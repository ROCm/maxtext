# Copyright 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Expert-parallel dispatch that sends each token to each expert shard once.

The standard path sends one all-to-all row per (token, expert) pair. Here the
sender moves one row per (token, destination shard) pair, together with the
token's local expert ids and routing weights on that shard. The receiver fans
each row out to its local experts and sums their outputs before the return
exchange, so both exchanges move at most `max_shards_per_token` rows per token.

All row movement is a masked gather whose custom gradient is again a gather, so
no step needs a scatter-add with repeated indices.
"""

from __future__ import annotations

import functools
import os
from typing import Callable, NamedTuple, Protocol, Sequence

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.extend import core as jax_core


class RowMap(NamedTuple):
  """Partial bijection between slots `[J, M]` and rows `[N]`.

  Row `i` holds slot `row_slots[i]`, flattened as `j * M + m`, when
  `row_valid[i]`. Slot `(j, m)` lives in row `slot_rows[j, m]` when
  `slot_valid[j, m]`. Indices of invalid entries are 0.
  """

  row_slots: jax.Array
  row_valid: jax.Array
  slot_rows: jax.Array
  slot_valid: jax.Array


class ExpertTiers(NamedTuple):
  """Expert-sorted capacity tiers of one shard's received slots."""

  maps: tuple[RowMap, ...]
  group_sizes: tuple[jax.Array, ...]
  assignments: jax.Array


class RowLayout(Protocol):
  """Placement of `C` compact expert-sorted rows into a padded row layout."""

  group_sizes: jax.Array
  compact_to_padded: jax.Array
  compact_valid: jax.Array
  padded_to_compact: jax.Array
  padded_valid: jax.Array


def _materialize(row_map: RowMap) -> RowMap:
  # Without the barrier XLA fuses the index arithmetic, including prefix sums,
  # into each consuming gather and repeats it for every hidden element.
  return RowMap(*jax.lax.optimization_barrier(tuple(row_map)))


def _take(values: jax.Array, indices: jax.Array, valid: jax.Array) -> jax.Array:
  """`values[indices]`, with rows where `valid` is false set to zero."""
  taken = values[jnp.where(valid, indices, 0)]
  mask = valid.reshape(valid.shape + (1,) * (values.ndim - 1))
  return jnp.where(mask, taken, jnp.zeros((), values.dtype))


def _sum_slots(values: Sequence[jax.Array], maps: Sequence[RowMap], dtype) -> jax.Array:
  """For each slot owner j, the f32 sum of its valid slot rows over all (values, map) pairs.

  One masked gather per slot column, so no `[J, M, ...]` intermediate is formed.
  """
  total = None
  for value, row_map in zip(values, maps):
    for m in range(row_map.slot_rows.shape[1]):
      term = _take(value, row_map.slot_rows[:, m], row_map.slot_valid[:, m]).astype(jnp.float32)
      total = term if total is None else total + term
  return total.astype(dtype)


def _slot_owner(row_map: RowMap) -> jax.Array:
  return row_map.row_slots // row_map.slot_rows.shape[1]


@jax.custom_vjp
def expand_rows(values: jax.Array, row_map: RowMap) -> jax.Array:
  """Row `i` is a copy of `values[j]`, where `j` owns the slot held by row `i`."""
  return _take(values, _slot_owner(row_map), row_map.row_valid)


def _expand_rows_fwd(values, row_map):
  return expand_rows(values, row_map), row_map


def _expand_rows_bwd(row_map, grad_output):
  return _sum_slots((grad_output,), (row_map,), grad_output.dtype), None


expand_rows.defvjp(_expand_rows_fwd, _expand_rows_bwd)


@jax.custom_vjp
def reduce_rows(values: tuple[jax.Array, ...], maps: tuple[RowMap, ...]) -> jax.Array:
  """`out[j]` is the sum of every row holding one of `j`'s slots, over all (values, map) pairs.

  The transpose of `expand_rows`. Several maps may cover disjoint subsets of
  the same slots, such as the capacity tiers of one shard.
  """
  return _sum_slots(values, maps, values[0].dtype)


def _reduce_rows_fwd(values, maps):
  return reduce_rows(values, maps), maps


def _reduce_rows_bwd(maps, grad_output):
  grads = tuple(_take(grad_output, _slot_owner(row_map), row_map.row_valid) for row_map in maps)
  return grads, None


reduce_rows.defvjp(_reduce_rows_fwd, _reduce_rows_bwd)


@jax.custom_vjp
def accumulate_rows(total: jax.Array, values: jax.Array, row_map: RowMap) -> jax.Array:
  """`total` plus `reduce_rows((values,), (row_map,))`, summed in `total`'s dtype."""
  return total + _sum_slots((values,), (row_map,), total.dtype)


def _accumulate_rows_fwd(total, values, row_map):
  # The empty array carries the dtype of `values` to backward.
  return accumulate_rows(total, values, row_map), (row_map, jnp.zeros((0,), values.dtype))


def _accumulate_rows_bwd(res, grad_output):
  row_map, values_dtype = res
  grad_values = _take(grad_output, _slot_owner(row_map), row_map.row_valid).astype(values_dtype.dtype)
  return grad_output, grad_values, None


accumulate_rows.defvjp(_accumulate_rows_fwd, _accumulate_rows_bwd)


@jax.custom_vjp
def slots_to_rows(values: jax.Array, row_map: RowMap) -> jax.Array:
  """Row `i` is `values[j, m]` for the slot `(j, m)` it holds; `values` is `[J, M, ...]`."""
  flat = values.reshape((-1,) + values.shape[2:])
  return _take(flat, row_map.row_slots, row_map.row_valid)


def _slots_to_rows_fwd(values, row_map):
  return slots_to_rows(values, row_map), row_map


def _slots_to_rows_bwd(row_map, grad_output):
  grad = _take(grad_output, row_map.slot_rows.reshape(-1), row_map.slot_valid.reshape(-1))
  return grad.reshape(row_map.slot_rows.shape + grad_output.shape[1:]), None


slots_to_rows.defvjp(_slots_to_rows_fwd, _slots_to_rows_bwd)


def sum_recomputed_tiers(
    count: jax.Array,
    starts: jax.Array,
    tier: Callable[[jax.Array], tuple[jax.Array, RowMap]],
    shape: Sequence[int],
) -> jax.Array:
  """f32 sum over `i < count` of `reduce_rows((values,), (row_map,))`, where `values, row_map = tier(starts[i])`.

  Forward and backward each loop over only the first `count` tiers and hold
  one tier's buffers at a time. The residuals are the arrays `tier` reads, so
  nothing the loop computes is saved: backward recomputes each tier and
  accumulates its cotangents.
  """
  # Every array `tier` closes over becomes an explicit input, including integer
  # ones: backward is traced after forward and cannot read forward's tracers.
  closed, out_shape = jax.make_jaxpr(tier, return_shape=True)(jax.ShapeDtypeStruct((), starts.dtype))
  out_tree = jax.tree.structure(out_shape)
  is_float = [jnp.issubdtype(c.dtype, jnp.inexact) for c in closed.consts]
  floats = tuple(c for c, f in zip(closed.consts, is_float) if f)
  others = tuple(c for c, f in zip(closed.consts, is_float) if not f)

  def call(start, floats, others):
    float_it, other_it = iter(floats), iter(others)
    consts = [next(float_it) if f else next(other_it) for f in is_float]
    outs = jax_core.jaxpr_as_fun(jax_core.ClosedJaxpr(closed.jaxpr, consts))(start)
    return jax.tree.unflatten(out_tree, outs)

  @jax.custom_vjp
  def run(count, starts, floats, others):
    def body(i, total):
      values, row_map = call(starts[i], floats, others)
      return accumulate_rows(total, values, row_map)

    return jax.lax.fori_loop(0, count, body, jnp.zeros(shape, jnp.float32))

  def run_fwd(count, starts, floats, others):
    return run(count, starts, floats, others), (count, starts, floats, others)

  def run_bwd(res, grad):
    count, starts, floats, others = res

    def body(i, acc):
      values, pull, row_map = jax.vjp(lambda floats: call(starts[i], floats, others), floats, has_aux=True)
      grad_values = _take(grad, _slot_owner(row_map), row_map.row_valid).astype(values.dtype)
      return jax.tree.map(jnp.add, acc, pull(grad_values)[0])

    grads = jax.lax.fori_loop(0, count, body, jax.tree.map(jnp.zeros_like, floats))
    return None, None, grads, tuple(None for _ in others)

  run.defvjp(run_fwd, run_bwd)
  return run(count, starts, floats, others)


def _invert(slot_rows: jax.Array, slot_valid: jax.Array, rows: int) -> jax.Array:
  """`row_slots` with `row_slots[slot_rows[s]] = s` for every valid flat slot `s`."""
  flat_rows = slot_rows.reshape(-1)
  slots = jnp.arange(flat_rows.size, dtype=jnp.int32)
  # Invalid slots target distinct out-of-range rows, which mode="drop" discards,
  # so every scatter index is unique.
  targets = jnp.where(slot_valid.reshape(-1), flat_rows, rows + slots)
  return jnp.zeros((rows,), jnp.int32).at[targets].set(slots, mode="drop", unique_indices=True)


def build_send_map(
    dest_shards: jax.Array, num_shards: int, slots_per_token: int
) -> tuple[RowMap, jax.Array, jax.Array]:
  """One send row per (token, destination shard) pair, grouped by destination.

  Slot `(t, s)` is token `t`'s `s`-th destination shard in shard order, so the
  send buffer has `T * slots_per_token` rows. Destinations past the first
  `slots_per_token` of a token are not sent.

  Args:
    dest_shards: `[T, K]` destination shard of each routed assignment.
    num_shards: number of expert shards.
    slots_per_token: static bound on distinct destination shards per token.

  Returns:
    The map between `[T, slots_per_token]` slots and send rows, the
    `[T, slots_per_token]` destination shard of each slot, and the number of
    rows sent to each shard.
  """
  tokens = dest_shards.shape[0]
  rows = tokens * slots_per_token
  shards = jnp.arange(num_shards, dtype=jnp.int32)
  sends = jnp.any(dest_shards[:, :, None] == shards[None, None, :], axis=1)
  order = jnp.cumsum(sends, axis=1, dtype=jnp.int32) - 1
  sends = sends & (order < slots_per_token)
  counts = jnp.sum(sends, axis=0, dtype=jnp.int32)
  starts = jnp.cumsum(counts) - counts
  shard_rows = starts[None, :] + jnp.cumsum(sends, axis=0, dtype=jnp.int32) - 1
  picks = sends[:, :, None] & (order[:, :, None] == jnp.arange(slots_per_token, dtype=jnp.int32))
  slot_valid = jnp.any(picks, axis=1)
  slot_shards = jnp.sum(jnp.where(picks, shards[None, :, None], 0), axis=1)
  slot_rows = jnp.sum(jnp.where(picks, shard_rows[:, :, None], 0), axis=1)
  row_valid = jnp.arange(rows, dtype=jnp.int32) < jnp.sum(counts)
  row_slots = _invert(slot_rows, slot_valid, rows)
  return _materialize(RowMap(row_slots, row_valid, slot_rows, slot_valid)), slot_shards, counts


def compose_row_map(row_map: RowMap, layout: RowLayout) -> RowMap:
  """`row_map` with its compact rows moved to their padded rows in `layout`."""
  compact = layout.padded_to_compact
  return _materialize(
      RowMap(
          row_slots=jnp.where(layout.padded_valid, row_map.row_slots[compact], 0),
          row_valid=layout.padded_valid & row_map.row_valid[compact],
          slot_rows=jnp.where(row_map.slot_valid, layout.compact_to_padded[row_map.slot_rows], 0),
          slot_valid=row_map.slot_valid & layout.compact_valid[row_map.slot_rows],
      )
  )


class SortedSlots(NamedTuple):
  """One shard's received slots `[R, K]` in local-expert order.

  Sorted position `p` holds flat slot `order[p]`; slot `(j, m)` sits at
  sorted position `position[j, m]`. Valid slots precede empty ones.
  """

  order: jax.Array
  position: jax.Array
  is_slot: jax.Array
  group_ends: jax.Array

  @property
  def assignments(self) -> jax.Array:
    return self.group_ends[-1]


def sort_slots(local_ids: jax.Array, num_experts: int) -> SortedSlots:
  """Stable sort of the slots of `local_ids` by local expert id, empty slots last."""
  key = jnp.where(local_ids >= 0, local_ids, num_experts).reshape(-1)
  sort_key = key.astype(jnp.uint8) if num_experts <= jnp.iinfo(jnp.uint8).max else key
  order = jnp.argsort(sort_key, stable=True).astype(jnp.int32)
  position = jnp.zeros_like(order).at[order].set(jnp.arange(order.size, dtype=jnp.int32), unique_indices=True)
  group_sizes = jnp.sum(key[:, None] == jnp.arange(num_experts, dtype=key.dtype)[None, :], axis=0, dtype=jnp.int32)
  return SortedSlots(
      order, position.reshape(local_ids.shape), (key < num_experts).reshape(local_ids.shape), jnp.cumsum(group_sizes)
  )


def expert_tier(slots: SortedSlots, start, capacity: int) -> tuple[RowMap, jax.Array]:
  """Map and group sizes of the tier holding sorted positions `[start, start + capacity)`.

  `start` may be traced; a static `start` must leave `capacity` sorted slots.
  """
  tier_position = slots.position - start
  slot_valid = slots.is_slot & (tier_position >= 0) & (tier_position < capacity)
  row_valid = jnp.arange(capacity, dtype=jnp.int32) < jnp.clip(slots.assignments - start, 0, capacity)
  if isinstance(start, int):
    row_slots = jax.lax.slice_in_dim(slots.order, start, start + capacity)
  else:
    row_slots = _take(slots.order, start + jnp.arange(capacity, dtype=jnp.int32), row_valid)
  row_map = _materialize(
      RowMap(
          row_slots=row_slots,
          row_valid=row_valid,
          slot_rows=jnp.where(slot_valid, tier_position, 0),
          slot_valid=slot_valid,
      )
  )
  tier_ends = jnp.clip(slots.group_ends - start, 0, capacity)
  return row_map, tier_ends - jnp.concatenate((jnp.zeros((1,), tier_ends.dtype), tier_ends[:-1]))


def _tiers(slots: SortedSlots, capacities: Sequence[int]) -> ExpertTiers:
  maps, tier_sizes = [], []
  start = 0
  for capacity in capacities:
    capacity = min(int(capacity), slots.order.size - start)
    if capacity <= 0:
      break
    row_map, sizes = expert_tier(slots, start, capacity)
    maps.append(row_map)
    tier_sizes.append(sizes)
    start += capacity
  return ExpertTiers(tuple(maps), tuple(tier_sizes), slots.assignments)


def build_expert_tiers(local_ids: jax.Array, num_experts: int, capacities: Sequence[int]) -> ExpertTiers:
  """Sort received slots by local expert and split them into capacity tiers.

  Args:
    local_ids: `[R, K]` local expert id of each received slot, -1 when empty.
    num_experts: number of local experts.
    capacities: static row capacity of each tier, filled in order. Slots
      beyond the total capacity are dropped, highest expert ids first.

  Returns:
    One map and one group-size vector per tier, and the number of valid slots.
  """
  return _tiers(sort_slots(local_ids, num_experts), capacities)


def exchange_params(traffic: jax.Array, shard_id, *, is_dispatch: bool):
  """`ragged_all_to_all` offsets and sizes for a packed `[sender, receiver]` row-count matrix."""
  matrix = traffic if is_dispatch else traffic.T
  send_sizes = matrix[shard_id]
  input_offsets = jnp.cumsum(send_sizes) - send_sizes
  output_offsets = (jnp.cumsum(matrix, axis=0) - matrix)[shard_id]
  recv_sizes = matrix[:, shard_id]
  return input_offsets, send_sizes, output_offsets, recv_sizes


def _exclusive_cumsum(values: jax.Array) -> jax.Array:
  return jnp.cumsum(values) - values


def two_stage_groups(num_shards: int, local_shards: int) -> tuple[list[list[int]], list[list[int]]]:
  """`axis_index_groups` of the two stages: shards on one process, then shards with one local index."""
  nodes = num_shards // local_shards
  within = [[n * local_shards + i for i in range(local_shards)] for n in range(nodes)]
  across = [[n * local_shards + i for n in range(nodes)] for i in range(local_shards)]
  return within, across


def _two_stage_plan(matrix: jax.Array, shard_id, local_shards: int):
  """`ragged_all_to_all` arguments of both stages of `two_stage_exchange`.

  Shard `n * L + i` is local index `i` of process `n`. Stage one sends each row
  within its process to the shard with its receiver's local index; that shard
  holds the rows in receiver-process-major, then sender order. Stage two sends
  each receiver process's block to the shard with the same local index there,
  where it lands exactly where a direct exchange would have put it.
  """
  num_shards = matrix.shape[0]
  nodes = num_shards // local_shards
  node, local = shard_id // local_shards, shard_id % local_shards
  # m4[sender node, sender local, receiver node, receiver local]
  m4 = matrix.reshape(nodes, local_shards, nodes, local_shards)
  # held[n, h, rn, si]: rows shard (n, h) holds after stage one, from (n, si) for (rn, h).
  held = jnp.transpose(m4, (0, 3, 2, 1))
  held_offsets = jax.vmap(jax.vmap(lambda v: _exclusive_cumsum(v.reshape(-1)).reshape(v.shape)))(held)
  sender_offsets = _exclusive_cumsum(matrix[shard_id])
  receiver_offsets = jnp.cumsum(matrix, axis=0) - matrix

  # Stage one, slice p * nodes + rn: rows for receiver (rn, p), sent to local peer p.
  peers = jnp.arange(local_shards)[:, None]
  rnodes = jnp.arange(nodes)[None, :]
  receivers = (rnodes * local_shards + peers).reshape(-1)
  stage1 = (
      sender_offsets[receivers],
      matrix[shard_id, receivers],
      held_offsets[node, peers, rnodes, local].reshape(-1),
      # Slice q * nodes + rn: rows from local sender q for receiver (rn, local).
      m4[node, :, :, local].reshape(-1),
  )
  # Stage two, slice rn: this shard's block for receiver (rn, local), sent to process rn.
  stage2 = (
      held_offsets[node, local, :, 0],
      jnp.sum(held[node, local], axis=1),
      receiver_offsets[node * local_shards, jnp.arange(nodes) * local_shards + local],
      jnp.sum(m4[:, :, node, local], axis=1),
  )
  return stage1, stage2, jnp.sum(matrix[:, shard_id])


def _two_stage_impl(operand, matrix, shard_id, output_rows, pair_rows, axis_name, local_shards, key, zero_unwritten, ragged):
  num_shards = matrix.shape[0]
  # A shard holds rows from its process's senders for one local index on every process.
  held_rows = min(pair_rows * num_shards, local_shards * operand.shape[0], num_shards // local_shards * output_rows)
  stage1, stage2, received = _two_stage_plan(matrix, shard_id, local_shards)
  within, across = two_stage_groups(num_shards, local_shards)
  held = ragged(operand, held_rows, *stage1, axis_name=axis_name, axis_index_groups=within, key=key)
  output = ragged(held, output_rows, *stage2, axis_name=axis_name, axis_index_groups=across, key=key + 1)
  if not zero_unwritten:
    return output
  # Received rows are contiguous from row 0, in sender order.
  written = jnp.arange(output_rows, dtype=jnp.int32) < received
  return jnp.where(written[:, None], output, jnp.minimum(received, 0).astype(output.dtype))


def fresh_ragged_all_to_all(
    operand, output_rows, input_offsets, send_sizes, output_offsets, recv_sizes, *, axis_name, axis_index_groups, key
):
  """`ragged_all_to_all` into a fresh uninitialized `output_rows` buffer."""
  from jax_aiter.ops.buffers import uninitialized  # pylint: disable=g-import-not-at-top

  output = uninitialized((output_rows,) + operand.shape[1:], operand.dtype, depends_on=recv_sizes, key=key)
  return jax.lax.ragged_all_to_all(
      operand,
      output,
      input_offsets,
      send_sizes,
      output_offsets,
      recv_sizes,
      axis_name=axis_name,
      axis_index_groups=axis_index_groups,
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8, 9))
def two_stage_exchange(
    operand,
    matrix,
    shard_id,
    output_rows,
    pair_rows,
    axis_name,
    local_shards,
    site,
    zero_unwritten,
    ragged,
):
  """The exchange of `exchange_params(matrix, ...)`, done within each process and then across processes.

  Every cross-process transfer pairs shards with the same local index, so on a
  network where each GPU's NIC reaches only the same NIC index on other nodes
  (a rail-isolated fabric) no transfer crosses rails.

  `matrix[s, r]` is the number of rows shard `s` sends shard `r`, at most
  `pair_rows`. This shard's rows are contiguous from row 0 in receiver order,
  and the `output_rows` result holds received rows contiguously from row 0 in
  sender order. Shards `[n * local_shards, (n + 1) * local_shards)` of
  `axis_name` must be one process's. `ragged` has the signature of
  `fresh_ragged_all_to_all`. The gradient is the same exchange of `matrix.T`.
  With `zero_unwritten` false, unwritten output rows and the gradients of
  unsent operand rows are uninitialized.
  """
  return _two_stage_impl(
      operand, matrix, shard_id, output_rows, pair_rows, axis_name, local_shards, 8 + 4 * site, zero_unwritten, ragged
  )


def _two_stage_exchange_fwd(operand, matrix, shard_id, output_rows, pair_rows, axis_name, local_shards, site, zero_unwritten, ragged):
  output = two_stage_exchange(
      operand, matrix, shard_id, output_rows, pair_rows, axis_name, local_shards, site, zero_unwritten, ragged
  )
  return output, (matrix, shard_id, jnp.zeros((operand.shape[0], 0), operand.dtype))


def _two_stage_exchange_bwd(output_rows, pair_rows, axis_name, local_shards, site, zero_unwritten, ragged, res, grad):
  del output_rows
  matrix, shard_id, operand_like = res
  grad_operand = _two_stage_impl(
      grad,
      matrix.T,
      shard_id,
      operand_like.shape[0],
      pair_rows,
      axis_name,
      local_shards,
      8 + 4 * site + 2,
      zero_unwritten,
      ragged,
  )
  return grad_operand, None, None


two_stage_exchange.defvjp(_two_stage_exchange_fwd, _two_stage_exchange_bwd)


def process_local_shards(mesh, axis_name: str) -> int:
  """Shards of `axis_name` per process, which must own equal contiguous blocks of the axis.

  The shards at one position of every block must also be the same local GPU of
  their process, so that `two_stage_exchange` keeps each transfer on one NIC rail.
  Other processes' devices carry no `local_hardware_id`, so each process checks
  that its own blocks list its GPUs in increasing order; with every process
  seeing the same GPUs, all blocks then share one order.
  """
  import numpy as np  # pylint: disable=g-import-not-at-top

  axis = mesh.axis_names.index(axis_name)
  devices = np.moveaxis(mesh.devices, axis, -1).reshape(-1, mesh.devices.shape[axis])
  procs = np.vectorize(lambda d: d.process_index)(devices)
  size = procs.shape[1]
  local = int(np.argmax(procs[0] != procs[0, 0])) or size
  blocks = procs[:, ::local]
  if (
      size % local
      or not np.array_equal(procs, np.repeat(blocks, local, axis=1))
      or any(len(set(row)) != len(row) for row in blocks)
  ):
    raise ValueError(f"processes do not own equal contiguous blocks of mesh axis {axis_name!r}: {procs.tolist()}")
  hw = np.vectorize(lambda d: getattr(d, "local_hardware_id", None), otypes=[object])(devices)
  for block in hw.reshape(-1, local).tolist():
    known = [h for h in block if h is not None]
    if known and (len(known) != local or any(a >= b for a, b in zip(known, known[1:]))):
      raise ValueError(f"local GPU order differs between processes on mesh axis {axis_name!r}: {hw.tolist()}")
  return local


def _maybe_log_dedup_route(shard, tokens, send_rows, recv_rows, assignments, tier_rows, dropped):
  from maxtext.layers import moe  # pylint: disable=g-import-not-at-top

  moe._append_route_record({  # pylint: disable=protected-access
      "kind": "dedup_dispatch",
      "pid": os.getpid(),
      "shard": int(shard),
      "tokens": int(tokens),
      "send_rows": int(send_rows),
      "recv_rows": int(recv_rows),
      "assignments": int(assignments),
      "tier_rows": [int(v) for v in tier_rows],
      "dropped": int(dropped),
  })


def dedup_dispatch_combine(
    x: jax.Array,
    expert_ids: jax.Array,
    weights: jax.Array,
    expert_fn: Callable[..., jax.Array],
    *,
    num_experts: int,
    num_shards: int,
    axis_name: str,
    max_shards_per_token: int,
    capacities: Sequence[int],
    check_capacity: bool = False,
    exchange: Callable[..., jax.Array] | None = None,
    row_layout: Callable[[jax.Array, int], RowLayout] | None = None,
    zero_unwritten_rows: bool = True,
    dropless_tier_rows: int = 0,
    local_shards: int = 0,
    two_stage_ragged: Callable[..., jax.Array] = fresh_ragged_all_to_all,
) -> jax.Array:
  """Routed-expert output for this shard's tokens, with one exchange row per (token, shard).

  Must run inside a `shard_map` over `axis_name`, with experts sharded
  contiguously: shard `s` owns experts `[s * E, (s + 1) * E)`.

  Args:
    x: `[T, H]` token activations.
    expert_ids: `[T, K]` global expert id of each assignment.
    weights: `[T, K]` routing weight of each assignment.
    expert_fn: `expert_fn(rows, row_weights, group_sizes, is_overflow=...,
      save_residuals=...)` maps `[C, H]` expert-sorted rows and their `[C]`
      routing weights to the weighted `[C, H_out]` expert outputs of one
      capacity tier. Rows not holding an assignment are zero and their outputs
      are ignored. With `save_residuals` false it must name no remat
      residuals, so that backward recomputes the tier from its inputs.
    num_experts: total number of routed experts.
    num_shards: size of `axis_name`.
    axis_name: expert-parallel mesh axis.
    max_shards_per_token: static bound on distinct shards per token.
    capacities: static row capacity of each local expert tier.
    check_capacity: raise on the host when assignments exceed the capacity.
    exchange: replacement for `moe._ragged_all_to_all_fresh`, with the same
      signature; zero-fills rows it does not receive when its last argument
      is true.
    row_layout: `row_layout(group_sizes, rows)` places a tier's compact rows
      in the padded layout `expert_fn` expects; `expert_fn` then receives
      padded rows and the layout's group sizes.
    zero_unwritten_rows: zero the rows the activation exchanges do not write,
      and their gradients. When false, those rows are uninitialized; every
      reader is a masked gather, so they never reach the result. The metadata
      exchange always zero-fills.
    dropless_tier_rows: when positive, tiers of this many rows follow
      `capacities` until every received slot has a row, so no received
      assignment drops. A shard whose slots fit `capacities` runs those tiers
      exactly as without dropless tiers. Otherwise it also runs the dropless
      tiers that hold rows, one at a time and with `save_residuals` false;
      backward recomputes each of them from the received rows, saved as
      `moe_dedup_received`.
    local_shards: when positive and below `num_shards`, the expert shards of
      one process, which owns a contiguous block of `axis_name`. Every
      exchange then runs as `two_stage_exchange`, so cross-process transfers
      pair only shards with the same local index; `exchange` is not used.
    two_stage_ragged: the grouped `ragged_all_to_all` of `two_stage_exchange`.

  Returns:
    `[T, H_out]` weighted sum of each token's expert outputs.
  """
  from maxtext.layers import moe  # pylint: disable=g-import-not-at-top

  tokens, top_k = expert_ids.shape
  local_experts = num_experts // num_shards
  shard_id = jax.lax.axis_index(axis_name)
  expert_ids = expert_ids.astype(jnp.int32)
  dest = expert_ids // local_experts
  send_rows = tokens * max_shards_per_token
  recv_rows = tokens * num_shards

  send_map, slot_shards, send_counts = build_send_map(dest, num_shards, max_shards_per_token)
  traffic = jax.lax.all_gather(send_counts, axis_name)
  # A saved plan spares backward the sort and the count exchange, a collective
  # that would also wait for the slowest shard to finish its expert gradients.
  send_map, slot_shards, traffic = checkpoint_name((send_map, slot_shards, traffic), "moe_dedup_plan")
  on_slot = (dest[:, None, :] == slot_shards[:, :, None]) & send_map.slot_valid[:, :, None]
  # Ids are sent as id + 1 so the zero rows of the fresh receive buffer read as empty.
  slot_ids = jnp.where(on_slot, (expert_ids - dest * local_experts + 1)[:, None, :], 0).astype(jnp.float32)
  slot_weights = jnp.where(on_slot, weights.astype(jnp.float32)[:, None, :], 0.0)
  send_meta = jnp.concatenate(
      (
          _take(slot_ids.reshape(-1, top_k), send_map.row_slots, send_map.row_valid),
          slots_to_rows(slot_weights, send_map),
      ),
      axis=1,
  )

  dispatch = exchange_params(traffic, shard_id, is_dispatch=True)
  combine = exchange_params(traffic, shard_id, is_dispatch=False)
  recv_starts = jnp.cumsum(dispatch[3]) - dispatch[3]
  send_starts = jnp.cumsum(dispatch[1]) - dispatch[1]

  if 0 < local_shards < num_shards:

    def run_exchange(operand, matrix, params, starts, rows, site, zero):
      del params, starts
      # A shard sends any one shard at most one row per token.
      return two_stage_exchange(
          operand, matrix, shard_id, rows, tokens, axis_name, local_shards, site, zero, two_stage_ragged
      )

  else:
    exchange = exchange or moe._ragged_all_to_all_fresh  # pylint: disable=protected-access

    def run_exchange(operand, matrix, params, starts, rows, site, zero):
      del matrix
      return exchange(operand, *params, starts, rows, axis_name, site, zero)

  recv_x = run_exchange(expand_rows(x, send_map), traffic, dispatch, recv_starts, recv_rows, 0, zero_unwritten_rows)
  recv_meta = run_exchange(send_meta, traffic, dispatch, recv_starts, recv_rows, 2, True)
  recv_meta = checkpoint_name(recv_meta, "moe_dispatched_meta")
  local_ids = jnp.round(recv_meta[:, :top_k]).astype(jnp.int32) - 1
  recv_weights = recv_meta[:, top_k:]

  slots = sort_slots(local_ids, local_experts)
  tiers = _tiers(slots, capacities)
  tier_rows = [jnp.sum(sizes) for sizes in tiers.group_sizes]
  covered = sum(row_map.row_slots.shape[0] for row_map in tiers.maps)
  extra_starts = tuple(range(covered, local_ids.size, dropless_tier_rows)) if dropless_tier_rows > 0 else ()
  if extra_starts:
    slots = checkpoint_name(slots, "moe_dedup_plan")
    recv_x = checkpoint_name(recv_x, "moe_dedup_received")
    tier_rows += [jnp.clip(slots.assignments - start, 0, dropless_tier_rows) for start in extra_starts]
  # Assignments this shard did not send (more destination shards than
  # `max_shards_per_token`) plus received ones beyond the tiers.
  unsent = jnp.sum(~jnp.any(on_slot, axis=1), dtype=jnp.int32)
  dropped = unsent + tiers.assignments - functools.reduce(jnp.add, tier_rows)
  if check_capacity:
    jax.debug.callback(moe._check_no_tier_route_drops, dropped)  # pylint: disable=protected-access
  if os.environ.get("MAXTEXT_ROUTE_STATS_PATH"):
    jax.debug.callback(
        _maybe_log_dedup_route,
        shard_id,
        jnp.int32(tokens),
        jnp.sum(send_counts),
        jnp.sum(dispatch[3]),
        tiers.assignments,
        jnp.stack(tier_rows),
        dropped,
    )

  maps, group_sizes = tiers.maps, tiers.group_sizes
  if row_layout is not None:
    layouts = [row_layout(sizes, row_map.row_slots.shape[0]) for sizes, row_map in zip(group_sizes, maps)]
    maps = tuple(compose_row_map(row_map, layout) for row_map, layout in zip(maps, layouts))
    group_sizes = tuple(layout.group_sizes for layout in layouts)
  maps, group_sizes, tier_rows = checkpoint_name((maps, group_sizes, tuple(tier_rows)), "moe_dedup_plan")

  def run_tier(row_map, sizes, *, is_overflow, save_residuals=True):
    return expert_fn(
        expand_rows(recv_x, row_map),
        slots_to_rows(recv_weights, row_map),
        sizes,
        is_overflow=is_overflow,
        save_residuals=save_residuals,
    )

  def skippable_tier(tier):
    rows = maps[tier].row_slots.shape[0]
    return jax.lax.cond(
        tier_rows[tier] > 0,
        lambda _: run_tier(maps[tier], group_sizes[tier], is_overflow=True),
        lambda _: jnp.zeros((rows,) + outputs[0].shape[1:], outputs[0].dtype),
        None,
    )

  def dropless_tier(start):
    row_map, sizes = expert_tier(slots, start, dropless_tier_rows)
    if row_layout is not None:
      layout = row_layout(sizes, dropless_tier_rows)
      row_map, sizes = compose_row_map(row_map, layout), layout.group_sizes
    return run_tier(row_map, sizes, is_overflow=True, save_residuals=False), row_map

  outputs = [run_tier(maps[0], group_sizes[0], is_overflow=False)]
  overflowed = functools.reduce(jnp.add, tier_rows[1:], jnp.int32(0)) > 0
  if extra_starts:
    starts = jnp.asarray(extra_starts, jnp.int32)
    active = jnp.sum(starts < slots.assignments, dtype=jnp.int32)

    def fixed_tiers():
      if len(maps) == 1:
        return ()
      # Tiers fill in order, so the first overflow tier holds rows whenever a later one does.
      first = run_tier(maps[1], group_sizes[1], is_overflow=True)
      return (first, *(skippable_tier(tier) for tier in range(2, len(maps))))

    def with_dropless_tiers(base):
      total = sum_recomputed_tiers(active, starts, dropless_tier, (local_ids.shape[0],) + base.shape[1:])
      for value, row_map in zip((base, *fixed_tiers()), maps):
        total = accumulate_rows(total, value, row_map)
      return total.astype(base.dtype)

    # Each branch computes its whole result, so no large array passes through
    # a branch unchanged and a step without dropless rows does exactly the
    # work, and saves exactly the residuals, of the configured tiers alone.
    branches = [lambda base: reduce_rows((base,), maps[:1])]
    index = jnp.int32(0)
    if len(maps) > 1:
      branches.append(lambda base: reduce_rows((base, *fixed_tiers()), maps))
      index += overflowed
    branches.append(with_dropless_tiers)
    index += active > 0
    recv_output = jax.lax.switch(index, branches, outputs[0])
  elif len(maps) > 1:
    outputs += [skippable_tier(tier) for tier in range(1, len(maps))]
    # Each slot column costs one gather per received row even when masked, so
    # the usually empty overflow tiers join the sum only when they hold rows.
    recv_output = jax.lax.cond(
        overflowed,
        lambda values: reduce_rows(values, maps),
        lambda values: reduce_rows(values[:1], maps[:1]),
        tuple(outputs),
    )
  else:
    recv_output = reduce_rows(tuple(outputs), maps)
  returned = run_exchange(recv_output, traffic.T, combine, send_starts, send_rows, 1, zero_unwritten_rows)
  returned = checkpoint_name(returned, "moe_combined")
  return reduce_rows((returned,), (send_map,))
