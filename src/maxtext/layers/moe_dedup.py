# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# JAX-AITER and Primus-Turbo are optional ROCm packages, and moe imports this module,
# so those imports happen where they are used.
# pylint: disable=import-outside-toplevel
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


def switch_on_base(
    index: jax.Array, base: Callable[[], jax.Array], branches: Sequence[Callable[[jax.Array], jax.Array]]
) -> jax.Array:
  """`jax.lax.switch(index, branches, base())`, without differentiating the switch.

  Differentiating a switch makes every branch return the residuals of every
  other branch, zero-filled, and return zero cotangents for the arrays it does
  not read. Here backward instead pulls the cotangent back through `branches[0]`
  and `base`'s own vjp, then, only when `index > 0`, recomputes branch `index`
  from the arrays it reads and adds its cotangents in place. Every branch must
  pass the cotangent to its argument exactly as `branches[0]` does.
  """
  if len(branches) == 1:
    return branches[0](base())
  closed = [jax.make_jaxpr(base)()]
  base_shape = jax.ShapeDtypeStruct(closed[0].out_avals[0].shape, closed[0].out_avals[0].dtype)
  closed += [jax.make_jaxpr(branch)(base_shape) for branch in branches]
  # Every array the functions close over becomes an explicit input, as in
  # `sum_recomputed_tiers`; the floating ones are differentiated.
  slot = {}
  consts = []
  for jaxpr in closed:
    for c in jaxpr.consts:
      if id(c) not in slot:
        slot[id(c)] = len(consts)
        consts.append(c)
  is_float = [jnp.issubdtype(c.dtype, jnp.inexact) for c in consts]
  place = [sum(is_float[:i]) if f else sum(not g for g in is_float[:i]) for i, f in enumerate(is_float)]
  floats = tuple(c for c, f in zip(consts, is_float) if f)
  others = tuple(c for c, f in zip(consts, is_float) if not f)

  def call(fn, args, floats, others):
    fn_consts = [
        floats[place[slot[id(c)]]] if is_float[slot[id(c)]] else others[place[slot[id(c)]]] for c in closed[fn].consts
    ]
    (out,) = jax_core.jaxpr_as_fun(jax_core.ClosedJaxpr(closed[fn].jaxpr, fn_consts))(*args)
    return out

  def taken(index, base_out, floats, others):
    return jax.lax.switch(
        index, [lambda value, fn=fn: call(fn, (value,), floats, others) for fn in range(1, len(closed))], base_out
    )

  @jax.custom_vjp
  def run(index, floats, others):
    return taken(index, call(0, (), floats, others), floats, others)

  def run_fwd(index, floats, others):
    base_out, pull_base = jax.vjp(lambda floats: call(0, (), floats, others), floats)
    return taken(index, base_out, floats, others), (index, pull_base, floats, others)

  def run_bwd(res, grad):
    index, pull_base, floats, others = res
    # Branches read their argument linearly, so its value does not reach a cotangent.
    stub = jnp.zeros(base_shape.shape, base_shape.dtype)
    grad_base = jax.vjp(lambda base_out: call(1, (base_out,), floats, others), stub)[1](grad)[0]
    (grads,) = pull_base(grad_base)

    def pull_branch(fn):
      return jax.vjp(lambda floats: call(fn, (stub,), floats, others), floats)[1](grad)[0]

    def add_taken(_, grads):
      extra = jax.lax.switch(index - 1, [functools.partial(pull_branch, fn) for fn in range(2, len(closed))])
      return jax.tree.map(jnp.add, grads, extra)

    # A loop rather than a cond, so the usual zero-trip case passes `grads` through without a copy.
    grads = jax.lax.fori_loop(0, (index > 0).astype(jnp.int32), add_taken, grads)
    return None, grads, tuple(None for _ in others)

  run.defvjp(run_fwd, run_bwd)
  return run(index, floats, others)


def _invert(slot_rows: jax.Array, slot_valid: jax.Array, rows: int) -> jax.Array:
  """`row_slots` with `row_slots[slot_rows[s]] = s` for every valid flat slot `s`."""
  flat_rows = slot_rows.reshape(-1)
  slots = jnp.arange(flat_rows.size, dtype=jnp.int32)
  # Invalid slots target distinct out-of-range rows, which mode="drop" discards,
  # so every scatter index is unique.
  targets = jnp.where(slot_valid.reshape(-1), flat_rows, rows + slots)
  return jnp.zeros((rows,), jnp.int32).at[targets].set(slots, mode="drop", unique_indices=True)


def build_send_map(dest_shards: jax.Array, num_shards: int, slots_per_token: int) -> tuple[RowMap, jax.Array, jax.Array]:
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


# On ROCm (gfx950), XLA's GPU sort intermittently misorders and duplicates
# entries of sorts whose shared-memory tile spans several wavefronts; rows of
# 256 keys, one 64-lane wavefront each, sort correctly.
_SORT_ROW = 256


def _stable_order(key: jax.Array, num_keys: int) -> jax.Array:
  """`jnp.argsort(key, stable=True)` of flat `key` in `[0, num_keys)`, sorting only rows of `_SORT_ROW` keys."""
  size = key.size
  row = max(1, min(_SORT_ROW, size))
  rows = -(-size // row)
  # Padding takes the largest key, so it sorts after every slot.
  blocks = jnp.pad(key, (0, rows * row - size), constant_values=num_keys - 1).reshape(rows, row)
  sort_key = blocks.astype(jnp.uint8) if num_keys <= 256 else blocks
  sorted_key, local = jax.lax.sort(
      (sort_key, jax.lax.broadcasted_iota(jnp.int32, blocks.shape, 1)), dimension=1, num_keys=1, is_stable=True
  )
  counts = jnp.sum(blocks[:, :, None] == jnp.arange(num_keys, dtype=key.dtype), axis=1, dtype=jnp.int32)
  totals = jnp.sum(counts, axis=0)
  # Sorted entry p of row r, holding key k, lands at the start of key k, plus
  # the k slots of earlier rows, plus its rank among row r's k slots.
  offset = (jnp.cumsum(totals) - totals) + (jnp.cumsum(counts, axis=0) - counts) - (jnp.cumsum(counts, axis=1) - counts)
  target = jnp.take_along_axis(offset, sorted_key.astype(jnp.int32), axis=1) + jnp.arange(row, dtype=jnp.int32)
  source = local + jnp.arange(0, rows * row, row, dtype=jnp.int32)[:, None]
  order = jnp.zeros(rows * row, jnp.int32).at[target.reshape(-1)].set(source.reshape(-1), unique_indices=True)
  return order[:size]


def sort_slots(local_ids: jax.Array, num_experts: int) -> SortedSlots:
  """Stable sort of the slots of `local_ids` by local expert id, empty slots last."""
  key = jnp.where(local_ids >= 0, local_ids, num_experts).reshape(-1)
  order = _stable_order(key, num_experts + 1)
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
  """Splits already sorted slots into tiers; see :func:`build_expert_tiers`."""
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


def setup_deepep(mesh, axis_name: str, hidden_bytes: int | None = None) -> None:
  """Bootstraps Primus-Turbo intranode DeepEP for `dedup_dispatch_combine(..., deepep=True)`.

  With several GPUs per process its kernels take their rank from the current
  GPU and their rank count from the process's GPU count, so at every position
  of the other mesh axes, `axis_name` must list exactly this process's GPUs in
  hardware order. With one GPU per process, `axis_name` spans one process per
  GPU and DeepEP shares buffers across them; `hidden_bytes` (row bytes) sizes
  those buffers.
  """
  import numpy as np
  import primus_turbo.jax
  from primus_turbo.jax.lax import moe as deepep_moe

  axis = mesh.axis_names.index(axis_name)
  devices = np.moveaxis(mesh.devices, axis, -1).reshape(-1, mesh.devices.shape[axis])
  if jax.local_device_count() == 1 and jax.process_count() > 1:
    if hidden_bytes is None:
      raise ValueError("DeepEP with one GPU per process needs hidden_bytes")
    # Larger groups take Primus-Turbo's internode path, which this has not used.
    if devices.shape[1] > 8:
      raise ValueError(f"DeepEP with one GPU per process supports at most 8 ranks on {axis_name!r}")
    primus_turbo.jax.initialize()
    deepep_moe.setup(mesh=mesh, ep_axis_name=axis_name, hidden_bytes=hidden_bytes, mode="per_process")
    return
  me, gpus = jax.process_index(), list(range(jax.local_device_count()))
  for row in devices:
    if any(d.process_index == me for d in row) and (
        any(d.process_index != me for d in row) or [d.local_hardware_id for d in row] != gpus
    ):
      raise ValueError(f"DeepEP needs mesh axis {axis_name!r} to be this process's GPUs {gpus} in order: {row.tolist()}")
  primus_turbo.jax.initialize()
  # An explicit mode="inproc" is rejected once jax.process_count() > 1; left unset,
  # a process with several GPUs still resolves to the in-process mode.
  deepep_moe.setup()


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
    deepep: bool = False,
    pass_row_source: bool = False,
    recompute_tier_switch: bool = False,
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
    deepep: exchange tokens with Primus-Turbo intranode DeepEP dispatch and
      combine, after `setup_deepep`, instead of `ragged_all_to_all`. `x` must
      be bfloat16, every destination shard of a token receives it, and
      `exchange` and `zero_unwritten_rows` do not apply.
    pass_row_source: also pass `expert_fn` the keyword `lhs_rows=(received,
      row_index)`: its row `r` is `received[row_index[r]]`, or zero for a
      negative index, so it can read the received rows without the gathered copy.
    recompute_tier_switch: with dropless tiers, differentiate the first tier
      and the switch over the extra tiers with `switch_on_base`, so a step
      without extra rows neither writes their zero-filled residuals nor zero
      cotangents. The first tier then runs inside a `custom_vjp`: a remat
      policy cannot save the arrays `expert_fn` names, and recomputes them.

  Returns:
    `[T, H_out]` weighted sum of each token's expert outputs.
  """
  from maxtext.layers import moe

  tokens, top_k = expert_ids.shape
  local_experts = num_experts // num_shards
  shard_id = jax.lax.axis_index(axis_name)
  expert_ids = expert_ids.astype(jnp.int32)
  dest = expert_ids // local_experts
  send_rows = tokens * max_shards_per_token
  recv_rows = tokens * num_shards

  if deepep:
    from primus_turbo.jax.lax import moe as deepep_moe

    recv_x, local_ids, recv_weights, handle = deepep_moe.moe_dispatch(
        x, expert_ids, weights.astype(jnp.float32), num_experts
    )
    received = handle[0][num_shards - 1, shard_id]
    # Rows past the received ones are uninitialized.
    local_ids = jnp.where(jnp.arange(recv_rows)[:, None] < received, local_ids, -1)
    local_ids, recv_weights = checkpoint_name((local_ids, recv_weights), "moe_dispatched_meta")
    unsent = jnp.int32(0)
    # DeepEP returns the rows through its own combine.
    combine = send_starts = None
  else:
    send_map, slot_shards, send_counts = build_send_map(dest, num_shards, max_shards_per_token)
    # A psum rather than an all-gather: XLA's pipelined all-gather would move
    # this exchange, and the attention and routing it depends on, into the
    # previous layer's iteration, computing them twice.
    traffic = jax.lax.psum(jnp.zeros((num_shards, num_shards), jnp.int32).at[shard_id].set(send_counts), axis_name)
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

    exchange = exchange or moe._ragged_all_to_all_fresh  # pylint: disable=protected-access

    def run_exchange(operand, params, starts, rows, site, zero):
      return exchange(operand, *params, starts, rows, axis_name, site, zero)

    recv_x = run_exchange(expand_rows(x, send_map), dispatch, recv_starts, recv_rows, 0, zero_unwritten_rows)
    recv_meta = run_exchange(send_meta, dispatch, recv_starts, recv_rows, 2, True)
    recv_meta = checkpoint_name(recv_meta, "moe_dispatched_meta")
    local_ids = jnp.round(recv_meta[:, :top_k]).astype(jnp.int32) - 1
    recv_weights = recv_meta[:, top_k:]
    # Assignments this shard did not send: more destination shards than `max_shards_per_token`.
    unsent = jnp.sum(~jnp.any(on_slot, axis=1), dtype=jnp.int32)

  slots = sort_slots(local_ids, local_experts)
  tiers = _tiers(slots, capacities)
  tier_rows = [jnp.sum(sizes) for sizes in tiers.group_sizes]
  covered = sum(row_map.row_slots.shape[0] for row_map in tiers.maps)
  extra_starts = tuple(range(covered, local_ids.size, dropless_tier_rows)) if dropless_tier_rows > 0 else ()
  if extra_starts:
    slots = checkpoint_name(slots, "moe_dedup_plan")
    recv_x = checkpoint_name(recv_x, "moe_dedup_received")
    tier_rows += [jnp.clip(slots.assignments - start, 0, dropless_tier_rows) for start in extra_starts]
  # Unsent assignments plus received ones beyond the tiers.
  dropped = unsent + tiers.assignments - functools.reduce(jnp.add, tier_rows)
  if check_capacity:
    jax.debug.callback(moe._check_no_tier_route_drops, dropped)  # pylint: disable=protected-access
  maps, group_sizes = tiers.maps, tiers.group_sizes
  if row_layout is not None:
    layouts = [row_layout(sizes, row_map.row_slots.shape[0]) for sizes, row_map in zip(group_sizes, maps)]
    maps = tuple(compose_row_map(row_map, layout) for row_map, layout in zip(maps, layouts))
    group_sizes = tuple(layout.group_sizes for layout in layouts)
  maps, group_sizes, tier_rows = checkpoint_name((maps, group_sizes, tuple(tier_rows)), "moe_dedup_plan")

  def run_tier(row_map, sizes, *, is_overflow, save_residuals=True):
    extra = {}
    if pass_row_source:
      extra["lhs_rows"] = (recv_x, jnp.where(row_map.row_valid, _slot_owner(row_map), -1))
    return expert_fn(
        expand_rows(recv_x, row_map),
        slots_to_rows(recv_weights, row_map),
        sizes,
        is_overflow=is_overflow,
        save_residuals=save_residuals,
        **extra,
    )

  def skippable_tier(tier, like):
    rows = maps[tier].row_slots.shape[0]
    return jax.lax.cond(
        tier_rows[tier] > 0,
        lambda _: run_tier(maps[tier], group_sizes[tier], is_overflow=True),
        lambda _: jnp.zeros((rows,) + like.shape[1:], like.dtype),
        None,
    )

  def dropless_tier(start):
    row_map, sizes = expert_tier(slots, start, dropless_tier_rows)
    if row_layout is not None:
      layout = row_layout(sizes, dropless_tier_rows)
      row_map, sizes = compose_row_map(row_map, layout), layout.group_sizes
    return run_tier(row_map, sizes, is_overflow=True, save_residuals=False), row_map

  base_tier = functools.partial(run_tier, maps[0], group_sizes[0], is_overflow=False)
  overflowed = functools.reduce(jnp.add, tier_rows[1:], jnp.int32(0)) > 0
  if extra_starts:
    starts = jnp.asarray(extra_starts, jnp.int32)
    active = jnp.sum(starts < slots.assignments, dtype=jnp.int32)

    def fixed_tiers(base):
      if len(maps) == 1:
        return ()
      # Tiers fill in order, so the first overflow tier holds rows whenever a later one does.
      first = run_tier(maps[1], group_sizes[1], is_overflow=True)
      return (first, *(skippable_tier(tier, base) for tier in range(2, len(maps))))

    def with_dropless_tiers(base):
      total = sum_recomputed_tiers(active, starts, dropless_tier, (local_ids.shape[0],) + base.shape[1:])
      for value, row_map in zip((base, *fixed_tiers(base)), maps):
        total = accumulate_rows(total, value, row_map)
      return total.astype(base.dtype)

    # Each branch computes its whole result, so no large array passes through
    # a branch unchanged and a step without dropless rows does exactly the
    # work, and saves exactly the residuals, of the configured tiers alone.
    branches = [lambda base: reduce_rows((base,), maps[:1])]
    index = jnp.int32(0)
    if len(maps) > 1:
      branches.append(lambda base: reduce_rows((base, *fixed_tiers(base)), maps))
      index += overflowed
    branches.append(with_dropless_tiers)
    index += active > 0
    if recompute_tier_switch:
      recv_output = switch_on_base(index, base_tier, branches)
    else:
      recv_output = jax.lax.switch(index, branches, base_tier())
  elif len(maps) > 1:
    outputs = [base_tier()]
    outputs += [skippable_tier(tier, outputs[0]) for tier in range(1, len(maps))]
    # Each slot column costs one gather per received row even when masked, so
    # the usually empty overflow tiers join the sum only when they hold rows.
    recv_output = jax.lax.cond(
        overflowed,
        lambda values: reduce_rows(values, maps),
        lambda values: reduce_rows(values[:1], maps[:1]),
        tuple(outputs),
    )
  else:
    recv_output = reduce_rows((base_tier(),), maps)
  if deepep:
    return deepep_moe.moe_combine(recv_output.astype(x.dtype), handle)
  returned = run_exchange(recv_output, combine, send_starts, send_rows, 1, zero_unwritten_rows)
  returned = checkpoint_name(returned, "moe_combined")
  return reduce_rows((returned,), (send_map,))
