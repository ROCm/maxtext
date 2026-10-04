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
  rows, slots_per_row = local_ids.shape
  key = jnp.where(local_ids >= 0, local_ids, num_experts).reshape(-1)
  sort_key = key.astype(jnp.uint8) if num_experts <= jnp.iinfo(jnp.uint8).max else key
  order = jnp.argsort(sort_key, stable=True).astype(jnp.int32)
  position = jnp.zeros_like(order).at[order].set(jnp.arange(order.size, dtype=jnp.int32), unique_indices=True)
  group_sizes = jnp.sum(key[:, None] == jnp.arange(num_experts, dtype=key.dtype)[None, :], axis=0, dtype=jnp.int32)
  group_ends = jnp.cumsum(group_sizes)
  assignments = group_ends[-1]
  is_slot = key < num_experts

  maps, tier_sizes = [], []
  start = 0
  for capacity in capacities:
    capacity = min(int(capacity), order.size - start)
    if capacity <= 0:
      break
    tier_position = position - start
    slot_valid = is_slot & (tier_position >= 0) & (tier_position < capacity)
    maps.append(
        _materialize(
            RowMap(
                row_slots=jax.lax.slice_in_dim(order, start, start + capacity),
                row_valid=jnp.arange(capacity, dtype=jnp.int32) < jnp.clip(assignments - start, 0, capacity),
                slot_rows=jnp.where(slot_valid, tier_position, 0).reshape(rows, slots_per_row),
                slot_valid=slot_valid.reshape(rows, slots_per_row),
            )
        )
    )
    tier_ends = jnp.clip(group_ends - start, 0, capacity)
    tier_sizes.append(tier_ends - jnp.concatenate((jnp.zeros((1,), tier_ends.dtype), tier_ends[:-1])))
    start += capacity
  return ExpertTiers(tuple(maps), tuple(tier_sizes), assignments)


def exchange_params(traffic: jax.Array, shard_id, *, is_dispatch: bool):
  """`ragged_all_to_all` offsets and sizes for a packed `[sender, receiver]` row-count matrix."""
  matrix = traffic if is_dispatch else traffic.T
  send_sizes = matrix[shard_id]
  input_offsets = jnp.cumsum(send_sizes) - send_sizes
  output_offsets = (jnp.cumsum(matrix, axis=0) - matrix)[shard_id]
  recv_sizes = matrix[:, shard_id]
  return input_offsets, send_sizes, output_offsets, recv_sizes


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
) -> jax.Array:
  """Routed-expert output for this shard's tokens, with one exchange row per (token, shard).

  Must run inside a `shard_map` over `axis_name`, with experts sharded
  contiguously: shard `s` owns experts `[s * E, (s + 1) * E)`.

  Args:
    x: `[T, H]` token activations.
    expert_ids: `[T, K]` global expert id of each assignment.
    weights: `[T, K]` routing weight of each assignment.
    expert_fn: `expert_fn(rows, row_weights, group_sizes, is_overflow=...)` maps
      `[C, H]` expert-sorted rows and their `[C]` routing weights to the
      weighted `[C, H_out]` expert outputs of one capacity tier. Rows not
      holding an assignment are zero and their outputs are ignored.
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

  exchange = exchange or moe._ragged_all_to_all_fresh  # pylint: disable=protected-access
  recv_x = exchange(
      expand_rows(x, send_map), *dispatch, recv_starts, recv_rows, axis_name, 0, zero_unwritten_rows
  )
  recv_meta = exchange(send_meta, *dispatch, recv_starts, recv_rows, axis_name, 2, True)
  recv_meta = checkpoint_name(recv_meta, "moe_dispatched_meta")
  local_ids = jnp.round(recv_meta[:, :top_k]).astype(jnp.int32) - 1
  recv_weights = recv_meta[:, top_k:]

  tiers = build_expert_tiers(local_ids, local_experts, capacities)
  tier_rows = [jnp.sum(sizes) for sizes in tiers.group_sizes]
  dropped = tiers.assignments - functools.reduce(jnp.add, tier_rows)
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

  def run_tier(tier, _):
    return expert_fn(
        expand_rows(recv_x, maps[tier]),
        slots_to_rows(recv_weights, maps[tier]),
        group_sizes[tier],
        is_overflow=tier > 0,
    )

  outputs = [run_tier(0, None)]
  for tier in range(1, len(maps)):
    rows = maps[tier].row_slots.shape[0]
    outputs.append(
        jax.lax.cond(
            tier_rows[tier] > 0,
            functools.partial(run_tier, tier),
            lambda _, rows=rows: jnp.zeros((rows,) + outputs[0].shape[1:], outputs[0].dtype),
            None,
        )
    )
  if len(outputs) > 1:
    # Each slot column costs one gather per received row even when masked, so
    # the usually empty overflow tiers join the sum only when they hold rows.
    recv_output = jax.lax.cond(
        functools.reduce(jnp.add, tier_rows[1:]) > 0,
        lambda values: reduce_rows(values, maps),
        lambda values: reduce_rows(values[:1], maps[:1]),
        tuple(outputs),
    )
  else:
    recv_output = reduce_rows(tuple(outputs), maps)
  returned = exchange(recv_output, *combine, send_starts, send_rows, axis_name, 1, zero_unwritten_rows)
  returned = checkpoint_name(returned, "moe_combined")
  return reduce_rows((returned,), (send_map,))
