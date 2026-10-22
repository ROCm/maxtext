# Copyright 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Receiver-side expert tiers for DeepEP: bounded expert-sorted rows instead of a top_k-fold fan-out.

DeepEP delivers one row per (token, destination shard) with that row's local expert ids. The naive way to feed
the grouped GEMM is to copy every received row `top_k` times and sort, which sizes every array for the worst case.
Here the received slots are sorted by local expert and the first `capacity` real ones become the GEMM input, so the
expert-side arrays are sized for real traffic. Assignments past `capacity` are dropped, highest expert ids first.

All row movement is a masked gather whose custom gradient is again a gather, so no step needs a scatter-add with
repeated indices.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.extend import core as jax_core


class RowMap(NamedTuple):
  """Partial bijection between slots `[J, M]` and rows `[N]`.

  Row `i` holds slot `row_slots[i]`, flattened as `j * M + m`, when `row_valid[i]`. Slot `(j, m)` lives in row
  `slot_rows[j, m]` when `slot_valid[j, m]`. Indices of invalid entries are 0.
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


def _materialize(row_map: RowMap) -> RowMap:
  # Without the barrier XLA fuses the index arithmetic, including prefix sums, into each consuming gather and
  # repeats it for every hidden element.
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

  The transpose of `expand_rows`.
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


def sum_recomputed_tiers(
    count: jax.Array,
    starts: jax.Array,
    tier: Callable[[jax.Array], tuple[jax.Array, RowMap]],
    shape: Sequence[int],
) -> jax.Array:
  """f32 sum over `i < count` of `reduce_rows((values,), (row_map,))`, where `values, row_map = tier(starts[i])`.

  Forward and backward each loop over only the first `count` tiers and hold one tier's buffers at a time. The residuals
  are the arrays `tier` reads, so nothing the loop computes is saved: backward recomputes each tier and accumulates its
  cotangents. `tier` must be a pure function of `start` and the arrays it closes over (the expert weights and the
  received rows among them), and its output shape must not depend on `start`.
  """
  # Every array `tier` closes over becomes an explicit input, including integer ones: backward is traced after forward
  # and cannot read forward's tracers.
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


class SortedSlots(NamedTuple):
  """One shard's received slots `[R, K]` in local-expert order.

  Sorted position `p` holds flat slot `order[p]`; slot `(j, m)` sits at sorted position `position[j, m]`. Valid
  slots precede empty ones.
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
  """Static tiers of the given capacities over already-sorted slots, filled in order."""
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
    capacities: static row capacity of each tier, filled in order. Slots beyond the total capacity are dropped,
      highest expert ids first.

  Returns:
    One map and one group-size vector per tier, and the number of valid slots.
  """
  return _tiers(sort_slots(local_ids, num_experts), capacities)


def dedup_receive_combine(
    recv_x: jax.Array,
    recv_idx: jax.Array,
    recv_weights: jax.Array,
    expert_fn: Callable[..., jax.Array],
    *,
    num_experts: int,
    capacity: int,
    dropless: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Expert outputs summed per received row, from rows already delivered by DeepEP.

  The receiver half of the MXFP4 branch's `dedup_dispatch_combine`; DeepEP does the exchange, so there is no send map.

  Args:
    recv_x: `[R, H]` received rows.
    recv_idx: `[R, K]` local expert id of each received slot, -1 when empty (rows past the received count included).
    recv_weights: `[R, K]` router weight of each received slot.
    expert_fn: `expert_fn(rows, row_weights, group_sizes, *, save_residuals)` maps `[C, H]` expert-sorted rows, their `[C]`
      router weights and the `[E]` per-expert row counts to the weighted `[C, H_out]` expert outputs. With
      `save_residuals` false it must tag nothing with `checkpoint_name`, so that backward recomputes the tier.
    num_experts: number of local experts.
    capacity: static rows of the primary tier. With `dropless`, every later tier has the same size.
    dropless: add tiers after the primary one until every received slot has a row, so nothing drops. The tiers run in a
      loop, one at a time and only as many as hold rows; nothing the loop computes is saved, and backward recomputes each
      tier from the received rows. The number of tiers is bounded at compile time by `R * K`, so this holds for any routing.

  Returns:
    `(aggregated, dropped, assignments)`: `[R, H_out]` per-row sums, and the number of received slots that did not get a row
    (0 with `dropless`) out of the total.
  """
  slots = sort_slots(recv_idx, num_experts)
  tiers = _tiers(slots, (capacity,))
  row_map, sizes = tiers.maps[0], tiers.group_sizes[0]
  covered = row_map.row_slots.shape[0]
  extra_starts = tuple(range(covered, recv_idx.size, covered)) if dropless else ()
  if extra_starts:
    slots = checkpoint_name(slots, "moe_dedup_plan")
    recv_x = checkpoint_name(recv_x, "moe_dedup_received")
  row_map, sizes = checkpoint_name((row_map, sizes), "moe_dedup_plan")

  def run_tier(tier_map, tier_sizes, *, save_residuals):
    return expert_fn(
        expand_rows(recv_x, tier_map),
        slots_to_rows(recv_weights, tier_map),
        tier_sizes,
        save_residuals=save_residuals,
    )

  base = run_tier(row_map, sizes, save_residuals=True)
  rows_done = jnp.sum(sizes)
  if not extra_starts:
    return reduce_rows((base,), (row_map,)), slots.assignments - rows_done, slots.assignments

  starts = jnp.asarray(extra_starts, jnp.int32)
  active = jnp.sum(starts < slots.assignments, dtype=jnp.int32)
  rows_done += jnp.sum(jnp.clip(slots.assignments - starts, 0, covered))

  def tier_from(start):
    """The expert outputs and row map of the extra tier starting at sorted position `start` (traced)."""
    tier_map, tier_sizes = expert_tier(slots, start, covered)
    return run_tier(tier_map, tier_sizes, save_residuals=False), tier_map

  def with_extra_tiers(values):
    total = sum_recomputed_tiers(active, starts, tier_from, (recv_x.shape[0],) + values.shape[1:])
    return accumulate_rows(total, values, row_map).astype(values.dtype)

  # A step without overflow does exactly the work, and saves exactly the residuals, of the primary tier alone.
  aggregated = jax.lax.switch(
      (active > 0).astype(jnp.int32), [lambda values: reduce_rows((values,), (row_map,)), with_extra_tiers], base
  )
  return aggregated, slots.assignments - rows_done, slots.assignments
