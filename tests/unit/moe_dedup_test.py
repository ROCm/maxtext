# Copyright 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P

from maxtext.kernels import jax_aiter_mxfp4
from maxtext.layers import moe_dedup


def _group_limited_ids(rng, tokens, num_shards, local_experts, top_k, shards_per_token):
  """Expert ids drawn from `shards_per_token` random shards per token, as DeepSeek group-limited routing."""
  ids = np.empty((tokens, top_k), np.int32)
  for t in range(tokens):
    shards = rng.choice(num_shards, shards_per_token, replace=False)
    candidates = (shards[:, None] * local_experts + np.arange(local_experts)).reshape(-1)
    ids[t] = rng.choice(candidates, top_k, replace=False)
  return ids


def _reference_expand(values, row_map):
  owner = np.asarray(row_map.row_slots) // row_map.slot_rows.shape[1]
  return jnp.where(row_map.row_valid[:, None], values[jnp.asarray(np.where(row_map.row_valid, owner, 0))], 0)


def test_send_map_has_one_row_per_token_and_shard():
  rng = np.random.default_rng(0)
  tokens, num_shards, local_experts, top_k, slots = 50, 8, 4, 6, 3
  ids = _group_limited_ids(rng, tokens, num_shards, local_experts, top_k, slots)
  row_map, slot_shards, counts = jax.jit(moe_dedup.build_send_map, static_argnums=(1, 2))(
      jnp.asarray(ids // local_experts), num_shards, slots
  )

  pairs = sorted({(int(d), t) for t in range(tokens) for d in ids[t] // local_experts})
  np.testing.assert_array_equal(counts, np.bincount([d for d, _ in pairs], minlength=num_shards))
  valid = np.asarray(row_map.row_valid)
  assert row_map.row_slots.shape == (tokens * slots,) and valid.sum() == len(pairs)
  held = np.asarray(row_map.row_slots)[valid]
  slot_shards = np.asarray(slot_shards)
  # Rows are grouped by destination shard, then ordered by token.
  assert [(int(slot_shards.reshape(-1)[s]), int(s // slots)) for s in held] == pairs
  slot_rows, slot_valid = np.asarray(row_map.slot_rows), np.asarray(row_map.slot_valid)
  for t in range(tokens):
    # Slots hold the token's destinations in shard order, valid ones first.
    dests = sorted(set(ids[t] // local_experts))
    np.testing.assert_array_equal(slot_valid[t], np.arange(slots) < len(dests))
    np.testing.assert_array_equal(slot_shards[t, : len(dests)], dests)
  np.testing.assert_array_equal(np.asarray(row_map.row_slots)[slot_rows[slot_valid]], np.flatnonzero(slot_valid))


def test_send_map_skips_destinations_beyond_the_slot_bound():
  dest = jnp.asarray([[0, 1, 2, 2], [3, 3, 3, 3]], jnp.int32)
  row_map, slot_shards, counts = moe_dedup.build_send_map(dest, 4, 2)
  np.testing.assert_array_equal(counts, [1, 1, 0, 1])
  np.testing.assert_array_equal(slot_shards, [[0, 1], [3, 0]])
  np.testing.assert_array_equal(row_map.slot_valid, [[True, True], [True, False]])
  assert int(np.sum(row_map.row_valid)) == 3


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_expand_and_reduce_rows_match_autodiff_reference(dtype):
  rng = np.random.default_rng(1)
  tokens, num_shards, local_experts = 40, 4, 4
  ids = _group_limited_ids(rng, tokens, num_shards, local_experts, 4, 3)
  row_map, _, _ = moe_dedup.build_send_map(jnp.asarray(ids // local_experts), num_shards, 3)
  x = jnp.asarray(rng.standard_normal((tokens, 8)), dtype)
  y = jnp.asarray(rng.standard_normal((tokens * 3, 8)), dtype)

  def reference_reduce(values):
    owner = jnp.where(row_map.row_valid, row_map.row_slots // 3, tokens)
    return jax.ops.segment_sum(values.astype(jnp.float32), owner, tokens + 1)[:tokens].astype(values.dtype)

  np.testing.assert_array_equal(moe_dedup.expand_rows(x, row_map), _reference_expand(x, row_map))
  np.testing.assert_allclose(
      np.asarray(moe_dedup.reduce_rows((y,), (row_map,)), np.float32),
      np.asarray(reference_reduce(y), np.float32),
      rtol=1e-6 if dtype == jnp.float32 else 1e-2,
      atol=1e-6 if dtype == jnp.float32 else 1e-2,
  )

  loss = lambda f: lambda v: jnp.sum(jnp.sin(f(v).astype(jnp.float32)) * jnp.arange(8.0))
  expand_grad = jax.grad(loss(lambda v: moe_dedup.expand_rows(v, row_map)))(x)
  expand_ref = jax.grad(loss(lambda v: _reference_expand(v, row_map)))(x)
  reduce_grad = jax.grad(loss(lambda v: moe_dedup.reduce_rows((v,), (row_map,))))(y)
  reduce_ref = jax.grad(loss(reference_reduce))(y)
  tolerance = 1e-6 if dtype == jnp.float32 else 2e-2
  np.testing.assert_allclose(np.asarray(expand_grad, np.float32), np.asarray(expand_ref, np.float32), rtol=tolerance, atol=tolerance)
  # The forward sums differ in summation order, so the loss cotangents differ in the last bits.
  np.testing.assert_allclose(np.asarray(reduce_grad, np.float32), np.asarray(reduce_ref, np.float32), rtol=tolerance, atol=tolerance)


def test_slots_to_rows_matches_autodiff_reference():
  rng = np.random.default_rng(2)
  ids = _group_limited_ids(rng, 30, 4, 4, 4, 2)
  row_map, _, _ = moe_dedup.build_send_map(jnp.asarray(ids // 4), 4, 2)
  values = jnp.asarray(rng.standard_normal((30, 2, 3)), jnp.float32)

  def reference(v):
    return jnp.where(row_map.row_valid[:, None], v.reshape(-1, 3)[row_map.row_slots], 0)

  np.testing.assert_array_equal(moe_dedup.slots_to_rows(values, row_map), reference(values))
  loss = lambda f: lambda v: jnp.sum(jnp.cos(f(v)) * jnp.arange(3.0))
  np.testing.assert_array_equal(
      jax.grad(loss(lambda v: moe_dedup.slots_to_rows(v, row_map)))(values), jax.grad(loss(reference))(values)
  )


@pytest.mark.parametrize(
    "capacities, expected_tiers",
    [
        ((16,), [[3, 0, 5, 2]]),  # everything fits in one tier
        ((6, 6), [[3, 0, 3, 0], [0, 0, 2, 2]]),  # an expert straddles the tier boundary
        ((4, 3), [[3, 0, 1, 0], [0, 0, 3, 0]]),  # three rows of the last experts are dropped
        ((32, 32), [[3, 0, 5, 2]]),  # capacity beyond the slot count is clipped
    ],
)
def test_expert_tiers_split_sorted_slots(capacities, expected_tiers):
  local_ids = jnp.asarray([[2, -1, 0, 3], [0, 2, -1, -1], [2, 2, 3, -1], [0, 2, -1, -1]], jnp.int32)
  tiers = moe_dedup.build_expert_tiers(local_ids, 4, capacities)
  assert int(tiers.assignments) == 10
  assert [np.asarray(sizes).tolist() for sizes in tiers.group_sizes] == expected_tiers

  flat = np.asarray(local_ids).reshape(-1)
  start = 0
  sorted_slots = np.argsort(np.where(flat >= 0, flat, 4), kind="stable")
  for row_map, sizes in zip(tiers.maps, tiers.group_sizes):
    count = int(np.sum(sizes))
    valid = np.asarray(row_map.row_valid)
    assert valid.sum() == count
    np.testing.assert_array_equal(np.asarray(row_map.row_slots)[:count], sorted_slots[start : start + count])
    slot_rows, slot_valid = np.asarray(row_map.slot_rows).reshape(-1), np.asarray(row_map.slot_valid).reshape(-1)
    np.testing.assert_array_equal(np.asarray(row_map.row_slots)[slot_rows[slot_valid]], np.flatnonzero(slot_valid))
    assert slot_valid.sum() == count
    start += count


def test_composed_map_equals_padding_the_compact_rows():
  rng = np.random.default_rng(4)
  local_ids = jnp.asarray([[2, -1, 0, 3], [0, 2, -1, -1], [2, 2, 3, -1], [0, 2, -1, -1]], jnp.int32)
  tiers = moe_dedup.build_expert_tiers(local_ids, 4, (6, 6))
  values = jnp.asarray(rng.standard_normal((4, 3)), jnp.float32)
  for row_map, sizes in zip(tiers.maps, tiers.group_sizes):
    layout = jax_aiter_mxfp4.expert_padding(sizes, row_map.row_slots.shape[0])
    padded_map = moe_dedup.compose_row_map(row_map, layout)
    padded = moe_dedup.expand_rows(values, padded_map)
    expected, _ = jax_aiter_mxfp4.pad_expert_rows(moe_dedup.expand_rows(values, row_map), sizes)
    np.testing.assert_array_equal(padded, expected)
    outputs = jnp.asarray(rng.standard_normal(padded.shape), jnp.float32)
    np.testing.assert_allclose(
        moe_dedup.reduce_rows((outputs,), (padded_map,)),
        moe_dedup.reduce_rows((jax_aiter_mxfp4.unpad_expert_rows(outputs, layout),), (row_map,)),
        rtol=1e-6,
    )


def test_expert_tiers_with_no_slots():
  tiers = moe_dedup.build_expert_tiers(jnp.full((3, 2), -1, jnp.int32), 4, (4, 2))
  assert int(tiers.assignments) == 0
  assert all(int(np.sum(sizes)) == 0 for sizes in tiers.group_sizes)
  assert not any(bool(np.any(row_map.row_valid)) or bool(np.any(row_map.slot_valid)) for row_map in tiers.maps)


@jax.custom_vjp
def _nan_gradient_rows(values, live):
  """Identity whose gradient is NaN in rows where `live` is false."""
  del live
  return values


def _nan_gradient_rows_fwd(values, live):
  return values, live


def _nan_gradient_rows_bwd(live, grad):
  return jnp.where(live.reshape(live.shape + (1,) * (grad.ndim - 1)), grad, jnp.nan), None


_nan_gradient_rows.defvjp(_nan_gradient_rows_fwd, _nan_gradient_rows_bwd)


def _emulated_exchange(
    operand,
    input_offsets,
    send_sizes,
    output_offsets,
    recv_sizes,
    output_row_starts,
    output_rows,
    axis_name,
    site,
    zero_unwritten=True,
):
  """`ragged_all_to_all` into a fresh buffer, built from `all_to_all` for backends without the ragged op.

  With `zero_unwritten` false, unwritten output rows are NaN, and so are the
  gradients of operand rows that were not sent.
  """
  del output_row_starts, site
  rows = operand.shape[0]
  i = jnp.arange(rows)
  in_range = i[None, :] < send_sizes[:, None]
  if not zero_unwritten:
    sent_rows = jnp.where(in_range, input_offsets[:, None] + i[None, :], rows)
    operand = _nan_gradient_rows(operand, jnp.zeros((rows,), bool).at[sent_rows].set(True, mode="drop"))
  chunks = _take_rows(operand, input_offsets[:, None] + i[None, :], in_range)
  received = jax.lax.all_to_all(chunks, axis_name, 0, 0)
  offsets = jax.lax.all_to_all(output_offsets[:, None], axis_name, 0, 0)[:, 0]
  targets = jnp.where(i[None, :] < recv_sizes[:, None], offsets[:, None] + i[None, :], output_rows)
  output = jnp.full((output_rows,) + operand.shape[1:], 0 if zero_unwritten else jnp.nan, operand.dtype)
  return output.at[targets.reshape(-1)].set(received.reshape((-1,) + operand.shape[1:]), mode="drop")


def _take_rows(values, indices, valid):
  taken = values[jnp.clip(indices, 0, values.shape[0] - 1)]
  return jnp.where(valid[..., None], taken, 0)


def _expected_kept(ids, num_shards, local_experts, capacity):
  """Assignments kept under the receiver's sort order: sender, token, then slot, within each local expert."""
  kept = np.zeros(ids.shape, bool)
  for shard in range(num_shards):
    slots = []
    for sender in range(ids.shape[0]):
      for t in range(ids.shape[1]):
        dests = ids[sender, t] // local_experts
        if shard not in dests:
          continue
        for k in range(ids.shape[2]):
          if dests[k] == shard:
            slots.append((ids[sender, t, k] % local_experts, len(slots), sender, t, k))
    for _, _, sender, t, k in sorted(slots)[:capacity]:
      kept[sender, t, k] = True
  return kept


def _exchange_or_skip(exchange):
  """The `exchange` argument for an eight-shard test, skipping where it cannot run."""
  devices = jax.devices()
  if len(devices) < 8:
    pytest.skip("requires eight devices; on CPU set --xla_force_host_platform_device_count=8")
  if exchange == "emulated":
    return _emulated_exchange
  if devices[0].platform != "gpu":
    pytest.skip("ragged_all_to_all needs a GPU backend")
  pytest.importorskip("jax_aiter.ops.buffers")
  return None


@pytest.mark.parametrize("zero_unwritten_rows", [True, False], ids=["zeroed", "unread_rows_nan"])
@pytest.mark.parametrize("layout", ["compact", "padded"])
@pytest.mark.parametrize("exchange", ["emulated", "fresh"])
@pytest.mark.parametrize("capacities", [(48, 160), (40, 20)], ids=["two_tier_dropless", "two_tier_drops"])
def test_dedup_dispatch_combine_matches_dense_reference(capacities, exchange, layout, zero_unwritten_rows):
  exchange_fn = _exchange_or_skip(exchange)
  devices = jax.devices()
  num_shards, local_experts, top_k, tokens, hidden, hidden_out = 8, 4, 4, 24, 8, 6
  num_experts = num_shards * local_experts
  rng = np.random.default_rng(3)
  ids = np.stack([_group_limited_ids(rng, tokens, num_shards, local_experts, top_k, 2) for _ in range(num_shards)])
  x = jnp.asarray(rng.standard_normal((num_shards, tokens, hidden)), jnp.float32)
  weights = jnp.asarray(rng.random((num_shards, tokens, top_k)), jnp.float32)
  kernels = jnp.asarray(rng.standard_normal((num_experts, hidden, hidden_out)) / 3, jnp.float32)
  cotangent = jnp.asarray(rng.standard_normal((num_shards, tokens, hidden_out)), jnp.float32)
  kept = _expected_kept(ids, num_shards, local_experts, sum(capacities))
  per_shard = np.bincount((ids // local_experts).reshape(-1), minlength=num_shards)
  if capacities == (48, 160):
    assert kept.all() and per_shard.max() > 48, "test needs the overflow tier and no drops"
  else:
    assert not kept.all()

  def expert_fn(rows, row_weights, group_sizes, *, is_overflow, local_kernels):
    del is_overflow
    expert = jnp.repeat(jnp.arange(local_experts), group_sizes, total_repeat_length=rows.shape[0])
    if zero_unwritten_rows:
      return jnp.einsum("ch,cho->co", rows, local_kernels[expert]) * row_weights[:, None]
    # Like a grouped GEMM with zero_tail off: rows past the last group are NaN
    # in the output and in the input gradient.
    live = jnp.arange(rows.shape[0]) < jnp.sum(group_sizes)
    output = jnp.einsum("ch,cho->co", _nan_gradient_rows(rows, live), local_kernels[expert]) * row_weights[:, None]
    return jnp.where(live[:, None], output, jnp.nan)

  mesh = Mesh(np.asarray(devices[:num_shards]), ("expert",))

  @jax.shard_map(
      mesh=mesh,
      in_specs=(P("expert"), P("expert"), P("expert"), P("expert")),
      out_specs=P("expert"),
      check_vma=False,
  )
  def dedup(x, ids, weights, kernels):
    return moe_dedup.dedup_dispatch_combine(
        x[0],
        ids[0],
        weights[0],
        functools.partial(expert_fn, local_kernels=kernels),
        num_experts=num_experts,
        num_shards=num_shards,
        axis_name="expert",
        max_shards_per_token=2,
        capacities=capacities,
        exchange=exchange_fn,
        row_layout=jax_aiter_mxfp4.expert_padding if layout == "padded" else None,
        zero_unwritten_rows=zero_unwritten_rows,
    )[None]

  def reference(x, ids, weights, kernels):
    outputs = jnp.einsum("sth,stkho->stko", x, kernels[ids])
    return jnp.sum(outputs * (weights * kept)[..., None], axis=2)

  def loss(f):
    return lambda x, w, k: jnp.sum(f(x, jnp.asarray(ids), w, k) * cotangent)

  ids_array = jnp.asarray(ids)
  np.testing.assert_allclose(
      jax.jit(dedup)(x, ids_array, weights, kernels), reference(x, ids_array, weights, kernels), rtol=1e-5, atol=1e-5
  )
  grads = jax.jit(jax.grad(loss(dedup), argnums=(0, 1, 2)))(x, weights, kernels)
  expected = jax.grad(loss(reference), argnums=(0, 1, 2))(x, weights, kernels)
  for actual, wanted in zip(grads, expected):
    np.testing.assert_allclose(actual, wanted, rtol=1e-5, atol=1e-5)


def _primitive_count(jaxpr, name):
  """Uses of primitive `name` in `jaxpr` and in every jaxpr nested in its equations."""
  count = 0
  for eqn in jaxpr.eqns:
    count += eqn.primitive.name == name
    for param in eqn.params.values():
      for sub in param if isinstance(param, (tuple, list)) else (param,):
        sub = getattr(sub, "jaxpr", sub)
        if hasattr(sub, "eqns"):
          count += _primitive_count(sub, name)
  return count


@pytest.mark.parametrize("exchange", ["emulated", "fresh"])
@pytest.mark.parametrize("saved, passes", [((), 2), (("moe_dedup_plan",), 1)], ids=["recomputed", "saved"])
def test_saved_plan_is_not_rebuilt_in_backward(saved, passes, exchange):
  exchange_fn = _exchange_or_skip(exchange)
  devices = jax.devices()
  num_shards, local_experts, top_k, tokens, hidden = 8, 4, 4, 24, 8
  rng = np.random.default_rng(5)
  ids = jnp.asarray(np.stack([_group_limited_ids(rng, tokens, num_shards, local_experts, top_k, 2) for _ in range(num_shards)]))
  x = jnp.asarray(rng.standard_normal((num_shards, tokens, hidden)), jnp.float32)
  weights = jnp.asarray(rng.random((num_shards, tokens, top_k)), jnp.float32)
  mesh = Mesh(np.asarray(devices[:num_shards]), ("expert",))

  @jax.shard_map(mesh=mesh, in_specs=(P("expert"),) * 3, out_specs=P("expert"), check_vma=False)
  def dedup(x, ids, weights):
    return moe_dedup.dedup_dispatch_combine(
        x[0],
        ids[0],
        weights[0],
        lambda rows, row_weights, group_sizes, *, is_overflow: jnp.tanh(rows) * row_weights[:, None],
        num_experts=num_shards * local_experts,
        num_shards=num_shards,
        axis_name="expert",
        max_shards_per_token=2,
        capacities=(48, 160),
        exchange=exchange_fn,
    )[None]

  def grad(f):
    return jax.grad(lambda x, w: jnp.sum(f(x, ids, w) ** 2), argnums=(0, 1))

  remat = jax.checkpoint(dedup, policy=jax.checkpoint_policies.save_only_these_names("moe_dispatched_meta", *saved))
  grad_jaxpr = jax.make_jaxpr(grad(remat))(x, weights).jaxpr
  assert _primitive_count(grad_jaxpr, "all_gather") == passes
  assert _primitive_count(grad_jaxpr, "sort") == passes
  for actual, wanted in zip(jax.jit(grad(remat))(x, weights), jax.jit(grad(dedup))(x, weights)):
    np.testing.assert_allclose(actual, wanted, rtol=1e-6, atol=1e-6)
