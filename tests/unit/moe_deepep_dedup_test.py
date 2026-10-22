# Copyright 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Receiver-side expert tiers for DeepEP (use_deepep_dedup_receive) against the worst-case fan-out/fan-in path.

Synthetic received rows stand in for DeepEP's output: `recv_x [R, H]`, `recv_topk_idx [R, K]` (local expert id or -1)
and `recv_topk_weights [R, K]`. A toy per-expert linear layer is the expert, so the old path (fan out, run, unsort,
weighted fan-in) and the new path (tier, weight the input, run, reduce) must agree on outputs and gradients.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.layers import moe, moe_dedup

R, K, E, H = 48, 4, 6, 16  # received rows, slots per row, local experts, hidden


def _inputs(seed=0, valid_rows=40):
  """Synthetic DeepEP output: received rows, local expert ids (-1 = empty slot), router weights, expert weights."""
  rng = np.random.default_rng(seed)
  idx = np.full((R, K), -1, np.int32)
  for r in range(valid_rows):
    n = rng.integers(1, K + 1)
    idx[r, :n] = rng.choice(E, size=n, replace=False)
    rng.shuffle(idx[r])  # valid ids need not sit in the first columns
  weights = rng.uniform(0.1, 1.0, (R, K)).astype(np.float32)
  recv_x = rng.standard_normal((R, H)).astype(np.float32)
  expert_w = rng.standard_normal((E, H, H)).astype(np.float32) / np.sqrt(H)
  return jnp.asarray(recv_x), jnp.asarray(idx), jnp.asarray(weights), jnp.asarray(expert_w)


def _expert(rows, expert_ids, expert_w):
  """Per-row linear layer: row i goes through the matrix of expert_ids[i]."""
  return jnp.einsum("nh,nhk->nk", rows, expert_w[expert_ids])


def _old_path(recv_x, idx, weights, expert_w):
  """The worst-case fan-out / unsort / weighted fan-in that use_deepep_dedup_receive replaces."""
  flat = idx.reshape(-1)
  expert_ids = jnp.where(flat >= 0, flat, E).astype(jnp.int32)
  group_sizes = jnp.bincount(expert_ids, length=E + 1)[:E]
  sort_idx = jnp.argsort(expert_ids)
  x = moe._deepep_dispatch_fan_out(recv_x, sort_idx, K)  # pylint: disable=protected-access
  x = jnp.where((jnp.arange(x.shape[0]) < jnp.sum(group_sizes))[:, None], x, 0)
  selected = jnp.where(expert_ids < E, expert_ids, 0)[sort_idx]
  out = _expert(x, selected, expert_w)
  out = jnp.where((jnp.arange(out.shape[0]) < jnp.sum(group_sizes))[:, None], out, 0)
  unsorted = out[jnp.argsort(sort_idx)]
  return moe.deepep_fan_in(unsorted, weights, idx, R)


def _new_path(recv_x, idx, weights, expert_w, capacity):
  tiers = moe_dedup.build_expert_tiers(idx, E, (capacity,))
  row_map, group_sizes = tiers.maps[0], tiers.group_sizes[0]
  x = moe_dedup.expand_rows(recv_x, row_map)
  row_w = moe_dedup.slots_to_rows(weights, row_map)
  selected = jnp.minimum(jnp.searchsorted(jnp.cumsum(group_sizes), jnp.arange(x.shape[0]), side="right"), E - 1)
  out = _expert(x * row_w[:, None], selected, expert_w)  # router weight on the expert input
  return moe_dedup.reduce_rows((out,), (row_map,)), tiers


class MoeDeepepDedupTest(unittest.TestCase):

  def test_outputs_match_old_path_when_nothing_drops(self):
    args = _inputs()
    want = _old_path(*args)
    got, tiers = _new_path(*args, capacity=R * K)
    self.assertEqual(int(tiers.assignments), int(jnp.sum(tiers.group_sizes[0])))
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)
    self.assertTrue(np.all(np.asarray(got)[40:] == 0), "invalid received rows must produce zero output")

  def test_gradients_match_old_path(self):
    recv_x, idx, weights, expert_w = _inputs(seed=1)
    cotangent = jnp.asarray(np.random.default_rng(7).standard_normal((R, H)).astype(np.float32))

    def old_loss(rx, w, ew):
      return jnp.sum(_old_path(rx, idx, w, ew) * cotangent)

    def new_loss(rx, w, ew):
      return jnp.sum(_new_path(rx, idx, w, ew, R * K)[0] * cotangent)

    want = jax.grad(old_loss, argnums=(0, 1, 2))(recv_x, weights, expert_w)
    got = jax.grad(new_loss, argnums=(0, 1, 2))(recv_x, weights, expert_w)
    for name, g, w in zip(("recv_x", "weights", "expert_w"), got, want):
      np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=1e-4, atol=1e-4, err_msg=name)

  def test_capacity_drops_exactly_the_overflow(self):
    args = _inputs(seed=2)
    total = int(jnp.sum(args[1] >= 0))
    capacity = total - 7
    got, tiers = _new_path(*args, capacity=capacity)
    self.assertEqual(int(tiers.assignments), total)
    self.assertEqual(int(jnp.sum(tiers.group_sizes[0])), capacity)
    self.assertEqual(int(tiers.assignments - jnp.sum(tiers.group_sizes[0])), 7)
    self.assertFalse(np.isnan(np.asarray(got)).any())

  def test_drop_check_raises_only_when_something_dropped(self):
    moe._check_no_deepep_dedup_drops(0, 1234)  # pylint: disable=protected-access
    with self.assertRaisesRegex(RuntimeError, "dropped 7 of 1234"):
      moe._check_no_deepep_dedup_drops(7, 1234)  # pylint: disable=protected-access

  def test_drop_check_agrees_with_the_tier_under_jit(self):
    """The dropped count the route feeds the check is assignments - rows that fit; a roomy tier gives 0, a tight one > 0."""
    _, idx, _, _ = _inputs(seed=4)
    total = int(jnp.sum(idx >= 0))
    for capacity, want in ((R * K, 0), (total - 5, 5)):
      tiers = jax.jit(lambda i, c=capacity: moe_dedup.build_expert_tiers(i, E, (c,)))(idx)
      self.assertEqual(int(tiers.assignments - jnp.sum(tiers.group_sizes[0])), want)

  def test_scale_rows_matches_float32_to_bf16_rounding(self):
    rng = np.random.default_rng(5)
    h = jnp.asarray(rng.standard_normal((64, 128)), jnp.bfloat16)
    w = jnp.asarray(rng.uniform(0.05, 1.0, (64, 1)), jnp.float32)
    got = moe._scale_rows(h, w)  # pylint: disable=protected-access
    want = h.astype(jnp.float32) * w
    self.assertEqual(got.dtype, jnp.bfloat16)
    # one extra bf16 rounding of the weight (2**-9) plus the product: well under 1%
    np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(want), rtol=1e-2, atol=1e-6)

  def test_group_sizes_count_assignments_per_expert(self):
    _, idx, _, _ = _inputs(seed=3)
    tiers = moe_dedup.build_expert_tiers(idx, E, (R * K,))
    want = np.bincount(np.asarray(idx).reshape(-1)[np.asarray(idx).reshape(-1) >= 0], minlength=E)
    np.testing.assert_array_equal(np.asarray(tiers.group_sizes[0]), want)


def _toy_expert_fn(expert_w):
  """Per-row linear layer; the expert of each row comes from the tier's group sizes, as in the real expert step."""

  def fn(rows, row_weights, group_sizes, *, save_residuals):
    del save_residuals
    expert = jnp.minimum(jnp.searchsorted(jnp.cumsum(group_sizes), jnp.arange(rows.shape[0]), side="right"), E - 1)
    return jnp.einsum("nh,nhk->nk", rows * row_weights[:, None], expert_w[expert])

  return fn


def _receive_combine(recv_x, idx, weights, expert_w, capacity, dropless):  # pylint: disable=too-many-positional-arguments
  return moe_dedup.dedup_receive_combine(
      recv_x, idx, weights, _toy_expert_fn(expert_w), num_experts=E, capacity=capacity, dropless=dropless
  )


class DedupDroplessTiersTest(unittest.TestCase):
  """`dedup_receive_combine`: the MXFP4 branch's dropless tiers on DeepEP's received rows."""

  def test_many_extra_tiers_match_the_fanout_path_forward_and_backward(self):
    recv_x, idx, weights, expert_w = _inputs(seed=6)
    total = int(jnp.sum(idx >= 0))
    capacity = 16
    self.assertGreater(total, 5 * capacity, "the test needs several extra tiers")
    cotangent = jnp.asarray(np.random.default_rng(8).standard_normal((R, H)).astype(np.float32))

    got, dropped, assignments = _receive_combine(recv_x, idx, weights, expert_w, capacity, dropless=True)
    self.assertEqual(int(dropped), 0)
    self.assertEqual(int(assignments), total)
    np.testing.assert_allclose(np.asarray(got), np.asarray(_old_path(recv_x, idx, weights, expert_w)), rtol=1e-5, atol=1e-5)

    def old_loss(rx, w, ew):
      return jnp.sum(_old_path(rx, idx, w, ew) * cotangent)

    def new_loss(rx, w, ew):
      return jnp.sum(_receive_combine(rx, idx, w, ew, capacity, True)[0] * cotangent)

    want = jax.grad(old_loss, argnums=(0, 1, 2))(recv_x, weights, expert_w)
    for name, g, w in zip(
        ("recv_x", "weights", "expert_w"), jax.jit(jax.grad(new_loss, argnums=(0, 1, 2)))(recv_x, weights, expert_w), want
    ):
      np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=1e-4, atol=1e-4, err_msg=name)

  def test_without_overflow_it_is_the_single_tier_result(self):
    recv_x, idx, weights, expert_w = _inputs(seed=7)
    single = _receive_combine(recv_x, idx, weights, expert_w, R * K, dropless=False)[0]
    tiered = _receive_combine(recv_x, idx, weights, expert_w, R * K, dropless=True)[0]
    np.testing.assert_array_equal(np.asarray(single), np.asarray(tiered))

  def test_dropped_count_is_the_overflow_and_zero_when_dropless(self):
    recv_x, idx, weights, expert_w = _inputs(seed=9)
    total = int(jnp.sum(idx >= 0))
    _, dropped, assignments = _receive_combine(recv_x, idx, weights, expert_w, 40, dropless=False)
    self.assertEqual((int(dropped), int(assignments)), (total - 40, total))
    self.assertEqual(int(_receive_combine(recv_x, idx, weights, expert_w, 40, dropless=True)[1]), 0)

  def test_no_assignments_gives_zeros(self):
    recv_x, _, weights, expert_w = _inputs(seed=10)
    empty = jnp.full((R, K), -1, jnp.int32)
    got, dropped, assignments = _receive_combine(recv_x, empty, weights, expert_w, 16, dropless=True)
    self.assertEqual((int(dropped), int(assignments)), (0, 0))
    self.assertTrue(np.all(np.asarray(got) == 0))

  def test_extra_tiers_save_nothing_beyond_the_single_tier_version(self):
    recv_x, idx, weights, expert_w = _inputs(seed=11)
    policy = jax.checkpoint_policies.save_only_these_names("moe_dedup_plan", "moe_dedup_received")

    def saved_float_shapes(dropless):
      fn = jax.checkpoint(lambda rx, w, ew: _receive_combine(rx, idx, w, ew, 16, dropless)[0], policy=policy)
      _, vjp = jax.vjp(fn, recv_x, weights, expert_w)
      return sorted(tuple(x.shape) for x in jax.tree.leaves(vjp) if jnp.issubdtype(x.dtype, jnp.inexact))

    # The checkpointed function's inputs (received rows, router weights, expert weights) are saved either way; the loop's
    # internals are recomputed in backward, so the extra tiers add no saved float array on top of them.
    self.assertEqual(saved_float_shapes(True), saved_float_shapes(False))
    self.assertIn((R, H), saved_float_shapes(True))


if __name__ == "__main__":
  unittest.main()
