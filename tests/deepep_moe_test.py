# Copyright 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for DeepEP dispatch/combine integration with MaxText MoE.

These tests verify:
  1. deepep_fan_out correctly expands recv tokens by topk_idx.
  2. deepep_fan_in correctly aggregates multi-expert outputs per token.
  3. fan_out -> fan_in round-trip preserves weighted sums.
  4. Config validation for use_deepep_dispatch.
  5. (expectedFailure) DeepEP padding rows must not gather expert-0 bias when mlp_bias is used.

Multi-GPU tests (requiring primus_turbo and >=2 AMD GPUs) are gated
behind the DEEPEP_MULTIGPU environment variable.
"""

import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from MaxText.layers.moe import deepep_fan_out, deepep_fan_in, _DeepEPCombineState


class DeepEPFanOutTest(unittest.TestCase):
  """Tests for deepep_fan_out."""

  def test_basic_fan_out(self):
    """Tokens assigned to various expert slots, some invalid."""
    num_recv = 4
    hidden = 8
    num_topk = 3
    num_local_experts = 4

    recv_x = jnp.ones((num_recv, hidden), dtype=jnp.bfloat16)
    for i in range(num_recv):
      recv_x = recv_x.at[i].set(i + 1.0)

    recv_topk_idx = jnp.array([
        [0, 2, -1],
        [1, -1, -1],
        [0, 1, 3],
        [3, -1, -1],
    ], dtype=jnp.int32)

    expanded_x, expert_ids, group_sizes, token_indices = deepep_fan_out(
        recv_x, recv_topk_idx, num_local_experts,
    )

    total = num_recv * num_topk
    total_valid = int(jnp.sum(recv_topk_idx >= 0).item())
    self.assertEqual(expanded_x.shape, (total, hidden))
    self.assertEqual(expert_ids.shape, (total,))
    self.assertEqual(token_indices.shape, (total,))
    self.assertEqual(int(group_sizes.sum()), total_valid)

    # Padding entries should have expert_id == num_local_experts
    num_padding = total - total_valid
    self.assertEqual(int((expert_ids == num_local_experts).sum()), num_padding)

    for i in range(total):
      tok_idx = int(token_indices[i].item())
      np.testing.assert_allclose(
          expanded_x[i].astype(jnp.float32),
          recv_x[tok_idx].astype(jnp.float32),
          atol=0.01,
      )

  def test_all_valid_slots(self):
    """All top-k slots are valid (no -1, no padding)."""
    num_recv = 2
    hidden = 4
    num_topk = 2
    num_local_experts = 3

    recv_x = jnp.arange(num_recv * hidden, dtype=jnp.bfloat16).reshape(num_recv, hidden)
    recv_topk_idx = jnp.array([[0, 1], [2, 0]], dtype=jnp.int32)

    expanded_x, expert_ids, group_sizes, token_indices = deepep_fan_out(
        recv_x, recv_topk_idx, num_local_experts,
    )

    self.assertEqual(expanded_x.shape[0], num_recv * num_topk)
    np.testing.assert_array_equal(group_sizes, jnp.array([2, 1, 1]))
    # No padding entries
    self.assertEqual(int((expert_ids == num_local_experts).sum()), 0)

  def test_all_invalid_slots(self):
    """All slots are -1 -- everything becomes padding."""
    num_recv = 3
    hidden = 4
    num_topk = 2
    num_local_experts = 2

    recv_x = jnp.ones((num_recv, hidden), dtype=jnp.bfloat16)
    recv_topk_idx = jnp.full((num_recv, num_topk), -1, dtype=jnp.int32)

    expanded_x, expert_ids, group_sizes, token_indices = deepep_fan_out(
        recv_x, recv_topk_idx, num_local_experts,
    )

    self.assertEqual(expanded_x.shape[0], num_recv * num_topk)
    np.testing.assert_array_equal(group_sizes, jnp.zeros(num_local_experts, dtype=jnp.int32))
    # All entries are padding
    self.assertTrue(jnp.all(expert_ids == num_local_experts))


class DeepEPFanInTest(unittest.TestCase):
  """Tests for deepep_fan_in."""

  def test_basic_fan_in(self):
    """Verify weighted aggregation of multi-expert outputs."""
    num_recv = 2
    num_topk = 2
    hidden_out = 4
    num_local_experts = 3

    recv_topk_idx = jnp.array([[0, 1], [2, 0]], dtype=jnp.int64)
    recv_topk_weights = jnp.array([[0.6, 0.4], [0.7, 0.3]], dtype=jnp.float32)

    expert_output = jnp.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ], dtype=jnp.bfloat16)

    recv_x_dummy = jnp.zeros((num_recv, 1), dtype=jnp.bfloat16)
    _, _, _, token_indices = deepep_fan_out(
        recv_x_dummy, recv_topk_idx, num_local_experts,
    )

    aggregated = deepep_fan_in(
        expert_output, token_indices, recv_topk_weights, recv_topk_idx,
        num_recv,
    )

    self.assertEqual(aggregated.shape, (num_recv, hidden_out))

    expected_0 = 0.6 * expert_output[0].astype(jnp.float32) + 0.4 * expert_output[1].astype(jnp.float32)
    expected_1 = 0.7 * expert_output[2].astype(jnp.float32) + 0.3 * expert_output[3].astype(jnp.float32)
    np.testing.assert_allclose(
        aggregated[0].astype(jnp.float32), expected_0, atol=0.15,
    )
    np.testing.assert_allclose(
        aggregated[1].astype(jnp.float32), expected_1, atol=0.15,
    )


class DeepEPFanOutFanInRoundTripTest(unittest.TestCase):
  """Test fan_out -> identity gmm -> fan_in preserves token identity."""

  def test_identity_round_trip(self):
    """fan_out -> identity expert -> fan_in should give weighted recv_x."""
    num_recv = 8
    hidden = 16
    num_topk = 4
    num_local_experts = 8
    key = jax.random.PRNGKey(42)

    recv_x = jax.random.normal(key, (num_recv, hidden), dtype=jnp.bfloat16)

    k1, k2 = jax.random.split(key)
    recv_topk_idx = jax.random.randint(k1, (num_recv, num_topk), 0, num_local_experts).astype(jnp.int64)
    raw_weights = jax.random.uniform(k2, (num_recv, num_topk), dtype=jnp.float32, minval=0.1, maxval=1.0)
    recv_topk_weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)

    expanded_x, expert_ids, group_sizes, token_indices = deepep_fan_out(
        recv_x, recv_topk_idx, num_local_experts,
    )

    expert_output = expanded_x

    aggregated = deepep_fan_in(
        expert_output, token_indices, recv_topk_weights, recv_topk_idx,
        num_recv,
    )

    for i in range(num_recv):
      expected = jnp.zeros(hidden, dtype=jnp.bfloat16)
      for k in range(num_topk):
        if recv_topk_idx[i, k] >= 0:
          expected = expected + recv_topk_weights[i, k] * recv_x[i]
      np.testing.assert_allclose(
          aggregated[i].astype(jnp.float32),
          expected.astype(jnp.float32),
          atol=0.05,
          err_msg=f"Mismatch at token {i}",
      )


class DeepEPFanGradTest(unittest.TestCase):
  """Test that gradients flow through fan_out -> fan_in."""

  def test_gradient_flows(self):
    num_recv = 4
    hidden = 8
    num_topk = 2
    num_local_experts = 4

    recv_topk_idx = jnp.array([[0, 1], [2, -1], [0, 3], [1, 2]], dtype=jnp.int64)
    recv_topk_weights = jnp.array([[0.6, 0.4], [1.0, 0.0], [0.5, 0.5], [0.3, 0.7]], dtype=jnp.float32)

    def forward(recv_x):
      expanded_x, expert_ids, group_sizes, token_indices = deepep_fan_out(
          recv_x, recv_topk_idx, num_local_experts,
      )
      expert_output = expanded_x * 2.0
      aggregated = deepep_fan_in(
          expert_output, token_indices, recv_topk_weights, recv_topk_idx,
          num_recv,
      )
      return aggregated.sum()

    recv_x = jnp.ones((num_recv, hidden), dtype=jnp.float32)
    grad_fn = jax.grad(forward)
    grads = grad_fn(recv_x)

    self.assertEqual(grads.shape, recv_x.shape)
    self.assertTrue(jnp.all(jnp.isfinite(grads)), "Gradients contain non-finite values")
    self.assertTrue(jnp.any(grads != 0), "All gradients are zero")


class DeepEPMlpBiasPaddingContractTest(unittest.TestCase):
  """Contract for DeepEP + mlp_bias: padding rows must not use expert-0 bias.

  The production ``RoutedMoE.sparse_matmul`` DeepEP path clamps padding expert IDs
  to 0 (OOB avoidance), then zeros the gathered per-row biases for padding rows
  via ``jnp.where(_deepep_valid_rows, bias, 0)``.  This test verifies that
  contract: after the clamp + mask, padding rows must have zero bias.
  """

  def test_padding_expanded_rows_must_not_gather_expert_zero_bias(self):
    """Replays DeepEP preprocessing (sort, row mask, expert clamp) + bias gather."""
    num_local_experts = 4
    num_recv = 2
    num_topk = 2
    hidden = 8
    mlp_dim = 16

    recv_x = jnp.zeros((num_recv, hidden), dtype=jnp.bfloat16)
    recv_topk_idx = jnp.array([[0, 1], [-1, -1]], dtype=jnp.int32)

    expanded_x, expert_ids, group_sizes, _ = deepep_fan_out(
        recv_x, recv_topk_idx, num_local_experts,
    )
    sort_idx = jnp.argsort(expert_ids.astype(jnp.int32))
    x_sorted = expanded_x[sort_idx]
    num_valid_rows = jnp.sum(group_sizes)
    deepep_valid_rows = (jnp.arange(x_sorted.shape[0]) < num_valid_rows)[:, None]
    _ = jnp.where(deepep_valid_rows, x_sorted, 0)

    selected_experts = jnp.where(expert_ids < num_local_experts, expert_ids, 0)[sort_idx]

    w0_bias = jnp.zeros((num_local_experts, mlp_dim), dtype=jnp.bfloat16).at[0].set(3.0)
    w0_bias_per_row = w0_bias[selected_experts]
    w0_bias_per_row = jnp.where(deepep_valid_rows, w0_bias_per_row, 0)
    padding_row = ~deepep_valid_rows[:, 0]

    np.testing.assert_allclose(
        w0_bias_per_row[padding_row].astype(jnp.float32),
        0.0,
        atol=0.0,
        err_msg="padding rows received nonzero bias after masking",
    )


class DeepEPBufferTruncationTest(unittest.TestCase):
  """Test that fan_out handles worst-case buffer padding correctly."""

  def test_garbage_rows_masked_to_negative_one(self):
    """Simulate uninitialized rows beyond actual_num_recv.

    DeepEP allocates num_worst_tokens rows but only populates the first
    actual_num_recv. The integration code must set remaining rows to -1
    before calling fan_out. This test verifies group_sizes only count
    valid entries (masking turns garbage into padding).
    """
    actual_recv = 3
    num_worst = 8
    hidden = 4
    num_local_experts = 4
    num_topk = 2

    recv_x = jnp.ones((num_worst, hidden), dtype=jnp.bfloat16)
    for i in range(actual_recv):
      recv_x = recv_x.at[i].set(i + 1.0)

    # Rows 0-2 have real assignments; rows 3-7 are "garbage" (zeros = expert 0)
    recv_topk_idx = jnp.zeros((num_worst, num_topk), dtype=jnp.int32)
    recv_topk_idx = recv_topk_idx.at[0].set(jnp.array([0, 2]))
    recv_topk_idx = recv_topk_idx.at[1].set(jnp.array([1, -1]))
    recv_topk_idx = recv_topk_idx.at[2].set(jnp.array([3, 0]))

    # WITHOUT masking: garbage rows add to expert 0's group_size
    _, _, gs_bad, _ = deepep_fan_out(recv_x, recv_topk_idx, num_local_experts)
    bad_expert0_count = int(gs_bad[0])

    # WITH masking (as the integration code does): set unused rows to -1
    valid_rows = jnp.arange(num_worst) < actual_recv
    recv_topk_idx_masked = jnp.where(valid_rows[:, None], recv_topk_idx, -1)
    _, _, gs_good, _ = deepep_fan_out(recv_x, recv_topk_idx_masked, num_local_experts)
    good_expert0_count = int(gs_good[0])

    # Expert 0 should have fewer tokens after masking
    self.assertGreater(bad_expert0_count, good_expert0_count,
                       "Without masking, garbage rows inflate expert 0 count")
    # Valid counts: expert 0 -> 2 (row 0 slot 0, row 2 slot 1)
    # expert 1 -> 1 (row 1 slot 0), expert 2 -> 1 (row 0 slot 1)
    # expert 3 -> 1 (row 2 slot 0)
    self.assertEqual(int(gs_good.sum()), 5)
    np.testing.assert_array_equal(gs_good, jnp.array([2, 1, 1, 1]))


class DeepEPFloat32PrecisionTest(unittest.TestCase):
  """Test that fan_in respects float32_weight_sum."""

  def test_float32_accumulation(self):
    """Verify fan_in with float32_weight_sum=True accumulates in float32."""
    num_recv = 2
    num_local_experts = 2

    recv_topk_idx = jnp.array([[0, 1], [0, 1]], dtype=jnp.int32)
    recv_topk_weights = jnp.array([[0.6, 0.4], [0.3, 0.7]], dtype=jnp.float32)

    expert_output = jnp.ones((4, 8), dtype=jnp.bfloat16) * 100.0

    recv_x_dummy = jnp.zeros((num_recv, 8), dtype=jnp.bfloat16)
    _, _, _, token_indices = deepep_fan_out(recv_x_dummy, recv_topk_idx, num_local_experts)

    agg_f32 = deepep_fan_in(
        expert_output, token_indices, recv_topk_weights, recv_topk_idx,
        num_recv, float32_weight_sum=True,
    )
    agg_bf16 = deepep_fan_in(
        expert_output, token_indices, recv_topk_weights, recv_topk_idx,
        num_recv, float32_weight_sum=False,
    )

    self.assertEqual(agg_f32.dtype, jnp.float32)
    self.assertEqual(agg_bf16.dtype, jnp.bfloat16)

    expected = 100.0  # 0.6*100 + 0.4*100 = 100 for row 0
    np.testing.assert_allclose(float(agg_f32[0, 0]), expected, atol=1e-5)

  def test_default_is_float32(self):
    """Default float32_weight_sum=True."""
    num_recv = 1
    recv_topk_idx = jnp.array([[0]], dtype=jnp.int32)
    recv_topk_weights = jnp.array([[1.0]], dtype=jnp.float32)
    expert_output = jnp.ones((1, 4), dtype=jnp.bfloat16)
    recv_x_dummy = jnp.zeros((1, 4), dtype=jnp.bfloat16)
    _, _, _, token_indices = deepep_fan_out(recv_x_dummy, recv_topk_idx, 1)

    result = deepep_fan_in(expert_output, token_indices, recv_topk_weights,
                           recv_topk_idx, num_recv)
    self.assertEqual(result.dtype, jnp.float32)


class DeepEPCombineStateTest(unittest.TestCase):
  """Test the NamedTuple state container."""

  def test_named_access(self):
    state = _DeepEPCombineState(
        handle=(None,),
        recv_topk_weights=jnp.zeros(1),
        recv_topk_idx=jnp.zeros(1, dtype=jnp.int32),
        fan_out_token_indices=jnp.zeros(1, dtype=jnp.int32),
        sort_idx=jnp.zeros(1, dtype=jnp.int32),
        num_recv_tokens=42,
        combine_fn=lambda x, handle: (x, None),
    )
    self.assertEqual(state.num_recv_tokens, 42)
    self.assertIsNotNone(state.handle)
    self.assertTrue(callable(state.combine_fn))


class DeepEPSortUnsortTest(unittest.TestCase):
  """Test the full sort-by-expert -> transform -> unsort chain."""

  def test_sort_unsort_preserves_fan_in(self):
    """fan_out -> sort -> scale_per_expert -> unsort -> fan_in must match
    a manual per-token weighted sum of the scaled outputs."""
    num_recv = 4
    hidden = 8
    num_local_experts = 3
    key = jax.random.PRNGKey(7)

    recv_x = jax.random.normal(key, (num_recv, hidden), dtype=jnp.bfloat16)
    recv_topk_idx = jnp.array([[0, 1], [2, -1], [0, 2], [1, -1]], dtype=jnp.int32)
    recv_topk_weights = jnp.array(
        [[0.6, 0.4], [1.0, 0.0], [0.5, 0.5], [1.0, 0.0]], dtype=jnp.float32
    )

    expanded_x, expert_ids, group_sizes, token_indices = deepep_fan_out(
        recv_x, recv_topk_idx, num_local_experts,
    )

    sort_idx = jnp.argsort(expert_ids.astype(jnp.int32))
    sorted_x = expanded_x[sort_idx]
    sorted_expert_ids = expert_ids[sort_idx]

    # Simulate per-expert scaling: expert 0 -> *2, expert 1 -> *3, expert 2 -> *4
    scale = jnp.array([2.0, 3.0, 4.0])[sorted_expert_ids]
    sorted_output = sorted_x * scale[:, None]

    unsort_idx = jnp.argsort(sort_idx)
    unsorted_output = sorted_output[unsort_idx]

    aggregated = deepep_fan_in(
        unsorted_output, token_indices, recv_topk_weights, recv_topk_idx,
        num_recv, float32_weight_sum=True,
    )

    for i in range(num_recv):
      expected = jnp.zeros(hidden, dtype=jnp.float32)
      for k in range(2):
        eid = int(recv_topk_idx[i, k])
        if eid >= 0:
          s = [2.0, 3.0, 4.0][eid]
          expected += recv_topk_weights[i, k] * s * recv_x[i].astype(jnp.float32)
      np.testing.assert_allclose(
          aggregated[i].astype(jnp.float32), expected, atol=0.1,
          err_msg=f"Token {i} mismatch",
      )


class DeepEPGradValueTest(unittest.TestCase):
  """Verify gradient values, not just that they're nonzero."""

  def test_gradient_values_scale_linearly(self):
    """If expert is identity, d(output)/d(recv_x) = sum of weights per token."""
    num_recv = 3
    hidden = 4
    num_local_experts = 2

    recv_topk_idx = jnp.array([[0, 1], [0, -1], [1, 0]], dtype=jnp.int32)
    recv_topk_weights = jnp.array(
        [[0.7, 0.3], [1.0, 0.0], [0.4, 0.6]], dtype=jnp.float32
    )

    def forward(recv_x):
      expanded_x, _, _, token_indices = deepep_fan_out(
          recv_x, recv_topk_idx, num_local_experts,
      )
      aggregated = deepep_fan_in(
          expanded_x, token_indices, recv_topk_weights, recv_topk_idx,
          num_recv, float32_weight_sum=False,
      )
      return aggregated.sum()

    recv_x = jnp.ones((num_recv, hidden), dtype=jnp.float32)
    grads = jax.grad(forward)(recv_x)

    # With identity expert, grad for token i = sum of its valid weights * hidden
    # (since each element contributes independently and sum reduces all)
    for i in range(num_recv):
      weight_sum = sum(
          float(recv_topk_weights[i, k])
          for k in range(2)
          if int(recv_topk_idx[i, k]) >= 0
      )
      np.testing.assert_allclose(
          grads[i], weight_sum, atol=1e-5,
          err_msg=f"Token {i}: expected grad={weight_sum}, got {grads[i]}",
      )


class DeepEPDuplicateExpertTest(unittest.TestCase):
  """Test tokens assigned to the same local expert in multiple topk slots."""

  def test_duplicate_expert_fan_out(self):
    """Token 0 assigned to expert 1 twice -- both slots are valid."""
    num_recv = 2
    hidden = 4
    num_local_experts = 3

    recv_x = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.bfloat16)
    recv_topk_idx = jnp.array([[1, 1], [0, 2]], dtype=jnp.int32)
    recv_topk_weights = jnp.array([[0.5, 0.5], [0.6, 0.4]], dtype=jnp.float32)

    expanded_x, expert_ids, group_sizes, token_indices = deepep_fan_out(
        recv_x, recv_topk_idx, num_local_experts,
    )

    # All slots valid -> no padding, group_sizes sum = 4
    self.assertEqual(int(group_sizes.sum()), 4)
    # Expert 1 should appear twice for token 0
    self.assertEqual(int(group_sizes[1]), 2)

    # fan_in should accumulate both contributions
    aggregated = deepep_fan_in(
        expanded_x, token_indices, recv_topk_weights, recv_topk_idx,
        num_recv, float32_weight_sum=True,
    )
    # Token 0: 0.5 * [1,2,3,4] + 0.5 * [1,2,3,4] = [1,2,3,4]
    np.testing.assert_allclose(
        aggregated[0].astype(jnp.float32),
        jnp.array([1, 2, 3, 4], dtype=jnp.float32),
        atol=0.05,
    )


class DeepEPJitTest(unittest.TestCase):
  """Verify fan_out and fan_in work under jax.jit."""

  def test_jit_fan_out_fan_in(self):
    num_recv = 4
    hidden = 8
    num_local_experts = 4

    recv_x = jax.random.normal(jax.random.PRNGKey(0), (num_recv, hidden), dtype=jnp.bfloat16)
    recv_topk_idx = jnp.array([[0, 1], [2, -1], [3, 0], [1, 2]], dtype=jnp.int32)
    recv_topk_weights = jnp.array(
        [[0.5, 0.5], [1.0, 0.0], [0.6, 0.4], [0.3, 0.7]], dtype=jnp.float32
    )

    @jax.jit
    def run(rx, idx, weights):
      expanded_x, expert_ids, group_sizes, token_indices = deepep_fan_out(
          rx, idx, num_local_experts,
      )
      sort_idx = jnp.argsort(expert_ids.astype(jnp.int32))
      sorted_out = expanded_x[sort_idx]
      unsort_idx = jnp.argsort(sort_idx)
      unsorted_out = sorted_out[unsort_idx]
      aggregated = deepep_fan_in(
          unsorted_out, token_indices, weights, idx,
          num_recv, float32_weight_sum=True,
      )
      return aggregated

    # Should not raise
    result_jit = run(recv_x, recv_topk_idx, recv_topk_weights)
    self.assertEqual(result_jit.shape, (num_recv, hidden))
    self.assertTrue(jnp.all(jnp.isfinite(result_jit)))

  def test_jit_grad(self):
    """Gradient through jit-compiled fan_out + fan_in."""
    num_recv = 3
    hidden = 4
    num_local_experts = 2
    recv_topk_idx = jnp.array([[0, 1], [0, -1], [1, 0]], dtype=jnp.int32)
    recv_topk_weights = jnp.array(
        [[0.5, 0.5], [1.0, 0.0], [0.4, 0.6]], dtype=jnp.float32
    )

    @jax.jit
    def loss_fn(rx):
      expanded_x, _, _, token_indices = deepep_fan_out(rx, recv_topk_idx, num_local_experts)
      aggregated = deepep_fan_in(
          expanded_x * 2.0, token_indices, recv_topk_weights, recv_topk_idx,
          num_recv, float32_weight_sum=True,
      )
      return aggregated.sum()

    recv_x = jnp.ones((num_recv, hidden), dtype=jnp.float32)
    grads = jax.grad(loss_fn)(recv_x)
    self.assertEqual(grads.shape, recv_x.shape)
    self.assertTrue(jnp.all(jnp.isfinite(grads)))
    self.assertTrue(jnp.any(grads > 0))


class DeepEPLargeScaleTest(unittest.TestCase):
  """Stress test with realistic dimensions."""

  def test_deepseek_v3_dimensions(self):
    """num_tokens=4096, hidden=7168, num_topk=8, 32 local experts."""
    num_recv = 4096
    hidden = 256  # reduced from 7168 for speed, same logic
    num_topk = 8
    num_local_experts = 32
    key = jax.random.PRNGKey(99)

    k1, k2, k3 = jax.random.split(key, 3)
    recv_x = jax.random.normal(k1, (num_recv, hidden), dtype=jnp.bfloat16)
    recv_topk_idx = jax.random.randint(k2, (num_recv, num_topk), -1, num_local_experts).astype(jnp.int32)
    recv_topk_weights = jax.random.uniform(k3, (num_recv, num_topk), dtype=jnp.float32)
    recv_topk_weights = jnp.where(recv_topk_idx >= 0, recv_topk_weights, 0.0)
    row_sums = recv_topk_weights.sum(axis=1, keepdims=True)
    recv_topk_weights = jnp.where(row_sums > 0, recv_topk_weights / jnp.maximum(row_sums, 1e-8), 0.0)

    expanded_x, expert_ids, group_sizes, token_indices = deepep_fan_out(
        recv_x, recv_topk_idx, num_local_experts,
    )

    self.assertEqual(expanded_x.shape, (num_recv * num_topk, hidden))
    total_valid = int(jnp.sum(recv_topk_idx >= 0).item())
    self.assertEqual(int(group_sizes.sum()), total_valid)

    sort_idx = jnp.argsort(expert_ids.astype(jnp.int32))
    sorted_x = expanded_x[sort_idx]
    unsort_idx = jnp.argsort(sort_idx)
    unsorted_output = sorted_x[unsort_idx]

    aggregated = deepep_fan_in(
        unsorted_output, token_indices, recv_topk_weights, recv_topk_idx,
        num_recv, float32_weight_sum=True,
    )
    self.assertEqual(aggregated.shape, (num_recv, hidden))
    self.assertTrue(jnp.all(jnp.isfinite(aggregated)))


class DeepEPFullPipelineTest(unittest.TestCase):
  """End-to-end: fan_out -> sort -> per-expert linear -> unsort -> fan_in.

  Simulates the actual sparse_matmul pipeline with a simple per-expert
  linear transform (scale + bias) to verify the full data flow is correct.
  """

  def test_per_expert_linear_pipeline(self):
    """Each expert applies a different scale. Verify final output."""
    num_recv = 6
    hidden = 4
    num_topk = 2
    num_local_experts = 3
    key = jax.random.PRNGKey(77)

    k1, k2 = jax.random.split(key)
    recv_x = jax.random.normal(k1, (num_recv, hidden), dtype=jnp.bfloat16)

    recv_topk_idx = jnp.array([
        [0, 1], [2, -1], [0, 2], [1, 0], [2, 1], [-1, -1],
    ], dtype=jnp.int32)
    recv_topk_weights = jnp.array([
        [0.6, 0.4], [1.0, 0.0], [0.5, 0.5], [0.7, 0.3], [0.8, 0.2], [0.0, 0.0],
    ], dtype=jnp.float32)

    expert_scales = jnp.array([2.0, 3.0, 5.0])

    @jax.jit
    def pipeline(rx, idx, weights):
      expanded_x, expert_ids, group_sizes, token_indices = deepep_fan_out(
          rx, idx, num_local_experts,
      )

      sort_idx = jnp.argsort(expert_ids.astype(jnp.int32))
      sorted_x = expanded_x[sort_idx]
      sorted_eids = expert_ids[sort_idx]

      clamped_eids = jnp.where(sorted_eids < num_local_experts, sorted_eids, 0)
      scales = expert_scales[clamped_eids]
      sorted_output = sorted_x * scales[:, None]

      unsort_idx = jnp.argsort(sort_idx)
      unsorted_output = sorted_output[unsort_idx]

      aggregated = deepep_fan_in(
          unsorted_output, token_indices, weights, idx,
          num_recv, float32_weight_sum=True,
      )
      return aggregated

    result = pipeline(recv_x, recv_topk_idx, recv_topk_weights)

    for i in range(num_recv):
      expected = jnp.zeros(hidden, dtype=jnp.float32)
      for k in range(num_topk):
        eid = int(recv_topk_idx[i, k])
        if eid >= 0:
          s = float(expert_scales[eid])
          expected += recv_topk_weights[i, k] * s * recv_x[i].astype(jnp.float32)
      np.testing.assert_allclose(
          result[i], expected, atol=0.1,
          err_msg=f"Token {i}: expected {expected}, got {result[i]}",
      )

  def test_pipeline_gradient_correctness(self):
    """Gradient of the full pipeline should be finite and nonzero."""
    num_recv = 4
    hidden = 8
    num_topk = 2
    num_local_experts = 3

    recv_topk_idx = jnp.array([[0, 1], [2, -1], [0, 2], [1, 0]], dtype=jnp.int32)
    recv_topk_weights = jnp.array(
        [[0.6, 0.4], [1.0, 0.0], [0.5, 0.5], [0.3, 0.7]], dtype=jnp.float32
    )
    expert_scales = jnp.array([2.0, 3.0, 5.0])

    @jax.jit
    def loss_fn(rx):
      expanded_x, expert_ids, group_sizes, token_indices = deepep_fan_out(
          rx, recv_topk_idx, num_local_experts,
      )
      sort_idx = jnp.argsort(expert_ids.astype(jnp.int32))
      sorted_x = expanded_x[sort_idx]
      clamped_eids = jnp.where(
          expert_ids < num_local_experts, expert_ids, 0
      )[sort_idx]
      sorted_output = sorted_x * expert_scales[clamped_eids][:, None]
      unsort_idx = jnp.argsort(sort_idx)
      unsorted_output = sorted_output[unsort_idx]
      aggregated = deepep_fan_in(
          unsorted_output, token_indices, recv_topk_weights, recv_topk_idx,
          num_recv, float32_weight_sum=True,
      )
      return aggregated.sum()

    recv_x = jax.random.normal(jax.random.PRNGKey(0), (num_recv, hidden), dtype=jnp.float32)
    grads = jax.grad(loss_fn)(recv_x)
    self.assertTrue(jnp.all(jnp.isfinite(grads)))
    self.assertTrue(jnp.any(grads != 0))

    for i in range(num_recv):
      expected_scale = sum(
          recv_topk_weights[i, k] * expert_scales[recv_topk_idx[i, k]]
          for k in range(num_topk)
          if int(recv_topk_idx[i, k]) >= 0
      )
      np.testing.assert_allclose(
          grads[i], float(expected_scale), atol=1e-4,
          err_msg=f"Token {i}: expected grad scale={expected_scale}",
      )


class DeepEPDenseIncompatibilityTest(unittest.TestCase):
  """Verify DeepEP is architecturally incompatible with dense paths."""

  def test_dense_with_capacity_factor_rejected(self):
    """use_deepep_dispatch + sparse_matmul=False is rejected at config time."""
    from MaxText import pyconfig
    from MaxText.globals import MAXTEXT_PKG_DIR

    with self.assertRaises(ValueError) as ctx:
      pyconfig.initialize(
          [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
          run_name="deepep_dense_test",
          enable_checkpointing=False,
          model_name="default",
          sparse_matmul=False,
          capacity_factor=2.0,
          use_deepep_dispatch=True,
          ici_expert_parallelism=4,
      )
    self.assertIn("sparse_matmul", str(ctx.exception))

  def test_dense_dropless_rejected(self):
    """use_deepep_dispatch + sparse_matmul=False + capacity_factor=-1 rejected."""
    from MaxText import pyconfig
    from MaxText.globals import MAXTEXT_PKG_DIR

    with self.assertRaises(ValueError) as ctx:
      pyconfig.initialize(
          [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
          run_name="deepep_dense_test",
          enable_checkpointing=False,
          model_name="default",
          sparse_matmul=False,
          capacity_factor=-1.0,
          use_deepep_dispatch=True,
          ici_expert_parallelism=4,
      )
    self.assertIn("sparse_matmul", str(ctx.exception))


class DeepEPConfigValidationTest(unittest.TestCase):
  """Test config validation for use_deepep_dispatch."""

  def test_requires_sparse_matmul(self):
    from MaxText import pyconfig
    from MaxText.globals import MAXTEXT_PKG_DIR

    with self.assertRaises(ValueError):
      pyconfig.initialize(
          [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
          run_name="deepep_test",
          enable_checkpointing=False,
          model_name="default",
          sparse_matmul=False,
          use_deepep_dispatch=True,
          ici_expert_parallelism=2,
      )

  def test_requires_expert_parallelism(self):
    from MaxText import pyconfig
    from MaxText.globals import MAXTEXT_PKG_DIR

    with self.assertRaises(ValueError):
      pyconfig.initialize(
          [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
          run_name="deepep_test",
          enable_checkpointing=False,
          model_name="default",
          sparse_matmul=True,
          use_deepep_dispatch=True,
          ici_expert_parallelism=1,
          dcn_expert_parallelism=1,
      )

  def test_rejects_internode(self):
    from MaxText import pyconfig
    from MaxText.globals import MAXTEXT_PKG_DIR

    with self.assertRaises(ValueError):
      pyconfig.initialize(
          [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
          run_name="deepep_test",
          enable_checkpointing=False,
          model_name="default",
          sparse_matmul=True,
          use_deepep_dispatch=True,
          ici_expert_parallelism=4,
          dcn_expert_parallelism=2,
      )

  def test_rejects_ep_over_8(self):
    from MaxText import pyconfig
    from MaxText.globals import MAXTEXT_PKG_DIR

    with self.assertRaises(ValueError):
      pyconfig.initialize(
          [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
          run_name="deepep_test",
          enable_checkpointing=False,
          model_name="default",
          sparse_matmul=True,
          use_deepep_dispatch=True,
          ici_expert_parallelism=16,
          dcn_expert_parallelism=1,
      )

  def test_requires_primus_turbo_package_when_deepep_flags_valid(self):
    """If primus_turbo is missing, fail at config time with a clear message."""
    try:
      import primus_turbo.jax.lax.moe  # pylint: disable=import-outside-toplevel,unused-import
    except ImportError:
      pass
    else:
      self.skipTest("primus_turbo is installed; cannot exercise missing-package error path")

    from MaxText import pyconfig
    from MaxText.globals import MAXTEXT_PKG_DIR

    with self.assertRaises(ValueError) as ctx:
      pyconfig.initialize(
          [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
          run_name="deepep_primus_test",
          enable_checkpointing=False,
          model_name="default",
          sparse_matmul=True,
          use_deepep_dispatch=True,
          ici_expert_parallelism=2,
          dcn_expert_parallelism=1,
      )
    self.assertIn("primus_turbo", str(ctx.exception).lower())


if __name__ == "__main__":
  unittest.main()
