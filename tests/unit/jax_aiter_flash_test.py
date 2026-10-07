# Copyright 2023–2025 Google LLC
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

"""Tests for attention=jax_aiter_flash: metadata, routing, and TE gating.

No kernel runs here; jax-aiter is replaced by recording fakes.
"""

import sys
import types
import unittest
from unittest import mock

from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.common.common_types import AttentionType
from maxtext.layers.attention_op import AttentionOp
from maxtext.utils import max_utils


def _metadata(segment_ids, max_segments):
  segment_ids = jnp.asarray(segment_ids, dtype=jnp.int32)
  build = jax.jit(AttentionOp._jax_aiter_flash_metadata, static_argnums=(1, 2))  # pylint: disable=protected-access
  seqstart, cu_seqlen = build(segment_ids, segment_ids.shape[1], max_segments)
  return np.asarray(seqstart), np.asarray(cu_seqlen)


def _bucket_metadata(segment_ids, max_segments):
  segment_ids = jnp.asarray(segment_ids, dtype=jnp.int32)
  build = jax.jit(AttentionOp._jax_aiter_flash_bucket_metadata, static_argnums=(1, 2))  # pylint: disable=protected-access
  buckets, overflow = build(segment_ids, segment_ids.shape[1], max_segments)
  return tuple(tuple(np.asarray(x) for x in bucket) for bucket in buckets), bool(overflow)


class JaxAiterFlashMetadataTest(unittest.TestCase):
  """Physical offsets and logical lengths handed to jax-aiter group mode."""

  def test_trailing_padding_is_excluded_from_logical_lengths(self):
    # Row 0: ids 1,1,1 then 2,2 then padding. Row 1: one segment, then padding.
    seqstart, cu_seqlen = _metadata([[1, 1, 1, 2, 2, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]], 4)
    total = 16
    np.testing.assert_array_equal(seqstart[:3], [0, 3, 8])
    np.testing.assert_array_equal(seqstart[3:], [total] * (len(seqstart) - 3))
    np.testing.assert_array_equal(cu_seqlen[:4], [0, 3, 5, 9])
    np.testing.assert_array_equal(cu_seqlen[4:], [9] * (len(cu_seqlen) - 4))

  def test_static_slot_count_is_independent_of_real_segment_count(self):
    one = [[1] * 8, [1] * 8]
    many = [[1, 1, 2, 2, 3, 3, 4, 4]] * 2
    for ids in (one, many):
      for arr in _metadata(ids, 4):
        self.assertEqual(arr.shape, (2 * 4 + 1,))
    seqstart, cu_seqlen = _metadata(many, 4)
    np.testing.assert_array_equal(seqstart, [0, 2, 4, 6, 8, 10, 12, 14, 16])
    np.testing.assert_array_equal(cu_seqlen, [0, 2, 4, 6, 8, 10, 12, 14, 16])

  def test_unpacked_padded_rows_use_one_slot_per_row(self):
    # packing=false on real data: one document per row, right-padded.
    seqstart, cu_seqlen = _metadata([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]], 1)
    np.testing.assert_array_equal(seqstart, [0, 6, 12])
    np.testing.assert_array_equal(cu_seqlen, [0, 3, 8])

  def test_offsets_are_monotonic_in_token_order(self):
    seqstart, cu_seqlen = _metadata([[1, 1, 0, 2, 2, 2, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]], 4)
    self.assertTrue(np.all(np.diff(seqstart) >= 0), msg=f"seqstart not sorted: {seqstart}")
    self.assertTrue(np.all(np.diff(cu_seqlen) >= 0), msg=f"cu_seqlen not sorted: {cu_seqlen}")
    np.testing.assert_array_equal(seqstart[:3], [0, 3, 8])
    np.testing.assert_array_equal(cu_seqlen[:4], [0, 2, 5, 6])

  def test_bucket_metadata_covers_each_real_token_once(self):
    rows = [
        [1] * 256 + [2] * 768 + [3] * 1536 + [4] * 1536,
        [1] * 4096,
        [1] * 512 + [2] * 1024 + [3] * 2048 + [4] * 512,
        [1] * 3000 + [2] * 1096,
    ]
    buckets, overflow = _bucket_metadata(rows, 64)
    self.assertFalse(overflow)
    owners = np.stack([bucket[2] for bucket in buckets])
    np.testing.assert_array_equal(owners.sum(axis=0), np.ones((4 * 4096,), dtype=np.int64))

  def test_bucket_metadata_reports_short_document_overflow(self):
    # Four rows with 64 short documents each exceed the common-case 128-slot
    # short bucket. Production must select the unbucketed fallback.
    row = np.repeat(np.arange(1, 65, dtype=np.int32), 64)
    buckets, overflow = _bucket_metadata(np.stack([row] * 4), 64)
    self.assertTrue(overflow)
    self.assertEqual(buckets[0][0].shape, (4 * 32 + 1,))


class JaxAiterFlashRoutingTest(parameterized.TestCase):
  """Which jax-aiter entry point each input selects, and what it is given."""

  def _attention(self, *, packing, dataset_type, max_segments_per_seq=2, dropout_rate=0.0,
                 attention_type=AttentionType.MLA):
    config = types.SimpleNamespace(
        context_sharding="context",
        packing=packing,
        dataset_type=dataset_type,
        max_segments_per_seq=max_segments_per_seq,
        head_dim=2,
        attention_kernel="jax_aiter_flash",
    )
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("context", "fsdp"))
    return AttentionOp(
        config=config,
        mesh=mesh,
        attention_kernel="jax_aiter_flash",
        max_target_length=4,
        num_query_heads=2,
        num_kv_heads=2,
        dtype=jnp.float32,
        dropout_rate=dropout_rate,
        attention_type=attention_type,
    )

  def _route(self, attention, segment_ids, full_row_layout="batch", seq_len=4, qk_dim=3, v_dim=2):
    calls = []

    def fake_auto(query, key, value, *, softmax_scale, causal=True):
      calls.append(("batch", query.shape, softmax_scale))
      return jnp.zeros(query.shape[:3] + value.shape[-1:], query.dtype)

    def fake_varlen(q, k, v, seqstart_q, seqstart_k, cu_q, cu_k, max_q, max_k, dropout_p, scale, causal, window):
      calls.append(("varlen", seqstart_q.shape, cu_q is not None, max_q, scale, causal, window))
      return jnp.zeros(q.shape[:2] + v.shape[-1:], q.dtype)

    def fake_buckets(q, k, v, buckets, owners, max_seqlens, scale, causal):
      calls.append(("buckets", tuple(ss.shape for ss, _ in buckets), len(owners), max_seqlens, scale, causal))
      return jnp.zeros(q.shape[:2] + v.shape[-1:], q.dtype)

    mha = types.ModuleType("jax_aiter.mha")
    mha.flash_attn_auto = fake_auto
    mha.flash_attn_varlen_auto = fake_varlen
    mha.flash_attn_varlen_buckets = fake_buckets
    mha.full_row_layout = lambda **kwargs: full_row_layout
    fake_modules = {"jax_aiter": types.ModuleType("jax_aiter"), "jax_aiter.mha": mha}
    query = jnp.zeros((2, seq_len, 2, qk_dim), jnp.float32)
    value = jnp.zeros((2, seq_len, 2, v_dim), jnp.float32)
    with mock.patch.dict(sys.modules, fake_modules):
      out = attention.jax_aiter_flash_attention(query, query, value, segment_ids)
    self.assertEqual(out.shape, (2, seq_len, 2, v_dim))
    return calls

  @parameterized.named_parameters(
      {"testcase_name": "synthetic_with_packing", "packing": True, "dataset_type": "synthetic"},
      {"testcase_name": "synthetic_without_packing", "packing": False, "dataset_type": "synthetic"},
  )
  def test_full_rows_use_batch_mode(self, packing, dataset_type):
    calls = self._route(self._attention(packing=packing, dataset_type=dataset_type), jnp.ones((2, 4), jnp.int32))
    self.assertEqual(calls, [("batch", (2, 4, 2, 3), 1.0)])

  def test_full_rows_follow_a_varlen_layout_choice(self):
    calls = self._route(self._attention(packing=False, dataset_type="synthetic"),
                        jnp.ones((2, 4), jnp.int32), full_row_layout="varlen")
    self.assertEqual(calls, [("varlen", (2 + 1,), False, 4, 1.0, True, (-1, -1))])

  def test_missing_segment_ids_use_batch_mode(self):
    calls = self._route(self._attention(packing=True, dataset_type="hf"), None)
    self.assertEqual([c[0] for c in calls], ["batch"])

  def test_diagnostic_force_batch_ignores_real_segment_boundaries(self):
    segment_ids = jnp.asarray([[1, 1, 2, 0], [1, 1, 1, 1]], jnp.int32)
    with mock.patch.dict("os.environ", {"MAXTEXT_DIAGNOSTIC_FORCE_BATCH_ATTN": "1"}):
      calls = self._route(
          self._attention(packing=True, dataset_type="hf", max_segments_per_seq=2),
          segment_ids,
      )
    self.assertEqual(calls, [("batch", (2, 4, 2, 3), 1.0)])

  def test_real_packed_data_uses_group_mode_with_configured_slots(self):
    calls = self._route(self._attention(packing=True, dataset_type="hf", max_segments_per_seq=2),
                        jnp.asarray([[1, 1, 2, 0], [1, 1, 1, 1]], jnp.int32))
    self.assertEqual(calls, [("varlen", (2 * 2 + 1,), True, 4, 1.0, True, (-1, -1))])

  def test_bucket_override_keeps_one_group_call(self):
    segment_ids = jnp.ones((2, 4096), jnp.int32)
    with mock.patch.dict("os.environ", {"MAXTEXT_JAX_AITER_FLASH_BUCKETS": "0"}):
      calls = self._route(
          self._attention(packing=True, dataset_type="hf", max_segments_per_seq=2),
          segment_ids,
          seq_len=4096,
          qk_dim=192,
          v_dim=128,
      )
    self.assertEqual(calls, [("varlen", (2 * 2 + 1,), True, 4096, 1.0, True, (-1, -1))])

  def _route_mla_buckets(self, mode):
    with mock.patch.dict("os.environ", {"MAXTEXT_JAX_AITER_FLASH_BUCKETS": mode}):
      return self._route(
          self._attention(packing=True, dataset_type="hf", max_segments_per_seq=2),
          jnp.ones((2, 4096), jnp.int32),
          seq_len=4096,
          qk_dim=192,
          v_dim=128,
      )

  def test_buckets_trace_overflow_fallback_and_four_bounded_calls(self):
    calls = self._route_mla_buckets("1")
    self.assertEqual(calls[0], ("varlen", (2 * 2 + 1,), True, 4096, 1.0, True, (-1, -1)))
    self.assertEqual([c[3] for c in calls[1:]], [512, 1024, 2048, 4096])

  def test_shared_buckets_use_one_in_place_call(self):
    calls = self._route_mla_buckets("shared")
    self.assertEqual(calls, [
        ("varlen", (2 * 2 + 1,), True, 4096, 1.0, True, (-1, -1)),
        # Bucket slots are capped by max_segments_per_seq=2.
        ("buckets", ((2 * 2 + 1,), (2 * 2 + 1,), (2 * 2 + 1,), (2 * 1 + 1,)), 4, (512, 1024, 2048, 4096), 1.0, True),
    ])

  def test_unknown_bucket_mode_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "MAXTEXT_JAX_AITER_FLASH_BUCKETS"):
      self._route_mla_buckets("yes")

  def test_real_unpacked_data_uses_one_slot_per_row(self):
    calls = self._route(self._attention(packing=False, dataset_type="hf", max_segments_per_seq=-1),
                        jnp.asarray([[1, 1, 0, 0], [1, 1, 1, 0]], jnp.int32))
    self.assertEqual(calls, [("varlen", (2 * 1 + 1,), True, 4, 1.0, True, (-1, -1))])

  def test_packing_without_slot_count_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "max_segments_per_seq"):
      self._route(self._attention(packing=True, dataset_type="hf", max_segments_per_seq=-1),
                  jnp.ones((2, 4), jnp.int32))

  def test_dropout_is_rejected(self):
    with self.assertRaisesRegex(NotImplementedError, "dropout"):
      self._route(self._attention(packing=True, dataset_type="hf", dropout_rate=0.1), jnp.ones((2, 4), jnp.int32))

  def test_sliding_window_is_rejected(self):
    with self.assertRaisesRegex(NotImplementedError, "global causal"):
      self._route(self._attention(packing=True, dataset_type="hf", attention_type=AttentionType.LOCAL_SLIDING),
                  jnp.ones((2, 4), jnp.int32))


class AiterTritonDiagnosticGuardTest(unittest.TestCase):
  """attention=aiter_triton accepts packed real data and synthetic full rows only."""

  def _run(self, *, packing, dataset_type, max_segments_per_seq):
    config = types.SimpleNamespace(
        context_sharding="context", packing=packing, dataset_type=dataset_type,
        max_segments_per_seq=max_segments_per_seq, head_dim=2, attention_kernel="aiter_triton",
    )
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("context", "fsdp"))
    attention = AttentionOp(
        config=config, mesh=mesh, attention_kernel="aiter_triton", max_target_length=4,
        num_query_heads=2, num_kv_heads=2, dtype=jnp.float32, attention_type=AttentionType.MLA,
    )
    triton = types.ModuleType("jax_aiter.triton.attention")
    triton.flash_attn_varlen_triton = lambda q, k, v, *a: jnp.zeros(q.shape[:2] + v.shape[-1:], q.dtype)
    fake = {
        "jax_aiter": types.ModuleType("jax_aiter"),
        "jax_aiter.triton": types.ModuleType("jax_aiter.triton"),
        "jax_aiter.triton.attention": triton,
    }
    query = jnp.zeros((2, 4, 2, 3), jnp.float32)
    with mock.patch.dict(sys.modules, fake):
      return attention.aiter_triton_attention(query, query, query, jnp.ones((2, 4), jnp.int32))

  def test_unpacked_real_data_is_rejected(self):
    with self.assertRaisesRegex(NotImplementedError, "packed real data or synthetic"):
      self._run(packing=False, dataset_type="hf", max_segments_per_seq=-1)

  def test_synthetic_full_rows_are_accepted(self):
    out = self._run(packing=False, dataset_type="synthetic", max_segments_per_seq=-1)
    self.assertEqual(out.shape, (2, 4, 2, 3))


@pytest.mark.parametrize(
    "hardware,attention,quantization,use_te_comm_gemm_overlap,expected",
    [
        ("gpu", "jax_aiter_flash", "", False, False),
        ("gpu", "aiter_triton", "", False, False),
        ("gpu", "cudnn_flash_te", "", False, True),
        ("gpu", "dot_product", "te_fp8_currentscaling", False, True),
        ("gpu", "jax_aiter_flash", "", True, True),
        ("cpu", "cudnn_flash_te", "te_fp8_currentscaling", True, False),
    ],
)
def test_should_use_transformer_engine(hardware, attention, quantization, use_te_comm_gemm_overlap, expected):
  config = mock.Mock(
      hardware=hardware,
      attention=attention,
      quantization=quantization,
      use_te_comm_gemm_overlap=use_te_comm_gemm_overlap,
  )
  assert max_utils._should_use_transformer_engine(config) is expected  # pylint: disable=protected-access


if __name__ == "__main__":
  unittest.main()
