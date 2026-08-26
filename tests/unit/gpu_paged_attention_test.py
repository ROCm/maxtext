"""Tests for the vendor-neutral half of `attention: "gpu_paged"`.

Copyright 2026 Advanced Micro Devices, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

These cover the metadata conversion and the config guards, none of which needs a
kernel. The end-to-end numerical check against the dense path lives in
`attention_test.py`, which already has the mesh and config fixtures.

Everything is marked `gpu_only`. That is not because these need a GPU -- they do
not -- but because `tests/conftest.py` auto-marks unmarked tests `cpu_only` and
then skips `cpu_only` on any accelerator testbed. An unmarked test here would
report success on the machines we actually care about without ever running.
"""

import sys
import unittest

from absl.testing import parameterized
import numpy as np
import pytest

import jax.numpy as jnp

from maxtext.configs import pyconfig
from maxtext.inference.kv_common import KvPageTableV1
from maxtext.layers import gpu_paged_attention as gpa
from tests.utils.test_helpers import get_test_config_path

TOKENS_PER_PAGE = 16


class _VllmMetadata:
  """The three fields the vLLM-shaped path reads, without tpu_inference."""

  def __init__(self, block_tables, seq_lens, query_start_loc):
    self.block_tables = block_tables
    self.seq_lens = seq_lens
    self.query_start_loc = query_start_loc


@pytest.mark.gpu_only
class MetadataConversionTest(parameterized.TestCase):
  """Both accepted metadata shapes must produce the same page bookkeeping."""

  def test_neutral_page_table_round_trips(self):
    table = KvPageTableV1(
        page_ids=[[1, 2], [3]],
        seq_lens=np.asarray([20, 5], np.int32),
        query_lens=np.asarray([20, 5], np.int32),
        write_positions=np.concatenate([np.arange(20), np.arange(5)]).astype(np.int32),
        request_order=np.asarray([0, 1], np.int32),
    )
    plan = gpa.plan_from_neutral(table, TOKENS_PER_PAGE)

    np.testing.assert_array_equal(np.asarray(plan.kv_indptr), [0, 2, 3])
    np.testing.assert_array_equal(np.asarray(plan.kv_page_indices), [1, 2, 3])
    np.testing.assert_array_equal(np.asarray(plan.kv_last_page_lens), [4, 5])
    np.testing.assert_array_equal(np.asarray(plan.cu_seqlens_q), [0, 20, 25])
    self.assertEqual(plan.max_seqlen_q, 20)
    self.assertEqual(plan.max_seqlen_k, 20)
    self.assertFalse(plan.is_decode)

  def test_vllm_metadata_packs_the_padded_block_table(self):
    """The 2D table is row-padded; the kernels want the ids packed contiguously."""
    # Two requests of 20 and 5 tokens, holding 2 and 1 pages. Row 1 is padded.
    md = _VllmMetadata(
        block_tables=jnp.asarray([[1, 2], [3, 0]], jnp.int32),
        seq_lens=jnp.asarray([20, 5], jnp.int32),
        query_start_loc=jnp.asarray([0, 20, 25], jnp.int32),
    )
    plan = gpa.plan_from_vllm(md, TOKENS_PER_PAGE, total_tokens=25, max_seqlen_k=20)

    np.testing.assert_array_equal(np.asarray(plan.kv_indptr), [0, 2, 3])
    # Only the first three entries are addressed by kv_indptr; the tail is slack.
    np.testing.assert_array_equal(np.asarray(plan.kv_page_indices)[:3], [1, 2, 3])
    np.testing.assert_array_equal(np.asarray(plan.kv_last_page_lens), [4, 5])

  def test_both_shapes_agree(self):
    """The two conversions describe the same batch, so they must coincide."""
    table = KvPageTableV1(
        page_ids=[[5, 9], [7]],
        seq_lens=np.asarray([18, 12], np.int32),
        query_lens=np.asarray([18, 12], np.int32),
        write_positions=np.concatenate([np.arange(18), np.arange(12)]).astype(np.int32),
        request_order=np.asarray([0, 1], np.int32),
    )
    md = _VllmMetadata(
        block_tables=jnp.asarray([[5, 9], [7, 0]], jnp.int32),
        seq_lens=jnp.asarray([18, 12], jnp.int32),
        query_start_loc=jnp.asarray([0, 18, 30], jnp.int32),
    )

    a = gpa.plan_from_neutral(table, TOKENS_PER_PAGE)
    b = gpa.plan_from_vllm(md, TOKENS_PER_PAGE, total_tokens=30, max_seqlen_k=18)

    n_pages = int(np.asarray(a.kv_indptr)[-1])
    np.testing.assert_array_equal(np.asarray(a.kv_indptr), np.asarray(b.kv_indptr))
    np.testing.assert_array_equal(
        np.asarray(a.kv_page_indices)[:n_pages], np.asarray(b.kv_page_indices)[:n_pages]
    )
    np.testing.assert_array_equal(np.asarray(a.kv_last_page_lens), np.asarray(b.kv_last_page_lens))
    np.testing.assert_array_equal(np.asarray(a.slot_mapping), np.asarray(b.slot_mapping))

  def test_slot_mapping_places_an_appended_token_after_its_context(self):
    """Decode: the new token lands at the first free offset, not at position 0."""
    md = _VllmMetadata(
        block_tables=jnp.asarray([[1, 2], [3, 0]], jnp.int32),
        seq_lens=jnp.asarray([21, 6], jnp.int32),   # one token longer than the prefill above
        query_start_loc=jnp.asarray([0, 1, 2], jnp.int32),
    )
    plan = gpa.plan_from_vllm(md, TOKENS_PER_PAGE, total_tokens=2, max_seqlen_k=21)

    # request 0: position 20 -> page_ids[20 // 16] = 2, offset 4  -> 2*16 + 4
    # request 1: position  5 -> page_ids[5 // 16]  = 3, offset 5  -> 3*16 + 5
    np.testing.assert_array_equal(np.asarray(plan.slot_mapping), [2 * 16 + 4, 3 * 16 + 5])
    self.assertTrue(plan.is_decode)

  def test_unrecognised_metadata_is_rejected_by_shape(self):
    with self.assertRaisesRegex(TypeError, "neutral KvPageTableV1|vLLM-shaped"):
      gpa.build_plan(object(), TOKENS_PER_PAGE, total_tokens=1, max_seqlen_k=1)

  def test_dispatch_recognises_each_shape(self):
    table = KvPageTableV1(
        page_ids=[[1]],
        seq_lens=np.asarray([4], np.int32),
        query_lens=np.asarray([4], np.int32),
        write_positions=np.arange(4, dtype=np.int32),
        request_order=np.asarray([0], np.int32),
    )
    md = _VllmMetadata(
        block_tables=jnp.asarray([[1]], jnp.int32),
        seq_lens=jnp.asarray([4], jnp.int32),
        query_start_loc=jnp.asarray([0, 4], jnp.int32),
    )
    self.assertTrue(gpa.is_neutral_page_table(table))
    self.assertFalse(gpa.is_vllm_metadata(table))
    self.assertTrue(gpa.is_vllm_metadata(md))
    self.assertFalse(gpa.is_neutral_page_table(md))


@pytest.mark.gpu_only
class BackendSelectionTest(parameterized.TestCase):

  def test_explicit_backend_is_taken_literally(self):
    self.assertEqual(gpa.resolve_backend("aiter"), "aiter")
    self.assertEqual(gpa.resolve_backend("flashinfer"), "flashinfer")

  def test_flashinfer_fails_with_a_pointer_rather_than_silently(self):
    with self.assertRaisesRegex(NotImplementedError, "flashinfer"):
      gpa.paged_attention_step(None, None, None, None, None, None, backend="flashinfer")

  def test_unknown_backend_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "unknown paged_attention_backend"):
      gpa.paged_attention_step(None, None, None, None, None, None, backend="nope")


@pytest.mark.gpu_only
class ConfigGuardTest(parameterized.TestCase):
  """`gpu_paged` must be accepted, and its one pathological combination refused."""

  base_arguments = {
      "per_device_batch_size": 1.0,
      "run_name": "test_gpu_paged_config",
      "enable_checkpointing": False,
      "max_target_length": 128,
  }

  def _config(self, **overrides):
    arguments = dict(self.base_arguments)
    arguments.update(overrides)
    return pyconfig.initialize([sys.argv[0], get_test_config_path()], **arguments)

  def _paged(self, **overrides):
    arguments = {"attention": "gpu_paged", "scan_layers": False, "paged_num_blocks": 256}
    arguments.update(overrides)
    return self._config(**arguments)

  def test_gpu_paged_is_an_accepted_attention_value(self):
    cfg = self._paged()
    self.assertEqual(cfg.attention, "gpu_paged")
    self.assertEqual(cfg.paged_attention_backend, "auto")
    self.assertEqual(cfg.paged_page_size, TOKENS_PER_PAGE)

  def test_scan_layers_is_refused(self):
    """Scanning stacks the caches, which would copy the whole pool every step."""
    with self.assertRaisesRegex(ValueError, "scan_layers=False"):
      self._paged(scan_layers=True)

  def test_a_pool_size_is_required(self):
    """Capacity has no safe default: too small caps concurrency silently."""
    with self.assertRaisesRegex(ValueError, "paged_num_blocks"):
      self._config(attention="gpu_paged", scan_layers=False)

  def test_the_context_length_defaults_to_the_target_length(self):
    """max_target_length already states the longest sequence to be served."""
    cfg = self._paged()
    self.assertEqual(cfg.paged_max_context_len, cfg.max_target_length)

  def test_a_pool_too_small_for_one_request_is_refused(self):
    """Otherwise this surfaces as backpressure that never clears."""
    with self.assertRaisesRegex(ValueError, "no request could ever finish"):
      self._paged(paged_num_blocks=2, paged_max_context_len=1024)

  def test_other_attention_kernels_are_unaffected_by_the_guards(self):
    cfg = self._config(attention="dot_product", scan_layers=True)
    self.assertEqual(cfg.attention, "dot_product")
    self.assertEqual(cfg.paged_num_blocks, 0)


if __name__ == "__main__":
  unittest.main()
