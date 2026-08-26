"""CPU-only tests for the neutral paged-KV vocabulary.

These run with no accelerator, which is the point of keeping the layer pure
numpy: pool sizing, slot arithmetic and last-page occupancy are all exactly the
sort of data-dependent host logic that cannot be traced, and all of it is
testable in ordinary CI.

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
"""

import unittest

import numpy as np

from maxtext.inference.kv_common import KvPageTableV1, KvStorageLayoutV1

# Imported by package path, which is cheap: `maxtext/__init__.py` resolves its
# heavy exports through a lazy `__getattr__`, so reaching a submodule loads
# neither jax nor the config stack. `kv_import_rule_test.py` is what holds that
# true, statically over the whole layer and once against a fresh interpreter.


class KvStorageLayoutTest(unittest.TestCase):
  """Pool geometry and sizing."""

  def _layout(self, **kw):
    """A realistic pool geometry, overridable field by field."""
    base = {
        "tokens_per_page": 16,
        "num_pages": 1024,
        "num_layers": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "dtype": "bfloat16",
    }
    base.update(kw)
    return KvStorageLayoutV1(**base)

  def test_bytes_per_page(self):
    layout = self._layout()
    # 16 tokens * 8 heads * 128 dim * 2 bytes
    self.assertEqual(layout.bytes_per_page(), 16 * 8 * 128 * 2)

  def test_pool_bytes_matches_hand_calculation(self):
    layout = self._layout()
    expected = 2 * 32 * 1024 * (16 * 8 * 128 * 2)
    self.assertEqual(layout.pool_bytes_per_shard(), expected)

  def test_bytes_per_token_is_the_capacity_unit(self):
    """Per-token cost is what turns pool size into concurrency."""
    layout = self._layout(num_layers=80)
    # 8 heads * 128 dim * 2 bytes * 2 (K,V) * 80 layers = 320 KiB
    self.assertEqual(layout.bytes_per_token(), 320 * 1024)

  def test_clean_partition_when_heads_divide_shards(self):
    layout = self._layout(num_kv_heads=8, kv_head_shards=4)
    self.assertEqual(layout.heads_per_shard(), 2)
    self.assertEqual(layout.replication_factor(), 1)

  def test_one_head_per_shard_at_the_boundary(self):
    """`kv_head_shards == num_kv_heads` partitions exactly, it does not replicate.

    The case an earlier version got wrong, and it is the common one: TP=8 on a
    model with 8 KV heads. Returning the full head count here sized every shard's
    pool a factor of TP too large, and disagreed with `replication_factor`, which
    correctly reported 1.
    """
    layout = self._layout(num_kv_heads=8, kv_head_shards=8)
    self.assertEqual(layout.heads_per_shard(), 1)
    self.assertEqual(layout.replication_factor(), 1)
    self.assertEqual(layout.total_pool_bytes(), layout.pool_bytes_per_shard() * 8)

  def test_replication_when_shards_exceed_heads(self):
    """GQA above its KV-head count replicates, multiplying the footprint.

    Each rank computes a subset of the *query* heads and therefore needs exactly
    the one KV head those map to -- not every head. So the shard holds one head
    and it is the number of copies that grows, which is what
    `replication_factor` reports and what `total_pool_bytes` multiplies by.
    """
    layout = self._layout(num_kv_heads=2, kv_head_shards=8)
    self.assertEqual(layout.heads_per_shard(), 1)
    self.assertEqual(layout.replication_factor(), 4)
    # 8 shards holding one head each is 4x the 2 logical heads, and the two ways
    # of counting the footprint have to agree or pool sizing is wrong.
    self.assertEqual(layout.total_pool_bytes(), layout.pool_bytes_per_shard() * 8)
    self.assertEqual(
        layout.total_pool_bytes(),
        layout.pool_bytes_per_shard() * layout.num_kv_heads * layout.replication_factor(),
    )

  def test_mqa_replicates_everywhere(self):
    layout = self._layout(num_kv_heads=1, kv_head_shards=8)
    self.assertEqual(layout.heads_per_shard(), 1)
    self.assertEqual(layout.replication_factor(), 8)

  def test_indivisible_sharding_is_rejected_at_construction(self):
    with self.assertRaises(ValueError):
      self._layout(num_kv_heads=6, kv_head_shards=4)

  def test_max_tokens_excludes_padding_page(self):
    layout = self._layout(num_pages=100, tokens_per_page=16)
    self.assertEqual(layout.max_tokens(), 99 * 16)


class KvPageTableTest(unittest.TestCase):
  """Per-step page bookkeeping."""

  def _decode_table(self):
    """Two requests, one new token each, mid-sequence."""
    return KvPageTableV1(
        page_ids=[[1, 2, 3], [4, 5]],
        seq_lens=np.array([33, 20], dtype=np.int32),
        query_lens=np.array([1, 1], dtype=np.int32),
        write_positions=np.array([32, 19], dtype=np.int32),
        request_order=np.array([0, 1], dtype=np.int32),
    )

  def test_validate_accepts_a_consistent_table(self):
    self._decode_table().validate(tokens_per_page=16)

  def test_validate_rejects_too_few_pages(self):
    table = KvPageTableV1(
        page_ids=[[1]],
        seq_lens=np.array([33], dtype=np.int32),
        query_lens=np.array([1], dtype=np.int32),
        write_positions=np.array([32], dtype=np.int32),
        request_order=np.array([0], dtype=np.int32),
    )
    with self.assertRaises(ValueError):
      table.validate(tokens_per_page=16)

  def test_validate_rejects_query_len_mismatch(self):
    table = self._decode_table()
    table.query_lens = np.array([2, 1], dtype=np.int32)
    with self.assertRaises(ValueError):
      table.validate(tokens_per_page=16)

  def test_indptr_is_exclusive_prefix_sum(self):
    np.testing.assert_array_equal(
        self._decode_table().indptr(), np.array([0, 3, 5], dtype=np.int32)
    )

  def test_flat_page_indices_is_request_ordered(self):
    np.testing.assert_array_equal(
        self._decode_table().flat_page_indices(),
        np.array([1, 2, 3, 4, 5], dtype=np.int32),
    )

  def test_last_page_lens_are_exact(self):
    """Exact occupancy is what stops a kernel reading a recycled page's tail."""
    table = self._decode_table()
    # seq_len 33 -> 33 % 16 == 1; seq_len 20 -> 20 % 16 == 4
    np.testing.assert_array_equal(
        table.last_page_lens(tokens_per_page=16), np.array([1, 4], dtype=np.int32)
    )

  def test_last_page_len_of_a_full_page_is_the_page_size(self):
    table = KvPageTableV1(
        page_ids=[[1, 2]],
        seq_lens=np.array([32], dtype=np.int32),
        query_lens=np.array([1], dtype=np.int32),
        write_positions=np.array([31], dtype=np.int32),
        request_order=np.array([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        table.last_page_lens(tokens_per_page=16), np.array([16], dtype=np.int32)
    )

  def test_slot_mapping_decode(self):
    table = self._decode_table()
    slots = table.slot_mapping(tokens_per_page=16)
    # req 0: position 32 -> page slot 2 -> page 3, offset 0 -> 3*16 + 0
    # req 1: position 19 -> page slot 1 -> page 5, offset 3 -> 5*16 + 3
    np.testing.assert_array_equal(slots, np.array([48, 83], dtype=np.int32))

  def test_slot_mapping_prefill_spans_pages(self):
    """A prefill writes several tokens per request, crossing a page boundary."""
    table = KvPageTableV1(
        page_ids=[[7, 9]],
        seq_lens=np.array([18], dtype=np.int32),
        query_lens=np.array([18], dtype=np.int32),
        write_positions=np.arange(18, dtype=np.int32),
        request_order=np.array([0], dtype=np.int32),
    )
    table.validate(tokens_per_page=16)
    slots = table.slot_mapping(tokens_per_page=16)
    expected = [7 * 16 + i for i in range(16)] + [9 * 16 + 0, 9 * 16 + 1]
    np.testing.assert_array_equal(slots, np.array(expected, dtype=np.int32))

  def test_padding_page_maps_to_skip_sentinel(self):
    """Tokens landing on the padding page become -1 so kernels drop them."""
    table = KvPageTableV1(
        page_ids=[[0]],
        seq_lens=np.array([4], dtype=np.int32),
        query_lens=np.array([4], dtype=np.int32),
        write_positions=np.arange(4, dtype=np.int32),
        request_order=np.array([0], dtype=np.int32),
    )
    slots = table.slot_mapping(tokens_per_page=16, padding_page_id=0)
    np.testing.assert_array_equal(slots, np.full((4,), -1, dtype=np.int32))

  def test_slot_mapping_rejects_position_beyond_held_pages(self):
    table = KvPageTableV1(
        page_ids=[[1]],
        seq_lens=np.array([20], dtype=np.int32),
        query_lens=np.array([1], dtype=np.int32),
        write_positions=np.array([19], dtype=np.int32),
        request_order=np.array([0], dtype=np.int32),
    )
    with self.assertRaises(ValueError):
      table.slot_mapping(tokens_per_page=16)


if __name__ == "__main__":
  unittest.main()
