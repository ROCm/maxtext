"""Moving pages between the pool and host memory, and back unchanged.

The round trip is the whole property: offload is only useful if a reloaded page
is byte-identical to what was evicted, and the failure mode if it is not is a
request resuming with subtly wrong KV rather than an error.

Skipped where `pinned_host` is not advertised, which is the case on the CPU
backend. That is a real skip rather than a hidden pass -- the milestone's crux
assumption is that ROCm advertises it, and a suite that silently reported green
on CPU would be testing nothing.

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
import pytest

import jax
import jax.numpy as jnp

from maxtext.inference.kv_common import KvStorageLayoutV1
from maxtext.inference.kv_execution.page_staging import (
    HOST_MEMORY_KIND,
    offload_pages,
    pad_page_indices,
    reload_pages,
)
from maxtext.inference.kv_execution.pool_factory import allocate_pool

PAGE = 8
PAGES = 16
LAYERS = 2


def host_memory_available() -> bool:
  return HOST_MEMORY_KIND in {m.kind for m in jax.devices()[0].addressable_memories()}


def a_pool():
  layout = KvStorageLayoutV1(
      tokens_per_page=PAGE,
      num_pages=PAGES,
      num_layers=LAYERS,
      num_kv_heads=2,
      head_dim=4,
      dtype="float32",
      padding_page_id=0,
  )
  return allocate_pool(layout, dtype=jnp.float32)


def paint(pool, page_ids, value):
  """Write a recognisable constant into whole pages of every layer."""
  for layer in range(pool.num_layers):
    k = pool.k_pages[layer].at[jnp.asarray(page_ids)].set(value)
    v = pool.v_pages[layer].at[jnp.asarray(page_ids)].set(value + 0.5)
    pool.replace_layer(layer, k, v)


def page_values(pool, layer, page_id):
  return np.asarray(pool.k_pages[layer])[page_id]


class PadIndicesTest(unittest.TestCase):
  """Runs anywhere: pure numpy."""

  def test_a_power_of_two_count_is_left_alone(self):
    np.testing.assert_array_equal(pad_page_indices([3, 1, 4, 1]), [3, 1, 4, 1])

  def test_a_ragged_count_is_padded_by_repeating_the_first_page(self):
    """Repetition rather than a sentinel, so the padding is idempotent in both
    directions: reading a page twice is free and writing it twice with the same
    bytes changes nothing."""
    np.testing.assert_array_equal(pad_page_indices([5, 6, 7]), [5, 6, 7, 5])

  def test_an_empty_list_stays_empty(self):
    self.assertEqual(pad_page_indices([]).size, 0)


@pytest.mark.gpu_only
@unittest.skipUnless(host_memory_available(), f"{HOST_MEMORY_KIND} is not advertised here")
class PageStagingTest(unittest.TestCase):

  def test_the_staged_copy_really_lives_in_host_memory(self):
    """If this fails the milestone is measuring a device-to-device copy and the
    capacity claim is empty."""
    pool = a_pool()
    paint(pool, [2, 3], 1.0)

    staged = offload_pages(pool, [2, 3])

    for array in staged.k + staged.v:
      self.assertEqual(array.sharding.memory_kind, HOST_MEMORY_KIND)

  def test_pages_come_back_byte_identical(self):
    pool = a_pool()
    paint(pool, [4, 5, 6], 7.0)
    before = [page_values(pool, layer, 5) for layer in range(LAYERS)]

    staged = offload_pages(pool, [4, 5, 6])
    paint(pool, [4, 5, 6], 0.0)  # stand in for the page being reused
    reload_pages(pool, staged)

    for layer in range(LAYERS):
      np.testing.assert_array_equal(page_values(pool, layer, 5), before[layer])

  def test_a_ragged_page_count_round_trips_through_the_padding(self):
    """Three pages pad to four by repeating the first. The repeated row must not
    corrupt the page it names, which it cannot only because gather and scatter
    are handed the same padded index array."""
    pool = a_pool()
    paint(pool, [9, 10, 11], 3.0)

    staged = offload_pages(pool, [9, 10, 11])
    self.assertEqual(staged.page_ids.size, 4)
    paint(pool, [9, 10, 11], 0.0)
    reload_pages(pool, staged)

    for page in (9, 10, 11):
      np.testing.assert_array_equal(
          page_values(pool, 0, page), np.full((PAGE, 2, 4), 3.0, np.float32)
      )

  def test_offload_does_not_disturb_the_pool(self):
    """It is a read. Freeing or scrubbing on offload would let a failed transfer
    look like a successful eviction."""
    pool = a_pool()
    paint(pool, [1], 2.0)

    offload_pages(pool, [1])

    np.testing.assert_array_equal(
        page_values(pool, 0, 1), np.full((PAGE, 2, 4), 2.0, np.float32)
    )

  def test_reload_leaves_every_other_page_alone(self):
    pool = a_pool()
    paint(pool, [12], 5.0)
    paint(pool, [13], 9.0)
    staged = offload_pages(pool, [12])
    paint(pool, [12], 0.0)

    reload_pages(pool, staged)

    np.testing.assert_array_equal(
        page_values(pool, 0, 13), np.full((PAGE, 2, 4), 9.0, np.float32)
    )

  def test_an_empty_offload_is_a_no_op_rather_than_an_error(self):
    """The common case on a pool that has not come under pressure yet."""
    pool = a_pool()
    staged = offload_pages(pool, [])

    self.assertEqual(staged.num_live, 0)
    reload_pages(pool, staged)

  def test_the_staged_size_reports_what_was_moved(self):
    """The number the capacity claim is built from, so it is worth asserting
    rather than trusting."""
    pool = a_pool()
    staged = offload_pages(pool, [1, 2])

    expected = 2 * LAYERS * 2 * PAGE * 2 * 4 * 4  # k+v, layers, pages, tokens, heads, dim, f32
    self.assertEqual(staged.nbytes(), expected)


if __name__ == "__main__":
  unittest.main()
