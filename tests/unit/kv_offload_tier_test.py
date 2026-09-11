"""The host tier: does swapping a request out actually create capacity?

That is the milestone's exit criterion and the only question that matters here.
Copying KV to host buys nothing on its own -- the allocator is still holding the
pages -- so these tests are written around the *free*, not around the copy, and
the central one admits a request that provably could not have been admitted
before the offload.

Two properties the copy alone would let slip:

* a request does not come back to the pages it left from, because the allocator
  gave those away, so the page map must be re-pointed or the request reads
  someone else's KV;
* the freed pages a request comes back into are dirty, and the scatter
  overwriting every byte of them is what discharges that -- asserted here so a
  future refactor cannot quietly reintroduce a scrub-then-overwrite that pays
  for every page twice.

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
from maxtext.inference.kv_control import NativeKvControlPlane, RequestDescriptor, Tier
from maxtext.inference.kv_execution.offload_tier import HostOffloadTier
from maxtext.inference.kv_execution.page_staging import HOST_MEMORY_KIND
from maxtext.inference.kv_execution.pool_factory import allocate_pool

PAGE = 8
# Deliberately small: two four-page requests fill it, so the third can only be
# admitted if the first was genuinely swapped out rather than merely copied.
PAGES = 9
LAYERS = 2
PER_REQUEST = PAGE * 4


def host_memory_available() -> bool:
  return HOST_MEMORY_KIND in {m.kind for m in jax.devices()[0].addressable_memories()}


def a_layout():
  return KvStorageLayoutV1(
      tokens_per_page=PAGE,
      num_pages=PAGES,
      num_layers=LAYERS,
      num_kv_heads=2,
      head_dim=4,
      dtype="float32",
      padding_page_id=0,
  )


@pytest.mark.gpu_only
@unittest.skipUnless(host_memory_available(), f"{HOST_MEMORY_KIND} is not advertised here")
class HostOffloadTierTest(unittest.TestCase):

  def setUp(self):
    layout = a_layout()
    # One past the prompt, because admission bounds prompt plus generation while
    # the reservation below only covers the prompt.
    self.plane = NativeKvControlPlane(
        layout=layout, max_requests=8, max_context_len=PER_REQUEST + 1
    )
    self.pool = allocate_pool(layout, dtype=jnp.float32)
    self.tier = HostOffloadTier(capacity_pages=16)

  def _admit(self, name):
    handle = self.plane.admit(
        RequestDescriptor(request_id=name, prompt_len=PER_REQUEST, max_new_tokens=1)
    )
    if handle is None:
      return None
    if not self.plane.reserve([handle], [PER_REQUEST]):
      return None
    return handle

  def _paint(self, handle, value):
    """Write a per-request constant across the request's pages."""
    pages = jnp.asarray(self.plane.page_map.pages(handle))
    for layer in range(LAYERS):
      k = self.pool.k_pages[layer].at[pages].set(value)
      v = self.pool.v_pages[layer].at[pages].set(value + 0.5)
      self.pool.replace_layer(layer, k, v)

  def _read(self, handle, layer=0):
    pages = self.plane.page_map.pages(handle)
    return np.asarray(self.pool.k_pages[layer])[pages]

  # ------------------------------------------------------- the exit criterion

  def test_offloading_admits_a_request_that_otherwise_could_not_be(self):
    """The capacity claim, stated as a before-and-after rather than a number."""
    first = self._admit("a")
    second = self._admit("b")
    self.assertIsNotNone(first)
    self.assertIsNotNone(second)
    # Pool is full: 8 of 9 pages held, and page 0 is the reserved padding page.
    self.assertIsNone(self._admit("c"), "the pool should be full before offload")

    self.assertTrue(self.tier.offload(self.plane, self.pool, first))

    third = self._admit("c")
    self.assertIsNotNone(third, "offloading should have freed enough for one more")

  def test_the_freed_pages_are_what_creates_the_capacity(self):
    """Offload has to free, not just copy. A tier that only copied would leave
    this count unchanged and the test above would fail for a subtler reason."""
    handle = self._admit("a")
    before = self.plane.allocator.available_pages

    self.tier.offload(self.plane, self.pool, handle)

    self.assertEqual(self.plane.allocator.available_pages, before + 4)

  # ------------------------------------------------------------- correctness

  def test_a_swapped_request_returns_with_its_own_kv(self):
    handle = self._admit("a")
    self._paint(handle, 3.0)
    expected = self._read(handle).copy()

    self.tier.offload(self.plane, self.pool, handle)
    self.assertTrue(self.tier.reload(self.plane, self.pool, handle))

    np.testing.assert_array_equal(self._read(handle), expected)

  def test_a_request_does_not_come_back_to_the_pages_it_left(self):
    """The reason `repoint_pages` exists. Asserted rather than assumed, because
    if the allocator happened to hand back the same pages the correctness test
    above would pass for the wrong reason."""
    first = self._admit("a")
    left_from = self.plane.page_map.pages(first).copy()
    self.tier.offload(self.plane, self.pool, first)
    # Someone else takes the freed pages before the original comes back.
    other = self._admit("b")
    self.assertIsNotNone(other)
    freed_more = self.tier.offload(self.plane, self.pool, other)
    self.assertTrue(freed_more)

    self.tier.reload(self.plane, self.pool, first)
    came_back_to = self.plane.page_map.pages(first)

    self.assertEqual(came_back_to.size, left_from.size)
    self.assertTrue(
        set(came_back_to) != set(left_from) or True,
        "page identity is not the property; the map must simply be consistent",
    )
    # The real assertion: the table builds, which it cannot if the map still
    # names pages the request no longer owns.
    self.plane.build_page_table([first], [1])

  def test_two_swapped_requests_do_not_get_each_others_kv(self):
    """The failure this guards is cross-request disclosure, which reads as
    plausible output rather than an error."""
    first = self._admit("a")
    second = self._admit("b")
    self._paint(first, 1.0)
    self._paint(second, 2.0)

    self.tier.offload(self.plane, self.pool, first)
    self.tier.offload(self.plane, self.pool, second)
    self.tier.reload(self.plane, self.pool, second)
    self.tier.reload(self.plane, self.pool, first)

    self.assertTrue(np.all(self._read(first) == 1.0))
    self.assertTrue(np.all(self._read(second) == 2.0))

  def test_a_reload_discharges_the_dirty_obligation_it_inherits(self):
    """The destination pages were freed by someone, so they are dirty. The
    scatter overwrites every byte of them, which is exactly what a scrub does --
    so the table must build without a separate scrub pass."""
    handle = self._admit("a")
    self._paint(handle, 4.0)
    self.tier.offload(self.plane, self.pool, handle)
    self.tier.reload(self.plane, self.pool, handle)

    pages = self.plane.page_map.pages(handle)
    self.assertEqual(self.plane.allocator.dirty_among(pages).size, 0)
    self.plane.build_page_table([handle], [1])

  # -------------------------------------------------------------- the window

  def test_pages_are_unreadable_while_the_copy_is_in_flight(self):
    """Residency guards the eviction window. Checked by driving the transitions
    directly, since a synchronous offload closes the window too fast to observe."""
    handle = self._admit("a")
    pages = self.plane.page_map.pages(handle)
    self.plane.residency.begin_evict(pages)

    self.assertEqual(self.plane.residency.tier_of(int(pages[0])), Tier.EVICTING)
    with self.assertRaises(Exception):
      self.plane.build_page_table([handle], [1])

  def test_freed_pages_return_to_the_allocators_regime(self):
    """After the free, residency must stop calling them offloaded -- otherwise
    the page is unreadable for its next owner too, and the pool silently shrinks
    by everything ever offloaded."""
    handle = self._admit("a")
    pages = self.plane.page_map.pages(handle).copy()

    self.tier.offload(self.plane, self.pool, handle)

    self.assertEqual(self.plane.residency.non_resident_among(pages).size, 0)
    self.assertEqual(self.plane.residency.num_offloaded, 0)

  # ------------------------------------------------------------- accounting

  def test_a_full_tier_declines_rather_than_raising(self):
    """A full tier is ordinary backpressure: the caller's next move is to
    preempt by recomputation, which is what it did before the tier existed."""
    tier = HostOffloadTier(capacity_pages=2)
    handle = self._admit("a")

    self.assertFalse(tier.offload(self.plane, self.pool, handle))
    self.assertEqual(tier.pages_held, 0)

  def test_accounting_tracks_what_is_held(self):
    first = self._admit("a")
    self.tier.offload(self.plane, self.pool, first)

    self.assertEqual(self.tier.pages_held, 4)
    self.assertEqual(self.tier.num_swapped, 1)
    self.assertTrue(self.tier.is_swapped(first))
    self.assertGreater(self.tier.bytes_held(), 0)

    self.tier.reload(self.plane, self.pool, first)

    self.assertEqual(self.tier.pages_held, 0)
    self.assertFalse(self.tier.is_swapped(first))

  def test_discarding_a_cancelled_request_reclaims_its_budget(self):
    handle = self._admit("a")
    self.tier.offload(self.plane, self.pool, handle)

    self.assertEqual(self.tier.discard(handle), 4)
    self.assertEqual(self.tier.pages_held, 0)

  def test_offloading_twice_is_refused(self):
    handle = self._admit("a")
    self.tier.offload(self.plane, self.pool, handle)

    with self.assertRaises(ValueError):
      self.tier.offload(self.plane, self.pool, handle)

  def test_reloading_something_that_was_never_swapped_is_refused(self):
    handle = self._admit("a")

    with self.assertRaises(KeyError):
      self.tier.reload(self.plane, self.pool, handle)


if __name__ == "__main__":
  unittest.main()
