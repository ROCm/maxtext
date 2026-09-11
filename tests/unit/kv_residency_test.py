"""Page residency, and the refusal that makes offload safe.

The property under test is the one M7's "no execution before readiness" exit
criterion names: a page whose bytes are in the host slab must not be readable by
a step. It matters for the same reason the dirty-page refusal matters -- the
failure is silent. A non-resident page still has device bytes, stale or
scrubbed, so a kernel reading it produces fluent wrong output rather than an
error, and no shape check or assertion downstream would notice.

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

from maxtext.inference.kv_common import KvStorageLayoutV1
from maxtext.inference.kv_control import (
    NativeKvControlPlane,
    NotResidentError,
    PageResidency,
    RequestDescriptor,
    Tier,
)

PAGE = 16
PAGES = 32


def a_layout(num_pages=PAGES):
  return KvStorageLayoutV1(
      tokens_per_page=PAGE,
      num_pages=num_pages,
      num_layers=1,
      num_kv_heads=2,
      head_dim=4,
      dtype="bfloat16",
      padding_page_id=0,
  )


class PageResidencyTest(unittest.TestCase):

  def test_every_page_starts_on_the_device(self):
    """The pool is allocated on device, so a page that was never offloaded is
    trivially resident. A deployment that never offloads must behave exactly as
    it did before this table existed."""
    residency = PageResidency(PAGES)

    self.assertEqual(residency.non_resident_among(np.arange(PAGES)).size, 0)
    self.assertEqual(residency.num_offloaded, 0)

  def test_a_page_stops_being_readable_the_moment_eviction_starts(self):
    """Not on completion. The window between "copy started" and "copy finished"
    is exactly where a step could read a page that is being moved out from under
    it, so the state changes at the start."""
    residency = PageResidency(PAGES)
    residency.begin_evict([5])

    self.assertEqual(residency.tier_of(5), Tier.EVICTING)
    self.assertEqual(list(residency.non_resident_among([5])), [5])

  def test_an_offloaded_page_is_not_readable(self):
    residency = PageResidency(PAGES)
    residency.begin_evict([5, 6])
    residency.confirm_evicted([5, 6])

    self.assertEqual(residency.tier_of(5), Tier.HOST)
    self.assertEqual(sorted(residency.non_resident_among([4, 5, 6])), [5, 6])
    self.assertEqual(residency.num_offloaded, 2)

  def test_reloading_makes_a_page_readable_again(self):
    residency = PageResidency(PAGES)
    residency.begin_evict([7])
    residency.confirm_evicted([7])
    residency.confirm_resident([7])

    self.assertEqual(residency.tier_of(7), Tier.DEVICE)
    self.assertEqual(residency.non_resident_among([7]).size, 0)
    self.assertEqual(residency.num_offloaded, 0)

  def test_evicting_an_already_offloaded_page_does_not_resurrect_it(self):
    """`begin_evict` only moves pages that are currently on device. Without that
    guard, a second offload of an already-offloaded page would put it back into
    EVICTING and a later `confirm_resident` would call it readable when its
    bytes are still on the host."""
    residency = PageResidency(PAGES)
    residency.begin_evict([9])
    residency.confirm_evicted([9])
    residency.begin_evict([9])

    self.assertEqual(residency.tier_of(9), Tier.HOST)

  def test_a_page_outside_the_pool_is_rejected(self):
    residency = PageResidency(PAGES)

    with self.assertRaises(ValueError):
      residency.begin_evict([PAGES])

  def test_offloaded_pages_can_be_listed_for_reload(self):
    residency = PageResidency(PAGES)
    residency.begin_evict([2, 3, 4])
    residency.confirm_evicted([2, 3])

    self.assertEqual(sorted(residency.pages_in(Tier.HOST)), [2, 3])
    self.assertEqual(sorted(residency.pages_in(Tier.EVICTING)), [4])


class ControlPlaneRefusalTest(unittest.TestCase):
  """The enforcement point, with a real control plane behind it."""

  def setUp(self):
    self.plane = NativeKvControlPlane(
        layout=a_layout(), max_requests=4, max_context_len=PAGE * 4
    )

  def _admit(self, name, tokens):
    handle = self.plane.admit(
        RequestDescriptor(request_id=name, prompt_len=tokens, max_new_tokens=1)
    )
    self.assertIsNotNone(handle, "the pool should have room for this request")
    self.assertTrue(self.plane.reserve([handle], [tokens]))
    return handle

  def test_a_resident_request_builds_a_table_as_before(self):
    """The negative control. Without it, a refusal test passes on a plane that
    refuses everything."""
    handle = self._admit("r", PAGE)

    table = self.plane.build_page_table([handle], [PAGE])

    self.assertEqual(table.num_requests, 1)

  def test_a_table_naming_an_offloaded_page_is_refused(self):
    handle = self._admit("r", PAGE)
    pages = self.plane.page_map.pages(handle)
    self.plane.residency.begin_evict(pages)
    self.plane.residency.confirm_evicted(pages)

    with self.assertRaises(NotResidentError):
      self.plane.build_page_table([handle], [PAGE])

  def test_a_table_naming_a_mid_eviction_page_is_refused(self):
    """The state that exists specifically so an asynchronous offload has
    somewhere to be between started and finished."""
    handle = self._admit("r", PAGE)
    self.plane.residency.begin_evict(self.plane.page_map.pages(handle))

    with self.assertRaises(NotResidentError):
      self.plane.build_page_table([handle], [PAGE])

  def test_the_table_builds_again_once_the_pages_are_back(self):
    handle = self._admit("r", PAGE)
    pages = self.plane.page_map.pages(handle)
    self.plane.residency.begin_evict(pages)
    self.plane.residency.confirm_evicted(pages)
    self.plane.residency.confirm_resident(pages)

    table = self.plane.build_page_table([handle], [PAGE])

    self.assertEqual(table.num_requests, 1)

  def test_offloading_another_requests_pages_does_not_block_this_one(self):
    """The refusal is over the table's own page list, not the pool at large --
    the same scoping the dirty-page check uses. Offload would be useless if
    evicting one request stalled every other."""
    mine = self._admit("mine", PAGE)
    theirs = self._admit("theirs", PAGE)
    self.plane.residency.begin_evict(self.plane.page_map.pages(theirs))

    table = self.plane.build_page_table([mine], [PAGE])

    self.assertEqual(table.num_requests, 1)


if __name__ == "__main__":
  unittest.main()
