"""CPU-only tests for the paged KV control plane.

All of this is host logic over small integer arrays, so all of it is testable
with no accelerator. That is the point of the layering, and these tests are what
make the claim more than an assertion.

A trap worth knowing about before trusting a green run here: `tests/conftest.py`
auto-marks any test without a hardware marker as `cpu_only`, and `cpu_only` tests
are skipped whenever an accelerator is visible. On a GPU box this whole file
therefore *skips* rather than passes, silently and quickly. Run it with
`JAX_PLATFORMS=cpu`, and read the count, not the colour.

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
    DirtyPageError,
    DoubleFreeError,
    KvControlPlane,
    NativeKvControlPlane,
    PageCapacityError,
    PageMap,
    PagedBlockAllocator,
    PageState,
    PageStateError,
    RequestDescriptor,
    RequestHandle,
    RequestState,
    StaleRequestHandleError,
    build_decode_table,
    build_page_table,
    decode_needs_new_page,
    last_page_occupancy,
    new_pages_for_extend,
    pages_for_tokens,
    token_slot,
)
from maxtext.inference.kv_control.logical_block import LogicalBlock, check_transition

PAGE = 16


def _layout(**kw) -> KvStorageLayoutV1:
  """A small but realistic pool geometry, overridable field by field."""
  base = {
      "tokens_per_page": PAGE,
      "num_pages": 64,
      "num_layers": 4,
      "num_kv_heads": 8,
      "head_dim": 128,
      "dtype": "bfloat16",
  }
  base.update(kw)
  return KvStorageLayoutV1(**base)


class PageArithmeticTest(unittest.TestCase):
  """The ceilings and offsets every other module leans on."""

  def test_pages_for_tokens(self):
    self.assertEqual(pages_for_tokens(0, PAGE), 0)
    self.assertEqual(pages_for_tokens(1, PAGE), 1)
    self.assertEqual(pages_for_tokens(PAGE, PAGE), 1)
    self.assertEqual(pages_for_tokens(PAGE + 1, PAGE), 2)

  def test_extend_continues_the_open_page_rather_than_paying_for_it(self):
    """A difference of ceilings, not a ceiling of the difference.

    Growing 8 -> 25 tokens needs one new page: the first page already holds
    positions 0-15, so eight of the seventeen new tokens land in it for free.
    Costing the extend as ceil(17/16) would claim two and over-allocate on every
    mid-page extend, which is the common case.
    """
    self.assertEqual(new_pages_for_extend(8, 25, PAGE), 1)
    self.assertNotEqual(new_pages_for_extend(8, 25, PAGE), pages_for_tokens(25 - 8, PAGE))

  def test_extend_from_empty_is_a_plain_ceiling(self):
    self.assertEqual(new_pages_for_extend(0, 33, PAGE), 3)

  def test_extend_within_the_open_page_is_free(self):
    self.assertEqual(new_pages_for_extend(1, PAGE, PAGE), 0)

  def test_extend_backwards_is_rejected(self):
    with self.assertRaises(ValueError):
      new_pages_for_extend(20, 10, PAGE)

  def test_decode_takes_a_page_only_when_it_crosses_a_boundary(self):
    self.assertTrue(decode_needs_new_page(1, PAGE))  # the very first token
    self.assertFalse(decode_needs_new_page(PAGE, PAGE))  # fills the first page
    self.assertTrue(decode_needs_new_page(PAGE + 1, PAGE))  # opens the second
    self.assertFalse(decode_needs_new_page(PAGE + 2, PAGE))

  def test_last_page_occupancy_reports_a_full_page_as_full(self):
    self.assertEqual(last_page_occupancy(0, PAGE), 0)
    self.assertEqual(last_page_occupancy(PAGE, PAGE), PAGE)
    self.assertEqual(last_page_occupancy(PAGE + 1, PAGE), 1)
    self.assertEqual(last_page_occupancy(33, PAGE), 1)

  def test_token_slot_is_page_major(self):
    self.assertEqual(token_slot(3, 32, PAGE), 48)
    self.assertEqual(token_slot(5, 19, PAGE), 83)


class PageStateTest(unittest.TestCase):
  """The readable-extent gate, expressed as a transition."""

  def test_a_fresh_page_is_not_readable_until_written(self):
    with self.assertRaises(PageStateError):
      check_transition(PageState.FREE, PageState.READY)

  def test_written_then_ready_is_the_normal_path(self):
    block = LogicalBlock(page_id=7, epoch=1)
    self.assertFalse(block.is_readable)
    block.set_state(PageState.READY)
    self.assertTrue(block.is_readable)

  def test_a_ready_page_may_be_reopened_for_more_tokens(self):
    block = LogicalBlock(page_id=7, epoch=1, state=PageState.READY)
    block.set_state(PageState.WRITING)
    self.assertFalse(block.is_readable)

  def test_restating_the_current_state_is_not_an_error(self):
    block = LogicalBlock(page_id=7, epoch=1)
    block.set_state(PageState.WRITING)
    self.assertIs(block.state, PageState.WRITING)


class AllocatorTest(unittest.TestCase):
  """Free-list mechanics, including the parts the reference gets away without."""

  def _allocator(self, num_pages=8, **kw):
    return PagedBlockAllocator(num_pages=num_pages, tokens_per_page=PAGE, debug_mode=True, **kw)

  def test_the_padding_page_is_never_handed_out(self):
    alloc = self._allocator(num_pages=8)
    self.assertEqual(alloc.capacity_pages, 7)
    pages = alloc.alloc(7)
    self.assertIsNotNone(pages)
    self.assertNotIn(0, pages.tolist())
    self.assertEqual(pages.tolist(), [1, 2, 3, 4, 5, 6, 7])

  def test_a_non_zero_reserved_page_is_also_respected(self):
    alloc = self._allocator(num_pages=8, padding_page_id=3)
    self.assertEqual(alloc.alloc(7).tolist(), [0, 1, 2, 4, 5, 6, 7])

  def test_allocation_pops_the_front_of_a_sorted_list(self):
    alloc = self._allocator()
    self.assertEqual(alloc.alloc(3).tolist(), [1, 2, 3])
    self.assertEqual(alloc.alloc(2).tolist(), [4, 5])

  def test_exhaustion_returns_none_rather_than_raising(self):
    alloc = self._allocator()
    self.assertIsNotNone(alloc.alloc(7))
    self.assertIsNone(alloc.alloc(1))

  def test_freeing_stages_the_pages_instead_of_sorting_immediately(self):
    """The cheap half of the two-tier list: no sort on the free path."""
    alloc = self._allocator()
    alloc.alloc(7)
    self.assertEqual(alloc.num_free_pages, 0)
    alloc.free([5, 2])
    self.assertEqual(alloc.num_free_pages, 0, "freed pages must not reach the allocation list yet")
    self.assertEqual(alloc.available_pages, 2)

  def test_the_merge_happens_when_allocation_would_otherwise_fail(self):
    alloc = self._allocator()
    alloc.alloc(7)
    alloc.free([5, 2])
    self.assertEqual(alloc.alloc(2).tolist(), [2, 5], "the merge must also restore sorted order")
    self.assertEqual(alloc.num_free_pages, 0)

  def test_an_allocation_that_fits_the_front_list_skips_the_merge(self):
    alloc = self._allocator()
    alloc.alloc(3)  # pages 1-3, leaving 4-7 in the front list
    alloc.free([2])
    self.assertEqual(alloc.alloc(1).tolist(), [4], "page 2 was staged, so it must not be reused yet")
    self.assertEqual(alloc.num_free_pages, 3, "pages 5-7 remain in the front list")
    self.assertEqual(alloc.available_pages, 4, "page 2 is still staged, and still counts as available")

  def test_many_separate_frees_merge_into_one_sorted_list(self):
    """The staging path with many entries, not the single-free case above.

    Freed one call at a time and in descending order, so a merge that failed to
    consider every staged entry, or to re-sort, would show up here.
    """
    alloc = self._allocator(num_pages=32)
    pages = alloc.alloc(31).tolist()
    for page in reversed(pages):
      alloc.free([page])
    self.assertEqual(alloc.num_free_pages, 0)
    self.assertEqual(alloc.available_pages, 31)
    self.assertEqual(alloc.alloc(31).tolist(), sorted(pages))

  def test_available_tokens_counts_staged_pages(self):
    alloc = self._allocator()
    alloc.alloc(7)
    alloc.free([1, 2])
    self.assertEqual(alloc.available_tokens, 2 * PAGE)

  def test_a_double_free_is_diagnosed_not_absorbed(self):
    """The reference deduplicates this silently; a page freed twice is a bug."""
    alloc = self._allocator()
    pages = alloc.alloc(3)
    alloc.free(pages)
    with self.assertRaises(DoubleFreeError):
      alloc.free(pages)

  def test_duplicates_within_one_free_call_are_fine(self):
    """They are what converting token slots to page ids produces."""
    alloc = self._allocator()
    alloc.alloc(3)
    self.assertEqual(alloc.free([2, 2, 3]).tolist(), [2, 3])

  def test_freeing_the_reserved_page_is_rejected(self):
    alloc = self._allocator()
    with self.assertRaises(ValueError):
      alloc.free([0])

  def test_freeing_outside_the_pool_is_rejected(self):
    alloc = self._allocator()
    with self.assertRaises(ValueError):
      alloc.free([99])

  def test_free_token_slots_drops_the_padding_sentinel(self):
    """A slot_mapping carries -1 for padded rows, which is correct, not an error."""
    alloc = self._allocator()
    alloc.alloc(4)
    freed = alloc.free_token_slots([-1, 2 * PAGE, 2 * PAGE + 5, 3 * PAGE, -1])
    self.assertEqual(freed.tolist(), [2, 3])

  def test_free_token_slots_on_an_all_padding_step_frees_nothing(self):
    alloc = self._allocator()
    self.assertEqual(alloc.free_token_slots([-1, -1]).size, 0)

  def test_the_epoch_advances_when_a_page_is_reused(self):
    alloc = self._allocator()
    page = int(alloc.alloc(1)[0])
    first = alloc.epoch_of(page)
    self.assertTrue(alloc.holds(page, first))

    alloc.free([page])
    self.assertFalse(alloc.holds(page, first), "a freed page must not satisfy a live reference")

    alloc.merge_released()
    self.assertEqual(int(alloc.alloc(1)[0]), page)
    self.assertEqual(alloc.epoch_of(page), first + 1)
    self.assertFalse(alloc.holds(page, first), "the previous owner's reference must not match")

  def test_clear_invalidates_every_outstanding_reference(self):
    alloc = self._allocator()
    page = int(alloc.alloc(1)[0])
    epoch = alloc.epoch_of(page)
    alloc.clear()
    self.assertEqual(alloc.available_pages, alloc.capacity_pages)
    self.assertFalse(alloc.holds(page, epoch))

  def test_churn_returns_every_page(self):
    """The M4 exit criterion in miniature: no leak under repeated turnover."""
    alloc = self._allocator(num_pages=32)
    rng = np.random.default_rng(0)
    for _ in range(500):
      want = int(rng.integers(1, 8))
      pages = alloc.alloc(want)
      if pages is None:
        continue
      alloc.free(pages)
    self.assertEqual(alloc.num_allocated_pages, 0)
    self.assertEqual(alloc.available_pages, alloc.capacity_pages)

  def test_from_layout_takes_the_pool_geometry(self):
    alloc = PagedBlockAllocator.from_layout(_layout(num_pages=64))
    self.assertEqual(alloc.capacity_pages, 63)
    self.assertEqual(alloc.tokens_per_page, PAGE)

  def test_a_never_allocated_page_is_clean(self):
    """The pool is zero-initialised, so a fresh pool costs no scrubbing at all."""
    alloc = self._allocator()
    self.assertEqual(alloc.num_dirty_pages, 0)
    self.assertEqual(alloc.dirty_among(alloc.alloc(3)).size, 0)

  def test_freeing_makes_a_page_dirty(self):
    alloc = self._allocator()
    pages = alloc.alloc(3)
    alloc.free(pages)
    self.assertEqual(alloc.num_dirty_pages, 3)
    for page in pages.tolist():
      self.assertTrue(alloc.is_dirty(page))

  def test_a_recycled_page_is_reported_dirty_on_reallocation(self):
    """The point of the tracking: this page holds someone else's KV."""
    alloc = self._allocator()
    first = alloc.alloc(2)
    alloc.free(first)
    alloc.merge_released()
    second = alloc.alloc(2)
    np.testing.assert_array_equal(np.sort(alloc.dirty_among(second)), np.sort(first))

  def test_marking_scrubbed_clears_the_obligation(self):
    alloc = self._allocator()
    pages = alloc.alloc(2)
    alloc.free(pages)
    alloc.mark_scrubbed(pages)
    self.assertEqual(alloc.num_dirty_pages, 0)
    alloc.merge_released()
    self.assertEqual(alloc.dirty_among(alloc.alloc(2)).size, 0)

  def test_clear_leaves_handed_out_pages_dirty(self):
    """Dropping the free list overwrites nothing, so it cannot clean anything."""
    alloc = self._allocator()
    pages = alloc.alloc(3)
    alloc.clear()
    for page in pages.tolist():
      self.assertTrue(alloc.is_dirty(page))


class PageMapTest(unittest.TestCase):
  """Rows, epochs, and the length bound that keeps reads inside a request."""

  def _map(self, max_requests=2, max_pages_per_request=4):
    return PageMap(
        max_requests=max_requests,
        max_pages_per_request=max_pages_per_request,
        tokens_per_page=PAGE,
    )

  def _descriptor(self, request_id="r0", prompt_len=16, max_new_tokens=16):
    return RequestDescriptor(request_id=request_id, prompt_len=prompt_len, max_new_tokens=max_new_tokens)

  def test_admit_then_release_returns_the_row(self):
    page_map = self._map()
    handle = page_map.admit(self._descriptor())
    self.assertEqual(page_map.num_live, 1)
    self.assertIs(page_map.state(handle), RequestState.WAITING)
    page_map.release(handle)
    self.assertEqual(page_map.num_live, 0)
    self.assertEqual(page_map.available_rows, 2)

  def test_admit_returns_none_when_every_row_is_taken(self):
    page_map = self._map(max_requests=1)
    self.assertIsNotNone(page_map.admit(self._descriptor("a")))
    self.assertIsNone(page_map.admit(self._descriptor("b")))

  def test_release_reports_the_pages_that_were_held(self):
    page_map = self._map()
    handle = page_map.admit(self._descriptor())
    page_map.append_pages(handle, [4, 9])
    np.testing.assert_array_equal(page_map.release(handle), np.array([4, 9], dtype=np.int32))

  def test_a_released_handle_is_refused(self):
    page_map = self._map()
    handle = page_map.admit(self._descriptor())
    page_map.release(handle)
    with self.assertRaises(StaleRequestHandleError):
      page_map.pages(handle)

  def test_a_handle_to_a_reused_row_is_refused(self):
    """The case an index alone cannot catch, and the reason for the epoch."""
    page_map = self._map(max_requests=1)
    first = page_map.admit(self._descriptor("first"))
    page_map.append_pages(first, [3])
    page_map.release(first)

    second = page_map.admit(self._descriptor("second"))
    self.assertEqual(second.row, first.row)
    self.assertNotEqual(second.epoch, first.epoch)
    page_map.append_pages(second, [7])
    with self.assertRaises(StaleRequestHandleError):
      page_map.pages(first)
    np.testing.assert_array_equal(page_map.pages(second), np.array([7], dtype=np.int32))

  def test_a_forged_handle_is_refused(self):
    page_map = self._map()
    page_map.admit(self._descriptor())
    with self.assertRaises(StaleRequestHandleError):
      page_map.pages(RequestHandle(request_id="r0", row=0, epoch=99))
    with self.assertRaises(StaleRequestHandleError):
      page_map.pages(RequestHandle(request_id="r0", row=17, epoch=0))

  def test_pages_are_returned_in_append_order(self):
    page_map = self._map()
    handle = page_map.admit(self._descriptor())
    page_map.append_pages(handle, [9])
    page_map.append_pages(handle, [4, 1])
    np.testing.assert_array_equal(page_map.pages(handle), np.array([9, 4, 1], dtype=np.int32))

  def test_pages_are_returned_as_a_copy(self):
    page_map = self._map()
    handle = page_map.admit(self._descriptor())
    page_map.append_pages(handle, [9])
    pages = page_map.pages(handle)
    pages[0] = 123
    np.testing.assert_array_equal(page_map.pages(handle), np.array([9], dtype=np.int32))

  def test_exceeding_a_row_is_reported(self):
    page_map = self._map(max_pages_per_request=2)
    handle = page_map.admit(self._descriptor())
    with self.assertRaises(PageCapacityError):
      page_map.append_pages(handle, [1, 2, 3])

  def test_advancing_past_the_held_pages_is_refused(self):
    """The bound that stops a kernel reading a page the request does not own."""
    page_map = self._map()
    handle = page_map.admit(self._descriptor())
    page_map.append_pages(handle, [1])
    self.assertEqual(page_map.advance(handle, PAGE), PAGE)
    with self.assertRaises(PageCapacityError):
      page_map.advance(handle, 1)

  def test_live_handles_are_current(self):
    page_map = self._map()
    first = page_map.admit(self._descriptor("a"))
    second = page_map.admit(self._descriptor("b"))
    self.assertEqual([h.request_id for h in page_map.live_handles()], ["a", "b"])
    page_map.release(first)
    self.assertEqual([h.request_id for h in page_map.live_handles()], ["b"])
    self.assertEqual(page_map.live_handles()[0], second)

  def test_from_layout_sizes_rows_from_the_context_length(self):
    page_map = PageMap.from_layout(_layout(), max_requests=4, max_context_len=33)
    self.assertEqual(page_map.max_pages_per_request, 3)


class MetadataTest(unittest.TestCase):
  """Building a `KvPageTableV1` from live bookkeeping."""

  def _map_with(self, specs):
    """`specs` is (request_id, pages, seq_len) per request."""
    page_map = PageMap(max_requests=8, max_pages_per_request=8, tokens_per_page=PAGE)
    handles = []
    for request_id, pages, seq_len in specs:
      handle = page_map.admit(
          RequestDescriptor(request_id=request_id, prompt_len=seq_len, max_new_tokens=0)
      )
      page_map.append_pages(handle, pages)
      page_map.advance(handle, seq_len)
      handles.append(handle)
    return page_map, handles

  def test_decode_table_matches_a_hand_calculation(self):
    page_map, handles = self._map_with([("a", [1, 2, 3], 33), ("b", [4, 5], 20)])
    table = build_decode_table(page_map, handles)

    np.testing.assert_array_equal(table.seq_lens, np.array([33, 20], dtype=np.int32))
    np.testing.assert_array_equal(table.query_lens, np.array([1, 1], dtype=np.int32))
    np.testing.assert_array_equal(table.write_positions, np.array([32, 19], dtype=np.int32))
    np.testing.assert_array_equal(table.indptr(), np.array([0, 3, 5], dtype=np.int32))
    np.testing.assert_array_equal(table.flat_page_indices(), np.array([1, 2, 3, 4, 5], dtype=np.int32))
    np.testing.assert_array_equal(table.last_page_lens(PAGE), np.array([1, 4], dtype=np.int32))
    # page 3 offset 0, and page 5 offset 3
    np.testing.assert_array_equal(table.slot_mapping(PAGE), np.array([48, 83], dtype=np.int32))

  def test_pages_beyond_the_current_length_are_trimmed(self):
    """Over-supply is silent corruption, because occupancy lands on the last page.

    Three pages for a 17-token context would put a last-page length of 1 on
    page 3, so a kernel would read all of page 2 -- whatever the page's previous
    owner left there -- and one token of real data.
    """
    page_map, handles = self._map_with([("a", [1, 2], 17)])
    page_map.append_pages(handles[0], [3])
    self.assertEqual(page_map.num_pages(handles[0]), 3)

    table = build_decode_table(page_map, handles)
    self.assertEqual(table.page_ids, [[1, 2]])
    np.testing.assert_array_equal(table.last_page_lens(PAGE), np.array([1], dtype=np.int32))

  def test_prefill_positions_cover_the_uncached_suffix(self):
    page_map, handles = self._map_with([("a", [1, 2], 18)])
    table = build_page_table(page_map, handles, [18])
    np.testing.assert_array_equal(table.write_positions, np.arange(18, dtype=np.int32))
    expected = [1 * PAGE + i for i in range(PAGE)] + [2 * PAGE, 2 * PAGE + 1]
    np.testing.assert_array_equal(table.slot_mapping(PAGE), np.array(expected, dtype=np.int32))

  def test_a_partial_prefill_starts_where_the_prefix_ended(self):
    """A chunked or prefix-cached request contributes only its suffix."""
    page_map, handles = self._map_with([("a", [1, 2], 20)])
    table = build_page_table(page_map, handles, [4])
    np.testing.assert_array_equal(table.write_positions, np.array([16, 17, 18, 19], dtype=np.int32))

  def test_request_order_names_the_row_behind_each_batch_position(self):
    page_map, handles = self._map_with([("a", [1], 5), ("b", [2], 5)])
    table = build_page_table(page_map, [handles[1], handles[0]], [1, 1])
    np.testing.assert_array_equal(table.request_order, np.array([1, 0], dtype=np.int32))

  def test_a_query_longer_than_the_context_is_rejected(self):
    page_map, handles = self._map_with([("a", [1], 4)])
    with self.assertRaises(ValueError):
      build_page_table(page_map, handles, [5])

  def test_mismatched_lengths_are_rejected(self):
    page_map, handles = self._map_with([("a", [1], 4)])
    with self.assertRaises(ValueError):
      build_page_table(page_map, handles, [1, 1])

  def test_an_empty_batch_builds_an_empty_table(self):
    page_map, _ = self._map_with([])
    table = build_page_table(page_map, [], [])
    self.assertEqual(table.num_requests, 0)
    self.assertEqual(table.num_tokens, 0)
    np.testing.assert_array_equal(table.indptr(), np.zeros((1,), dtype=np.int32))


class NativeKvControlPlaneTest(unittest.TestCase):
  """Admission, reservation, release, and the all-or-nothing property."""

  def _plane(self, num_pages=16, max_requests=4, max_context_len=64, **kw):
    return NativeKvControlPlane(
        layout=_layout(num_pages=num_pages),
        max_requests=max_requests,
        max_context_len=max_context_len,
        debug_mode=True,
        **kw,
    )

  def _admit(self, plane, request_id="r0", prompt_len=16, max_new_tokens=16):
    return plane.admit(
        RequestDescriptor(request_id=request_id, prompt_len=prompt_len, max_new_tokens=max_new_tokens)
    )

  def test_it_satisfies_the_control_plane_protocol(self):
    self.assertIsInstance(self._plane(), KvControlPlane)

  def test_a_request_longer_than_the_configured_context_is_refused(self):
    plane = self._plane(max_context_len=32)
    with self.assertRaises(ValueError):
      self._admit(plane, prompt_len=32, max_new_tokens=1)

  def test_a_pool_too_small_for_one_request_fails_at_construction(self):
    """Better than surfacing later as backpressure that never clears."""
    with self.assertRaises(ValueError):
      self._plane(num_pages=3, max_context_len=64)

  def test_prefill_reserves_exactly_the_pages_the_prompt_needs(self):
    plane = self._plane()
    handle = self._admit(plane, prompt_len=33)
    before = plane.allocator.available_pages
    self.assertTrue(plane.reserve([handle], [33]))
    self.assertEqual(plane.allocator.available_pages, before - 3)
    self.assertEqual(plane.page_map.seq_len(handle), 33)

  def test_decode_takes_a_page_only_on_a_boundary(self):
    plane = self._plane()
    handle = self._admit(plane, prompt_len=PAGE)
    self.assertTrue(plane.reserve([handle], [PAGE]))
    self.assertEqual(plane.page_map.num_pages(handle), 1)

    self.assertTrue(plane.reserve_decode([handle]))  # position 16 opens page 2
    self.assertEqual(plane.page_map.num_pages(handle), 2)

    self.assertTrue(plane.reserve_decode([handle]))  # position 17 continues it
    self.assertEqual(plane.page_map.num_pages(handle), 2)
    self.assertEqual(plane.page_map.seq_len(handle), PAGE + 2)

  def test_a_batch_reservation_that_cannot_fit_changes_nothing(self):
    """A partial reservation would leave a batch no page table can describe."""
    plane = self._plane(num_pages=4, max_requests=3, max_context_len=32)
    handles = [self._admit(plane, f"r{i}", prompt_len=PAGE, max_new_tokens=PAGE) for i in range(3)]
    self.assertTrue(plane.reserve(handles, [PAGE] * 3))
    self.assertEqual(plane.allocator.available_pages, 0)

    self.assertFalse(plane.reserve_decode(handles), "three boundary crossings cannot fit in zero pages")
    for handle in handles:
      self.assertEqual(plane.page_map.seq_len(handle), PAGE)
      self.assertEqual(plane.page_map.num_pages(handle), 1)

  def test_generating_past_the_admitted_bound_is_reported(self):
    plane = self._plane(max_context_len=PAGE)
    handle = self._admit(plane, prompt_len=PAGE, max_new_tokens=0)
    self.assertTrue(plane.reserve([handle], [PAGE]))
    with self.assertRaises(PageCapacityError):
      plane.reserve_decode([handle])

  def test_release_returns_the_pages_to_the_pool(self):
    plane = self._plane()
    capacity = plane.allocator.capacity_pages
    handle = self._admit(plane, prompt_len=33)
    plane.reserve([handle], [33])
    reclaimed = plane.release(handle)
    self.assertEqual(reclaimed.size, 3)
    self.assertEqual(plane.allocator.available_pages, capacity)
    self.assertEqual(plane.allocator.num_allocated_pages, 0)

  def test_releasing_twice_is_reported(self):
    plane = self._plane()
    handle = self._admit(plane)
    plane.reserve([handle], [16])
    plane.release(handle)
    with self.assertRaises(StaleRequestHandleError):
      plane.release(handle)

  def test_a_reserve_of_zero_tokens_is_a_no_op(self):
    plane = self._plane()
    handle = self._admit(plane)
    before = plane.allocator.available_pages
    self.assertTrue(plane.reserve([handle], [0]))
    self.assertEqual(plane.allocator.available_pages, before)
    self.assertEqual(plane.page_map.seq_len(handle), 0)

  def test_an_empty_batch_reserves_successfully(self):
    self.assertTrue(self._plane().reserve([], []))

  def test_a_mixed_length_batch_under_churn_leaks_nothing(self):
    """The M4 exit criterion, host side: turnover must return every page.

    Requests of assorted lengths arrive, decode a while, and leave. If either
    the free list or the page map dropped a page, the pool would not come back
    to its full capacity.
    """
    plane = self._plane(num_pages=64, max_requests=4, max_context_len=128)
    capacity = plane.allocator.capacity_pages
    rng = np.random.default_rng(0)
    live: list[RequestHandle] = []

    for step in range(200):
      if len(live) < 4 and rng.random() < 0.5:
        prompt_len = int(rng.integers(1, 40))
        handle = self._admit(plane, f"r{step}", prompt_len=prompt_len, max_new_tokens=32)
        if handle is not None and plane.reserve([handle], [prompt_len]):
          plane.confirm_scrubbed(plane.pending_scrub())
          plane.page_map.set_state(handle, RequestState.DECODE)
          live.append(handle)
        elif handle is not None:
          plane.release(handle)

      if live:
        if plane.reserve_decode(live):
          # Standing in for the driver's device-side scrub. Without it the next
          # line raises, which is the whole point of the gate.
          plane.confirm_scrubbed(plane.pending_scrub())
          table = plane.build_decode_table(live)
          table.validate(PAGE)
          self.assertEqual(table.num_tokens, len(live))
        finished = [h for h in live if rng.random() < 0.25]
        for handle in finished:
          plane.release(handle)
          live = [h for h in live if h != handle]

    for handle in live:
      plane.release(handle)
    self.assertEqual(plane.num_live, 0)
    self.assertEqual(plane.allocator.num_allocated_pages, 0)
    self.assertEqual(plane.allocator.available_pages, capacity)

  def test_a_fresh_pool_needs_no_scrubbing(self):
    plane = self._plane()
    handle = self._admit(plane, prompt_len=33)
    self.assertTrue(plane.reserve([handle], [33]))
    self.assertEqual(plane.pending_scrub().size, 0)
    plane.build_page_table([handle], [33])  # must not raise

  def test_a_recycled_page_must_be_scrubbed_before_it_can_be_described(self):
    """The acceptance gate, host side: a dirty page cannot reach a kernel.

    A page table is read together with its last-page lengths, so a page holding
    a previous request's KV inside a live extent produces plausible attention
    output and no diagnostic. Refusing to build the table is what turns that into
    an error at the one point where it is still cheap to notice.
    """
    # Exactly one allocatable page, so the second request is forced to take the
    # first one's. A larger pool would hand out a fresh page and never recycle.
    plane = self._plane(num_pages=2, max_requests=2, max_context_len=PAGE)
    first = self._admit(plane, "first", prompt_len=PAGE, max_new_tokens=0)
    plane.reserve([first], [PAGE])
    plane.release(first)

    second = self._admit(plane, "second", prompt_len=PAGE, max_new_tokens=0)
    self.assertTrue(plane.reserve([second], [PAGE]))
    recycled = plane.pending_scrub()
    self.assertEqual(recycled.size, 1, "the reused page must be reported as needing a scrub")

    with self.assertRaises(DirtyPageError):
      plane.build_page_table([second], [PAGE])

    plane.confirm_scrubbed(recycled)
    self.assertEqual(plane.pending_scrub().size, 0)
    plane.build_page_table([second], [PAGE])  # now permitted

  def test_a_partial_confirmation_still_blocks(self):
    """Scrubbing some of what was recycled must not clear the whole obligation."""
    plane = self._plane(num_pages=5, max_requests=2, max_context_len=64)
    first = self._admit(plane, "first", prompt_len=3 * PAGE, max_new_tokens=0)
    plane.reserve([first], [3 * PAGE])
    plane.release(first)

    second = self._admit(plane, "second", prompt_len=3 * PAGE, max_new_tokens=0)
    plane.reserve([second], [3 * PAGE])
    recycled = plane.pending_scrub()
    self.assertEqual(recycled.size, 3)

    plane.confirm_scrubbed(recycled[:2])
    self.assertEqual(plane.pending_scrub().size, 1)
    with self.assertRaises(DirtyPageError):
      plane.build_page_table([second], [3 * PAGE])

  def test_a_dirty_page_trimmed_as_unreadable_does_not_block(self):
    """The check covers the readable extent, not every page a request holds."""
    plane = self._plane(num_pages=8, max_requests=2, max_context_len=64)
    donor = self._admit(plane, "donor", prompt_len=PAGE, max_new_tokens=0)
    plane.reserve([donor], [PAGE])
    dirtied = int(plane.page_map.pages(donor)[0])
    plane.release(donor)
    self.assertTrue(plane.allocator.is_dirty(dirtied))

    reader = self._admit(plane, "reader", prompt_len=PAGE, max_new_tokens=PAGE)
    plane.reserve([reader], [PAGE])
    plane.confirm_scrubbed(plane.pending_scrub())

    # Held, but past the recorded length, so the table trims it away and no
    # kernel can reach it this step.
    plane.page_map.append_pages(reader, [dirtied])
    table = plane.build_page_table([reader], [PAGE])
    self.assertNotIn(dirtied, table.flat_page_indices().tolist())

  def test_the_table_a_step_produces_is_internally_consistent(self):
    plane = self._plane()
    handles = [self._admit(plane, f"r{i}", prompt_len=17 + i, max_new_tokens=4) for i in range(3)]
    self.assertTrue(plane.reserve(handles, [17, 18, 19]))
    table = plane.build_page_table(handles, [17, 18, 19])

    table.validate(PAGE)
    self.assertEqual(table.num_requests, 3)
    self.assertEqual(table.num_tokens, 17 + 18 + 19)
    slots = table.slot_mapping(PAGE)
    self.assertEqual(np.unique(slots).size, slots.size, "two tokens must never share a pool slot")
    self.assertTrue(np.all(slots >= PAGE), "no token may land on the reserved padding page")


if __name__ == "__main__":
  unittest.main()
