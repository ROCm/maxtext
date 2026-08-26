"""The authoritative control plane when MaxText owns the pages.

Assembles the allocator, the page map, and the metadata builder into the one
object a driver holds. Small on purpose: everything interesting lives in the
parts, and what this adds is two properties none of them can provide alone.

**A batch reservation either succeeds for every request or changes nothing.**
That matters more than it looks. A reservation that half-succeeds leaves some
requests advanced and some not, and there is no page table describing that
state, so the caller's only recovery would be to unwind bookkeeping it does not
own. Allocating the batch's pages in a single request from the free list, after
checking every request can record them, removes the case entirely.

**No page table naming a dirty page can be built.** A recycled page holds its
previous occupant's KV until something overwrites it, so a step must scrub what
it recycled before any kernel reads it. That gives a fixed order per step --
reserve, scrub, confirm, build the table, run -- and `build_page_table` refuses
outright if the confirm has not happened. Making it an error rather than a
convention is the point: the failure it prevents is one request reading
another's KV, which produces plausible tokens and no diagnostic.

The scrub covers the whole recycled page rather than just the tail beyond the
new writes. Exact last-page lengths already stop a well-behaved kernel reading
past valid data, so the redundancy is deliberate -- it costs one page write per
page-boundary crossing and removes any dependence on every kernel, present and
future, honouring that bound.

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

from __future__ import annotations

from typing import Sequence

import numpy as np

from maxtext.inference.kv_common import KvPageTableV1, KvStorageLayoutV1
from maxtext.inference.kv_control import metadata
from maxtext.inference.kv_control.allocator import PagedBlockAllocator
from maxtext.inference.kv_control.logical_block import new_pages_for_extend, pages_for_tokens
from maxtext.inference.kv_control.page_map import PageCapacityError, PageMap
from maxtext.inference.kv_control.request import RequestDescriptor, RequestHandle


class DirtyPageError(RuntimeError):
  """A step would have read a page still holding another request's KV."""


class NativeKvControlPlane:
  """Page accounting owned by MaxText, satisfying `KvControlPlane`."""

  def __init__(
      self,
      layout: KvStorageLayoutV1,
      max_requests: int,
      max_context_len: int,
      debug_mode: bool = False,
  ):
    if max_context_len < 1:
      raise ValueError(f"max_context_len must be at least 1, got {max_context_len}")

    self._layout = layout
    self.max_context_len = int(max_context_len)
    self.allocator = PagedBlockAllocator.from_layout(layout, debug_mode=debug_mode)
    self.page_map = PageMap.from_layout(layout, max_requests=max_requests, max_context_len=max_context_len)
    self._pending_scrub: list[np.ndarray] = []

    # Caught here rather than as a mid-serving allocation failure, which would
    # look like ordinary backpressure and never resolve.
    pages_needed = pages_for_tokens(max_context_len, layout.tokens_per_page)
    if pages_needed > self.allocator.capacity_pages:
      raise ValueError(
          f"a single {max_context_len}-token request needs {pages_needed} pages but the pool has only "
          f"{self.allocator.capacity_pages} allocatable; raise the pool size or lower max_context_len"
      )

  @property
  def layout(self) -> KvStorageLayoutV1:
    return self._layout

  @property
  def available_tokens(self) -> int:
    return self.allocator.available_tokens

  @property
  def num_live(self) -> int:
    return self.page_map.num_live

  def admit(self, descriptor: RequestDescriptor) -> RequestHandle | None:
    if descriptor.max_total_len > self.max_context_len:
      raise ValueError(
          f"request {descriptor.request_id!r} declares a maximum length of {descriptor.max_total_len} "
          f"tokens, past the configured max_context_len of {self.max_context_len}"
      )
    return self.page_map.admit(descriptor)

  def reserve(
      self,
      handles: Sequence[RequestHandle],
      num_new_tokens: Sequence[int] | np.ndarray,
  ) -> bool:
    """Reserve pages for `num_new_tokens` per request, all or nothing."""
    counts = np.asarray(num_new_tokens, dtype=np.int64).reshape(-1)
    if counts.size != len(handles):
      raise ValueError(f"got {len(handles)} handles but {counts.size} token counts")
    if counts.size == 0:
      return True
    if np.any(counts < 0):
      raise ValueError(f"token counts must be non-negative, got {counts.tolist()}")

    tokens_per_page = self._layout.tokens_per_page
    # Both loops run before anything is mutated: the first would otherwise leave
    # a request advanced with no pages recorded for it if a later request in the
    # same batch turned out not to fit its row.
    per_request_pages = []
    for handle, count in zip(handles, counts):
      prefix_len = self.page_map.seq_len(handle)
      new_len = prefix_len + int(count)
      if pages_for_tokens(new_len, tokens_per_page) > self.page_map.max_pages_per_request:
        raise PageCapacityError(
            f"request {handle.request_id!r} would reach {new_len} tokens, past the "
            f"{self.max_context_len} it was admitted under"
        )
      per_request_pages.append(new_pages_for_extend(prefix_len, new_len, tokens_per_page))

    pages = self.allocator.alloc(int(sum(per_request_pages)))
    if pages is None:
      return False

    recycled = self.allocator.dirty_among(pages)
    if recycled.size:
      self._pending_scrub.append(recycled)

    offset = 0
    for handle, count, needed in zip(handles, counts, per_request_pages):
      self.page_map.append_pages(handle, pages[offset : offset + needed])
      offset += needed
      self.page_map.advance(handle, int(count))
    return True

  def pending_scrub(self) -> np.ndarray:
    """Pages reserved this step that must be overwritten before any read.

    Reading this does not discharge the obligation; `confirm_scrubbed` does.
    Idempotent, so a caller may inspect it without committing to anything.
    """
    if not self._pending_scrub:
      return np.empty((0,), dtype=np.int32)
    return np.unique(np.concatenate(self._pending_scrub))

  def confirm_scrubbed(self, page_ids: Sequence[int] | np.ndarray) -> None:
    """Record that `page_ids` have been overwritten on the device.

    Call this only after the write has actually been issued. It is the single
    point at which the guarantee can be broken, which is why it is explicit
    rather than folded into reservation.
    """
    self.allocator.mark_scrubbed(page_ids)
    scrubbed = set(np.asarray(page_ids, dtype=np.int32).reshape(-1).tolist())
    remaining = [
        pages[~np.isin(pages, list(scrubbed))] for pages in self._pending_scrub
    ]
    self._pending_scrub = [pages for pages in remaining if pages.size]

  def reserve_decode(self, handles: Sequence[RequestHandle]) -> bool:
    return self.reserve(handles, np.ones((len(handles),), dtype=np.int32))

  def release(self, handle: RequestHandle) -> np.ndarray:
    """Reclaim the request's pages and report them.

    The pages are returned so a caller can zero or poison them before they can
    be handed out again. Nothing here does that -- the pool is device memory and
    this layer never touches a device -- which is why the pages have to leave
    through the return value rather than being quietly recycled.
    """
    pages = self.page_map.release(handle)
    if pages.size:
      self.allocator.free(pages)
    return pages

  def build_page_table(
      self,
      handles: Sequence[RequestHandle],
      query_lens: Sequence[int] | np.ndarray,
  ) -> KvPageTableV1:
    """Describe this step, refusing to describe a page that is still dirty.

    The check is over the table's own page list, so it covers exactly the extent
    a kernel is about to read -- no more, since pages trimmed as beyond the
    current length are not readable this step, and no less.
    """
    table = metadata.build_page_table(self.page_map, handles, query_lens)
    unscrubbed = self.allocator.dirty_among(table.flat_page_indices())
    if unscrubbed.size:
      raise DirtyPageError(
          f"pages {unscrubbed[:8].tolist()} still hold a previous request's KV and would be readable "
          f"by this step. Scrub the pages from pending_scrub() and call confirm_scrubbed() before "
          f"building the table."
      )
    return table

  def build_decode_table(self, handles: Sequence[RequestHandle]) -> KvPageTableV1:
    return self.build_page_table(handles, np.ones((len(handles),), dtype=np.int32))
