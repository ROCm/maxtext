"""Page-granular free list for the KV pool.

Two design choices carried over from the reference implementation, and one
deliberate divergence.

Carried over: the two-tier free list. `free()` does not return pages to the
allocation list but to a staging area, and the two are merged and re-sorted only
when an allocation would otherwise fail. The sort then happens once, at the
moment it actually buys something, and allocation pops from the front of the
sorted list so a request's pages stay as clustered as the pool's history allows.

The staging area is a list of arrays rather than one array grown by
concatenation. Concatenating on each free would make the free path cost
proportional to everything already staged, so a workload that frees steadily
while the front list is still long -- a large pool serving short requests, which
is not an exotic case -- pays quadratically in the pool size between merges. That
is the cost the two tiers exist to avoid, so it would be an unfortunate place to
reintroduce it.

Also carried over: one page id is reserved and never allocated. It is a landing
zone -- padded gather entries default to it and read harmlessly -- which is why
the pool must be zero-initialised rather than merely allocated. It buys nothing
against stale KV in a recycled *real* page; that is a separate obligation
discharged by writing a page's full readable extent before marking it readable.

The divergence: the reference allocates in token indices, because its
request-to-token table is token-granular. Ours is page-granular and carries
explicit write positions, so the reference's three-part extend fill -- continue
the open page, then whole pages, then a partial page -- falls out of
`position // tokens_per_page` rather than needing to be computed. What remains
is the page *count*, which is `new_pages_for_extend`. `free_token_slots` exists
for callers that only have slots to hand.

Two things the reference does not do, and this does.

It diagnoses a double free. The reference set-differences the free lists, which
makes a double free idempotent rather than reported, so a page freed while
another request still holds it is invisible. Here an allocation bitmap separates
the two cases -- the same page appearing twice in one call is just token-index
deduplication and is fine, while freeing a page that is not currently allocated
is a use-after-free and raises.

And it tracks which pages are dirty. A freed page still holds the previous
occupant's KV until something overwrites it, so handing it to a new request
without scrubbing it means the new request's readable extent covers bytes it
does not own. The reference has no notion of this at all. A page is dirty from
the moment it is freed until a caller confirms it has been scrubbed; since the
pool is zero-initialised, a page that has never been allocated starts clean, so
a fresh pool costs no scrubbing at all.

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

from maxtext.inference.kv_common import KvStorageLayoutV1


class DoubleFreeError(RuntimeError):
  """A page was freed that the allocator does not consider allocated."""


class PagedBlockAllocator:
  """Hands out pool pages and takes them back.

  Holds no refcounts. Sharing a page between requests is prefix-index policy,
  and keeping it out of here is what lets the free-list mechanics be reasoned
  about -- and tested -- on their own.
  """

  def __init__(
      self,
      num_pages: int,
      tokens_per_page: int,
      padding_page_id: int = 0,
      debug_mode: bool = False,
  ):
    if num_pages < 1:
      raise ValueError(f"num_pages must be at least 1, got {num_pages}")
    if tokens_per_page < 1:
      raise ValueError(f"tokens_per_page must be at least 1, got {tokens_per_page}")
    if not 0 <= padding_page_id < num_pages:
      raise ValueError(f"padding_page_id {padding_page_id} is outside the pool's {num_pages} pages")

    self.num_pages = int(num_pages)
    self.tokens_per_page = int(tokens_per_page)
    self.padding_page_id = int(padding_page_id)
    self.debug_mode = bool(debug_mode)

    all_pages = np.arange(self.num_pages, dtype=np.int32)
    self._allocatable = all_pages[all_pages != self.padding_page_id]
    self._free_pages = self._allocatable.copy()
    self._released: list[np.ndarray] = []
    self._num_released = 0
    # int64 so a long-running server cannot wrap an epoch and make a stale
    # reference look current again.
    self._epochs = np.zeros((self.num_pages,), dtype=np.int64)
    self._allocated = np.zeros((self.num_pages,), dtype=bool)
    # Clean at startup because the pool is zero-initialised. That is a real
    # dependency, not an assumption: `pool_factory` must allocate zeros, and the
    # reserved padding page relies on the same thing.
    self._dirty = np.zeros((self.num_pages,), dtype=bool)

  @classmethod
  def from_layout(cls, layout: KvStorageLayoutV1, debug_mode: bool = False) -> "PagedBlockAllocator":
    """Build from pool geometry, which is where these numbers are decided."""
    return cls(
        num_pages=layout.num_pages,
        tokens_per_page=layout.tokens_per_page,
        padding_page_id=layout.padding_page_id,
        debug_mode=debug_mode,
    )

  @property
  def capacity_pages(self) -> int:
    """Allocatable pages, excluding the reserved padding page."""
    return int(self._allocatable.size)

  @property
  def num_free_pages(self) -> int:
    """Pages available without a merge. Use `available_pages` for the real figure."""
    return int(self._free_pages.size)

  @property
  def available_pages(self) -> int:
    """Pages an allocation could reach, counting the staging area."""
    return int(self._free_pages.size) + self._num_released

  @property
  def available_tokens(self) -> int:
    return self.available_pages * self.tokens_per_page

  @property
  def num_allocated_pages(self) -> int:
    return int(np.count_nonzero(self._allocated))

  def merge_released(self) -> None:
    """Fold the staging area back into the allocation list, sorted.

    Public because a driver may want to pay this cost at a quiet moment rather
    than in the middle of a step that is about to fail.
    """
    if self._released:
      self._free_pages = np.sort(np.concatenate([self._free_pages, *self._released]))
      self._released = []
      self._num_released = 0

  def alloc(self, num_pages: int) -> np.ndarray | None:
    """Take `num_pages` pages, or return None if the pool cannot supply them.

    None rather than an exception: exhaustion is a scheduling outcome the driver
    must branch on every step, not an error condition.
    """
    if num_pages < 0:
      raise ValueError(f"num_pages must be non-negative, got {num_pages}")
    if num_pages == 0:
      return np.empty((0,), dtype=np.int32)

    if num_pages > self._free_pages.size:
      self.merge_released()
    if num_pages > self._free_pages.size:
      return None

    pages = self._free_pages[:num_pages].copy()
    self._free_pages = self._free_pages[num_pages:]
    self._epochs[pages] += 1
    self._allocated[pages] = True
    if self.debug_mode:
      self._check_invariants()
    return pages

  def free(self, page_ids: Sequence[int] | np.ndarray) -> np.ndarray:
    """Return pages to the staging list.

    Returns the deduplicated pages actually released, which is what a caller
    needs in order to poison or zero them before they can be handed out again.
    """
    pages = np.unique(np.asarray(page_ids, dtype=np.int32))
    if pages.size == 0:
      return pages

    if pages[0] < 0 or pages[-1] >= self.num_pages:
      raise ValueError(f"page ids {pages[0]}..{pages[-1]} fall outside the pool's {self.num_pages} pages")
    if np.any(pages == self.padding_page_id):
      raise ValueError(
          f"page {self.padding_page_id} is the reserved padding page and is never allocated, "
          f"so it cannot be freed"
      )

    unowned = pages[~self._allocated[pages]]
    if unowned.size:
      raise DoubleFreeError(
          f"pages {unowned.tolist()} are not currently allocated. Either they were freed twice, "
          f"or a stale handle is still naming pages that were reclaimed."
      )

    self._allocated[pages] = False
    self._dirty[pages] = True
    self._released.append(pages)
    self._num_released += int(pages.size)
    if self.debug_mode:
      self._check_invariants()
    return pages

  def free_token_slots(self, token_slots: Sequence[int] | np.ndarray) -> np.ndarray:
    """Free the pages covering `token_slots`.

    Negative slots are dropped rather than rejected: a padded row's slot is -1
    by construction, so a caller passing a `slot_mapping` straight through is
    behaving correctly, not making a mistake.
    """
    slots = np.asarray(token_slots, dtype=np.int64)
    slots = slots[slots >= 0]
    if slots.size == 0:
      return np.empty((0,), dtype=np.int32)
    return self.free(np.unique(slots // self.tokens_per_page).astype(np.int32))

  def dirty_among(self, page_ids: Sequence[int] | np.ndarray) -> np.ndarray:
    """The subset of `page_ids` still holding a previous occupant's KV.

    The caller must overwrite these before any kernel is allowed to read them,
    and say so via `mark_scrubbed`. Returning the subset rather than a flag is
    what lets a step scrub only the pages it actually recycled: on a fresh pool
    that is none of them.
    """
    pages = np.asarray(page_ids, dtype=np.int32).reshape(-1)
    if pages.size == 0:
      return pages
    return pages[self._dirty[pages]]

  def is_dirty(self, page_id: int) -> bool:
    return bool(self._dirty[page_id])

  @property
  def num_dirty_pages(self) -> int:
    return int(np.count_nonzero(self._dirty))

  def mark_scrubbed(self, page_ids: Sequence[int] | np.ndarray) -> None:
    """Record that `page_ids` have been overwritten and are safe to read.

    Only the caller can know this, because the pool is device memory and this
    layer never touches a device. Calling it without having done the write is
    the one way to defeat the guarantee, which is why it is a separate step
    rather than a side effect of allocation.
    """
    pages = np.asarray(page_ids, dtype=np.int32).reshape(-1)
    if pages.size == 0:
      return
    if pages.min() < 0 or pages.max() >= self.num_pages:
      raise ValueError(f"page ids {pages.min()}..{pages.max()} fall outside the pool's {self.num_pages} pages")
    self._dirty[pages] = False

  def epoch_of(self, page_id: int) -> int:
    """Current epoch of `page_id`, incremented on each allocation."""
    return int(self._epochs[page_id])

  def is_allocated(self, page_id: int) -> bool:
    return bool(self._allocated[page_id])

  def holds(self, page_id: int, epoch: int) -> bool:
    """Whether a reference taken at `epoch` still names a live allocation.

    False both for a page that has been freed and for one already handed to a
    later request, which are the two ways a stale reference goes wrong.
    """
    return bool(self._allocated[page_id]) and int(self._epochs[page_id]) == epoch

  def clear(self) -> None:
    """Return every page to the free list and invalidate all outstanding references.

    Epochs advance rather than resetting, so a reference taken before the clear
    cannot be mistaken for a current one afterwards. Pages that had been handed
    out stay dirty: dropping the free list does not overwrite anything.
    """
    self._free_pages = self._allocatable.copy()
    self._released = []
    self._num_released = 0
    self._dirty |= self._allocated
    self._allocated[:] = False
    self._epochs += 1

  def _check_invariants(self) -> None:
    """Assert the free lists and the allocation bitmap still agree."""
    free_size = int(self._free_pages.size)
    combined = np.concatenate([self._free_pages, *self._released])
    if np.unique(combined).size != free_size + self._num_released:
      raise AssertionError("a page appears more than once across the allocation and staging lists")
    if combined.size and np.any(self._allocated[combined]):
      raise AssertionError("a page is marked allocated while sitting in a free list")
    if self.num_allocated_pages + free_size + self._num_released != self.capacity_pages:
      raise AssertionError(
          f"page accounting lost pages: {self.num_allocated_pages} allocated + {free_size} free + "
          f"{self._num_released} staged != {self.capacity_pages} allocatable"
      )
