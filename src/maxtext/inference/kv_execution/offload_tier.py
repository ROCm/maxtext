"""The host tier: swap a request's KV out to host memory and back.

This is the piece that turns `page_staging`'s copy into actual capacity, and the
reason it exists as its own object is that **copying is not the part that buys
anything**. Offloading a page's contents while the page stays allocated extends
nothing; the allocator is still holding it. Capacity comes from *freeing* the
pages after the copy lands, which means a request does not come back to the
pages it left from -- the allocator will have handed those to somebody else. So a
tier has to re-point the page map on the way back in, and that is what makes this
more than a wrapper around a gather.

**The M9 seam is this class's method surface, not its implementation.** `offload`
and `reload` are the two operations a driver needs, and everything host-specific
is confined to the two `page_staging` calls inside them. Swapping the backend for
`jax.experimental.transfer` at M8 or MORI at M9 replaces those two lines and
leaves the ordering, the residency transitions, the capacity accounting and the
page-map re-pointing untouched. That ordering is the part worth getting right
once.

**Why this is a registry and not the contiguous slab the plan asked for.** A
pre-allocated host slab written at computed slots would need a scatter whose
destination operand is host-resident, and XLA rejects that at compile time
(`CompileTimeHostOffloadOutputLocationMismatch`) rather than inserting the copy
itself. Bringing the slab to device to scatter into it would defeat the entire
point. So each offload owns exactly the host arrays it needs, and the bound is
enforced by counting pages rather than by managing slots. That loses nothing --
there are no slots to fragment -- and it removes a free list.

**Two divisions of labour worth stating, because both are easy to get wrong.**

`PageResidency` guards the *eviction window* only. Between `begin_evict` and the
free, the pages hold data that is mid-copy and must not be read, and residency is
what refuses them. Once the pages are freed they return to the allocator's
regime, where the dirty bitmap already refuses them -- so residency is reset
rather than left saying `HOST`, which would make the page unreadable for its next
owner too.

And a reload *discharges* the dirty obligation on its destination pages rather
than paying it. The freed pages a request comes back into are dirty, but the
scatter overwrites every byte of every one of them, which is exactly what a scrub
would have done. Scrubbing first and then overwriting would write each page
twice, on the critical path of every resume.

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

from typing import Any

import numpy as np

from maxtext.inference.kv_execution.page_staging import (
    StagedPages,
    offload_pages,
    reload_pages,
)


class TierCapacityError(RuntimeError):
  """The host tier cannot hold this many more pages."""


class HostOffloadTier:
  """Holds swapped-out requests in host memory, bounded by a page budget.

  Bounded in *pages* rather than bytes because that is the unit the pool, the
  allocator and the capacity claim all already speak, so a deployment can state
  the host tier as a multiple of the device pool and have the comparison mean
  something.
  """

  def __init__(self, capacity_pages: int, pool_sharding: Any | None = None) -> None:
    if capacity_pages < 0:
      raise ValueError(f"capacity_pages must not be negative, got {capacity_pages}")
    self.capacity_pages = int(capacity_pages)
    self.pool_sharding = pool_sharding
    self._staged: dict[Any, StagedPages] = {}
    self._pages_held = 0

  # ------------------------------------------------------------- accounting

  @property
  def pages_held(self) -> int:
    return self._pages_held

  @property
  def pages_free(self) -> int:
    return self.capacity_pages - self._pages_held

  @property
  def num_swapped(self) -> int:
    return len(self._staged)

  def bytes_held(self) -> int:
    """Host memory actually occupied, padding included.

    Padding included because it is really allocated -- a three-page offload
    holds four pages of host memory -- and a capacity figure that ignored it
    would overstate what the tier can take.
    """
    return sum(s.nbytes() for s in self._staged.values())

  def is_swapped(self, handle: Any) -> bool:
    return handle in self._staged

  def swapped_handles(self) -> tuple:
    return tuple(self._staged)

  # -------------------------------------------------------------- movement

  def offload(self, plane: Any, pool: Any, handle: Any) -> bool:
    """Copy a request's KV to host and free its pages. Returns False if full.

    False rather than raising, because a full tier is ordinary backpressure: the
    caller's next move is to preempt by recomputation instead, which is what it
    did before this tier existed.
    """
    if handle in self._staged:
      raise ValueError(f"request {handle!r} is already swapped out")

    pages = plane.page_map.pages(handle)
    if pages.size == 0:
      return False
    if pages.size > self.pages_free:
      return False

    # Unreadable from here on, so a step cannot observe a page mid-copy. Set
    # before the gather rather than after it, because the window is the hazard.
    plane.residency.begin_evict(pages)
    staged = offload_pages(pool, pages, self.pool_sharding)
    plane.residency.confirm_evicted(pages)

    # The line that actually extends capacity. Everything above it only made a
    # copy; the allocator is what was holding the memory.
    plane.allocator.free(pages)
    # Back under the allocator's regime, where the dirty bitmap refuses them.
    # Leaving them marked HOST would refuse them for their next owner too.
    plane.residency.confirm_resident(pages)

    self._staged[handle] = staged
    self._pages_held += int(pages.size)
    return True

  def reload(self, plane: Any, pool: Any, handle: Any) -> bool:
    """Bring a request's KV back into the pool. Returns False if the pool is full.

    The pages are new, so the page map is re-pointed. The caller must not assume
    a request resumes on the pages it left.
    """
    staged = self._staged.get(handle)
    if staged is None:
      raise KeyError(f"request {handle!r} is not swapped out")

    pages = plane.allocator.alloc(staged.num_live)
    if pages is None:
      return False

    reload_pages(pool, staged, self.pool_sharding, dest_page_ids=pages)

    # The scatter wrote every byte of every destination page, which is precisely
    # what a scrub would have done -- so the obligation is discharged, not paid.
    # Scrubbing first would write each page twice on every resume.
    plane.allocator.mark_scrubbed(pages)
    plane.page_map.repoint_pages(handle, pages)
    plane.residency.confirm_resident(pages)

    del self._staged[handle]
    self._pages_held -= int(staged.num_live)
    return True

  def discard(self, handle: Any) -> int:
    """Drop a swapped request's host copy without reloading it.

    For a request that was cancelled while swapped out. Returns the pages
    reclaimed, so a caller can check its own accounting against the tier's.
    """
    staged = self._staged.pop(handle, None)
    if staged is None:
      return 0
    self._pages_held -= int(staged.num_live)
    return int(staged.num_live)

  def clear(self) -> None:
    self._staged.clear()
    self._pages_held = 0
