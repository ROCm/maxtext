"""Which tier each pool page currently lives in.

The counterpart to the allocator's dirty bitmap, and deliberately a separate
object rather than another array on `PagedBlockAllocator`. Dirtiness is a
property of the free list -- a page becomes dirty by being freed -- so it
belongs where free and alloc live. Residency is a property of *tiering*, which
the allocator knows nothing about and should not learn: a page can be offloaded
while still very much allocated, and can be resident while free.

**The obligation this exists to enforce is that offloaded is not resident.** It
is the same shape of hole the dirty bitmap closes, and it fails the same way --
silently. A page whose bytes are in host memory still has device bytes; they are
simply stale or scrubbed. A kernel reading it gets plausible numbers rather than
an error, and the request produces fluent wrong output. So `build_page_table`
refuses to describe a non-resident page, exactly as it refuses a dirty one, and
that refusal is what makes M7's "no execution before readiness" structural
rather than a convention the driver is trusted to follow.

The three states are worth stating apart because the middle one is what a
concurrent design needs and what a synchronous one can ignore:

* `DEVICE`   -- bytes are in the pool and readable.
* `EVICTING` -- a transfer is in flight; the device bytes may be mid-copy.
* `HOST`     -- bytes are in the host slab; the device page is not readable.

`EVICTING` is not readable either. It exists so that a future asynchronous
offload has a state to occupy between "started" and "finished" without either
lying about readability or blocking, and so that a page cannot be handed back
out while a copy is still reading it.

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

import enum
from typing import Sequence

import numpy as np


class Tier(enum.IntEnum):
  """Stored as small ints so the whole table is one numpy array."""

  DEVICE = 0
  EVICTING = 1
  HOST = 2


class NotResidentError(RuntimeError):
  """A page table named a page whose bytes are not on the device."""


class PageResidency:
  """Per-page tier, with the same shape and lifetime as the pool."""

  def __init__(self, num_pages: int) -> None:
    if num_pages < 1:
      raise ValueError(f"num_pages must be at least 1, got {num_pages}")
    self.num_pages = int(num_pages)
    # Everything starts on device: the pool is allocated there, and a page that
    # has never been offloaded is trivially resident.
    self._tier = np.full((self.num_pages,), Tier.DEVICE, dtype=np.int8)

  def _check_range(self, pages: np.ndarray) -> None:
    if pages.size and (pages.min() < 0 or pages.max() >= self.num_pages):
      raise ValueError(
          f"page ids must lie in [0, {self.num_pages}); got "
          f"min {int(pages.min())} max {int(pages.max())}"
      )

  def _as_pages(self, page_ids: Sequence[int] | np.ndarray) -> np.ndarray:
    pages = np.asarray(page_ids, dtype=np.int64).reshape(-1)
    self._check_range(pages)
    return pages

  def begin_evict(self, page_ids: Sequence[int] | np.ndarray) -> None:
    """Mark pages as being copied out. They stop being readable immediately.

    Immediately, rather than on completion, because the point of the state is to
    close the window in which a step could read a page whose copy has started.
    """
    pages = self._as_pages(page_ids)
    resident = pages[self._tier[pages] == Tier.DEVICE]
    self._tier[resident] = Tier.EVICTING

  def confirm_evicted(self, page_ids: Sequence[int] | np.ndarray) -> None:
    """Record that the copy landed and the device bytes are now expendable."""
    pages = self._as_pages(page_ids)
    self._tier[pages] = Tier.HOST

  def confirm_resident(self, page_ids: Sequence[int] | np.ndarray) -> None:
    """Record that pages have been written back into the pool."""
    pages = self._as_pages(page_ids)
    self._tier[pages] = Tier.DEVICE

  def non_resident_among(self, page_ids: Sequence[int] | np.ndarray) -> np.ndarray:
    """Which of these pages a kernel must not read. The enforcement primitive.

    Shaped like `PagedBlockAllocator.dirty_among` on purpose: the control plane
    asks both the same question about the same page list, and a reader comparing
    the two guards should not have to hold two idioms in mind.
    """
    pages = self._as_pages(page_ids)
    if pages.size == 0:
      return np.empty((0,), dtype=np.int32)
    offending = np.unique(pages[self._tier[pages] != Tier.DEVICE])
    return offending.astype(np.int32)

  def tier_of(self, page_id: int) -> Tier:
    return Tier(int(self._tier[int(page_id)]))

  def pages_in(self, tier: Tier) -> np.ndarray:
    return np.flatnonzero(self._tier == tier).astype(np.int32)

  @property
  def num_offloaded(self) -> int:
    """Pages whose bytes are in the host slab. The capacity extension itself."""
    return int(np.count_nonzero(self._tier == Tier.HOST))

  @property
  def num_evicting(self) -> int:
    return int(np.count_nonzero(self._tier == Tier.EVICTING))

  def clear(self) -> None:
    self._tier[:] = Tier.DEVICE
