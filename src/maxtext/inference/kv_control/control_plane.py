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

**Prefix sharing, when enabled, changes which pages a request owns but not how a
step is described.** A shared page is attached to the request's row like any
other, so it appears in the page table and the kernel reads it. What keeps it
read-only is that the request's query length covers only its uncached suffix,
and write positions are derived from that -- so a shared page cannot enter a
`slot_mapping` without the query length being wrong first. Under `debug_mode`
that implication is checked rather than assumed.

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

import dataclasses
from typing import Sequence

import numpy as np

from maxtext.inference.kv_common import CacheNamespace, KvPageTableV1, KvStorageLayoutV1
from maxtext.inference.kv_control import metadata
from maxtext.inference.kv_control.allocator import PagedBlockAllocator
from maxtext.inference.kv_control.logical_block import new_pages_for_extend, pages_for_tokens
from maxtext.inference.kv_control.page_map import PageCapacityError, PageMap
from maxtext.inference.kv_control.prefix_index import PrefixIndex, PrefixMatch, PrefixNode
from maxtext.inference.kv_control.request import RequestDescriptor, RequestHandle

_DEFAULT_NAMESPACE = CacheNamespace()


class DirtyPageError(RuntimeError):
  """A step would have read a page still holding another request's KV."""


class SharedPageWriteError(RuntimeError):
  """A step would have written into a page another request is reading."""


@dataclasses.dataclass
class _PrefixState:
  """What a request borrowed from the prefix index, and under whose identity.

  Held per live request because neither the page map nor the allocator can
  distinguish a borrowed page from an owned one, and releasing a request must
  free exactly the ones it owns.

  Recorded even when the lookup missed, because the namespace is needed again at
  release to publish under. Inferring it then would mean guessing, and a wrong
  guess publishes one configuration's K/V where another will find it.
  """

  namespace: CacheNamespace
  node: PrefixNode | None = None
  shared_pages: np.ndarray = dataclasses.field(default_factory=lambda: np.empty((0,), dtype=np.int32))


class NativeKvControlPlane:
  """Page accounting owned by MaxText, satisfying `KvControlPlane`."""

  def __init__(
      self,
      layout: KvStorageLayoutV1,
      max_requests: int,
      max_context_len: int,
      debug_mode: bool = False,
      enable_prefix_cache: bool = False,
  ):
    if max_context_len < 1:
      raise ValueError(f"max_context_len must be at least 1, got {max_context_len}")

    self._layout = layout
    self.max_context_len = int(max_context_len)
    self.debug_mode = bool(debug_mode)
    self.allocator = PagedBlockAllocator.from_layout(layout, debug_mode=debug_mode)
    self.page_map = PageMap.from_layout(layout, max_requests=max_requests, max_context_len=max_context_len)
    self._pending_scrub: list[np.ndarray] = []
    # Off unless asked for: sharing is only sound when the caller supplies a
    # namespace that genuinely distinguishes its configurations, and defaulting
    # it on would make that someone else's problem to discover.
    self.prefix_index = PrefixIndex(layout.tokens_per_page, enabled=enable_prefix_cache)
    self._prefix_state: dict[RequestHandle, _PrefixState] = {}

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

  @property
  def prefix_cache_enabled(self) -> bool:
    return self.prefix_index.enabled

  def attach_prefix(
      self,
      handle: RequestHandle,
      token_ids: Sequence[int] | np.ndarray,
      namespace: CacheNamespace = _DEFAULT_NAMESPACE,
  ) -> PrefixMatch:
    """Lend the request every already-computed page of its prompt.

    The shared pages are attached to the request's row and its length advanced
    over them, so the request begins as though it had already prefilled that
    much. The caller then prefills only `prompt_len - match.num_tokens` tokens,
    and that difference is the entire benefit.

    Must be called before the request holds any pages, since a prefix is by
    definition the front of the sequence and the page list is positional.
    """
    if self.page_map.num_pages(handle) or self.page_map.seq_len(handle):
      raise ValueError(
          f"request {handle.request_id!r} already holds "
          f"{self.page_map.num_pages(handle)} pages at length {self.page_map.seq_len(handle)}; "
          "a prefix can only be attached to a request that has not yet reserved anything"
      )

    match = self.prefix_index.match(token_ids, namespace)
    # Recorded on a miss too: this is where the caller states which
    # configuration the request belongs to, and release needs it to publish
    # under the same one.
    self._prefix_state[handle] = _PrefixState(namespace=namespace)
    if not match:
      return match

    self.prefix_index.acquire(match.node)
    self.page_map.append_pages(handle, match.pages)
    self.page_map.advance(handle, match.num_tokens)
    self._prefix_state[handle] = _PrefixState(
        namespace=namespace,
        node=match.node,
        shared_pages=match.pages,
    )
    return match

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

    needed = int(sum(per_request_pages))
    pages = self.allocator.alloc(needed)
    if pages is None:
      # Cached pages are the one reclaimable thing left: they hold K/V nothing
      # currently needs, and recomputing them is strictly better than refusing
      # to make progress. Evicting only the shortfall keeps the rest of the
      # cache for whoever asks next.
      reclaimed = self.evict_cached(needed - self.allocator.available_pages)
      if reclaimed.size == 0:
        return False
      pages = self.allocator.alloc(needed)
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

  def evict_cached(self, num_pages: int) -> np.ndarray:
    """Drop `num_pages` unreferenced cached pages and return them to the pool.

    The pages come back dirty, as any freed page does, so a later step that
    reserves them will scrub them through the ordinary path.
    """
    reclaimed = self.prefix_index.evict(num_pages)
    if reclaimed.size:
      self.allocator.free(reclaimed)
    return reclaimed

  def release(
      self,
      handle: RequestHandle,
      token_ids: Sequence[int] | np.ndarray | None = None,
      num_valid_tokens: int | None = None,
      namespace: CacheNamespace | None = None,
  ) -> np.ndarray:
    """Reclaim the request's pages and report the ones actually freed.

    With prefix sharing on and `token_ids` supplied, the request's full pages are
    offered to the index first, and the ones it adopts stay allocated for the
    next request with the same prefix. Those are excluded from the return value,
    which continues to mean "pages that are now free and may be poisoned" -- so a
    caller that poisons what it gets back cannot corrupt a page it just donated.

    Pages borrowed at admission are likewise never freed here: the index owns
    them and this request was only reading them.

    Publishing needs to know which configuration produced the K/V. That comes
    from the `attach_prefix` call that admitted the request, or from `namespace`
    for a caller that never made one. It is never defaulted: publishing under a
    namespace the request did not belong to files its K/V where a different
    configuration will find and trust it.
    """
    state = self._prefix_state.pop(handle, None)
    held = self.page_map.release(handle)

    keep = np.zeros((0,), dtype=np.int32)
    if state is not None:
      if state.node is not None:
        self.prefix_index.release(state.node)
        keep = state.shared_pages
      if namespace is not None and namespace != state.namespace:
        raise ValueError(
            f"request {handle.request_id!r} was admitted under namespace ({state.namespace.describe()}) "
            f"but is being released under a different one ({namespace.describe()})"
        )
      namespace = state.namespace

    if token_ids is not None and self.prefix_index.enabled:
      if namespace is None:
        raise ValueError(
            f"cannot publish request {handle.request_id!r} without a namespace. Either admit it via "
            "attach_prefix, which records one, or pass namespace= here."
        )
      published = self.prefix_index.publish(token_ids, held, namespace, num_valid_tokens)
      if published.adopted.size:
        keep = np.union1d(keep, published.adopted)

    pages = held[~np.isin(held, keep)] if keep.size else held
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
    if self.debug_mode and self._prefix_state:
      self._check_no_shared_writes(table)
    return table

  def _check_no_shared_writes(self, table: KvPageTableV1) -> None:
    """Confirm this step writes into no page it is only borrowing.

    The property already follows from query lengths covering only the uncached
    suffix, so this can never fire against correct arithmetic. It is here
    because the failure it would catch -- one request overwriting the cached
    prefix that several others are reading -- corrupts those requests silently
    and at a distance, and is worth paying a debug-mode set intersection to rule
    out while the arithmetic is still new.
    """
    borrowed = np.unique(np.concatenate([s.shared_pages for s in self._prefix_state.values()]))
    slots = table.slot_mapping(self._layout.tokens_per_page, self.allocator.padding_page_id).reshape(-1)
    written = np.unique(slots[slots >= 0] // self._layout.tokens_per_page).astype(np.int32)
    clash = np.intersect1d(written, borrowed)
    if clash.size:
      raise SharedPageWriteError(
          f"pages {clash[:8].tolist()} are shared prefix pages that other requests are reading, but "
          f"this step's slot_mapping writes into them. A query length is covering tokens that were "
          f"served from the prefix cache."
      )

  def build_decode_table(self, handles: Sequence[RequestHandle]) -> KvPageTableV1:
    return self.build_page_table(handles, np.ones((len(handles),), dtype=np.int32))
