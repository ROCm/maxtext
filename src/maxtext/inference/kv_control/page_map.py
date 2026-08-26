"""Which pages each live request holds.

A dense `[max_requests, max_pages_per_request]` int32 table plus a free list of
rows, following the reference design, with two changes.

The reference's table is token-granular, `[max_requests, max_context_len]`, and
recovers a request's pages by reading its row and filtering zeros. This one is
page-granular, so it is `tokens_per_page` times smaller -- 256 requests at 32k
context is 33 MB there and 2 MB here at 16 tokens per page -- and it tracks each
row's length explicitly instead of filtering a sentinel. Explicit lengths also
mean the table stays correct if the reserved page id is ever something other
than zero, which sentinel filtering silently would not.

The second change is the epoch. Rows are recycled, so a caller holding a handle
to a released request would otherwise read whichever request now occupies the
row. Every lookup checks the handle's epoch against the row's, which converts
that from a wrong answer into an exception.

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
from maxtext.inference.kv_control.logical_block import pages_for_tokens
from maxtext.inference.kv_control.request import RequestDescriptor, RequestHandle, RequestState


class StaleRequestHandleError(RuntimeError):
  """A handle named a request that has been released, and its row reused."""


class PageCapacityError(RuntimeError):
  """A request needs more pages than its row can record."""


class PageMap:
  """Per-request page bookkeeping, in dense arrays.

  Owns sequence lengths as well as page ids, because the two are inseparable: a
  page list without the length is not enough to say how much of the final page a
  kernel may read, and that bound is the whole defence against reading a
  recycled page's tail.
  """

  def __init__(
      self,
      max_requests: int,
      max_pages_per_request: int,
      tokens_per_page: int,
      padding_page_id: int = 0,
  ):
    if max_requests < 1:
      raise ValueError(f"max_requests must be at least 1, got {max_requests}")
    if max_pages_per_request < 1:
      raise ValueError(f"max_pages_per_request must be at least 1, got {max_pages_per_request}")
    if tokens_per_page < 1:
      raise ValueError(f"tokens_per_page must be at least 1, got {tokens_per_page}")

    self.max_requests = int(max_requests)
    self.max_pages_per_request = int(max_pages_per_request)
    self.tokens_per_page = int(tokens_per_page)
    self.padding_page_id = int(padding_page_id)

    self._pages = np.full((self.max_requests, self.max_pages_per_request), self.padding_page_id, dtype=np.int32)
    self._num_pages = np.zeros((self.max_requests,), dtype=np.int32)
    self._seq_lens = np.zeros((self.max_requests,), dtype=np.int32)
    self._epochs = np.zeros((self.max_requests,), dtype=np.int64)
    self._states: list[RequestState | None] = [None] * self.max_requests
    self._descriptors: list[RequestDescriptor | None] = [None] * self.max_requests
    self._free_rows: list[int] = list(range(self.max_requests))

  @classmethod
  def from_layout(cls, layout: KvStorageLayoutV1, max_requests: int, max_context_len: int) -> "PageMap":
    """Size the table from pool geometry and the longest context to be served."""
    return cls(
        max_requests=max_requests,
        max_pages_per_request=pages_for_tokens(max_context_len, layout.tokens_per_page),
        tokens_per_page=layout.tokens_per_page,
        padding_page_id=layout.padding_page_id,
    )

  @property
  def num_live(self) -> int:
    return self.max_requests - len(self._free_rows)

  @property
  def available_rows(self) -> int:
    return len(self._free_rows)

  def admit(self, descriptor: RequestDescriptor) -> RequestHandle | None:
    """Take a row for `descriptor`, or return None if every row is occupied.

    None rather than an exception, for the same reason page exhaustion returns
    None: a full batch is a scheduling outcome, not a fault.
    """
    if not self._free_rows:
      return None
    row = self._free_rows.pop(0)
    self._num_pages[row] = 0
    self._seq_lens[row] = 0
    self._pages[row, :] = self.padding_page_id
    self._states[row] = RequestState.WAITING
    self._descriptors[row] = descriptor
    return RequestHandle(request_id=descriptor.request_id, row=row, epoch=int(self._epochs[row]))

  def release(self, handle: RequestHandle) -> np.ndarray:
    """Give up the row and report the pages it held.

    The pages are returned rather than freed here: this class does not own the
    free list, and separating "no longer referenced" from "available again" is
    what leaves room to zero or poison a page in between.
    """
    row = self._row(handle)
    held = self._pages[row, : self._num_pages[row]].copy()
    self._num_pages[row] = 0
    self._seq_lens[row] = 0
    self._pages[row, :] = self.padding_page_id
    self._states[row] = None
    self._descriptors[row] = None
    self._epochs[row] += 1
    self._free_rows.append(row)
    return held

  def pages(self, handle: RequestHandle) -> np.ndarray:
    """The request's pages in sequence order, as a copy."""
    row = self._row(handle)
    return self._pages[row, : self._num_pages[row]].copy()

  def num_pages(self, handle: RequestHandle) -> int:
    return int(self._num_pages[self._row(handle)])

  def append_pages(self, handle: RequestHandle, page_ids: Sequence[int] | np.ndarray) -> None:
    """Extend the request's page list, in sequence order.

    Order is load-bearing: sequence position `p` is held by page slot
    `p // tokens_per_page`, so appending out of order silently misdirects every
    subsequent read and write.
    """
    row = self._row(handle)
    pages = np.asarray(page_ids, dtype=np.int32).reshape(-1)
    if pages.size == 0:
      return
    start = int(self._num_pages[row])
    end = start + pages.size
    if end > self.max_pages_per_request:
      raise PageCapacityError(
          f"request {handle.request_id!r} would hold {end} pages but a row records at most "
          f"{self.max_pages_per_request}; raise max_context_len or lower the request's length cap"
      )
    self._pages[row, start:end] = pages
    self._num_pages[row] = end

  def seq_len(self, handle: RequestHandle) -> int:
    return int(self._seq_lens[self._row(handle)])

  def advance(self, handle: RequestHandle, num_tokens: int) -> int:
    """Grow the recorded context by `num_tokens` and return the new length.

    Refuses to advance past what the held pages can address, because that is the
    precise condition under which a kernel would read beyond the request's own
    data.
    """
    if num_tokens < 0:
      raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    row = self._row(handle)
    new_len = int(self._seq_lens[row]) + num_tokens
    addressable = int(self._num_pages[row]) * self.tokens_per_page
    if new_len > addressable:
      raise PageCapacityError(
          f"request {handle.request_id!r} would reach length {new_len} but its "
          f"{int(self._num_pages[row])} pages address only {addressable} tokens: reserve pages "
          f"before advancing the length, never after"
      )
    self._seq_lens[row] = new_len
    return new_len

  def state(self, handle: RequestHandle) -> RequestState:
    return self._states[self._row(handle)]

  def set_state(self, handle: RequestHandle, state: RequestState) -> None:
    self._states[self._row(handle)] = state

  def descriptor(self, handle: RequestHandle) -> RequestDescriptor:
    return self._descriptors[self._row(handle)]

  def live_handles(self) -> list[RequestHandle]:
    """Handles for every occupied row, in row order."""
    occupied = set(range(self.max_requests)) - set(self._free_rows)
    return [
        RequestHandle(
            request_id=self._descriptors[row].request_id,
            row=row,
            epoch=int(self._epochs[row]),
        )
        for row in sorted(occupied)
    ]

  def _row(self, handle: RequestHandle) -> int:
    """Validate `handle` and return its row."""
    row = handle.row
    if not 0 <= row < self.max_requests:
      raise StaleRequestHandleError(f"handle row {row} is outside the table's {self.max_requests} rows")
    if self._descriptors[row] is None:
      raise StaleRequestHandleError(
          f"request {handle.request_id!r} has been released; its row {row} is free"
      )
    if int(self._epochs[row]) != handle.epoch:
      raise StaleRequestHandleError(
          f"request {handle.request_id!r} holds epoch {handle.epoch} but row {row} is at epoch "
          f"{int(self._epochs[row])}: the row has been reused by another request"
      )
    return row
