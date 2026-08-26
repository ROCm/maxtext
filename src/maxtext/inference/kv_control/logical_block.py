"""Block identity, page states, and the page arithmetic everything else shares.

The arithmetic here is small enough to look not worth centralising, and that is
exactly why it is centralised: `ceil(seq / page) - ceil(prefix / page)` inlined
in four places is four places to get an off-by-one wrong, and the symptom of
getting it wrong is a kernel reading a page the request does not own.

One page state matters more than the others. A page is readable only once its
full readable extent has been written, because a recycled page still holds the
previous occupant's KV until something overwrites it. `WRITING` versus `READY`
is where that obligation is expressed, rather than being implicit in whether an
append happened to have run yet.

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
import enum


class PageState(enum.Enum):
  """Whether a page may be allocated, written, or read.

  `WRITING` covers both a fresh allocation and a partial last page being
  extended, since in both cases some of the page's readable extent is not yet
  valid for this request. A prefill-decode destination page awaiting a transfer
  is the same condition with a different filler, and will reuse this state
  rather than adding a parallel one.
  """

  FREE = "free"
  WRITING = "writing"
  READY = "ready"


_LEGAL_TRANSITIONS = {
    PageState.FREE: frozenset({PageState.WRITING}),
    # WRITING -> FREE is a request cancelled or preempted mid-prefill.
    PageState.WRITING: frozenset({PageState.READY, PageState.FREE}),
    # READY -> WRITING is a partial last page being extended by the next step.
    PageState.READY: frozenset({PageState.WRITING, PageState.FREE}),
}


class PageStateError(RuntimeError):
  """An illegal page state transition, which is always a control-plane bug."""


def check_transition(current: PageState, target: PageState) -> None:
  """Raise unless `current -> target` is legal.

  Self-transitions are allowed: re-marking a page WRITING as more tokens arrive
  is the normal extend path, not a mistake.
  """
  if current is target:
    return
  if target not in _LEGAL_TRANSITIONS[current]:
    raise PageStateError(f"illegal page state transition {current.value} -> {target.value}")


@dataclasses.dataclass
class LogicalBlock:
  """One page's worth of a request's context.

  Attributes:
    page_id: the physical page backing this block.
    epoch: the allocator's epoch for `page_id` at the moment it was allocated.
      Comparing it against the allocator's current epoch is what turns a
      use-after-free into an error instead of a plausible read of someone
      else's KV.
    num_tokens: tokens of this request written into the page so far. Never the
      page's capacity unless the page is genuinely full; an over-stated value is
      precisely how a kernel reads a recycled page's tail.
    state: see `PageState`.
    ref_count: holders of this block. Stays at 1 for the whole of M4, because
      only one request can own a page until prefix sharing exists; the prefix
      index is what raises it, and it lives outside the allocator on purpose so
      that sharing policy and free-list mechanics stay separable.
  """

  page_id: int
  epoch: int
  num_tokens: int = 0
  state: PageState = PageState.WRITING
  ref_count: int = 1

  def set_state(self, target: PageState) -> None:
    check_transition(self.state, target)
    self.state = target

  @property
  def is_readable(self) -> bool:
    return self.state is PageState.READY


def pages_for_tokens(num_tokens: int, tokens_per_page: int) -> int:
  """Pages needed to hold `num_tokens` contiguous tokens."""
  if num_tokens < 0:
    raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
  return -(-num_tokens // tokens_per_page)


def new_pages_for_extend(prefix_len: int, seq_len: int, tokens_per_page: int) -> int:
  """Pages a request must acquire to grow from `prefix_len` to `seq_len`.

  The already-open last page is continued in place and costs nothing, which is
  why this is a difference of two ceilings rather than a ceiling of the
  difference. Those disagree exactly when the extend starts mid-page, which is
  the common case.
  """
  if seq_len < prefix_len:
    raise ValueError(f"seq_len {seq_len} is shorter than prefix_len {prefix_len}")
  return pages_for_tokens(seq_len, tokens_per_page) - pages_for_tokens(prefix_len, tokens_per_page)


def decode_needs_new_page(seq_len_after: int, tokens_per_page: int) -> bool:
  """Whether appending the token that brings the context to `seq_len_after` crosses a page boundary."""
  if seq_len_after <= 0:
    raise ValueError(f"seq_len_after must be positive, got {seq_len_after}")
  return (seq_len_after - 1) % tokens_per_page == 0


def last_page_occupancy(seq_len: int, tokens_per_page: int) -> int:
  """Tokens occupying the final page of a `seq_len`-token context.

  A full final page reports the page size, not zero.
  """
  if seq_len < 0:
    raise ValueError(f"seq_len must be non-negative, got {seq_len}")
  if seq_len == 0:
    return 0
  remainder = seq_len % tokens_per_page
  return remainder if remainder else tokens_per_page


def token_slot(page_id: int, position: int, tokens_per_page: int) -> int:
  """Absolute pool slot of the token at sequence `position` held in `page_id`.

  Slots are `page_id * tokens_per_page + offset`, and downstream code depends on
  that contiguity: it is what lets an append kernel scatter with a single index
  array and what lets a page transfer address a page as one byte range.
  """
  return page_id * tokens_per_page + position % tokens_per_page
