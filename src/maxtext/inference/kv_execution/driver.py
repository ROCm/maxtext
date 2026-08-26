"""Page-based continuous batching, beside the slot-based loop rather than inside it.

`InferenceWorker._run_continuous_batching` is built on a static set of integer
slots: one slot is one fixed reservation, held for a request's whole life, handed
out by `empty_decode_slots.pop()` and returned by the detokenisation thread. That
is the right shape for the dense two-region cache and the wrong shape here, where
a request owns a set of pages that changes size on every step and where admitting
one depends on how much *pool* is free rather than on whether a slot is spare.
Generalising the existing loop to cover both would make it unreviewable and put
the working dense path at risk, so this is a sibling.

**Execution is injected.** The driver owns admission, reservation, scrubbing,
shape selection and release; it does not own the forward pass. A caller supplies
a step function taking the padded `StepView` and the pool and returning one token
per active request. That keeps the scheduling logic testable without a model, and
it is the seam a real engine adapter plugs into.

Two policies are worth naming because they are choices, not consequences:

  * **Prefill is preferred over decode when both are possible.** It favours time
    to first token at the cost of inter-token latency for requests already
    running. A production scheduler would make this configurable; the milestone
    only needs it to be deliberate.
  * **Decode running out of pages preempts the newest request by recomputation.**
    Its pages are released and it returns to the queue to be prefilled again.
    This loses work, but it is the simplest policy that cannot deadlock, and a
    loop that can deadlock under churn fails the stability criterion outright.
    Preempting the newest rather than the oldest keeps the requests closest to
    finishing.

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
from typing import Callable, Iterable, Sequence

import numpy as np

from maxtext.inference.kv_control import (
    NativeKvControlPlane,
    RequestDescriptor,
    RequestHandle,
    RequestState,
    pages_for_tokens,
)
from maxtext.inference.kv_execution.bucketing import StepShape, StepShapePlanner
from maxtext.inference.kv_execution.pool_factory import PagedKvPool
from maxtext.inference.kv_execution.pool_ops import poison_pages, scrub_pages
from maxtext.inference.kv_execution.step_view import StepView, build_step_view


@dataclasses.dataclass(eq=False)
class PagedRequest:
  """A request as the driver tracks it, across preemption and requeueing.

  Compared by identity, not by field values. The driver moves these between the
  waiting, live and done lists by `remove`, and two requests that happen to
  agree on every field -- same length, same tokens generated so far, which is
  entirely possible -- would otherwise be interchangeable to `list.remove` and
  the wrong one would be dropped.
  """

  request_id: str
  prompt_len: int
  max_new_tokens: int
  handle: RequestHandle | None = None
  generated: list[int] = dataclasses.field(default_factory=list)
  state: RequestState = RequestState.WAITING
  finish_reason: str | None = None
  preemptions: int = 0

  @property
  def is_finished(self) -> bool:
    return self.finish_reason is not None

  def descriptor(self) -> RequestDescriptor:
    """The remaining work, not the original request.

    After a preemption the prompt has to be recomputed but the tokens already
    generated are part of the context, so the prompt to replay is longer than
    the one submitted. Describing the original would under-reserve.
    """
    return RequestDescriptor(
        request_id=self.request_id,
        prompt_len=self.prompt_len + len(self.generated),
        max_new_tokens=max(self.max_new_tokens - len(self.generated), 0),
    )


@dataclasses.dataclass(frozen=True)
class StepOutcome:
  """What one executed step did, for a caller that wants to observe the loop."""

  shape: StepShape
  is_decode: bool
  num_requests: int
  num_tokens: int
  preempted: int = 0


# Takes the padded view and the pool; returns one token per *active* request and
# the rebound pool arrays, which alias the ones passed in.
StepFn = Callable[[StepView, PagedKvPool], np.ndarray]


class PagedDriver:
  """Admits, schedules, reserves for and releases paged requests."""

  def __init__(
      self,
      control_plane: NativeKvControlPlane,
      pool: PagedKvPool,
      step_fn: StepFn,
      *,
      max_batch: int,
      max_batched_tokens: int | None = None,
      eos_ids: Iterable[int] = (),
      poison_on_free: bool = False,
  ):
    layout = control_plane.layout
    self.plane = control_plane
    self.pool = pool
    self.step_fn = step_fn
    self.max_batch = int(max_batch)
    self.eos_ids = frozenset(int(t) for t in eos_ids)
    self.poison_on_free = bool(poison_on_free)
    self.planner = StepShapePlanner(
        tokens_per_page=layout.tokens_per_page,
        max_batch=max_batch,
        max_context_len=control_plane.max_context_len,
        pool_pages=layout.num_pages,
        max_batched_tokens=max_batched_tokens,
    )
    self._waiting: list[PagedRequest] = []
    self._live: list[PagedRequest] = []
    self._done: list[PagedRequest] = []
    self.observed_shapes: set[StepShape] = set()

  @property
  def num_distinct_shapes(self) -> int:
    """Distinct traced shapes so far. The direct measure of whether bucketing works."""
    return len(self.observed_shapes)

  @property
  def num_waiting(self) -> int:
    return len(self._waiting)

  @property
  def num_live(self) -> int:
    return len(self._live)

  def submit(self, requests: Sequence[PagedRequest]) -> None:
    self._waiting.extend(requests)

  def run(self, max_steps: int = 100_000) -> list[PagedRequest]:
    """Drive every submitted request to completion.

    `max_steps` is a guard, not a schedule: hitting it means the loop failed to
    make progress, and raising says so rather than returning half an answer.
    """
    steps = 0
    while self._waiting or self._live:
      if steps >= max_steps:
        raise RuntimeError(
            f"the driver made no progress in {max_steps} steps with {len(self._waiting)} waiting and "
            f"{len(self._live)} live requests"
        )
      if self.step() is None:
        break
      steps += 1
    return list(self._done)

  def step(self) -> StepOutcome | None:
    """Run one step. Prefill if anything can be admitted, else decode.

    Returns None when there is nothing left to do.
    """
    admitted = self._admit()
    if admitted:
      return self._run_phase(admitted, [r.descriptor().prompt_len for r in admitted], is_decode=False)
    if self._live:
      return self._run_phase(self._live, [1] * len(self._live), is_decode=True)
    return None

  def _admit(self) -> list[PagedRequest]:
    """Take as many waiting requests as rows, pages and the token bucket allow.

    Budgeted against `available_pages` before anything is admitted, because
    reservation is all-or-nothing: admitting a request the pool cannot back would
    fail the whole batch rather than just that request.
    """
    if not self._waiting:
      return []

    tokens_per_page = self.plane.layout.tokens_per_page
    token_budget = self.planner.token_rungs[-1]
    page_budget = self.plane.allocator.available_pages
    room = self.max_batch - len(self._live)

    admitted: list[PagedRequest] = []
    tokens = 0
    for request in list(self._waiting):
      if len(admitted) >= room or self.plane.page_map.available_rows == 0:
        break
      prompt_len = request.descriptor().prompt_len
      needed = pages_for_tokens(prompt_len, tokens_per_page)
      if needed > page_budget or tokens + prompt_len > token_budget:
        break
      handle = self.plane.admit(request.descriptor())
      if handle is None:
        break
      request.handle = handle
      request.state = RequestState.PREFILL
      page_budget -= needed
      tokens += prompt_len
      admitted.append(request)
      self._waiting.remove(request)
    return admitted

  def _run_phase(
      self,
      requests: Sequence[PagedRequest],
      query_lens: Sequence[int],
      *,
      is_decode: bool,
  ) -> StepOutcome:
    """Reserve, scrub, build, execute, and retire -- in that order.

    The order is not incidental. Scrubbing has to happen after reservation,
    since that is what decides which pages were recycled, and before the table is
    built, because the control plane refuses to describe a page it still
    considers dirty.
    """
    batch = list(requests)
    lens = list(query_lens)
    preempted = 0

    while batch and not self.plane.reserve([r.handle for r in batch], lens):
      if not is_decode:
        # A prefill batch was budgeted against the free pool, so a failure here
        # means the budget was wrong rather than that the pool is under pressure.
        raise RuntimeError(
            f"reservation failed for an admitted prefill batch of {len(batch)} requests; "
            f"{self.plane.available_tokens} tokens available"
        )
      self._preempt_newest(batch)
      preempted += 1
      lens = [1] * len(batch)

    if not batch:
      return StepOutcome(
          shape=StepShape(0, 0, 0, 0, is_decode),
          is_decode=is_decode,
          num_requests=0,
          num_tokens=0,
          preempted=preempted,
      )

    self._scrub_recycled_pages()

    handles = [r.handle for r in batch]
    table = self.plane.build_page_table(handles, lens)
    max_seq_len = int(table.seq_lens.max()) if table.num_requests else 1
    shape = (
        self.planner.decode_shape(len(batch), max_seq_len)
        if is_decode
        else self.planner.extend_shape(sum(lens), max_seq_len)
    )
    self.observed_shapes.add(shape)

    view = build_step_view(
        table,
        shape,
        tokens_per_page=self.plane.layout.tokens_per_page,
        padding_page_id=self.plane.layout.padding_page_id,
    )
    next_tokens = np.asarray(self.step_fn(view, self.pool)).reshape(-1)
    if next_tokens.size < len(batch):
      raise ValueError(f"the step function returned {next_tokens.size} tokens for {len(batch)} requests")

    self._retire(batch, next_tokens[: len(batch)], is_decode=is_decode)
    return StepOutcome(
        shape=shape,
        is_decode=is_decode,
        num_requests=len(batch),
        num_tokens=sum(lens),
        preempted=preempted,
    )

  def _scrub_recycled_pages(self) -> None:
    """Zero every page this step recycled, then say so.

    Both halves matter. Without the write a later read sees the previous
    occupant's KV; without the confirmation the control plane refuses to build
    the table, which is the mechanism that stops the write being forgotten.
    """
    pending = self.plane.pending_scrub()
    if not pending.size:
      return
    for layer in range(self.pool.num_layers):
      k, v = scrub_pages(self.pool.k_pages[layer], self.pool.v_pages[layer], pending)
      self.pool.replace_layer(layer, k, v)
    self.plane.confirm_scrubbed(pending)

  def _preempt_newest(self, batch: list[PagedRequest]) -> None:
    """Return the newest request's pages to the pool and requeue it.

    Its generated tokens are kept, so the replay is a longer prompt rather than
    lost output. What is lost is the compute that built its KV, which is the
    price of not deadlocking.
    """
    if not batch:
      return
    victim = batch.pop()
    self._release_pages(victim)
    victim.state = RequestState.WAITING
    victim.preemptions += 1
    if victim in self._live:
      self._live.remove(victim)
    self._waiting.insert(0, victim)

  def _release_pages(self, request: PagedRequest) -> None:
    """Give up a request's pages, poisoning them first when asked.

    Poison is applied *before* the pages leave the request, and does not count as
    a scrub: the control plane keeps them dirty, so the next occupant still has
    to zero them. That ordering is what makes the sentinel a detector of a missed
    scrub rather than a substitute for one.
    """
    if request.handle is None:
      return
    pages = self.plane.page_map.pages(request.handle)
    if self.poison_on_free and pages.size:
      for layer in range(self.pool.num_layers):
        k, v = poison_pages(self.pool.k_pages[layer], self.pool.v_pages[layer], pages)
        self.pool.replace_layer(layer, k, v)
    self.plane.release(request.handle)
    request.handle = None

  def _retire(self, batch: Sequence[PagedRequest], tokens: np.ndarray, *, is_decode: bool) -> None:
    """Record this step's tokens and release whatever finished.

    Releasing is deferred to a second pass because the stop conditions read the
    request's recorded length, which only exists while it still holds a handle.
    """
    finished: list[PagedRequest] = []
    for request, token in zip(batch, tokens.tolist()):
      request.generated.append(int(token))
      request.state = RequestState.DECODE
      if int(token) in self.eos_ids:
        request.finish_reason = "stop"
      elif len(request.generated) >= request.max_new_tokens:
        request.finish_reason = "length"
      elif self.plane.page_map.seq_len(request.handle) >= self.plane.max_context_len:
        request.finish_reason = "context"

      if request.is_finished:
        finished.append(request)
      elif not is_decode:
        self._live.append(request)

    for request in finished:
      request.state = RequestState.FINISHED
      self._release_pages(request)
      if request in self._live:
        self._live.remove(request)
      self._done.append(request)
