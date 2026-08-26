"""Wiring between `MaxEngine` and the paged control plane.

`MaxEngine.release_pages(slot)` was a no-op that printed a warning, with a
docstring referring to a `PageManager` that exists nowhere in the file, and three
call sites that all pass a fixed slot integer and ignore the result. It is
tempting to just implement it. That would be a mistake, because the signature
encodes the dense cache's assumption that a request *is* a slot -- one fixed
reservation for its whole life -- and building the new runtime around that would
bake in the abstraction the paging work exists to replace.

So `release(handle)` is the real API and `release_pages(slot)` becomes a shim over
it. The shim needs a slot-to-handle map, which this adapter keeps; that map is the
entire cost of keeping the three legacy call sites working, and it is confined to
one object that a paged deployment can eventually stop constructing.

`MaxEngine.prefill` already accepts a `request_id` it never forwards, which is the
natural anchor for a handle and is why this adapter keys on request id as well as
slot.

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

from maxtext.inference.kv_control import NativeKvControlPlane, RequestDescriptor, RequestHandle
from maxtext.inference.kv_execution.bucketing import StepShape, StepShapePlanner
from maxtext.inference.kv_execution.pool_factory import PagedKvPool
from maxtext.inference.kv_execution.pool_ops import poison_pages, scrub_pages
from maxtext.inference.kv_execution.step_view import StepView, build_step_view


class PagedRuntime:
  """The paged control plane and pool, as one attachable object.

  `MaxEngine` holds at most one of these and delegates to it. Keeping the two
  together matters because releasing pages and poisoning them are the same
  operation from a caller's point of view, and poisoning needs the pool.

  It also owns `prepare_step`, which is the fixed per-step order — reserve,
  scrub, confirm, build, pad — expressed once. `MaxEngine` should not be
  reimplementing that sequence, and neither should a second driver: getting the
  order wrong is how a kernel ends up reading a page it does not own.
  """

  def __init__(
      self,
      control_plane: NativeKvControlPlane,
      pool: PagedKvPool,
      planner: StepShapePlanner | None = None,
      poison_on_free: bool = False,
  ):
    self.control_plane = control_plane
    self.pool = pool
    self.planner = planner
    self.poison_on_free = bool(poison_on_free)
    # Every bucketed shape this runtime has ever asked for. One compiled program
    # per entry, so this is the compile count -- the Section 6.4 measurement, and
    # the only way to tell a warmup that covered the shape space from one that
    # merely ran for a while.
    self.observed_shapes: set[StepShape] = set()
    self._by_slot: dict[int, RequestHandle] = {}
    self._by_request_id: dict[str, RequestHandle] = {}
    self._cached_tokens: dict[str, int] = {}

  def admit(self, request_id: str, prompt_len: int, max_new_tokens: int) -> RequestHandle | None:
    """Start tracking a request, returning None when there is no room."""
    return self.control_plane.admit(
        RequestDescriptor(request_id=request_id, prompt_len=prompt_len, max_new_tokens=max_new_tokens)
    )

  def attach_prefix(self, handle: RequestHandle, token_ids, namespace=None) -> int:
    """Lend the request any cached pages of its prompt. Returns tokens skipped.

    The caller must then shorten the step's query to the tokens that remain, and
    offset their positions by what was skipped -- the suffix sits at absolute
    positions `cached..prompt_len`, and RoPE is not translation invariant.
    """
    if not self.control_plane.prefix_cache_enabled or token_ids is None:
      return 0
    if namespace is None:
      cached = self.control_plane.attach_prefix(handle, token_ids).num_tokens
    else:
      cached = self.control_plane.attach_prefix(handle, token_ids, namespace).num_tokens
    self._cached_tokens[handle.request_id] = cached
    return cached

  def cached_tokens(self, handle: RequestHandle) -> int:
    """Tokens this request did not have to prefill. Zero without a hit."""
    return self._cached_tokens.get(handle.request_id, 0)

  def prepare_step(
      self,
      handles: Sequence[RequestHandle],
      query_lens: Sequence[int],
      *,
      is_decode: bool,
      num_requests: int | None = None,
  ) -> StepView | None:
    """Reserve pages for this step and return the padded device arrays.

    Returns None if the pool cannot back the step, which is backpressure rather
    than an error. The scrub sits between reservation and table construction
    because reservation is what decides which pages were recycled, and the
    control plane refuses to describe a page it still considers dirty.
    """
    if self.planner is None:
      raise ValueError("this runtime has no StepShapePlanner, so it cannot decide a bucketed shape")
    if not self.control_plane.reserve(handles, query_lens):
      return None

    self.scrub_recycled()

    table = self.control_plane.build_page_table(handles, query_lens)
    max_seq_len = int(table.seq_lens.max()) if table.num_requests else 1
    shape = (
        self.planner.decode_shape(len(handles), max_seq_len)
        if is_decode
        else self.planner.extend_shape(int(sum(query_lens)), max_seq_len, num_requests=num_requests)
    )
    self.observed_shapes.add(shape)
    layout = self.control_plane.layout
    return build_step_view(
        table, shape, tokens_per_page=layout.tokens_per_page, padding_page_id=layout.padding_page_id
    )

  def scrub_recycled(self) -> np.ndarray:
    """Zero every page this step recycled, then record that it was done.

    Both halves are necessary: without the write a later read sees the previous
    occupant's KV, and without the confirmation the control plane refuses to
    build the table -- which is the mechanism that stops the write being skipped.
    """
    pending = self.control_plane.pending_scrub()
    if not pending.size:
      return pending
    for layer in range(self.pool.num_layers):
      k, v = scrub_pages(self.pool.k_pages[layer], self.pool.v_pages[layer], pending)
      self.pool.replace_layer(layer, k, v)
    self.control_plane.confirm_scrubbed(pending)
    return pending

  def track(self, handle: RequestHandle, slot: int | None = None) -> None:
    """Record a handle so it can be found again by request id, or by slot.

    The slot argument exists only for the legacy API. A caller that has a handle
    should keep it and pass it to `release` directly.
    """
    self._by_request_id[handle.request_id] = handle
    if slot is not None:
      self._by_slot[int(slot)] = handle

  def handle_for_slot(self, slot: int) -> RequestHandle | None:
    return self._by_slot.get(int(slot))

  def handle_for_request(self, request_id: str) -> RequestHandle | None:
    return self._by_request_id.get(str(request_id))

  def release(self, handle: RequestHandle, token_ids=None) -> np.ndarray:
    """Reclaim everything the request holds. The canonical API.

    Idempotent at this level: an untracked handle is a no-op rather than an
    error, because the three legacy call sites fire on sequence termination and
    do not coordinate with each other. The control plane underneath is *not*
    idempotent, and that is the right split -- a double release through a handle
    the adapter still knows about is a bug worth reporting, while a release of
    something already forgotten is just a duplicate notification.

    `token_ids` is the full context, offered to the prefix cache. The recorded
    sequence length bounds what may be published, since that is exactly the
    extent whose K/V has been written; the final sampled token is in the list but
    its own K/V never got computed.
    """
    known = self._by_request_id.get(handle.request_id)
    if known is None or known != handle:
      return np.empty((0,), dtype=np.int32)
    valid = self.control_plane.page_map.seq_len(handle)
    freed = self.control_plane.release(handle, token_ids, num_valid_tokens=valid)
    # Poisoning what came free rather than what the request held: with sharing on
    # they differ, and a page the cache adopted is about to be read as valid K/V.
    if self.poison_on_free and freed.size:
      for layer in range(self.pool.num_layers):
        k, v = poison_pages(self.pool.k_pages[layer], self.pool.v_pages[layer], freed)
        self.pool.replace_layer(layer, k, v)
    self._forget(handle)
    return freed

  def release_slot(self, slot: int) -> np.ndarray:
    """Shim for `release_pages(slot)`."""
    handle = self._by_slot.get(int(slot))
    if handle is None:
      return np.empty((0,), dtype=np.int32)
    return self.release(handle)

  def _forget(self, handle: RequestHandle) -> None:
    self._by_request_id.pop(handle.request_id, None)
    self._cached_tokens.pop(handle.request_id, None)
    for slot, tracked in list(self._by_slot.items()):
      if tracked == handle:
        del self._by_slot[slot]
