"""The control-plane contract a driver programs against.

This exists because there will be more than one implementation, and only one may
be authoritative at a time. MaxText's own allocator and page map are one; a
frontend that owns its own page accounting -- vLLM's cache manager, SGLang's
allocator and radix cache -- is another, reached through an adapter. Running two
page owners for the same request means two free lists, two refcounts, and two
disagreeing views of which pages a request holds, so which one is live has to be
a decision asserted at startup rather than something inferred from whichever
object a caller happened to be handed.

Runtime-checkable so that assertion can be a one-line check. It verifies method
presence only, which is enough to catch an adapter that has drifted from the
contract and not enough to be mistaken for type checking.

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

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from maxtext.inference.kv_common import KvPageTableV1, KvStorageLayoutV1
from maxtext.inference.kv_control.request import RequestDescriptor, RequestHandle


@runtime_checkable
class KvControlPlane(Protocol):
  """Admission, page reservation, release, and per-step metadata."""

  @property
  def layout(self) -> KvStorageLayoutV1:
    """Pool geometry this control plane is accounting for."""

  @property
  def available_tokens(self) -> int:
    """Tokens the pool could still accept. The number backpressure is decided on."""

  def admit(self, descriptor: RequestDescriptor) -> RequestHandle | None:
    """Start tracking a request, or return None if there is no room to track it."""

  def reserve(
      self,
      handles: Sequence[RequestHandle],
      num_new_tokens: Sequence[int] | np.ndarray,
  ) -> bool:
    """Reserve pages for the batch's new tokens, all of them or none.

    All-or-nothing because a partial reservation leaves the batch in a state the
    caller has no way to describe: some requests advanced, some not, and no
    page table that covers both.
    """

  def reserve_decode(self, handles: Sequence[RequestHandle]) -> bool:
    """Reserve one token per request."""

  def pending_scrub(self) -> np.ndarray:
    """Pages reserved this step that must be overwritten before any kernel reads them.

    Part of the contract rather than an implementation detail: a frontend that
    owns its own page accounting still has to answer this, or the recycled-page
    guarantee is only as good as whichever allocator happens to be live.
    """

  def confirm_scrubbed(self, page_ids: Sequence[int] | np.ndarray) -> None:
    """Record that `page_ids` have been overwritten on the device."""

  def release(self, handle: RequestHandle) -> np.ndarray:
    """Give up everything the request holds and report the pages reclaimed.

    Request-based, not slot-based. A slot is the dense cache's unit -- one fixed
    reservation per request -- whereas a paged request owns a set of pages that
    changed size on every step, so the handle is the only durable name for it.
    """

  def build_page_table(
      self,
      handles: Sequence[RequestHandle],
      query_lens: Sequence[int] | np.ndarray,
  ) -> KvPageTableV1:
    """Describe this step in the neutral vocabulary."""
