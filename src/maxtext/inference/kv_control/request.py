"""Request identity for the paged control plane.

A dense KV cache identifies a request by the slot it occupies, because one slot
is one fixed-size reservation held for the request's whole life. A paged request
owns a varying set of pages instead, so its identity has to survive that set
growing and has to remain distinguishable from a later request that reuses the
same bookkeeping row. That is the whole reason `RequestHandle` carries an epoch:
without one, a caller holding a released handle reads whichever request now
occupies the row, silently and with plausible-looking results.

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


class RequestState(enum.Enum):
  """Lifecycle of an admitted request, as the control plane sees it.

  Deliberately coarser than a scheduler's own state machine. These are only the
  distinctions that change what the control plane may do with the request's
  pages, which is why there is no queued/running/preempted split here.
  """

  WAITING = "waiting"
  PREFILL = "prefill"
  DECODE = "decode"
  FINISHED = "finished"


@dataclasses.dataclass(frozen=True)
class RequestHandle:
  """A reference to one live request's page bookkeeping.

  Frozen, so it is hashable and can key driver-side state, and so a caller
  cannot mutate the row or epoch it was handed.

  Attributes:
    request_id: caller-facing identity. Opaque here: the control plane never
      interprets it, and two handles with the same id but different epochs are
      different requests.
    row: dense index into the page map's tables. What makes lookup an array
      index rather than a dict probe.
    epoch: bumped every time `row` is handed out again. A handle whose epoch no
      longer matches the row's is naming a request that has already been
      released, and the page map refuses it rather than answering about whoever
      holds the row now.
  """

  request_id: str
  row: int
  epoch: int


@dataclasses.dataclass(frozen=True)
class RequestDescriptor:
  """What the control plane needs in order to size a request's page demand.

  Lengths only. Prompt token ids are deliberately absent: nothing in M4 reads
  them, and admitting them now would force a representation choice -- hashable
  tuple, numpy array, shared buffer -- on behalf of prefix sharing, which is the
  consumer that will actually have an opinion. Copying every prompt into a tuple
  purely to keep this dataclass frozen is a real per-request cost for no present
  benefit.

  Attributes:
    request_id: caller-facing identity, carried through to the handle.
    prompt_len: tokens in the prompt, and so the length of the prefill.
    max_new_tokens: upper bound on generated tokens. Only a bound: a request
      that stops early simply releases its pages sooner.
  """

  request_id: str
  prompt_len: int
  max_new_tokens: int

  def __post_init__(self):
    if self.prompt_len < 0:
      raise ValueError(f"prompt_len must be non-negative, got {self.prompt_len}")
    if self.max_new_tokens < 0:
      raise ValueError(f"max_new_tokens must be non-negative, got {self.max_new_tokens}")

  @property
  def max_total_len(self) -> int:
    """Longest context this request can reach, and so its worst-case page count."""
    return self.prompt_len + self.max_new_tokens
