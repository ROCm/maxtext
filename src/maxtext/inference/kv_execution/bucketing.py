"""Power-of-two shape ladders, so a churning batch traces a fixed set of shapes.

This is the JAX-specific obligation of the whole design. Every array crossing
into a step is data-dependent in *size*: how many requests are live, how many
tokens they contribute, how many pages they hold. Left alone, a mixed-length
workload under churn presents a new shape almost every step and recompiles
forever, which does not merely cost time -- it makes the steady-state latency the
milestone is supposed to demonstrate unmeasurable.

Two shape families, because the two phases vary along different axes:

  * **Decode** contributes exactly one token per request, so the token count is
    not free -- it *is* the batch bucket. Only the batch size varies.
  * **Extend** varies in both request count and total tokens, and independently.
    Bucketing both would multiply out, so the batch axis is pinned to its largest
    bucket and only the token count varies.

**One refinement on the plan's three ladders.** The gather table gets no ladder of
its own; it is derived as `num_requests * ceil(max_seqlen_k / tokens_per_page)`,
clamped to the pool. That is a correct upper bound -- no request can hold more
pages than its own capped length needs, and no batch more than the pool has --
and deriving it removes an entire dimension from the compile cross-product rather
than adding one. What does need a ladder, and is easy to miss, is `max_seqlen_k`:
the kernels take it as a static configuration value, so passing the true maximum
would retrace on nearly every step no matter how well the array shapes were
bucketed.

Deliberately free of `jax`. Which shapes exist is host arithmetic, and keeping it
that way means the ladders can be reasoned about and tested without a device.

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

# The floor on the token ladder, matching what MaxText's existing prefill
# bucketing uses (`2**i for i in range(6, ...)`). Below 64 tokens a separate
# trace buys nothing.
MIN_TOKEN_BUCKET = 1 << 6


def bucket_up(value: int, ladder: Sequence[int]) -> int:
  """Smallest ladder entry at least `value`.

  Raises rather than clamping when `value` exceeds the ladder: silently
  bucketing 5000 tokens down to a 4096-token shape would truncate the batch, and
  a truncated batch loses tokens instead of running slowly.
  """
  for step in ladder:
    if value <= step:
      return step
  raise ValueError(f"{value} exceeds the largest bucket {ladder[-1]}; the ladder is mis-sized for this workload")


def _powers_of_two(low: int, high: int) -> tuple[int, ...]:
  """Powers of two from `low` up to the first one at or above `high`."""
  rungs, step = [], low
  while step < high:
    rungs.append(step)
    step *= 2
  rungs.append(step)
  return tuple(rungs)


def batch_ladder(max_batch: int) -> tuple[int, ...]:
  """1, 2, 4, ... up to `max_batch`."""
  if max_batch < 1:
    raise ValueError(f"max_batch must be at least 1, got {max_batch}")
  return _powers_of_two(1, max_batch)


def token_ladder(max_tokens: int) -> tuple[int, ...]:
  """64, 128, ... up to `max_tokens`, or a single rung if that is below 64."""
  if max_tokens < 1:
    raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")
  if max_tokens <= MIN_TOKEN_BUCKET:
    return (MIN_TOKEN_BUCKET,)
  return _powers_of_two(MIN_TOKEN_BUCKET, max_tokens)


def seqlen_ladder(tokens_per_page: int, max_context_len: int) -> tuple[int, ...]:
  """Page-aligned powers of two up to `max_context_len`.

  Starts at the page size because a shorter context still occupies one whole
  page, so a finer rung would describe a shape that cannot occur.
  """
  if max_context_len < 1:
    raise ValueError(f"max_context_len must be at least 1, got {max_context_len}")
  return _powers_of_two(tokens_per_page, max(max_context_len, tokens_per_page))


@dataclasses.dataclass(frozen=True)
class StepShape:
  """The padded shape family one step is traced for.

  Hashable, so a driver can count distinct shapes -- which is the direct measure
  of whether bucketing is working, and what the milestone's exit criterion is
  stated in terms of.
  """

  num_requests: int
  num_tokens: int
  num_pages: int
  max_seqlen_k: int
  is_decode: bool


class StepShapePlanner:
  """Owns the ladders and maps a live batch onto a `StepShape`."""

  def __init__(
      self,
      tokens_per_page: int,
      max_batch: int,
      max_context_len: int,
      pool_pages: int,
      max_batched_tokens: int | None = None,
  ):
    """
    Args:
      max_batched_tokens: token budget for one extend step, and so the top of
        the token ladder. Distinct from `max_context_len`, which bounds a single
        request: an extend step batches several requests, so its total can
        exceed the longest one. Sizing the ladder from `max_context_len` instead
        makes a perfectly legal batch unbucketable. Defaults to
        `max_context_len`, which admits one full-length request per step.
    """
    self.tokens_per_page = int(tokens_per_page)
    self.pool_pages = int(pool_pages)
    self.max_batched_tokens = int(max_batched_tokens or max_context_len)
    self.batch_rungs = batch_ladder(max_batch)
    self.token_rungs = token_ladder(self.max_batched_tokens)
    self.seqlen_rungs = seqlen_ladder(tokens_per_page, max_context_len)

  @property
  def max_batch_bucket(self) -> int:
    return self.batch_rungs[-1]

  def _pages_for(self, num_requests: int, max_seqlen_k: int) -> int:
    """Upper bound on pages the batch can reference, clamped to the pool."""
    per_request = -(-max_seqlen_k // self.tokens_per_page)
    return min(num_requests * per_request, self.pool_pages)

  def decode_shape(self, num_requests: int, max_seq_len: int) -> StepShape:
    """One token per request, so only the batch axis varies."""
    requests = bucket_up(max(num_requests, 1), self.batch_rungs)
    seqlen = bucket_up(max(max_seq_len, 1), self.seqlen_rungs)
    return StepShape(
        num_requests=requests,
        num_tokens=requests,
        num_pages=self._pages_for(requests, seqlen),
        max_seqlen_k=seqlen,
        is_decode=True,
    )

  def extend_shape(self, num_tokens: int, max_seq_len: int, num_requests: int | None = None) -> StepShape:
    """Batch pinned to its largest bucket; only the token count varies.

    Args:
      num_requests: pass a count to bucket the batch axis instead of pinning it.
        Pinning exists to stop the request count and the token count multiplying
        out into a cross-product of shapes, so it buys nothing for a caller whose
        request count is fixed -- `MaxEngine.prefill_paged` always prefills one
        prompt — and there it only pads the per-request arrays to the maximum
        batch for no reason.
    """
    requests = self.max_batch_bucket if num_requests is None else bucket_up(max(num_requests, 1), self.batch_rungs)
    tokens = bucket_up(max(num_tokens, 1), self.token_rungs)
    seqlen = bucket_up(max(max_seq_len, 1), self.seqlen_rungs)
    return StepShape(
        num_requests=requests,
        num_tokens=tokens,
        num_pages=self._pages_for(requests, seqlen),
        max_seqlen_k=seqlen,
        is_decode=False,
    )

  def max_distinct_shapes(self) -> int:
    """Upper bound on shapes this planner can ever produce.

    Worth being able to state, because "bounded" is only meaningful with a
    number attached. Decode contributes batch times seqlen rungs, extend
    contributes token times seqlen rungs.
    """
    seqlens = len(self.seqlen_rungs)
    return len(self.batch_rungs) * seqlens + len(self.token_rungs) * seqlens
