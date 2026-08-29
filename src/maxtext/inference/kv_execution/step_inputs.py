"""What a step feeds the model: tokens, positions, and where to sample.

`StepView` describes *pages*. This describes *tokens*. Both are needed for a
step, and keeping them apart is deliberate -- the page bookkeeping comes from the
control plane and knows nothing about token ids, while these arrays come from the
requests and know nothing about pages.

**One position rule, and it is the reason this module exists.** A slice's
``tokens`` occupy absolute positions ``start .. start + len(tokens) - 1``. Callers
differ only in where they start:

  * prefill: ``start = cached_tokens``, feeding the context the cache did not
    supply
  * decode:  ``start = prompt_len + len(generated) - 1``, feeding the single token
    the previous step produced

Decode is not a special case in the rule, only in the arithmetic that fills it.

Getting positions wrong is the failure this module is built to prevent, and it is
not a crash. RoPE encodes absolute position and is not translation invariant, so
a suffix rotated as though it began the sequence produces K/V that does not
belong after the prefix it follows -- plausible text from a wrong computation. The
two ways to reach that are a prefix-cache hit, where the query starts at
``cached_tokens``, and a replay after preemption, where the retained generated
tokens make the prompt longer than the one submitted.

**Numpy, not `jnp`, and that is not an oversight.** Building these with `jnp`
compiles a fresh program per prompt length, because `arange(n) < prompt_len`
bakes the length into the jaxpr as a literal. An earlier measurement in this
project ended up three-quarters compile time that way while reporting zero
unwarmed shapes. These arrays are a few hundred bytes; numpy has no cache to miss.

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

from maxtext.inference.kv_execution.bucketing import StepShape

# Matches `maxtext.common.common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR`.
# Duplicated as a literal rather than imported because that module pulls in jax,
# and this one is deliberately host-only so it stays testable with no accelerator.
_ACTIVE_SEGMENT = 1


@dataclasses.dataclass(frozen=True)
class RequestSlice:
  """One request's contribution to a step: what to feed, and where it sits.

  `tokens` is *exactly* what this step feeds -- not the whole context with an
  index into it. An earlier shape took the full context plus a start offset, and
  it forced `generate_paged` to fabricate a zero-filled array as long as the
  sequence just to make a single-token decode indexable. That is wasteful on a
  per-step path and it invents context that does not exist.

  `query_len` is carried anyway, redundantly with `len(tokens)`, because they come
  from different places: `query_len` is what the page table reserved and
  `len(tokens)` is what the caller actually assembled. Requiring them to agree is
  what catches a caller feeding fewer tokens than it reserved positions for, which
  would otherwise leave the pool holding K/V for positions no query covered.

  Deliberately *not* `PagedRequest`. `MaxEngine.prefill_paged` has a handle, a
  padded prompt and a true length; it has no `PagedRequest` and should not, since
  that is the driver's scheduling record. If assembly demanded one, the engine
  entry points could not share it and input assembly would exist twice -- which
  is the thing this design is for.
  """

  tokens: np.ndarray
  start: int
  query_len: int


@dataclasses.dataclass(frozen=True)
class StepInputs:
  """The token-side operands of one step, padded to the bucketed shape.

  Shapes differ by phase, inherited from the attention layer rather than chosen
  here: prefill packs requests along the sequence axis at batch 1, decode batches
  them along the batch axis.

  ============  ====================  ===================
  field         prefill               decode
  ============  ====================  ===================
  tokens        ``[1, width]``        ``[num_requests, 1]``
  positions     ``[1, width]``        ``[num_requests, 1]``
  segment_ids   ``[1, width]``        ``None``
  sample_rows   all zero              ``arange(n)``
  sample_at     last index per req    all zero
  ============  ====================  ===================

  `sample_rows` is carried explicitly so the sampling gather is one expression,
  ``logits[sample_rows, sample_at]``, in both phases. That is also what makes
  batched prefill possible: with a single packed row, several requests differ in
  their sample *position* rather than their sample row.
  """

  tokens: np.ndarray
  positions: np.ndarray
  segment_ids: np.ndarray | None
  sample_rows: np.ndarray
  sample_at: np.ndarray


def _validate(slices: Sequence[RequestSlice], shape: StepShape, is_decode: bool) -> None:
  """Fail on the four ways a caller can present an impossible step."""
  if not slices:
    raise ValueError("a step needs at least one request slice")

  for index, item in enumerate(slices):
    if item.query_len < 1:
      raise ValueError(f"slice {index} has query_len {item.query_len}; a step must run at least one token")
    if item.start < 0:
      raise ValueError(f"slice {index} has start {item.start}; positions are absolute and cannot be negative")
    supplied = int(np.asarray(item.tokens).reshape(-1).size)
    if supplied != item.query_len:
      # The page table reserved `query_len` positions. Feeding a different number
      # would leave the pool holding K/V for positions no query covered, or run
      # tokens the table has nowhere to put. Loud beats plausible.
      raise ValueError(
          f"slice {index} reserved {item.query_len} positions but supplied {supplied} tokens; "
          f"these must agree or the pool and the query disagree about what this step ran"
      )

  if is_decode:
    if any(item.query_len != 1 for item in slices):
      raise ValueError("a decode step runs exactly one token per request")
    if len(slices) > shape.num_requests:
      raise ValueError(
          f"{len(slices)} requests do not fit the decode batch bucket of {shape.num_requests}"
      )
  else:
    total = sum(item.query_len for item in slices)
    if total > shape.num_tokens:
      raise ValueError(
          f"{total} query tokens do not fit the token bucket of {shape.num_tokens}"
      )


def build_step_inputs(
    slices: Sequence[RequestSlice],
    shape: StepShape,
    *,
    is_decode: bool,
) -> StepInputs:
  """Assemble one step's token-side operands.

  Takes a `StepShape` rather than a `StepView` on purpose. Everything needed here
  is a host int -- the token bucket and the batch bucket -- and accepting the view
  would put its `jax.Array` fields within reach, which is how host arithmetic
  quietly acquires a device dependency. Sample indices come from cumulating the
  slices' own `query_len`, not from the view's `cu_seqlens_q`.

  Args:
    slices: one per active request, in the same order the page table was built.
    shape: the bucketed shape this step is traced for.
    is_decode: selects the layout, since the two phases pack differently.

  Returns:
    Arrays padded to `shape`, with every padded element inert.
  """
  _validate(slices, shape, is_decode)

  if is_decode:
    width = shape.num_requests
    tokens = np.zeros((width, 1), np.int32)
    positions = np.zeros((width, 1), np.int32)
    for row, item in enumerate(slices):
      tokens[row, 0] = int(np.asarray(item.tokens).reshape(-1)[0])
      positions[row, 0] = int(item.start)
    # One row each, so the sample row varies and the position does not. Padded
    # rows sample position 0 of their own row, which holds a zero token; their
    # slot mapping points at the reserved padding page, so nothing they compute
    # is read.
    return StepInputs(
        tokens=tokens,
        positions=positions,
        # None, not zeros: the model refuses segment ids in autoregressive mode,
        # where every token is by definition in the active sequence.
        segment_ids=None,
        sample_rows=np.arange(width, dtype=np.int32),
        sample_at=np.zeros((width,), np.int32),
    )

  width = shape.num_tokens
  tokens = np.zeros((1, width), np.int32)
  positions = np.zeros((1, width), np.int32)
  segment_ids = np.zeros((1, width), np.int32)
  sample_at = np.zeros((len(slices),), np.int32)

  cursor = 0
  for index, item in enumerate(slices):
    end = cursor + item.query_len
    tokens[0, cursor:end] = np.asarray(item.tokens).reshape(-1)
    # Absolute, so a cached prefix or a preemption replay lands where RoPE
    # expects rather than at zero.
    positions[0, cursor:end] = np.arange(item.start, item.start + item.query_len, dtype=np.int32)
    segment_ids[0, cursor:end] = _ACTIVE_SEGMENT
    # The logits this step produces cover only the tokens it ran, so the sample
    # index is within the packed row rather than within the sequence.
    sample_at[index] = end - 1
    cursor = end

  return StepInputs(
      tokens=tokens,
      positions=positions,
      segment_ids=segment_ids,
      # One packed row, so every request samples from row zero and they differ
      # only in position. This is precisely what a `rows = arange(batch)` gather
      # cannot express, and why batched prefill returned a single token before.
      sample_rows=np.zeros((len(slices),), np.int32),
      sample_at=sample_at,
  )
