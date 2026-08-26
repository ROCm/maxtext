"""A `KvPageTableV1` as the padded device arrays a step actually consumes.

The padding convention is the whole content of this module, and it is chosen so
that a padded row is *inert* rather than merely ignored, because "ignored"
depends on every kernel agreeing to ignore it:

  * `slot_mapping` pads with -1, which the append kernel drops. Not with a real
    slot, which would write a padded row's garbage into a live page.
  * `kv_indptr` and `cu_seqlens_q` repeat their final value, so every padded
    request has a zero-length page range and a zero-length query. A kernel
    iterating the full bucketed batch does no work for them without needing a
    mask.
  * `kv_page_indices` pads with the reserved page, and `seq_lens` and
    `kv_last_page_lens` pad with zero. The reserved page reads as zeros, so even
    a kernel that addressed a padded entry despite the flat indptr would read
    zeros rather than another request's KV.

Every one of those is a second line of defence behind the flat indptr. That is
deliberate: padding bugs are silent, and the failure mode is a wrong token rather
than a crash.

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
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

from maxtext.inference.kv_common import KvPageTableV1
from maxtext.inference.kv_execution.bucketing import StepShape


@dataclasses.dataclass(frozen=True)
class StepView:
  """One step's page bookkeeping, padded to a bucketed shape.

  The active counts are carried alongside because they are what a caller needs
  to slice a result back down; they are host ints, not traced, so using one in a
  shape would defeat the bucketing.
  """

  slot_mapping: jax.Array  # int32 [num_tokens]
  kv_indptr: jax.Array  # int32 [num_requests + 1]
  kv_page_indices: jax.Array  # int32 [num_pages]
  kv_last_page_lens: jax.Array  # int32 [num_requests]
  cu_seqlens_q: jax.Array  # int32 [num_requests + 1]
  seq_lens: jax.Array  # int32 [num_requests]
  shape: StepShape
  num_active_requests: int
  num_active_tokens: int
  max_seqlen_q: int

  def to_paged_plan(self) -> Any:
    """Adapt to the M3 attention path's `PagedPlan`.

    Imported lazily so the control and execution layers stay usable without
    pulling in the attention module, which reaches a vendor backend.
    """
    from maxtext.layers.gpu_paged_attention import PagedPlan  # pylint: disable=import-outside-toplevel

    return PagedPlan(
        slot_mapping=self.slot_mapping,
        kv_indptr=self.kv_indptr,
        kv_page_indices=self.kv_page_indices,
        kv_last_page_lens=self.kv_last_page_lens,
        cu_seqlens_q=self.cu_seqlens_q,
        max_seqlen_q=self.max_seqlen_q,
        max_seqlen_k=self.shape.max_seqlen_k,
        is_decode=self.shape.is_decode,
    )


def _pad_to(values: np.ndarray, size: int, fill: int, name: str) -> np.ndarray:
  if values.size > size:
    raise ValueError(f"{name} has {values.size} entries, past the bucketed {size}")
  if values.size == size:
    return values
  return np.concatenate([values, np.full((size - values.size,), fill, dtype=np.int32)])


def _pad_cumulative(values: np.ndarray, size: int, name: str) -> np.ndarray:
  """Extend a prefix-sum array by repeating its last entry.

  Repetition is what makes a padded request zero-length rather than
  out-of-range, which is the difference between a kernel skipping it and a
  kernel reading whatever sits at index zero.
  """
  if values.size > size:
    raise ValueError(f"{name} has {values.size} entries, past the bucketed {size}")
  if values.size == size:
    return values
  tail = int(values[-1]) if values.size else 0
  return np.concatenate([values, np.full((size - values.size,), tail, dtype=np.int32)])


def build_step_view(
    table: KvPageTableV1,
    shape: StepShape,
    tokens_per_page: int,
    padding_page_id: int = 0,
) -> StepView:
  """Pad `table` out to `shape` and move it to the device.

  Validates the table first: an inconsistent table padded into a static shape is
  considerably harder to diagnose than one rejected on the spot.
  """
  table.validate(tokens_per_page)

  num_requests = shape.num_requests
  query_lens = np.asarray(table.query_lens, dtype=np.int32)
  seq_lens = np.asarray(table.seq_lens, dtype=np.int32)

  cu_seqlens_q = np.zeros((query_lens.size + 1,), dtype=np.int32)
  if query_lens.size:
    np.cumsum(query_lens, out=cu_seqlens_q[1:])

  active_tokens = int(query_lens.sum())
  max_seq_len = int(seq_lens.max()) if seq_lens.size else 0
  if max_seq_len > shape.max_seqlen_k:
    raise ValueError(
        f"a request is {max_seq_len} tokens long but the step shape is configured for "
        f"{shape.max_seqlen_k}; the sequence-length ladder is mis-sized"
    )

  return StepView(
      slot_mapping=jnp.asarray(
          _pad_to(table.slot_mapping(tokens_per_page, padding_page_id), shape.num_tokens, -1, "slot_mapping"),
          jnp.int32,
      ),
      kv_indptr=jnp.asarray(_pad_cumulative(table.indptr(), num_requests + 1, "kv_indptr"), jnp.int32),
      kv_page_indices=jnp.asarray(
          _pad_to(table.flat_page_indices(), shape.num_pages, padding_page_id, "kv_page_indices"), jnp.int32
      ),
      kv_last_page_lens=jnp.asarray(
          _pad_to(table.last_page_lens(tokens_per_page), num_requests, 0, "kv_last_page_lens"), jnp.int32
      ),
      cu_seqlens_q=jnp.asarray(_pad_cumulative(cu_seqlens_q, num_requests + 1, "cu_seqlens_q"), jnp.int32),
      seq_lens=jnp.asarray(_pad_to(seq_lens, num_requests, 0, "seq_lens"), jnp.int32),
      shape=shape,
      num_active_requests=table.num_requests,
      num_active_tokens=active_tokens,
      # Static, from the bucket rather than from the data: a traced maximum would
      # be a fresh kernel configuration on almost every step.
      max_seqlen_q=1 if shape.is_decode else shape.num_tokens,
  )
