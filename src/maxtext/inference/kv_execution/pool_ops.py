"""Device-side page hygiene: scrub a recycled page, or poison a freed one.

The control plane can say which pages hold another request's KV, but it cannot
do anything about it -- the pool is device memory and `kv_control` never touches
a device. These two functions are where that obligation is actually discharged.

`scrub_pages` zeroes. That is what makes a recycled page safe to hand on.
`poison_pages` fills with a recognisable sentinel instead, which is strictly a
debugging aid: it makes an unscrubbed read *loud* rather than plausible. Poison
is not a scrub, and the control plane deliberately keeps a poisoned page marked
dirty so it cannot be read until it has really been zeroed.

**The page-count axis is bucketed, and it has to be.** These run inside a serving
loop, so a fresh trace for every distinct number of recycled pages would defeat
the whole of Step 5. The index array is padded up to a power of two with repeats
of a page that is already being written, so the padding is idempotent rather than
needing a mask.

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

import functools
from typing import Sequence

import numpy as np

import jax
import jax.numpy as jnp

# A power of two, and that is the whole reason for this value. A sentinel is only
# useful if a test can say "this byte is the sentinel" exactly, and every dtype a
# KV pool might use -- bfloat16, float16, both fp8 variants -- carries far too
# little mantissa to store a round decimal like -8e4 without rounding it. Powers
# of two are exact in all of them, and -256 is inside fp8 e4m3's range of 448
# where -65536 would overflow float16 outright.
#
# Otherwise: large, finite, negative, and nothing a real activation produces. Not
# a NaN, which would propagate through any downstream reduction and destroy the
# evidence of where the read actually happened.
POISON_SENTINEL = -256.0


def _pad_page_indices(page_ids: Sequence[int] | np.ndarray) -> np.ndarray:
  """Pad to a power of two by repeating the first page.

  Repetition rather than a sentinel plus a mask: both fills are idempotent, so
  writing a page twice is free and needs no branch in the kernel.
  """
  pages = np.asarray(page_ids, dtype=np.int32).reshape(-1)
  if pages.size == 0:
    return pages
  padded_size = 1 << (int(pages.size) - 1).bit_length()
  if padded_size == pages.size:
    return pages
  return np.concatenate([pages, np.full((padded_size - pages.size,), pages[0], dtype=np.int32)])


@functools.partial(jax.jit, donate_argnums=(0, 1))
def _fill(k_pages: jax.Array, v_pages: jax.Array, page_ids: jax.Array, value: jax.Array):
  """Set every element of the named pages to `value`, in place."""
  fill_shape = (page_ids.shape[0],) + k_pages.shape[1:]
  block = jnp.full(fill_shape, value, dtype=k_pages.dtype)
  return k_pages.at[page_ids].set(block), v_pages.at[page_ids].set(block)


@functools.partial(jax.jit, donate_argnums=(0, 1))
def _fill_all(k_pages: list, v_pages: list, page_ids: jax.Array, value: jax.Array):
  """`_fill` over every layer at once.

  One dispatch rather than one per layer, and at 80 layers that difference is
  the whole point. Every layer's pool has the same shape and takes the same page
  indices, so the per-layer loop was issuing eighty identical launches -- each
  with its own donation bookkeeping across every device -- on the critical path
  of any step that recycled a page. It costs nothing at the four-layer scale the
  earlier measurements used and shows up as a throughput cliff at real depth,
  which is exactly the shape of defect Section 6.4 exists to catch.

  Taking lists rather than stacked arrays keeps the pool's per-layer identity, so
  donation still aliases each layer's own buffer and nothing is repacked.
  """
  def fill(arr):
    block = jnp.full((page_ids.shape[0],) + arr.shape[1:], value, dtype=arr.dtype)
    return arr.at[page_ids].set(block)

  return [fill(k) for k in k_pages], [fill(v) for v in v_pages]


def scrub_pages_all_layers(
    k_pages: Sequence[jax.Array],
    v_pages: Sequence[jax.Array],
    page_ids: Sequence[int] | np.ndarray,
) -> tuple[list[jax.Array], list[jax.Array]]:
  """Zero the named pages in every layer, in a single dispatch.

  Returns the rebound arrays, which alias the inputs. A no-op for an empty list,
  which is the common case on a pool that has not wrapped around yet.
  """
  pages = _pad_page_indices(page_ids)
  if pages.size == 0:
    return list(k_pages), list(v_pages)
  return _fill_all(list(k_pages), list(v_pages), jnp.asarray(pages), jnp.zeros((), k_pages[0].dtype))


def scrub_pages(
    k_pages: jax.Array,
    v_pages: jax.Array,
    page_ids: Sequence[int] | np.ndarray,
) -> tuple[jax.Array, jax.Array]:
  """Zero the named pages. Returns the rebound arrays, which alias the inputs.

  A no-op for an empty list, which is the common case on a pool that has not
  wrapped around yet.
  """
  pages = _pad_page_indices(page_ids)
  if pages.size == 0:
    return k_pages, v_pages
  return _fill(k_pages, v_pages, jnp.asarray(pages), jnp.zeros((), k_pages.dtype))


def poison_pages(
    k_pages: jax.Array,
    v_pages: jax.Array,
    page_ids: Sequence[int] | np.ndarray,
    value: float = POISON_SENTINEL,
) -> tuple[jax.Array, jax.Array]:
  """Fill the named pages with a sentinel, for debugging only.

  Does not discharge a scrub: a poisoned page is still dirty, and the control
  plane will still refuse to build a page table naming it.
  """
  pages = _pad_page_indices(page_ids)
  if pages.size == 0:
    return k_pages, v_pages
  return _fill(k_pages, v_pages, jnp.asarray(pages), jnp.asarray(value, k_pages.dtype))
