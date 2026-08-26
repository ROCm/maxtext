"""Allocate the paged KV pool.

The pool is a pair of NHD arrays per layer, `[num_pages, tokens_per_page,
heads_per_shard, head_dim]`, matching what the M3 attention path already reads so
nothing is repacked between append, prefill and decode.

**Zeros, not `empty`.** Two separate guarantees depend on it. The reserved
padding page is a landing zone that padded gather rows read from, and it must
read as zeros rather than as whatever was in that memory. And the allocator
treats a never-allocated page as clean, so it issues no scrub for it -- which is
only sound if the page really does start zeroed. `jnp.zeros` under `jit` is
materialised by the compiler rather than transferred, so this costs a kernel
launch, not a host copy.

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

import jax
import jax.numpy as jnp

from maxtext.inference.kv_common import KvStorageLayoutV1


@dataclasses.dataclass
class PagedKvPool:
  """The per-layer K and V arrays, plus the geometry that describes them.

  Mutable because the arrays are rebound on every step: the append op donates
  them and returns aliased results, so holding the originals after a step is how
  a caller ends up reading a stale buffer.
  """

  layout: KvStorageLayoutV1
  k_pages: list[jax.Array]
  v_pages: list[jax.Array]

  @property
  def num_layers(self) -> int:
    return len(self.k_pages)

  @property
  def page_shape(self) -> tuple[int, ...]:
    layout = self.layout
    return (layout.num_pages, layout.tokens_per_page, layout.heads_per_shard(), layout.head_dim)

  def bytes_per_shard(self) -> int:
    return self.layout.pool_bytes_per_shard()

  def replace_layer(self, layer: int, k: jax.Array, v: jax.Array) -> None:
    """Rebind one layer's pages after an aliased write."""
    self.k_pages[layer] = k
    self.v_pages[layer] = v


def allocate_pool(
    layout: KvStorageLayoutV1,
    sharding: Any | None = None,
    dtype: Any | None = None,
) -> PagedKvPool:
  """Allocate a zero-initialised pool for every layer.

  Args:
    layout: pool geometry. `heads_per_shard()` is already the per-device head
      count, including the replication case where TP exceeds the KV head count,
      so the shape below is the local shard rather than the global pool.
    sharding: optional sharding for each array. Passed through to
      `jax.device_put`; a `NamedSharding` carrying a `memory_kind` is how the
      pool would later be placed in a collective memory space.
    dtype: overrides the layout's dtype. Only useful for tests that want a
      dtype numpy can print.

  Returns:
    A `PagedKvPool` whose arrays are all zeros.
  """
  shape = (layout.num_pages, layout.tokens_per_page, layout.heads_per_shard(), layout.head_dim)
  resolved = jnp.dtype(dtype) if dtype is not None else jnp.dtype(layout.dtype)

  k_pages, v_pages = [], []
  for _ in range(layout.num_layers):
    k = jnp.zeros(shape, resolved)
    v = jnp.zeros(shape, resolved)
    if sharding is not None:
      k = jax.device_put(k, sharding)
      v = jax.device_put(v, sharding)
    k_pages.append(k)
    v_pages.append(v)
  return PagedKvPool(layout=layout, k_pages=k_pages, v_pages=v_pages)
