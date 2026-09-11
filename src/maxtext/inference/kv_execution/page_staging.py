"""Move a set of pages between the device pool and a host-resident slab.

This is M7's substance, and it is deliberately not written as "host offload".
M7, M8 and M9 all need the same primitive -- gather a scattered set of pages
into a contiguous staging buffer, move it, scatter it into a different scattered
set at the far end -- and differ only in where the buffer goes. Host is simply
the first backend, and the cheapest one to get right because it needs no second
process, no sockets and no rendezvous.

**Offload is a staging tier, not an execution tier.** Nothing ever attends from
host memory. Measured on MI325X, host is roughly 19 GB/s reading and 55 GB/s
writing against HBM's ~6 TB/s, so serving a kernel from there is not a slow path
but a broken one. What offload buys is capacity, and capacity converts to
throughput here because decode is bandwidth-bound: M4.6 measured step time flat
from batch 2 to 16 while throughput rose 41x, so a larger batch amortises the
weight read and KV footprint is what caps the batch. It also buys back the
prefill of a preempted request, which is otherwise recomputed.

Three things about the JAX side are not obvious and were established by running
them rather than reading:

* **The gather fuses with the destination memory space.** `jnp.take` under a jit
  whose `out_shardings` names `pinned_host` runs the gather on device and lands
  the result in host memory, one dispatch, no intermediate device copy.
* **The scatter does not fuse.** A scatter whose operand is host-resident fails
  at compile time with `CompileTimeHostOffloadOutputLocationMismatch`, so the
  staged array has to be moved back explicitly before it can be written into the
  pool. That asymmetry is why `fetch` has two steps and `store` has one.
* **A stray op against a host array silently copies it back.** `dynamic_slice` on
  a host-resident array returned a *device* array with no diagnostic, XLA having
  inserted an implicit H2D. On a multi-gigabyte slab that is an invisible PCIe
  transfer, which is the argument for keeping every host-side access in this
  module rather than letting callers touch the slab.

The page-count axis is bucketed for the same reason `pool_ops` buckets it: these
run in a serving loop, and a fresh trace per distinct page count would defeat the
whole of the bucketing work. Padding repeats an index that is already being
moved, so it is idempotent in both directions and needs no mask -- and because
gather and scatter are handed the *same* padded index array, the padded rows
round-trip consistently.

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
from typing import Any, Sequence

import numpy as np

import jax
import jax.numpy as jnp

HOST_MEMORY_KIND = "pinned_host"
DEVICE_MEMORY_KIND = "device"


def pad_page_indices(page_ids: Sequence[int] | np.ndarray) -> np.ndarray:
  """Pad to a power of two by repeating the first page.

  Repetition rather than a sentinel plus a mask, matching `pool_ops`: reading a
  page twice is free, and writing it twice with the same bytes is idempotent, so
  neither direction needs a branch in the kernel.
  """
  pages = np.asarray(page_ids, dtype=np.int32).reshape(-1)
  if pages.size == 0:
    return pages
  padded = 1 << (int(pages.size) - 1).bit_length()
  if padded == pages.size:
    return pages
  return np.concatenate([pages, np.full((padded - pages.size,), pages[0], dtype=np.int32)])


def host_sharding_like(pool_sharding: Any | None, device: Any | None = None) -> Any:
  """The pool's sharding, restated in host memory.

  Keeping the pool's `PartitionSpec` matters under tensor parallelism: each
  device offloads the head shard it owns, so the slab is host memory *per
  device* rather than one host copy gathered from all of them. Gathering would
  turn an offload into a collective.
  """
  if pool_sharding is None:
    target = device if device is not None else jax.devices()[0]
    return jax.sharding.SingleDeviceSharding(target, memory_kind=HOST_MEMORY_KIND)
  return jax.sharding.NamedSharding(
      pool_sharding.mesh, pool_sharding.spec, memory_kind=HOST_MEMORY_KIND
  )


def device_sharding_like(host_sharding: Any) -> Any:
  """The inverse of `host_sharding_like`, for the return leg."""
  if isinstance(host_sharding, jax.sharding.NamedSharding):
    return jax.sharding.NamedSharding(
        host_sharding.mesh, host_sharding.spec, memory_kind=DEVICE_MEMORY_KIND
    )
  return jax.sharding.SingleDeviceSharding(
      next(iter(host_sharding.device_set)), memory_kind=DEVICE_MEMORY_KIND
  )


@functools.lru_cache(maxsize=16)
def _gather_fn(num_layers: int, host: Any):
  """The gather, jitted once per (depth, destination) rather than per call.

  Cached because `out_shardings` has to be baked in -- it is what makes the
  gather land in host memory instead of taking a device round trip -- and a
  freshly built `jax.jit` on each offload would retrace every time, which is the
  opposite of what bucketing the page count is for.

  One dispatch across all layers, for `pool_ops._fill_all`'s reason: at 80
  layers a per-layer loop issues eighty launches on the critical path of every
  offload. That costs nothing at test depth and is a cliff at real depth.
  """
  return jax.jit(
      lambda k, v, idx: ([jnp.take(x, idx, axis=0) for x in k],
                         [jnp.take(x, idx, axis=0) for x in v]),
      out_shardings=([host] * num_layers, [host] * num_layers),
  )


@functools.partial(jax.jit, donate_argnums=(0, 1))
def _scatter_all_layers(k_pages: list, v_pages: list, page_ids: jax.Array,
                        k_staged: list, v_staged: list):
  """Write the staged pages back into every layer, in one dispatch.

  Donated so the write aliases the pool rather than replacing it; the caller
  must rebind from the returned handles.
  """
  return ([k.at[page_ids].set(s) for k, s in zip(k_pages, k_staged)],
          [v.at[page_ids].set(s) for v, s in zip(v_pages, v_staged)])


class StagedPages:
  """One offload's worth of pages, resident in host memory.

  Carries the padded index array it was gathered with, because the scatter has
  to use the identical one: the padded rows only round-trip consistently if both
  directions agree on which page each row came from.
  """

  __slots__ = ("k", "v", "page_ids", "num_live")

  def __init__(self, k: list, v: list, page_ids: np.ndarray, num_live: int) -> None:
    self.k = k
    self.v = v
    self.page_ids = page_ids
    self.num_live = num_live

  @property
  def num_layers(self) -> int:
    return len(self.k)

  def nbytes(self) -> int:
    return sum(int(a.size) * int(a.dtype.itemsize) for a in self.k + self.v)


def offload_pages(pool: Any, page_ids: Sequence[int] | np.ndarray,
                  pool_sharding: Any | None = None) -> StagedPages:
  """Copy the named pool pages into host memory and return the staged copy.

  Does not free or scrub the pages. Deciding a page may be reused is the control
  plane's call, and coupling it here would let a failed transfer look like a
  successful eviction.
  """
  live = np.asarray(page_ids, dtype=np.int32).reshape(-1)
  padded = pad_page_indices(live)
  if padded.size == 0:
    return StagedPages([], [], padded, 0)

  host = host_sharding_like(pool_sharding)
  gather = _gather_fn(pool.num_layers, host)
  k_staged, v_staged = gather(pool.k_pages, pool.v_pages, jnp.asarray(padded))
  return StagedPages(k_staged, v_staged, padded, int(live.size))


def reload_pages(pool: Any, staged: StagedPages, pool_sharding: Any | None = None) -> None:
  """Write a staged copy back into the pool, in place.

  The pool arrays are donated, so `pool.replace_layer` is not optional -- the
  arrays passed in are dead once this returns.

  Two steps rather than one because a scatter cannot take a host-resident
  operand: XLA rejects it at compile time rather than inserting the copy itself,
  so the move back to device is explicit here.
  """
  if staged.num_live == 0:
    return

  device = (device_sharding_like(host_sharding_like(pool_sharding))
            if pool_sharding is not None
            else jax.sharding.SingleDeviceSharding(
                next(iter(staged.k[0].sharding.device_set)), memory_kind=DEVICE_MEMORY_KIND))

  k_device = [jax.device_put(a, device) for a in staged.k]
  v_device = [jax.device_put(a, device) for a in staged.v]
  k_pages, v_pages = _scatter_all_layers(
      list(pool.k_pages), list(pool.v_pages), jnp.asarray(staged.page_ids), k_device, v_device
  )
  for layer in range(pool.num_layers):
    pool.replace_layer(layer, k_pages[layer], v_pages[layer])
