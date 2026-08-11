"""Logical geometry of a paged KV pool.

This type exists to produce two things: the pool byte count that the allocator
needs, and enough shape information for a backend to derive whatever physical
layout its kernels want. It deliberately stops short of that physical layout --
strides, packing and vendor shapes belong to the backend's own ABI.

The byte count is also the entire accelerator-facing surface of the pool. XLA's
total knowledge of a KV pool is a size in and a pointer out; it never learns what
a page means. That is the design, not an omission: fp8 KV, MLA's fused tensor and
x-packing each change this class or a vendor ABI, and none of them should require
rebuilding a runtime plugin.

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

import numpy as np

KV_STORAGE_LAYOUT_VERSION = 1

# Element sizes are tabulated rather than taken from numpy because the dtypes
# that matter most for KV are ones numpy does not have: bfloat16 and the fp8
# variants live in ml_dtypes, which this layer may not import. Sizing a pool is
# too load-bearing to depend on an optional package.
_ITEMSIZE_BY_NAME = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "float8_e4m3fn": 1,
    "float8_e4m3fnuz": 1,
    "float8_e5m2": 1,
    "float8_e5m2fnuz": 1,
    "int8": 1,
    "uint8": 1,
}


@dataclasses.dataclass(frozen=True)
class KvStorageLayoutV1:
    """Pool geometry, in logical terms.

    Attributes:
        tokens_per_page: page size in tokens; "block_size" in vLLM vocabulary.
        num_pages: total pages in the pool, per layer, per shard.
        num_layers: decoder layers backed by this pool.
        num_kv_heads: global KV head count, before any tensor-parallel sharding.
        head_dim: per-head dimension.
        dtype: numpy dtype name of the stored K/V.
        kv_head_shards: tensor-parallel width the pool is sharded over.
        padding_page_id: page reserved as a padding target and never allocated.
            This buys safe targets for padded rows and simpler invalid-page
            handling. It is emphatically *not* a security mechanism: stale KV in a
            recycled real page is a separate obligation.
    """

    version: int = KV_STORAGE_LAYOUT_VERSION
    tokens_per_page: int = 16
    num_pages: int = 0
    num_layers: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    dtype: str = "bfloat16"
    kv_head_shards: int = 1
    padding_page_id: int = 0

    def __post_init__(self):
        if self.tokens_per_page <= 0:
            raise ValueError(f"tokens_per_page must be positive, got {self.tokens_per_page}")
        if self.kv_head_shards <= 0:
            raise ValueError(f"kv_head_shards must be positive, got {self.kv_head_shards}")
        if self.num_kv_heads and self.kv_head_shards > self.num_kv_heads:
            if self.kv_head_shards % self.num_kv_heads != 0:
                raise ValueError(
                    f"kv_head_shards {self.kv_head_shards} exceeds num_kv_heads "
                    f"{self.num_kv_heads} without dividing evenly: no clean "
                    f"replication factor exists. Reject this configuration at "
                    f"startup rather than mis-sharding."
                )
        elif self.num_kv_heads and self.num_kv_heads % self.kv_head_shards != 0:
            raise ValueError(
                f"num_kv_heads {self.num_kv_heads} is not divisible by "
                f"kv_head_shards {self.kv_head_shards}"
            )

    def itemsize(self) -> int:
        """Bytes per stored element."""
        if self.dtype in _ITEMSIZE_BY_NAME:
            return _ITEMSIZE_BY_NAME[self.dtype]
        try:
            return int(np.dtype(self.dtype).itemsize)
        except TypeError as exc:
            raise ValueError(
                f"unknown KV dtype {self.dtype!r}; add it to _ITEMSIZE_BY_NAME"
            ) from exc

    def heads_per_shard(self) -> int:
        """KV heads held by one shard.

        Two regimes, and conflating them mis-sizes the pool. When the head count
        divides the shard count the heads are partitioned. When the shard count
        exceeds the head count -- GQA at high TP, and MQA always -- each head is
        *replicated*, so a shard still holds at least one whole head and the
        total footprint is multiplied rather than divided.
        """
        if self.kv_head_shards >= self.num_kv_heads:
            return max(self.num_kv_heads, 1) if self.num_kv_heads else 0
        return self.num_kv_heads // self.kv_head_shards

    def replication_factor(self) -> int:
        """How many shards hold a copy of the same KV head."""
        if self.num_kv_heads and self.kv_head_shards > self.num_kv_heads:
            return self.kv_head_shards // self.num_kv_heads
        return 1

    def bytes_per_page(self) -> int:
        """Bytes of one page of one of K or V, for one layer, on one shard.

        This is the only layout-derived quantity a page transfer needs, which is
        what keeps a transfer ABI from becoming a second copy of the attention
        ABI.
        """
        return self.tokens_per_page * self.heads_per_shard() * self.head_dim * self.itemsize()

    def bytes_per_token(self) -> int:
        """Bytes of K and V across all layers for one token, on one shard."""
        return 2 * self.num_layers * self.heads_per_shard() * self.head_dim * self.itemsize()

    def pool_bytes_per_shard(self) -> int:
        """Total pool bytes on one shard: K and V, all layers, all pages.

        This is the value handed to the allocator, and the entire
        accelerator-facing surface of this class.
        """
        return 2 * self.num_layers * self.num_pages * self.bytes_per_page()

    def total_pool_bytes(self) -> int:
        """Pool bytes summed over shards, including any replication."""
        return self.pool_bytes_per_shard() * self.kv_head_shards

    def max_tokens(self) -> int:
        """Live tokens the pool can hold, excluding the padding page."""
        return max(self.num_pages - 1, 0) * self.tokens_per_page
