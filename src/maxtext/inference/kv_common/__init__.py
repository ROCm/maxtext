"""Neutral vocabulary for the paged KV runtime.

This package is the durable interoperability surface. It describes pool geometry
and per-step page tables in kernel-neutral terms: no strides, no packing, no
vendor shapes, and no accelerator framework.

Import rule, enforced by CI: this package may import only the standard library
and ``numpy``. It must never import ``jax``, ``jax_aiter``, ``maxtext.layers``, or
``maxtext.models``. A second frontend or a second kernel vendor meets these types,
not a vendor ABI, which is why the restriction is worth policing rather than
merely documenting.

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

from maxtext.inference.kv_common.namespace import CACHE_NAMESPACE_VERSION, CacheNamespace
from maxtext.inference.kv_common.page_table import KV_PAGE_TABLE_VERSION, KvPageTableV1
from maxtext.inference.kv_common.storage_layout import (
    KV_STORAGE_LAYOUT_VERSION,
    KvStorageLayoutV1,
)

__all__ = [
    "CACHE_NAMESPACE_VERSION",
    "KV_PAGE_TABLE_VERSION",
    "KV_STORAGE_LAYOUT_VERSION",
    "CacheNamespace",
    "KvPageTableV1",
    "KvStorageLayoutV1",
]
