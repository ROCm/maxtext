"""MaxText config plus mesh into a `KvStorageLayoutV1`.

One function, and its only interesting job is deciding `kv_head_shards` from the
mesh rather than from a config field, because a mismatch between the two is
exactly the kind of thing that produces a pool a factor of TP too small and a
wrong-answer failure much later.

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

from typing import Any

from maxtext.inference.kv_common import KvStorageLayoutV1

# MaxText spells the KV-head tensor-parallel axes several ways depending on the
# model; these are the ones that shard `num_kv_heads`.
_KV_HEAD_MESH_AXES = ("tensor", "tensor_transpose", "tensor_sequence")


def kv_head_shards(mesh: Any | None) -> int:
  """Product of the mesh axes that shard KV heads, or 1 with no mesh."""
  if mesh is None:
    return 1
  shape = dict(getattr(mesh, "shape", {}) or {})
  shards = 1
  for axis in _KV_HEAD_MESH_AXES:
    shards *= int(shape.get(axis, 1))
  return max(shards, 1)


def build_storage_layout(config: Any, mesh: Any | None = None) -> KvStorageLayoutV1:
  """Derive pool geometry from a MaxText config.

  `num_pages` includes the reserved padding page, so the pool holds
  `paged_num_blocks` usable pages plus one. Sizing the pool as exactly
  `paged_num_blocks` and then reserving one out of it would silently cost a page
  of capacity relative to what the config asked for.
  """
  num_kv_heads = int(getattr(config, "num_kv_heads", 0) or 0)
  head_dim = int(getattr(config, "head_dim", 0) or 0)
  num_layers = int(getattr(config, "num_decoder_layers", 0) or getattr(config, "base_num_decoder_layers", 0) or 0)
  tokens_per_page = int(getattr(config, "paged_page_size", 16))
  usable_pages = int(getattr(config, "paged_num_blocks", 0) or 0)
  if usable_pages < 1:
    raise ValueError(
        f"paged_num_blocks must be at least 1 for attention='gpu_paged', got {usable_pages}"
    )

  dtype = str(getattr(config, "dtype", "bfloat16"))

  return KvStorageLayoutV1(
      tokens_per_page=tokens_per_page,
      num_pages=usable_pages + 1,
      num_layers=num_layers,
      num_kv_heads=num_kv_heads,
      head_dim=head_dim,
      dtype=dtype,
      kv_head_shards=kv_head_shards(mesh),
      padding_page_id=0,
  )
