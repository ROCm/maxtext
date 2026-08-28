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

import numpy as np

from maxtext.inference.kv_common import KvStorageLayoutV1

# MaxText spells the KV-head tensor-parallel axes several ways depending on the
# model; these are the ones that shard `num_kv_heads`.
_KV_HEAD_MESH_AXES = ("tensor", "tensor_transpose", "tensor_sequence")

# Axis names for the pool's own mesh. Deliberately not MaxText's: the pool needs
# the tensor-parallel axis split into the part that selects a KV head and the
# part that replicates it, and MaxText's mesh has them fused into one axis.
KV_SHARD_AXIS = "kv_head_shard"
KV_REPLICA_AXIS = "kv_head_replica"


def kv_head_shards(mesh: Any | None) -> int:
  """Product of the mesh axes that shard KV heads, or 1 with no mesh."""
  if mesh is None:
    return 1
  shape = dict(getattr(mesh, "shape", {}) or {})
  shards = 1
  for axis in _KV_HEAD_MESH_AXES:
    shards *= int(shape.get(axis, 1))
  return max(shards, 1)


def pool_replicas(mesh: Any | None) -> int:
    """Devices holding a copy of each KV-head shard, from the non-KV mesh axes.

    The pool's `PartitionSpec` names only the KV-head axes, so it is replicated
    across every other axis of the mesh -- one copy per device on them. That is
    the only route to a replicated KV footprint that MaxText permits, since it
    refuses to build a model whose KV heads are sharded more ways than it has
    heads. On eight devices with four KV heads, `tensor=4, fsdp=2` gives four
    shards and two replicas, and every device holds exactly one head.
    """
    if mesh is None:
        return 1
    devices = int(getattr(mesh, "size", 0) or 0)
    if devices <= 0:
        return 1
    shards = kv_head_shards(mesh)
    return max(devices // max(shards, 1), 1)


def kv_pool_sharding(mesh: Any | None, layout: KvStorageLayoutV1) -> Any | None:
  """Where each KV head of the pool lives, as a `NamedSharding`.

  The pool arrays are *globally* shaped -- `num_kv_heads` on the head axis -- and
  this sharding is what puts the right heads on the right devices. Allocating
  per-device shards directly would be the other option, but it puts the caller in
  charge of device order and makes the arrays unusable as ordinary jit inputs.

  **The tensor-parallel axis has to be split in two, which is why the pool builds
  its own mesh.** MaxText's mesh fuses them: `tensor` of width 8 says nothing
  about whether eight ranks hold eight distinct KV heads or two heads replicated
  four ways. Both happen, and they need different device assignments.

  The split is `(kv_head_shard, kv_head_replica)`, row-major over the same device
  order MaxText uses, because that is the assignment the model already implies.
  Rank `i` computes query head `i`, and query head `i` reads KV head
  `i // replication_factor` -- so consecutive ranks share a KV head, which is
  exactly what a row-major reshape produces. Getting this backwards would put a
  rank's KV on another rank's device and force a gather on every step, which is
  the cost M6 exists to remove and which would show up as a correct-but-slow run
  rather than a failure.

  Returns None when there is nothing to shard, so a single-device caller passes
  the result straight through and gets the ordinary unsharded pool.
  """
  # pylint: disable=import-outside-toplevel
  import jax

  shards = int(layout.kv_head_shards)
  if mesh is None or shards <= 1:
    return None

  # The head axis is what decides whether the model's own mesh can express this,
  # and it is *not* the same question as whether the pool ends up replicated.
  # Naming only the KV-head axes in the spec leaves the pool replicated across
  # every other mesh axis automatically, which is exactly the layout wanted when
  # surplus parallelism sits on `fsdp` -- four shards over `tensor`, two copies
  # over `fsdp`, one head per device. Branching on `replication_factor()` instead
  # sends that perfectly ordinary case down the private-mesh path and fails.
  if not layout.num_kv_heads or shards <= layout.num_kv_heads:
    # The model's own mesh already expresses it, so use that rather than a
    # private one: `shard_map` needs the pool and the activations on the *same*
    # mesh, and a second mesh over the same devices is not the same mesh as far
    # as it is concerned.
    axes = tuple(a for a in _KV_HEAD_MESH_AXES if int(dict(mesh.shape).get(a, 1)) > 1)
    spec = jax.sharding.PartitionSpec(None, None, axes if len(axes) > 1 else axes[0], None)
    return jax.sharding.NamedSharding(mesh, spec)

  # Only reachable when the head axis itself is over-sharded, which MaxText
  # refuses to build a model for. Kept because this counts mesh axes directly
  # while MaxText counts them through the logical axis rules.
  replication = layout.replication_factor()
  devices = np.asarray(getattr(mesh, "devices", None))
  if devices.size != shards:
    raise ValueError(
        f"the pool is sharded {shards} ways but the mesh holds {devices.size} devices. The layout's "
        f"kv_head_shards must come from this mesh, or the pool will be laid out for a different one."
    )

  kv_axis = shards // replication
  # Row-major, and matching MaxText's device order rather than imposing one.
  grid = devices.reshape(kv_axis, replication)
  pool_mesh = jax.sharding.Mesh(grid, (KV_SHARD_AXIS, KV_REPLICA_AXIS))
  # Only the head axis is partitioned; pages, tokens within a page, and head_dim
  # are whole on every device. Replication across the second axis is implicit in
  # not naming it.
  spec = jax.sharding.PartitionSpec(None, None, KV_SHARD_AXIS, None)
  return jax.sharding.NamedSharding(pool_mesh, spec)


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
      pool_replicas=pool_replicas(mesh),
      padding_page_id=0,
  )
