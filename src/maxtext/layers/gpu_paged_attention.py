"""Paged attention over a GPU KV pool, for `attention: "gpu_paged"`.

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

Everything here is vendor-neutral except `_call_backend`, which is the single
leaf that knows which kernel provider is in use. That is the whole point of the
split: FlashInfer on NVIDIA slots in beside aiter on ROCm without a second code
path through MaxText.

The KV pool is a pair of NHD arrays, `[num_pages, tokens_per_page, num_kv_heads,
head_dim]`, carried as the layer's `kv_cache`. One step writes the new K/V into
the pool and then attends over it, so both halves see the same pages and nothing
is repacked when a request moves between prefill and decode.

Two metadata shapes are accepted, because the serving harness that produces them
differs by platform and neither package can be assumed present:

  * a neutral `KvPageTableV1` (MaxText's own vocabulary, `inference/kv_common/`),
    recognised by its `indptr()` / `flat_page_indices()` / `last_page_lens()`
    methods. Its fields are concrete host values, so the conversion is numpy.
  * a vLLM-shaped metadata object with `block_tables` / `seq_lens` /
    `query_start_loc`, as `tpu_inference.AttentionMetadata` supplies. Those are
    traced arrays, so that conversion has to be `jnp` and shape-static.

Both are duck-typed. Importing either producer here would defeat the neutrality
the split exists for.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class PagedPlan:
  """One step's page bookkeeping, as the device arrays the kernels take."""

  slot_mapping: jax.Array       # [total_tokens]     where each new token is written
  kv_indptr: jax.Array          # [num_seqs + 1]     prefix sum over page counts
  kv_page_indices: jax.Array    # [>= total_pages]   page ids, packed in request order
  kv_last_page_lens: jax.Array  # [num_seqs]         occupancy of each final page
  cu_seqlens_q: jax.Array       # [num_seqs + 1]     prefix sum over query lengths
  max_seqlen_q: int
  max_seqlen_k: int
  is_decode: bool


def is_neutral_page_table(metadata: Any) -> bool:
  """True for MaxText's `KvPageTableV1`, without importing it."""
  return all(
      callable(getattr(metadata, name, None))
      for name in ("indptr", "flat_page_indices", "last_page_lens", "slot_mapping")
  )


def is_vllm_metadata(metadata: Any) -> bool:
  """True for a vLLM-shaped metadata object, without importing tpu_inference."""
  return all(hasattr(metadata, name) for name in ("block_tables", "seq_lens", "query_start_loc"))


def plan_from_neutral(page_table: Any, tokens_per_page: int, padding_page_id: int = 0) -> PagedPlan:
  """Convert a `KvPageTableV1`. Its members are host values, so this is numpy."""
  import numpy as np  # pylint: disable=import-outside-toplevel

  page_table.validate(tokens_per_page)

  query_lens = np.asarray(page_table.query_lens, dtype=np.int32)
  seq_lens = np.asarray(page_table.seq_lens, dtype=np.int32)
  cu_seqlens_q = np.zeros((query_lens.size + 1,), dtype=np.int32)
  if query_lens.size:
    np.cumsum(query_lens, out=cu_seqlens_q[1:])

  return PagedPlan(
      slot_mapping=jnp.asarray(page_table.slot_mapping(tokens_per_page, padding_page_id), jnp.int32),
      kv_indptr=jnp.asarray(page_table.indptr(), jnp.int32),
      kv_page_indices=jnp.asarray(page_table.flat_page_indices(), jnp.int32),
      kv_last_page_lens=jnp.asarray(page_table.last_page_lens(tokens_per_page), jnp.int32),
      cu_seqlens_q=jnp.asarray(cu_seqlens_q, jnp.int32),
      max_seqlen_q=int(query_lens.max()) if query_lens.size else 0,
      max_seqlen_k=int(seq_lens.max()) if seq_lens.size else 0,
      is_decode=bool(query_lens.size and np.all(query_lens == 1)),
  )


def plan_from_vllm(metadata: Any, tokens_per_page: int, total_tokens: int, max_seqlen_k: int) -> PagedPlan:
  """Convert vLLM-shaped metadata. These are traced arrays, so this is `jnp`.

  The 2D `block_tables` is `[num_seqs, max_blocks_per_seq]` and rows are padded,
  while the kernels want the page ids packed contiguously with `kv_indptr`
  pointing at each request's run. Packing is a scatter: every valid `(seq, block)`
  lands at `kv_indptr[seq] + block`, and invalid entries are sent to a scratch
  slot past the end so the shape stays static under `jit`.
  """
  block_tables = jnp.asarray(metadata.block_tables, jnp.int32)
  seq_lens = jnp.asarray(metadata.seq_lens, jnp.int32)
  query_start_loc = jnp.asarray(metadata.query_start_loc, jnp.int32)

  if block_tables.ndim != 2:
    raise ValueError(f"block_tables must be [num_seqs, max_blocks_per_seq], got shape {block_tables.shape}")
  num_seqs, max_blocks = block_tables.shape

  pages_per_seq = (seq_lens + tokens_per_page - 1) // tokens_per_page
  kv_indptr = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(pages_per_seq).astype(jnp.int32)])

  block = jnp.arange(max_blocks, dtype=jnp.int32)[None, :]
  valid = block < pages_per_seq[:, None]
  dest = jnp.where(valid, kv_indptr[:num_seqs, None] + block, num_seqs * max_blocks)
  packed = jnp.zeros((num_seqs * max_blocks + 1,), jnp.int32).at[dest.reshape(-1)].set(block_tables.reshape(-1))
  kv_page_indices = packed[:-1]

  # An empty sequence has no final page; an overstated length is how a kernel
  # reads bytes left behind by a recycled page's previous occupant.
  last_page_lens = jnp.where(seq_lens > 0, ((seq_lens - 1) % tokens_per_page) + 1, 0).astype(jnp.int32)

  # Each new token's absolute position within its own sequence, hence its slot.
  query_lens = jnp.diff(query_start_loc)
  token = jnp.arange(total_tokens, dtype=jnp.int32)
  seq_id = jnp.searchsorted(query_start_loc[1:], token, side="right")
  within = token - query_start_loc[seq_id]
  position = seq_lens[seq_id] - query_lens[seq_id] + within
  page = block_tables[seq_id, position // tokens_per_page]
  slot_mapping = (page * tokens_per_page + position % tokens_per_page).astype(jnp.int32)

  return PagedPlan(
      slot_mapping=slot_mapping,
      kv_indptr=kv_indptr,
      kv_page_indices=kv_page_indices,
      kv_last_page_lens=last_page_lens,
      cu_seqlens_q=query_start_loc,
      # Query lengths are traced, so the caller supplies the static bounds the
      # kernels bake into their configuration.
      max_seqlen_q=total_tokens,
      max_seqlen_k=max_seqlen_k,
      is_decode=(total_tokens == int(num_seqs)),
  )


def build_plan(metadata: Any, tokens_per_page: int, total_tokens: int, max_seqlen_k: int) -> PagedPlan:
  """Dispatch on the metadata's shape rather than on its type."""
  if isinstance(metadata, PagedPlan):
    return metadata
  if is_neutral_page_table(metadata):
    return plan_from_neutral(metadata, tokens_per_page)
  if is_vllm_metadata(metadata):
    return plan_from_vllm(metadata, tokens_per_page, total_tokens, max_seqlen_k)
  raise TypeError(
      "gpu_paged attention metadata must be either a neutral KvPageTableV1 "
      "(indptr/flat_page_indices/last_page_lens/slot_mapping) or a vLLM-shaped object "
      f"(block_tables/seq_lens/query_start_loc); got {type(metadata).__name__}."
  )


def resolve_backend(configured: str) -> str:
  """`auto` follows the platform; anything else is taken literally."""
  if configured != "auto":
    return configured
  try:
    platform = jax.devices()[0].platform
    backend = jax.devices()[0].client.platform_version
  except Exception:  # pylint: disable=broad-exception-caught
    return "aiter"
  if platform == "gpu" and "cuda" in backend.lower():
    return "flashinfer"
  return "aiter"


def _call_backend(backend, query, k_pool, v_pool, plan, scale, causal):
  """The only vendor-specific code in this file."""
  if backend == "aiter":
    # Imported lazily: MaxText must not require jax-aiter to be installed for
    # any other attention mode.
    # pylint: disable=import-outside-toplevel
    from jax_aiter.ops.append_kv import append_kv
    from jax_aiter.ops.paged_attention import paged_attention
    from jax_aiter.ops.paged_prefill import paged_prefill

    return append_kv, paged_attention, paged_prefill

  if backend == "flashinfer":
    raise NotImplementedError(
        "paged_attention_backend='flashinfer' is not wired up yet; only 'aiter' is implemented. "
        "The MaxText side is backend-agnostic, so adding it is a change to this function alone."
    )

  raise ValueError(f"unknown paged_attention_backend {backend!r}; expected 'auto', 'aiter' or 'flashinfer'")


def paged_attention_step(query, key, value, k_pool, v_pool, plan, *, backend="auto", scale=None, causal=True):
  """Write this step's K/V into the pool, then attend over it.

  Returns `(output, [k_pool, v_pool])`. The pools are donated and returned
  aliased, so callers must rebind rather than keep the originals.
  """
  append_kv, paged_attention, paged_prefill = _call_backend(
      resolve_backend(backend), query, k_pool, v_pool, plan, scale, causal
  )

  k_pool, v_pool = append_kv(key, value, plan.slot_mapping, k_pool, v_pool)

  if plan.is_decode:
    out = paged_attention(
        query,
        k_pool,
        v_pool,
        plan.kv_indptr,
        plan.kv_page_indices,
        plan.kv_last_page_lens,
        max_seq_len=plan.max_seqlen_k,
        scale=scale,
    )
  else:
    out = paged_prefill(
        query,
        k_pool,
        v_pool,
        plan.cu_seqlens_q,
        plan.kv_indptr,
        plan.kv_page_indices,
        plan.kv_last_page_lens,
        max_seqlen_q=plan.max_seqlen_q,
        max_seqlen_k=plan.max_seqlen_k,
        scale=scale,
        causal=causal,
    )
  return out, [k_pool, v_pool]


def kv_head_axes(mesh: Any, candidates: tuple[str, ...] = ("tensor", "tensor_transpose", "tensor_sequence")):
  """The mesh axes that actually shard KV heads, or () when nothing does."""
  if mesh is None:
    return ()
  shape = dict(getattr(mesh, "shape", {}) or {})
  return tuple(axis for axis in candidates if int(shape.get(axis, 1)) > 1)


def paged_attention_step_sharded(
    query, key, value, k_pool, v_pool, plan, *, mesh, axes, backend="auto", scale=None, causal=True
):
  """`paged_attention_step` under `shard_map`, so each device sees its own heads.

  XLA cannot partition an FFI custom call. Handed a sharded pool it neither
  gathers nor splits and the step hangs, so the kernels have to be entered in
  manual mode where every device runs the same code on its local shard and the
  kernel sees a narrower pool than the global one. Nothing vendor-side changes:
  the jax-aiter KV ops carry no `custom_partitioning`, which is what would
  otherwise conflict with `shard_map`.

  **The plan arrays are replicated, and that is the whole reason this is free.**
  Page ids, slot offsets and last-page occupancies describe *pages*, and pages
  are not sharded -- every device holds the same pages and differs only in which
  heads it stores. So each device can compute its slice of the attention with no
  knowledge of any other, and the step needs no cross-device KV traffic at all,
  which is precisely the property M6 exists to establish.

  `plan` is not a pytree, so its arrays are passed positionally and its static
  fields are closed over.
  """
  qkv_spec = jax.sharding.PartitionSpec(None, axes, None)
  pool_spec = jax.sharding.PartitionSpec(None, None, axes, None)
  replicated = jax.sharding.PartitionSpec()

  def body(query, key, value, k_pool, v_pool, slot_mapping, kv_indptr, page_indices, last_page_lens, cu_seqlens_q):
    local = dataclasses.replace(
        plan,
        slot_mapping=slot_mapping,
        kv_indptr=kv_indptr,
        kv_page_indices=page_indices,
        kv_last_page_lens=last_page_lens,
        cu_seqlens_q=cu_seqlens_q,
    )
    # Flattened, because `shard_map` matches out_specs against the output tree
    # and a nested list there is one more level than the specs can describe.
    out, pools = paged_attention_step(
        query, key, value, k_pool, v_pool, local, backend=backend, scale=scale, causal=causal
    )
    return out, pools[0], pools[1]

  # check_vma is off because the replicated plan arrays are read by every device
  # and the pools are written per-device; the varying-manual-axes check cannot
  # see that an aliased FFI write is confined to the local shard.
  out, k_pool, v_pool = jax.shard_map(
      body,
      mesh=mesh,
      # Only the KV-head axes go manual. Left to default, `shard_map` makes
      # *every* mesh axis manual, and MaxText's mesh carries a dozen -- so the
      # region would be entered under a different set of manual axes than the
      # computation around it, which is not what the specs describe. A
      # single-axis mesh cannot expose this, which is how an isolated test stays
      # bit-exact while the model does not.
      axis_names=frozenset(axes if isinstance(axes, tuple) else (axes,)),
      in_specs=(
          qkv_spec, qkv_spec, qkv_spec, pool_spec, pool_spec,
          replicated, replicated, replicated, replicated, replicated,
      ),
      out_specs=(qkv_spec, pool_spec, pool_spec),
      check_vma=False,
  )(
      query, key, value, k_pool, v_pool,
      plan.slot_mapping, plan.kv_indptr, plan.kv_page_indices, plan.kv_last_page_lens, plan.cu_seqlens_q,
  )
  return out, [k_pool, v_pool]
