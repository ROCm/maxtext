# SPDX-License-Identifier: Apache-2.0
"""FlyDSL MoE block: token sort + 2-stage grouped GEMM from preshuffled weights."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

from flydsl_moe.gemm import moe_gemm1_jax, moe_gemm2_jax, moe_gemm2_reduce_jax


def pick_tile_m(tokens: int) -> int:
    """tile_m heuristic. Override in the caller if you tune per-shape."""
    if tokens <= 1:
        return 16
    if tokens <= 32:
        return 32
    return 64


def moe_sort_jax(
    topk_ids: jax.Array,
    topk_weights: jax.Array,
    *,
    num_experts: int,
    block_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Device-side MoE sort. Output layout matches ``moe_sorting_numpy``."""
    T, topk = topk_ids.shape
    bs = block_size

    max_padded = T * topk + num_experts * (bs - 1)
    max_blocks = (max_padded + bs - 1) // bs
    sentinel = jnp.int32((topk << 24) | T)

    flat_experts = topk_ids.reshape(-1).astype(jnp.int32)
    flat_token_idx = jnp.repeat(jnp.arange(T, dtype=jnp.int32), topk)
    flat_slot_idx = jnp.tile(jnp.arange(topk, dtype=jnp.int32), T)
    flat_weights = topk_weights.reshape(-1).astype(jnp.float32)
    packed = (flat_slot_idx.astype(jnp.int32) << 24) | flat_token_idx

    sort_perm = jnp.argsort(flat_experts, stable=True)
    sorted_packed_d = packed[sort_perm]
    sorted_weights_d = flat_weights[sort_perm]
    sorted_experts_d = flat_experts[sort_perm]

    counts = jnp.bincount(flat_experts, length=num_experts).astype(jnp.int32)
    padded_counts = (counts + bs - 1) // bs * bs

    dense_offsets = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(counts)])
    padded_offsets = jnp.concatenate(
        [jnp.zeros(1, jnp.int32), jnp.cumsum(padded_counts)]
    )

    e_of_i = sorted_experts_d
    padded_pos = (
        padded_offsets[e_of_i]
        + jnp.arange(T * topk, dtype=jnp.int32)
        - dense_offsets[e_of_i]
    )

    sorted_ids = jnp.full(max_padded, sentinel, dtype=jnp.int32)
    sorted_weights = jnp.zeros(max_padded, dtype=jnp.float32)
    sorted_ids = sorted_ids.at[padded_pos].set(sorted_packed_d)
    sorted_weights = sorted_weights.at[padded_pos].set(sorted_weights_d)

    blocks_per_expert = padded_counts // bs
    block_offsets = jnp.concatenate(
        [jnp.zeros(1, jnp.int32), jnp.cumsum(blocks_per_expert)]
    )
    block_idx = jnp.arange(max_blocks, dtype=jnp.int32)
    expert_ids = jnp.searchsorted(block_offsets[1:], block_idx, side="right").astype(
        jnp.int32
    )
    total_blocks = block_offsets[-1]
    expert_ids = jnp.where(block_idx < total_blocks, expert_ids, jnp.int32(-1))

    num_valid_ids = jnp.array([padded_counts.sum()], dtype=jnp.int32)
    return sorted_ids, sorted_weights, expert_ids, num_valid_ids


def moe_block_fly_bf16_atomic(
    x: jax.Array,
    fly_params: dict,
    sorted_ids: jax.Array,
    expert_ids: jax.Array,
    sorted_weights: jax.Array,
    num_valid_ids: jax.Array,
    *,
    tokens: int,
    model_dim: int,
    inter_dim: int,
    num_experts: int,
    topk: int,
    tile_m: int | None = None,
    stage2_mode: str | None = None,
) -> jax.Array:
    """FlyDSL MoE block, bf16 in -> bf16 out, stage-2 atomic-add reduction."""
    tile_m = tile_m if tile_m is not None else pick_tile_m(tokens)
    if stage2_mode is None:
        stage2_mode = os.environ.get("FLY_STAGE2", "atomic")

    out1_f16 = moe_gemm1_jax(
        x,
        fly_params["w1_shuffled"],
        jnp.ones((tokens,), jnp.float32),
        fly_params["scale_w1"],
        sorted_ids,
        expert_ids,
        sorted_weights,
        num_valid_ids,
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=num_experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=128,
        tile_k=128,
        in_dtype="bf16",
        out_dtype="f16",
        # FlyDSL 0.2.0's stage-1 bf16 CShuffle epilogue fails MLIR verification;
        # the "direct" epilogue is numerically equivalent. Stage 2 keeps CShuffle
        # (required for f16 output). Drop this once the wheel pin fixes it.
        use_cshuffle_epilog=False,
    )
    out1_bf16 = out1_f16.astype(jnp.bfloat16)

    stage2_kw = dict(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=num_experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=256,
        tile_k=128,
        in_dtype="bf16",
        out_dtype="f16",
    )
    stage2_in = (
        out1_bf16.reshape(tokens * topk, inter_dim),
        fly_params["w2_shuffled"],
        jnp.ones((tokens * topk,), jnp.float32),
        fly_params["scale_w2"],
        sorted_ids,
        expert_ids,
        sorted_weights,
        num_valid_ids,
    )
    if stage2_mode == "reduce":
        out2_f16 = moe_gemm2_reduce_jax(*stage2_in, **stage2_kw)
    else:
        out2_f16 = moe_gemm2_jax(*stage2_in, **stage2_kw)
    return out2_f16.astype(jnp.bfloat16)
