# SPDX-License-Identifier: Apache-2.0
"""JAX calls into the FlyDSL MoE 2-stage GEMM kernels (via flydsl_call).

Requires kernels.moe_gemm_2stage to be importable (see flydsl/__init__.py).
"""

from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp


def _detect_kernel() -> bool:
    """True if kernels.moe_gemm_2stage is importable (checked lazily, not at import)."""
    try:
        return importlib.util.find_spec("kernels.moe_gemm_2stage") is not None
    except (ImportError, ValueError):
        return False


_HAS_KERNEL = _detect_kernel()


def moe_gemm_available() -> bool:
    """True if kernels/moe_gemm_2stage.py is on PYTHONPATH."""
    return _HAS_KERNEL


def moe_gemm1_jax(
    x_q: jax.Array,
    w1_shuffled: jax.Array,
    scale_x: jax.Array,
    scale_w1: jax.Array,
    sorted_ids: jax.Array,
    expert_ids: jax.Array,
    sorted_weights: jax.Array,
    num_valid_ids: jax.Array,
    *,
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    in_dtype: str = "fp8",
    out_dtype: str = "f16",
    doweight_stage1: bool = False,
    use_cshuffle_epilog: bool | None = None,
) -> jax.Array:
    """Call FlyDSL moe_gemm1 (gate + up matmul + gated activation) from JAX."""
    from jax_flydsl.flydsl_lib import flydsl_call
    from kernels.moe_gemm_2stage import compile_moe_gemm1

    blocks = int(expert_ids.shape[0])
    out_dtype_jnp = jnp.float16 if out_dtype == "f16" else jnp.bfloat16

    launcher = compile_moe_gemm1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=doweight_stage1,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        use_cshuffle_epilog=use_cshuffle_epilog,
    )

    return flydsl_call(
        x_q,
        w1_shuffled,
        scale_x,
        scale_w1,
        sorted_ids,
        expert_ids,
        sorted_weights,
        num_valid_ids,
        kernel=launcher,
        out_shape=jax.ShapeDtypeStruct((tokens, topk, inter_dim), out_dtype_jnp),
        scalars={
            "i32_tokens_in": tokens,
            "i32_inter_in": inter_dim,
            "i32_k_in": model_dim,
            "i32_size_expert_ids_in": blocks,
        },
        output_positions=[0],
    )


def moe_gemm2_jax(
    a2_q: jax.Array,
    w2_shuffled: jax.Array,
    scale_a2: jax.Array,
    scale_w2: jax.Array,
    sorted_ids: jax.Array,
    expert_ids: jax.Array,
    sorted_weights: jax.Array,
    num_valid_ids: jax.Array,
    *,
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    in_dtype: str = "fp8",
    out_dtype: str = "f16",
    doweight_stage2: bool = True,
) -> jax.Array:
    """Call FlyDSL moe_gemm2 (down matmul + atomic reduce) from JAX."""
    from jax_flydsl.flydsl_lib import flydsl_call
    from kernels.moe_gemm_2stage import compile_moe_gemm2

    blocks = int(expert_ids.shape[0])
    out_dtype_jnp = jnp.float16 if out_dtype == "f16" else jnp.bfloat16

    launcher = compile_moe_gemm2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=doweight_stage2,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accumulate=True,
        use_cshuffle_epilog=None,
    )

    out_zeroed = jnp.zeros((tokens, model_dim), dtype=out_dtype_jnp)
    return flydsl_call(
        a2_q,
        w2_shuffled,
        scale_a2,
        scale_w2,
        sorted_ids,
        expert_ids,
        sorted_weights,
        num_valid_ids,
        out_zeroed,
        kernel=launcher,
        out_shape=jax.ShapeDtypeStruct((tokens, model_dim), out_dtype_jnp),
        scalars={
            "i32_tokens_in": tokens,
            "i32_n_in": model_dim,
            "i32_k_in": inter_dim,
            "i32_size_expert_ids_in": blocks,
        },
        output_positions=[0],
        input_output_aliases={8: 0},
    )


def moe_gemm2_reduce_jax(
    a2_q: jax.Array,
    w2_shuffled: jax.Array,
    scale_a2: jax.Array,
    scale_w2: jax.Array,
    sorted_ids: jax.Array,
    expert_ids: jax.Array,
    sorted_weights: jax.Array,
    num_valid_ids: jax.Array,
    *,
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    in_dtype: str = "fp8",
    out_dtype: str = "f16",
    doweight_stage2: bool = True,
) -> jax.Array:
    """Call FlyDSL moe_gemm2 in REDUCE mode (no atomics + JAX-side sum)."""
    from jax_flydsl.flydsl_lib import flydsl_call
    from kernels.moe_gemm_2stage import compile_moe_gemm2

    blocks = int(expert_ids.shape[0])
    out_dtype_jnp = jnp.float16 if out_dtype == "f16" else jnp.bfloat16

    launcher = compile_moe_gemm2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=doweight_stage2,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accumulate=False,
        use_cshuffle_epilog=None,
    )

    intermediate_zeroed = jnp.zeros((tokens * topk, model_dim), dtype=out_dtype_jnp)
    intermediate = flydsl_call(
        a2_q,
        w2_shuffled,
        scale_a2,
        scale_w2,
        sorted_ids,
        expert_ids,
        sorted_weights,
        num_valid_ids,
        intermediate_zeroed,
        kernel=launcher,
        out_shape=jax.ShapeDtypeStruct((tokens * topk, model_dim), out_dtype_jnp),
        scalars={
            "i32_tokens_in": tokens,
            "i32_n_in": model_dim,
            "i32_k_in": inter_dim,
            "i32_size_expert_ids_in": blocks,
        },
        output_positions=[0],
        input_output_aliases={8: 0},
    )

    return intermediate.reshape(tokens, topk, model_dim).sum(axis=1)
