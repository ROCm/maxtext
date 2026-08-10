# SPDX-License-Identifier: Apache-2.0
"""Adapter between MaxText routed-MoE and the FlyDSL grouped MXFP8 GEMM.

Substitutes for the three per-expert matmuls of the `sparse_matmul` path.
Routing, the activation and the unpermute stay MaxText's own; only the
contractions move, and they stay differentiable because the operation carries
its own `custom_vjp`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def group_offsets_from_sizes(group_sizes: jax.Array) -> jax.Array:
  """Per-expert token counts [G] -> the [G+1] offsets the grouped GEMM reads."""
  sizes = group_sizes.astype(jnp.int32)
  return jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(sizes)])


def grouped_gemm_mxfp8_experts(
    x: jax.Array,
    kernel: jax.Array,
    group_offsets: jax.Array,
) -> jax.Array:
  """One per-expert matmul of the MoE FFN, contracted in MXFP8. Differentiable.

  Two layout notes. MaxText stores every expert kernel contracting-dim-first --
  wi_* as [E, D, M] and wo as [E, M, D] -- while the kernel declares its weights
  free-dim-first, so each one is transposed on the way in. And the grouped GEMM
  writes each group and nothing else, so when the token buffer has more rows
  than the groups fill, as it does whenever ragged_buffer_factor gives a
  fixed-capacity receiver, the tail holds whatever the allocator left there.
  Masking it here rather than at the far end of the FFN keeps that garbage from
  reaching the next quantizer, which would otherwise pick its block scales off
  values no expert ever produced.

  Args:
    x: [tokens, K] this shard's tokens, already stacked by local expert.
    kernel: [G, K, N] MaxText expert weight -- wi_0, wi_1, or wo.
    group_offsets: [G + 1], from :func:`group_offsets_from_sizes`.

  Returns:
    [tokens, N] in the dtype of ``x``.
  """
  from jax_flydsl.ops import grouped_gemm_mxfp8

  if x.ndim != 2:
    raise ValueError(f"grouped_gemm_mxfp8_experts: x must be 2D [tokens, K], got {x.shape}")
  if kernel.ndim != 3:
    raise ValueError(f"grouped_gemm_mxfp8_experts: kernel must be 3D [G, K, N], got {kernel.shape}")

  out = grouped_gemm_mxfp8(
      x,
      jnp.swapaxes(kernel, 1, 2).astype(x.dtype),
      group_offsets,
  )
  live = jnp.arange(out.shape[0], dtype=jnp.int32)[:, None] < group_offsets[-1]
  return jnp.where(live, out, 0).astype(out.dtype)
