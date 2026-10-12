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

  MaxText stores every expert kernel contracting-dim-first -- wi_* as [E, D, M]
  and wo as [E, M, D] -- which is the layout the op takes, so the weight goes
  through untouched. Turning it here instead would cost a full copy of it on
  every step: the op needs both orientations anyway and turns the narrower fp8
  payload internally.

  The grouped GEMM writes each group and nothing else, so when the token buffer
  has more rows than the groups fill, as it does whenever ragged_buffer_factor
  gives a fixed-capacity receiver, the tail holds whatever the allocator left
  there. Masking it here rather than at the far end of the FFN keeps that
  garbage from reaching the next quantizer, which would otherwise pick its block
  scales off values no expert ever produced.

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
      kernel.astype(x.dtype),
      group_offsets,
  )
  return _mask_ragged_tail(out, group_offsets)


def grouped_gemm_mxfp8_gate_up(
    x: jax.Array,
    w0: jax.Array,
    w1: jax.Array,
    group_offsets: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Gate and up projections, with ``x`` quantized once. Differentiable.

  Same contraction as two :func:`grouped_gemm_mxfp8_experts` calls. The
  activation quantizer runs once and both GEMMs read it; each weight still
  quantizes on its own.

  Returns:
    ``(gate, up)``, each ``[tokens, N]`` in the dtype of ``x``, tails zeroed.
  """
  from jax_flydsl.ops.grouped_gemm_mxfp8 import grouped_gemm_mxfp8_gate_up as _gate_up

  if x.ndim != 2:
    raise ValueError(f"grouped_gemm_mxfp8_gate_up: x must be 2D [tokens, K], got {x.shape}")
  for name, kernel in (("w0", w0), ("w1", w1)):
    if kernel.ndim != 3:
      raise ValueError(f"grouped_gemm_mxfp8_gate_up: {name} must be 3D [G, K, N], got {kernel.shape}")

  gate, up = _gate_up(x, w0.astype(x.dtype), w1.astype(x.dtype), group_offsets)
  return _mask_ragged_tail(gate, group_offsets), _mask_ragged_tail(up, group_offsets)


def _mask_ragged_tail(out: jax.Array, group_offsets: jax.Array) -> jax.Array:
  """Zero rows past the last real group. The GEMM leaves that tail untouched."""
  live = jnp.arange(out.shape[0], dtype=jnp.int32)[:, None] < group_offsets[-1]
  return jnp.where(live, out, 0).astype(out.dtype)


def grouped_gemm_mxfp8_glu(
    x: jax.Array,
    w_packed: jax.Array,
    group_offsets: jax.Array,
    clamp_limit: float | None,
) -> jax.Array:
  """``silu(gate) * up`` from one GEMM. Differentiable.

  ``w_packed`` is ``[G, K, 2I]`` already in tile order. The kernel applies SiLU
  in the epilogue and writes ``[tokens, I]``. The tail past the last real group
  is zeroed, same as the other grouped GEMMs.
  """
  from jax_flydsl.ops.grouped_gemm_mxfp8 import grouped_gemm_mxfp8_glu as _glu

  if x.ndim != 2:
    raise ValueError(f"grouped_gemm_mxfp8_glu: x must be 2D [tokens, K], got {x.shape}")
  if w_packed.ndim != 3:
    raise ValueError(f"grouped_gemm_mxfp8_glu: weight must be 3D [G, K, 2I], got {w_packed.shape}")

  out = _glu(x, w_packed.astype(x.dtype), group_offsets, clamp_limit)
  return _mask_ragged_tail(out, group_offsets)


def grouped_glu_down_mxfp8_experts(
    l1: jax.Array,
    kernel: jax.Array,
    group_offsets: jax.Array,
    clamp_limit: float | None,
) -> jax.Array:
  """The FFN's activation and down projection in one op. Differentiable.

  ``l1`` is ``[gate || up]`` from a fused gate/up grouped GEMM. The GLU and both
  MXFP8 orientations of its result come out of one kernel reading ``l1`` once,
  in place of ``apply_ffn_activation`` materializing the intermediate and the
  down GEMM quantizing it again.

  Router weights stay in MaxText's ``unpermute``, as on the unfused path.

  Args:
    l1: [tokens, 2 * I], this shard's tokens stacked by local expert.
    kernel: [G, I, D] MaxText ``wo``.
    group_offsets: [G + 1], from :func:`group_offsets_from_sizes`.
    clamp_limit: DeepSeek's activation bound, or None.

  Returns:
    [tokens, D] in the dtype of ``l1``, with rows past the last group zeroed as
    in :func:`grouped_gemm_mxfp8_experts`.
  """
  from jax_flydsl.ops.glu_mlp import grouped_glu_down_mxfp8

  if l1.ndim != 2:
    raise ValueError(f"grouped_glu_down_mxfp8_experts: l1 must be 2D [tokens, 2*I], got {l1.shape}")
  if kernel.ndim != 3:
    raise ValueError(f"grouped_glu_down_mxfp8_experts: kernel must be 3D [G, I, D], got {kernel.shape}")

  out = grouped_glu_down_mxfp8(l1, kernel.astype(l1.dtype), group_offsets, clamp_limit)
  live = jnp.arange(out.shape[0], dtype=jnp.int32)[:, None] < group_offsets[-1]
  return jnp.where(live, out, 0).astype(out.dtype)
