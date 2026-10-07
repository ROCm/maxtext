# Copyright 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""JAX-AITER MXFP4 backend for one local routed-expert shard.

MaxText owns routing and the custom training graph. This module only adapts its
compact expert-sorted rows to the collaborator kernel contract and supplies the
explicit grouped-GEMM backward rule.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec

# Row alignment of each expert segment, per grouped-GEMM kernel family. The
# whole-loop weight gradient walks each expert in 256-row contraction steps.
_GROUP_ALIGNMENT = {"ragged": 128, "wholeloop": 256}
_DENSE_BATCH_MESH_AXES = ("data", "fsdp", "fsdp_transpose", "expert")


def _group_alignment(grouped_kernel: str) -> int:
  if grouped_kernel not in _GROUP_ALIGNMENT:
    raise ValueError(
        f"Unknown jax_aiter_mxfp4 grouped kernel {grouped_kernel!r}; "
        f"expected one of {sorted(_GROUP_ALIGNMENT)}"
    )
  return _GROUP_ALIGNMENT[grouped_kernel]


def _check_no_overflow(total_tokens, buffer_capacity):
  """Abort if routed tokens exceed the dispatch buffer (token truncation)."""
  total = int(total_tokens)
  cap = int(buffer_capacity)
  if total > cap:
    raise RuntimeError(
        f"MXFP4 route-capacity overflow: {total} routed tokens exceed "
        f"buffer capacity {cap}. Increase ragged_buffer_factor or use "
        f"worst-case sizing (ragged_buffer_factor=-1)."
    )


class ExpertPadding(NamedTuple):
  """Static-capacity mapping between compact and per-expert padded rows."""

  group_sizes: jax.Array
  group_end_offsets: jax.Array
  compact_to_padded: jax.Array
  compact_valid: jax.Array
  padded_to_compact: jax.Array
  padded_valid: jax.Array


def _round_up_groups(group_sizes: jax.Array, alignment: int) -> jax.Array:
  group_sizes = group_sizes.astype(jnp.int32)
  return jnp.where(
      group_sizes == 0,
      0,
      ((group_sizes + alignment - 1) // alignment) * alignment,
  )


def _gather_rows(
    values: jax.Array, indices: jax.Array, valid: jax.Array
) -> jax.Array:
  gathered = values[indices]
  return jnp.where(valid[:, None], gathered, jnp.zeros((), values.dtype))


@jax.custom_vjp
def _gather_pad_values(
    values: jax.Array,
    compact_to_padded: jax.Array,
    compact_valid: jax.Array,
    padded_to_compact: jax.Array,
    padded_valid: jax.Array,
) -> jax.Array:
  return _gather_rows(values, padded_to_compact, padded_valid)


def _gather_pad_values_fwd(
    values,
    compact_to_padded,
    compact_valid,
    padded_to_compact,
    padded_valid,
):
  output = _gather_rows(values, padded_to_compact, padded_valid)
  return output, (compact_to_padded, compact_valid)


def _gather_pad_values_bwd(residuals, grad_output):
  compact_to_padded, compact_valid = residuals
  grad_values = _gather_rows(
      grad_output, compact_to_padded, compact_valid
  )
  return grad_values, None, None, None, None


_gather_pad_values.defvjp(
    _gather_pad_values_fwd, _gather_pad_values_bwd
)


@jax.custom_vjp
def _gather_unpad_values(
    values: jax.Array,
    compact_to_padded: jax.Array,
    compact_valid: jax.Array,
    padded_to_compact: jax.Array,
    padded_valid: jax.Array,
) -> jax.Array:
  return _gather_rows(values, compact_to_padded, compact_valid)


def _gather_unpad_values_fwd(
    values,
    compact_to_padded,
    compact_valid,
    padded_to_compact,
    padded_valid,
):
  output = _gather_rows(values, compact_to_padded, compact_valid)
  return output, (padded_to_compact, padded_valid)


def _gather_unpad_values_bwd(residuals, grad_output):
  padded_to_compact, padded_valid = residuals
  grad_values = _gather_rows(
      grad_output, padded_to_compact, padded_valid
  )
  return grad_values, None, None, None, None


_gather_unpad_values.defvjp(
    _gather_unpad_values_fwd, _gather_unpad_values_bwd
)


def pad_expert_rows(
    values: jax.Array,
    group_sizes: jax.Array,
    grouped_kernel: str = "ragged",
    check_route_capacity: bool = True,
) -> tuple[jax.Array, ExpertPadding]:
  """Pad each compact expert segment to the grouped kernel's row alignment.

  The padding needs no host synchronization when ``check_route_capacity`` is
  false. Keep the check enabled until the configured route buffer has been
  validated for the workload.
  """
  if values.ndim != 2:
    raise ValueError(f"values must be rank 2, got {values.shape}")
  if check_route_capacity:
    jax.debug.callback(
        _check_no_overflow,
        jnp.sum(group_sizes.astype(jnp.int32)),
        jnp.int32(values.shape[0]),
    )
  padding = expert_padding(group_sizes, values.shape[0], grouped_kernel)
  padded = _gather_pad_values(
      values,
      padding.compact_to_padded,
      padding.compact_valid,
      padding.padded_to_compact,
      padding.padded_valid,
  )
  return padded, padding


def max_compact_capacity(cols: int, num_experts: int, grouped_kernel: str = "ragged") -> int:
  """Largest `expert_padding` compact capacity whose padded `[rows, cols]` buffer has int32-indexable elements.

  The JAX-AITER quantizer indexes its operand with signed int32.
  """
  alignment = _group_alignment(grouped_kernel)
  padded_rows = (2**31 - 1) // cols // alignment * alignment
  return padded_rows - (alignment - 1) * num_experts


def expert_padding(
    group_sizes: jax.Array,
    compact_capacity: int,
    grouped_kernel: str = "ragged",
) -> ExpertPadding:
  """Row mapping of `pad_expert_rows` for `compact_capacity` expert-sorted rows."""
  if group_sizes.ndim != 1:
    raise ValueError(f"group_sizes must be rank 1, got {group_sizes.shape}")

  alignment = _group_alignment(grouped_kernel)
  num_experts = group_sizes.shape[0]
  padded_capacity = (
      compact_capacity + (alignment - 1) * num_experts + alignment - 1
  ) // alignment * alignment

  group_sizes = group_sizes.astype(jnp.int32)
  compact_ends = jnp.cumsum(group_sizes, dtype=jnp.int32)
  compact_starts = compact_ends - group_sizes
  padded_sizes = _round_up_groups(group_sizes, alignment)
  padded_ends = jnp.cumsum(padded_sizes, dtype=jnp.int32)
  padded_starts = padded_ends - padded_sizes

  compact_rows = jnp.arange(compact_capacity, dtype=jnp.int32)
  compact_expert = jnp.sum(
      compact_rows[:, None] >= compact_ends[None, :], axis=1, dtype=jnp.int32
  )
  compact_valid = compact_expert < num_experts
  compact_safe_expert = jnp.minimum(compact_expert, num_experts - 1)
  compact_to_padded = (
      padded_starts[compact_safe_expert]
      + compact_rows
      - compact_starts[compact_safe_expert]
  )
  compact_to_padded = jnp.where(
      compact_valid, compact_to_padded, 0
  )

  padded_rows = jnp.arange(padded_capacity, dtype=jnp.int32)
  padded_expert = jnp.sum(
      padded_rows[:, None] >= padded_ends[None, :],
      axis=1,
      dtype=jnp.int32,
  )
  padded_safe_expert = jnp.minimum(padded_expert, num_experts - 1)
  padded_row_in_expert = padded_rows - padded_starts[padded_safe_expert]
  padded_valid = (padded_expert < num_experts) & (
      padded_row_in_expert < group_sizes[padded_safe_expert]
  )
  padded_to_compact = (
      compact_starts[padded_safe_expert] + padded_row_in_expert
  )
  padded_to_compact = jnp.where(padded_valid, padded_to_compact, 0)

  return ExpertPadding(
      group_sizes=padded_sizes,
      group_end_offsets=padded_ends,
      compact_to_padded=compact_to_padded,
      compact_valid=compact_valid,
      padded_to_compact=padded_to_compact,
      padded_valid=padded_valid,
  )


def unpad_expert_rows(values: jax.Array, padding: ExpertPadding) -> jax.Array:
  """Restore MaxText's compact fixed-capacity row layout."""
  return _gather_unpad_values(
      values,
      padding.compact_to_padded,
      padding.compact_valid,
      padding.padded_to_compact,
      padding.padded_valid,
  )


def _group_offsets(group_sizes: jax.Array, grouped_kernel: str) -> jax.Array:
  """E cumulative ends for the ragged kernels, [0, ends...] for whole-loop."""
  ends = jnp.cumsum(group_sizes.astype(jnp.int32), dtype=jnp.int32)
  if grouped_kernel == "ragged":
    return ends
  return jnp.concatenate((jnp.zeros((1,), jnp.int32), ends))


def _check_grouped_operands(lhs, rhs, group_sizes):
  if lhs.ndim != 2 or rhs.ndim != 3:
    raise ValueError(f"Expected lhs[M,K] and rhs[E,K,N], got {lhs.shape}, {rhs.shape}")
  if lhs.dtype != jnp.bfloat16 or rhs.dtype != jnp.bfloat16:
    raise TypeError("JAX-AITER MXFP4 grouped GEMM requires bfloat16 operands")
  if group_sizes.shape != (rhs.shape[0],):
    raise ValueError("group_sizes must contain one value per local expert")


def _quantize_rhs(rhs: jax.Array, *, transpose_rhs: bool):
  """Quantize rhs[E,K,N] to the kernel's [E, out, contraction/2] layout."""
  from jax_aiter.ops.moe_mxfp4 import quantize_mxfp4_dim0

  if transpose_rhs:
    # lhs[M,N] @ rhs[E,K,N]^T -> out[M,K]
    rhs_for_kernel = rhs
  else:
    # lhs[M,K] @ rhs[E,K,N] -> out[M,N]
    rhs_for_kernel = rhs.swapaxes(1, 2)
  experts, output_size, contraction = rhs_for_kernel.shape
  rhs_packed, rhs_scales = quantize_mxfp4_dim0(
      rhs_for_kernel.reshape(experts * output_size, contraction)
  )
  return (
      rhs_packed.reshape(experts, output_size, contraction // 2),
      rhs_scales.reshape(experts, output_size, contraction // 32),
  )


def _grouped_gmm_quantized_rhs(
    lhs: jax.Array,
    rhs_packed: jax.Array,
    rhs_scales: jax.Array,
    group_sizes: jax.Array,
    *,
    grouped_kernel: str,
    zero_tail: bool = True,
) -> jax.Array:
  from jax_aiter.ops.moe_mxfp4 import (
      grouped_mxfp4_fwd_da,
      grouped_mxfp4_fwd_da_wholeloop,
      quantize_mxfp4_dim0,
  )

  contraction = rhs_packed.shape[2] * 2
  if lhs.shape[1] != contraction:
    raise ValueError(
        f"Grouped contraction mismatch: lhs {lhs.shape}, "
        f"quantized rhs {rhs_packed.shape}"
    )
  lhs_packed, lhs_scales = quantize_mxfp4_dim0(lhs)
  offsets = _group_offsets(group_sizes, grouped_kernel)
  if grouped_kernel == "ragged":
    return grouped_mxfp4_fwd_da(lhs_packed, rhs_packed, lhs_scales, rhs_scales, offsets)
  return grouped_mxfp4_fwd_da_wholeloop(
      lhs_packed, rhs_packed, lhs_scales, rhs_scales, offsets, zero_tail=zero_tail
  )


def _grouped_gmm(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    transpose_rhs: bool,
    grouped_kernel: str,
    zero_tail: bool = True,
) -> jax.Array:
  _check_grouped_operands(lhs, rhs, group_sizes)
  rhs_packed, rhs_scales = _quantize_rhs(rhs, transpose_rhs=transpose_rhs)
  return _grouped_gmm_quantized_rhs(
      lhs,
      rhs_packed,
      rhs_scales,
      group_sizes,
      grouped_kernel=grouped_kernel,
      zero_tail=zero_tail,
  )


def _grouped_weight_grad(
    lhs: jax.Array,
    grad_output: jax.Array,
    group_sizes: jax.Array,
    grouped_kernel: str,
) -> jax.Array:
  """d(lhs @ rhs)/d rhs for lhs[M,K] and grad_output[M,N], as [E,K,N]."""
  from jax_aiter.ops.moe_mxfp4 import (
      grouped_mxfp4_dw,
      grouped_mxfp4_dw_wholeloop,
      quantize_mxfp4_dim1,
  )

  grad_transposed, grad_scales = quantize_mxfp4_dim1(grad_output)
  lhs_transposed, lhs_scales = quantize_mxfp4_dim1(lhs)
  weight_grad = (
      grouped_mxfp4_dw
      if grouped_kernel == "ragged"
      else grouped_mxfp4_dw_wholeloop
  )
  # Raw dW is [E,N,K]; MaxText stores expert weights as [E,K,N].
  return weight_grad(
      grad_transposed,
      grad_scales,
      lhs_transposed,
      lhs_scales,
      _group_offsets(group_sizes, grouped_kernel),
  ).swapaxes(1, 2)


def _make_grouped_gmm(grouped_kernel: str, zero_tail: bool):
  @jax.custom_vjp
  def grouped_gmm_for_kernel(
      lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array
  ) -> jax.Array:
    return _grouped_gmm(
        lhs,
        rhs,
        group_sizes,
        transpose_rhs=False,
        grouped_kernel=grouped_kernel,
        zero_tail=zero_tail,
    )

  def forward(lhs, rhs, group_sizes):
    output = _grouped_gmm(
        lhs,
        rhs,
        group_sizes,
        transpose_rhs=False,
        grouped_kernel=grouped_kernel,
        zero_tail=zero_tail,
    )
    return output, (lhs, rhs, group_sizes)

  def backward(residuals, grad_output):
    lhs, rhs, group_sizes = residuals
    grad_output = grad_output.astype(jnp.bfloat16)
    grad_lhs = _grouped_gmm(
        grad_output,
        rhs,
        group_sizes,
        transpose_rhs=True,
        grouped_kernel=grouped_kernel,
        zero_tail=zero_tail,
    )
    grad_rhs = _grouped_weight_grad(lhs, grad_output, group_sizes, grouped_kernel)
    return grad_lhs, grad_rhs, None

  grouped_gmm_for_kernel.defvjp(forward, backward)
  return grouped_gmm_for_kernel


_GROUPED_GMM = {
    (kernel, zero_tail): _make_grouped_gmm(kernel, zero_tail)
    for kernel in _GROUP_ALIGNMENT
    for zero_tail in (True, False)
}


def _gate_up_forward(lhs, w0, w1, group_sizes, grouped_kernel, zero_tail):
  _check_grouped_operands(lhs, w0, group_sizes)
  _check_grouped_operands(lhs, w1, group_sizes)
  if w0.shape != w1.shape:
    raise ValueError(f"Gate and up weights differ: {w0.shape} vs {w1.shape}")
  # The concatenate fuses into the transpose that already feeds the quantizer.
  return _grouped_gmm(
      lhs,
      jnp.concatenate((w0, w1), axis=2),
      group_sizes,
      transpose_rhs=False,
      grouped_kernel=grouped_kernel,
      zero_tail=zero_tail,
  )


def _make_grouped_gmm_gate_up(grouped_kernel: str, zero_tail: bool):
  @jax.custom_vjp
  def gate_up_for_kernel(lhs, w0, w1, group_sizes):
    return _gate_up_forward(lhs, w0, w1, group_sizes, grouped_kernel, zero_tail)

  def forward(lhs, w0, w1, group_sizes):
    output = _gate_up_forward(lhs, w0, w1, group_sizes, grouped_kernel, zero_tail)
    return output, (lhs, w0, w1, group_sizes)

  def backward(residuals, grad_output):
    lhs, w0, w1, group_sizes = residuals
    grad_output = grad_output.astype(jnp.bfloat16)
    # Each MXFP4 row block covers 32 contraction values, so quantizing w0 and
    # w1 separately and concatenating along the contraction axis is
    # bit-identical to quantizing their concatenation, without a bf16 copy.
    packed0, scales0 = _quantize_rhs(w0, transpose_rhs=True)
    packed1, scales1 = _quantize_rhs(w1, transpose_rhs=True)
    grad_lhs = _grouped_gmm_quantized_rhs(
        grad_output,
        jnp.concatenate((packed0, packed1), axis=2),
        jnp.concatenate((scales0, scales1), axis=2),
        group_sizes,
        grouped_kernel=grouped_kernel,
        zero_tail=zero_tail,
    )
    grad_w = _grouped_weight_grad(lhs, grad_output, group_sizes, grouped_kernel)
    grad_w0, grad_w1 = jax.lax.split(grad_w, (w0.shape[2], w1.shape[2]), axis=2)
    return grad_lhs, grad_w0, grad_w1, None

  gate_up_for_kernel.defvjp(forward, backward)
  return gate_up_for_kernel


_GROUPED_GMM_GATE_UP = {
    (kernel, zero_tail): _make_grouped_gmm_gate_up(kernel, zero_tail)
    for kernel in _GROUP_ALIGNMENT
    for zero_tail in (True, False)
}


def grouped_gmm(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    grouped_kernel: str = "ragged",
    zero_tail: bool = True,
) -> jax.Array:
  """Differentiable ``lhs[group] @ rhs[group]`` in dynamic MXFP4.

  ``group_sizes`` must be multiples of the kernel's row alignment, as
  produced by :func:`pad_expert_rows` with the same ``grouped_kernel``.

  Rows past the last group are zero when ``zero_tail`` is true, in the output
  and in the input gradient. When false, the whole-loop kernel leaves them
  uninitialized: a full-buffer pass is saved, so pass false only when every
  reader of those rows masks them with a select.
  """
  _group_alignment(grouped_kernel)
  return _GROUPED_GMM[grouped_kernel, zero_tail](lhs, rhs, group_sizes)


def grouped_gmm_gate_up(
    lhs: jax.Array,
    w0: jax.Array,
    w1: jax.Array,
    group_sizes: jax.Array,
    grouped_kernel: str = "ragged",
    zero_tail: bool = True,
) -> jax.Array:
  """Gate and up projections as one grouped GEMM, returning ``[M, 2N]``.

  Columns ``[:N]`` equal ``grouped_gmm(lhs, w0)`` and ``[N:]`` equal
  ``grouped_gmm(lhs, w1)``. The backward pass computes the input gradient with
  one GEMM over the combined contraction, so the two partial input gradients
  are never materialized and summed. ``zero_tail`` is as in
  :func:`grouped_gmm`.
  """
  _group_alignment(grouped_kernel)
  return _GROUPED_GMM_GATE_UP[grouped_kernel, zero_tail](lhs, w0, w1, group_sizes)


def dense_linear(
    lhs: jax.Array,
    rhs: jax.Array,
    mesh: Mesh,
) -> jax.Array:
  """Differentiable E=1 whole-loop MXFP4 linear with EP-local execution.

  ``lhs`` and the result are sharded over the active data-parallel mesh axes.
  ``rhs`` is replicated because its contraction dimension must be complete on
  every rank.  The transpose of ``shard_map`` sums the replicated weight's
  per-rank gradients over those same axes.
  """
  if lhs.ndim != 2 or rhs.ndim != 2:
    raise ValueError(f"Expected lhs[M,K] and rhs[K,N], got {lhs.shape}, {rhs.shape}")
  if lhs.dtype != jnp.bfloat16 or rhs.dtype != jnp.bfloat16:
    raise TypeError("JAX-AITER MXFP4 dense linear requires bfloat16 operands")
  rows, contraction = lhs.shape
  if rhs.shape[0] != contraction:
    raise ValueError(f"Dense contraction mismatch: lhs {lhs.shape}, rhs {rhs.shape}")

  active_batch_axes = tuple(
      axis for axis in _DENSE_BATCH_MESH_AXES if mesh.shape.get(axis, 1) > 1
  )
  batch_shards = 1
  for axis in active_batch_axes:
    batch_shards *= mesh.shape[axis]
  if rows % batch_shards:
    raise ValueError(
        f"Dense rows {rows} are not divisible by batch sharding {batch_shards}"
    )
  local_rows = rows // batch_shards
  output = rhs.shape[1]
  if local_rows % 256 or contraction % 256 or output % 256:
    raise ValueError(
        "Whole-loop MXFP4 dense linear requires local M, K, and N divisible "
        f"by 256; got local M={local_rows}, K={contraction}, N={output}"
    )

  if not active_batch_axes:
    batch_axis = None
  elif len(active_batch_axes) == 1:
    batch_axis = active_batch_axes[0]
  else:
    batch_axis = active_batch_axes

  def local_linear(local_lhs, local_rhs):
    group_sizes = jnp.asarray([local_lhs.shape[0]], dtype=jnp.int32)
    return grouped_gmm(
        local_lhs,
        local_rhs[None, :, :],
        group_sizes,
        grouped_kernel="wholeloop",
    )

  return jax.shard_map(
      local_linear,
      mesh=mesh,
      in_specs=(PartitionSpec(batch_axis, None), PartitionSpec(None, None)),
      out_specs=PartitionSpec(batch_axis, None),
      check_vma=False,
  )(lhs, rhs)


__all__ = [
    "ExpertPadding",
    "dense_linear",
    "grouped_gmm",
    "pad_expert_rows",
    "unpad_expert_rows",
]
