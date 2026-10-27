# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# JAX-AITER is an optional ROCm package, imported where it is used.
# pylint: disable=import-outside-toplevel
"""JAX-AITER MXFP4 backend for one local routed-expert shard.

MaxText owns routing and the custom training graph. This module only adapts its
compact expert-sorted rows to the JAX-AITER whole-loop kernel contract and
supplies the explicit grouped-GEMM backward rule.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import numpy as np
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh, PartitionSpec

# Row alignment of each expert segment. The whole-loop weight gradient walks
# each expert in 256-row contraction steps.
_GROUP_ALIGNMENT = 256
_DENSE_BATCH_MESH_AXES = ("data", "fsdp", "fsdp_transpose", "expert")


def padded_feature_dim(size: int) -> int:
  """Smallest K or N >= ``size`` that the grouped kernels accept.

  The input gradient contracts over the forward output dimension, so both
  weight dimensions need the forward contraction constraint, K % 256.
  """
  return -(-size // _GROUP_ALIGNMENT) * _GROUP_ALIGNMENT


def _pad_axis(value: jax.Array, axis: int, size: int) -> jax.Array:
  extra = size - value.shape[axis]
  if extra == 0:
    return value
  widths = [(0, 0)] * value.ndim
  widths[axis] = (0, extra)
  return jnp.pad(value, widths)


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


def _gather_rows(values: jax.Array, indices: jax.Array, valid: jax.Array) -> jax.Array:
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
  grad_values = _gather_rows(grad_output, compact_to_padded, compact_valid)
  return grad_values, None, None, None, None


_gather_pad_values.defvjp(_gather_pad_values_fwd, _gather_pad_values_bwd)


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
  grad_values = _gather_rows(grad_output, padded_to_compact, padded_valid)
  return grad_values, None, None, None, None


_gather_unpad_values.defvjp(_gather_unpad_values_fwd, _gather_unpad_values_bwd)


def pad_expert_rows(
    values: jax.Array,
    group_sizes: jax.Array,
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
  padding = expert_padding(group_sizes, values.shape[0])
  padded = _gather_pad_values(
      values,
      padding.compact_to_padded,
      padding.compact_valid,
      padding.padded_to_compact,
      padding.padded_valid,
  )
  return padded, padding


def max_compact_capacity(cols: int, num_experts: int) -> int:
  """Largest `expert_padding` compact capacity whose padded `[rows, cols]` buffer has int32-indexable elements.

  The JAX-AITER quantizer indexes its operand with signed int32.
  """
  alignment = _GROUP_ALIGNMENT
  padded_rows = (2**31 - 1) // cols // alignment * alignment
  return padded_rows - (alignment - 1) * num_experts


def expert_padding(group_sizes: jax.Array, compact_capacity: int) -> ExpertPadding:
  """Row mapping of `pad_expert_rows` for `compact_capacity` expert-sorted rows."""
  if group_sizes.ndim != 1:
    raise ValueError(f"group_sizes must be rank 1, got {group_sizes.shape}")

  alignment = _GROUP_ALIGNMENT
  num_experts = group_sizes.shape[0]
  padded_capacity = (compact_capacity + (alignment - 1) * num_experts + alignment - 1) // alignment * alignment

  group_sizes = group_sizes.astype(jnp.int32)
  compact_ends = jnp.cumsum(group_sizes, dtype=jnp.int32)
  compact_starts = compact_ends - group_sizes
  padded_sizes = _round_up_groups(group_sizes, alignment)
  padded_ends = jnp.cumsum(padded_sizes, dtype=jnp.int32)
  padded_starts = padded_ends - padded_sizes

  compact_rows = jnp.arange(compact_capacity, dtype=jnp.int32)
  compact_expert = jnp.sum(compact_rows[:, None] >= compact_ends[None, :], axis=1, dtype=jnp.int32)
  compact_valid = compact_expert < num_experts
  compact_safe_expert = jnp.minimum(compact_expert, num_experts - 1)
  compact_to_padded = padded_starts[compact_safe_expert] + compact_rows - compact_starts[compact_safe_expert]
  compact_to_padded = jnp.where(compact_valid, compact_to_padded, 0)

  padded_rows = jnp.arange(padded_capacity, dtype=jnp.int32)
  padded_expert = jnp.sum(
      padded_rows[:, None] >= padded_ends[None, :],
      axis=1,
      dtype=jnp.int32,
  )
  padded_safe_expert = jnp.minimum(padded_expert, num_experts - 1)
  padded_row_in_expert = padded_rows - padded_starts[padded_safe_expert]
  padded_valid = (padded_expert < num_experts) & (padded_row_in_expert < group_sizes[padded_safe_expert])
  padded_to_compact = compact_starts[padded_safe_expert] + padded_row_in_expert
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


def expert_row_ids(group_sizes: jax.Array, rows: int) -> jax.Array:
  """EP-local expert of each row in an expert-grouped layout.

  Rows past the last group get the out-of-range id ``len(group_sizes)``.
  Gather per-expert values with ``mode="fill"`` so those rows read zeros and
  their gradients are dropped: with an unzeroed tail they hold uninitialized
  values, which a scatter-add into a real expert would accumulate.
  """
  ends = jnp.cumsum(group_sizes.astype(jnp.int32), dtype=jnp.int32)
  return jnp.searchsorted(ends, jnp.arange(rows, dtype=jnp.int32), side="right").astype(jnp.int32)


@jax.custom_vjp
def gather_expert_rows(values: jax.Array, row_ids: jax.Array) -> jax.Array:
  """``values[row_ids]``, with zeros for the out-of-range ids of ``expert_row_ids``.

  The backward sums the cotangent rows of each expert with a one-hot GEMM. The
  gather's own transpose is a scatter-add of every row into a few expert rows,
  which serializes on the destination rows.
  """
  return values.at[row_ids].get(mode="fill", fill_value=0)


def _gather_expert_rows_fwd(values, row_ids):
  return gather_expert_rows(values, row_ids), (values, row_ids)


def _gather_expert_rows_bwd(res, grad):
  values, row_ids = res
  num_experts = values.shape[0]
  # Out-of-range rows can carry uninitialized cotangents. Select them away
  # before the GEMM, where a zero one-hot weight would not stop a NaN.
  grad = jnp.where((row_ids < num_experts)[:, None], grad, jnp.zeros((), grad.dtype))
  one_hot = jax.nn.one_hot(row_ids, num_experts, dtype=grad.dtype)
  dvalues = jax.lax.dot_general(one_hot, grad, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
  return dvalues.astype(values.dtype), None


gather_expert_rows.defvjp(_gather_expert_rows_fwd, _gather_expert_rows_bwd)


def _group_bounds(group_sizes: jax.Array) -> jax.Array:
  """``[0, end_0, ..., end_{E-1}]``, the whole-loop kernels' group bounds."""
  ends = jnp.cumsum(group_sizes.astype(jnp.int32), dtype=jnp.int32)
  return jnp.concatenate((jnp.zeros((1,), jnp.int32), ends))


def _quant_row_limit(group_sizes: jax.Array, zero_tail: bool) -> dict:
  """Rows past the last group are read by nothing when the tail is not zeroed."""
  if zero_tail:
    return {}
  return {"row_limit": jnp.sum(group_sizes.astype(jnp.int32))}


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
  rhs_packed, rhs_scales = quantize_mxfp4_dim0(rhs_for_kernel.reshape(experts * output_size, contraction))
  return (
      rhs_packed.reshape(experts, output_size, contraction // 2),
      rhs_scales.reshape(experts, output_size, contraction // 32),
  )


def _quantize_grad_output_dual(grad_output: jax.Array, group_sizes: jax.Array, zero_tail: bool):
  """``grad_output`` quantized for the input gradient (dim0) and the weight gradient (dim1) from one read.

  Bit-identical to ``quantize_mxfp4_dim0`` and ``quantize_mxfp4_dim1`` of ``grad_output``.
  """
  from jax_aiter.ops.moe_mxfp4_ptquant import quantize_mxfp4_dual

  row_limit = _quant_row_limit(group_sizes, zero_tail)
  packed, scales, transposed, transposed_scales = quantize_mxfp4_dual(grad_output, **row_limit)
  return (packed, scales), (transposed, transposed_scales)


def _grouped_gmm_quantized_rhs(
    lhs: jax.Array,
    rhs_packed: jax.Array,
    rhs_scales: jax.Array,
    group_sizes: jax.Array,
    *,
    zero_tail: bool = True,
    lhs_rows: tuple[jax.Array, jax.Array] | None = None,
    lhs_quantized: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
  """Grouped ``lhs[M,K] @ rhs`` with rhs already in the kernel's MXFP4 ``[E, out, contraction/2]`` layout.

  ``lhs`` is quantized here unless ``lhs_quantized`` supplies it; ``lhs_rows``
  quantizes gathered rows of a larger source instead of ``lhs``.
  """
  from jax_aiter.ops.moe_mxfp4 import (
      grouped_mxfp4_fwd_da_wholeloop,
      quantize_mxfp4_dim0,
      quantize_mxfp4_dim0_gather,
  )

  contraction = rhs_packed.shape[2] * 2
  if lhs.shape[1] != contraction:
    raise ValueError(f"Grouped contraction mismatch: lhs {lhs.shape}, " f"quantized rhs {rhs_packed.shape}")
  row_limit = _quant_row_limit(group_sizes, zero_tail)
  if lhs_quantized is not None:
    lhs_packed, lhs_scales = lhs_quantized
  elif lhs_rows is None:
    lhs_packed, lhs_scales = quantize_mxfp4_dim0(lhs, **row_limit)
  else:
    lhs_packed, lhs_scales = quantize_mxfp4_dim0_gather(*lhs_rows, **row_limit)
  return grouped_mxfp4_fwd_da_wholeloop(
      lhs_packed, rhs_packed, lhs_scales, rhs_scales, _group_bounds(group_sizes), zero_tail=zero_tail
  )


def _grouped_gmm(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    transpose_rhs: bool,
    zero_tail: bool = True,
    lhs_rows: tuple[jax.Array, jax.Array] | None = None,
    lhs_quantized: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
  """Grouped MXFP4 ``lhs @ rhs`` for bf16 ``rhs[E,K,N]``, or ``lhs @ rhs^T`` with ``transpose_rhs``."""
  _check_grouped_operands(lhs, rhs, group_sizes)
  rhs_packed, rhs_scales = _quantize_rhs(rhs, transpose_rhs=transpose_rhs)
  return _grouped_gmm_quantized_rhs(
      lhs,
      rhs_packed,
      rhs_scales,
      group_sizes,
      zero_tail=zero_tail,
      lhs_rows=lhs_rows,
      lhs_quantized=lhs_quantized,
  )


def _grouped_weight_grad(
    lhs: jax.Array | None,
    grad_output: jax.Array,
    group_sizes: jax.Array,
    zero_tail: bool = True,
    lhs_rows: tuple[jax.Array, jax.Array] | None = None,
    grad_quantized: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
  """d(lhs @ rhs)/d rhs for lhs[M,K] and grad_output[M,N], as [E,K,N].

  ``grad_quantized`` is ``grad_output`` already in the dim1 cast layout.
  """
  from jax_aiter.ops.moe_mxfp4 import (
      grouped_mxfp4_dw_wholeloop,
      quantize_mxfp4_dim1,
      quantize_mxfp4_dim1_gather,
  )

  row_limit = _quant_row_limit(group_sizes, zero_tail)
  if grad_quantized is None:
    grad_transposed, grad_scales = quantize_mxfp4_dim1(grad_output, **row_limit)
  else:
    grad_transposed, grad_scales = grad_quantized
  if lhs_rows is None:
    lhs_transposed, lhs_scales = quantize_mxfp4_dim1(lhs, **row_limit)
  else:
    lhs_transposed, lhs_scales = quantize_mxfp4_dim1_gather(*lhs_rows, **row_limit)
  # Raw dW is [E,N,K]; MaxText stores expert weights as [E,K,N].
  return grouped_mxfp4_dw_wholeloop(
      grad_transposed,
      grad_scales,
      lhs_transposed,
      lhs_scales,
      _group_bounds(group_sizes),
  ).swapaxes(1, 2)


def _make_grouped_gmm(zero_tail: bool, dual_grad_cast: bool):
  """Grouped MXFP4 GEMM whose backward runs the input and weight gradients as MXFP4 GEMMs too."""

  @jax.custom_vjp
  def grouped_gmm_for_kernel(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, lhs_rows=None) -> jax.Array:
    return _grouped_gmm(
        lhs,
        rhs,
        group_sizes,
        transpose_rhs=False,
        zero_tail=zero_tail,
        lhs_rows=lhs_rows,
    )

  def forward(lhs, rhs, group_sizes, lhs_rows=None):
    output = _grouped_gmm(
        lhs,
        rhs,
        group_sizes,
        transpose_rhs=False,
        zero_tail=zero_tail,
        lhs_rows=lhs_rows,
    )
    return output, (lhs if lhs_rows is None else None, rhs, group_sizes, lhs_rows)

  def backward(residuals, grad_output):
    lhs, rhs, group_sizes, lhs_rows = residuals
    grad_output = grad_output.astype(jnp.bfloat16)
    grad_rows = grad_cols = None
    if dual_grad_cast:
      grad_rows, grad_cols = _quantize_grad_output_dual(grad_output, group_sizes, zero_tail)
    grad_lhs = _grouped_gmm(
        grad_output,
        rhs,
        group_sizes,
        transpose_rhs=True,
        zero_tail=zero_tail,
        lhs_quantized=grad_rows,
    )
    grad_rhs = _grouped_weight_grad(lhs, grad_output, group_sizes, zero_tail, lhs_rows, grad_quantized=grad_cols)
    return grad_lhs, grad_rhs, None, None

  grouped_gmm_for_kernel.defvjp(forward, backward)
  return grouped_gmm_for_kernel


_GROUPED_GMM = {
    (zero_tail, dual): _make_grouped_gmm(zero_tail, dual) for zero_tail in (True, False) for dual in (False, True)
}


class GatheredExpertWeight(NamedTuple):
  """EP-local expert weight ``[E,K,N]`` gathered as MXFP4 in both GEMM layouts.

  ``fwd_*`` is the forward kernel operand and ``bwd_*`` the input-gradient
  operand. ``grad_carrier`` is a full-size zero whose cotangent is the
  gathered weight gradient; it lets every grouped GEMM that reads this weight,
  including ones inside tier conditionals, return its gradient without a
  collective, leaving one reduce-scatter at the gather.
  """

  fwd_packed: jax.Array
  fwd_scales: jax.Array
  bwd_packed: jax.Array
  bwd_scales: jax.Array
  grad_carrier: jax.Array

  @property
  def shape(self):
    return self.grad_carrier.shape


def _kernel_shard_axis(shard_dim: int, transpose_rhs: bool) -> int:
  """Axis of the ``[E, out, contraction]`` kernel layout holding rhs dim ``shard_dim``."""
  if shard_dim == 1:
    return 1 if transpose_rhs else 2
  if shard_dim == 2:
    return 2 if transpose_rhs else 1
  raise ValueError(f"Expert weights may only be sharded on dim 1 or 2, got {shard_dim}")


def _gather_quantized_rhs(rhs_shard, *, transpose_rhs, axis_name, shard_dim):
  # Compact MXFP4 blocks run along the contraction axis and never span a
  # shard boundary, so gathering quantized shards is bit-identical to
  # quantizing the gathered weight.
  packed, scales = _quantize_rhs(rhs_shard, transpose_rhs=transpose_rhs)
  axis = _kernel_shard_axis(shard_dim, transpose_rhs)
  return (
      jax.lax.all_gather(packed, axis_name, axis=axis, tiled=True),
      jax.lax.all_gather(scales, axis_name, axis=axis, tiled=True),
  )


def _gather_expert_weight_fwd(rhs_shard, axis_name, shard_dim):
  """Forward of :func:`gather_expert_weight`; it saves no residuals."""
  if rhs_shard.ndim != 3 or rhs_shard.dtype != jnp.bfloat16:
    raise TypeError(f"Expected bfloat16 rhs[E,K,N] shard, got {rhs_shard.dtype}{rhs_shard.shape}")
  if rhs_shard.shape[1] % 32 or rhs_shard.shape[2] % 32:
    raise ValueError(f"MXFP4 FSDP gather needs both local weight dims divisible by 32, got {rhs_shard.shape}")
  fwd_packed, fwd_scales = _gather_quantized_rhs(rhs_shard, transpose_rhs=False, axis_name=axis_name, shard_dim=shard_dim)
  bwd_packed, bwd_scales = _gather_quantized_rhs(rhs_shard, transpose_rhs=True, axis_name=axis_name, shard_dim=shard_dim)
  full_shape = list(rhs_shard.shape)
  full_shape[shard_dim] *= jax.lax.axis_size(axis_name)
  carrier = jnp.zeros(full_shape, rhs_shard.dtype)
  return GatheredExpertWeight(fwd_packed, fwd_scales, bwd_packed, bwd_scales, carrier), None


def _gather_expert_weight_bwd(axis_name, shard_dim, _, cotangent):
  return (jax.lax.psum_scatter(cotangent.grad_carrier, axis_name, scatter_dimension=shard_dim, tiled=True),)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def gather_expert_weight(rhs_shard: jax.Array, axis_name, shard_dim: int) -> GatheredExpertWeight:
  """All-gather an FSDP-sharded expert weight as MXFP4 inside ``shard_map``.

  ``rhs_shard`` is the local ``[E,K,N]`` slice sharded over ``axis_name`` on
  ``shard_dim``. Its gradient is the sum over ``axis_name`` of the gathered
  weight gradient, reduce-scattered back to the shard.
  """
  return _gather_expert_weight_fwd(rhs_shard, axis_name, shard_dim)[0]


gather_expert_weight.defvjp(_gather_expert_weight_fwd, _gather_expert_weight_bwd)


def _float0_zeros(shape):
  return np.zeros(shape, dtype=jax.dtypes.float0)


def _make_gathered_grouped_gmm(zero_tail: bool, dual_grad_cast: bool):
  """Grouped MXFP4 GEMM on a :class:`GatheredExpertWeight`, returning its gradient on ``grad_carrier``."""

  def forward_output(lhs, weight, group_sizes, lhs_rows):
    _check_grouped_operands(lhs, weight.grad_carrier, group_sizes)
    return _grouped_gmm_quantized_rhs(
        lhs,
        weight.fwd_packed,
        weight.fwd_scales,
        group_sizes,
        zero_tail=zero_tail,
        lhs_rows=lhs_rows,
    )

  @jax.custom_vjp
  def gathered_gmm_for_kernel(lhs, weight, group_sizes, lhs_rows=None):
    return forward_output(lhs, weight, group_sizes, lhs_rows)

  def forward(lhs, weight, group_sizes, lhs_rows=None):
    output = forward_output(lhs, weight, group_sizes, lhs_rows)
    saved_lhs = lhs if lhs_rows is None else None
    return output, (saved_lhs, weight.bwd_packed, weight.bwd_scales, group_sizes, lhs_rows)

  def backward(residuals, grad_output):
    lhs, bwd_packed, bwd_scales, group_sizes, lhs_rows = residuals
    grad_output = grad_output.astype(jnp.bfloat16)
    grad_rows = grad_cols = None
    if dual_grad_cast:
      grad_rows, grad_cols = _quantize_grad_output_dual(grad_output, group_sizes, zero_tail)
    grad_lhs = _grouped_gmm_quantized_rhs(
        grad_output,
        bwd_packed,
        bwd_scales,
        group_sizes,
        zero_tail=zero_tail,
        lhs_quantized=grad_rows,
    )
    grad_rhs = _grouped_weight_grad(lhs, grad_output, group_sizes, zero_tail, lhs_rows, grad_quantized=grad_cols)
    # bwd_packed is [E, K, N/2]; the forward operand is [E, N, K/2].
    experts, k_dim, packed_n = bwd_packed.shape
    n_dim = packed_n * 2
    weight_cotangent = GatheredExpertWeight(
        _float0_zeros((experts, n_dim, k_dim // 2)),
        _float0_zeros((experts, n_dim, k_dim // 32)),
        _float0_zeros(bwd_packed.shape),
        _float0_zeros(bwd_scales.shape),
        grad_rhs.astype(jnp.bfloat16),
    )
    return grad_lhs, weight_cotangent, None, None

  gathered_gmm_for_kernel.defvjp(forward, backward)
  return gathered_gmm_for_kernel


_GATHERED_GROUPED_GMM = {
    (zero_tail, dual): _make_gathered_grouped_gmm(zero_tail, dual)
    for zero_tail in (True, False)
    for dual in (False, True)
}


def grouped_gmm(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    zero_tail: bool = True,
    lhs_rows: tuple[jax.Array, jax.Array] | None = None,
    dual_grad_cast: bool = False,
) -> jax.Array:
  """Differentiable ``lhs[group] @ rhs[group]`` in dynamic MXFP4.

  ``group_sizes`` must be multiples of the kernel's row alignment, as
  produced by :func:`pad_expert_rows`.

  Rows past the last group are zero when ``zero_tail`` is true, in the output
  and in the input gradient. When false, the whole-loop kernel leaves them
  uninitialized: a full-buffer pass is saved, so pass false only when every
  reader of those rows masks them with a select.

  ``rhs`` may be a :class:`GatheredExpertWeight` from
  :func:`gather_expert_weight`.

  ``lhs_rows = (values, row_index)`` states that ``lhs`` row ``r`` is
  ``values[row_index[r]]``, or zero for a negative index. Both passes then
  quantize straight from ``values`` and never read ``lhs``, so the gather
  producing ``lhs`` is dead code.   ``lhs`` still receives the input gradient;
  ``lhs_rows`` receives none.

  ``dual_grad_cast`` quantizes the output gradient for the input and weight
  gradients with one fused cast, which needs N % 256 == 0 after padding.
  Results are bit-identical.

  K and N that the kernels cannot take are zero-padded to
  :func:`padded_feature_dim`; the padding is exact because MXFP4 blocks are 32
  wide and both sizes are multiples of 32.
  """
  if lhs_rows is not None:
    values, row_index = lhs_rows
    lhs_rows = (jax.lax.stop_gradient(values).astype(jnp.bfloat16), row_index.astype(jnp.int32))
  _, k_dim, n_dim = rhs.shape
  k_padded = padded_feature_dim(k_dim)
  n_padded = padded_feature_dim(n_dim)
  if isinstance(rhs, GatheredExpertWeight):
    if (k_padded, n_padded) != (k_dim, n_dim):
      raise ValueError(f"MXFP4 FSDP weight gather needs kernel-aligned expert weights, got {rhs.shape}")
    return _GATHERED_GROUPED_GMM[zero_tail, dual_grad_cast](lhs, rhs, group_sizes, lhs_rows)
  if (k_padded, n_padded) == (k_dim, n_dim):
    return _GROUPED_GMM[zero_tail, dual_grad_cast](lhs, rhs, group_sizes, lhs_rows)
  lhs = _pad_axis(lhs, 1, k_padded)
  rhs = _pad_axis(_pad_axis(rhs, 1, k_padded), 2, n_padded)
  if lhs_rows is not None:
    lhs_rows = (_pad_axis(lhs_rows[0], 1, k_padded), lhs_rows[1])
  return _GROUPED_GMM[zero_tail, dual_grad_cast](lhs, rhs, group_sizes, lhs_rows)[:, :n_dim]


def _active_axes(entry, mesh: Mesh) -> tuple:
  axes = entry if isinstance(entry, tuple) else (entry,)
  return tuple(axis for axis in axes if axis is not None and mesh.shape.get(axis, 1) > 1)


def dense_linear(
    lhs: jax.Array,
    rhs: jax.Array,
    mesh: Mesh,
    rhs_spec: PartitionSpec | None = None,
    dual_grad_cast: bool = False,
) -> jax.Array:
  """Differentiable E=1 whole-loop MXFP4 linear with EP-local execution.

  ``lhs`` and the result are sharded over the active data-parallel mesh axes.
  The contraction dimension of ``rhs`` must be complete on every rank.  By
  default ``rhs`` is replicated before the call and the transpose of
  ``shard_map`` sums its per-rank gradients over those same axes.

  ``rhs_spec`` instead passes ``rhs`` sharded over data-parallel axes and
  all-gathers it inside the ``shard_map`` under the name ``gathered_weight``.
  A remat policy that saves that name keeps the gathered weight per device for
  backward rather than gathering it again; the gradient is reduce-scattered
  back to ``rhs_spec``. ``dual_grad_cast`` is as in :func:`grouped_gmm`.
  """
  if lhs.ndim != 2 or rhs.ndim != 2:
    raise ValueError(f"Expected lhs[M,K] and rhs[K,N], got {lhs.shape}, {rhs.shape}")
  if lhs.dtype != jnp.bfloat16 or rhs.dtype != jnp.bfloat16:
    raise TypeError("JAX-AITER MXFP4 dense linear requires bfloat16 operands")
  rows, contraction = lhs.shape
  if rhs.shape[0] != contraction:
    raise ValueError(f"Dense contraction mismatch: lhs {lhs.shape}, rhs {rhs.shape}")

  active_batch_axes = tuple(axis for axis in _DENSE_BATCH_MESH_AXES if mesh.shape.get(axis, 1) > 1)
  batch_shards = 1
  for axis in active_batch_axes:
    batch_shards *= mesh.shape[axis]
  if rows % batch_shards:
    raise ValueError(f"Dense rows {rows} are not divisible by batch sharding {batch_shards}")
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

  rhs_spec = PartitionSpec(None, None) if rhs_spec is None else rhs_spec
  gathers = []
  for dim, entry in enumerate(tuple(rhs_spec) + (None,) * (2 - len(rhs_spec))):
    axes = _active_axes(entry, mesh)
    # A gathered axis must also shard lhs rows: its reduce-scatter then sums
    # distinct per-rank gradients, never copies of one.
    if not set(axes) <= set(active_batch_axes):
      raise ValueError(f"rhs axes {axes} are not data-parallel axes {active_batch_axes}")
    if axes:
      gathers.append((dim, axes))

  def local_linear(local_lhs, local_rhs):
    for dim, axes in gathers:
      local_rhs = jax.lax.all_gather(local_rhs, axes, axis=dim, tiled=True)
    if gathers:
      local_rhs = checkpoint_name(local_rhs, "gathered_weight")
    group_sizes = jnp.asarray([local_lhs.shape[0]], dtype=jnp.int32)
    return grouped_gmm(local_lhs, local_rhs[None, :, :], group_sizes, dual_grad_cast=dual_grad_cast)

  return jax.shard_map(
      local_linear,
      mesh=mesh,
      in_specs=(PartitionSpec(batch_axis, None), rhs_spec),
      out_specs=PartitionSpec(batch_axis, None),
      check_vma=False,
  )(lhs, rhs)


__all__ = [
    "ExpertPadding",
    "GatheredExpertWeight",
    "dense_linear",
    "gather_expert_weight",
    "grouped_gmm",
    "pad_expert_rows",
    "unpad_expert_rows",
]
