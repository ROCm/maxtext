# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quantization library."""

import functools
import json
import re
from typing import Tuple, Sequence
from dataclasses import dataclass

from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2.flax import aqt_flax
from aqt.jax.v2 import tiled_dot_general
from aqt.jax.v2 import calibration

import qwix

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten_with_path, tree_unflatten

from flax.linen import fp8_ops
from flax.linen import initializers as flax_initializers
import flax.linen as nn

from maxtext.common.common_types import DType, Config
from maxtext.inference.kvcache import KVQuant

# Params used to define mixed precision quantization configs
DEFAULT = "__default__"  # default config
_W_BITS = "w_bits"  # Number of bits used to represent weights
_A_BITS = "a_bits"  # Number of bits used to represent activations
_W_SCALE = "w_scale"  # Clipping scale for weights
_A_SCALE = "a_scale"  # Clipping scale for activations
_TILE_SIZE = "tile_size"  # Tile size for subchannel


@dataclass
class Quantization:
  """Base class for quantization configurations"""

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Placeholder for dot_general implementation in subclasses."""

  def einsum(self, dtype: DType = jnp.float32):
    """Placeholder for einsum implementation in subclasses."""


def _tiling_fn(lhs, rhs, dimension_numbers, tile_size):
  """apply tiling function"""
  del lhs, rhs

  (lhs_ca, rhs_ca), _ = dimension_numbers
  ret = tiled_dot_general.Cfg(
      lhs=tiled_dot_general.TensorTiling(contraction_axes=[], remaining_axes=[]),
      rhs=tiled_dot_general.TensorTiling(contraction_axes=[], remaining_axes=[]),
  )

  for lhs_idx, rhs_idx in zip(lhs_ca, rhs_ca):
    ret.lhs.contraction_axes.append(tiled_dot_general.AxisTiling(axis=lhs_idx, tile_size=tile_size, tile_count=None))
    ret.rhs.contraction_axes.append(tiled_dot_general.AxisTiling(axis=rhs_idx, tile_size=tile_size, tile_count=None))

  return ret


def _rhs_axis_metadata_wrapper(
    x: jnp.ndarray,
    tile_map,
    no_sharding_axis: Sequence[int],
    mesh_axes: Tuple[str, ...],
    is_tiled: bool,
    replicate_scale: bool = False,
):
  """right-hand-side axis metadata wrapper"""
  if replicate_scale:
    # Temporarily using the shape to identify the scale.
    # TODO: remove the replication once the 2d sharding quantization
    # works as expected.
    if len(x.shape) == 1:
      return nn.with_logical_partitioning((lambda: x), tuple(None for _ in mesh_axes))()

  mesh_axes = list(mesh_axes)
  if is_tiled:
    # tile_map is a mapping between original rank and a list of new, tiled rank.
    if len(mesh_axes) < len(tile_map):
      mesh_axes = [None] * (len(tile_map) - len(mesh_axes)) + mesh_axes
    new_mesh_axes = [None] * len(x.shape)
    for orig_rank, new_rank in tile_map.items():
      assert new_rank
      assert len(new_rank) <= 2
      new_mesh_axes[new_rank[-1]] = mesh_axes[orig_rank]
    mesh_axes = new_mesh_axes

  if mesh_axes is not None and len(mesh_axes) > 0:
    for no_shard_idx in no_sharding_axis:
      if no_shard_idx < len(mesh_axes):
        mesh_axes[no_shard_idx] = None

  return nn.with_logical_partitioning((lambda: x), mesh_axes)()


@dataclass
class AqtQuantization:
  """Configures AQT quantization github.com/google/aqt."""

  quant_dg: aqt_config.DotGeneral
  quant_mode: aqt_flax.QuantMode = aqt_flax.QuantMode.TRAIN
  replicate_scale: bool = False

  def _get_mixed_precision_cfg(self):
    """get configuration for mixed precision"""
    quant_dg = None
    is_tiled = False
    tiling_fn = None
    # pylint: disable=protected-access
    module_path = "/".join(nn.module._context.module_stack[-1].path)
    tile_size = -1
    for layer_name_re, layer_quant_dg in self.quant_dg.items():
      if re.fullmatch(layer_name_re, module_path):
        quant_dg, tile_size = layer_quant_dg
    if quant_dg is None:
      quant_dg, tile_size = self.quant_dg[DEFAULT]
    if tile_size != -1:
      is_tiled = True
      tiling_fn = functools.partial(_tiling_fn, tile_size=tile_size)
    return quant_dg, is_tiled, tiling_fn

  def _get_rhs_axis_metadata_wrapper(
      self, mesh_axes: Tuple[str, ...] = (), is_tiled: bool = False, replicate_scale: bool = False
  ):
    if self.quant_mode == aqt_flax.QuantMode.CONVERT:
      return None
    return functools.partial(
        _rhs_axis_metadata_wrapper, mesh_axes=mesh_axes, is_tiled=is_tiled, replicate_scale=replicate_scale
    )

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns dot_general configured with aqt params."""
    if isinstance(self.quant_dg, dict):
      quant_dg, is_tiled, tiling_fn = self._get_mixed_precision_cfg()
    else:
      quant_dg, is_tiled, tiling_fn = self.quant_dg, False, None
    rhs_axis_metadata_wrapper = self._get_rhs_axis_metadata_wrapper(
        mesh_axes, is_tiled, replicate_scale=self.replicate_scale
    )
    # module_path = "/".join(nn.module._context.module_stack[-1].path)
    # print(f"quant_dg: {quant_dg}, is_tiled: {is_tiled}, module_path: {module_path}")
    aqt_dg_cls = functools.partial(
        aqt_flax.AqtDotGeneral,
        quant_dg,
        rhs_quant_mode=self.quant_mode,
        lhs_freeze_mode=aqt_flax.FreezerMode.NONE,
        rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
        rhs_axis_metadata_wrapper=rhs_axis_metadata_wrapper,
        use_legacy_freezer=False,
        tiling_fn=tiling_fn,
    )
    return aqt_dg_cls

  def einsum(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns einsum configured with aqt params."""
    if isinstance(self.quant_dg, dict):
      quant_dg, is_tiled, tiling_fn = self._get_mixed_precision_cfg()
    else:
      quant_dg, is_tiled, tiling_fn = self.quant_dg, False, None

    rhs_axis_metadata_wrapper = self._get_rhs_axis_metadata_wrapper(
        mesh_axes, is_tiled, replicate_scale=self.replicate_scale
    )
    aqt_einsum = functools.partial(
        aqt_flax.AqtEinsum(
            cfg=quant_dg,
            rhs_quant_mode=self.quant_mode,
            lhs_freeze_mode=aqt_flax.FreezerMode.NONE,
            rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
            rhs_axis_metadata_wrapper=rhs_axis_metadata_wrapper,
            use_legacy_freezer=False,
            tiling_fn=tiling_fn,
        )
    )
    return aqt_einsum


@dataclass
class Fp8Quantization(Quantization):
  """Configures Fp8 quantization for NVIDIA GPUs"""

  quant_mode = "train"

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns dot_general configured with aqt params."""
    return nn.Fp8DirectDotGeneralOp

  def einsum(self, dtype: DType = jnp.float32):
    return _Fp8EinsumWrapper(dtype=dtype)


class _Fp8EinsumWrapper(nn.Module):
  """Wrapper for nn.Fp8Einsum to handle computation dtype."""

  dtype: DType

  @nn.compact
  def __call__(self, eqn, lhs, rhs, **kwargs):
    # nn.Fp8Einsum determines compute dtype from rhs.
    # We cast rhs to the desired computation dtype.
    # nn.Fp8Einsum will then cast lhs to the same dtype.
    rhs = rhs.astype(self.dtype)
    return nn.Fp8Einsum(name="fp8_einsum")(eqn, lhs, rhs, **kwargs)


class Fp8Einsum(nn.Module):
  """An fp8 einsum op."""

  #: size of the amax history.
  amax_history_length: int = 1024
  #: e4m3 variants, e.g., e4m3fn, e4m3fnuz.
  e4m3_dtype: DType = jnp.float8_e4m3fn
  #: e5m2 variants, e.g., e5m2, e5m2fnuz.
  e5m2_dtype: DType = jnp.float8_e5m2
  #: computation dtype.
  dtype: DType = jnp.float32

  def setup(self) -> None:
    """init with input_amax_history, kernel_amax_history, output_grad_amax_history,
    input_scale, kernel_scale, output_grad_scale"""
    scale_args = (
        flax_initializers.ones_init(),
        jax.random.PRNGKey(0),
        (1,),
        jnp.float32,
    )
    amax_history_args = (
        flax_initializers.zeros_init(),
        jax.random.PRNGKey(0),
        (self.amax_history_length,),
        jnp.float32,
    )

    OVERWRITE_WITH_GRADIENT = "_overwrite_with_gradient"
    self.input_amax_history = self.variable(OVERWRITE_WITH_GRADIENT, "input_amax_history", *amax_history_args)
    self.kernel_amax_history = self.variable(OVERWRITE_WITH_GRADIENT, "kernel_amax_history", *amax_history_args)
    self.output_grad_amax_history = self.variable(OVERWRITE_WITH_GRADIENT, "output_grad_amax_history", *amax_history_args)

    self.input_scale = self.variable(OVERWRITE_WITH_GRADIENT, "input_scale", *scale_args)
    self.kernel_scale = self.variable(OVERWRITE_WITH_GRADIENT, "kernel_scale", *scale_args)
    self.output_grad_scale = self.variable(OVERWRITE_WITH_GRADIENT, "output_grad_scale", *scale_args)

  def __call__(self, eqn, *args, **kwargs):
    assert len(args) == 2
    x = args[0]
    k = args[1]

    comp_dtype = self.dtype
    k = jnp.asarray(k, comp_dtype)
    x = jnp.asarray(x, comp_dtype)

    x_qdq = fp8_ops.in_qdq(comp_dtype, self.e4m3_dtype, x, self.input_scale.value, self.input_amax_history.value)
    k_qdq = fp8_ops.in_qdq(comp_dtype, self.e4m3_dtype, k, self.kernel_scale.value, self.kernel_amax_history.value)

    y_qdq = jnp.einsum(eqn, x_qdq, k_qdq, _dot_general=fp8_ops.dot_general_with_precision)

    y = fp8_ops.out_qdq(
        comp_dtype,
        self.e5m2_dtype,
        y_qdq,
        self.output_grad_scale.value,
        self.output_grad_amax_history.value,
    )
    return y


@dataclass
class NANOOFp8Quantization(Quantization):
  """Configures NANOO Fp8 quantization for AMD MI300/MI325 GPUs"""

  quant_mode = "train"

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns dot_general configured with aqt params."""
    return nn.NANOOFp8DotGeneralOp


class AiterBf16DotGeneralOp(nn.Module):
  """Drop-in dot_general replacement using AITER ASM BF16 GEMM.

  Uses AITER hand-tuned ASM kernels via FFI for all GEMM operations.
  Supports multi-GPU via custom_partitioning (sharding_rule="m k, n k -> m n").
  """

  @nn.compact
  def __call__(self, inputs, kernel, dimension_numbers, precision=None, **kwargs):
    from jax_aiter.gemm import gemm as aiter_gemm

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    lhs_contract = tuple(lhs_contract)
    rhs_contract = tuple(rhs_contract)

    inp_shape = inputs.shape
    ker_shape = kernel.shape
    inp_ndim = len(inp_shape)
    ker_ndim = len(ker_shape)

    m_axes = [i for i in range(inp_ndim) if i not in lhs_contract]
    n_axes = [i for i in range(ker_ndim) if i not in rhs_contract]

    K = 1
    for ax in lhs_contract:
      K *= inp_shape[ax]
    M = 1
    for ax in m_axes:
      M *= inp_shape[ax]
    N = 1
    for ax in n_axes:
      N *= ker_shape[ax]

    perm_lhs = m_axes + list(lhs_contract)
    is_identity_lhs = perm_lhs == list(range(inp_ndim))
    if is_identity_lhs:
      a_2d = jnp.reshape(inputs, (M, K))
    else:
      a_2d = jnp.reshape(jnp.transpose(inputs, perm_lhs), (M, K))

    perm_rhs = n_axes + list(rhs_contract)
    is_identity_rhs = perm_rhs == list(range(ker_ndim))
    if is_identity_rhs:
      b_nk = jnp.reshape(kernel, (N, K))
    else:
      b_nk = jnp.reshape(jnp.transpose(kernel, perm_rhs), (N, K))

    a_2d = a_2d.astype(jnp.bfloat16)
    b_nk = b_nk.astype(jnp.bfloat16)

    out_2d = aiter_gemm(a_2d, b_nk)

    out_m_shape = tuple(inp_shape[ax] for ax in m_axes)
    out_n_shape = tuple(ker_shape[ax] for ax in n_axes)
    return jnp.reshape(out_2d, out_m_shape + out_n_shape)


@dataclass
class AiterBf16Quantization(Quantization):
  """AITER ASM BF16 GEMM for AMD MI350 (gfx950).

  Uses hand-tuned assembly kernels via FFI.
  Set quantization='aiter_bf16' to enable.
  """

  quant_mode = "train"

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    return AiterBf16DotGeneralOp


AITER_FP8_AMAX_HISTORY_LEN = 1024


def _aiter_fp8_compute_scale(amax_history, prev_scale, fp8_max=448.0):
  """Compute FP8 scale from amax history (delayed scaling).

  Returns amax / fp8_max (a DIVISOR scale: x_quantized = x / scale).
  When history has no valid entries (all zeros, e.g. first step), falls back
  to prev_scale (initialized to 1.0) so the full FP8 range is usable.
  """
  amax = jnp.max(amax_history)
  safe_fallback = prev_scale
  new_scale = jnp.where(amax > 0, amax / fp8_max, safe_fallback)
  new_scale = jnp.where(jnp.isfinite(amax), new_scale, safe_fallback)
  return new_scale


def _aiter_fp8_update_history(x, amax_history):
  """Update rolling amax history with current tensor's max."""
  amax_update = jnp.max(jnp.abs(x)).astype(amax_history.dtype)
  return jnp.roll(amax_history, shift=-1).at[-1].set(amax_update)


class AiterFp8DotGeneralOp(nn.Module):
  """Drop-in dot_general replacement using AITER FP8 block-scale GEMM for MI350.

  Uses per-call scaling (scale computed from current tensor amax each call).
  Quantization + weight shuffle happen inside custom_partitioning so they
  operate on local shard shapes under FSDP. Automatic BF16 fallback when
  kernel constraints (N%256, K%128, M>=16, K>=512) are not met on the local
  shard. Backward uses BF16 GEMM (STE pattern via lax.dot_general).
  """

  @nn.compact
  def __call__(self, inputs, kernel, dimension_numbers, precision=None, **kwargs):
    from jax_aiter.gemm_fp8 import gemm_fp8_mi350 as aiter_fp8_gemm
    from jax_aiter.gemm_fp8 import fp8_supported_for_shape
    from jax_aiter.gemm import gemm as aiter_gemm

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    lhs_contract = tuple(lhs_contract)
    rhs_contract = tuple(rhs_contract)

    inp_shape = inputs.shape
    ker_shape = kernel.shape
    inp_ndim = len(inp_shape)
    ker_ndim = len(ker_shape)

    m_axes = [i for i in range(inp_ndim) if i not in lhs_contract]
    n_axes = [i for i in range(ker_ndim) if i not in rhs_contract]

    K = 1
    for ax in lhs_contract:
      K *= inp_shape[ax]
    M = 1
    for ax in m_axes:
      M *= inp_shape[ax]
    N = 1
    for ax in n_axes:
      N *= ker_shape[ax]

    perm_lhs = m_axes + list(lhs_contract)
    is_identity_lhs = perm_lhs == list(range(inp_ndim))
    if is_identity_lhs:
      a_2d = jnp.reshape(inputs, (M, K))
    else:
      a_2d = jnp.reshape(jnp.transpose(inputs, perm_lhs), (M, K))

    perm_rhs = n_axes + list(rhs_contract)
    is_identity_rhs = perm_rhs == list(range(ker_ndim))
    if is_identity_rhs:
      b_nk = jnp.reshape(kernel, (N, K))
    else:
      b_nk = jnp.reshape(jnp.transpose(kernel, perm_rhs), (N, K))

    a_bf16 = a_2d.astype(jnp.bfloat16)
    b_bf16 = b_nk.astype(jnp.bfloat16)

    # Pre-dispatch: check shape support BEFORE entering FP8 custom_partitioning.
    # Never mix lax.dot_general fallback inside FP8 custom_partitioning (breaks FSDP).
    if fp8_supported_for_shape(M, N, K):
      out_2d = aiter_fp8_gemm(a_bf16, b_bf16)
    else:
      out_2d = aiter_gemm(a_bf16, b_bf16)

    out_m_shape = tuple(inp_shape[ax] for ax in m_axes)
    out_n_shape = tuple(ker_shape[ax] for ax in n_axes)
    return jnp.reshape(out_2d, out_m_shape + out_n_shape)


# ---------------------------------------------------------------------------
# Delayed scaling GEMM with custom_vjp following Flax fp8_ops.in_qdq pattern.
# The backward returns updated scale/history as "gradients" so the optimizer
# overwrites the _overwrite_with_gradient variables with the new values.
# ---------------------------------------------------------------------------

def _aiter_delayed_compute_scale(amax_history, prev_scale, fp8_max=448.0):
  """Compute FP8 scale from amax history (Flax convention: divisor scale).

  scale = max(amax_history) / fp8_max
  When history is all zeros, falls back to prev_scale.
  """
  amax = jnp.max(amax_history)
  sf = jnp.where(amax > 0.0, amax / fp8_max, prev_scale)
  sf = jnp.where(jnp.isfinite(amax), sf, prev_scale)
  return sf


def _aiter_delayed_update_history(x, amax_history):
  """Update amax history: roll left, insert current amax at end."""
  amax_update = jnp.max(jnp.abs(x)).astype(amax_history.dtype)
  return jnp.roll(amax_history, shift=-1, axis=0).at[-1].set(amax_update)


@functools.partial(jax.custom_vjp, nondiff_argnums=())
def _aiter_fp8_gemm_delayed(
    a_bf16, b_bf16,
    input_scale, kernel_scale,
    input_history, kernel_history
):
  """FP8 GEMM with delayed scaling — forward uses scale from amax history.

  This function participates in jax.grad. The backward returns updated
  scale/history as "gradients" for _overwrite_with_gradient variables.
  """
  from jax_aiter.gemm_fp8.gemm_fp8_mi350 import _gemm_fp8_delayed_partitioned

  # Compute new scale from history (delayed: based on PREVIOUS steps' amax)
  new_input_scale = _aiter_delayed_compute_scale(input_history, input_scale)
  new_kernel_scale = _aiter_delayed_compute_scale(kernel_history, kernel_scale)

  # Run FP8 GEMM with delayed scales
  out = _gemm_fp8_delayed_partitioned(
      a_bf16, b_bf16,
      new_input_scale.reshape(1), new_kernel_scale.reshape(1))
  return out


def _aiter_fp8_gemm_delayed_fwd(
    a_bf16, b_bf16,
    input_scale, kernel_scale,
    input_history, kernel_history
):
  from jax_aiter.gemm_fp8.gemm_fp8_mi350 import _gemm_fp8_delayed_partitioned

  # Compute new scale from history
  new_input_scale = _aiter_delayed_compute_scale(input_history, input_scale)
  new_kernel_scale = _aiter_delayed_compute_scale(kernel_history, kernel_scale)

  # Run FP8 GEMM
  out = _gemm_fp8_delayed_partitioned(
      a_bf16, b_bf16,
      new_input_scale.reshape(1), new_kernel_scale.reshape(1))

  # Update histories with current tensor amax (for next step)
  new_input_history = _aiter_delayed_update_history(a_bf16, input_history)
  new_kernel_history = _aiter_delayed_update_history(b_bf16, kernel_history)

  # Recompute scales from updated history (these are what the optimizer stores)
  final_input_scale = _aiter_delayed_compute_scale(new_input_history, new_input_scale)
  final_kernel_scale = _aiter_delayed_compute_scale(new_kernel_history, new_kernel_scale)

  # Save residuals for backward
  return out, (a_bf16, b_bf16,
               final_input_scale, final_kernel_scale,
               new_input_history, new_kernel_history)


def _aiter_fp8_gemm_delayed_bwd(res, g):
  (a_bf16, b_bf16,
   new_input_scale, new_kernel_scale,
   new_input_history, new_kernel_history) = res

  g = g.astype(jnp.bfloat16)

  # BF16 backward for activations and weights (same as per-call version)
  da = jax.lax.dot_general(g, b_bf16, (((1,), (0,)), ((), ())))
  db = jax.lax.dot_general(g, a_bf16, (((0,), (0,)), ((), ()))).astype(jnp.bfloat16)

  # Return updated scale/history as "gradients" — the optimizer overwrites
  # _overwrite_with_gradient variables with these values
  return (da, db,
          new_input_scale, new_kernel_scale,
          new_input_history, new_kernel_history)


_aiter_fp8_gemm_delayed.defvjp(_aiter_fp8_gemm_delayed_fwd, _aiter_fp8_gemm_delayed_bwd)


class AiterFp8DelayedDotGeneralOp(nn.Module):
  """Drop-in dot_general replacement using AITER FP8 GEMM with delayed scaling.

  Uses TE-style delayed scaling following the Flax fp8_ops.in_qdq pattern:
  the FP8 scale is computed from a rolling amax_history, and the backward
  pass returns updated scale/history as "gradients" so the optimizer's
  _overwrite_with_gradient mechanism carries them across training steps.

  This provides:
    1. Smooth scale transitions between steps (scale from history, not current tensor)
    2. Consistent forward/backward behavior
    3. Proper state persistence via the gradient mechanism (no mutable collections needed)

  Backward uses BF16 GEMM (STE pattern via lax.dot_general).
  """
  amax_history_length: int = AITER_FP8_AMAX_HISTORY_LEN

  @nn.compact
  def __call__(self, inputs, kernel, dimension_numbers, precision=None, **kwargs):
    OVERWRITE_WITH_GRADIENT = "_overwrite_with_gradient"

    scale_args = (
        flax_initializers.ones_init(), jax.random.PRNGKey(0),
        (1,), jnp.float32,
    )
    amax_history_args = (
        flax_initializers.zeros_init(), jax.random.PRNGKey(0),
        (self.amax_history_length,), jnp.float32,
    )

    # Delayed scaling state — persisted via _overwrite_with_gradient
    input_scale = self.variable(OVERWRITE_WITH_GRADIENT, "input_scale", *scale_args)
    kernel_scale = self.variable(OVERWRITE_WITH_GRADIENT, "kernel_scale", *scale_args)
    input_history = self.variable(OVERWRITE_WITH_GRADIENT, "input_amax_history", *amax_history_args)
    kernel_history = self.variable(OVERWRITE_WITH_GRADIENT, "kernel_amax_history", *amax_history_args)

    # --- Reshape inputs to 2D for GEMM ---
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    lhs_contract = tuple(lhs_contract)
    rhs_contract = tuple(rhs_contract)

    inp_shape = inputs.shape
    ker_shape = kernel.shape
    inp_ndim = len(inp_shape)
    ker_ndim = len(ker_shape)

    m_axes = [i for i in range(inp_ndim) if i not in lhs_contract]
    n_axes = [i for i in range(ker_ndim) if i not in rhs_contract]

    K = 1
    for ax in lhs_contract:
      K *= inp_shape[ax]
    M = 1
    for ax in m_axes:
      M *= inp_shape[ax]
    N = 1
    for ax in n_axes:
      N *= ker_shape[ax]

    perm_lhs = m_axes + list(lhs_contract)
    is_identity_lhs = perm_lhs == list(range(inp_ndim))
    if is_identity_lhs:
      a_2d = jnp.reshape(inputs, (M, K))
    else:
      a_2d = jnp.reshape(jnp.transpose(inputs, perm_lhs), (M, K))

    perm_rhs = n_axes + list(rhs_contract)
    is_identity_rhs = perm_rhs == list(range(ker_ndim))
    if is_identity_rhs:
      b_nk = jnp.reshape(kernel, (N, K))
    else:
      b_nk = jnp.reshape(jnp.transpose(kernel, perm_rhs), (N, K))

    a_bf16 = a_2d.astype(jnp.bfloat16)
    b_bf16 = b_nk.astype(jnp.bfloat16)

    # --- Forward with delayed scaling ---
    # Scale/history variables participate in jax.grad via custom_vjp.
    # The backward returns updated values as "gradients" for _overwrite_with_gradient.
    from jax_aiter.gemm_fp8 import fp8_supported_for_shape
    if fp8_supported_for_shape(M, N, K):
      out_2d = _aiter_fp8_gemm_delayed(
          a_bf16, b_bf16,
          input_scale.value, kernel_scale.value,
          input_history.value, kernel_history.value)
    else:
      from jax_aiter.gemm import gemm as aiter_gemm
      out_2d = aiter_gemm(a_bf16, b_bf16)

    out_m_shape = tuple(inp_shape[ax] for ax in m_axes)
    out_n_shape = tuple(ker_shape[ax] for ax in n_axes)
    return jnp.reshape(out_2d, out_m_shape + out_n_shape)


@dataclass
class AiterFp8Quantization(Quantization):
  """AITER FP8 block-scale GEMM for AMD MI350 (gfx950) with per-call scaling.

  Forward in FP8 (2x compute throughput), backward in BF16 (accuracy).
  Set quantization='aiter_fp8' to enable.
  """

  quant_mode = "train"

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    return AiterFp8DotGeneralOp


@dataclass
class AiterFp8DelayedQuantization(Quantization):
  """AITER FP8 GEMM with TE-style delayed scaling for AMD MI350 (gfx950).

  Uses rolling amax history to compute smooth FP8 scales across training steps.
  This prevents NaN from quantization error accumulation through deep
  transformer stacks (32+ layers) at large batch sizes.

  Forward in FP8 with delayed scales, backward in BF16 (accuracy).
  Set quantization='aiter_fp8_delayed' to enable.
  """

  quant_mode = "train"

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    return AiterFp8DelayedDotGeneralOp


class AiterNanooFp8DotGeneralOp(nn.Module):
  """Drop-in dot_general using AITER FP8 GEMM with module_context gating.

  Follows the BF16-style architecture:
    Forward + dA: AITER FP8 FFI (auto-dispatches ASM vs CK by shape)
    dB: lax.dot_general -> hipBLASLt (native XLA, zero overhead)

  Scope control via AITER_NANOO_FP8_SCOPE env var:
    "all"      — FP8 for all projections (may NaN from error accumulation)
    "mlp"      — FP8 for MLP only (gate, up, down), BF16 for attention
    "mlp_down" — FP8 for down_proj only (most conservative, proven stable)
    Default: "mlp" (balances throughput and stability)

  Non-MLP projections always use BF16 AITER ASM GEMM.
  """
  module_context: str = ""

  @nn.compact
  def __call__(self, inputs, kernel, dimension_numbers, precision=None, **kwargs):
    import os
    from jax_aiter.gemm_fp8 import gemm_fp8
    from jax_aiter.gemm import gemm as bf16_gemm

    scope = os.environ.get("AITER_NANOO_FP8_SCOPE", "mlp").lower()

    if scope == "all":
      use_fp8 = True
    elif scope == "mlp":
      use_fp8 = "mlp" in self.module_context
    elif scope == "mlp_down":
      use_fp8 = self.module_context == "mlp_down"
    else:
      allowed = set(p.strip() for p in scope.split(","))
      use_fp8 = self.module_context in allowed

    depth_limit = os.environ.get("AITER_NANOO_FP8_DEPTH", "")
    if use_fp8 and depth_limit:
      import re
      max_layer = int(depth_limit)
      scope_path = "/".join(self.scope.path)
      layer_match = re.search(r'layers_(\d+)', scope_path)
      if layer_match:
        layer_idx = int(layer_match.group(1))
        if layer_idx >= max_layer:
          use_fp8 = False

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    lhs_contract = tuple(lhs_contract)
    rhs_contract = tuple(rhs_contract)

    inp_shape = inputs.shape
    ker_shape = kernel.shape
    inp_ndim = len(inp_shape)
    ker_ndim = len(ker_shape)

    m_axes = [i for i in range(inp_ndim) if i not in lhs_contract]
    n_axes = [i for i in range(ker_ndim) if i not in rhs_contract]

    K = 1
    for ax in lhs_contract:
      K *= inp_shape[ax]
    M = 1
    for ax in m_axes:
      M *= inp_shape[ax]
    N = 1
    for ax in n_axes:
      N *= ker_shape[ax]

    perm_lhs = m_axes + list(lhs_contract)
    is_identity_lhs = perm_lhs == list(range(inp_ndim))
    if is_identity_lhs:
      a_2d = jnp.reshape(inputs, (M, K))
    else:
      a_2d = jnp.reshape(jnp.transpose(inputs, perm_lhs), (M, K))

    perm_rhs = n_axes + list(rhs_contract)
    is_identity_rhs = perm_rhs == list(range(ker_ndim))
    if is_identity_rhs:
      b_nk = jnp.reshape(kernel, (N, K))
    else:
      b_nk = jnp.reshape(jnp.transpose(kernel, perm_rhs), (N, K))

    a_bf16 = a_2d.astype(jnp.bfloat16)
    b_bf16 = b_nk.astype(jnp.bfloat16)

    if use_fp8:
      out_2d = gemm_fp8(a_bf16, b_bf16)
      out_clip = float(os.environ.get("AITER_FP8_OUT_CLIP", "0"))
      if out_clip > 0:
        out_2d = jnp.clip(out_2d, -out_clip, out_clip)
      clip_scope = os.environ.get("AITER_FP8_OUT_CLIP_SCOPE", "")
      if clip_scope:
        clip_val = float(clip_scope.split(":")[1]) if ":" in clip_scope else 64.0
        clip_projs = set(p.strip() for p in clip_scope.split(":")[0].split(",")) if ":" in clip_scope else {"all"}
        if "all" in clip_projs or self.module_context in clip_projs:
          out_2d = jnp.clip(out_2d, -clip_val, clip_val)
    else:
      out_2d = bf16_gemm(a_bf16, b_bf16)

    out_m_shape = tuple(inp_shape[ax] for ax in m_axes)
    out_n_shape = tuple(ker_shape[ax] for ax in n_axes)
    return jnp.reshape(out_2d, out_m_shape + out_n_shape)


@dataclass
class AiterNanooFp8Quantization(Quantization):
  """Unified AITER FP8 GEMM with hipBLASLt FP8 backward (nanoo-style).

  Forward: AITER FP8 FFI (ASM block-scale or CK per-token, auto-dispatched)
  Backward dA: AITER FP8 FFI (same dispatch)
  Backward dB: native dot_general -> hipBLASLt FP8

  Set quantization='aiter_nanoo_fp8' to enable.
  """

  quant_mode = "train"

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    return AiterNanooFp8DotGeneralOp


# ---------------------------------------------------------------------------
# FP8 Quantize-Dequantize (QDQ) — noise injection with AITER BF16 GEMM compute.
#
# Forward: quantize inputs to FP8 e4m3 → dequantize back to BF16 (noise injection)
#          → compute GEMM using AITER BF16 ASM kernel (not FP8 kernel).
# Backward: STE through QDQ; AITER GEMM handles its own custom_vjp backward.
# Scale/history tracked via _overwrite_with_gradient variables (Flax convention).
# ---------------------------------------------------------------------------

AITER_QDQ_AMAX_HISTORY_LEN = 1024


class AiterFp8QdqDotGeneralOp(nn.Module):
  """Drop-in dot_general replacement using FP8 QDQ noise + AITER BF16 GEMM.

  Quantize-Dequantize (QDQ) injects FP8 quantization noise into BF16 tensors
  before computing the GEMM with AITER's fast BF16 ASM kernel. This gives:
    1. FP8-aware training (model learns robustness to FP8 quantization)
    2. AITER BF16 GEMM performance (~846 TFLOP/s, beats TE by ~0.6%)
    3. No NaN — BF16 compute precision prevents error accumulation

  Scale/history are tracked via _overwrite_with_gradient variables following
  the Flax fp8_ops.in_qdq pattern. The backward pass returns updated
  scale/history as "gradients" so the optimizer carries them across steps.
  """
  amax_history_length: int = AITER_QDQ_AMAX_HISTORY_LEN

  @nn.compact
  def __call__(self, inputs, kernel, dimension_numbers, precision=None, **kwargs):
    from jax_aiter.gemm import gemm as aiter_gemm

    OVERWRITE_WITH_GRADIENT = "_overwrite_with_gradient"

    scale_args = (
        flax_initializers.ones_init(), jax.random.PRNGKey(0),
        (1,), jnp.float32,
    )
    amax_history_args = (
        flax_initializers.zeros_init(), jax.random.PRNGKey(0),
        (self.amax_history_length,), jnp.float32,
    )

    # Delayed scaling state — persisted via _overwrite_with_gradient
    input_scale = self.variable(OVERWRITE_WITH_GRADIENT, "input_scale", *scale_args)
    kernel_scale = self.variable(OVERWRITE_WITH_GRADIENT, "kernel_scale", *scale_args)
    input_history = self.variable(OVERWRITE_WITH_GRADIENT, "input_amax_history", *amax_history_args)
    kernel_history = self.variable(OVERWRITE_WITH_GRADIENT, "kernel_amax_history", *amax_history_args)

    # --- Reshape inputs to 2D for GEMM (same as AiterBf16DotGeneralOp) ---
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    lhs_contract = tuple(lhs_contract)
    rhs_contract = tuple(rhs_contract)

    inp_shape = inputs.shape
    ker_shape = kernel.shape
    inp_ndim = len(inp_shape)
    ker_ndim = len(ker_shape)

    m_axes = [i for i in range(inp_ndim) if i not in lhs_contract]
    n_axes = [i for i in range(ker_ndim) if i not in rhs_contract]

    K = 1
    for ax in lhs_contract:
      K *= inp_shape[ax]
    M = 1
    for ax in m_axes:
      M *= inp_shape[ax]
    N = 1
    for ax in n_axes:
      N *= ker_shape[ax]

    perm_lhs = m_axes + list(lhs_contract)
    is_identity_lhs = perm_lhs == list(range(inp_ndim))
    if is_identity_lhs:
      a_2d = jnp.reshape(inputs, (M, K))
    else:
      a_2d = jnp.reshape(jnp.transpose(inputs, perm_lhs), (M, K))

    perm_rhs = n_axes + list(rhs_contract)
    is_identity_rhs = perm_rhs == list(range(ker_ndim))
    if is_identity_rhs:
      b_nk = jnp.reshape(kernel, (N, K))
    else:
      b_nk = jnp.reshape(jnp.transpose(kernel, perm_rhs), (N, K))

    a_bf16 = a_2d.astype(jnp.bfloat16)
    b_bf16 = b_nk.astype(jnp.bfloat16)

    # --- Apply QDQ noise injection using Flax's in_qdq ---
    # in_qdq: quantize to FP8 e4m3 → dequantize back to BF16 (noise injection)
    # The custom_vjp in in_qdq handles:
    #   - Forward: updates scale from history, does QDQ round-trip
    #   - Backward: STE (passes gradient through), returns updated scale/history
    compute_dtype = jnp.bfloat16
    e4m3_dtype = jnp.float8_e4m3fn

    a_qdq = fp8_ops.in_qdq(
        compute_dtype, e4m3_dtype,
        a_bf16, input_scale.value, input_history.value)
    b_qdq = fp8_ops.in_qdq(
        compute_dtype, e4m3_dtype,
        b_bf16, kernel_scale.value, kernel_history.value)

    # --- Compute GEMM using AITER BF16 ASM kernel ---
    # aiter_gemm expects (M, K) @ (N, K)^T → (M, N) and has its own custom_vjp
    out_2d = aiter_gemm(a_qdq, b_qdq)

    out_m_shape = tuple(inp_shape[ax] for ax in m_axes)
    out_n_shape = tuple(ker_shape[ax] for ax in n_axes)
    return jnp.reshape(out_2d, out_m_shape + out_n_shape)


class AiterFp8MlpOnlyDotGeneralOp(nn.Module):
  """Drop-in dot_general using FP8 GEMM for MLP layers, BF16 GEMM for attention.

  Uses the `module_context` attribute (set by DenseGeneral from the parent
  module context) to decide which kernel to use:
    - module_context contains "mlp" → FP8 GEMM (gate_proj, up_proj, down_proj)
    - Otherwise → BF16 ASM GEMM (q_proj, k_proj, v_proj, o_proj, logits)

  This reduces FP8 error accumulation through the residual chain by roughly
  half (3 FP8 GEMMs per layer instead of 7), while still getting FP8 speedup
  on the largest GEMMs (MLP layers account for ~57% of FLOP).

  Forward: FP8 or BF16 depending on layer type.
  Backward: Always BF16 via the underlying kernel's custom_vjp.
  """
  module_context: str = ""

  @nn.compact
  def __call__(self, inputs, kernel, dimension_numbers, precision=None, **kwargs):
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    lhs_contract = tuple(lhs_contract)
    rhs_contract = tuple(rhs_contract)

    inp_shape = inputs.shape
    ker_shape = kernel.shape
    inp_ndim = len(inp_shape)
    ker_ndim = len(ker_shape)

    m_axes = [i for i in range(inp_ndim) if i not in lhs_contract]
    n_axes = [i for i in range(ker_ndim) if i not in rhs_contract]

    K = 1
    for ax in lhs_contract:
      K *= inp_shape[ax]
    M = 1
    for ax in m_axes:
      M *= inp_shape[ax]
    N = 1
    for ax in n_axes:
      N *= ker_shape[ax]

    perm_lhs = m_axes + list(lhs_contract)
    is_identity_lhs = perm_lhs == list(range(inp_ndim))
    if is_identity_lhs:
      a_2d = jnp.reshape(inputs, (M, K))
    else:
      a_2d = jnp.reshape(jnp.transpose(inputs, perm_lhs), (M, K))

    perm_rhs = n_axes + list(rhs_contract)
    is_identity_rhs = perm_rhs == list(range(ker_ndim))
    if is_identity_rhs:
      b_nk = jnp.reshape(kernel, (N, K))
    else:
      b_nk = jnp.reshape(jnp.transpose(kernel, perm_rhs), (N, K))

    a_bf16 = a_2d.astype(jnp.bfloat16)
    b_bf16 = b_nk.astype(jnp.bfloat16)

    # Decide FP8 vs BF16 based on module_context (set by DenseGeneral from parent).
    # module_context is one of: "mlp_gate", "mlp_up", "mlp_down", "mlp_fused", or ""
    #
    # AITER_FP8_MLP_PROJECTIONS env var controls which projections get FP8.
    # Default (unset or "all"): all MLP projections.
    # Example: "mlp_gate" = only gate_proj, "mlp_gate,mlp_up" = gate + up.
    import os
    fp8_projs = os.environ.get("AITER_FP8_MLP_PROJECTIONS", "all")
    if fp8_projs == "all":
      use_fp8 = "mlp" in self.module_context
    elif fp8_projs.startswith("all_except_"):
      # FP8 for ALL projections (MLP + attention) EXCEPT the listed ones.
      # Example: "all_except_mlp_down" = FP8 for gate, up, q, k, v, o; BF16 for down.
      excluded = set(p.strip() for p in fp8_projs[len("all_except_"):].split(","))
      use_fp8 = self.module_context not in excluded
    else:
      allowed = set(p.strip() for p in fp8_projs.split(","))
      use_fp8 = self.module_context in allowed

    if use_fp8:
      from jax_aiter.gemm_fp8 import gemm_fp8_mi350 as aiter_fp8_gemm
      from jax_aiter.gemm_fp8 import fp8_supported_for_shape
      from jax_aiter.gemm import gemm as aiter_gemm
      # Clip activations BEFORE entering FP8 custom_vjp boundary.
      # This ensures JAX autograd applies the clip gradient mask
      # (zero gradient where |x| >= threshold), preventing weight
      # explosion from extreme activation outliers at the gated MLP
      # boundary (silu(gate)*up) and other FP8 GEMM inputs.
      _act_clip = float(os.environ.get("AITER_FP8_ACT_CLIP", "0"))
      # AITER_FP8_ACT_CLIP_PROJS controls which projections get the clip.
      # Default "all" = all MLP projections. E.g. "mlp_down" or "mlp_down,mlp_gate".
      _clip_projs = os.environ.get("AITER_FP8_ACT_CLIP_PROJS", "all")
      if _act_clip > 0:
        _should_clip = (_clip_projs == "all" and "mlp" in self.module_context) or \
                       (self.module_context in set(p.strip() for p in _clip_projs.split(",")))
        if _should_clip:
          clip_val = jnp.asarray(_act_clip, dtype=a_bf16.dtype)
          a_bf16 = jax.lax.clamp(-clip_val, a_bf16, clip_val)

      # --- Per-projection internal FWD_CLIP override ---
      # AITER_FP8_FWD_CLIP_DOWN: If set, overrides AITER_FP8_FWD_CLIP for
      # mlp_down only. This clips silu(gate)*up INSIDE the custom_vjp
      # boundary (invisible to outer autograd), enabling raw da (no a_bf16
      # dependency) while still protecting the FP8 quantization from outliers.
      # The override is set at trace time, so each projection bakes in its
      # own clip value.
      _fwd_clip_down = os.environ.get("AITER_FP8_FWD_CLIP_DOWN", "")
      _saved_fwd_clip = None
      if _fwd_clip_down and self.module_context == "mlp_down":
        _saved_fwd_clip = os.environ.get("AITER_FP8_FWD_CLIP", None)
        os.environ["AITER_FP8_FWD_CLIP"] = _fwd_clip_down

      # Pre-dispatch: check shape support BEFORE entering FP8 custom_partitioning.
      fp8_ok = fp8_supported_for_shape(M, N, K)
      if os.environ.get("AITER_FP8_LOG_DISPATCH", "0") == "1":
        print(f"FP8_DISPATCH: context={self.module_context} fp8={fp8_ok} M={M} N={N} K={K}"
              f" fwd_clip={os.environ.get('AITER_FP8_FWD_CLIP', 'unset')}")
      if fp8_ok:
        out_2d = aiter_fp8_gemm(a_bf16, b_bf16)
      else:
        out_2d = aiter_gemm(a_bf16, b_bf16)

      # Restore original FWD_CLIP after trace
      if _saved_fwd_clip is not None:
        os.environ["AITER_FP8_FWD_CLIP"] = _saved_fwd_clip
      elif _fwd_clip_down and self.module_context == "mlp_down":
        if "AITER_FP8_FWD_CLIP" in os.environ and _saved_fwd_clip is None:
          del os.environ["AITER_FP8_FWD_CLIP"]
    else:
      from jax_aiter.gemm import gemm as aiter_gemm
      if os.environ.get("AITER_FP8_LOG_DISPATCH", "0") == "1":
        print(f"FP8_DISPATCH: context={self.module_context} fp8=False(not_mlp) M={M} N={N} K={K}")
      out_2d = aiter_gemm(a_bf16, b_bf16)

    # --- NaN guard (AITER_FP8_NAN_GUARD) ---
    # "1" = Replace NaN with 0 in FP8 output
    # "2" = Force FP8 output to all zeros (graph structure test)
    nan_guard = os.environ.get("AITER_FP8_NAN_GUARD", "0")
    if nan_guard == "1" and use_fp8:
      out_2d = jnp.where(out_2d == out_2d, out_2d, jnp.zeros_like(out_2d))
    elif nan_guard == "2" and use_fp8:
      out_2d = jnp.zeros_like(out_2d)

    out_m_shape = tuple(inp_shape[ax] for ax in m_axes)
    out_n_shape = tuple(ker_shape[ax] for ax in n_axes)
    return jnp.reshape(out_2d, out_m_shape + out_n_shape)


@dataclass
class AiterFp8MlpOnlyQuantization(Quantization):
  """AITER FP8 for MLP layers only, BF16 for attention (AMD MI350 gfx950).

  Uses FP8 GEMM for MLP projections (gate, up, down — ~57% of FLOPs) and
  AITER BF16 ASM GEMM for attention projections (q, k, v, o — ~28% of FLOPs).
  This halves the number of consecutive FP8 GEMMs in the residual chain
  (3 per layer instead of 7), preventing NaN from quantization error
  accumulation through 32 transformer layers at large batch sizes.

  Expected throughput: between FP8-all (~866 TFLOP/s) and BF16-all (~846),
  targeting above TE baseline (841 TFLOP/s).

  Set quantization='aiter_fp8_mlp_only' to enable.
  """

  quant_mode = "train"

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    return AiterFp8MlpOnlyDotGeneralOp


@dataclass
class AiterFp8QdqQuantization(Quantization):
  """AITER FP8 QDQ (Quantize-Dequantize) for AMD MI350 (gfx950).

  Injects FP8 quantization noise into BF16 tensors before computing with
  AITER's fast BF16 ASM GEMM kernel. This provides:
    1. FP8-aware training (model learns robustness to quantization)
    2. Higher throughput than TE baseline via AITER BF16 GEMM
    3. Convergence at all batch sizes (no NaN from error accumulation)

  Set quantization='aiter_fp8_qdq' to enable.
  """

  quant_mode = "train"

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    return AiterFp8QdqDotGeneralOp


class AiterFp4DotGeneralOp(nn.Module):
  """Drop-in dot_general using AITER FP4 (MXFP4) ASM GEMM.

  Single FP4 recipe (TE parity, MI350/MI355X gfx950):

    Forward:  Activation and weight are dual-cast to MXFP4 (E2M1 packed FP4
              + E8M0 per-block scales) via the fused ``CastMxfp4DualJA`` HIP
              kernel. The FP4 GEMM is ``GemmFp4FwdJA``.
    Backward: dA via FP4 ASM with the saved columnwise weight; dB via FP4
              wgrad GEMM (NT layout) with FSDP-aware ``jax.lax.psum``
              sharding. ``grad_out`` is dual-cast WITH Hadamard transform
              applied inside the fused HIP kernel (decorrelates outliers in
              the gradient distribution).

  FP4 is applied to all projections whose contraction K is a multiple of 64.
  Attention projections (Q/K/V/O) opt in via ``AITER_FP4_ATTN=1``; without it
  they use AITER BF16 ASM. MLP projections always use FP4.

  Set ``quantization='aiter_fp4'`` to enable.
  """
  module_context: str = ""

  @nn.compact
  def __call__(self, inputs, kernel, dimension_numbers, precision=None, **kwargs):
    import os

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    lhs_contract = tuple(lhs_contract)
    rhs_contract = tuple(rhs_contract)

    inp_shape = inputs.shape
    ker_shape = kernel.shape
    inp_ndim = len(inp_shape)
    ker_ndim = len(ker_shape)

    m_axes = [i for i in range(inp_ndim) if i not in lhs_contract]
    n_axes = [i for i in range(ker_ndim) if i not in rhs_contract]

    K = 1
    for ax in lhs_contract:
      K *= inp_shape[ax]
    M = 1
    for ax in m_axes:
      M *= inp_shape[ax]
    N = 1
    for ax in n_axes:
      N *= ker_shape[ax]

    perm_lhs = m_axes + list(lhs_contract)
    if perm_lhs == list(range(inp_ndim)):
      a_2d = jnp.reshape(inputs, (M, K))
    else:
      a_2d = jnp.reshape(jnp.transpose(inputs, perm_lhs), (M, K))

    perm_rhs = n_axes + list(rhs_contract)
    if perm_rhs == list(range(ker_ndim)):
      b_nk = jnp.reshape(kernel, (N, K))
    else:
      b_nk = jnp.reshape(jnp.transpose(kernel, perm_rhs), (N, K))

    a_bf16 = a_2d.astype(jnp.bfloat16)
    b_bf16 = b_nk.astype(jnp.bfloat16)

    fp4_attn = os.environ.get("AITER_FP4_ATTN", "0") == "1"
    use_fp4 = K % 64 == 0 and ("mlp" in self.module_context or fp4_attn)

    if use_fp4:
      from jax_aiter.gemm_fp4 import gemm_fp4_bf16 as aiter_fp4_gemm
      out_2d = aiter_fp4_gemm(a_bf16, b_bf16)
    else:
      from jax_aiter.gemm import gemm as aiter_gemm
      out_2d = aiter_gemm(a_bf16, b_bf16)

    out_m_shape = tuple(inp_shape[ax] for ax in m_axes)
    out_n_shape = tuple(ker_shape[ax] for ax in n_axes)
    return jnp.reshape(out_2d, out_m_shape + out_n_shape)


@dataclass
class AiterFp4Quantization(Quantization):
  """AITER FP4 (MXFP4) for AMD MI350/MI355X gfx950.

  All MLP projections (gate, up, down) use FP4 ASM GEMM. Attention
  projections (Q, K, V, O) use FP4 too when ``AITER_FP4_ATTN=1``. Forward
  + dA + dB all run in FP4 (no FP8 fallback). ``grad_out`` is cast with
  Hadamard transform (TE parity) for tighter convergence.

  Set ``quantization='aiter_fp4'`` to enable.
  """

  quant_mode = "train"

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    return AiterFp4DotGeneralOp


def _get_int8_quant_config(config):
  drhs_bits = None
  drhs_accumulator_dtype = None
  drhs_local_aqt = None
  if config.quantization_local_shard_count != 0:
    drhs_bits = 8
    drhs_accumulator_dtype = jnp.int32
    drhs_local_aqt = aqt_config.LocalAqt(contraction_axis_shard_count=config.quantization_local_shard_count)
  return aqt_config.config_v3(
      fwd_bits=8,
      dlhs_bits=8,
      drhs_bits=drhs_bits,
      rng_type="jax.uniform",
      dlhs_local_aqt=None,
      drhs_local_aqt=drhs_local_aqt,
      fwd_accumulator_dtype=jnp.int32,
      dlhs_accumulator_dtype=jnp.int32,
      drhs_accumulator_dtype=drhs_accumulator_dtype,
  )


@dataclass(frozen=True)
class ConstantBoundConfig:
  fwd_lhs_bound: float | None = None
  fwd_rhs_bound: float | None = None
  dlhs_lhs_bound: float | None = None
  dlhs_rhs_bound: float | None = None
  drhs_lhs_bound: float | None = None
  drhs_rhs_bound: float | None = None


def _build_const_scale_config(
    aqt_dg: aqt_config.DotGeneral,
    cst_bound_config: ConstantBoundConfig,
) -> aqt_config.DotGeneral:
  """Build a constant scale config for AQT dot general.

  Args:
    aqt_dg: The AQT dot general config.
    cst_bound_config: The constant bound config.

  Returns:
    The AQT dot general config with constant scale config.
  """
  if cst_bound_config.fwd_lhs_bound is not None:
    aqt_dg.fwd.dg_quantizer.lhs.calibration = functools.partial(
        calibration.ConstantCalibration, bound=cst_bound_config.fwd_lhs_bound
    )
  if cst_bound_config.fwd_rhs_bound is not None:
    aqt_dg.fwd.dg_quantizer.rhs.calibration = functools.partial(
        calibration.ConstantCalibration, bound=cst_bound_config.fwd_rhs_bound
    )
  if cst_bound_config.dlhs_lhs_bound:
    aqt_dg.dlhs.dg_quantizer.lhs.calibration = functools.partial(
        calibration.ConstantCalibration, bound=cst_bound_config.dlhs_lhs_bound
    )

  if cst_bound_config.dlhs_rhs_bound is not None:
    aqt_dg.dlhs.dg_quantizer.rhs.calibration = functools.partial(
        calibration.ConstantCalibration, bound=cst_bound_config.dlhs_rhs_bound
    )

  if cst_bound_config.drhs_lhs_bound is not None:
    aqt_dg.drhs.dg_quantizer.lhs.calibration = functools.partial(
        calibration.ConstantCalibration, bound=cst_bound_config.drhs_lhs_bound
    )

  if cst_bound_config.drhs_rhs_bound is not None:
    aqt_dg.drhs.dg_quantizer.rhs.calibration = functools.partial(
        calibration.ConstantCalibration, bound=cst_bound_config.drhs_rhs_bound
    )

  return aqt_dg


@dataclass
class PerTensorScales:
  fwd_lhs: bool = False
  fwd_rhs: bool = False
  dlhs_lhs: bool = False
  dlhs_rhs: bool = False
  drhs_lhs: bool = False
  drhs_rhs: bool = False


def _build_per_tensor_config(
    aqt_dg: aqt_config.DotGeneral,
    per_tensor_scales: PerTensorScales,
) -> aqt_config.DotGeneral:
  """Build a per tensor config for AQT dot general.

  Args:
    aqt_dg: The AQT dot general config.
    per_tensor_scales: The per tensor scales config.

  Returns:
    The AQT dot general config with per tensor config.
  """
  if per_tensor_scales.fwd_lhs:
    aqt_dg.fwd.dg_quantizer.lhs.calib_shared_axes = "per_tensor"
  if per_tensor_scales.fwd_rhs:
    aqt_dg.fwd.dg_quantizer.rhs.calib_shared_axes = "per_tensor"
  if per_tensor_scales.dlhs_lhs:
    aqt_dg.dlhs.dg_quantizer.lhs.calib_shared_axes = "per_tensor"
  if per_tensor_scales.dlhs_rhs:
    aqt_dg.dlhs.dg_quantizer.rhs.calib_shared_axes = "per_tensor"
  if per_tensor_scales.drhs_lhs:
    aqt_dg.drhs.dg_quantizer.lhs.calib_shared_axes = "per_tensor"
  if per_tensor_scales.drhs_rhs:
    aqt_dg.drhs.dg_quantizer.rhs.calib_shared_axes = "per_tensor"
  return aqt_dg


# fp8 training recipe of dynamic scaling with configurable constant_bound_config for static scaling option
def _get_aqt_fp8_default_config(config):
  """Get aqt for 8-bit floating point quantization configuration."""
  aqt_dg = aqt_config.config_v4(
      fwd_bits="e4m3",
      dlhs_bits="e5m2",
      drhs_bits="e5m2",
      use_dummy_static_bound=False,
      fwd_accumulator_dtype=jnp.bfloat16,
      dlhs_accumulator_dtype=jnp.bfloat16,
      drhs_accumulator_dtype=jnp.bfloat16,
      dlhs_use_fwd_quant=False,
      drhs_use_fwd_quant=False,
  )
  constant_bound_config = None

  if len(config.constant_bound_config) == 6:
    fwd_lhs_bound, fwd_rhs_bound, dlhs_lhs_bound, dlhs_rhs_bound, drhs_lhs_bound, drhs_rhs_bound = (
        config.constant_bound_config
    )
    constant_bound_config = ConstantBoundConfig(
        fwd_lhs_bound=fwd_lhs_bound,
        fwd_rhs_bound=fwd_rhs_bound,
        dlhs_lhs_bound=dlhs_lhs_bound,
        dlhs_rhs_bound=dlhs_rhs_bound,
        drhs_lhs_bound=drhs_lhs_bound,
        drhs_rhs_bound=drhs_rhs_bound,
    )
    aqt_dg = _build_const_scale_config(aqt_dg, constant_bound_config)

  aqt_config.set_stochastic_rounding(
      aqt_dg,
      vjp_lhs_stochastic_rounding=False,
      vjp_rhs_stochastic_rounding=False,
      implementation="jax.uniform",
  )

  per_tensor_scales = PerTensorScales(
      fwd_lhs=True,
      fwd_rhs=True,
      dlhs_lhs=True,
      dlhs_rhs=True,
      drhs_lhs=True,
      drhs_rhs=True,
  )
  return _build_per_tensor_config(aqt_dg, per_tensor_scales)


def _get_aqt_fp8_quant_config(config):
  """get aqt for 8-bit floating point quantization configuration"""
  cfg = aqt_config.config_v4(fwd_bits="e4m3", dlhs_bits=None, drhs_bits=None, fwd_accumulator_dtype=jnp.bfloat16)
  return cfg


def _dot_general_make(quant_cfg):
  """Create quantization configs for input matrices to a matmul"""
  lhs_bits = quant_cfg[_A_BITS]
  lhs_scale = quant_cfg[_A_SCALE]
  rhs_bits = quant_cfg[_W_BITS]
  rhs_scale = quant_cfg[_W_SCALE]
  aqt_dg = aqt_config.dot_general_make(lhs_bits=lhs_bits, rhs_bits=rhs_bits)
  if lhs_scale < 1.0:
    aqt_dg.fwd.dg_quantizer.lhs.calibration = functools.partial(calibration.AbsMaxCalibration, scale=lhs_scale)
  if rhs_scale < 1.0:
    aqt_dg.fwd.dg_quantizer.rhs.calibration = functools.partial(calibration.AbsMaxCalibration, scale=rhs_scale)
  return aqt_dg


def _get_default_mp_config(default=None):
  default_config = {_W_BITS: None, _A_BITS: None, _W_SCALE: 1.0, _A_SCALE: 1.0, _TILE_SIZE: -1}
  if default:
    default_config.update(default)
  return default_config


def _get_mixed_precision_quant_config(mixed_precision_config):
  """Set quantization params based on user configuration."""
  ret_config = {}
  default_mp_config = _get_default_mp_config(default=mixed_precision_config.get(DEFAULT, None))
  for layer_name_re, layer_quantization_config in mixed_precision_config.items():
    # Make a copy of default_mp_config to avoid updating original dict
    quant_config = default_mp_config.copy()
    # print(f"Mixed precision config: processing
    # {layer_name_re} - {layer_quantization_config}, default config - {quant_config}")
    if layer_name_re != DEFAULT:
      for k in quant_config:
        quant_config[k] = layer_quantization_config.get(k, default_mp_config[k])
    ret_config[layer_name_re] = [_dot_general_make(quant_config), quant_config["tile_size"]]
  return ret_config


def _get_quant_config(config):
  """Set quantization params based on user configuration."""
  if not config.quantization or config.quantization == "":
    return None
  if config.quantization == "int8":
    return _get_int8_quant_config(config)
  if config.quantization == "intmp":
    assert config.quant_cfg_path, "Must specify quant_cfg for mixed precision quantization"
    with open(config.quant_cfg_path, "rt", encoding="utf8") as config_file:
      mixed_precision_config = json.load(config_file)
    return _get_mixed_precision_quant_config(mixed_precision_config)
  if config.quantization == "fp8":
    return "fp8"
  if config.quantization == "nanoo_fp8":
    return "nanoo_fp8"
  if config.quantization == "aiter_bf16":
    return "aiter_bf16"
  if config.quantization == "aiter_fp8":
    return "aiter_fp8"
  if config.quantization == "aiter_fp8_delayed":
    return "aiter_fp8_delayed"
  if config.quantization == "aiter_fp8_qdq":
    return "aiter_fp8_qdq"
  if config.quantization == "aiter_fp8_mlp_only":
    return "aiter_fp8_mlp_only"
  if config.quantization == "aiter_nanoo_fp8":
    return "aiter_nanoo_fp8"
  if config.quantization == "aiter_fp4":
    return "aiter_fp4"
  if config.quantization == "aqt_fp8":
    return _get_aqt_fp8_quant_config(config)
  if config.quantization == "aqt_fp8_full":
    return _get_aqt_fp8_default_config(config)
  if config.quantization.startswith("te_"):
    return config.quantization

  raise ValueError(f"Invalid value configured for quantization {config.quantization}.")


def in_convert_mode(quant):
  return quant and (quant.quant_mode == aqt_flax.QuantMode.CONVERT)


def in_serve_mode(quant):
  return quant and (quant.quant_mode == aqt_flax.QuantMode.SERVE)


def get_quant_mode(quant_mode_str: str = "train"):
  """Set quant mode."""
  if quant_mode_str == "train":
    return aqt_flax.QuantMode.TRAIN
  elif quant_mode_str == "serve":
    return aqt_flax.QuantMode.SERVE
  elif quant_mode_str == "convert":
    return aqt_flax.QuantMode.CONVERT
  else:
    raise ValueError(f"Invalid quantization mode {quant_mode_str}.")
  return None


def configure_quantization(config: Config, quant_mode_str: str = "train"):
  """Configure quantization based on user config and quant mode."""
  if config.use_qwix_quantization:
    return None
  quant_cfg = _get_quant_config(config)
  if quant_cfg:
    if quant_cfg == "fp8":
      return Fp8Quantization()
    elif quant_cfg == "nanoo_fp8":
      return NANOOFp8Quantization()
    elif quant_cfg == "aiter_bf16":
      return AiterBf16Quantization()
    elif quant_cfg == "aiter_fp8":
      return AiterFp8Quantization()
    elif quant_cfg == "aiter_fp8_delayed":
      return AiterFp8DelayedQuantization()
    elif quant_cfg == "aiter_fp8_qdq":
      return AiterFp8QdqQuantization()
    elif quant_cfg == "aiter_fp8_mlp_only":
      return AiterFp8MlpOnlyQuantization()
    elif quant_cfg == "aiter_nanoo_fp8":
      return AiterNanooFp8Quantization()
    elif quant_cfg == "aiter_fp4":
      return AiterFp4Quantization()
    elif isinstance(quant_cfg, str) and quant_cfg.startswith("te_"):
      return TransformerEngineQuantization(config)
    quant_mode = get_quant_mode(quant_mode_str)
    replicate_scale = config.replicate_quant_scale if config.replicate_quant_scale else False
    return AqtQuantization(quant_dg=quant_cfg, quant_mode=quant_mode, replicate_scale=replicate_scale)
  return None


def match_aqt_and_unquantized_param(aqt_params, params):
  """match aqt and unquantized params"""
  aqt_param_flat, aqt_tree_def = jax.tree_util.tree_flatten_with_path(
      aqt_params, is_leaf=lambda x: isinstance(x, aqt_tensor.QTensor)
  )
  param_tree_flat, _ = jax.tree_util.tree_flatten_with_path(params)
  aqt_paths = []
  # Original path of quantized AQT param path.
  param_paths = []

  for aqt_k, _ in aqt_param_flat:
    index = None
    for index, (k, _) in enumerate(param_tree_flat):
      path_depth = len(k)
      # every quantized parameter has AQT.. as the leaf node
      # AqtDotGeneral and AqtEinsum replace leaf node.
      # Therefore, leaf node should be ignored for path matching
      # Note: Aqt only operates on kernels so don't pop bias parameters.
      # Ref: https://github.com/AI-Hypercomputer/maxtext/compare/main...quantize_r1
      if k[: path_depth - 1] == aqt_k[: path_depth - 1] and k[-1].key != "bias":
        aqt_paths.append(aqt_k)
        param_paths.append(k)
        break
    assert index is not None
    # since the parameter is already added, we can delete it.
    param_tree_flat.pop(index)
  return jax.tree_util.tree_unflatten(aqt_tree_def, param_paths)


def _get_aqt_key_paths(aqt_vars, params):
  """Generate a list of paths which have aqt state"""
  aqt_to_unquantized_key_path = match_aqt_and_unquantized_param(aqt_vars, params)
  aqt_key_paths, _ = jax.tree_util.tree_flatten(aqt_to_unquantized_key_path, is_leaf=lambda x: isinstance(x, tuple))
  return list(aqt_key_paths)


def remove_quantized_params(params, aqt_vars):
  """Remove param values with aqt tensors to Null to optimize memory."""
  quantized_param_paths = _get_aqt_key_paths(aqt_vars, params)
  tree_flat, tree_struct = tree_flatten_with_path(params)
  for i, (k, v) in enumerate(tree_flat):
    if k in quantized_param_paths:
      v = {}
    tree_flat[i] = v
  return tree_unflatten(tree_struct, tree_flat)


def configure_kv_quant(config):
  return None if not config.quantize_kvcache else KVQuant(config)


class NvidaFp8Provider(qwix.QtProvider):
  """Wraps nn.Fp8DirectDotGeneralOp with Qwix's provider interface."""

  def dot_general(self, *args, **kwargs):
    # Here we only check if the rule is None or not.
    rule, op_id = self._get_current_rule_and_op_id("dot_general")
    if rule is None:
      return jax.lax.dot_general(*args, **kwargs)
    return nn.Fp8DirectDotGeneralOp(name=op_id)(*args, **kwargs)

  def einsum(self, *args, **kwargs):
    rule, op_id = self._get_current_rule_and_op_id("einsum")
    if rule is None:
      return jnp.einsum(*args, **kwargs)
    return nn.Fp8Einsum(name=op_id)(*args, **kwargs)


class NANOOFp8Provider(qwix.QtProvider):

  def dot_general(self, *args, **kwargs):
    # Here we only check if the rule is None or not.
    rule, op_id = self._get_current_rule_and_op_id("dot_general")
    if rule is None:
      return jax.lax.dot_general(*args, **kwargs)
    return nn.NANOOFp8DotGeneralOp(name=op_id)(*args, **kwargs)


def get_fp8_full_qwix_rule(config: Config):
  return qwix.QtRule(
      module_path="decoder/.*layers.*",
      weight_qtype=jnp.float8_e4m3fn,
      act_qtype=jnp.float8_e4m3fn,
      bwd_qtype=jnp.float8_e5m2,
      weight_calibration_method=config.weight_quantization_calibration_method,
      act_calibration_method=config.act_quantization_calibration_method,
      bwd_calibration_method=config.bwd_quantization_calibration_method,
      op_names=("dot_general", "gmm", "ragged_dot"),
  )


def get_quantization_rule(config: Config):
  match config.quantization:
    case "int4":
      return qwix.QtRule(
          module_path="decoder/.*layers.*",
          weight_qtype=jnp.int4,
          act_qtype=jnp.int4,
          bwd_qtype=jnp.int4,
          bwd_weight_grad_tile_size=1 / config.quantization_local_shard_count,
          op_names=("dot_general",),
      )
    case "int8":
      return qwix.QtRule(
          module_path="decoder/.*layers.*",
          weight_qtype=jnp.int8,
          act_qtype=jnp.int8,
          bwd_qtype=jnp.int8,
          bwd_weight_grad_tile_size=1 / config.quantization_local_shard_count,
          op_names=("dot_general",),
      )
    case "fp8":
      return qwix.QtRule(
          module_path="decoder/.*layers.*",
          weight_qtype=jnp.float8_e4m3fn,
          act_qtype=jnp.float8_e4m3fn,
          bwd_qtype=jnp.float8_e4m3fn,
          bwd_weight_grad_tile_size=1 / config.quantization_local_shard_count,
          op_names=("dot_general",),
      )
    case "fp8_full":
      return get_fp8_full_qwix_rule(config)
    case "fp8_gpu":
      return qwix.QtRule(
          module_path="decoder/.*layers.*",
          weight_qtype=jnp.float8_e4m3fn,
          act_qtype=jnp.float8_e4m3fn,
          bwd_qtype=jnp.float8_e4m3fn,
          bwd_weight_grad_tile_size=1 / config.quantization_local_shard_count,
          op_names=("dot_general",),
      )
    case "fp8_nanoo":
      return qwix.QtRule(
          module_path="decoder/.*layers.*",
          weight_qtype=jnp.float8_e4m3fn,
          act_qtype=jnp.float8_e4m3fn,
          bwd_qtype=jnp.float8_e4m3fn,
          bwd_weight_grad_tile_size=1 / config.quantization_local_shard_count,
          op_names=("dot_general",),
      )
    case "":
      return None


def get_qt_provider(config):
  """Get quantization rules based on the config."""
  match config.quantization:
    case "int8":
      return qwix.QtProvider([get_quantization_rule(config)])
    case "int4":
      return qwix.QtProvider([get_quantization_rule(config)])
    case "fp8":
      return qwix.QtProvider([get_quantization_rule(config)])
    case "fp8_full":
      return qwix.QtProvider([get_quantization_rule(config)])
    case "fp8_gpu":
      return NvidaFp8Provider([get_quantization_rule(config)])
    case "fp8_nanoo":
      return NANOOFp8Provider([get_quantization_rule(config)])
  return None


def maybe_quantize_model(model, config):
  """Quantize the model if quantization is enabled."""
  if config.use_qwix_quantization:
    quantization_provider = get_qt_provider(config)
    if quantization_provider:
      model = qwix.quantize_model(model, quantization_provider)
  return model


class TransformerEngineQuantization(Quantization):
  """Class for TransformerEngine quantization recipes."""

  def __init__(self, config):
    """Initialize TransformerEngine quantization."""

    self.quant_mode = "train"

    if not config.quantization.startswith("te_"):
      raise ValueError(f"Invalid TransformerEngine quantization config: {config.quantization}")

    self._recipe = TransformerEngineQuantization._get_recipe(config.quantization)

  def __hash__(self):
    return hash((self.quant_mode, self._recipe))

  def __eq__(self, other):
    if not isinstance(other, TransformerEngineQuantization):
      return False
    return (self.quant_mode, self._recipe) == (other.quant_mode, other._recipe)

  @staticmethod
  def _get_recipe(recipe_name: str):
    """Get the TransformerEngine recipe based on the name."""
    from transformer_engine.common import recipe  # pylint: disable=import-outside-toplevel # pytype: disable=import-error

    RECIPES = {
        "te_fp8_delayedscaling": recipe.DelayedScaling,
        "te_fp8_currentscaling": recipe.Float8CurrentScaling,
        "te_mxfp8": recipe.MXFP8BlockScaling,
        "te_nvfp4": recipe.NVFP4BlockScaling,  # pytype: disable=module-attr
        "te_nvfp4_no_rht": functools.partial(recipe.NVFP4BlockScaling, disable_rht=True),  # pytype: disable=module-attr
    }
    if recipe_name not in RECIPES:
      raise ValueError(f"Invalid TransformerEngine recipe: {recipe_name}")
    return RECIPES[recipe_name]()

  def get_block_size(self):
    """Get the block size for quantization for recipes that require blocks.

    If there is no block requirement for the current recipe, returns 1.
    """
    from transformer_engine.common import recipe  # pylint: disable=import-outside-toplevel # pytype: disable=import-error

    if isinstance(self._recipe, recipe.MXFP8BlockScaling):
      return 32
    if isinstance(self._recipe, recipe.NVFP4BlockScaling):  # pytype: disable=module-attr
      return 128  # TODO(set this to 16 when unfused RHT is supported)
    return 1

  def _wrap(self, f, name=None):
    """Wraps the given function `f` to support TransformerEngine quantization.

    This method does a couple things:


    1. Wraps the given function in a context that specifies MaxText's physical mesh axes to
    TransformerEngine. This ensures our collective operations in TransformerEngine are using
    the correct axes.

    2. Wraps the given function in a Flax linen module. This module does not store any Flax
    parameters but can store Flax variables for quantizers if required by the recipe.

    3. When the wrapper is called, it provides an additional argument to the given function `f`,
    'generate_quantizer_set' as the first argument. 'generate_quantizer_set' is a function that
    can be called to generate a TransformerEngine/JAX quantizer set object used in
    TransformerEngine/JAX APIs. 'generate_quantizer_set' will generate quantizers based on the
    recipe of this TransformerEngineQuantizer object.

    Args:
      f: The function to wrap. The first argument must be 'generate_quantizer_set'.
      name: The name of this wrapped operation. If unspecified, will use `f.__name__`.

    Returns:
      A Flax linen module that wraps the given function.
    """

    import transformer_engine.jax  # pylint: disable=import-outside-toplevel # pytype: disable=import-error

    fp8_recipe = self._recipe

    class TEWrapper(transformer_engine.jax.flax.module.TransformerEngineBase):
      """Wrapper module for TransformerEngine quantization."""

      def generate_quantizer_set(self, postfix: str = ""):
        OVERWRITE_WITH_GRADIENT = "_overwrite_with_gradient"
        return super().generate_quantizer_set(  # pytype: disable=wrong-keyword-args
            postfix=postfix,
            variable_collection=OVERWRITE_WITH_GRADIENT,
            quantization_checkpoint_name="quantization",
            fp8_recipe=fp8_recipe,
        )

      @nn.compact
      def __call__(self, *args, **kwargs):
        return f(self.generate_quantizer_set, *args, **kwargs)

    TEWrapper.__name__ = f"TEWrapper_{name if name else f.__name__}"

    return TEWrapper

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Placeholder for dot_general implementation in subclasses."""
    import transformer_engine.jax  # pylint: disable=import-outside-toplevel # pytype: disable=import-error

    def te_dot_general(generate_quantizer_set, x, kernel, dims, **kwargs):
      contracting_dims, batch_dims = dims
      assert batch_dims == ((), ()), "Batch dimensions must be empty for TransformerEngine dot."

      quantizer_set = generate_quantizer_set()
      return transformer_engine.jax.dense.dense(
          x,
          kernel,
          contracting_dims=contracting_dims,
          quantizer_set=quantizer_set,
      )

    return self._wrap(te_dot_general, "dot_general")

  def einsum(self, dtype: DType = jnp.float32):
    """Placeholder for einsum implementation in subclasses."""
    # quant.einsum is only required for MoE or for inference with KVCache.
    raise ValueError("Einsum is not yet supported for TransformerEngine quantization.")
