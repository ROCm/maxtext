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

"""Primus-Turbo grouped GEMM for MoE sparse matmul on AMD GPUs."""

import jax
import jax.numpy as jnp

from maxtext.utils import max_logging

BACKENDS = ("ck", "hipblaslt", "flydsl")

_logged = set()


def _import(backend):
  try:
    import primus_turbo.jax  # pylint: disable=import-outside-toplevel

    if backend == "ck":
      from primus_turbo.jax.lax.grouped_gemm import grouped_gemm as fn  # pylint: disable=import-outside-toplevel
    elif backend == "flydsl":
      from primus_turbo.jax.lax.grouped_gemm_flydsl import grouped_gemm_flydsl as fn  # pylint: disable=import-outside-toplevel
    else:
      from primus_turbo.jax.lax.grouped_gemm_hipblaslt import grouped_gemm_hipblaslt as fn  # pylint: disable=import-outside-toplevel
  except ImportError as e:
    raise ImportError(
        f"use_turbo_grouped_gemm=True with backend '{backend}' requires a primus_turbo build that provides it."
    ) from e
  primus_turbo.jax.initialize()
  return fn


def _ck_gmm(fn, lhs, rhs, group_sizes):
  # The CK kernels leave rows past the last group unwritten (ragged-buffer and tile padding), in the
  # output and in the lhs gradient. Masking lhs zeroes the lhs-gradient tail through its transpose.
  rows = jax.lax.broadcasted_iota(jnp.int32, (lhs.shape[0], 1), 0)
  valid = rows < jnp.sum(group_sizes)
  lhs = jnp.where(valid, lhs, jnp.zeros_like(lhs))
  # The CK kernel only accepts int64 group sizes. Global x64 breaks the MoE routing
  # (XLA-ROCm s32/s64 scatter mismatch), so scope x64 to this call.
  with jax.enable_x64(True):
    out = fn(lhs, rhs, group_sizes.astype(jnp.int64), transA=False, transB=False, num_cu=-1)
  return jnp.where(valid, out, jnp.zeros_like(out))


def _fp8_gmm(lhs, rhs, group_sizes, backend):
  """Per-tensor fp8 (e4m3) grouped GEMM with fp8 forward, dA and dW."""
  import primus_turbo.jax  # pylint: disable=import-outside-toplevel

  primus_turbo.jax.initialize()
  if backend == "flydsl":
    from primus_turbo.jax.lax import grouped_gemm_flydsl_fp8 as mod  # pylint: disable=import-outside-toplevel

    fn = mod.grouped_gemm_fp8_flydsl
  elif backend == "hipblaslt":
    from primus_turbo.jax.lax import grouped_gemm_hipblaslt_fp8 as mod  # pylint: disable=import-outside-toplevel

    fn = mod.grouped_gemm_fp8_hipblaslt
  else:
    raise ValueError(f"turbo_grouped_gemm_quant=fp8 needs backend flydsl or hipblaslt, got {backend!r}.")
  if f"fp8-{backend}" not in _logged:
    max_logging.log(f"Using primus_turbo fp8 grouped_gemm ({backend}) in MoE sparse matmul: lhs {lhs.shape}, rhs {rhs.shape}")
    _logged.add(f"fp8-{backend}")
  return fn(lhs, rhs, group_sizes.astype(jnp.int32))


def gmm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, backend: str = "ck", exact_rows: bool = False, quant: str = "bf16") -> jax.Array:
  """Grouped matmul with `jax.lax.ragged_dot` semantics.

  Args:
    lhs: [m, k] rows sorted by group.
    rhs: [num_groups, k, n].
    group_sizes: [num_groups] integer row counts; their sum may be less than m.
    backend: "ck" (CK grouped GEMM kernels) or "hipblaslt" (one hipBLASLt grouped GEMM launch for all
      groups; the group sizes stay on the device, so nothing synchronizes with the host).

  Returns:
    [m, n] in lhs.dtype. Rows past sum(group_sizes) are zero.
  """
  if group_sizes.shape[0] > rhs.shape[0]:
    # ring-of-experts EP: local experts first (roll_to_expert_id), global sizes passed
    group_sizes = group_sizes[: rhs.shape[0]]
  if backend not in BACKENDS:
    raise ValueError(f"Unknown turbo grouped GEMM backend {backend!r}; expected one of {BACKENDS}.")
  if quant == "fp8":
    return _fp8_gmm(lhs, rhs, group_sizes, backend)
  fn = _import(backend)
  if backend not in _logged:
    max_logging.log(f"Using primus_turbo grouped_gemm ({backend}) in MoE sparse matmul: lhs {lhs.shape}, rhs {rhs.shape}")
    _logged.add(backend)
  if lhs.dtype != rhs.dtype or lhs.dtype not in (jnp.bfloat16, jnp.float16):
    raise ValueError(f"primus_turbo grouped_gemm needs matching bf16/fp16 operands, got {lhs.dtype} and {rhs.dtype}.")
  if backend == "ck":
    return _ck_gmm(fn, lhs, rhs, group_sizes)
  if backend == "flydsl":
    # exact_rows: sum(group_sizes) == m, so there are no padding rows to zero (flydsl skips a memset).
    return fn(lhs, rhs, group_sizes.astype(jnp.int32), not exact_rows)
  return fn(lhs, rhs, group_sizes)




def gmm_window(lhs, rhs, group_sizes, group_offset, backend="ck", exact_rows=False, quant="bf16"):
  """megablox-style group_offset: rhs holds experts [group_offset, group_offset + rhs.shape[0]) of the global group_sizes and the rows of those experts sit at
  [sum(group_sizes[:group_offset]), ...) of lhs (ring + ragged sort). Slice the window, run the gmm, write the result back at the same rows (zeros elsewhere)."""
  m = lhs.shape[0]
  e_local = rhs.shape[0]
  gs = group_sizes.astype(jnp.int32)
  row_start = jnp.sum(jax.lax.dynamic_slice_in_dim(jnp.cumsum(gs) - gs, group_offset, 1))
  lhs_pad = jnp.concatenate([lhs, jnp.zeros_like(lhs)], axis=0)
  lhs_loc = jax.lax.dynamic_slice_in_dim(lhs_pad, row_start, m, axis=0)
  gs_loc = jax.lax.dynamic_slice_in_dim(gs, group_offset, e_local)
  out_loc = gmm(lhs_loc, rhs, gs_loc, backend, False, quant)
  out_pad = jax.lax.dynamic_update_slice(jnp.zeros((2 * m, out_loc.shape[1]), out_loc.dtype), out_loc, (row_start, 0))
  return out_pad[:m]
