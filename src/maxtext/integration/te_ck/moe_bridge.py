# SPDX-License-Identifier: Apache-2.0
"""Adapter between MaxText routed-MoE and Transformer Engine's grouped GEMM,
which dispatches to Composable Kernel when NVTE_USE_CK_GROUPED_GEMM=1.

Substitutes for the three per-expert matmuls of the `sparse_matmul` path, the
same three the FlyDSL MXFP8 bridge replaces, so the two backends are measured
over identical routing, activation and unpermute code. Differentiable because
TE's `grouped_dense` carries its own `custom_vjp`: the forward GEMM, the dgrad
and the wgrad all become CK grouped-GEMM launches.

One layout caveat worth stating, since it decides a third of the cost. MaxText
stores every expert kernel contracting-dim-first, wi_* as [E, D, M] and wo as
[E, M, D], which is exactly the [G, K, N] that `grouped_dense` declares, so
nothing is transposed on the way in. But that makes the forward GEMM CK's NN
(RowMajor/RowMajor) instantiation, which measures ~1.35x its NT one at Qwen3
expert shapes; only dgrad (NT) and wgrad (TN) get CK's better kernels. Feeding
the weight in as [G, N, K] instead would put the forward on NT, but
`grouped_dense`'s backward rule only accepts ((1,), (1,)), so that would mean
hand-writing the VJP. Left as is, and the forward penalty is a known term in
the comparison rather than a hidden one.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def grouped_gemm_ck_experts(
    x: jax.Array,
    kernel: jax.Array,
    group_sizes: jax.Array,
) -> jax.Array:
  """One per-expert matmul of the MoE FFN, contracted by TE/CK. Differentiable.

  The grouped GEMM writes each group and nothing else, so when the token buffer
  has more rows than the groups fill -- as it does whenever ragged_buffer_factor
  gives a fixed-capacity receiver -- the tail holds whatever the allocator left
  there. Masking it here rather than at the far end of the FFN keeps that
  garbage from reaching the next GEMM.

  Args:
    x: [tokens, K] this shard's tokens, already stacked by local expert.
    kernel: [G, K, N] MaxText expert weight -- wi_0, wi_1, or wo.
    group_sizes: [G] per-expert token counts, from routing.

  Returns:
    [tokens, N] in the dtype of ``x``.
  """
  from transformer_engine.jax.dense import grouped_dense  # local import: optional dep

  if x.ndim != 2:
    raise ValueError(f"grouped_gemm_ck_experts: x must be 2D [tokens, K], got {x.shape}")
  if kernel.ndim != 3:
    raise ValueError(f"grouped_gemm_ck_experts: kernel must be 3D [G, K, N], got {kernel.shape}")

  sizes = group_sizes.astype(jnp.int32)
  live_rows = jnp.sum(sizes)

  # `grouped_dense` takes sizes, not offsets, and derives each group's start as a
  # running sum -- so it requires the buffer to be exactly filled, and asserts
  # sum(group_sizes) == x.shape[0]. MaxText's receiver is provisioned for the
  # worst case where every token routes to this shard, so under EP it is mostly
  # padding. TE's own message prescribes the remedy: fold the padding into
  # group_sizes. The slack goes on the last group, whose extra rows are
  # multiplied by that expert's weights and then dropped by the mask below.
  #
  # Correct, but not free: TE contracts the padding, where a kernel that takes
  # offsets skips it. At EP=8 the buffer is 8x the live rows, so that is 8x the
  # expert FLOPs -- a property of the packed-layout contract, not of CK itself.
  padded_sizes = sizes.at[-1].add(x.shape[0] - live_rows)

  out = grouped_dense(
      x,
      kernel.astype(x.dtype),
      padded_sizes,
      contracting_dims=((1,), (1,)),
  )
  live = jnp.arange(out.shape[0], dtype=jnp.int32)[:, None] < live_rows
  return jnp.where(live, out, 0).astype(out.dtype)
