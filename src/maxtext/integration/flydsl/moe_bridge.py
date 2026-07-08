# SPDX-License-Identifier: Apache-2.0
"""Adapter between MaxText routed-MoE and the FlyDSL 2-stage grouped GEMM.

MaxText expert weights: wi_0/wi_1 (gate/up) [E, D, M], wo (down) [E, M, D].
FlyDSL wants a fused, MFMA-shuffled layout (see flydsl_moe.preshuffle).
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp


def fly_compute_dtype(out_dtype) -> tuple[str, Any]:
  """Map a MaxText dtype to the FlyDSL MoE compute dtype: ('fp16'|'bf16', jnp dtype).

  fp16 uses the hipBLASLt-comparable f16 MFMA path; anything else -> bf16.
  """
  if jnp.dtype(out_dtype) == jnp.float16:
    return "fp16", jnp.float16
  return "bf16", jnp.bfloat16


def preshuffle_expert_weights(
    wi_0: jax.Array,
    wi_1: jax.Array,
    wo: jax.Array,
    dtype=jnp.bfloat16,
) -> tuple[jax.Array, jax.Array]:
  """MaxText expert weights -> FlyDSL (w1_shuffled, w2_shuffled) in `dtype`."""
  from flydsl_moe.preshuffle import make_w1_shuffled, make_w2_shuffled
  wi_0 = wi_0.astype(dtype)
  wi_1 = wi_1.astype(dtype)
  wo = wo.astype(dtype)
  w1_shuffled = make_w1_shuffled(wi_0, wi_1).astype(dtype)
  w2_shuffled = make_w2_shuffled(wo).astype(dtype)
  return w1_shuffled, w2_shuffled


_FLY_KEYS = ("wi_fly_w1", "wi_fly_w2")


def strip_preshuffle_params(params):
  """Drop the derived wi_fly_* leaves so orbax restore ignores them."""
  if not isinstance(params, dict):
    return params
  return {k: strip_preshuffle_params(v) for k, v in params.items() if k not in _FLY_KEYS}


def fill_preshuffle_params(params, shardings=None):
  """Recompute wi_fly_w1/w2 from each expert block (holding wi_0), in place."""

  def recurse(node, shard):
    if not isinstance(node, dict):
      return
    if "wi_0" in node:
      w1_shuffled, w2_shuffled = preshuffle_expert_weights(node["wi_0"], node["wi_1"], node["wo"])
      if shard is not None:
        w1_shuffled = jax.device_put(w1_shuffled, shard["wi_fly_w1"])
        w2_shuffled = jax.device_put(w2_shuffled, shard["wi_fly_w2"])
      node["wi_fly_w1"] = w1_shuffled
      node["wi_fly_w2"] = w2_shuffled
      return
    for key, child in node.items():
      recurse(child, shard[key] if shard is not None else None)

  recurse(params, shardings)
  return params


def _bf16_fly_params(w1_shuffled: jax.Array, w2_shuffled: jax.Array) -> dict[str, Any]:
  """bf16 fly_params dict; scale_w* are unused but required positionally."""
  return {
      "w1_shuffled": w1_shuffled,
      "scale_w1": jnp.ones((w1_shuffled.shape[0],), jnp.float32),
      "w2_shuffled": w2_shuffled,
      "scale_w2": jnp.ones((w2_shuffled.shape[0],), jnp.float32),
  }


def flydsl_routed_moe(
    inputs: jax.Array,
    top_k_weights: jax.Array,
    top_k_indices: jax.Array,
    w0_kernel: jax.Array,
    w1_kernel: jax.Array,
    wo_kernel: jax.Array,
    *,
    num_experts: int,
    num_experts_per_tok: int,
    out_dtype: jnp.dtype,
    w1_shuffled: jax.Array | None = None,
    w2_shuffled: jax.Array | None = None,
    inter_dim: int | None = None,
) -> jax.Array:
  """Routed-MoE forward via the FlyDSL bf16 2-stage grouped GEMM.

  Pass w1_shuffled/w2_shuffled to use offline-preshuffled weights; if omitted they
  are shuffled inline (correct, but costs a per-step repack).
  """
  from flydsl_moe.block import moe_block_fly_bf16_atomic, moe_sort_jax, pick_tile_m

  compute_dtype, compute_jnp = fly_compute_dtype(out_dtype)

  *lead_shape, model_dim = inputs.shape
  tokens = int(math.prod(lead_shape))
  if inter_dim is None:
    inter_dim = int(w0_kernel.shape[-1])

  x = inputs.reshape(tokens, model_dim).astype(compute_jnp)
  topk_ids = top_k_indices.reshape(tokens, num_experts_per_tok).astype(jnp.int32)
  topk_weights = top_k_weights.reshape(tokens, num_experts_per_tok).astype(jnp.float32)

  tile_m = pick_tile_m(tokens)
  sorted_ids, sorted_weights, expert_ids, num_valid_ids = moe_sort_jax(
      topk_ids,
      topk_weights,
      num_experts=num_experts,
      block_size=tile_m,
  )

  if w1_shuffled is None or w2_shuffled is None:
    w1_shuffled, w2_shuffled = preshuffle_expert_weights(
        w0_kernel, w1_kernel, wo_kernel, dtype=compute_jnp
    )

  out = moe_block_fly_bf16_atomic(
      x,
      _bf16_fly_params(w1_shuffled, w2_shuffled),
      sorted_ids,
      expert_ids,
      sorted_weights,
      num_valid_ids,
      tokens=tokens,
      model_dim=model_dim,
      inter_dim=inter_dim,
      num_experts=num_experts,
      topk=num_experts_per_tok,
      tile_m=tile_m,
      compute_dtype=compute_dtype,
  )
  return out.reshape(*lead_shape, model_dim).astype(out_dtype)
