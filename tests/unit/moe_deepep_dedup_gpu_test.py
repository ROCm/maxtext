# Copyright 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dropless overflow tiers with the real FlyDSL MXFP8 expert step, on one GPU.

The CPU tests use a toy expert. Here the expert is the production one (GLU gate/up GEMM, router weight on the
down-projection input, down GEMM), so this checks what the CPU tests cannot: the tier loop's custom gradient around the
FlyDSL custom calls, in particular that the expert-weight gradients are not silently lost. A run with many small tiers must
agree with one big tier. MXFP8 quantizes per row, so a row's result does not depend on which tier holds it; what differs
is only accumulation order.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("jax_flydsl", reason="requires jax-flydsl on PYTHONPATH")

from jax_flydsl.ops.grouped_gemm_mxfp8 import repack_glu_halves  # pylint: disable=wrong-import-position

from maxtext.integration.flydsl import moe_bridge  # pylint: disable=wrong-import-position
from maxtext.layers import moe_dedup  # pylint: disable=wrong-import-position

R, K, E, H, INTER = 2048, 4, 8, 256, 256  # received rows, slots per row, local experts, hidden, expert width
TOTAL_TIERS = 5  # the small run must need at least this many tiers in all


def _inputs():
  rng = np.random.default_rng(0)
  idx = np.full((R, K), -1, np.int32)
  for r in range(R - 64):  # the last rows are unreal, like DeepEP's unused tail
    n = rng.integers(1, K + 1)
    idx[r, :n] = rng.choice(E, size=n, replace=False)
  recv_x = jnp.asarray(rng.standard_normal((R, H)), jnp.bfloat16)
  weights = jnp.asarray(rng.uniform(0.1, 1.0, (R, K)), jnp.float32)
  w0 = jnp.asarray(rng.standard_normal((E, H, INTER)) / np.sqrt(H), jnp.bfloat16)
  w1 = jnp.asarray(rng.standard_normal((E, H, INTER)) / np.sqrt(H), jnp.bfloat16)
  wo = jnp.asarray(rng.standard_normal((E, INTER, H)) / np.sqrt(INTER), jnp.bfloat16)
  return recv_x, jnp.asarray(idx), weights, repack_glu_halves(w0, w1), wo


def _expert_fn(w_packed, wo):
  def fn(rows, row_weights, group_sizes, *, save_residuals):
    del save_residuals
    offsets = moe_bridge.group_offsets_from_sizes(group_sizes)
    activated = moe_bridge.grouped_gemm_mxfp8_glu(rows, w_packed, offsets, None)
    scaled = activated * row_weights[:, None].astype(activated.dtype)
    return moe_bridge.grouped_gemm_mxfp8_experts(scaled, wo, offsets)

  return fn


def _run(recv_x, idx, weights, w_packed, wo, capacity):
  return moe_dedup.dedup_receive_combine(
      recv_x, idx, weights, _expert_fn(w_packed, wo), num_experts=E, capacity=capacity, dropless=True
  )


def _rel(a, b):
  a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
  return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def test_small_tiers_match_one_big_tier_forward_and_backward():
  recv_x, idx, weights, w_packed, wo = _inputs()
  total = int(jnp.sum(idx >= 0))
  small = total // TOTAL_TIERS + 1
  assert total > (TOTAL_TIERS - 1) * small, "the small run needs several extra tiers"
  cotangent = jnp.asarray(np.random.default_rng(1).standard_normal((R, H)), jnp.float32)

  big, big_dropped, _ = _run(recv_x, idx, weights, w_packed, wo, capacity=R * K)
  tiered, dropped, assignments = _run(recv_x, idx, weights, w_packed, wo, capacity=small)
  assert int(big_dropped) == 0 and int(dropped) == 0 and int(assignments) == total
  assert _rel(tiered, big) < 1e-2, "tiered output differs from the one-tier output"

  def loss(capacity):
    def fn(rx, w, wp, wo_):
      return jnp.sum(_run(rx, idx, w, wp, wo_, capacity)[0].astype(jnp.float32) * cotangent)

    return jax.jit(jax.grad(fn, argnums=(0, 1, 2, 3)))

  want = loss(R * K)(recv_x, weights, w_packed, wo)
  got = loss(small)(recv_x, weights, w_packed, wo)
  for name, g, w in zip(("recv_x", "router_weights", "w_gate_up", "w_down"), got, want):
    assert float(jnp.max(jnp.abs(w))) > 0, f"{name}: reference gradient is all zero, the test is vacuous"
    assert _rel(g, w) < 2e-2, f"{name}: gradient differs between tiered and one-tier runs ({_rel(g, w):.4f})"
