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

"""Tests for the Qwen3 gated delta-net causal conv and chunked scan."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.models import qwen3
from tests.utils.test_helpers import get_test_config_path

TAPS = 4


def conv_reference(conv_input, kernel, out_len, dtype):
  """What Qwen3NextGatedDeltaNet computed before: a CAUSAL depthwise nnx.Conv in `dtype`."""
  conv = nnx.Conv(
      in_features=kernel.shape[-1],
      out_features=kernel.shape[-1],
      kernel_size=(kernel.shape[0],),
      feature_group_count=kernel.shape[-1],
      padding="CAUSAL",
      use_bias=False,
      dtype=dtype,
      precision=jax.lax.Precision.HIGHEST,
      rngs=nnx.Rngs(0),
  )
  conv.kernel[...] = kernel
  return conv(conv_input)[:, -out_len:, :]


def vjp_all(fn, cotangent, *args):
  out, pullback = jax.vjp(fn, *args)
  return out, *pullback(cotangent)


def test_causal_depthwise_conv1d_matches_nnx_conv():
  channels, seq_len = 96, 128
  kernel = jax.random.normal(jax.random.key(0), (TAPS, 1, channels), jnp.float32)
  x = jax.random.normal(jax.random.key(1), (2, TAPS - 1 + seq_len, channels), jnp.float32)
  cotangent = jax.random.normal(jax.random.key(2), (2, seq_len, channels), jnp.float32)

  expected = jax.jit(lambda x, w: vjp_all(lambda a, b: conv_reference(a, b, seq_len, jnp.float32), cotangent, x, w))(
      x, kernel
  )
  actual = jax.jit(
      lambda x, w: vjp_all(lambda a, b: qwen3.causal_depthwise_conv1d(a, b, seq_len, jnp.float32), cotangent, x, w)
  )(x, kernel)
  for name, e, a in zip(("output", "grad_input", "grad_kernel"), expected, actual, strict=True):
    np.testing.assert_allclose(a, e, rtol=1e-5, atol=1e-5, err_msg=name)

  x_bf16 = x.astype(jnp.bfloat16)
  expected_bf16 = conv_reference(x_bf16, kernel, seq_len, jnp.bfloat16)
  actual_bf16 = qwen3.causal_depthwise_conv1d(x_bf16, kernel, seq_len, jnp.bfloat16)
  assert actual_bf16.dtype == jnp.bfloat16
  np.testing.assert_allclose(
      np.asarray(actual_bf16, np.float32), np.asarray(expected_bf16, np.float32), rtol=2e-2, atol=2e-2
  )


@pytest.mark.parametrize("with_state", [False, True])
def test_chunk_gated_delta_rule_matches_naive_with_gradients(with_state):
  # 160 tokens: two full chunks plus a padded one.
  batch, seq_len, heads, dim = 2, 160, 4, 32
  keys = jax.random.split(jax.random.key(6), 8)
  args = (
      jax.random.normal(keys[0], (batch, seq_len, heads, dim)),
      jax.random.normal(keys[1], (batch, seq_len, heads, dim)),
      jax.random.normal(keys[2], (batch, seq_len, heads, dim)),
      -jax.nn.softplus(jax.random.normal(keys[3], (batch, seq_len, heads))),
      jax.nn.sigmoid(jax.random.normal(keys[4], (batch, seq_len, heads))),
      0.1 * jax.random.normal(keys[5], (batch, heads, dim, dim)) if with_state else None,
  )
  out_cotangent = jax.random.normal(keys[6], (batch, seq_len, heads, dim))
  state_cotangent = jax.random.normal(keys[7], (batch, heads, dim, dim))

  def loss(rule, **kwargs):
    def fn(q, k, v, g, beta, state):
      out, final = rule(q, k, v, g, beta, chunk_size=64, initial_state=state, use_qk_norm_in_gdn=True, **kwargs)
      return jnp.sum(out * out_cotangent) + (jnp.sum(final * state_cotangent) if with_state else 0.0)

    return jax.jit(jax.value_and_grad(fn, argnums=tuple(range(6 if with_state else 5))))

  expected = loss(qwen3.naive_jax_chunk_gated_delta_rule)(*args)
  actual = loss(qwen3.jax_chunk_gated_delta_rule, compute_dtype=jnp.float32)(*args)
  np.testing.assert_allclose(actual[0], expected[0], rtol=1e-4)
  for name, e, a in zip(("q", "k", "v", "g", "beta", "state"), expected[1], actual[1]):
    np.testing.assert_allclose(a, e, rtol=1e-3, atol=1e-4, err_msg=f"grad {name}")


def test_gated_delta_net_grads_match_nnx_conv(monkeypatch):
  cfg = pyconfig.initialize(
      [
          None,
          get_test_config_path(),
          "run_name=qwen3_gdn_conv_test",
          "dtype=float32",
          "weight_dtype=float32",
          "matmul_precision=highest",
          "decoder_block=qwen3_next",
          "base_emb_dim=128",
          "gdn_num_value_heads=4",
          "gdn_num_key_heads=2",
          "gdn_key_head_dim=32",
          "gdn_value_head_dim=32",
          f"gdn_conv_kernel_dim={TAPS}",
          "gdn_chunk_size=64",
          "use_qk_norm_in_gdn=True",
          "skip_jax_distributed_system=True",
      ]
  )
  x = jax.random.normal(jax.random.key(3), (2, 128, cfg.emb_dim), jnp.float32)
  layer = qwen3.Qwen3NextGatedDeltaNet(config=cfg, rngs=nnx.Rngs(4), inputs_shape=x.shape)
  graphdef, params, rest = nnx.split(layer, nnx.Param, ...)
  cotangent = jax.random.normal(jax.random.key(5), x.shape, jnp.float32)

  def output(params, x):
    out, _ = nnx.merge(graphdef, params, rest)(x)
    return out

  actual = jax.jit(lambda p, x: vjp_all(output, cotangent, p, x))(params, x)
  monkeypatch.setattr(qwen3, "causal_depthwise_conv1d", conv_reference)
  expected = jax.jit(lambda p, x: vjp_all(output, cotangent, p, x))(params, x)

  # Gradients such as A_log's sum fp32 rounding differences over every token of
  # the recurrence; an indexing error in the conv would be off by O(1).
  for name, e, a, rtol in zip(("output", "grad_params", "grad_input"), expected, actual, (1e-4, 1e-3, 1e-3), strict=True):
    for (path, el), al in zip(jax.tree_util.tree_leaves_with_path(e), jax.tree_util.tree_leaves(a), strict=True):
      np.testing.assert_allclose(al, el, rtol=rtol, atol=1e-5, err_msg=f"{name} {jax.tree_util.keystr(path)}")
