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

"""Tests for the Primus-Turbo grouped GEMM used by MoE sparse matmul (AMD GPUs)."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.kernels import turbo_grouped_gemm

try:
  import primus_turbo.jax  # pylint: disable=unused-import

  HAS_PRIMUS_TURBO = True
except ImportError:
  HAS_PRIMUS_TURBO = False


def _reference(lhs, rhs, group_sizes):
  out = jax.lax.ragged_dot(lhs.astype(jnp.float32), rhs.astype(jnp.float32), group_sizes)
  rows = jnp.arange(lhs.shape[0])[:, None]
  return jnp.where(rows < jnp.sum(group_sizes), out, 0.0)


def _rel_err(x, ref):
  x, ref = np.asarray(x, np.float32), np.asarray(ref, np.float32)
  return np.linalg.norm(x - ref) / max(np.linalg.norm(ref), 1e-30)


@pytest.mark.gpu_only
@unittest.skipUnless(HAS_PRIMUS_TURBO, "primus_turbo is not installed")
class TurboGroupedGemmTest(unittest.TestCase):
  """Forward and gradients of turbo_grouped_gemm.gmm against a float32 ragged_dot reference."""

  def _check(self, backend, group_sizes, m, k, n):
    group_sizes = jnp.asarray(group_sizes, jnp.int32)
    num_groups, valid = group_sizes.shape[0], int(jnp.sum(group_sizes))
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    lhs = jax.random.normal(k1, (m, k), jnp.bfloat16)
    rhs = (jax.random.normal(k2, (num_groups, k, n), jnp.float32) / np.sqrt(k)).astype(jnp.bfloat16)
    cotangent = jax.random.normal(k3, (m, n), jnp.float32)

    def loss(fn, x, w):
      return jnp.sum(fn(x, w).astype(jnp.float32) * cotangent)

    turbo = lambda x, w: turbo_grouped_gemm.gmm(x, w, group_sizes, backend=backend)
    ref = lambda x, w: _reference(x, w, group_sizes)
    out = jax.jit(turbo)(lhs, rhs)
    d_lhs, d_rhs = jax.jit(jax.grad(lambda x, w: loss(turbo, x, w), argnums=(0, 1)))(lhs, rhs)
    ref_out = ref(lhs, rhs)
    ref_d_lhs, ref_d_rhs = jax.grad(lambda x, w: loss(ref, x, w), argnums=(0, 1))(lhs, rhs)

    self.assertEqual(out.dtype, jnp.bfloat16)
    self.assertLess(_rel_err(out, ref_out), 1e-2)
    self.assertLess(_rel_err(d_lhs, ref_d_lhs), 1e-2)
    self.assertLess(_rel_err(d_rhs, ref_d_rhs), 1e-2)
    # Rows past the last group are padding and must come back as exact zeros.
    self.assertEqual(float(jnp.max(jnp.abs(out[valid:].astype(jnp.float32)), initial=0.0)), 0.0)
    self.assertEqual(float(jnp.max(jnp.abs(d_lhs[valid:].astype(jnp.float32)), initial=0.0)), 0.0)

  def test_ck_balanced(self):
    self._check("ck", [256] * 8, m=2048, k=256, n=512)

  def test_ck_empty_groups_and_padding(self):
    self._check("ck", [0, 300, 17, 0, 512, 1, 200, 0], m=1536, k=256, n=384)

  def test_hipblaslt_balanced(self):
    self._check("hipblaslt", [256] * 8, m=2048, k=256, n=512)

  def test_hipblaslt_empty_groups_and_padding(self):
    self._check("hipblaslt", [0, 300, 17, 0, 512, 1, 200, 0], m=1536, k=256, n=384)

  def test_hipblaslt_single_group(self):
    self._check("hipblaslt", [1000], m=1024, k=512, n=256)

  def test_flydsl_balanced(self):
    self._check("flydsl", [256] * 8, m=2048, k=256, n=512)

  def test_flydsl_empty_groups_and_padding(self):
    self._check("flydsl", [0, 300, 17, 0, 512, 1, 200, 0], m=1536, k=256, n=384)

  def test_flydsl_single_group(self):
    self._check("flydsl", [1000], m=1024, k=512, n=256)

  def test_flydsl_skewed_many_groups(self):
    sizes = [0] * 31 + [3000] + [1] * 32
    self._check("flydsl", sizes, m=3200, k=192, n=640)

  def test_rejects_float32(self):
    x = jnp.zeros((16, 8), jnp.float32)
    w = jnp.zeros((2, 8, 8), jnp.float32)
    with self.assertRaises(ValueError):
      turbo_grouped_gemm.gmm(x, w, jnp.array([8, 8], jnp.int32))


if __name__ == "__main__":
  unittest.main()
