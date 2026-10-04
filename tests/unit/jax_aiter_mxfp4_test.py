# Copyright 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

pytest.importorskip("flydsl.compiler")
pytest.importorskip("jax_aiter")

from maxtext.kernels import jax_aiter_mxfp4


def test_pad_and_unpad_expert_rows():
  values = jnp.arange(10 * 2, dtype=jnp.float32).reshape(10, 2)
  group_sizes = jnp.asarray([3, 0, 4], dtype=jnp.int32)

  padded, metadata = jax.jit(jax_aiter_mxfp4.pad_expert_rows)(
      values, group_sizes
  )
  assert padded.shape == (512, 2)
  np.testing.assert_array_equal(metadata.group_sizes, [128, 0, 128])
  np.testing.assert_array_equal(metadata.group_end_offsets, [128, 128, 256])
  np.testing.assert_array_equal(np.asarray(padded[:3]), np.asarray(values[:3]))
  np.testing.assert_array_equal(
      np.asarray(padded[128:132]), np.asarray(values[3:7])
  )

  restored = jax_aiter_mxfp4.unpad_expert_rows(padded, metadata)
  np.testing.assert_array_equal(np.asarray(restored[:7]), np.asarray(values[:7]))
  np.testing.assert_array_equal(np.asarray(restored[7:]), 0)


def test_expert_padding_matches_pad_expert_rows():
  values = jnp.arange(10 * 2, dtype=jnp.float32).reshape(10, 2)
  group_sizes = jnp.asarray([3, 0, 4], dtype=jnp.int32)
  _, expected = jax_aiter_mxfp4.pad_expert_rows(values, group_sizes, grouped_kernel="wholeloop")

  padding = jax_aiter_mxfp4.expert_padding(group_sizes, 10, grouped_kernel="wholeloop")
  for got, want in zip(padding, expected):
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_wholeloop_pads_expert_rows_to_256():
  values = jnp.arange(10 * 2, dtype=jnp.float32).reshape(10, 2)
  group_sizes = jnp.asarray([3, 0, 4], dtype=jnp.int32)

  padded, metadata = jax.jit(
      jax_aiter_mxfp4.pad_expert_rows, static_argnames="grouped_kernel"
  )(values, group_sizes, grouped_kernel="wholeloop")
  assert padded.shape == (1024, 2)
  np.testing.assert_array_equal(metadata.group_sizes, [256, 0, 256])
  np.testing.assert_array_equal(metadata.group_end_offsets, [256, 256, 512])
  np.testing.assert_array_equal(
      np.asarray(padded[256:260]), np.asarray(values[3:7])
  )
  restored = jax_aiter_mxfp4.unpad_expert_rows(padded, metadata)
  np.testing.assert_array_equal(np.asarray(restored[:7]), np.asarray(values[:7]))


def test_route_capacity_callback_can_be_disabled():
  values = jnp.arange(10 * 2, dtype=jnp.float32).reshape(10, 2)
  group_sizes = jnp.asarray([3, 0, 4], dtype=jnp.int32)

  guarded_hlo = jax.jit(jax_aiter_mxfp4.pad_expert_rows).lower(
      values, group_sizes
  ).as_text()
  unguarded_hlo = jax.jit(
      functools.partial(
          jax_aiter_mxfp4.pad_expert_rows, check_route_capacity=False
      )
  ).lower(values, group_sizes).as_text()

  assert "xla_ffi_python" in guarded_hlo
  assert "xla_ffi_python" not in unguarded_hlo


def test_unknown_grouped_kernel_is_rejected():
  with pytest.raises(ValueError, match="Unknown jax_aiter_mxfp4 grouped kernel"):
    jax_aiter_mxfp4.pad_expert_rows(
        jnp.zeros((4, 2)), jnp.asarray([4], jnp.int32), grouped_kernel="dense"
    )


def test_wholeloop_grouped_gmm_matches_ragged_forward_and_gradients():
  experts, rows, contraction, output = 3, 1024, 1024, 1024
  key_lhs, key_rhs, key_grad = jax.random.split(jax.random.key(37), 3)
  lhs = jax.random.normal(
      key_lhs, (rows, contraction), dtype=jnp.float32
  ).astype(jnp.bfloat16)
  rhs = jax.random.normal(
      key_rhs, (experts, contraction, output), dtype=jnp.float32
  ).astype(jnp.bfloat16)
  grad_output = jax.random.normal(
      key_grad, (rows, output), dtype=jnp.float32
  ).astype(jnp.bfloat16)
  # 256-aligned groups, an empty expert, and 256 unrouted slack rows.
  group_sizes = jnp.asarray([256, 0, 512], dtype=jnp.int32)

  @functools.partial(jax.jit, static_argnames="grouped_kernel")
  def run(x, w, sizes, cotangent, *, grouped_kernel):
    output_value, pullback = jax.vjp(
        lambda a, b: jax_aiter_mxfp4.grouped_gmm(
            a, b, sizes, grouped_kernel=grouped_kernel
        ),
        x,
        w,
    )
    return (output_value, *pullback(cotangent))

  operands = (lhs, rhs, group_sizes, grad_output)
  ragged = run(*operands, grouped_kernel="ragged")
  wholeloop = run(*operands, grouped_kernel="wholeloop")
  for name, expected, actual in zip(
      ("output", "grad_lhs", "grad_rhs"), ragged, wholeloop
  ):
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint16),
        np.asarray(expected).view(np.uint16),
        err_msg=name,
    )
  np.testing.assert_array_equal(np.asarray(wholeloop[0][768:]), 0)
  np.testing.assert_array_equal(np.asarray(wholeloop[2][1]), 0)


@pytest.mark.parametrize("grouped_kernel", ["ragged", "wholeloop"])
def test_grouped_gmm_gate_up_matches_separate_projections(grouped_kernel):
  experts, rows, contraction, output = 3, 1024, 1024, 1024
  keys = jax.random.split(jax.random.key(53), 4)
  lhs, w0, w1 = (
      jax.random.normal(key, shape, dtype=jnp.float32).astype(jnp.bfloat16)
      for key, shape in zip(
          keys[:3],
          ((rows, contraction), (experts, contraction, output), (experts, contraction, output)),
      )
  )
  grad_output = jax.random.normal(
      keys[3], (rows, 2 * output), dtype=jnp.float32
  ).astype(jnp.bfloat16)
  group_sizes = jnp.asarray([256, 0, 512], dtype=jnp.int32)

  @jax.jit
  def fused(x, a, b, cotangent):
    value, pullback = jax.vjp(
        lambda x, a, b: jax_aiter_mxfp4.grouped_gmm_gate_up(
            x, a, b, group_sizes, grouped_kernel=grouped_kernel
        ),
        x,
        a,
        b,
    )
    return (value, *pullback(cotangent))

  @jax.jit
  def separate(x, a, b, cotangent):
    def project(x, a, b):
      return jnp.concatenate(
          [
              jax_aiter_mxfp4.grouped_gmm(
                  x, w, group_sizes, grouped_kernel=grouped_kernel
              )
              for w in (a, b)
          ],
          axis=1,
      )

    value, pullback = jax.vjp(project, x, a, b)
    return (value, *pullback(cotangent))

  @jax.jit
  def concatenated_weight_grad_lhs(cotangent, a, b):
    return jax_aiter_mxfp4._grouped_gmm(
        cotangent,
        jnp.concatenate((a, b), axis=2),
        group_sizes,
        transpose_rhs=True,
        grouped_kernel=grouped_kernel,
    )

  actual = fused(lhs, w0, w1, grad_output)
  expected = separate(lhs, w0, w1, grad_output)
  for name, actual_value, expected_value in zip(
      ("output", "grad_w0", "grad_w1"),
      (actual[0], actual[2], actual[3]),
      (expected[0], expected[2], expected[3]),
  ):
    np.testing.assert_array_equal(
        np.asarray(actual_value).view(np.uint16),
        np.asarray(expected_value).view(np.uint16),
        err_msg=name,
    )

  # One GEMM over the combined contraction rounds once, unlike the sum of two
  # bf16 partial gradients, so it matches the concatenated-weight GEMM exactly.
  np.testing.assert_array_equal(
      np.asarray(actual[1]).view(np.uint16),
      np.asarray(concatenated_weight_grad_lhs(grad_output, w0, w1)).view(np.uint16),
  )
  error = jnp.linalg.norm(
      actual[1].astype(jnp.float32) - expected[1].astype(jnp.float32)
  ) / jnp.linalg.norm(expected[1].astype(jnp.float32))
  assert float(error) < 1e-2, float(error)
  np.testing.assert_array_equal(np.asarray(actual[1][768:]), 0)


@pytest.mark.parametrize("fused", [False, True], ids=["grouped_gmm", "gate_up"])
def test_wholeloop_zero_tail_off_keeps_group_rows_and_weight_grads(fused):
  experts, rows, contraction, output = 3, 1024, 1024, 1024
  live_rows = 768
  keys = jax.random.split(jax.random.key(61), 4)
  lhs, w0, w1 = (
      jax.random.normal(key, shape, dtype=jnp.float32).astype(jnp.bfloat16)
      for key, shape in zip(
          keys[:3],
          ((rows, contraction), (experts, contraction, output), (experts, contraction, output)),
      )
  )
  grad_output = jax.random.normal(
      keys[3], (rows, 2 * output if fused else output), dtype=jnp.float32
  ).astype(jnp.bfloat16)
  group_sizes = jnp.asarray([256, 0, 512], dtype=jnp.int32)
  # Rows past the groups hold NaN, as inputs and cotangents may when the
  # producing GEMM leaves its tail uninitialized.
  tail = jnp.arange(rows)[:, None] >= live_rows
  lhs = jnp.where(tail, jnp.nan, lhs)
  grad_output = jnp.where(tail, jnp.nan, grad_output)

  @functools.partial(jax.jit, static_argnames="zero_tail")
  def run(x, a, b, cotangent, *, zero_tail):
    if fused:
      project = lambda x, a, b: jax_aiter_mxfp4.grouped_gmm_gate_up(
          x, a, b, group_sizes, grouped_kernel="wholeloop", zero_tail=zero_tail
      )
      value, pullback = jax.vjp(project, x, a, b)
    else:
      project = lambda x, a: jax_aiter_mxfp4.grouped_gmm(
          x, a, group_sizes, grouped_kernel="wholeloop", zero_tail=zero_tail
      )
      value, pullback = jax.vjp(project, x, a)
    return (value, *pullback(cotangent))

  zeroed = run(lhs, w0, w1, grad_output, zero_tail=True)
  untouched = run(lhs, w0, w1, grad_output, zero_tail=False)
  for name, expected, actual in zip(("output", "grad_lhs"), zeroed[:2], untouched[:2]):
    np.testing.assert_array_equal(np.asarray(expected[live_rows:]), 0, err_msg=name)
    np.testing.assert_array_equal(
        np.asarray(actual[:live_rows]).view(np.uint16),
        np.asarray(expected[:live_rows]).view(np.uint16),
        err_msg=name,
    )
  for index, (expected, actual) in enumerate(zip(zeroed[2:], untouched[2:])):
    assert bool(jnp.isfinite(expected).all()), f"weight grad {index}"
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint16),
        np.asarray(expected).view(np.uint16),
        err_msg=f"weight grad {index}",
    )


def test_dense_linear_matches_e1_grouped_gmm_forward_and_gradients():
  rows = contraction = output = 256
  key_lhs, key_rhs, key_grad = jax.random.split(jax.random.key(41), 3)
  lhs = jax.random.normal(
      key_lhs, (rows, contraction), dtype=jnp.float32
  ).astype(jnp.bfloat16)
  rhs = jax.random.normal(
      key_rhs, (contraction, output), dtype=jnp.float32
  ).astype(jnp.bfloat16)
  grad_output = jax.random.normal(
      key_grad, (rows, output), dtype=jnp.float32
  ).astype(jnp.bfloat16)
  group_sizes = jnp.asarray([rows], dtype=jnp.int32)
  mesh = Mesh(np.asarray(jax.devices()[:1]), ("expert",))

  def run(fn):
    value, pullback = jax.vjp(fn, lhs, rhs)
    return value, *pullback(grad_output)

  expected = run(
      lambda x, w: jax_aiter_mxfp4.grouped_gmm(
          x,
          w[None, :, :],
          group_sizes,
          grouped_kernel="wholeloop",
      )
  )
  actual = run(
      lambda x, w: jax_aiter_mxfp4.dense_linear(x, w, mesh)
  )
  for name, expected_value, actual_value in zip(
      ("output", "grad_lhs", "grad_rhs"),
      expected,
      actual,
      strict=True,
  ):
    np.testing.assert_array_equal(
        np.asarray(actual_value).view(np.uint16),
        np.asarray(expected_value).view(np.uint16),
        err_msg=name,
    )


@pytest.mark.skipif(jax.device_count() != 8, reason="requires the target EP=8 mesh")
def test_dense_linear_ep8_sharding_and_gradients():
  rows, contraction, output = 2048, 256, 256
  key_lhs, key_rhs, key_grad = jax.random.split(jax.random.key(43), 3)
  mesh = Mesh(np.asarray(jax.devices()), ("expert",))
  batch_sharding = NamedSharding(mesh, PartitionSpec("expert", None))
  replicated = NamedSharding(mesh, PartitionSpec(None, None))
  lhs = jax.device_put(
      jax.random.normal(
          key_lhs, (rows, contraction), dtype=jnp.bfloat16
      ),
      batch_sharding,
  )
  rhs = jax.device_put(
      jax.random.normal(
          key_rhs, (contraction, output), dtype=jnp.bfloat16
      ),
      replicated,
  )
  grad_output = jax.device_put(
      jax.random.normal(
          key_grad, (rows, output), dtype=jnp.bfloat16
      ),
      batch_sharding,
  )

  def run(linear):
    value, pullback = jax.vjp(linear, lhs, rhs)
    return value, *pullback(grad_output)

  actual = jax.jit(
      lambda: run(
          lambda x, w: jax_aiter_mxfp4.dense_linear(x, w, mesh)
      )
  )()
  expected = jax.jit(
      lambda: run(lambda x, w: x @ w)
  )()

  assert actual[0].sharding.spec[0] == "expert"
  assert actual[1].sharding.spec[0] == "expert"
  assert actual[2].sharding.is_fully_replicated
  for name, actual_value, expected_value in zip(
      ("output", "grad_lhs", "grad_rhs"),
      actual,
      expected,
      strict=True,
  ):
    error = jnp.linalg.norm(
        actual_value.astype(jnp.float32)
        - expected_value.astype(jnp.float32)
    ) / jnp.linalg.norm(expected_value.astype(jnp.float32))
    assert float(error) < 0.25, (name, float(error))


def test_pad_and_unpad_autodiff_uses_gathers():
  key_values, key_weights, key_cot = jax.random.split(jax.random.key(7), 3)
  values = jax.random.normal(key_values, (10, 2), dtype=jnp.float32)
  weights = jax.random.normal(key_weights, (512, 2), dtype=jnp.float32)
  cotangent = jax.random.normal(key_cot, (10, 2), dtype=jnp.float32)
  group_sizes = jnp.asarray([3, 0, 4], dtype=jnp.int32)

  def scale_padded(v):
    padded, metadata = jax_aiter_mxfp4.pad_expert_rows(v, group_sizes)
    return jax_aiter_mxfp4.unpad_expert_rows(padded * weights, metadata)

  output, pullback = jax.vjp(scale_padded, values)
  (grad,) = pullback(cotangent)
  routed_rows = np.r_[0:3, 128:132]
  np.testing.assert_array_equal(
      np.asarray(output[:7]), np.asarray(values[:7] * weights[routed_rows])
  )
  np.testing.assert_array_equal(np.asarray(output[7:]), 0)
  np.testing.assert_array_equal(
      np.asarray(grad[:7]), np.asarray(cotangent[:7] * weights[routed_rows])
  )
  np.testing.assert_array_equal(np.asarray(grad[7:]), 0)

  hlo = (
      jax.jit(lambda v, c: jax.vjp(scale_padded, v)[1](c))
      .lower(values, cotangent)
      .as_text()
  )
  assert "stablehlo.gather" in hlo
  assert "stablehlo.scatter" not in hlo


def test_grouped_gmm_custom_vjp_is_finite():
  experts, rows, contraction, output = 2, 256, 1024, 1024
  key_lhs, key_rhs, key_grad = jax.random.split(jax.random.key(31), 3)
  lhs = jax.random.normal(
      key_lhs, (rows, contraction), dtype=jnp.float32
  ).astype(jnp.bfloat16)
  rhs = jax.random.normal(
      key_rhs, (experts, contraction, output), dtype=jnp.float32
  ).astype(jnp.bfloat16)
  grad_output = jax.random.normal(
      key_grad, (rows, output), dtype=jnp.float32
  ).astype(jnp.bfloat16)
  group_sizes = jnp.asarray([128, 128], dtype=jnp.int32)

  output_value, pullback = jax.vjp(
      lambda x, w: jax_aiter_mxfp4.grouped_gmm(x, w, group_sizes),
      lhs,
      rhs,
  )
  grad_lhs, grad_rhs = pullback(grad_output)

  assert output_value.shape == (rows, output)
  assert grad_lhs.shape == lhs.shape
  assert grad_rhs.shape == rhs.shape
  assert bool(jnp.isfinite(output_value).all())
  assert bool(jnp.isfinite(grad_lhs).all())
  assert bool(jnp.isfinite(grad_rhs).all())
