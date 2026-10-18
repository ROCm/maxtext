# Copyright 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools

import flax.linen as nn
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

pytest.importorskip("flydsl.compiler")
pytest.importorskip("jax_aiter")

from maxtext.configs import pyconfig
from maxtext.kernels import jax_aiter_mxfp4
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


def test_pad_and_unpad_expert_rows():
  values = jnp.arange(10 * 2, dtype=jnp.float32).reshape(10, 2)
  group_sizes = jnp.asarray([3, 0, 4], dtype=jnp.int32)

  padded, metadata = jax.jit(jax_aiter_mxfp4.pad_expert_rows)(
      values, group_sizes
  )
  assert padded.shape == (1024, 2)
  np.testing.assert_array_equal(metadata.group_sizes, [256, 0, 256])
  np.testing.assert_array_equal(metadata.group_end_offsets, [256, 256, 512])
  np.testing.assert_array_equal(np.asarray(padded[:3]), np.asarray(values[:3]))
  np.testing.assert_array_equal(
      np.asarray(padded[256:260]), np.asarray(values[3:7])
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


def test_max_compact_capacity_is_the_largest_int32_indexable_padding():
  cols, num_experts = 7168, 32

  def padded_rows(capacity):
    padding = jax.eval_shape(
        functools.partial(jax_aiter_mxfp4.expert_padding, compact_capacity=capacity),
        jax.ShapeDtypeStruct((num_experts,), jnp.int32),
    )
    return padding.padded_to_compact.shape[0]

  capacity = jax_aiter_mxfp4.max_compact_capacity(cols, num_experts)
  assert padded_rows(capacity) * cols <= 2**31 - 1
  assert padded_rows(capacity + 1) * cols > 2**31 - 1


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


@pytest.mark.parametrize("grouped_kernel", ["dense", "ragged"])
def test_unknown_grouped_kernel_is_rejected(grouped_kernel):
  with pytest.raises(ValueError, match="Unknown jax_aiter_mxfp4 grouped kernel"):
    jax_aiter_mxfp4.pad_expert_rows(
        jnp.zeros((4, 2)), jnp.asarray([4], jnp.int32), grouped_kernel=grouped_kernel
    )


def test_grouped_gmm_matches_full_precision_forward_and_gradients():
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

  @jax.jit
  def run(x, w, cotangent):
    output_value, pullback = jax.vjp(
        lambda a, b: jax_aiter_mxfp4.grouped_gmm(a, b, group_sizes), x, w
    )
    return (output_value, *pullback(cotangent))

  output_value, grad_lhs, grad_rhs = run(lhs, rhs, grad_output)
  x, w, g = (value.astype(jnp.float32) for value in (lhs, rhs, grad_output))
  segments = ((0, slice(0, 256)), (2, slice(256, 768)))
  checks = [(f"output {e}", output_value[s], x[s] @ w[e]) for e, s in segments]
  checks += [(f"grad_lhs {e}", grad_lhs[s], g[s] @ w[e].T) for e, s in segments]
  checks += [(f"grad_rhs {e}", grad_rhs[e], x[s].T @ g[s]) for e, s in segments]
  for name, actual, expected in checks:
    error = jnp.linalg.norm(actual.astype(jnp.float32) - expected) / jnp.linalg.norm(expected)
    assert float(error) < 0.25, (name, float(error))
  np.testing.assert_array_equal(np.asarray(output_value[768:]), 0)
  np.testing.assert_array_equal(np.asarray(grad_lhs[768:]), 0)
  np.testing.assert_array_equal(np.asarray(grad_rhs[1]), 0)


@pytest.mark.parametrize(("size", "expected"), [(2880, 3072), (1024, 1024)])
def test_padded_feature_dim(size, expected):
  assert jax_aiter_mxfp4.padded_feature_dim(size, "wholeloop") == expected


def test_unaligned_feature_dims_are_padded_exactly():
  """GPT-OSS K = N = 2880 pads to 3072; zero-padding further, to 3328, must not change a bit."""
  experts, rows, contraction, output = 3, 1024, 2880, 2880
  wider = 3328
  keys = jax.random.split(jax.random.key(41), 5)
  lhs = jax.random.normal(keys[0], (rows, contraction), jnp.float32).astype(jnp.bfloat16)
  w0, w1 = (
      jax.random.normal(key, (experts, contraction, output), jnp.float32).astype(jnp.bfloat16)
      for key in keys[1:3]
  )
  grad_single = jax.random.normal(keys[3], (rows, output), jnp.float32).astype(jnp.bfloat16)
  grad_fused = jax.random.normal(keys[4], (rows, 2 * output), jnp.float32).astype(jnp.bfloat16)
  group_sizes = jnp.asarray([256, 0, 512], dtype=jnp.int32)

  def widen_lhs(x):
    return jnp.pad(x, ((0, 0), (0, wider - contraction)))

  def widen_weight(w):
    return jnp.pad(w, ((0, 0), (0, wider - contraction), (0, wider - output)))

  def single(x, a, *, widened):
    if not widened:
      return jax_aiter_mxfp4.grouped_gmm(x, a, group_sizes)
    return jax_aiter_mxfp4.grouped_gmm(widen_lhs(x), widen_weight(a), group_sizes)[:, :output]

  def gate_up(x, a, b, *, widened):
    if not widened:
      return jax_aiter_mxfp4.grouped_gmm_gate_up(x, a, b, group_sizes)
    value = jax_aiter_mxfp4.grouped_gmm_gate_up(widen_lhs(x), widen_weight(a), widen_weight(b), group_sizes)
    return jnp.concatenate((value[:, :output], value[:, wider : wider + output]), axis=1)

  @functools.partial(jax.jit, static_argnames="widened")
  def run(x, a, b, single_cotangent, fused_cotangent, *, widened):
    out, pullback = jax.vjp(functools.partial(single, widened=widened), x, a)
    out_fused, pullback_fused = jax.vjp(functools.partial(gate_up, widened=widened), x, a, b)
    return (out, *pullback(single_cotangent), out_fused, *pullback_fused(fused_cotangent))

  operands = (lhs, w0, w1, grad_single, grad_fused)
  padded = run(*operands, widened=False)
  widened = run(*operands, widened=True)
  names = ("output", "grad_lhs", "grad_w", "fused_output", "fused_grad_lhs", "fused_grad_w0", "fused_grad_w1")
  shapes = ((rows, output), lhs.shape, w0.shape, (rows, 2 * output), lhs.shape, w0.shape, w1.shape)
  for name, shape, expected, actual in zip(names, shapes, widened, padded):
    assert actual.shape == expected.shape == shape, name
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint16), np.asarray(expected).view(np.uint16), err_msg=name
    )
  # Rows past the last group stay zero, and the fused halves are the two projections.
  np.testing.assert_array_equal(np.asarray(padded[0][768:]), 0)
  np.testing.assert_array_equal(
      np.asarray(padded[3][:, :output]).view(np.uint16), np.asarray(padded[0]).view(np.uint16)
  )
  segments = (slice(0, 256), slice(256, 256), slice(256, 768))
  reference = jnp.concatenate(
      [lhs[rows_].astype(jnp.float32) @ w0[e].astype(jnp.float32) for e, rows_ in enumerate(segments)]
  )
  error = jnp.linalg.norm(padded[0][:768].astype(jnp.float32) - reference) / jnp.linalg.norm(reference)
  assert float(error) < 0.25, float(error)


def test_grouped_gmm_gate_up_matches_separate_projections():
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
        lambda x, a, b: jax_aiter_mxfp4.grouped_gmm_gate_up(x, a, b, group_sizes),
        x,
        a,
        b,
    )
    return (value, *pullback(cotangent))

  @jax.jit
  def separate(x, a, b, cotangent):
    def project(x, a, b):
      return jnp.concatenate(
          [jax_aiter_mxfp4.grouped_gmm(x, w, group_sizes) for w in (a, b)],
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


@pytest.mark.parametrize("zero_tail", [True, False])
@pytest.mark.parametrize("projection", ["grouped_gmm", "gate_up", "gathered", "dense"])
def test_dual_grad_cast_matches_separate_casts(projection, zero_tail):
  pytest.importorskip("jax_aiter.ops.moe_mxfp4_ptquant")
  if projection == "dense" and not zero_tail:
    pytest.skip("dense_linear always zeroes the tail")
  experts, rows, contraction, output = 3, 1024, 512, 768
  live_rows = rows if projection == "dense" else 768
  keys = jax.random.split(jax.random.key(67), 4)
  lhs = jax.random.normal(keys[0], (rows, contraction), jnp.float32).astype(jnp.bfloat16)
  w0, w1 = (
      jax.random.normal(key, (experts, contraction, output), jnp.float32).astype(jnp.bfloat16)
      for key in keys[1:3]
  )
  group_sizes = jnp.asarray([256, 0, 512], dtype=jnp.int32)
  mesh = Mesh(np.asarray(jax.devices()[:1]), ("fsdp",))
  options = {"grouped_kernel": "wholeloop", "zero_tail": zero_tail}
  tail = jnp.arange(rows)[:, None] >= live_rows
  lhs = jnp.where(tail, jnp.nan, lhs)

  def project(dual):
    if projection == "grouped_gmm":
      return lambda x, a, b: jax_aiter_mxfp4.grouped_gmm(x, a, group_sizes, **options, dual_grad_cast=dual)
    if projection == "gate_up":
      return lambda x, a, b: jax_aiter_mxfp4.grouped_gmm_gate_up(x, a, b, group_sizes, **options, dual_grad_cast=dual)
    if projection == "dense":
      return lambda x, a, b: jax_aiter_mxfp4.dense_linear(x, a[0], mesh, dual_grad_cast=dual)

    def gathered(x, a, b):
      weight = jax_aiter_mxfp4.gather_expert_weight(a, "fsdp", 1)
      return jax_aiter_mxfp4.grouped_gmm(x, weight, group_sizes, **options, dual_grad_cast=dual)

    weight_spec = PartitionSpec(None, "fsdp", None)
    return jax.shard_map(
        gathered,
        mesh=mesh,
        in_specs=(PartitionSpec(), weight_spec, weight_spec),
        out_specs=PartitionSpec(),
        check_vma=False,
    )

  def run(x, a, b, *, dual):
    value, pullback = jax.vjp(project(dual), x, a, b)
    cotangent = jax.random.normal(keys[3], value.shape, jnp.float32).astype(jnp.bfloat16)
    return value, *pullback(jnp.where(tail, jnp.nan, cotangent))

  separate_run, dual_run = (
      jax.jit(functools.partial(run, dual=dual)) for dual in (False, True)
  )
  casts = [
      fn.lower(lhs, w0, w1).as_text().count("FlydslDispatchJA") for fn in (separate_run, dual_run)
  ]
  assert casts[1] == casts[0] - 1, casts
  separate, dual = separate_run(lhs, w0, w1), dual_run(lhs, w0, w1)
  for name, expected, actual in zip(("output", "grad_lhs", "grad_w0", "grad_w1"), separate, dual):
    if name in ("output", "grad_lhs"):
      expected, actual = expected[:live_rows], actual[:live_rows]
    else:
      assert bool(jnp.isfinite(actual).all()), name
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint16), np.asarray(expected).view(np.uint16), err_msg=name
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


def test_expert_row_ids_follow_grouped_rows():
  group_sizes = jnp.asarray([2, 0, 3, 1], dtype=jnp.int32)
  ids = jax.jit(jax_aiter_mxfp4.expert_row_ids, static_argnums=1)(group_sizes, 9)
  np.testing.assert_array_equal(np.asarray(ids), [0, 0, 2, 2, 2, 3, 4, 4, 4])


def test_expert_bias_grad_ignores_unzeroed_tail_rows():
  group_sizes = jnp.asarray([2, 0, 3, 1], dtype=jnp.int32)
  # Rows past the last group carry an uninitialized, here NaN, cotangent.
  cotangent = jnp.full((9, 2), jnp.nan, jnp.float32).at[:6].set(1.0)

  def loss(bias):
    ids = jax_aiter_mxfp4.expert_row_ids(group_sizes, cotangent.shape[0])
    return jnp.sum(bias.at[ids].get(mode="fill", fill_value=0) * cotangent)

  grad = jax.jit(jax.grad(loss))(jnp.ones((4, 2), jnp.float32))
  np.testing.assert_array_equal(np.asarray(grad), [[2.0, 2.0], [0.0, 0.0], [3.0, 3.0], [1.0, 1.0]])


def _gpt_oss_moe(**overrides):
  settings = dict(
      run_name="jax_aiter_mxfp4_mlp_bias_test",
      enable_checkpointing=False,
      model_name="gpt-oss-20b",
      override_model_config=True,
      base_emb_dim=256,
      base_mlp_dim=256,
      base_moe_mlp_dim=256,
      num_experts=16,
      num_experts_per_tok=4,
      per_device_batch_size=1,
      max_target_length=128,
      dtype="bfloat16",
      weight_dtype="float32",
      megablox=False,
      sparse_matmul=True,
      ici_expert_parallelism=8,
      ragged_buffer_factor=2.0,
  )
  settings.update(overrides)
  cfg = pyconfig.initialize([None, get_test_config_path()], **settings)
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = moe.get_routed_moe(
      name="MoeBlock",
      config=cfg,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      mesh=mesh,
      kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes=("embed", "mlp"),
      intermediate_dim=cfg.moe_mlp_dim,
      weight_dtype=cfg.weight_dtype,
      dtype=cfg.dtype,
  )
  return model, mesh, cfg.logical_axis_rules


@pytest.mark.skipif(jax.device_count() != 8, reason="requires the target EP=8 mesh")
@pytest.mark.parametrize("zero_kernels", [True, False])
@pytest.mark.parametrize("dim", [256, 2880])
@pytest.mark.parametrize("skip_unread_zeroing", [False, True])
def test_dedup_mlp_bias_matches_fp32_reference(zero_kernels, dim, skip_unread_zeroing):
  """GPT-OSS expert biases and clamped activation through MXFP4 dedup dispatch.

  2880 is the GPT-OSS width, which the grouped kernels take only zero-padded.
  """
  dims = dict(base_emb_dim=dim, base_mlp_dim=dim, base_moe_mlp_dim=dim)
  reference = _gpt_oss_moe(dtype="float32", **dims)
  mxfp4 = _gpt_oss_moe(
      moe_local_expert_backend="jax_aiter_mxfp4",
      jax_aiter_mxfp4_dedup_dispatch=True,
      jax_aiter_mxfp4_fresh_a2a_buffers=True,
      jax_aiter_mxfp4_check_route_capacity=False,
      jax_aiter_mxfp4_dedup_skip_unread_zeroing=skip_unread_zeroing,
      shardy=False,
      **dims,
  )
  keys = jax.random.split(jax.random.key(11), 6)
  hidden = jax.random.normal(keys[0], (8, 128, dim), jnp.bfloat16)
  model, mesh, rules = reference
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(rules):
    params = nn.meta.unbox(model.init({"params": keys[1], "dropout": keys[1]}, hidden))["params"]
  for name, key in zip(("wi_0_bias", "wi_1_bias", "wo_bias"), keys[2:5]):
    params[name] = jax.random.normal(key, params[name].shape, params[name].dtype)
  if zero_kernels:
    # MXFP4 GEMMs of zero weights are exact, so only the biases reach the output.
    for name in ("wi_0", "wi_1", "wo"):
      params[name] = jnp.zeros_like(params[name])
  cotangent = jax.random.normal(keys[5], hidden.shape, jnp.float32)

  def run(moe_block, p):
    model, mesh, rules = moe_block
    x = hidden.astype(jnp.float32) if moe_block is reference else hidden

    def loss(p):
      out = model.apply({"params": p}, x)[0]
      return jnp.sum(out.astype(jnp.float32) * cotangent), out

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(rules):
      (_, out), grads = jax.jit(jax.value_and_grad(loss, has_aux=True))(p)
    return out.astype(jnp.float32), grads

  def rel(a, b):
    return float(jnp.linalg.norm(a - b) / jnp.maximum(jnp.linalg.norm(b), 1e-30))

  expected, expected_grads = run(reference, params)
  actual, actual_grads = run(mxfp4, params)
  assert bool(jnp.isfinite(actual).all())
  # Against FP32: BF16 activations alone give about 3%; MXFP4 GEMMs add up to about 16%
  # on the bias gradients that flow back through quantized wo and wi.
  tolerance = 5e-2 if zero_kernels else 0.25
  assert rel(actual, expected) < tolerance, rel(actual, expected)
  bias_names = ("wo_bias",) if zero_kernels else ("wi_0_bias", "wi_1_bias", "wo_bias")
  for name in bias_names:
    error = rel(actual_grads[name].astype(jnp.float32), expected_grads[name].astype(jnp.float32))
    assert error < tolerance, (name, error)
  if not zero_kernels:
    unbiased = dict(params, **{n: jnp.zeros_like(params[n]) for n in ("wi_0_bias", "wi_1_bias", "wo_bias")})
    bias_effect = rel(run(reference, unbiased)[0], expected)
    assert bias_effect > 4 * rel(actual, expected), (bias_effect, rel(actual, expected))


def test_pad_and_unpad_autodiff_uses_gathers():
  key_values, key_weights, key_cot = jax.random.split(jax.random.key(7), 3)
  values = jax.random.normal(key_values, (10, 2), dtype=jnp.float32)
  weights = jax.random.normal(key_weights, (1024, 2), dtype=jnp.float32)
  cotangent = jax.random.normal(key_cot, (10, 2), dtype=jnp.float32)
  group_sizes = jnp.asarray([3, 0, 4], dtype=jnp.int32)

  def scale_padded(v):
    padded, metadata = jax_aiter_mxfp4.pad_expert_rows(v, group_sizes)
    return jax_aiter_mxfp4.unpad_expert_rows(padded * weights, metadata)

  output, pullback = jax.vjp(scale_padded, values)
  (grad,) = pullback(cotangent)
  routed_rows = np.r_[0:3, 256:260]
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
  experts, rows, contraction, output = 2, 512, 1024, 1024
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
  group_sizes = jnp.asarray([256, 256], dtype=jnp.int32)

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


@pytest.mark.parametrize("scatter_dimension", [1, 2])
def test_fp8_psum_scatter_matches_psum_scatter(scatter_dimension):
  devices = jax.devices()
  if len(devices) < 8:
    pytest.skip("requires eight devices; on CPU set --xla_force_host_platform_device_count=8")
  shards = 8
  mesh = Mesh(np.asarray(devices[:shards]), ("fsdp",))
  shape = (3, 512, 256)
  key_value, key_scale = jax.random.split(jax.random.key(7))
  scale = jnp.exp(2.0 * jax.random.normal(key_scale, (shards, 3, 1, 1)))
  x = (jax.random.normal(key_value, (shards,) + shape) * scale).astype(jnp.bfloat16)
  x = x.at[:, 1].set(0)

  def run(reduce):
    f = jax.shard_map(
        lambda g: reduce(g[0])[None], mesh=mesh,
        in_specs=PartitionSpec("fsdp"), out_specs=PartitionSpec("fsdp"),
    )
    return np.asarray(jax.jit(f)(x), np.float32)

  exact = run(lambda g: jax.lax.psum_scatter(
      g.astype(jnp.float32), "fsdp", scatter_dimension=scatter_dimension, tiled=True))
  fp8 = run(lambda g: jax_aiter_mxfp4.fp8_psum_scatter(g, "fsdp", scatter_dimension))

  assert fp8.shape == exact.shape
  np.testing.assert_array_equal(fp8[:, 1], 0)
  rel = np.linalg.norm(fp8 - exact) / np.linalg.norm(exact)
  assert rel < 4e-2, rel
