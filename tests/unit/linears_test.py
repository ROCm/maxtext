# Copyright 2026 Google LLC
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

"""Tests for linears.py."""

import sys
import unittest
from unittest import mock
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.layers import linears
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


class UtilsTest(unittest.TestCase):
  """Tests for utility functions in linears.py."""

  def test_normalize_axes(self):
    self.assertEqual(linears.normalize_axes((1, 2), 4), (1, 2))
    self.assertEqual(linears.normalize_axes((-1, -2), 4), (3, 2))
    self.assertEqual(linears.normalize_axes((0, -1), 3), (0, 2))

  def test_canonicalize_tuple(self):
    self.assertEqual(linears.canonicalize_tuple(1), (1,))
    self.assertEqual(linears.canonicalize_tuple((1, 2)), (1, 2))
    self.assertEqual(linears.canonicalize_tuple([1, 2]), (1, 2))

  # pylint: disable=protected-access
  def test_convert_to_activation_function(self):
    lin_fn = linears._convert_to_activation_function("linear")
    x = jnp.array([1.0, 2.0])
    np.testing.assert_array_equal(lin_fn(x), x)

    relu_fn = linears._convert_to_activation_function("relu")
    x = jnp.array([-1.0, 2.0])
    np.testing.assert_array_equal(relu_fn(x), jnp.array([0.0, 2.0]))

    # Test with callable
    def dummy_fn(x):
      return x + 1

    self.assertEqual(linears._convert_to_activation_function(dummy_fn), dummy_fn)

    with self.assertRaises(ValueError):
      linears._convert_to_activation_function(123)


class DenseGeneralTest(unittest.TestCase):
  """Tests for DenseGeneral."""

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(params=0)

  def test_basic_call(self):
    batch_size = 2
    in_features = 4
    out_features = 8

    layer = linears.DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=out_features,
        rngs=self.rngs,
    )

    inputs = jnp.ones((batch_size, in_features))
    outputs = layer(inputs)

    self.assertEqual(outputs.shape, (batch_size, out_features))

  def test_bias(self):
    batch_size = 2
    in_features = 4
    out_features = 8

    layer = linears.DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=out_features,
        use_bias=True,
        rngs=self.rngs,
    )

    inputs = jnp.ones((batch_size, in_features))
    outputs = layer(inputs)

    self.assertEqual(outputs.shape, (batch_size, out_features))
    self.assertIsNotNone(layer.bias)

  def test_retain_gathered_weight_marks_replicated_kernel(self):
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("expert",))
    gathered_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(None, None),
    )
    layer = linears.DenseGeneral(
        in_features_shape=4,
        out_features_shape=8,
        kernel_axes=("embed", "mlp"),
        mesh=mesh,
        retain_gathered_weight=True,
        rngs=self.rngs,
    )

    with mock.patch(
        "maxtext.layers.linears.get_physical_spec_without_axes",
        return_value=gathered_sharding,
    ) as get_gathered_sharding, mock.patch(
        "maxtext.layers.linears.maybe_shard_with_name",
        side_effect=lambda value, unused_sharding, **unused_kwargs: value,
    ) as gather, mock.patch(
        "maxtext.layers.linears.checkpoint_name",
        side_effect=lambda value, unused_name: value,
    ) as checkpoint:
      outputs = layer(jnp.ones((2, 4)))

    self.assertEqual(outputs.shape, (2, 8))
    get_gathered_sharding.assert_called_once_with(
        jax.sharding.PartitionSpec("embed", "mlp"),
        mesh,
        linears._DENSE_BATCH_MESH_AXES,
    )
    gather.assert_called_once_with(
        mock.ANY,
        gathered_sharding,
        shard_mode=linears.ShardMode.AUTO,
        debug_sharding=False,
    )
    checkpoint.assert_called_once_with(mock.ANY, "gathered_weight")

  def _run_dense_test(self, axis, in_feat_shape, expected_shape):
    batch_size = 2
    seq_len = 3
    in_features = 4
    out_features = 8

    layer = linears.DenseGeneral(
        in_features_shape=in_feat_shape,
        out_features_shape=out_features,
        axis=axis,
        rngs=self.rngs,
    )

    inputs = jnp.ones((batch_size, seq_len, in_features))
    outputs = layer(inputs)

    self.assertEqual(outputs.shape, expected_shape)

  def test_axis_neg_1(self):
    self._run_dense_test(-1, 4, (2, 3, 8))

  def test_axis_1(self):
    self._run_dense_test(1, 3, (2, 4, 8))

  def test_axis_0(self):
    self._run_dense_test(0, 2, (3, 4, 8))

  def test_jax_aiter_mxfp4_flattens_trailing_contraction_axes(self):
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:1]),
        ("expert",),
    )
    layer = linears.DenseGeneral(
        in_features_shape=(2, 2),
        out_features_shape=(2, 3),
        axis=(-2, -1),
        dtype=jnp.bfloat16,
        weight_dtype=jnp.bfloat16,
        mesh=mesh,
        use_jax_aiter_mxfp4=True,
        rngs=self.rngs,
    )
    inputs = jnp.ones((2, 3, 2, 2), dtype=jnp.bfloat16)

    with mock.patch(
        "maxtext.kernels.jax_aiter_mxfp4.dense_linear",
        side_effect=lambda lhs, rhs, unused_mesh: lhs @ rhs,
    ) as dense_linear, mock.patch(
        "maxtext.layers.linears.checkpoint_name",
        side_effect=lambda value, unused_name: value,
    ) as checkpoint:
      outputs = layer(inputs)

    self.assertEqual(outputs.shape, (2, 3, 2, 3))
    matrix_inputs, matrix_kernel, called_mesh = dense_linear.call_args.args
    self.assertEqual(matrix_inputs.shape, (6, 4))
    self.assertEqual(matrix_kernel.shape, (4, 6))
    self.assertIs(called_mesh, mesh)
    checkpoint.assert_called_once_with(mock.ANY, "mxfp4_linear")


class MlpBlockTest(unittest.TestCase):
  """Tests for MlpBlock."""

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(params=0, dropout=1)

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 128,
        "fused_mlp": False,
    }
    argv = [sys.argv[0], get_test_config_path()]
    self.cfg = pyconfig.initialize(argv, **config_arguments)

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = jax.sharding.Mesh(devices_array, self.cfg.mesh_axes)

  def test_basic_call(self):
    batch_size = 2
    seq_len = 3
    in_features = 4
    intermediate_dim = 8

    layer = linears.MlpBlock(
        config=self.cfg,
        mesh=self.mesh,
        in_features=in_features,
        intermediate_dim=intermediate_dim,
        rngs=self.rngs,
    )

    inputs = jnp.ones((batch_size, seq_len, in_features))
    outputs = layer(inputs)

    self.assertEqual(outputs.shape, (batch_size, seq_len, in_features))
    self.assertEqual(layer.wi.kernel[...].shape, (in_features, intermediate_dim))

  def test_fused_mlp(self):
    batch_size = 2
    seq_len = 3
    in_features = 4
    intermediate_dim = 8

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 128,
        "fused_mlp": True,
    }
    argv = [sys.argv[0], get_test_config_path()]
    cfg_fused = pyconfig.initialize(argv, **config_arguments)

    layer = linears.MlpBlock(
        config=cfg_fused,
        mesh=self.mesh,
        in_features=in_features,
        intermediate_dim=intermediate_dim,
        rngs=self.rngs,
    )

    inputs = jnp.ones((batch_size, seq_len, in_features))
    outputs = layer(inputs)

    self.assertEqual(outputs.shape, (batch_size, seq_len, in_features))
    self.assertEqual(layer.wi.kernel[...].shape, (in_features, 1, intermediate_dim))

  @pytest.mark.tpu_only
  def test_dense_fsdp_two_stage_all_gather_tpu_only(self):
    """`dense_fsdp_use_two_stage_all_gather` is a pure resharding and must not change MLP numerics.

    The flag replaces the single combined 2-axis (fsdp x fsdp_transpose) kernel all-gather with two
    single-axis gathers separated by an `optimization_barrier`. That changes only the weight's layout,
    not its values, so on a 2D-FSDP mesh the MLP output must match the default single-all-gather path.
    Covers both the non-fused (2D `wi`/`wo` kernels) and fused (3D `wi` kernel) MLP paths.
    """
    # Imperative skip: calling jax.device_count() in a decorator would force JAX init during collection.
    if jax.device_count() != 4:
      self.skipTest("Dense FSDP two-stage all-gather test requires exactly 4 devices")

    in_features = 128
    intermediate_dim = 256
    batch_size = 4
    seq_len = 8

    def build_and_run(use_two_stage, fused_mlp):
      config_arguments = {
          "per_device_batch_size": 1.0,
          "run_name": "test",
          "enable_checkpointing": False,
          "max_target_length": seq_len,
          "dtype": "bfloat16",
          "fused_mlp": fused_mlp,
          "ici_fsdp_parallelism": 2,
          "ici_fsdp_transpose_parallelism": 2,
          "dense_fsdp_use_two_stage_all_gather": use_two_stage,
      }
      argv = [sys.argv[0], get_test_config_path()]
      cfg = pyconfig.initialize(argv, **config_arguments)

      devices_array = maxtext_utils.create_device_mesh(cfg)
      mesh = jax.sharding.Mesh(devices_array, cfg.mesh_axes)
      # Fresh, fixed-seed Rngs yields deterministic, identical weights across both builds.
      with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
        layer = linears.MlpBlock(
            config=cfg,
            mesh=mesh,
            in_features=in_features,
            intermediate_dim=intermediate_dim,
            rngs=nnx.Rngs(params=0, dropout=1),
        )
        inputs = jax.random.uniform(jax.random.PRNGKey(1), (batch_size, seq_len, in_features), dtype=cfg.dtype)
        return nnx.jit(lambda m, x: m(x))(layer, inputs)

    for fused_mlp in (False, True):
      with self.subTest(fused_mlp=fused_mlp):
        reference = build_and_run(use_two_stage=False, fused_mlp=fused_mlp)
        two_stage = build_and_run(use_two_stage=True, fused_mlp=fused_mlp)

        self.assertEqual(two_stage.shape, (batch_size, seq_len, in_features))
        self.assertTrue(
            jnp.allclose(reference, two_stage, rtol=1e-2, atol=1e-2, equal_nan=False),
            msg=(
                f"two-stage all-gather changed MLP output (fused_mlp={fused_mlp}): max abs diff="
                f"{jnp.max(jnp.abs(reference.astype(jnp.float32) - two_stage.astype(jnp.float32)))}"
            ),
        )


if __name__ == "__main__":
  unittest.main()
