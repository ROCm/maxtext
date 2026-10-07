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

"""Tests for embeddings.py."""

import sys
import unittest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.layers import embeddings
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


class EmbedTest(unittest.TestCase):
  """Tests for Embed."""

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(params=0)

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 128,
    }
    argv = [sys.argv[0], get_test_config_path()]
    self.cfg = pyconfig.initialize(argv, **config_arguments)

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = jax.sharding.Mesh(devices_array, self.cfg.mesh_axes)

  def test_basic_call(self):
    num_embeddings = 100
    num_features = 16
    batch_size = 2
    seq_len = 3

    layer = embeddings.Embed(
        num_embeddings=num_embeddings,
        num_features=num_features,
        config=self.cfg,
        mesh=self.mesh,
        rngs=self.rngs,
    )

    inputs = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    outputs = layer(inputs)

    self.assertEqual(outputs.shape, (batch_size, seq_len, num_features))

  def test_attend(self):
    num_embeddings = 100
    num_features = 16
    batch_size = 2
    seq_len = 3

    layer = embeddings.Embed(
        num_embeddings=num_embeddings,
        num_features=num_features,
        config=self.cfg,
        mesh=self.mesh,
        rngs=self.rngs,
    )

    query = jnp.ones((batch_size, seq_len, num_features))
    outputs = layer.attend(query)

    self.assertEqual(outputs.shape, (batch_size, seq_len, num_embeddings))


class RotaryEmbeddingTest(unittest.TestCase):
  """Tests for RotaryEmbedding."""

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(params=0)

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 128,
    }
    argv = [sys.argv[0], get_test_config_path()]
    self.cfg = pyconfig.initialize(argv, **config_arguments)

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = jax.sharding.Mesh(devices_array, self.cfg.mesh_axes)

  def test_basic_call(self):
    layer = embeddings.RotaryEmbedding(
        min_timescale=1,
        max_timescale=10000,
        mesh=self.mesh,
        embedding_dims=4,
        rngs=self.rngs,
    )

    inputs = jnp.ones((1, 2, 1, 4))
    position = jnp.array([[0, 1]])

    outputs = layer(inputs, position=position)

    self.assertEqual(outputs.shape, (1, 2, 1, 4))

    # Snapshot verification
    expected = jnp.array([[[[1.0, 1.0, 1.0, 1.0]], [[-0.300781, 0.988281, 1.38281, 1.00781]]]])
    np.testing.assert_allclose(outputs, expected, atol=1e-5)


class LLaMARotaryEmbeddingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(params=0)

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 128,
    }
    argv = [sys.argv[0], get_test_config_path()]
    self.cfg = pyconfig.initialize(argv, **config_arguments)

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = jax.sharding.Mesh(devices_array, self.cfg.mesh_axes)

  def test_basic_call(self):
    layer = embeddings.LLaMARotaryEmbedding(
        min_timescale=1,
        max_timescale=10000,
        mesh=self.mesh,
        embedding_dims=4,
        use_scale=True,
        rngs=self.rngs,
    )
    inputs = jnp.ones((1, 2, 1, 4))
    position = jnp.array([[0, 1]])
    outputs = layer(inputs, position=position)
    self.assertEqual(outputs.shape, (1, 2, 1, 4))

    # Snapshot verification
    expected = jnp.array([[[[1.0, 1.0, 1.0, 1.0]], [[-0.300781, 1.38281, 0.988281, 1.00781]]]])
    np.testing.assert_allclose(outputs, expected, atol=1e-5)


class YarnRotaryEmbeddingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(params=0)

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 128,
    }
    argv = [sys.argv[0], get_test_config_path()]
    self.cfg = pyconfig.initialize(argv, **config_arguments)

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = jax.sharding.Mesh(devices_array, self.cfg.mesh_axes)

  def test_basic_call(self):
    layer = embeddings.YarnRotaryEmbedding(
        embedding_dims=4,
        mesh=self.mesh,
        max_position_embeddings=16384,
        original_max_position_embeddings=4096,
        rngs=self.rngs,
    )
    inputs = jnp.ones((1, 2, 1, 4))
    position = jnp.array([[0, 1]])
    outputs = layer(inputs, position=position)
    self.assertEqual(outputs.shape, (1, 2, 1, 4))

    # Snapshot verification
    expected = jnp.array([[[[1.0, 1.0, 1.0, 1.0]], [[-0.300781, 0.996094, 1.38281, 1.00781]]]])
    np.testing.assert_allclose(outputs, expected, atol=1e-5)


@unittest.skipIf(jax.device_count() != 8, "requires eight devices; on CPU set --xla_force_host_platform_device_count=8")
class RailAlignedEmbeddingLookupTest(unittest.TestCase):

  def test_matches_gather_forward_and_gradient(self):
    P = jax.sharding.PartitionSpec
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("expert",))
    table_pspec, out_pspec = P(None, "expert"), P("expert", None, None)
    k_table, k_ids, k_cot = jax.random.split(jax.random.key(3), 3)
    table = jax.device_put(
        jax.random.normal(k_table, (37, 8 * 3), jnp.float32), jax.sharding.NamedSharding(mesh, table_pspec)
    )
    ids = jax.device_put(
        jax.random.randint(k_ids, (16, 5), 0, 37), jax.sharding.NamedSharding(mesh, P("expert", None))
    )
    cotangent = jax.random.normal(k_cot, (16, 5, 24), jnp.float32)

    for local_shards in (2, 4, 8):
      lookup = jax.jit(
          lambda t, i, n=local_shards: embeddings.rail_aligned_embedding_lookup(t, i, mesh, table_pspec, out_pspec, n)
      )
      output, pullback = jax.vjp(lambda t, f=lookup: f(t, ids), table)
      np.testing.assert_array_equal(np.asarray(output), np.asarray(table)[np.asarray(ids)], err_msg=str(local_shards))
      expected_grad = np.zeros(table.shape, np.float32)
      np.add.at(expected_grad, np.asarray(ids), np.asarray(cotangent))
      np.testing.assert_allclose(np.asarray(pullback(cotangent)[0]), expected_grad, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
  unittest.main()
