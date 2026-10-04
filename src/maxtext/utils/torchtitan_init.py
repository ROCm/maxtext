# Copyright 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TorchTitan-equivalent DeepSeek-V3 weight initialization for MaxText NNX models.

TorchTitan initializes DeepSeek-V3 with std 0.02 for input projections and the
MLP/expert gate projection, 0.02/sqrt(2*(layer_id+1)) for residual output
projections, the router, and the MLP/expert up and down projections, std 1.0
for the token embedding, and a +/-3 sigma truncated dim**-0.5 output head.
MaxText's default fan-in init differs, and for stacked routed-expert kernels it
counts the expert axis in the fan-in.
"""

import math
import re
import zlib

import jax
import jax.numpy as jnp
from flax import nnx

from maxtext.utils import max_logging

_BASE_STD = 0.02

# First match wins. "keep" leaves the parameter at its MaxText init, which
# already matches TorchTitan (RMSNorm scales = 1, router correction bias = 0).
_RULES = (
    (re.compile(r"/scale$"), "keep"),
    (re.compile(r"/gate/bias$"), "keep"),
    (re.compile(r"^token_embedder/embedding$"), "embed"),
    (re.compile(r"^decoder/logits_dense/kernel$"), "head"),
    (re.compile(r"/self_attention/(wq_a|wq_b|wkv_a|wkv_b)/kernel$"), "base"),
    (re.compile(r"/self_attention/out/kernel$"), "depth"),
    (re.compile(r"/gate/kernel$"), "depth"),
    (re.compile(r"/wi_0(/kernel)?$"), "base"),
    (re.compile(r"/(wi_1|wo)(/kernel)?$"), "depth"),
)


def _depth_std(layer_id):
  return _BASE_STD / math.sqrt(2 * (layer_id + 1))


def _key_name(key):
  for attr in ("key", "name", "idx"):
    if hasattr(key, attr):
      return str(getattr(key, attr))
  return str(key)


def _layer_group(label, config):
  """Returns (first layer_id, number of stacked layers) for a scanned layer group, else None."""
  num_dense = config.first_num_dense_layers
  if label.startswith("decoder/dense_layers/"):
    return 0, num_dense
  if label.startswith("decoder/moe_layers/"):
    return num_dense, config.num_decoder_layers - num_dense
  return None


def _layer_axis(var, shape, num_layers):
  axis = var.get_metadata().get("param_scan_axis")
  if axis is not None and axis < len(shape) and shape[axis] == num_layers:
    return axis
  candidates = [i for i, n in enumerate(shape) if n == num_layers]
  if len(candidates) != 1:
    raise ValueError(f"cannot locate the layer axis of shape {shape} for {num_layers} stacked layers")
  return candidates[0]


def _target_std(kind, label, var, shape, config):
  """Returns (std as a float or a per-layer array broadcastable to shape, layer axis or None)."""
  if kind == "embed":
    return 1.0, None
  if kind == "head":
    return config.emb_dim**-0.5, None
  group = _layer_group(label, config)
  if group is None:
    raise ValueError(f"{label}: per-layer rule outside a scanned layer group")
  first_layer, num_layers = group
  if kind == "base":
    return _BASE_STD, None
  if num_layers == 1:
    return _depth_std(first_layer), None
  axis = _layer_axis(var, shape, num_layers)
  per_layer = [_depth_std(first_layer + i) for i in range(num_layers)]
  bshape = [1] * len(shape)
  bshape[axis] = num_layers
  return jnp.asarray(per_layer, jnp.float32).reshape(bshape), axis


def apply_torchtitan_dsv3_init(model, config):
  """Resamples every trainable parameter of `model` in place with TorchTitan's DeepSeek-V3 scheme."""
  params = nnx.state(model, nnx.Param)
  base_key = jax.random.PRNGKey(config.init_weights_seed)
  counts = {}

  def resample(path, var):
    label = "/".join(n for n in (_key_name(k) for k in path) if n not in ("value", "raw_value"))
    kind = next((k for pattern, k in _RULES if pattern.search(label)), None)
    if kind is None:
      raise ValueError(f"torchtitan_dsv3_init has no rule for parameter {label}")
    counts[kind] = counts.get(kind, 0) + 1
    if kind == "keep":
      return var
    value = var.get_value()
    shape, dtype = value.shape, value.dtype
    std, layer_axis = _target_std(kind, label, var, shape, config)
    key = jax.random.fold_in(base_key, zlib.crc32(label.encode()))

    def sample(old, k):
      del old
      if kind == "head":
        noise = jax.random.truncated_normal(k, -3.0, 3.0, shape, jnp.float32)
      else:
        noise = jax.random.normal(k, shape, jnp.float32)
      return (noise * std).astype(dtype)

    # Donating the old value writes the new one into the same buffer. Allocating fresh
    # buffers instead fragments device memory enough that the first train step's
    # scratch allocation fails.
    new_value = jax.jit(sample, donate_argnums=0, out_shardings=value.sharding)(value, key)
    reduce_axes = tuple(a for a in range(len(shape)) if a != layer_axis)
    measured = jax.jit(lambda x: jnp.std(x.astype(jnp.float32), axis=reduce_axes))(new_value)
    target = jnp.ravel(jnp.asarray(std)).tolist()
    max_logging.log(
        f"torchtitan_dsv3_init {label} shape={shape} rule={kind} "
        f"target_std={[round(t, 6) for t in target]} measured_std={[round(float(m), 6) for m in jnp.ravel(measured)]}"
    )
    return var.replace(value=new_value)

  new_params = jax.tree_util.tree_map_with_path(resample, params, is_leaf=lambda x: isinstance(x, nnx.Variable))
  nnx.update(model, new_params)
  max_logging.log(f"torchtitan_dsv3_init: applied, rule counts {dict(sorted(counts.items()))}")
