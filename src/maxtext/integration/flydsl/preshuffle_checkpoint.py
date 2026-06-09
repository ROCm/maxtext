# SPDX-License-Identifier: Apache-2.0
r"""Offline FlyDSL preshuffle for a MaxText parameter checkpoint.

Bakes the FlyDSL MFMA preshuffle into a checkpoint *once*, offline, so that
``moe_backend=flydsl`` inference restores ready-to-use ``fly_w1_shuffled`` /
``fly_w2_shuffled`` weights directly (no runtime inline shuffle, no abstract
placeholder that breaks ``init_decode_state``).

Why this exists
---------------
The flydsl backend declares ``fly_w1_shuffled`` / ``fly_w2_shuffled`` as model
params. A stock checkpoint does not contain those keys, so restoring it with
``moe_backend=flydsl`` leaves them abstract -> the engine's internal decode
state stays abstract and ``init_decode_state`` crashes. Injecting at runtime
only patches the local params dict, not the engine state, so it is fragile.

This tool instead loads the stock checkpoint with the *default* backend (so the
abstract tree matches the on-disk keys), computes the shuffled weights from
``wi_0 / wi_1 / wo`` per MoE block, inserts them alongside the originals, and
writes a new parameter checkpoint. Restoring *that* with ``moe_backend=flydsl``
gives concrete fly weights everywhere -> decode just works, at full speed.

Run the converter and the eventual inference with the *same* model config
(notably ``scan_layers``), so the param tree structure matches on restore.

Example
-------
python3 -m maxtext.integration.flydsl.preshuffle_checkpoint \
    src/maxtext/configs/base.yml \
    model_name=mixtral-8x7b tokenizer_path=${TOKENIZER_PATH?} \
    load_parameters_path=${STOCK_CKPT?}/0/items \
    save_quantized_params_path=${OUT_CKPT?}/0/items \
    scan_layers=false weight_dtype=bfloat16 per_device_batch_size=1 \
    ici_fsdp_parallelism=1 ici_tensor_parallelism=1 async_checkpointing=false \
    checkpoint_storage_use_ocdbt=false checkpoint_storage_use_zarr3=false

Then run inference with ``moe_backend=flydsl`` and
``load_parameters_path=${OUT_CKPT}/0/items`` (no inject step needed).
"""

from __future__ import annotations

import functools
from typing import Any, Sequence

from absl import app
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp

from maxtext.common import common_types
from maxtext.common import checkpointing
from maxtext.configs import pyconfig
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils


def _read_array(node):
    """Extract a jax.Array from an nnx.Param, linen {"kernel": ...}, or raw array."""
    if hasattr(node, "value"):
        return jnp.asarray(node.value)
    if hasattr(node, "__getitem__") and hasattr(node, "keys"):
        try:
            if "kernel" in node:
                return jnp.asarray(node["kernel"])
        except (TypeError, KeyError):
            pass
    return jnp.asarray(node)


def _is_moe_block(d) -> bool:
    """A dict-like subtree holding the routed-MoE expert weights."""
    try:
        keys = set(d.keys())
    except (AttributeError, TypeError):
        return False
    return {"wi_0", "wi_1", "wo"}.issubset(keys)


def _shuffle_block(wi_0, wi_1, wo):
    """Compute (fly_w1_shuffled, fly_w2_shuffled), handling scanned layers.

    Per-layer weights are ``[E, D, M]`` / ``[E, M, D]``. With ``scan_layers``
    they gain a leading layer axis ``[L, E, ...]``; we vmap the shuffle over it
    so the fly weights keep the matching ``[L, ...]`` leading axis.
    """
    from flydsl_moe.preshuffle import make_w1_shuffled, make_w2_shuffled

    if wi_0.ndim == 4:  # scanned: [L, E, D, M]
        w1 = jax.vmap(make_w1_shuffled)(wi_0, wi_1)
        w2 = jax.vmap(make_w2_shuffled)(wo)
    else:  # per-layer: [E, D, M]
        w1 = make_w1_shuffled(wi_0, wi_1)
        w2 = make_w2_shuffled(wo)
    return w1, w2


def _is_routed_expert_weight(wi_0, num_experts) -> bool:
    """True only for routed-MoE expert weights, not dense MLP weights.

    Routed-MoE ``wi_0`` carries an expert axis: ``[E, D, M]`` (per-layer) or
    ``[L, E, D, M]`` (scanned). A dense ``MlpBlock`` shares the ``wi_0/wi_1/wo``
    key names but has no expert axis (``[D, M]`` / ``[L, D, M]``), so it must be
    skipped or the gate/up fuse unpacks the wrong number of dims.
    """
    return wi_0.ndim >= 3 and num_experts in wi_0.shape[:-2]


def insert_preshuffled_weights(params, *, weight_dtype, num_experts):
    """Add ``fly_w1_shuffled`` / ``fly_w2_shuffled`` to every routed-MoE block.

    Mutates ``params`` in-place. Returns the count of blocks converted so the
    caller can fail loudly if a flydsl run would find nothing to use. Dense MLP
    blocks (same key names, no expert axis) are left untouched.
    """
    n = [0]

    def _walk(obj, _seen=None):
        if _seen is None:
            _seen = set()
        if id(obj) in _seen:
            return
        _seen.add(id(obj))
        if hasattr(obj, "keys"):
            try:
                keys = list(obj.keys())
            except Exception:
                return
            if _is_moe_block(obj) and "fly_w1_shuffled" not in obj:
                wi_0 = _read_array(obj["wi_0"]).astype(weight_dtype)
                if _is_routed_expert_weight(wi_0, num_experts):
                    wi_1 = _read_array(obj["wi_1"]).astype(weight_dtype)
                    wo = _read_array(obj["wo"]).astype(weight_dtype)
                    w1_sh, w2_sh = _shuffle_block(wi_0, wi_1, wo)
                    jax.block_until_ready((w1_sh, w2_sh))
                    obj["fly_w1_shuffled"] = w1_sh
                    obj["fly_w2_shuffled"] = w2_sh
                    n[0] += 1
            for k in keys:
                try:
                    _walk(obj[k], _seen)
                except (KeyError, TypeError):
                    pass
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v, _seen)

    _walk(params)
    return n[0]


def _load_stock_params(config, rng):
    """Load the full parameter tree from ``config.load_parameters_path``."""
    devices_array = maxtext_utils.create_device_mesh(config=config)
    mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
    quant = quantizations.configure_quantization(config)
    model = models.transformer_as_linen(
        config, mesh=mesh, quant=quant, model_mode=common_types.MODEL_MODE_TRAIN
    )
    init_state_fn = functools.partial(maxtext_utils.init_initial_state, model, None, config, False, rng)
    unboxed_abstract_state, _, _ = maxtext_utils.get_abstract_state(config, mesh, init_state_fn, False)
    with nn_partitioning.axis_rules(config.logical_axis_rules):
        params = checkpointing.load_params_from_path(
            config.load_parameters_path,
            unboxed_abstract_state.params,
            config.checkpoint_storage_concurrent_gb,
            config.checkpoint_storage_use_ocdbt,
            config.checkpoint_storage_use_zarr3,
        )
    return params


def main(argv: Sequence[str]) -> None:
    config = pyconfig.initialize(argv)
    _validate(config)
    max_utils.print_system_information()

    rng = jax.random.PRNGKey(1234)
    max_logging.log(f"[flydsl] loading stock params from {config.load_parameters_path}")
    params = _load_stock_params(config, rng)

    max_logging.log("[flydsl] computing MFMA preshuffle for each MoE block...")
    converted = insert_preshuffled_weights(
        params, weight_dtype=config.weight_dtype, num_experts=config.num_experts
    )
    if converted == 0:
        raise ValueError(
            "No MoE blocks (wi_0/wi_1/wo) found in the checkpoint. Is this an MoE "
            "model loaded with the right model_name / scan_layers?"
        )
    max_logging.log(f"[flydsl] preshuffled {converted} MoE block(s)")

    out = config.save_quantized_params_path
    checkpointing.save_params_to_path(
        checkpoint_dir=out,
        params=params,
        use_ocdbt=config.checkpoint_storage_use_ocdbt,
        use_zarr3=config.checkpoint_storage_use_zarr3,
    )
    max_logging.log(
        f"[flydsl] wrote preshuffled checkpoint to {out}\n"
        f"[flydsl] run inference with: moe_backend=flydsl load_parameters_path={out}"
    )


def _validate(config):
    assert config.load_parameters_path, "Set load_parameters_path to the stock checkpoint (.../0/items)."
    assert config.save_quantized_params_path, (
        "Set save_quantized_params_path to the output checkpoint dir (.../0/items)."
    )
    assert config.load_full_state_path == "", "Convert a parameter checkpoint, not a full training state."


if __name__ == "__main__":
    app.run(main)
