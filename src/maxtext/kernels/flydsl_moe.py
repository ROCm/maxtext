# SPDX-License-Identifier: Apache-2.0
"""FlyDSL routed-MoE backend for MaxText (selected via ``moe_backend=flydsl``).

This module bridges MaxText's ``RoutedMoE`` to the FlyDSL 2-stage grouped-GEMM
kernels exposed by the ``jax_flydsl`` pip package. It reuses MaxText's own
router (``RoutedMoE.get_topk``) so all model-specific routing (Gemma4 softmax
position, Mixtral top-k, ``norm_topk_prob``) stays in one place.

Requirements (provided by the jax-flydsl container / ``setup_env.sh``):
  * the ``jax_flydsl`` bridge + the ``flydsl_moe`` MoE kernel API on PYTHONPATH
  * FlyDSL ``kernels.*`` checkout on ``PYTHONPATH`` (``flydsl_moe.moe_gemm_available()``)

Weights stay in the stock ``wi_0 / wi_1 / wo`` layout in the checkpoint; the
FlyDSL MFMA preshuffle is applied here (XLA fuses the reshape/transpose).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from maxtext.common import common_types as ctypes


def flydsl_available() -> bool:
    """True if jax_flydsl + the FlyDSL kernels checkout are importable."""
    try:
        from flydsl_moe import moe_gemm_available

        return bool(moe_gemm_available())
    except Exception:
        return False


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


def _wrap_like(orig, new_array):
    """Place ``new_array`` back in the same wrapper kind ``orig`` had."""
    if hasattr(orig, "value"):
        new_p = type(orig).__new__(type(orig))
        new_p.__dict__.update(orig.__dict__)
        new_p.value = new_array
        return new_p
    if hasattr(orig, "__getitem__") and hasattr(orig, "keys"):
        try:
            if "kernel" in orig:
                d = dict(orig)
                d["kernel"] = new_array
                return d
        except (TypeError, KeyError):
            pass
    return new_array


def _is_flydsl_moe_block(d) -> bool:
    """Dict-like sub-tree with the routed-MoE weights AND the fly placeholders."""
    try:
        keys = set(d.keys())
    except (AttributeError, TypeError):
        return False
    return {"wi_0", "wi_1", "wo", "fly_w1_shuffled", "fly_w2_shuffled"}.issubset(keys)


def inject_preshuffled_weights(params, *, verbose: bool = True):
    """Fill every MoE block's ``fly_w1/w2_shuffled`` from its ``wi_0/wi_1/wo`` ONCE.

    Call this once after ``engine.load_params`` (and before ``aot_compile`` /
    the timed loop) when ``moe_backend=flydsl``. The preshuffle then lives in the
    params, not the forward graph, so it is paid a single time. Mutates and
    returns the same ``params`` tree. No-op if no flydsl MoE blocks are found.
    """
    from flydsl_moe.preshuffle import make_w1_shuffled, make_w2_shuffled

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
            if _is_flydsl_moe_block(obj):
                wi_0 = _read_array(obj["wi_0"])
                wi_1 = _read_array(obj["wi_1"])
                wo = _read_array(obj["wo"])
                w1_sh = make_w1_shuffled(wi_0, wi_1)
                w2_sh = make_w2_shuffled(wo)
                jax.block_until_ready((w1_sh, w2_sh))
                obj["fly_w1_shuffled"] = _wrap_like(obj["fly_w1_shuffled"], w1_sh)
                obj["fly_w2_shuffled"] = _wrap_like(obj["fly_w2_shuffled"], w2_sh)
                n[0] += 1
            for k in keys:
                try:
                    _walk(obj[k], _seen)
                except (KeyError, TypeError):
                    pass
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v, _seen)
        elif hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                _walk(v, _seen)

    _walk(params)
    if verbose:
        print(f"[flydsl] inject_preshuffled_weights: filled {n[0]} MoE block(s)")
    return params


def _stage1_activation(cfg) -> str:
    """Map ``cfg.mlp_activations[0]`` to the FlyDSL stage-1 activation.

    gemma4 uses ``["gelu", "linear"]``; Mixtral uses ``["silu", "linear"]``.
    The FlyDSL stage-1 epilogue supports ``silu`` and ``gelu`` (tanh approx,
    matching ``jax.nn.gelu``). Anything else falls back to silu.
    """
    acts = getattr(cfg, "mlp_activations", None) or ["silu"]
    gate_act = str(acts[0]).lower()
    if gate_act in ("gelu", "gelu_approximate", "gelu_tanh"):
        return "gelu"
    return "silu"


def flydsl_matmul(layer, inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel):
    """RoutedMoE compute via FlyDSL kernels.

    Reads the MFMA-preshuffled weights from ``layer.fly_w1_shuffled`` /
    ``layer.fly_w2_shuffled`` (filled ONCE at load time by
    :func:`inject_preshuffled_weights`), so the shuffle is never recomputed in
    the forward pass. ``w0_kernel/w1_kernel/wo_kernel`` are unused here.

    Returns ``(output, lb_loss, bias_updates)`` to match the other RoutedMoE
    backends. Load-balance loss / bias updates are not produced on this
    inference path (returns ``None, None``).
    """
    from flydsl_moe.block import (
        moe_block_fly_bf16_atomic,
        moe_sort_jax,
        pick_tile_m,
    )
    from flydsl_moe.preshuffle import (
        make_w1_shuffled,
        make_w2_shuffled,
        pad_inter_dim,
    )

    cfg = layer.config
    num_experts = layer.num_experts
    topk = layer.num_experts_per_tok

    # Router: reuse MaxText's own top-k (handles every model variant).
    topk_weights, topk_ids = layer.get_topk(gate_logits, pre_bias_logits)

    # Flatten [B, S, D] -> [T, D].
    if inputs.ndim == 3:
        B, S, D = inputs.shape
        T = B * S
        x_2d = inputs.reshape(T, D)
        topk_ids_flat = topk_ids.reshape(T, -1)
        topk_w_flat = topk_weights.reshape(T, -1)
    else:
        T, D = inputs.shape
        x_2d, topk_ids_flat, topk_w_flat = inputs, topk_ids, topk_weights

    # Fast path: preshuffled weights filled once at load (inject_preshuffled_weights,
    # random-init / microbenchmark). Fallback: shuffle inline from wi_0/wi_1/wo when
    # the placeholders aren't populated -- e.g. a real checkpoint restore, where the
    # fly_w*_shuffled params aren't in the checkpoint, so the nnx.Param exists but
    # its .value is None.
    def _fly_value(p):
        # Usable only if populated with a real array/tracer. On a checkpoint
        # restore that lacked these keys the nnx.Param comes back as None or an
        # abstract jax.ShapeDtypeStruct -> treat as "not populated" so the caller
        # falls back to inline shuffle.
        if p is None:
            return None
        v = getattr(p, "value", p)
        if v is None or isinstance(v, jax.ShapeDtypeStruct):
            return None
        return v

    fly_w1_v = _fly_value(getattr(layer, "fly_w1_shuffled", None))
    fly_w2_v = _fly_value(getattr(layer, "fly_w2_shuffled", None))
    if fly_w1_v is not None and fly_w2_v is not None:
        fly_w1 = jnp.asarray(fly_w1_v, layer.dtype)
        fly_w2 = jnp.asarray(fly_w2_v, layer.dtype)
    else:
        fly_w1 = make_w1_shuffled(w0_kernel, w1_kernel)
        fly_w2 = make_w2_shuffled(wo_kernel)
    fly_params = {
        "w1_shuffled": fly_w1,
        "w2_shuffled": fly_w2,
        "scale_w1": jnp.ones((fly_w1.shape[0],), jnp.float32),
        "scale_w2": jnp.ones((fly_w2.shape[0],), jnp.float32),
    }

    tile_m = pick_tile_m(T)
    sorted_ids, sorted_weights, expert_ids, num_valid = moe_sort_jax(
        topk_ids_flat.astype(jnp.int32),
        topk_w_flat.astype(jnp.float32),
        num_experts=num_experts,
        block_size=tile_m,
    )

    inter_dim_padded = pad_inter_dim(cfg.moe_mlp_dim)
    # gemma4 (topk=8): fp16 atomic-add reordering can flip near-tie argmax vs
    # ragged_dot, so use the deterministic JAX-side reduction there. The
    # FLY_STAGE2 env var (atomic|reduce) overrides for diagnostics.
    import os

    is_gemma4 = getattr(cfg, "decoder_block", None) == ctypes.DecoderBlockType.GEMMA4
    stage2_mode = os.environ.get("FLY_STAGE2") or ("reduce" if is_gemma4 else "atomic")

    moe_out_2d = moe_block_fly_bf16_atomic(
        x_2d,
        fly_params,
        sorted_ids,
        expert_ids,
        sorted_weights,
        num_valid,
        tokens=T,
        model_dim=cfg.emb_dim,
        inter_dim=inter_dim_padded,
        num_experts=num_experts,
        topk=topk,
        tile_m=tile_m,
        activation=_stage1_activation(cfg),
        stage2_mode=stage2_mode,
    )

    output = moe_out_2d.reshape(B, S, D) if inputs.ndim == 3 else moe_out_2d
    return output, None, None
