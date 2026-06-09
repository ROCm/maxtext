"""MoE active-params correction for MaxText's dense-formula throughput metrics.

MaxText reports prefill TFLOPs as ``2 * num_params * seq_len`` (dense formula;
see ``maxtext/utils/maxtext_utils.py::calculate_prefill_tflops_per_device``)
plus a small attention term, and AR bandwidth as
``(model_size + cache_size) / step_time``. For top-k MoE models only the
active experts touch each token, so absolute numbers are inflated by roughly
``num_experts / num_experts_per_tok``. Per-backend speedup ratios are
unaffected (both scale identically).

``apply_moe_correction(out, params, config)`` rewrites ``out`` in place
with active-params values and preserves the dense values as ``*_dense``
fields. No-op on dense models (no routed-MoE weights detected).
"""

from __future__ import annotations

from typing import Any


_ROUTED_NAMES_3D = ("wi_0", "wi_1", "wo")
_FLY_NAMES = ("fly_w1_shuffled", "fly_w2_shuffled")


def _leaf_array(node: Any):
    """Get the underlying array from an nnx.Param / linen {"kernel": ...} /
    raw array. Returns None for non-array leaves."""
    if hasattr(node, "value") and not hasattr(node, "keys"):
        return node.value
    if hasattr(node, "__getitem__") and hasattr(node, "keys"):
        try:
            if "kernel" in node:
                return node["kernel"]
        except (TypeError, KeyError):
            pass
    if hasattr(node, "size") and hasattr(node, "dtype"):
        return node
    return None


def _leaf_bytes(node: Any) -> int:
    arr = _leaf_array(node)
    if arr is None:
        return 0
    try:
        return int(arr.size) * int(arr.dtype.itemsize)
    except (AttributeError, TypeError):
        return 0


def _leaf_ndim(node: Any) -> int:
    arr = _leaf_array(node)
    return 0 if arr is None else int(getattr(arr, "ndim", 0))


def _is_kernel_dict(node: Any) -> bool:
    if not hasattr(node, "keys"):
        return False
    try:
        return "kernel" in node
    except (TypeError, KeyError):
        return False


def _walk_param_bytes(params: Any) -> tuple[int, int]:
    """Walk ``params``, return ``(routed_expert_bytes, total_bytes)``.

    Routed-MoE leaves are identified by name + rank: ``fly_w*_shuffled`` (any
    rank) or ``wi_0/wi_1/wo`` with ``ndim == 3``. The 3D guard distinguishes
    routed experts ``[E, D, M]`` from gemma's shared MLP ``wi_*`` which is 2D.
    """
    seen: set[int] = set()
    expert = [0]
    total = [0]

    def _walk(obj: Any, last_key: Any = None) -> None:
        if id(obj) in seen:
            return
        seen.add(id(obj))

        arr = _leaf_array(obj)
        if arr is not None and not (hasattr(obj, "keys") and not (
            hasattr(obj, "value") or _is_kernel_dict(obj)
        )):
            n = _leaf_bytes(obj)
            total[0] += n
            if last_key in _FLY_NAMES:
                expert[0] += n
            elif last_key in _ROUTED_NAMES_3D and _leaf_ndim(obj) == 3:
                expert[0] += n
            return

        if hasattr(obj, "keys"):
            try:
                keys = list(obj.keys())
            except Exception:
                return
            for k in keys:
                try:
                    _walk(obj[k], k)
                except (KeyError, TypeError):
                    pass
            return

        if isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v, last_key)
            return

        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                _walk(v, k)

    _walk(params)
    return expert[0], total[0]


def moe_active_ratio(
    params: Any, *, num_experts: int, num_experts_per_tok: int
) -> dict:
    """Return active-params fraction info computed from a params tree.

    ``active_ratio = 1.0`` for dense models (no routed-MoE weights detected).
    """
    expert_bytes, total_bytes = _walk_param_bytes(params)
    if total_bytes == 0 or num_experts <= 0 or num_experts_per_tok <= 0:
        return {
            "expert_bytes": 0,
            "non_expert_bytes": total_bytes,
            "total_bytes": total_bytes,
            "active_bytes": total_bytes,
            "active_ratio": 1.0,
            "expert_share": 0.0,
            "num_experts": int(num_experts),
            "num_experts_per_tok": int(num_experts_per_tok),
        }
    non_expert = total_bytes - expert_bytes
    active = non_expert + (num_experts_per_tok / num_experts) * expert_bytes
    return {
        "expert_bytes": expert_bytes,
        "non_expert_bytes": non_expert,
        "total_bytes": total_bytes,
        "active_bytes": int(round(active)),
        "active_ratio": active / total_bytes,
        "expert_share": expert_bytes / total_bytes,
        "num_experts": int(num_experts),
        "num_experts_per_tok": int(num_experts_per_tok),
    }


def apply_moe_correction(out: dict, params: Any, config: Any) -> dict:
    """Rewrite ``out``'s TFLOPs/sec and bandwidth fields with active-params
    values, in place. Preserves dense values under ``*_dense`` suffixes and
    records the full info dict under ``out["moe_correction"]``. No-op on
    dense models.

    Prefill: only the learnable-weight FLOPs (``2 * num_params * seq_len``)
    are scaled; the small causal-attention term passes through dense.
    AR bandwidth: only ``model_size`` is scaled; ``cache_size`` (KV cache)
    is fully read each step.
    """
    info = moe_active_ratio(
        params,
        num_experts=int(getattr(config, "num_experts", 0) or 0),
        num_experts_per_tok=int(getattr(config, "num_experts_per_tok", 0) or 0),
    )
    out["moe_correction"] = info
    ratio = info["active_ratio"]
    if ratio >= 0.9999:
        return out

    sizes = out.get("sizes") or {}
    num_params = float(sizes.get("model_params_in_billions", 0.0)) * 1e9
    model_size_bytes = float(sizes.get("model_size_in_gb", 0.0)) * 1e9
    cache_size_bytes = float(sizes.get("cache_size_in_gb", 0.0)) * 1e9

    for seq_key, entry in (out.get("prefill") or {}).items():
        if not isinstance(entry, dict) or "total_tflops_per_device" not in entry:
            continue
        try:
            seq = int(seq_key)
        except (TypeError, ValueError):
            continue
        time_ms = float(entry.get("time_in_ms", 0.0))
        total_dense = float(entry["total_tflops_per_device"])
        learnable_dense = 2.0 * num_params * seq / 1e12
        attn_dense = max(0.0, total_dense - learnable_dense)
        total_active = ratio * learnable_dense + attn_dense

        entry["total_tflops_per_device_dense"] = total_dense
        entry["tflops_per_sec_per_device_dense"] = float(entry["tflops_per_sec_per_device"])
        entry["total_tflops_per_device"] = total_active
        entry["tflops_per_sec_per_device"] = (
            total_active / (time_ms / 1000.0) if time_ms > 0 else 0.0
        )

    ar = out.get("autoregressive")
    if isinstance(ar, dict) and "bw_per_device_GB_per_second" in ar:
        step_ms = float(ar.get("step_in_ms", 0.0))
        active_total_bytes = ratio * model_size_bytes + cache_size_bytes
        ar["bw_per_device_GB_per_second_dense"] = float(ar["bw_per_device_GB_per_second"])
        ar["bw_per_device_GB_per_second"] = (
            active_total_bytes / 1e9 / (step_ms / 1000.0) if step_ms > 0 else 0.0
        )

    # Rewrite size/param primary fields to active values; keep dense as *_dense.
    if "model_size_in_gb" in sizes:
        sizes["model_size_in_gb_dense"] = sizes["model_size_in_gb"]
        sizes["model_size_in_gb"] = info["active_bytes"] / 1e9
    if "model_params_in_billions" in sizes:
        sizes["model_params_in_billions_dense"] = sizes["model_params_in_billions"]
        sizes["model_params_in_billions"] = (
            float(sizes["model_params_in_billions_dense"]) * ratio
        )
    sizes["moe_active_ratio"] = ratio
    sizes["moe_expert_share_of_params"] = info["expert_share"]
    out["sizes"] = sizes

    return out
