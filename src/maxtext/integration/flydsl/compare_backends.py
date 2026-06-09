"""Three-way MoE backend comparison: ragged_dot vs dense vs fly, with a report.

Builds three MaxEngines at the SAME rng_seed (so weights are identical) -- one
per MoE path -- runs the same prompts through each, and reports:
  * the predicted next token (decoded) from each backend, per prompt, and
  * pairwise logit agreement (top-1 match, top-k set overlap, cosine, mean abs).

Because the weights are identical, any difference is attributable to the MoE
path alone. ``ragged_dot`` (sparse, dropless, exact) and ``fly`` (sparse,
dropless, exact) should agree to the bf16 floor; ``dense`` (capacity-limited)
may diverge if routing exceeds expert capacity.

Backends:
  * ragged : sparse_matmul=True,  megablox=False  (stock exact, slow on ROCm)
  * dense  : sparse_matmul=False, megablox=False  (stock, capacity-drops)
  * fly    : FlyRoutedMoE via fly_moe_backend       (FlyDSL kernel, exact)

Usage:
  python3 -m maxtext.integration.flydsl.compare_backends --model mixtral --maxtext_dir /workspace/maxtext
  python3 -m maxtext.integration.flydsl.compare_backends --model gemma   --maxtext_dir /workspace/maxtext
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import sys

import numpy as np

# model_tag -> (model_name, tokenizer_asset)
_MODELS = {
    "mixtral": ("mixtral-8x7b", "tokenizer.mistral-v1"),
    "gemma": ("gemma4-26b", "tokenizer.gemma3"),
}

_PROMPTS = [
    "The capital of France is",
    "In a galaxy far, far away,",
    "def fibonacci(n):",
    "Photosynthesis is the process by which",
    "The quick brown fox jumps over the",
    "Once upon a time, there was a",
    "The first president of the United States was",
    "Water boils at a temperature of",
    "The largest planet in our solar system is",
    "import numpy as np",
]


def _setup_paths(maxtext_dir: str, jaxflydsl_dir: str) -> None:
    for p in (jaxflydsl_dir, f"{maxtext_dir}/src"):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ["MAXTEXT_SRC"] = f"{maxtext_dir}/src"
    os.environ["JAXFLYDSL_DIR"] = jaxflydsl_dir


def _build_config(
    maxtext_dir,
    model_name,
    tokenizer_name,
    seq,
    out_dir,
    sparse_matmul,
    load_parameters_path=None,
    scan_layers=False,
    moe_backend="default",
):
    from maxtext.configs import pyconfig

    cfg = f"{maxtext_dir}/src/maxtext/configs/base.yml"
    tok = f"{maxtext_dir}/src/maxtext/assets/tokenizers/{tokenizer_name}"
    overrides = [
        f"model_name={model_name}", "hardware=gpu", "attention=dot_product",
        "dtype=bfloat16", "weight_dtype=bfloat16",
        f"sparse_matmul={sparse_matmul}", "megablox=False", "capacity_factor=1.25",
        f"moe_backend={moe_backend}",
        "quantization=", "per_device_batch_size=1",
        "ici_fsdp_parallelism=1", "ici_expert_parallelism=1",
        "max_target_length=4096", f"max_prefill_predict_length={max(seq, 2048)}",
        f"enable_checkpointing={'true' if load_parameters_path else 'false'}",
        f"scan_layers={'true' if scan_layers else 'false'}",
        f"base_output_directory={out_dir}", "run_name=compare_backends",
        f"tokenizer_path={tok}",
    ]
    if load_parameters_path:
        overrides.append(f"load_parameters_path={load_parameters_path}")
    return pyconfig.initialize([sys.argv[0], cfg] + overrides)


def _run(config, seed, seq, prompts):
    """Build a fresh engine, prefill each prompt, return (logits_per_prompt, tok).

    The MoE backend is selected entirely by ``config`` (``moe_backend=flydsl``
    vs ``default``) -- no monkeypatching. Engine + params are freed before
    returning; the tokenizer is kept.
    """
    import jax
    import jax.numpy as jnp
    from maxtext.inference.maxengine import maxengine

    out = []
    engine = maxengine.MaxEngine(config)
    params = engine.load_params(jax.random.PRNGKey(seed))

    tok = engine.build_tokenizer(engine.get_tokenizer())
    is_bos = tok.bos_id is not None
    i32 = jax.ShapeDtypeStruct((), int)
    rng_shape = jax.ShapeDtypeStruct([4], jnp.dtype("uint32"))
    key_shape = jax.ShapeDtypeStruct([seq], jnp.dtype("int32"))
    exe = jax.jit(
        engine.prefill_aot, in_shardings=(engine.param_layouts, None, None, None)
    ).lower(params, key_shape, i32, rng_shape).compile(compiler_options=None)

    rng = jax.random.PRNGKey(seed)
    for prompt in prompts:
        tokens, true_length = tok.encode(prompt, is_bos=is_bos, prefill_lengths=[seq])
        res, _ = exe(params, tokens, true_length, rng)
        jax.block_until_ready(res)
        logits = np.asarray(res["logits"]).reshape(-1, np.asarray(res["logits"]).shape[-1])
        out.append((logits[-1].astype(np.float32), int(true_length), np.asarray(tokens)))
        del res
    del exe, params, engine

    gc.collect()
    try:
        jax.clear_caches()
    except AttributeError:
        pass
    gc.collect()
    return out, tok


def _decode(tok, tid: int) -> str:
    for meth in ("decode", "detokenize"):
        fn = getattr(tok, meth, None)
        if fn is not None:
            try:
                s = fn([int(tid)])
                return (s if isinstance(s, str) else str(s)).replace("\n", "\\n")
            except Exception:
                pass
    return f"<{tid}>"


def _pair(a: np.ndarray, b: np.ndarray, k: int = 5) -> dict:
    a = a.astype(np.float64); b = b.astype(np.float64)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    ta = set(np.argsort(-a)[:k].tolist()); tb = set(np.argsort(-b)[:k].tolist())
    return {
        "top1_match": int(int(np.argmax(a)) == int(np.argmax(b))),
        "top5_overlap": len(ta & tb) / k,
        "cosine": cos,
        "mean_abs": float(np.abs(a - b).mean()),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(_MODELS), required=True)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--rng_seed", type=int, default=4242)
    p.add_argument("--maxtext_dir", default=os.environ.get("MAXTEXT_DIR", "/workspace/maxtext"))
    p.add_argument("--jaxflydsl_dir", default=os.environ.get("JAXFLYDSL_DIR", "/workspace/jax-flydsl"))
    p.add_argument("--out_dir", default=None)
    p.add_argument("--load_parameters_path", default=None,
                   help="Orbax checkpoint path (real weights) instead of random init")
    args = p.parse_args()

    model_name, tok_name = _MODELS[args.model]
    if args.out_dir is None:
        args.out_dir = f"{args.jaxflydsl_dir}/results/compare_{args.model}"
    _setup_paths(args.maxtext_dir, args.jaxflydsl_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    import jax
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")

    def cfg(sparse, moe_backend="default"):
        return _build_config(
            args.maxtext_dir, model_name, tok_name, args.seq, args.out_dir, sparse,
            load_parameters_path=args.load_parameters_path, moe_backend=moe_backend,
        )

    print(f"[1/3] ragged_dot (sparse_matmul=True)...")
    ragged, tok = _run(cfg("True"), args.rng_seed, args.seq, prompts=_PROMPTS)
    print(f"[2/3] dense (sparse_matmul=False)...")
    dense, _ = _run(cfg("False"), args.rng_seed, args.seq, prompts=_PROMPTS)
    print(f"[3/3] fly (moe_backend=flydsl)...")
    fly, _ = _run(cfg("True", moe_backend="flydsl"), args.rng_seed, args.seq, prompts=_PROMPTS)

    lines = []
    lines.append(f"# Backend comparison report: {model_name}")
    lines.append("")
    lines.append(f"- seq={args.seq}, rng_seed={args.rng_seed}, per_device_batch=1, bf16, single device")
    lines.append("- Same weights across all three (identical seed) -> differences are the MoE path only.")
    lines.append(f"- weights: {args.load_parameters_path or 'random init (no checkpoint)'}")
    lines.append("- ragged & fly are exact/dropless; dense is capacity-limited (may drop tokens).")
    lines.append("")
    lines.append("## Predicted next token per prompt")
    lines.append("")
    lines.append("| # | prompt | ragged | dense | fly | all agree? |")
    lines.append("|---|--------|--------|-------|-----|-----------|")
    for i, prompt in enumerate(_PROMPTS):
        rg = _decode(tok, int(np.argmax(ragged[i][0])))
        dn = _decode(tok, int(np.argmax(dense[i][0])))
        fl = _decode(tok, int(np.argmax(fly[i][0])))
        agree = "yes" if rg == dn == fl else "NO"
        lines.append(f"| {i} | {prompt!r} | {rg!r} | {dn!r} | {fl!r} | {agree} |")
    lines.append("")
    lines.append("## Pairwise logit agreement (per prompt)")
    lines.append("")
    lines.append("| # | pair | top-1 | top-5 overlap | cosine | mean_abs |")
    lines.append("|---|------|-------|---------------|--------|----------|")
    for i in range(len(_PROMPTS)):
        for label, a, b in [("fly vs ragged", fly[i][0], ragged[i][0]),
                            ("fly vs dense", fly[i][0], dense[i][0]),
                            ("ragged vs dense", ragged[i][0], dense[i][0])]:
            m = _pair(a, b)
            lines.append(f"| {i} | {label} | {'Y' if m['top1_match'] else 'N'} | "
                        f"{m['top5_overlap']*100:.0f}% | {m['cosine']:.5f} | {m['mean_abs']:.3e} |")
    report = "\n".join(lines)
    path = f"{args.out_dir}/report.md"
    with open(path, "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    print(f"\n[report written to {path}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
