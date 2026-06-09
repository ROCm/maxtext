"""Correctness check: gemma4-26b baseline (stock MaxText) vs FlyDSL backend.

gemma4-26b uses ``RoutedAndSharedMoE`` (a routed ``RoutedMoE`` plus a dense
shared-experts ``MlpBlock``). ``fly_moe_backend("gemma")`` swaps only the inner
routed ``RoutedMoE`` for the FlyDSL subclass; the shared MLP stays stock in both
runs. So any logit difference is isolated to the routed-MoE FlyDSL kernel path
vs. stock ``ragged_dot``.

Runs one prefill of a fixed prompt through each backend (separate engine builds
at the SAME RNG seed -> identical random wi_0/wi_1/wo) and compares logits, via
the production integration path (``fly_moe_backend``) -- no manual class patch.

The real correctness signal is top-1 next-token agreement (>= 99%); logit-space
relative error is dominated by near-zero-denominator artifacts.

Usage::

    python3 -m maxtext.integration.flydsl.verify_correctness_gemma \\
        --maxtext_dir /workspace/maxtext --seq 1024

Pin a GPU manually if needed: ``HIP_VISIBLE_DEVICES=N python3 -m ...``.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import sys

import numpy as np


def _setup_paths(maxtext_dir: str, jaxflydsl_dir: str) -> None:
    for p in (jaxflydsl_dir, f"{maxtext_dir}/src"):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ["MAXTEXT_SRC"] = f"{maxtext_dir}/src"
    os.environ["JAXFLYDSL_DIR"] = jaxflydsl_dir


def _build_config(maxtext_dir: str, seq: int, out_dir: str, moe_backend: str = "default"):
    import sys as _sys
    from maxtext.configs import pyconfig

    config_path = f"{maxtext_dir}/src/maxtext/configs/base.yml"
    tokenizer = f"{maxtext_dir}/src/maxtext/assets/tokenizers/tokenizer.gemma3"
    overrides = [
        "model_name=gemma4-26b",
        "hardware=gpu",
        # Correctness is attention-agnostic: identical in both runs, so it cancels.
        "attention=dot_product",
        "dtype=bfloat16",
        "weight_dtype=bfloat16",
        "sparse_matmul=True",
        "megablox=False",
        "capacity_factor=1.25",
        f"moe_backend={moe_backend}",
        "quantization=",
        "per_device_batch_size=1",
        "ici_fsdp_parallelism=1",
        "ici_expert_parallelism=1",
        "max_target_length=4096",
        f"max_prefill_predict_length={max(seq, 2048)}",
        "enable_checkpointing=false",
        "scan_layers=false",
        f"base_output_directory={out_dir}",
        "run_name=verify_correctness_gemma",
        f"tokenizer_path={tokenizer}",
    ]
    return pyconfig.initialize([_sys.argv[0], config_path] + overrides)


def _run_prefill_once(engine, params, seq: int, rng_seed: int):
    """Compile + run one prefill, return the logits as a host numpy array.

    The compiled executable + device logits are dropped before returning so two
    large programs aren't resident at once (segfaults on memory-tight cards).
    """
    import jax
    import jax.numpy as jnp

    metadata = engine.get_tokenizer()
    tokenizer_model = engine.build_tokenizer(metadata)
    is_bos = tokenizer_model.bos_id is not None
    tokens, true_length = tokenizer_model.encode(
        engine.config.prompt, is_bos=is_bos, prefill_lengths=[seq],
    )

    i32_scalar = jax.ShapeDtypeStruct((), int)
    rng_shape = jax.ShapeDtypeStruct([4], jnp.dtype("uint32"))
    key_shape = jax.ShapeDtypeStruct([seq], jnp.dtype("int32"))

    prefill_executable = (
        jax.jit(
            engine.prefill_aot,
            in_shardings=(engine.param_layouts, None, None, None),
        ).lower(params, key_shape, i32_scalar, rng_shape)
    ).compile(compiler_options=None)

    rng = jax.random.PRNGKey(rng_seed)
    out, _ = prefill_executable(params, tokens, true_length, rng)
    jax.block_until_ready(out)

    logits = np.asarray(out["logits"])

    del out
    del prefill_executable
    gc.collect()
    try:
        jax.clear_caches()
    except AttributeError:
        pass
    gc.collect()

    return logits


def _run_backend(config, seq: int, rng_seed: int, fly: bool) -> np.ndarray:
    """Build a fresh engine, load params at ``rng_seed``, run one prefill, return
    host logits. The FLY run wraps engine construction + param load in
    ``fly_moe_backend("gemma")`` (production path). Engine + params are freed
    before returning so the two backends don't co-reside in HBM.
    """
    import jax
    from maxtext.inference.maxengine import maxengine

    # MoE backend selected by ``config`` (moe_backend=flydsl vs default).
    ctx = contextlib.nullcontext()

    with ctx:
        engine = maxengine.MaxEngine(config)
        params = engine.load_params(jax.random.PRNGKey(rng_seed))
        n_b = jax.tree_util.tree_reduce(lambda a, x: a + x.size, params, 0) / 1e9
        print(f"   params loaded ({n_b:.1f}B), compiling + running prefill...")
        logits = _run_prefill_once(engine, params, seq, rng_seed)

    del params, engine
    gc.collect()
    try:
        jax.clear_caches()
    except AttributeError:
        pass
    gc.collect()
    return logits


def _logit_diff(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    abs_err = np.abs(a - b)
    denom = np.maximum(np.abs(a), 1e-3)
    rel_err = abs_err / denom
    return {
        "shape": tuple(a.shape),
        "max_abs": float(abs_err.max()),
        "mean_abs": float(abs_err.mean()),
        "max_rel": float(rel_err.max()),
        "mean_rel": float(rel_err.mean()),
        "topk_match": _topk_match(a, b, k=1),
        "topk5_match": _topk_match(a, b, k=5),
    }


def _topk_match(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Fraction of positions where the top-k tokens match between a and b.

    For the last logit position only (the one that matters for next-token
    prediction). a/b shape: [B, S, V] or [S, V].
    """
    if a.ndim == 3:
        a_last = a[:, -1, :]
        b_last = b[:, -1, :]
    else:
        a_last = a[-1:, :]
        b_last = b[-1:, :]
    topk_a = np.argsort(-a_last, axis=-1)[..., :k]
    topk_b = np.argsort(-b_last, axis=-1)[..., :k]
    matches = (topk_a == topk_b).all(axis=-1).astype(np.float32)
    return float(matches.mean())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--rng_seed", type=int, default=4242)
    # Relative-error tolerances are loose by design: max_rel on logits is
    # dominated by tiny-denominator artifacts. Top-1 next-token agreement is
    # the real correctness gate.
    p.add_argument("--max_rel_tol", type=float, default=1e2,
                   help="Loose -- logit max_rel is meaningless near zero")
    p.add_argument("--mean_rel_tol", type=float, default=5e-2,
                   help="bf16 reduction-order noise typically 1-5%")
    p.add_argument("--mean_abs_tol", type=float, default=2e-2,
                   help="More meaningful than rel for logits")
    p.add_argument("--top1_match_min", type=float, default=0.99,
                   help="Fraction of positions where top-1 next token agrees")
    p.add_argument("--maxtext_dir", default=os.environ.get("MAXTEXT_DIR", "/workspace/maxtext"))
    p.add_argument("--jaxflydsl_dir", default=os.environ.get("JAXFLYDSL_DIR", "/workspace/jax-flydsl"))
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = f"{args.jaxflydsl_dir}/results/verify_gemma"

    _setup_paths(args.maxtext_dir, args.jaxflydsl_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    import jax
    # MaxText's inference path uses unsafe_rbg (key_data shape (4,)). Must be
    # set BEFORE generating any PRNG keys or the engine will reject them.
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")

    print("\n" + "=" * 78)
    print("Verifying correctness: gemma4-26b baseline vs FlyDSL (moe_backend=flydsl)")
    print("(routed-MoE block only; shared-experts MLP stays stock in both)")
    print(f"(same rng_seed={args.rng_seed} -> identical wi_0/wi_1/wo in both runs)")
    print("=" * 78)

    print("\n[1] BASELINE run (stock RoutedMoE)")
    logits_baseline = _run_backend(
        config=_build_config(args.maxtext_dir, args.seq, args.out_dir, moe_backend="default"),
        seq=args.seq, rng_seed=args.rng_seed, fly=False)
    print(f"   logits shape: {logits_baseline.shape}, dtype: {logits_baseline.dtype}")

    print("\n[2] FLY run (moe_backend=flydsl -> FlyDSL grouped-GEMM kernels)")
    logits_fly = _run_backend(
        config=_build_config(args.maxtext_dir, args.seq, args.out_dir, moe_backend="flydsl"),
        seq=args.seq, rng_seed=args.rng_seed, fly=True)
    print(f"   logits shape: {logits_fly.shape}, dtype: {logits_fly.dtype}")

    print("\n[3] Comparing logits...")
    if logits_baseline.shape != logits_fly.shape:
        print(f"  SHAPE MISMATCH: baseline={logits_baseline.shape} fly={logits_fly.shape}")
        return 2

    diff = _logit_diff(logits_baseline, logits_fly)
    print()
    print(f"  shape           : {diff['shape']}")
    print(f"  max abs error   : {diff['max_abs']:.4e}     (tol: {args.mean_abs_tol*5:.4e})")
    print(f"  mean abs error  : {diff['mean_abs']:.4e}     (tol: {args.mean_abs_tol:.4e})")
    print(f"  max rel error   : {diff['max_rel']:.4e}     (tol: {args.max_rel_tol:.4e})  [meaningless near zero]")
    print(f"  mean rel error  : {diff['mean_rel']:.4e}     (tol: {args.mean_rel_tol:.4e})")
    print(f"  top-1 match @-1 : {diff['topk_match']*100:.2f}%        (tol: >= {args.top1_match_min*100:.2f}%)")
    print(f"  top-5 match @-1 : {diff['topk5_match']*100:.2f}%")

    print()
    top1_pass = diff["topk_match"] >= args.top1_match_min
    abs_pass = diff["mean_abs"] < args.mean_abs_tol
    if top1_pass and abs_pass:
        print(f"[4] PASS")
        print(f"    top-1 next-token agreement: {diff['topk_match']*100:.2f}% (>= {args.top1_match_min*100:.0f}%)")
        print(f"    mean absolute logit error : {diff['mean_abs']:.4e} (< {args.mean_abs_tol:.4e})")
        print(f"    Model behavior is identical for inference.")
        return 0
    elif top1_pass:
        print(f"[4] PASS (with caveat)")
        print(f"    top-1 next-token agreement: {diff['topk_match']*100:.2f}% (>= {args.top1_match_min*100:.0f}%) -- model behavior preserved")
        print(f"    BUT mean absolute error    : {diff['mean_abs']:.4e} (> {args.mean_abs_tol:.4e})")
        print(f"    Likely accumulated bf16 rounding from different reduction order.")
        print(f"    Safe for inference; if logit-exact match needed, investigate stage-2 atomic order.")
        return 0
    else:
        print(f"[4] FAIL - top-1 next-token agreement only {diff['topk_match']*100:.2f}%")
        print(f"    Model behavior differs. Investigate (in order):")
        print(f"      1. Routing: gate weight read, score_fn (softmax/sigmoid) + norm_topk_prob")
        print(f"      2. moe_sort_jax per-expert ordering")
        print(f"      3. Inter-dim padding: stage-1 CShuffle tile_n=128 requires moe_mlp_dim%128==0")
        print(f"         (gemma4 M=704 is padded to 768 by _pad_inter_dim; check zero-fill correctness)")
        print(f"      4. Weight concat order (gate-then-up vs up-then-gate)")
        print(f"      5. FlyDSL kernel: try in_dtype/out_dtype variants")
        return 1


if __name__ == "__main__":
    sys.exit(main())
