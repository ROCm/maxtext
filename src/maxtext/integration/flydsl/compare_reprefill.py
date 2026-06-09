"""Re-prefill vs KV-cache decode diagnostic (not a fix).

For each prompt:
  1. prefill(prompt) -> token1
  2. re-prefill(prompt + token1) -> token2_rep (no KV cache)
  3. prefill -> insert -> generate() -> token2_dec (KV cache)

When token2_rep is coherent but token2_dec is garbage, the bug is in MaxText's
KV-cache decode path — not FlyDSL MoE. Gemma4-IT shows 0/10 rep=dec on ragged
and fly alike on the current pin.

Usage:
  python3 -m maxtext.integration.flydsl.compare_reprefill --model gemma \\
      --load_parameters_path results/gemma4_it_ckpt/0/items
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import sys

import numpy as np

from maxtext.integration.flydsl.compare_backends import (
    _MODELS,
    _PROMPTS,
    _build_config,
    _decode,
    _setup_paths,
)



def _last_logits(logits) -> np.ndarray:
    arr = np.asarray(logits)
    if arr.ndim == 3:
        return arr[0, -1].astype(np.float32)
    if arr.ndim == 2:
        return arr[-1].astype(np.float32)
    raise ValueError(f"unexpected logits shape {arr.shape}")


def _run(config, model, seed, seq, fly, prompts):
    # MoE backend is selected by ``config`` (moe_backend=flydsl vs default);
    # ``fly``/``model`` are retained only for call-site compatibility.
    import jax
    import jax.numpy as jnp
    from maxtext.inference.maxengine import maxengine

    ctx = contextlib.nullcontext()

    rows = []
    with ctx:
        engine = maxengine.MaxEngine(config)
        params = engine.load_params(jax.random.PRNGKey(seed))

        tok = engine.build_tokenizer(engine.get_tokenizer())
        is_bos = tok.bos_id is not None

        i32 = jax.ShapeDtypeStruct((), int)
        rng_shape = jax.ShapeDtypeStruct([4], jnp.dtype("uint32"))
        key_shape = jax.ShapeDtypeStruct([seq], jnp.dtype("int32"))
        prefill_exe = (
            jax.jit(
                engine.prefill_aot,
                in_shardings=(engine.param_layouts, None, None, None),
            )
            .lower(params, key_shape, i32, rng_shape)
            .compile(compiler_options=None)
        )

        rng = jax.random.PRNGKey(seed)
        for prompt in prompts:
            tokens, true_length = tok.encode(prompt, is_bos=is_bos, prefill_lengths=[seq])
            true_length = int(true_length)
            rng = jax.random.PRNGKey(seed)

            prefix, _ = engine.prefill(
                params=params,
                padded_tokens=tokens,
                true_length=true_length,
                rng=rng,
            )
            token1 = int(np.asarray(prefix["tokens"]).reshape(-1)[0])

            tokens_ext = np.asarray(tokens).copy()
            tokens_ext[true_length] = token1
            tl2 = true_length + 1
            rep_out, _ = prefill_exe(params, jnp.asarray(tokens_ext), tl2, rng)
            jax.block_until_ready(rep_out)
            rep_logits = _last_logits(rep_out["logits"])
            token2_rep = int(np.argmax(rep_logits))

            slot_state = engine.init_decode_state(rng=jax.random.PRNGKey(seed))
            slot_state = engine.insert(prefix, slot_state, slot=0)
            rng, k = jax.random.split(rng)
            slot_state, res = engine.generate(params, slot_state, rng=k)
            data = np.asarray(res.data)
            token2_dec = int(data.reshape(data.shape[0], -1)[0, 0])

            rows.append({
                "prompt": prompt,
                "token1": token1,
                "token2_rep": token2_rep,
                "token2_dec": token2_dec,
                "rep_dec_top1": int(token2_rep == token2_dec),
            })
            del rep_out, prefix, slot_state

        del prefill_exe, params, engine

    gc.collect()
    try:
        jax.clear_caches()
    except AttributeError:
        pass
    gc.collect()
    return rows, tok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(_MODELS), required=True)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--rng_seed", type=int, default=4242)
    p.add_argument("--backends", default="ragged,fly")
    p.add_argument("--maxtext_dir", default=os.environ.get("MAXTEXT_DIR", "/workspace/maxtext"))
    p.add_argument("--jaxflydsl_dir", default=os.environ.get("JAXFLYDSL_DIR", "/workspace/jax-flydsl"))
    p.add_argument("--load_parameters_path", default=None)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    model_name, tok_name = _MODELS[args.model]
    if args.out_dir is None:
        args.out_dir = f"{args.jaxflydsl_dir}/results/reprefill_{args.model}"
    _setup_paths(args.maxtext_dir, args.jaxflydsl_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    import jax
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")

    wanted = [b.strip() for b in args.backends.split(",") if b.strip()]
    results = {}
    tok = None
    for n, name in enumerate(wanted):
        fly = name != "ragged"
        print(f"[{n + 1}/{len(wanted)}] {name}...")
        cfg = _build_config(
            args.maxtext_dir, model_name, tok_name, args.seq, args.out_dir,
            "True", load_parameters_path=args.load_parameters_path,
            moe_backend="flydsl" if fly else "default",
        )
        rows, t = _run(cfg, args.model, args.rng_seed, args.seq, fly, _PROMPTS)
        results[name] = rows
        if tok is None:
            tok = t

    ref = wanted[0]
    lines = [
        f"# Re-prefill vs decode diagnostic: {model_name}",
        "",
        f"- seq={args.seq}, rng_seed={args.rng_seed}, greedy, bf16, single device",
        f"- weights: {args.load_parameters_path or 'random init'}",
        "- token1 from prefill; token2 from re-prefill(prompt+token1) vs KV-cache generate()",
        "",
        f"## Per-prompt ({ref})",
        "",
        "| # | prompt | token1 | re-prefill token2 | decode token2 | rep=dec? |",
        "|---|--------|--------|-------------------|---------------|----------|",
    ]
    rep_dec_match = 0
    for i, row in enumerate(results[ref]):
        t1 = _decode(tok, row["token1"])
        t2r = _decode(tok, row["token2_rep"])
        t2d = _decode(tok, row["token2_dec"])
        ok = row["rep_dec_top1"]
        rep_dec_match += ok
        lines.append(
            f"| {i} | {row['prompt']!r} | {t1!r} | {t2r!r} | {t2d!r} | "
            f"{'Y' if ok else 'N'} |"
        )
    lines.append("")
    lines.append(f"**{ref}: re-prefill token2 == decode token2 on {rep_dec_match}/{len(_PROMPTS)} prompts.**")
    lines.append("")

    if len(wanted) > 1:
        other = wanted[1]
        lines.append(f"## {ref} vs {other} — path agreement")
        lines.append("")
        lines.append("| # | token1 | rep token2 | dec token2 | rep top1 | dec top1 |")
        lines.append("|---|--------|------------|------------|----------|----------|")
        rep_fly = dec_fly = 0
        for i in range(len(_PROMPTS)):
            a, b = results[ref][i], results[other][i]
            r_ok = int(a["token2_rep"] == b["token2_rep"])
            d_ok = int(a["token2_dec"] == b["token2_dec"])
            rep_fly += r_ok
            dec_fly += d_ok
            lines.append(
                f"| {i} | {'Y' if a['token1'] == b['token1'] else 'N'} | "
                f"{'Y' if r_ok else 'N'} | {'Y' if d_ok else 'N'} | "
                f"{_decode(tok, a['token2_rep'])!r} / {_decode(tok, b['token2_rep'])!r} | "
                f"{_decode(tok, a['token2_dec'])!r} / {_decode(tok, b['token2_dec'])!r} |"
            )
        lines.append("")
        lines.append(
            f"**{other} vs {ref}: re-prefill top-1 {rep_fly}/{len(_PROMPTS)}, "
            f"decode top-1 {dec_fly}/{len(_PROMPTS)}.**"
        )
        lines.append("")

    report = "\n".join(lines)
    path = f"{args.out_dir}/report.md"
    with open(path, "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    print(f"\n[report written to {path}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
