"""Greedy autoregressive DECODE comparison: fly vs ragged, real weights.

Prefill is the many-token regime; decode runs the MoE kernel one token at a
time (tokens = batch), a different tile_m / kernel path. This validates that
fly's decode-path MoE agrees with MaxText's exact backend.

Each backend builds an engine at the same seed, prefills each prompt, then
greedily generates ``--gen_steps`` tokens via the real KV-cache decode path
(engine.prefill -> insert -> generate loop, decode_sampling_strategy=greedy).
We compare the generated token sequences (leading-match length + exact match).

With trained weights and greedy decoding, fly and ragged should produce the
same text; a divergence point is a bf16 near-tie (and is reported).

Usage:
  python3 -m maxtext.integration.flydsl.compare_decode --model gemma   --backends ragged,fly \
      --load_parameters_path <ckpt>/0/items --gen_steps 16
  python3 -m maxtext.integration.flydsl.compare_decode --model mixtral --backends ragged,fly \
      --load_parameters_path <ckpt>/0/items
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import sys

import numpy as np

from maxtext.integration.flydsl.compare_backends import _MODELS, _setup_paths, _build_config, _PROMPTS


def _gen(config, seed, seq, prompts, gen_steps):
    """Build engine, greedily generate gen_steps tokens per prompt.

    MoE backend is selected by ``config`` (``moe_backend=flydsl`` vs ``default``).
    Returns (list[list[int]] generated token ids per prompt, tokenizer).
    """
    import jax
    from maxtext.inference.maxengine import maxengine

    out = []
    engine = maxengine.MaxEngine(config)
    params = engine.load_params(jax.random.PRNGKey(seed))

    tok = engine.build_tokenizer(engine.get_tokenizer())
    is_bos = tok.bos_id is not None
    decode_state = engine.init_decode_state(rng=jax.random.PRNGKey(seed))

    for prompt in prompts:
        tokens, true_length = tok.encode(prompt, is_bos=is_bos, prefill_lengths=[seq])
        rng = jax.random.PRNGKey(seed)
        prefix, _ = engine.prefill(
            params=params, padded_tokens=tokens, true_length=true_length, rng=rng
        )
        gen = [int(np.asarray(prefix["tokens"]).reshape(-1)[0])]
        decode_state = engine.insert(prefix, decode_state, slot=0)
        for _ in range(gen_steps - 1):
            rng, k = jax.random.split(rng)
            decode_state, res = engine.generate(params, decode_state, rng=k)
            data = np.asarray(res.data)
            gen.append(int(data.reshape(data.shape[0], -1)[0, 0]))
        out.append(gen)
    del params, engine

    gc.collect()
    try:
        jax.clear_caches()
    except AttributeError:
        pass
    gc.collect()
    return out, tok


def _leading_match(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(_MODELS), required=True)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--gen_steps", type=int, default=16)
    p.add_argument("--rng_seed", type=int, default=4242)
    p.add_argument("--backends", default="ragged,fly", help="subset of: ragged,fly,fly_reduce")
    p.add_argument("--maxtext_dir", default=os.environ.get("MAXTEXT_DIR", "/workspace/maxtext"))
    p.add_argument("--jaxflydsl_dir", default=os.environ.get("JAXFLYDSL_DIR", "/workspace/jax-flydsl"))
    p.add_argument("--load_parameters_path", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--scan_layers", action="store_true",
                   help="Use MaxText scan_layers=true (default false; Gemma4 scanned blocks)")
    args = p.parse_args()

    model_name, tok_name = _MODELS[args.model]
    if args.out_dir is None:
        args.out_dir = f"{args.jaxflydsl_dir}/results/decode_{args.model}"
    _setup_paths(args.maxtext_dir, args.jaxflydsl_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    import jax
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")

    wanted = [b.strip() for b in args.backends.split(",") if b.strip()]
    # (sparse_matmul, moe_backend, FLY_STAGE2 override)
    _SPEC = {"ragged": ("True", "default", None), "fly": ("True", "flydsl", "atomic"),
             "fly_reduce": ("True", "flydsl", "reduce")}

    R, tok = {}, None
    for n, name in enumerate(wanted):
        sparse, moe_backend, stage2 = _SPEC[name]
        if stage2 is not None:
            os.environ["FLY_STAGE2"] = stage2
        else:
            os.environ.pop("FLY_STAGE2", None)
        print(f"[{n+1}/{len(wanted)}] {name} (decode, gen_steps={args.gen_steps})...")
        cfg = _build_config(args.maxtext_dir, model_name, tok_name, args.seq, args.out_dir,
                            sparse, load_parameters_path=args.load_parameters_path,
                            scan_layers=args.scan_layers, moe_backend=moe_backend)
        seqs, t = _gen(cfg, args.rng_seed, args.seq, _PROMPTS, args.gen_steps)
        R[name] = seqs
        if tok is None:
            tok = t

    ref = wanted[0]
    others = wanted[1:]
    lines = [f"# Decode comparison report: {model_name}", "",
             f"- seq={args.seq}, gen_steps={args.gen_steps}, scan_layers={args.scan_layers}, "
             f"greedy, dtype=bfloat16, single device",
             f"- backends: {', '.join(wanted)}",
             f"- weights: {args.load_parameters_path or 'random init (no checkpoint)'}",
             "- Greedy autoregressive decode via the real KV-cache path; compares generated token sequences.",
             ""]
    for other in others:
        lines.append(f"## {ref} vs {other}")
        lines.append("")
        lines.append("| # | prompt | leading match | exact? | " + f"{ref} text | {other} text |")
        lines.append("|---|--------|---------------|--------|------|------|")
        exact = 0
        for i, prompt in enumerate(_PROMPTS):
            a, b = R[ref][i], R[other][i]
            lm = _leading_match(a, b)
            ok = (a == b)
            exact += int(ok)
            ta = _decode_seq(tok, a)
            tb = _decode_seq(tok, b)
            lines.append(f"| {i} | {prompt!r} | {lm}/{len(a)} | {'Y' if ok else 'N'} | {ta!r} | {tb!r} |")
        lines.append("")
        lines.append(f"**{other}: {exact}/{len(_PROMPTS)} sequences exactly match {ref}.**")
        lines.append("")

    report = "\n".join(lines)
    path = f"{args.out_dir}/report.md"
    with open(path, "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    print(f"\n[report written to {path}]")
    return 0


def _decode_seq(tok, ids):
    for meth in ("decode", "detokenize"):
        fn = getattr(tok, meth, None)
        if fn is not None:
            try:
                s = fn([int(x) for x in ids])
                return (s if isinstance(s, str) else str(s)).replace("\n", "\\n")
            except Exception:
                pass
    return " ".join(str(int(x)) for x in ids)


if __name__ == "__main__":
    raise SystemExit(main())
