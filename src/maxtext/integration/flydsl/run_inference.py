# SPDX-License-Identifier: Apache-2.0
"""Inference microbenchmark with the FlyDSL one-time weight preshuffle.

This is the thin runner needed for ``moe_backend=flydsl``: the stock
``inference_microbenchmark`` has no seam to fill the preshuffled-weight params
between ``load_params`` and the timed loop, so we replicate its prefill/AR
benchmark and insert :func:`inject_preshuffled_weights` once after load.

The MoE backend is selected entirely by config (``moe_backend=flydsl`` vs
``default``); for ``default`` the inject is a no-op (no fly params), so this
runner also works as a stock baseline.

We avoid ``engine.aot_compile`` and ``prefill_insert_benchmark`` (XLA
layout-solver bug on this jaxlib/ROCm stack): prefill is JIT-compiled directly,
``decode_state`` comes from ``engine.init_decode_state``, and AR uses non-AOT
``engine.generate``.

Usage:
  cd $MAXTEXT_DIR
  python3 -m maxtext.integration.flydsl.run_inference configs/base.yml \\
      model_name=mixtral-8x7b hardware=gpu moe_backend=flydsl \\
      inference_microbenchmark_prefill_lengths="256,512" \\
      inference_microbenchmark_stages=prefill,generate ...
"""

from __future__ import annotations

import json
import sys

import jax
from absl import app

from maxtext.configs import pyconfig
from maxtext.inference.inference_microbenchmark import (
    _FLATTEN_MICROBENCHMARK_RESULTS,
    ar_benchmark,
    collate_results,
    prefill_benchmark,
    print_results_for_analyze,
    summarize_prefill_result,
    write_results,
)
from maxtext.inference.maxengine import maxengine
from maxtext.utils import max_utils

from maxtext.kernels.flydsl_moe import inject_preshuffled_weights  # noqa: E402


def run_benchmarks(config) -> dict:
    engine = maxengine.MaxEngine(config)
    rng, rng_load, rng_init = jax.random.split(jax.random.PRNGKey(1234), 3)
    params = engine.load_params(rng_load)

    # One-time FlyDSL preshuffle fill (no-op unless moe_backend=flydsl).
    if getattr(config, "moe_backend", "default") == "flydsl":
        params = inject_preshuffled_weights(params)

    prefill_lengths = [int(x) for x in config.inference_microbenchmark_prefill_lengths.split(",")]
    stages = config.inference_microbenchmark_stages.split(",")
    iters = int(config.inference_microbenchmark_loop_iters)

    tokenizer_model = engine.build_tokenizer(engine.get_tokenizer())
    is_bos = tokenizer_model.bos_id is not None
    text = config.prompt

    decode_state = engine.init_decode_state(rng=rng_init)
    _, cache_size, _ = max_utils.summarize_pytree_data(decode_state["cache"], name="Cache")
    num_model_params, model_size, _ = max_utils.summarize_pytree_data(params, name="Model")

    results: dict = {}
    i32 = jax.ShapeDtypeStruct((), int)
    rng_shape = jax.ShapeDtypeStruct([4], jax.numpy.dtype("uint32"))

    if "prefill" in stages:
        results["prefill-result-sizes"] = {}
        results["prefill"] = {}
        prefill_tokens, prefill_lens, prefill_exe = {}, {}, {}
        for L in prefill_lengths:
            toks, tlen = tokenizer_model.encode(text, is_bos=is_bos, prefill_lengths=[L])
            prefill_tokens[L], prefill_lens[L] = toks, tlen
            key_shape = jax.ShapeDtypeStruct([L], jax.numpy.dtype("int32"))
            prefill_exe[L] = (
                jax.jit(engine.prefill_aot, in_shardings=(engine.param_layouts, None, None, None))
                .lower(params, key_shape, i32, rng_shape)
                .compile(compiler_options=None)
            )
            results["prefill-result-sizes"][L] = summarize_prefill_result(
                prefill_exe[L], params, toks, tlen
            )
        for L in prefill_lengths:
            results["prefill"][L] = prefill_benchmark(
                config, prefill_exe[L], params, prefill_tokens[L], prefill_lens[L], num_model_params, iters
            )

    if "generate" in stages:
        print("\n[ar_benchmark] Using engine.generate (non-AOT) to bypass layout mismatch")
        decode_state = engine.init_decode_state(rng=rng_init)
        results["autoregressive"], _ = ar_benchmark(
            config, engine.generate, params, decode_state,
            engine.max_concurrent_decodes, cache_size, model_size, iters,
        )

    out = collate_results(config, results, model_size, cache_size, num_model_params)
    out["backend"] = getattr(config, "moe_backend", "default")
    print_results_for_analyze(out)
    write_results(out, config.inference_microbenchmark_log_file_path, _FLATTEN_MICROBENCHMARK_RESULTS)
    return out


def main(argv):
    if len(argv) < 2:
        raise ValueError("Usage: run_inference <config.yml> [k=v ...] (add moe_backend=flydsl for FlyDSL)")
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    config = pyconfig.initialize(argv)
    results = run_benchmarks(config)
    json.dump(results, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    app.run(main)
