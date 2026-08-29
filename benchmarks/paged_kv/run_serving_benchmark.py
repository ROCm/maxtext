"""Run the paged-versus-dense serving benchmark on one GPU.

The Section 6.2 A/B: same model, same precision, same request trace, one engine
on the dense two-region cache and one on the page pool. This is the comparison
that proves or refutes the thesis in Section 1.1, and it is deliberately a single
device -- sharding is a separate milestone and would only add a variable.

Usage, from the MaxText root:

    DECOUPLE_GCLOUD=TRUE JA_ROOT_DIR=/path/to/jax-aiter \\
    PYTHONPATH=src:/path/to/jax-aiter \\
    python3 benchmarks/paged_kv/run_serving_benchmark.py --mode both

`--mode paged` needs neither JetStream nor a checkpoint. `--mode dense` needs
JetStream for real, because the dense `_prefill_jit` returns its `ResultTokens`
from inside `jit` and the decoupled stub is not a pytree; the paged path builds
its own outside `jit` and is unaffected.

Copyright 2026 Advanced Micro Devices, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import copy
import sys

import jax
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxtext.common.common_types import MODEL_MODE_PREFILL
from maxtext.configs import pyconfig
from maxtext.inference.kv_execution import benchmark
from maxtext.utils import maxtext_utils, model_creation_utils

# Small enough to run in a couple of minutes, large enough that the KV footprint
# is the binding constraint rather than a rounding error. head_dim 128 with equal
# query and KV head counts keeps gqa_ratio at 1, inside the prebuilt kernel set.
BASE_CONFIG = {
    "base_emb_dim": 512,
    "base_mlp_dim": 1024,
    "base_num_query_heads": 4,
    "base_num_kv_heads": 4,
    "base_num_decoder_layers": 4,
    "head_dim": 128,
    "vocab_size": 256,
    "per_device_batch_size": 1.0,
    "scan_layers": False,
    "sparse_matmul": False,
    # The paged kernels take bfloat16 or float16 only, so both arms use bfloat16.
    # Comparing a bf16 pool against an fp32 dense cache would measure the dtype.
    "dtype": "bfloat16",
    "weight_dtype": "float32",
    "decode_sampling_strategy": "greedy",
    "enable_checkpointing": False,
    "skip_jax_distributed_system": True,
    "pure_nnx": True,
}


def build_config(mode: str, args) -> object:
  """Config for one arm of the A/B.

  The two arms are given *equal KV memory*, which is what makes the comparison
  fair and is the whole point of the exercise. The dense side commits
  `slots x max_target_length`; the paged side is handed the same number of tokens
  as a pool. Any concurrency difference is then attributable to fungibility
  rather than to one arm having been given more memory.
  """
  overrides = dict(BASE_CONFIG)
  overrides["max_target_length"] = args.max_context
  overrides["max_prefill_predict_length"] = args.max_prompt
  # Model scale is the decisive variable for the throughput comparison, not a
  # detail. The paged win runs through batch size amortising the per-step weight
  # read, so it can only appear once weights dominate a step. At toy scale the
  # extra per-step work is all that is left to measure.
  overrides["base_num_decoder_layers"] = args.layers
  overrides["base_emb_dim"] = args.emb_dim
  overrides["base_mlp_dim"] = args.emb_dim * 2
  dense_kv_tokens = args.max_batch * args.max_context

  if mode == "dense":
    overrides["attention"] = "dot_product"
    overrides["per_device_batch_size"] = float(args.max_batch)
  else:
    overrides["attention"] = "gpu_paged"
    overrides["paged_page_size"] = args.page_size
    overrides["paged_num_blocks"] = dense_kv_tokens // args.page_size
    overrides["paged_max_context_len"] = args.max_context
    overrides["per_device_batch_size"] = float(args.max_batch)
    overrides["paged_enable_prefix_cache"] = bool(args.prefix_cache)

  overrides["run_name"] = f"paged_bench_{mode}"
  return pyconfig.initialize([sys.argv[0], args.config_path], **overrides)


def build_params(cfg, devices):
  """Random weights on one device. No checkpoint, and no network."""
  mesh = jax.sharding.Mesh(
      maxtext_utils.create_device_mesh(config=cfg, devices=devices), cfg.mesh_axes
  )
  with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
    model = model_creation_utils.create_model(
        cfg, mesh, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(params=0, dropout=0)
    )
  _, params_state, _ = nnx.split(model, nnx.Param, ...)
  return params_state


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("paged", "dense", "both"), default="paged")
  parser.add_argument(
      "--shared-prefix",
      type=int,
      default=0,
      help=(
          "tokens of common leading context per request; 0 uses the independent-prompt trace. "
          "Non-zero is the workload prefix sharing targets, so pair it with --prefix-cache"
      ),
  )
  parser.add_argument(
      "--prefix-variants",
      type=int,
      default=1,
      help="distinct shared prefixes to spread requests across, as a deployment serving several system prompts",
  )
  parser.add_argument(
      "--prefix-cache",
      action="store_true",
      help=(
          "share already-computed pages between requests with a common prefix (paged arm only). "
          "Expect little or no saving here: this harness admits the whole trace at once, and a "
          "request can only reuse pages another request has already finished with. Use "
          "run_prefix_cache_benchmark.py to measure sharing"
      ),
  )
  parser.add_argument("--requests", type=int, default=24)
  parser.add_argument("--mean-prompt", type=int, default=48)
  parser.add_argument("--mean-output", type=int, default=32)
  parser.add_argument("--max-batch", type=int, default=8, help="dense slot count, and the KV budget both arms get")
  parser.add_argument(
      "--paged-max-batch",
      type=int,
      default=0,
      help="paged concurrency cap; 0 derives one high enough that the pool binds instead",
  )
  parser.add_argument("--max-context", type=int, default=256)
  parser.add_argument("--max-prompt", type=int, default=128)
  parser.add_argument("--page-size", type=int, default=16)
  parser.add_argument("--layers", type=int, default=4, help="decoder layers; scale this to find the crossover")
  parser.add_argument("--emb-dim", type=int, default=512, help="model width; mlp is twice this")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
      "--repeats",
      type=int,
      default=2,
      help="passes over the trace; the last is reported and the spread is the reportability check",
  )
  parser.add_argument("--json-out", default=None)
  parser.add_argument(
      "--config-path",
      default="src/maxtext/configs/base.yml",
      help="base.yml to layer overrides onto",
  )
  args = parser.parse_args()

  # pylint: disable=import-outside-toplevel
  from maxtext.inference.maxengine import maxengine

  devices = jax.devices()[:1]
  print(f"device: {devices[0].device_kind}  ({len(jax.devices())} visible, using 1)")

  results = {}
  arms = ("paged", "dense") if args.mode == "both" else (args.mode,)
  for arm in arms:
    cfg = build_config(arm, args)
    params_state = build_params(cfg, devices)
    engine = maxengine.MaxEngine(cfg, devices)
    params = engine.load_params(params=params_state)

    # A factory rather than one list: each repeat needs requests with no
    # timestamps on them, and each arm needs an independent copy.
    def fresh_trace():
      if args.shared_prefix:
        # The unique tail is what is left of the mean prompt once the shared
        # block is accounted for, so both traces present the same total prompt
        # length and the comparison is about sharing rather than about size.
        unique = max(args.mean_prompt - args.shared_prefix, args.page_size)
        trace = benchmark.shared_prefix_trace(
            args.requests,
            args.shared_prefix,
            unique,
            args.mean_output,
            seed=args.seed,
            num_variants=args.prefix_variants,
        )
      else:
        trace = benchmark.synthetic_trace(
            args.requests, args.mean_prompt, args.mean_output, seed=args.seed
        )
      return copy.deepcopy(trace)

    if arm == "paged":
      # The batch cap must not bind, or the measurement reports the flag rather
      # than the pool. The dense arm's concurrency is structurally its slot
      # count; the paged arm's should be whatever the identical KV budget can
      # actually hold, which is the entire claim being tested.
      # Capped, and the cap is not cosmetic. The derived value grows with the KV
      # budget, and a batch bucket above 32 combined with the upper
      # sequence-length rungs falls outside the AOT-prebuilt pa_ragged set, so
      # every such shape triggers an aiter JIT build. An unbounded derivation
      # took over fifty minutes in warmup and produced nothing. Raise it
      # deliberately with --paged-max-batch once the prebuild covers the shapes.
      derived = max(args.max_batch, (args.max_batch * args.max_context) // max(args.mean_prompt, 1))
      paged_batch = args.paged_max_batch or min(derived, 32)
      if not args.paged_max_batch and derived > paged_batch:
        print(
            f"note: capping paged concurrency at {paged_batch} (pool would allow ~{derived}); "
            f"larger batch buckets need an aiter prebuild, so pass --paged-max-batch to override"
        )
      engine.init_paged_runtime(max_requests=paged_batch, max_batched_tokens=args.max_prompt)
      warmed = benchmark.warmup_paged(
          engine,
          params,
          max_prompt=args.max_prompt,
          max_batch=paged_batch,
          # The longest context the trace can reach. Warming past it would
          # compile shapes the run never presents; stopping short leaves
          # compilation inside the measured window.
          target_context=min(
              args.max_context, int(args.mean_prompt * 1.6) + int(args.mean_output * 1.6) + 2
          ),
      )
      summary = benchmark.run_repeated(
          engine,
          params,
          fresh_trace,
          max_batch=paged_batch,
          warmed_shapes=warmed,
          repeats=args.repeats,
      )
      summary["kv_tokens_committed"] = summary.get("pool_capacity_tokens")
      summary["batch_cap"] = paged_batch
    else:
      state = engine.init_decode_state()
      benchmark.warmup_dense(engine, params, state, prompt_len=min(8, args.max_prompt))
      durations = []
      for _ in range(max(args.repeats, 1)):
        summary = benchmark.run_dense(engine, params, fresh_trace(), max_batch=args.max_batch)
        durations.append(summary["duration_s"])
      summary["repeat_durations_s"] = durations
      summary["stability_ratio"] = benchmark.stability_ratio(durations)
      summary["latency_is_reportable"] = benchmark.is_stable(durations)
      summary["kv_tokens_committed"] = args.max_batch * args.max_context
      summary["batch_cap"] = args.max_batch

    summary["config"] = {
        "attention": cfg.attention,
        "max_context": args.max_context,
        "max_batch": args.max_batch,
        "requests": args.requests,
        "mean_prompt": args.mean_prompt,
        "mean_output": args.mean_output,
    }
    results[arm] = summary
    print(benchmark.report(arm, summary))
    print(f"  KV tokens committed: {summary['kv_tokens_committed']}")
    if summary.get("prefix_cache_enabled"):
      print(
          f"  prefix cache: {summary['prefill_tokens_saved']} of {summary['prompt_tokens']} prompt tokens "
          f"not recomputed ({summary['prefill_saving_fraction']:.1%}), "
          f"page hit rate {summary['prefix_cache_page_hit_rate']:.1%}"
      )

  if "paged" in results and "dense" in results:
    paged, dense = results["paged"], results["dense"]
    print("\n=== paged vs dense, equal KV memory ===")
    ratio = paged["output_throughput_tok_per_s"] / max(dense["output_throughput_tok_per_s"], 1e-9)
    print(f"  output throughput  x{ratio:.2f}")
    print(
        f"  TTFT p50  {dense['ttft']['p50_ms']:.2f} -> {paged['ttft']['p50_ms']:.2f} ms"
        f"   ITL p50  {dense['itl']['p50_ms']:.2f} -> {paged['itl']['p50_ms']:.2f} ms"
    )
    occ = paged.get("occupancy", {})
    if occ:
      print(
          f"  concurrency: dense fixed at {dense['config']['max_batch']} slots,"
          f" paged reached {occ['max_concurrency']} on the same KV budget"
      )

  if args.json_out:
    benchmark.write_json(args.json_out, results)
    print(f"\nwrote {args.json_out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
