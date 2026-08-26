"""M5's measurement: what prefix sharing costs and what it saves.

Two arms of the *same* paged engine on the *same* trace, differing only in
whether the prefix cache is on. That isolation is the point: a paged-versus-dense
comparison mixes in every other difference between the two paths, whereas this
changes one flag.

Two things are reported, and the distinction matters.

**Prefill tokens avoided** is exact, and it is arithmetic rather than a
measurement -- prompt tokens minus tokens actually run. It does not depend on
model scale, kernel quality or what else the machine is doing, so it is the
number to quote when the question is whether sharing works.

**Time to first token** is the number anyone actually cares about, and it is only
meaningful once prefill compute dominates a step. At toy width the run is launch
bound, the avoided compute is a rounding error against fixed per-step overhead,
and the TTFT delta will understate the saving badly. Scale the model with
`--layers` and `--emb-dim` before believing it. This is the same trap the M4.5
throughput comparison fell into.

Warmup is a full discarded pass over the trace rather than a shape sweep. That
compiles exactly the shapes the measured pass will present -- no more, and
crucially no fewer -- and it sidesteps `warmup_paged`, whose enumeration of the
shape space currently trips an aiter failure in this container that has nothing
to do with paging.

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

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxtext.common.common_types import MODEL_MODE_PREFILL
from maxtext.configs import pyconfig
from maxtext.inference.kv_common import CacheNamespace
from maxtext.inference.kv_execution import benchmark
from maxtext.utils import maxtext_utils, model_creation_utils

NAMESPACE = CacheNamespace(model_fingerprint="prefix-bench", tokenizer="synthetic")

BASE_CONFIG = {
    "base_num_query_heads": 8,
    "base_num_kv_heads": 8,
    "head_dim": 128,
    "vocab_size": 32000,
    "per_device_batch_size": 1.0,
    "scan_layers": False,
    "sparse_matmul": False,
    "dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    "decode_sampling_strategy": "greedy",
    "enable_checkpointing": False,
    "skip_jax_distributed_system": True,
    "pure_nnx": True,
    "attention": "gpu_paged",
}


def build_config(args, prefix_cache: bool):
  overrides = dict(BASE_CONFIG)
  overrides["base_num_decoder_layers"] = args.layers
  overrides["base_emb_dim"] = args.emb_dim
  overrides["base_mlp_dim"] = args.emb_dim * 4
  overrides["max_target_length"] = args.max_context
  overrides["max_prefill_predict_length"] = args.max_prompt
  overrides["paged_page_size"] = args.page_size
  overrides["paged_num_blocks"] = args.pool_tokens // args.page_size
  overrides["paged_max_context_len"] = args.max_context
  overrides["paged_enable_prefix_cache"] = prefix_cache
  overrides["run_name"] = f"prefix_bench_{'on' if prefix_cache else 'off'}"
  return pyconfig.initialize([sys.argv[0], args.config_path], **overrides)


def build_params(cfg, devices):
  mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(config=cfg, devices=devices), cfg.mesh_axes)
  with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
    model = model_creation_utils.create_model(
        cfg, mesh, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(params=0, dropout=0)
    )
  _, params_state, _ = nnx.split(model, nnx.Param, ...)
  return params_state


def serve(engine, params, requests, *, max_batch: int, measure: bool):
  """One pass over the trace. Returns per-request TTFT and prefill token counts.

  Requests are admitted greedily and every live request is advanced one token per
  step, which is the same policy the driver and the A/B harness use.
  """
  runtime = engine.paged_runtime
  waiting, live = list(requests), []
  ttfts, prefill_tokens = [], []

  while waiting or live:
    while waiting and len(live) < max_batch:
      candidate = waiting[0]
      tokens = candidate.token_ids
      started = time.perf_counter()
      handle, first = engine.prefill_paged(
          params=params,
          padded_tokens=jnp.asarray(tokens, jnp.int32),
          true_length=int(tokens.size),
          request_id=candidate.request_id,
          max_new_tokens=candidate.max_new_tokens,
          prompt_token_ids=tokens,
          namespace=NAMESPACE,
      )
      if handle is None:
        break
      token = int(first.data[0, 0])  # blocks, so the timing is of real work
      if measure:
        ttfts.append(time.perf_counter() - started)
        prefill_tokens.append(int(tokens.size) - runtime.cached_tokens(handle))
      waiting.pop(0)
      candidate.handle = handle
      candidate.generated.append(token)
      live.append(candidate)

    if not live:
      raise RuntimeError("nothing is live and nothing could be admitted; the pool is too small")

    result, ok = engine.generate_paged(
        params, [r.handle for r in live], next_tokens=jnp.asarray([r.generated[-1] for r in live], jnp.int32)
    )
    if not ok:
      victim = live.pop()
      engine.release(victim.handle)
      victim.handle, victim.generated = None, []
      waiting.insert(0, victim)
      continue

    tokens_out = np.asarray(result.data[:, 0]).reshape(-1)
    for row, request in enumerate(live):
      request.generated.append(int(tokens_out[row]))
    for request in [r for r in live if len(r.generated) >= r.max_new_tokens]:
      engine.release(request.handle, request.context_tokens())
      request.handle = None
      live.remove(request)

  return ttfts, prefill_tokens


def run_arm(args, devices, prefix_cache: bool):
  # pylint: disable=import-outside-toplevel
  from maxtext.inference.maxengine import maxengine

  cfg = build_config(args, prefix_cache)
  engine = maxengine.MaxEngine(cfg, devices)
  params = engine.load_params(params=build_params(cfg, devices))
  engine.init_paged_runtime(max_requests=args.max_batch, max_batched_tokens=args.max_prompt)

  def trace():
    return benchmark.shared_prefix_trace(
        args.requests,
        args.shared_prefix,
        args.unique,
        args.output,
        seed=args.seed,
        num_variants=args.variants,
    )

  # Discarded, and it compiles every shape the measured pass will present.
  serve(engine, params, trace(), max_batch=args.max_batch, measure=False)
  # A cold cache for the measured pass, so the reported saving comes from
  # requests sharing with each other rather than with the warmup.
  plane = engine.paged_runtime.control_plane
  plane.evict_cached(plane.prefix_index.num_cached_pages)

  start = time.perf_counter()
  ttfts, prefill_tokens = serve(engine, params, trace(), max_batch=args.max_batch, measure=True)
  duration = time.perf_counter() - start

  prompted = sum(int(r.token_ids.size) for r in trace())
  return {
      "prefix_cache": prefix_cache,
      "duration_s": duration,
      "ttft_p50_ms": statistics.median(ttfts) * 1e3,
      "ttft_mean_ms": statistics.fmean(ttfts) * 1e3,
      "prompt_tokens": prompted,
      "prefill_tokens_run": sum(prefill_tokens),
      "prefill_tokens_saved": prompted - sum(prefill_tokens),
      "prefill_saving_fraction": (prompted - sum(prefill_tokens)) / prompted,
      "page_hit_rate": plane.prefix_index.hit_rate,
      "pages_retained": plane.prefix_index.num_cached_pages,
  }


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--requests", type=int, default=24)
  parser.add_argument("--shared-prefix", type=int, default=512, help="tokens of common context per request")
  parser.add_argument("--unique", type=int, default=128, help="mean tokens of per-request context")
  parser.add_argument("--variants", type=int, default=1, help="distinct shared prefixes to spread across")
  parser.add_argument("--output", type=int, default=16)
  parser.add_argument("--max-batch", type=int, default=8)
  parser.add_argument("--max-context", type=int, default=1024)
  parser.add_argument("--max-prompt", type=int, default=1024)
  parser.add_argument("--page-size", type=int, default=16)
  parser.add_argument("--pool-tokens", type=int, default=16384)
  parser.add_argument("--layers", type=int, default=8)
  parser.add_argument("--emb-dim", type=int, default=1024)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--json-out", default=None)
  parser.add_argument("--config-path", default="src/maxtext/configs/base.yml")
  args = parser.parse_args()

  devices = jax.devices()[:1]
  print(f"device: {devices[0].device_kind}   model: {args.layers}L x {args.emb_dim}d")
  print(
      f"trace: {args.requests} requests, {args.shared_prefix} shared + ~{args.unique} unique prompt tokens, "
      f"{args.variants} prefix variant(s), {args.output} output tokens"
  )

  results = {}
  for prefix_cache in (False, True):
    arm = "on" if prefix_cache else "off"
    summary = run_arm(args, devices, prefix_cache)
    results[arm] = summary
    print(f"\n=== prefix cache {arm} ===")
    print(f"  prompt tokens        {summary['prompt_tokens']}")
    print(f"  prefill tokens run   {summary['prefill_tokens_run']}")
    print(f"  prefill saved        {summary['prefill_tokens_saved']} ({summary['prefill_saving_fraction']:.1%})")
    print(f"  TTFT p50             {summary['ttft_p50_ms']:.2f} ms")
    print(f"  wall clock           {summary['duration_s']:.2f} s")

  off, on = results["off"], results["on"]
  print("\n=== prefix sharing, same engine and same trace ===")
  print(
      f"  prefill work    {off['prefill_tokens_run']} -> {on['prefill_tokens_run']} tokens "
      f"({on['prefill_saving_fraction']:.1%} avoided)"
  )
  print(f"  TTFT p50        {off['ttft_p50_ms']:.2f} -> {on['ttft_p50_ms']:.2f} ms")
  print(f"  wall clock      {off['duration_s']:.2f} -> {on['duration_s']:.2f} s")
  print(f"  page hit rate   {on['page_hit_rate']:.1%},  pages retained {on['pages_retained']}")

  if args.json_out:
    with open(args.json_out, "w", encoding="utf-8") as handle:
      json.dump(results, handle, indent=2)
    print(f"\nwrote {args.json_out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
