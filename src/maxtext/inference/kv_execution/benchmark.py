"""Serving benchmark for the paged KV runtime, and for the dense path beside it.

Section 6.1 of the design notes that the MaxText native path has no serving
harness and that this is the one piece that has to be written rather than reused.
This is it. The metric names are chosen to match what `vllm/benchmarks/
benchmark_serving.py` and `sgl_jax/bench_serving.py` report, so results from the
three land in one table without translation.

**Half of what this measures is not performance.** Alongside TTFT and ITL it
records pool occupancy against live tokens, concurrency sustained at a fixed pool
size, and peak device memory across the run. Those are capacity and correctness
properties, and they fail silently: a leaked page, a declined donation or an
unbounded compile count all produce plausible tokens and merely worse numbers, so
a harness reporting only latency would pass with any of them present.

**One methodological warning, learned the hard way here.** Counting compiled
shapes is not enough to know whether a latency figure is real. An earlier version
of this file reported zero unwarmed shapes for a run that was three-quarters
compilation, because padding a variable-length array with `jnp` compiles once per
length and no shape *bucket* changed. Build per-call arrays in numpy, and trust
`run_repeated`: passes that agree with each other cannot both contain a one-off
cost, whereas a compilation counter only ever catches the cases someone
remembered to count.

**On the memory numbers specifically, to head off a misreading.** The pool is one
fixed allocation made at startup, so paged peak memory is *flat* — it does not
track load, and a run where it grew would indicate a leak rather than a success.
What paging buys is that every byte of that allocation is fungible between
requests, which shows up as concurrency at a fixed budget rather than as a
smaller footprint. `occupancy_*` is the series that tracks live tokens.

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

import dataclasses
import json
import time
from typing import Any, Sequence

import numpy as np

import jax
import jax.numpy as jnp

from maxtext.inference.kv_execution.driver import PagedDriver, PagedRequest


@dataclasses.dataclass
class RequestMetrics:
  """Timing for one request, held beside it rather than on it.

  There used to be a `Request` here that was `PagedRequest` plus timestamps, and
  the two drifted: the harness re-implemented scheduling to collect these, and its
  copy of preemption discarded generated tokens where the driver's keeps them,
  which livelocked a 70B sweep. The driver is now the only scheduler, so the
  harness owns *measurement* and nothing else.

  Keyed by `request_id` rather than by object identity, because `run_repeated`
  deep-copies a trace between passes.
  """

  arrival: float = 0.0
  first_token_at: float | None = None
  token_times: list[float] = dataclasses.field(default_factory=list)
  # Tokens this request actually computed, summed over its prefill steps. Not
  # derivable afterwards: `cached_tokens` is reset when the request is released,
  # and a preempted request prefills more than once.
  prefill_tokens: int = 0

  @property
  def ttft(self) -> float | None:
    """Arrival to first token. None while the request has not produced one."""
    return None if self.first_token_at is None else self.first_token_at - self.arrival

  def inter_token_latencies(self) -> list[float]:
    """Gaps between successive tokens *after* the first.

    The first token's cost is TTFT and belongs to prefill; folding it in here
    would flatter ITL on short outputs and inflate it on long ones.
    """
    return [b - a for a, b in zip(self.token_times, self.token_times[1:])]


def _metrics_for(table: dict[str, RequestMetrics], request: PagedRequest) -> RequestMetrics:
  """The metrics row for `request`, created on first sight."""
  return table.setdefault(request.request_id, RequestMetrics())


def synthetic_trace(
    num_requests: int,
    mean_prompt: int,
    mean_output: int,
    *,
    seed: int = 0,
    length_spread: float = 0.6,
) -> list[PagedRequest]:
  """A batch of requests with varied lengths, all arriving at once.

  Lengths are what this plan's claims are about, so they vary; arrival times are
  not, so every request is available from the start. That measures saturated
  throughput rather than a rate-driven trace, which is the right first
  measurement: a rate low enough to keep the pool empty would hide exactly the
  capacity behaviour under test.
  """
  rng = np.random.default_rng(seed)

  def spread(mean: int) -> np.ndarray:
    low = max(1, int(mean * (1.0 - length_spread)))
    high = max(low + 1, int(mean * (1.0 + length_spread)))
    return rng.integers(low, high, size=num_requests)

  prompts, outputs = spread(mean_prompt), spread(mean_output)
  # Token ids are distinct per request on purpose. A constant filler would make
  # every prompt a prefix of every other, so switching the prefix cache on would
  # report a hit rate this workload does not represent -- the baseline has to be a
  # workload with nothing to share.
  return [
      PagedRequest(
          request_id=f"r{i}",
          prompt_len=int(prompts[i]),
          max_new_tokens=int(outputs[i]),
          prompt_tokens=rng.integers(1, 30000, size=int(prompts[i]), dtype=np.int64),
      )
      for i in range(num_requests)
  ]


def shared_prefix_trace(
    num_requests: int,
    shared_prefix: int,
    mean_unique: int,
    mean_output: int,
    *,
    seed: int = 0,
    length_spread: float = 0.6,
    num_variants: int = 1,
) -> list[PagedRequest]:
  """Requests that begin with a common block, as a served system prompt does.

  This is the workload prefix sharing exists for, and the one where a paged
  runtime's advantage over a dense one is largest: the shared block is prefilled
  once and read by everything after it. `num_variants` splits the population
  across several distinct prefixes, which is the realistic case -- a deployment
  serves a handful of system prompts, not one -- and it also keeps the measurement
  honest about eviction, since several prefixes compete for the same pool.
  """
  rng = np.random.default_rng(seed)
  prefixes = [rng.integers(1, 30000, size=shared_prefix, dtype=np.int64) for _ in range(max(num_variants, 1))]

  def spread(mean: int) -> np.ndarray:
    low = max(1, int(mean * (1.0 - length_spread)))
    high = max(low + 1, int(mean * (1.0 + length_spread)))
    return rng.integers(low, high, size=num_requests)

  uniques, outputs = spread(mean_unique), spread(mean_output)
  requests = []
  for i in range(num_requests):
    tail = rng.integers(1, 30000, size=int(uniques[i]), dtype=np.int64)
    tokens = np.concatenate([prefixes[i % len(prefixes)], tail])
    requests.append(
        PagedRequest(
            request_id=f"r{i}",
            prompt_len=int(tokens.size),
            max_new_tokens=int(outputs[i]),
            prompt_tokens=tokens,
        )
    )
  return requests


def warmup_paged(engine, params, *, max_prompt: int, max_batch: int, target_context: int) -> set:
  """Compile every shape the measured run can present, then discard the results.

  Not optional, and not a detail. XLA compiles per shape, so an uncompiled bucket
  pays seconds on its first use and those seconds land in the TTFT and ITL
  percentiles. The first measured run here reported a p50 TTFT of 25 seconds and
  an ITL p99 of 10 seconds, all of it compilation. Bucketing is what makes
  pre-compilation possible at all -- an unbucketed implementation has no finite
  shape set to enumerate.

  **Three ladders, not two, and the third is the one that gets missed.** A shape
  is `(batch bucket, token bucket, sequence-length bucket)`. Warming the first
  two still left multi-second outliers, because the sequence-length rung is a
  *static kernel argument*: a request whose context grows past a rung presents a
  new program even at an already-compiled batch width. So this sweeps context
  length as well, by admitting a full batch and decoding it forward to
  `target_context`, touching every batch width at every rung along the way.

  Returns the set of shapes compiled, so a caller can diff it against what the
  measured run actually used and see whether the sweep was complete.
  """
  runtime = engine.paged_runtime
  planner = runtime.planner
  batch_rungs = [b for b in planner.batch_rungs if b <= max_batch] or [1]

  # Phase 1: every prefill shape. One prompt per sequence-length rung a prompt
  # can reach, since a fresh request's context is exactly its prompt length.
  for rung in planner.seqlen_rungs:
    prompt_len = min(rung, max_prompt)
    handle, _ = engine.prefill_paged(
        params=params,
        padded_tokens=jnp.ones((prompt_len,), jnp.int32),
        true_length=prompt_len,
        request_id=f"warmup-prefill-{rung}",
        max_new_tokens=1,
    )
    if handle is not None:
      engine.release(handle)
    runtime.control_plane.allocator.merge_released()

  # Phase 2: every decode shape. Admit the widest batch the pool allows, then
  # walk it forward so each batch width is exercised at each length rung.
  handles, tokens = [], []
  for index in range(max_batch):
    handle, result = engine.prefill_paged(
        params=params,
        padded_tokens=jnp.ones((1,), jnp.int32),
        true_length=1,
        request_id=f"warmup-decode-{index}",
        max_new_tokens=target_context,
    )
    if handle is None:
      break
    handles.append(handle)
    tokens.append(int(result.data[0, 0]))

  page_map = runtime.control_plane.page_map
  while handles and page_map.seq_len(handles[0]) < target_context:
    for width in batch_rungs:
      if width > len(handles):
        continue
      result, ok = engine.generate_paged(
          params, handles[:width], next_tokens=jnp.asarray(tokens[:width], jnp.int32)
      )
      if not ok:
        handles, tokens = handles[:-1], tokens[:-1]
        break
      fresh = np.asarray(result.data[:, 0]).reshape(-1)
      for row in range(width):
        tokens[row] = int(fresh[row])

  for handle in handles:
    engine.release(handle)
  runtime.control_plane.allocator.merge_released()
  return set(runtime.observed_shapes)


def warmup_dense(engine, params, decode_state, *, prompt_len: int):
  """Compile the dense prefill and generate shapes. Returns the decode state."""
  padded = jnp.asarray(
      [1] * prompt_len + [0] * (engine.config.max_prefill_predict_length - prompt_len), jnp.int32
  )
  prefix, _ = engine.prefill(params=params, padded_tokens=padded, true_length=prompt_len)
  decode_state = engine.insert(prefix, decode_state, slot=0)
  decode_state, _ = engine.generate(params, decode_state)
  return decode_state


def _peak_bytes() -> int | None:
  """Peak device bytes in use, or None where the backend does not report it."""
  try:
    stats = jax.local_devices()[0].memory_stats() or {}
  except Exception:  # pylint: disable=broad-exception-caught
    return None
  return stats.get("peak_bytes_in_use")


def _percentiles(values: Sequence[float]) -> dict[str, float]:
  """p50 and p99 in milliseconds, reported together because the tail is the point.

  Paging and prefix sharing show up in tail behaviour; a mean can hide a stall
  entirely, which is why Section 6 asks for both.
  """
  if not values:
    return {"p50_ms": float("nan"), "p99_ms": float("nan"), "mean_ms": float("nan")}
  arr = np.asarray(values, dtype=np.float64) * 1e3
  return {
      "p50_ms": float(np.percentile(arr, 50)),
      "p99_ms": float(np.percentile(arr, 99)),
      "mean_ms": float(arr.mean()),
  }


def run_paged(
    engine,
    params,
    requests: Sequence[Request],
    *,
    max_batch: int,
    warmed_shapes: set | None = None,
) -> dict[str, Any]:
  """Serve `requests` on the paged path, measuring what `PagedDriver` does.

  **The harness no longer schedules.** It used to run its own admission, its own
  preemption and its own step order, which is how it drifted from the driver:
  its preemption discarded a victim's generated tokens where the driver's keeps
  them, so an overcommitted pool replayed the same failed attempt forever. That
  livelocked a 70B capacity sweep at full GPU utilisation with zero progress.

  Now the driver owns policy and this owns measurement. The loop is
  timestamp-around-`driver.step()`, attributing per-request times through
  `StepOutcome.batch`, which is exactly the information the harness previously
  re-implemented a scheduler to obtain.
  """
  runtime = engine.paged_runtime
  allocator = runtime.control_plane.allocator
  page_size = runtime.control_plane.layout.tokens_per_page

  # The engine's runtime is passed in rather than letting the driver build one, so
  # the pool, the shape bookkeeping and the prefix-cache accounting stay
  # single-copy -- `runtime.observed_shapes` below is read back from it.
  driver = PagedDriver(
      runtime.control_plane,
      runtime.pool,
      engine.paged_step_fn(params),
      max_batch=max_batch,
      runtime=runtime,
  )
  metrics: dict[str, RequestMetrics] = {}
  occupancy: list[tuple[float, int, int, int]] = []
  steps = 0
  before = set(runtime.observed_shapes)

  start = time.perf_counter()
  for request in requests:
    _metrics_for(metrics, request).arrival = start
  driver.submit(list(requests))

  while True:
    outcome = driver.step()
    if outcome is None:
      break
    steps += 1
    now = time.perf_counter()

    for request, query_len in zip(outcome.batch, outcome.query_lens):
      entry = _metrics_for(metrics, request)
      if not outcome.is_decode:
        # Tokens this prefill actually ran, which is the prompt minus whatever the
        # cache supplied. Accumulated rather than assigned, because a preempted
        # request prefills more than once and every one of those is real work.
        entry.prefill_tokens += query_len
      if entry.first_token_at is None:
        entry.first_token_at = now
      entry.token_times.append(now)

    live = driver.live_requests()
    live_tokens = sum(r.prompt_len + len(r.generated) for r in live)
    occupancy.append((now - start, allocator.num_allocated_pages * page_size, live_tokens, len(live)))

  elapsed = time.perf_counter() - start
  done = driver.completed()
  index = runtime.control_plane.prefix_index
  summary = _summarise(
      done, metrics, occupancy, elapsed, steps, allocator, page_size, retained_pages=index.num_cached_pages
  )
  summary.update(_summarise_prefix_cache(done, metrics, runtime.control_plane))

  # Shape accounting, which is necessary but *not sufficient*: it sees bucketed
  # `StepShape`s and is blind to eager host-side ops whose values enter a jaxpr as
  # literals. An earlier version of this harness reported zero unwarmed shapes
  # for a run that was three-quarters compilation, because padding a
  # variable-length array with `jnp` compiles per length and no shape bucket
  # changed. The empirical check in `run_repeated` is what actually settles it.
  used = set(runtime.observed_shapes)
  fresh = used - before if warmed_shapes is None else used - set(warmed_shapes)
  summary["distinct_shapes_total"] = len(used)
  summary["shapes_compiled_during_measurement"] = len(fresh)
  summary["all_shapes_prewarmed"] = not fresh
  if fresh:
    summary["unwarmed_shapes"] = [dataclasses.asdict(s) for s in sorted(fresh, key=str)]
  return summary


def _summarise_prefix_cache(requests, metrics, control_plane) -> dict[str, Any]:
  """What sharing avoided, in tokens rather than in seconds.

  Reported separately from latency because it is the direct measurement: a
  latency difference between two runs mixes in the pool pressure the retained
  pages cause, whereas prompt tokens minus prefilled tokens is exactly the work
  that did not happen.
  """
  prompted = sum(r.prompt_len for r in requests)
  prefilled = sum(metrics[r.request_id].prefill_tokens for r in requests if r.request_id in metrics)
  index = control_plane.prefix_index
  return {
      "prefix_cache_enabled": index.enabled,
      "prompt_tokens": prompted,
      "prefill_tokens_run": prefilled,
      "prefill_tokens_saved": prompted - prefilled,
      "prefill_saving_fraction": (prompted - prefilled) / prompted if prompted else 0.0,
      "prefix_cache_page_hit_rate": index.hit_rate,
      "prefix_cache_pages_retained": index.num_cached_pages,
  }


def run_repeated(engine, params, trace_factory, *, max_batch: int, warmed_shapes=None, repeats: int = 2):
  """Run the same trace several times and report the last, with the spread.

  This is the reportability check that cannot be fooled. Counting compilations
  requires knowing every place one can happen, and the previous attempt at that
  missed a whole class; running the identical workload twice does not need to know
  anything -- if the first pass was paying for compilation, the second is
  materially faster, and the ratio says so. A run is reportable when successive
  passes agree.
  """
  plane = engine.paged_runtime.control_plane
  passes = []
  for _ in range(max(repeats, 1)):
    # Each pass starts with a cold prefix cache. Otherwise a later pass finds the
    # previous pass's pages waiting for it, which both inflates the reported
    # saving past what this trace's own requests share with each other and makes
    # the passes incomparable -- defeating the stability check, which is the
    # whole reason for repeating.
    plane.evict_cached(plane.prefix_index.num_cached_pages)
    summary = run_paged(
        engine, params, trace_factory(), max_batch=max_batch, warmed_shapes=warmed_shapes
    )
    passes.append(summary)

  final = passes[-1]
  durations = [p["duration_s"] for p in passes]
  final["repeat_durations_s"] = durations
  final["latency_is_reportable"] = is_stable(durations) and final["all_shapes_prewarmed"]
  final["stability_ratio"] = stability_ratio(durations)
  return final


def stability_ratio(durations: Sequence[float]) -> float:
  """Spread across the passes *after* the first, as max over min.

  The first pass is excluded on purpose. Warmup cannot pre-compile everything a
  trace touches -- the first request through a code path still pays for whatever
  the sweep did not reach -- so comparing the first pass to the last always looks
  alarming and says nothing. What matters is whether the passes that follow agree
  with each other, because two passes that agree cannot both contain a
  one-off cost.
  """
  tail = list(durations[1:]) or list(durations)
  low, high = min(tail), max(tail)
  return high / low if low else float("inf")


def is_stable(durations: Sequence[float], tolerance: float = 1.15) -> bool:
  """True when repeated passes agree closely enough to report a latency figure."""
  if len(durations) < 2:
    return False
  return stability_ratio(durations) < tolerance


def run_dense(engine, params, requests: Sequence[PagedRequest], *, max_batch: int) -> dict[str, Any]:
  """Serve `requests` on the dense two-region cache, for the A/B.

  Deliberately the ordinary MaxText path -- `prefill`, `init_decode_state`,
  `insert`, `generate` -- because the comparison is against what exists today,
  not against an idealised dense implementation. Its batch is a fixed set of
  slots, and every slot advances in lockstep whether or not it holds a request,
  which is the behaviour under examination rather than an inefficiency to correct.

  This keeps its own loop rather than driving `PagedDriver`, and that is not an
  oversight: the driver's whole subject is page allocation, which the dense
  two-region cache does not have. A slot here is fixed for a request's lifetime,
  there is nothing to reserve, recycle or preempt, and forcing it through a paged
  scheduler would measure the scheduler instead of the cache.
  """
  waiting, done = list(requests), []
  metrics: dict[str, RequestMetrics] = {}
  start = time.perf_counter()
  for request in waiting:
    _metrics_for(metrics, request).arrival = start

  decode_state = engine.init_decode_state()
  slots: dict[int, PagedRequest] = {}
  steps = 0

  while waiting or slots:
    while waiting and len(slots) < max_batch:
      free = next(s for s in range(max_batch) if s not in slots)
      candidate = waiting.pop(0)
      padded = jnp.asarray(
          [1] * candidate.prompt_len
          + [0] * (engine.config.max_prefill_predict_length - candidate.prompt_len),
          dtype=jnp.int32,
      )
      prefix, result = engine.prefill(
          params=params, padded_tokens=padded, true_length=candidate.prompt_len
      )
      decode_state = engine.insert(prefix, decode_state, slot=free)
      candidate.generated.append(int(result.data[0, 0]))
      now = time.perf_counter()
      entry = _metrics_for(metrics, candidate)
      entry.first_token_at = now
      entry.token_times.append(now)
      # The dense path has no prefix cache, so every prompt token is computed.
      entry.prefill_tokens += candidate.prompt_len
      slots[free] = candidate
      steps += 1

    if not slots:
      break

    decode_state, result = engine.generate(params, decode_state)
    steps += 1
    now = time.perf_counter()
    for slot, request in list(slots.items()):
      request.generated.append(int(result.data[slot, 0]))
      _metrics_for(metrics, request).token_times.append(now)
      if len(request.generated) >= request.max_new_tokens:
        del slots[slot]
        done.append(request)

  elapsed = time.perf_counter() - start
  return _summarise(done, metrics, [], elapsed, steps, None, 0)


def _summarise(
    done,
    metrics: dict[str, RequestMetrics],
    occupancy: Sequence[tuple[float, int, int, int]],
    elapsed: float,
    steps: int,
    allocator: Any,
    page_size: int,
    retained_pages: int = 0,
) -> dict[str, Any]:
  """Collapse a run into the reported schema.

  `retained_pages` is what the prefix cache is deliberately still holding once
  every request has finished. Those pages are allocated on purpose, so counting
  them as leaked would report a leak on every run with sharing enabled -- and an
  alarm that always fires is one nobody reads, which would hide the real leak
  this metric exists to catch.
  """
  output_tokens = sum(len(r.generated) for r in done)
  prompt_tokens = sum(r.prompt_len for r in done)
  timings = [metrics[r.request_id] for r in done if r.request_id in metrics]
  itls = [gap for entry in timings for gap in entry.inter_token_latencies()]

  summary: dict[str, Any] = {
      "completed_requests": len(done),
      "prompt_tokens": prompt_tokens,
      "output_tokens": output_tokens,
      "duration_s": elapsed,
      "output_throughput_tok_per_s": output_tokens / elapsed if elapsed else 0.0,
      "request_throughput_per_s": len(done) / elapsed if elapsed else 0.0,
      "engine_steps": steps,
      # Non-zero means the pool was overcommitted and requests were replayed.
      # Prompt and output token counts are shifted for those requests, so a run
      # with preemptions is a capacity result rather than a throughput one.
      "preemptions": sum(r.preemptions for r in done),
      "requests_preempted": sum(1 for r in done if r.preemptions),
      "ttft": _percentiles([e.ttft for e in timings if e.ttft is not None]),
      "itl": _percentiles(itls),
      "peak_device_bytes": _peak_bytes(),
  }

  if occupancy:
    times, pool_tokens, live_tokens, concurrency = (np.asarray(c) for c in zip(*occupancy))
    summary["occupancy"] = {
        # The claim: pages held track tokens held. A ratio pinned above 1 that
        # never comes down means pages are not being reclaimed.
        "max_pool_tokens_held": int(pool_tokens.max()),
        "max_live_tokens": int(live_tokens.max()),
        "mean_overhead_ratio": float((pool_tokens / np.maximum(live_tokens, 1)).mean()),
        "max_concurrency": int(concurrency.max()),
        "mean_concurrency": float(concurrency.mean()),
        "samples": len(occupancy),
        "series_time_s": times.tolist(),
        "series_pool_tokens": pool_tokens.tolist(),
        "series_live_tokens": live_tokens.tolist(),
    }
  if allocator is not None:
    summary["pages_retained_by_cache"] = int(retained_pages)
    summary["pages_leaked"] = int(allocator.num_allocated_pages) - int(retained_pages)
    summary["pool_capacity_tokens"] = allocator.capacity_pages * page_size
  return summary


def report(name: str, summary: dict[str, Any]) -> str:
  """A short human-readable form. The JSON is the artifact; this is for reading."""
  lines = [f"=== {name} ==="]
  lines.append(
      f"  requests {summary['completed_requests']:4d}"
      f"  output_tok {summary['output_tokens']:6d}"
      f"  {summary['duration_s']:.2f}s"
      f"  {summary['output_throughput_tok_per_s']:8.1f} tok/s"
      f"  steps {summary['engine_steps']}"
  )
  lines.append(
      f"  TTFT  p50 {summary['ttft']['p50_ms']:8.2f} ms   p99 {summary['ttft']['p99_ms']:8.2f} ms"
  )
  lines.append(
      f"  ITL   p50 {summary['itl']['p50_ms']:8.2f} ms   p99 {summary['itl']['p99_ms']:8.2f} ms"
  )
  if summary.get("peak_device_bytes"):
    lines.append(f"  peak device memory {summary['peak_device_bytes'] / 2**30:.3f} GiB")
  occ = summary.get("occupancy")
  if occ:
    ratio = occ["mean_overhead_ratio"]
    # Below 1 means the pool is holding fewer tokens than the requests
    # collectively address, which only happens when several of them are reading
    # the same pages. Calling that "overhead" would report the benefit as a cost.
    label = f"page overhead x{ratio:.3f}" if ratio >= 1 else f"sharing dividend x{1 / ratio:.3f}"
    lines.append(
        f"  pool tokens held max {occ['max_pool_tokens_held']}"
        f"  vs live tokens max {occ['max_live_tokens']}  ({label})"
    )
    lines.append(
        f"  concurrency max {occ['max_concurrency']}  mean {occ['mean_concurrency']:.2f}"
        f"  of pool capacity {summary.get('pool_capacity_tokens')} tokens"
    )
  if "pages_leaked" in summary:
    retained = summary.get("pages_retained_by_cache", 0)
    suffix = f"  (plus {retained} deliberately retained by the prefix cache)" if retained else ""
    lines.append(f"  pages leaked {summary['pages_leaked']}{suffix}")
  if "distinct_shapes_total" in summary:
    lines.append(
        f"  shapes {summary['distinct_shapes_total']} total,"
        f" {summary['shapes_compiled_during_measurement']} unwarmed"
    )
  if "repeat_durations_s" in summary:
    spread = " ".join(f"{d:.3f}" for d in summary["repeat_durations_s"])
    verdict = "reportable" if summary.get("latency_is_reportable") else "NOT reportable"
    lines.append(
        f"  repeats {spread} s  (post-first spread x{summary.get('stability_ratio', float('nan')):.3f})"
        f"  -> latency {verdict}"
    )
  return "\n".join(lines)


def write_json(path: str, payload: dict[str, Any]) -> None:
  with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
