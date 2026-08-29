"""Continuous batching for `OfflineEngine` on the paged KV pool.

The paged sibling of `offline_engine.InferenceWorker`, selected when
`attention="gpu_paged"`. It presents the same contract -- `run_inference(data,
rng)` returning one `CompletionOutput` per input -- and shares nothing else,
because the two differ in the thing that organises them.

**Why this is a separate worker rather than a flag on the dense one.** The dense
worker is built around a fixed decode *slot* per request: `empty_decode_slots`,
`slot_to_id`, a `DecodeState` whose batch dimension is the slot count, and a
`generate` that advances every slot in lockstep whether or not it holds a
request. A paged request owns a varying set of pages instead, which is why M4
made the release API request-based rather than `release_pages(slot)`. Threading a
pool through the slot machinery would mean keeping both models of ownership alive
in one loop; scheduling is delegated to `PagedDriver` instead, which already owns
admission, reservation, recycled-page scrubbing and recompute preemption.

**Detokenisation is synchronous here, deliberately.** The dense worker runs a
background thread emitting tokens as they arrive, because its loop cannot yield
between slots. This one has the whole token history per request when the driver
finishes, and offline inference has nobody waiting on a first token. A thread
would add ordering and shutdown hazards to buy latency nothing measures.

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

from typing import Any, Hashable

import numpy as np

import jax

from maxtext.inference import inference_utils
from maxtext.inference.kv_execution.driver import PagedDriver, PagedRequest
from maxtext.inference.maxengine.maxengine import MaxEngine
from maxtext.utils import max_logging


class PagedInferenceWorker:
  """Runs a batch of requests to completion on the page pool."""

  def __init__(
      self,
      config: Any,
      params: Any | None,
      devices: list[Any],
      tokenizer: Any,
      eos_ids: list[int] | None,
      max_decode_length: int,
      *,
      max_requests: int | None = None,
      max_batched_tokens: int | None = None,
      rng: jax.random.PRNGKey = None,
      mesh: Any = None,
      debug: bool = False,
  ):
    self.config = config
    self.devices = devices
    self.tokenizer = tokenizer
    self.eos_ids = eos_ids
    self.max_decode_length = int(max_decode_length)
    self.mesh = mesh
    self.rng = jax.random.PRNGKey(0) if rng is None else rng
    self.debug = debug

    self.engine = MaxEngine(self.config, self.devices)
    self.params = self.engine.load_params(params=params, rng=self.rng)
    if self.tokenizer is None:
      self.tokenizer = self._build_tokenizer()
    if self.eos_ids is None:
      if self.tokenizer is None:
        raise ValueError(
            "no tokenizer and no eos_ids: the worker cannot tell when a request has finished, so every "
            "one would generate to its length cap. Pass eos_ids for a token-in, token-out caller, or a "
            "tokenizer_path for text."
        )
      self.eos_ids = [self.tokenizer.eos_id]

    self.runtime = self.engine.init_paged_runtime(
        max_requests=max_requests, max_batched_tokens=max_batched_tokens
    )
    # `max_requests` bounds the page map's rows and the batch bucket ladder, so it
    # is the concurrency ceiling the driver schedules against.
    self.max_batch = self.runtime.control_plane.page_map.max_requests
    max_logging.log(
        f"Paged inference worker ready: pool {self.runtime.pool.k_pages[0].shape}, "
        f"{self.max_batch} concurrent requests"
    )

  def _build_tokenizer(self):
    """A tokenizer for this config, without JetStream and without torch.

    Order of preference, and each fallback is a real narrowing rather than a
    stylistic one:

    1. Nothing, when `eos_ids` was supplied and no `tokenizer_path` is set. A
       token-in, token-out caller -- the benchmark harnesses, the parity tests --
       needs no tokenizer at all, and building one would demand a path it has no
       reason to have.
    2. `hf_tokenizer.build_tokenizer`, which reads `tokenizer.json` through the
       `tokenizers` package. No JetStream, and no torch, which matters because
       `transformers` imports torch when it finds it and a second HIP runtime
       aborts RCCL clique setup above one device.
    3. `MaxEngine.build_tokenizer`, the JetStream route, only if asked for a
       tokenizer type this cannot serve. It raises a clear message under
       `DECOUPLE_GCLOUD=TRUE`, which is the honest outcome: that path genuinely
       needs a package that was archived in February 2026.
    """
    path = getattr(self.config, "tokenizer_path", "") or ""
    if not path:
      return None

    # `.value` first: `tokenizer_type` is a `TokenizerType` enum, whose `str()` is
    # "TokenizerType.HUGGINGFACE" rather than "huggingface". Comparing the string
    # form silently matched nothing and fell through to the JetStream branch.
    declared = getattr(self.config, "tokenizer_type", "") or ""
    tokenizer_type = str(getattr(declared, "value", declared)).lower()
    if tokenizer_type in ("", "huggingface"):
      # pylint: disable=import-outside-toplevel
      from maxtext.inference import hf_tokenizer

      eos = self.eos_ids[0] if self.eos_ids else None
      return hf_tokenizer.build_tokenizer(path, eos_id=eos)

    max_logging.log(
        f"tokenizer_type={tokenizer_type!r} is not served by the torch-free loader; falling back to "
        f"MaxEngine.build_tokenizer, which requires JetStream."
    )
    return self.engine.build_tokenizer(self.engine.get_tokenizer())

  def update_params(self, params: Any) -> None:
    """Update the model weights. The pool is unaffected and is not reallocated."""
    self.params = params

  def run_inference(self, data, rng=None) -> list:
    """Run every input to completion and return one output each, in input order.

    Args:
      data: `offline_engine.InputData`, whose `tokens` are padded and whose
        `true_length` says how much of that is real.
      rng: overrides the worker's key when given.

    Returns:
      One `offline_engine.CompletionOutput` per input, in the order supplied --
      *not* completion order, which paging makes arbitrary.
    """
    # pylint: disable=import-outside-toplevel
    from maxtext.inference.offline_engine import CompletionOutput

    if rng is not None:
      self.rng = rng
    if not data:
      return []

    # Log probabilities are part of `CompletionOutput`, and the driver's step
    # contract returns tokens only. The step function stashes each step's logits
    # here and the loop attributes them through `StepOutcome.batch` -- the same
    # shape of side-table the benchmark harness uses for timing, and for the same
    # reason: the driver reports what it advanced, so an observer needs nothing
    # more from the step itself.
    pending_logprobs: dict[str, Any] = {}

    def step(view, inputs, pool):
      del pool
      sampled, selected = self.engine.paged_step(
          params=self.params, view=view, inputs=inputs, is_decode=view.shape.is_decode
      )
      tokens = np.asarray(sampled).reshape(-1)[: inputs.sample_at.size]
      logprobs = np.asarray(
          inference_utils.log_prob_of_chosen_token(selected, sampled)
      ).reshape(-1)[: inputs.sample_at.size]
      pending_logprobs["last"] = logprobs
      return tokens

    driver = PagedDriver(
        self.runtime.control_plane,
        self.runtime.pool,
        step,
        max_batch=self.max_batch,
        eos_ids=self.eos_ids,
        runtime=self.runtime,
    )

    by_id: dict[Hashable, PagedRequest] = {}
    requests = []
    for row in data:
      prompt = np.asarray(row.tokens).reshape(-1)[: int(row.true_length)].astype(np.int64)
      request = PagedRequest(
          request_id=str(row.id),
          prompt_len=int(row.true_length),
          # Bounded by the pool's own context limit as well as the caller's, since
          # a request that cannot fit is rejected at admission rather than part way
          # through generating.
          max_new_tokens=min(self.max_decode_length, self.config.paged_max_context_len - int(row.true_length)),
          prompt_tokens=prompt,
      )
      requests.append(request)
      by_id[row.id] = request
    driver.submit(requests)

    logprobs_by_id: dict[Hashable, list[np.ndarray]] = {r.request_id: [] for r in requests}
    while True:
      outcome = driver.step()
      if outcome is None:
        break
      supplied = pending_logprobs.get("last")
      for index, request in enumerate(outcome.batch):
        if supplied is not None and index < supplied.size:
          logprobs_by_id[request.request_id].append(supplied[index])

    outputs = []
    for row in data:
      request = by_id[row.id]
      generated = np.asarray(request.generated, dtype=np.int32)
      logps = np.asarray(logprobs_by_id[request.request_id], dtype=np.float32)
      outputs.append(
          CompletionOutput(
              index=row.id,
              token_ids=generated,
              # Trimmed to the tokens actually returned: a preempted request is
              # replayed, so it produces more step observations than final tokens.
              logprobs=logps[: generated.size],
              prompt_length=int(row.true_length),
          )
      )
    if self.debug:
      max_logging.log(
          f"Paged worker completed {len(outputs)} requests, "
          f"{sum(r.preemptions for r in requests)} preemptions"
      )
    return outputs
