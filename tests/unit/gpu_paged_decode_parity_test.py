"""Model-level greedy decode: `gpu_paged` must match the dense path token for token.

This is M3's actual exit criterion, and it is a stronger claim than the
attention-layer parity test in `attention_test.py`. That one compares one layer
on one forward pass with weights copied across. This one runs a whole model
through `MaxEngine` twice, for a prompt and several generated tokens, and asks
whether the *sequence* agrees. The difference matters because a paged decode
gets its context from pages written on earlier steps, so an error in slot
arithmetic, page ordering or last-page occupancy shows up only after the context
crosses a page boundary — which a single forward pass never does.

It is also the wiring a benchmark needs. Both paths go through `MaxEngine`:
dense via `prefill`/`init_decode_state`/`insert`/`generate`, paged via the
sibling `init_paged_runtime`/`prefill_paged`/`generate_paged`. Nothing here
reaches around the engine into the model, so what passes here is what a serving
harness would drive.

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

import sys
import unittest

from absl.testing import parameterized
import numpy as np
import pytest

import jax
import jax.numpy as jnp
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils, model_creation_utils

try:
  from maxtext.inference.maxengine import maxengine
except ModuleNotFoundError as _exc:  # pragma: no cover - environment dependent
  # `maxengine` reaches JetStream for its engine base class and tokenizer types.
  # Naming both remedies matters: installing google-jetstream pulls TensorFlow and
  # two dozen other packages, which is a heavy intervention in a container built
  # around a self-built ROCm jaxlib, whereas DECOUPLE_GCLOUD=TRUE substitutes
  # stubs that carry everything this test reads.
  pytest.skip(
      f"MaxEngine needs JetStream ({_exc}). Either install google-jetstream under a constraints file "
      f"pinning jax and jaxlib, or run with DECOUPLE_GCLOUD=TRUE to use the built-in stubs.",
      allow_module_level=True,
  )

from tests.utils.test_helpers import get_test_config_path  # pylint: disable=wrong-import-position

PROMPT = [3, 17, 42, 5, 9, 21, 33, 2]
# 8 prompt tokens plus 45 generated is 53, so the context spans four pages and
# crosses three boundaries. That is the point: a decode that stayed inside its
# first page would exercise no page ordering and no gather, and would pass
# whatever the page arithmetic did.
STEPS = 45
PAGE = 16

# How close two logits may be and still count as tied. One bfloat16 ulp at unit
# magnitude, which is the resolution the reference itself is computed at.
#
# This exists because comparing two greedy *trajectories* token for token does
# not work here, and the reason took a while to see. Logits are computed in
# bfloat16, so the gap between the top two candidates is quantised, and exact
# ties are common — with this model they occur every few dozen steps and *not at
# reproducible places*, because bf16 quantises an accumulation whose order XLA is
# free to vary between processes. At a tied step argmax is decided by
# tie-breaking, two arithmetically correct implementations can pick differently,
# and every later token is downstream of that coin flip. A trajectory comparison
# is therefore flaky by construction: it looked at various times like a page
# ordering bug, a numerics drift and a clean pass, from the same code.
#
# So the comparison below is teacher-forced instead. It replays the paged path's
# own tokens through a full forward pass and asks, at every step, whether the
# token the paged path chose was *an* argmax. That is tie-tolerant, never
# diverges, and is a stronger statement than trajectory equality.
TIE_TOLERANCE = 2.0**-8

_COMMON = {
    "base_emb_dim": 512,
    "base_mlp_dim": 512,
    "base_num_query_heads": 4,
    "base_num_kv_heads": 4,
    "base_num_decoder_layers": 2,
    # head_dim 128 with equal query and KV head counts puts gqa_ratio at 1, which
    # is inside the prebuilt pa_ragged configuration set. head_dim 128 is also the
    # only size the ASM path accepts.
    "head_dim": 128,
    # 256 rather than 64. A small vocabulary over randomly initialised weights
    # makes exact logit ties likely and makes the model collapse to emitting one
    # token, both of which defeat a token-for-token comparison. See
    # MIN_LOGIT_MARGIN.
    "vocab_size": 256,
    "max_prefill_predict_length": 64,
    "max_target_length": 128,
    "per_device_batch_size": 1,
    "scan_layers": False,
    "sparse_matmul": False,
    # bfloat16, not float32: the paged kernels accept bfloat16 and float16 only,
    # and comparing a bf16 paged pool against an fp32 dense cache would measure
    # the dtype rather than the paging.
    "dtype": "bfloat16",
    "weight_dtype": "float32",
    "matmul_precision": "highest",
    "decode_sampling_strategy": "greedy",
    "enable_checkpointing": False,
    "skip_jax_distributed_system": True,
    "pure_nnx": True,
}

_DENSE = {"attention": "dot_product"}
_PAGED = {
    "attention": "gpu_paged",
    "paged_page_size": PAGE,
    # 1024 tokens of pool, comfortably more than one request needs, so nothing
    # here depends on recycling. The recycling path has its own tests.
    "paged_num_blocks": 64,
}


def _require_kernels():
  """Skip unless jax-aiter is importable and its KV shims are built."""
  try:
    from jax_aiter.ffi.registry import standalone_symbol_available  # pylint: disable=import-outside-toplevel
  except ImportError as exc:
    raise unittest.SkipTest("jax-aiter is not importable; set PYTHONPATH to the jax-aiter checkout") from exc
  for symbol in ("AppendKvJA", "PagedAttentionJA", "PagedPrefillJA"):
    if not standalone_symbol_available(symbol):
      raise unittest.SkipTest(f"{symbol} is not built; run 'make -f Makefile.kv ja_kv' and set JA_ROOT_DIR")


def _config(**overrides):
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **(_COMMON | overrides))


def _devices():
  """One device, explicitly, however many the host exposes.

  Parity is a claim about arithmetic, and sharding is a separate milestone, so a
  single device is the right scope. It is also the only scope that runs here: an
  8-way sharded model forward pass puts a collective in the program, and RCCL
  clique initialisation aborts in this container with a rocprofiler
  double-registration fatal once torch is loaded (transformers pulls it in). That
  is unrelated to paging, but it would present as a core dump in this test, so
  the device list is pinned rather than inherited.
  """
  return jax.devices()[:1]


def _mesh(cfg):
  return jax.sharding.Mesh(
      maxtext_utils.create_device_mesh(config=cfg, devices=_devices()), cfg.mesh_axes
  )


@pytest.mark.gpu_only
class GpuPagedDecodeParityTest(parameterized.TestCase):
  """A paged decode must reproduce the dense decode exactly."""

  def setUp(self):
    super().setUp()
    _require_kernels()

  def _build_params(self, cfg):
    """One set of random weights, to be loaded into both engines.

    Built once and shared rather than built twice from the same seed: sharing is
    what makes the comparison about paging instead of about whether two
    initialisations happened to agree.
    """
    mesh = _mesh(cfg)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
      model = model_creation_utils.create_model(
          cfg, mesh, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(params=0, dropout=0)
      )
    _, params_state, _ = nnx.split(model, nnx.Param, ...)
    return params_state

  def _verify_tokens_are_argmax(self, cfg, params_state, generated):
    """Replay `generated` through full forward passes and check each was an argmax.

    Teacher-forced, so the reference follows the sequence under test rather than
    running its own trajectory. That removes the failure mode a trajectory
    comparison cannot avoid: one tied step no longer invalidates everything after
    it, because the reference is re-anchored on the actual prefix every step.

    The reference has no KV cache at all, so it cannot share a bug with the thing
    under test. Returns the per-step margin between the chosen token and the best
    alternative, for reporting.
    """
    mesh = _mesh(cfg)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
      model = model_creation_utils.create_model(
          cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=0)
      )
    nnx.update(model, params_state)

    pad = cfg.max_target_length
    # Train mode shards the batch axis over fsdp, which absorbs every device, so
    # a batch of one cannot be laid out at all. Every row is the same prompt and
    # only row 0 is read.
    batch = cfg.micro_batch_size_to_train_on
    margins = []
    for step, chosen in enumerate(generated):
      context = list(PROMPT) + list(generated[:step])
      row = context + [0] * (pad - len(context))
      ids = jnp.tile(jnp.asarray([row], dtype=jnp.int32), (batch, 1))
      positions = jnp.tile(jnp.asarray([list(range(pad))], dtype=jnp.int32), (batch, 1))
      mask = [1] * len(context) + [0] * (pad - len(context))
      segment_ids = jnp.tile(jnp.asarray([mask], dtype=jnp.int32), (batch, 1))
      with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
        logits = model(
            ids, positions, decoder_segment_ids=segment_ids, enable_dropout=False, model_mode=MODEL_MODE_TRAIN
        )
      scores = np.asarray(logits[0, len(context) - 1].astype(jnp.float32))
      best = float(scores.max())
      self.assertGreaterEqual(
          float(scores[chosen]),
          best - TIE_TOLERANCE,
          f"at generated token {step} (context length {len(context)}, page boundary every {PAGE}) the "
          f"paged path chose {chosen} scoring {float(scores[chosen]):.5f}, but the best was "
          f"{int(scores.argmax())} scoring {best:.5f} — a gap of {best - float(scores[chosen]):.5f}, far "
          f"beyond the {TIE_TOLERANCE:.5f} a bfloat16 tie can explain",
      )
      margins.append(best - float(np.partition(scores, -2)[-2]))
    return margins

  def _dense_rollout(self, cfg, params_state, steps=STEPS):
    """Greedy decode on the dense two-region cache, through MaxEngine."""
    engine = maxengine.MaxEngine(cfg, _devices())
    params = engine.load_params(params=params_state)
    padded = jnp.asarray(
        PROMPT + [0] * (cfg.max_prefill_predict_length - len(PROMPT)), dtype=jnp.int32
    )
    prefix, first = engine.prefill(params=params, padded_tokens=padded, true_length=len(PROMPT))
    generated = [int(first.data[0, 0])]

    decode_state = engine.init_decode_state()
    decode_state = engine.insert(prefix, decode_state, slot=0)
    for _ in range(steps):
      decode_state, result = engine.generate(params, decode_state)
      generated.append(int(result.data[0, 0]))
    return generated

  def _paged_rollout(self, cfg, params_state, steps=STEPS):
    """Greedy decode on the page pool, through the sibling entry points."""
    engine = maxengine.MaxEngine(cfg, _devices())
    params = engine.load_params(params=params_state)
    runtime = engine.init_paged_runtime()

    padded = jnp.asarray(
        PROMPT + [0] * (cfg.max_prefill_predict_length - len(PROMPT)), dtype=jnp.int32
    )
    handle, first = engine.prefill_paged(
        params=params, padded_tokens=padded, true_length=len(PROMPT), request_id="parity"
    )
    self.assertIsNotNone(handle, "the pool refused a single request, so it is mis-sized for this test")
    generated = [int(first.data[0, 0])]

    for _ in range(steps):
      result, ok = engine.generate_paged(params, [handle], next_tokens=jnp.asarray([generated[-1]], jnp.int32))
      self.assertTrue(ok, "the pool ran out of pages during a single-request decode")
      generated.append(int(result.data[0, 0]))

    # Housekeeping is part of the contract, so assert it rather than assume it.
    pages_held = runtime.control_plane.page_map.num_pages(handle)
    self.assertGreaterEqual(
        pages_held, 3, f"the context only reached {pages_held} pages, so page ordering was barely exercised"
    )
    engine.release(handle)
    self.assertEqual(runtime.control_plane.allocator.num_allocated_pages, 0)
    return generated

  def _assert_same_tokens(self, reference, actual, label):
    """Compare sequences, reporting *where* they diverge.

    The index is the diagnosis. Token 0 comes from prefill, so a difference there
    points at the projection or the prefill kernel; a difference first appearing
    near a multiple of the page size points at page arithmetic; a drift that
    starts late and grows points at numerics.
    """
    self.assertEqual(len(reference), len(actual))
    first_diff = next((i for i, (a, b) in enumerate(zip(reference, actual)) if a != b), None)
    self.assertIsNone(
        first_diff,
        f"paged decode diverged from {label} at generated token {first_diff} "
        f"(context length {len(PROMPT) + (first_diff or 0)}, page boundary every {PAGE}):"
        f"\n  {label} = {reference}\n  paged = {actual}",
    )

  def test_every_paged_token_is_an_argmax_of_a_full_forward_pass(self):
    """M3's exit criterion, in the form that is actually decidable.

    The strong statement: at every step, over a 53-token context spanning four
    pages, the token the paged path produced was the argmax of a cacheless
    forward pass over exactly the prefix the paged path had built. A page
    ordering error, a slot-arithmetic error or a stale last-page length would all
    put a wrong token somewhere in that sequence.
    """
    params_state = self._build_params(_config(**_DENSE))
    paged = self._paged_rollout(_config(**_PAGED), params_state)
    margins = self._verify_tokens_are_argmax(_config(**_DENSE), params_state, paged)

    self.assertEqual(len(paged), STEPS + 1)
    # Guard against the check passing because everything was tied. A run where
    # most steps are decidable is a run where the assertion above meant something.
    decidable = sum(1 for m in margins if m > TIE_TOLERANCE)
    self.assertGreater(
        decidable,
        len(margins) // 2,
        f"only {decidable} of {len(margins)} steps had a decidable argmax, so this proved little",
    )

  def test_paged_decode_matches_the_dense_engine_path_token_for_token(self):
    """The same comparison against the dense cached path, engine to engine.

    The dense `_prefill_jit` returns its `ResultTokens` from inside `jit`, so the
    type has to be a registered pytree. The paged siblings build theirs outside
    `jit` and are unaffected, which is why the forward-pass comparison above runs
    everywhere.

    The skip tests that capability directly rather than asking whether JetStream
    is stubbed. Those are different questions: the stub is registered as a pytree
    precisely so this path works without the real package, and a
    provenance-based check would skip a comparison that is perfectly able to run.
    """
    probe = maxengine.engine_api.ResultTokens(
        data=jnp.zeros((1, 3), jnp.int32),
        tokens_idx=(0, 1),
        valid_idx=(1, 2),
        length_idx=(2, 3),
        log_prob=None,
        samples_per_slot=1,
    )
    if not jax.tree_util.tree_leaves(probe):
      self.skipTest(
          "engine_api.ResultTokens is not a registered pytree, so the dense prefill path cannot "
          "return one from inside jit; install google-jetstream or register the stub"
      )
    if jax.device_count() != len(_devices()):
      # `pyconfig` derives the batch from `jax.device_count()`, so it sizes the
      # dense cache for every visible device while the engine mesh is pinned to
      # one. Prefill then runs at batch 1 against a batch-N cache and asserts
      # deep inside the attention op. The paged path is unaffected because its
      # pool geometry comes from the layout rather than from the device count.
      self.skipTest(
          f"the dense path needs the process to see exactly the devices the engine uses; "
          f"{jax.device_count()} are visible and the engine is pinned to {len(_devices())}. "
          f"Re-run with HIP_VISIBLE_DEVICES=0 (or CUDA_VISIBLE_DEVICES=0)."
      )
    params_state = self._build_params(_config(**_DENSE))
    dense = self._dense_rollout(_config(**_DENSE), params_state)
    # Verified against the forward pass rather than against each other, because a
    # single tied step would otherwise make the two trajectories diverge for
    # reasons that have nothing to do with paging.
    self._verify_tokens_are_argmax(_config(**_DENSE), params_state, dense)
    paged = self._paged_rollout(_config(**_PAGED), params_state)
    self._verify_tokens_are_argmax(_config(**_DENSE), params_state, paged)

  def test_the_driver_reproduces_the_engine_entry_points(self):
    """`PagedDriver` driving the engine must match `prefill_paged`/`generate_paged`.

    The equivalence that licenses the refactor. Both now go through the same
    `build_step_inputs` and `paged_step`, but they reach them differently: the
    entry points admit one request and assemble a slice from their arguments,
    while the driver schedules a queue and assembles slices from `PagedRequest`.
    A divergence here means the driver's half of the position rule disagrees with
    the engine's, which is precisely the failure a shared seam is meant to make
    impossible -- so it is worth checking rather than assuming.
    """
    # pylint: disable=import-outside-toplevel
    from maxtext.inference.kv_execution.driver import PagedDriver, PagedRequest

    cfg = _config(**_PAGED)
    params_state = self._build_params(cfg)
    reference = self._paged_rollout(cfg, params_state, steps=STEPS)

    engine = maxengine.MaxEngine(cfg, _devices())
    params = engine.load_params(params=params_state)
    runtime = engine.init_paged_runtime()
    driver = PagedDriver(
        runtime.control_plane,
        runtime.pool,
        engine.paged_step_fn(params),
        max_batch=1,
        max_batched_tokens=cfg.max_prefill_predict_length,
    )
    # The driver counts its own generated tokens, so it wants exactly as many as
    # the reference produced: one from prefill plus STEPS decodes.
    driver.submit(
        [
            PagedRequest(
                request_id="driven",
                prompt_len=len(PROMPT),
                max_new_tokens=STEPS + 1,
                prompt_tokens=np.asarray(PROMPT, dtype=np.int64),
            )
        ]
    )
    done = driver.run()

    self.assertEqual(len(done), 1)
    self._assert_same_tokens(reference, done[0].generated, "driver against the engine entry points")
    self.assertEqual(
        runtime.control_plane.allocator.num_allocated_pages, 0, "the driver must release what it took"
    )

  def test_a_batched_prefill_samples_every_request(self):
    """Two requests in one prefill step must each get their own token.

    New capability, and the reason it needs its own test: prefill packs requests
    along the sequence axis at batch one, so before `sample_rows` existed the
    gather was `logits[arange(batch), sample_at]` and returned a *single* token
    however many requests were packed. The driver has always batched prefill, so
    that was a live trap rather than a hypothetical -- and a wrong-length result is
    the good case, because a driver that receives one token for two requests
    assigns the wrong token to the second.

    Compared against prefilling the same two prompts separately, which is the only
    reference that distinguishes "packed correctly" from "packed consistently".
    """
    # pylint: disable=import-outside-toplevel
    from maxtext.inference.kv_execution.step_inputs import RequestSlice, build_step_inputs

    cfg = _config(**_PAGED)
    params_state = self._build_params(cfg)

    # Two distinct prompts, so a swap or a duplicate is visible.
    first, second = PROMPT[:6], PROMPT[2:10]
    self.assertNotEqual(first, second, "the two prompts must differ for this test to mean anything")

    separate = []
    for index, prompt in enumerate((first, second)):
      engine = maxengine.MaxEngine(cfg, _devices())
      params = engine.load_params(params=params_state)
      engine.init_paged_runtime()
      padded = jnp.asarray(
          list(prompt) + [0] * (cfg.max_prefill_predict_length - len(prompt)), dtype=jnp.int32
      )
      handle, result = engine.prefill_paged(
          params=params, padded_tokens=padded, true_length=len(prompt), request_id=f"solo{index}"
      )
      self.assertIsNotNone(handle)
      separate.append(int(result.data[0, 0]))
      engine.release(handle)

    # Now both in one packed step, driven at the seam rather than through the
    # single-request entry point.
    engine = maxengine.MaxEngine(cfg, _devices())
    params = engine.load_params(params=params_state)
    # Two rows explicitly: `init_paged_runtime` defaults to the dense batch width,
    # which is one here, and a one-row page map cannot hold a two-request batch.
    runtime = engine.init_paged_runtime(
        max_requests=2, max_batched_tokens=cfg.max_prefill_predict_length
    )
    handles = [
        runtime.admit(request_id=f"packed{i}", prompt_len=len(p), max_new_tokens=1)
        for i, p in enumerate((first, second))
    ]
    self.assertTrue(all(h is not None for h in handles), "the pool refused a two-request batch")

    query_lens = [len(first), len(second)]
    view = runtime.prepare_step(handles, query_lens, is_decode=False, num_requests=2)
    self.assertIsNotNone(view, "the pool could not back a two-request prefill")
    inputs = build_step_inputs(
        [
            RequestSlice(tokens=np.asarray(p, np.int64), start=0, query_len=len(p))
            for p in (first, second)
        ],
        view.shape,
        is_decode=False,
    )
    self.assertEqual(inputs.sample_at.size, 2, "one sample index per packed request")
    sampled, _ = engine.paged_step(params=params, view=view, inputs=inputs, is_decode=False)

    packed = np.asarray(sampled).reshape(-1)[:2].tolist()
    self.assertEqual(
        packed,
        separate,
        "a packed prefill must give each request the token it would have got alone",
    )
    for handle in handles:
      engine.release(handle)

  def test_offline_engine_selects_the_paged_worker_and_agrees_with_the_engine(self):
    """`OfflineEngine.batch_inference` is the production entry point, so wire it.

    Until the step seam existed there was no paged worker to select, and this is
    the check that selecting one produces the same answer. Several prompts at
    once, so it exercises the continuous batching that a single-request rollout
    cannot: requests at different positions sharing one pool.

    The reference is the engine's own paged entry points rather than the dense
    worker, and that is a container limitation rather than a choice -- the dense
    `_prefill_jit` returns a `ResultTokens` from inside `jit`, which has to be a
    registered pytree, and the `DECOUPLE_GCLOUD` stub is a plain class. The parity
    tests above already tie those entry points to dense, so agreement here is
    transitive.
    """
    # pylint: disable=import-outside-toplevel
    import jax.numpy as jnp

    from maxtext.inference import offline_engine

    prompts = [list(range(3, 15)), list(range(40, 48)), list(range(90, 106))]
    cfg = _config(**_PAGED, return_log_prob=True)

    # Reference: one request at a time through the entry points.
    reference = {}
    engine = maxengine.MaxEngine(cfg, _devices())
    params_state = self._build_params(cfg)
    params = engine.load_params(params=params_state)
    engine.init_paged_runtime(max_requests=4, max_batched_tokens=cfg.max_prefill_predict_length)
    budget = cfg.max_target_length - cfg.max_prefill_predict_length
    for index, prompt in enumerate(prompts):
      padded = jnp.asarray(
          list(prompt) + [0] * (cfg.max_prefill_predict_length - len(prompt)), jnp.int32
      )
      handle, first = engine.prefill_paged(
          params=params, padded_tokens=padded, true_length=len(prompt), request_id=f"ref{index}"
      )
      self.assertIsNotNone(handle)
      tokens = [int(first.data[0, 0])]
      for _ in range(min(budget, cfg.paged_max_context_len - len(prompt)) - 1):
        result, ok = engine.generate_paged(
            params, [handle], next_tokens=jnp.asarray([tokens[-1]], jnp.int32)
        )
        if not ok:
          break
        tokens.append(int(result.data[0, 0]))
      engine.release(handle)
      reference[f"p{index}"] = tokens

    # Through OfflineEngine, which must pick the paged worker off the attention
    # setting. `eos_ids` is supplied so no tokenizer is needed -- these prompts are
    # token ids, and a tokenizer would need JetStream.
    offline = offline_engine.OfflineEngine(
        config=cfg, params=params_state, tokenizer=object(), eos_ids=[-1]
    )
    self.assertEqual(
        type(offline.worker).__name__,
        "PagedInferenceWorker",
        "attention='gpu_paged' must select the paged worker",
    )
    outputs = offline.batch_inference(
        [
            offline_engine.InputData(id=f"p{i}", tokens=np.asarray(p, np.int32), true_length=len(p))
            for i, p in enumerate(prompts)
        ]
    )

    self.assertEqual(len(outputs), len(prompts))
    for out in outputs:
      expected = reference[out.index]
      actual = np.asarray(out.token_ids).tolist()
      self._assert_same_tokens(expected[: len(actual)], actual[: len(expected)], f"{out.index}")
      # Logprobs are part of the contract and `_validate_config` insists on them,
      # so an empty array would satisfy the token check and still be wrong.
      self.assertEqual(
          np.asarray(out.logprobs).size,
          len(actual),
          "one log probability per returned token",
      )
      self.assertEqual(out.prompt_length, len(prompts[int(out.index[1:])]))

  def test_the_paged_path_allocates_no_dense_cache(self):
    """A dense cache alongside the pool would waste gigabytes at real sizes."""
    cfg = _config(**_PAGED)
    engine = maxengine.MaxEngine(cfg, _devices())
    params = engine.load_params(params=self._build_params(cfg))
    runtime = engine.init_paged_runtime()

    expected_bytes = runtime.control_plane.layout.pool_bytes_per_shard()
    actual = sum(
        int(np.prod(a.shape)) * a.dtype.itemsize
        for layer in range(runtime.pool.num_layers)
        for a in (runtime.pool.k_pages[layer], runtime.pool.v_pages[layer])
    )
    self.assertEqual(actual, expected_bytes)
    del params

  def test_the_pool_is_mutated_in_place_rather_than_replaced(self):
    """The M0 aliasing invariant, observed at the top of the stack.

    A pool that is copied rather than aliased still produces correct tokens, so
    nothing else in this file would notice. Checking that the arrays change
    identity exactly once per step -- because donation hands back a new handle to
    the same buffer -- is not the invariant either. What is observable here and
    worth pinning is that the pool does not grow: a replacement allocation per
    step would show up as a rising live-buffer count.
    """
    cfg = _config(**_PAGED)
    engine = maxengine.MaxEngine(cfg, _devices())
    params = engine.load_params(params=self._build_params(cfg))
    runtime = engine.init_paged_runtime()

    padded = jnp.asarray(PROMPT + [0] * (cfg.max_prefill_predict_length - len(PROMPT)), dtype=jnp.int32)
    handle, first = engine.prefill_paged(
        params=params, padded_tokens=padded, true_length=len(PROMPT), request_id="alias"
    )
    shapes_before = [(a.shape, a.dtype) for a in runtime.pool.k_pages]
    token = int(first.data[0, 0])
    for _ in range(8):
      result, ok = engine.generate_paged(params, [handle], next_tokens=jnp.asarray([token], jnp.int32))
      self.assertTrue(ok)
      token = int(result.data[0, 0])

    self.assertEqual([(a.shape, a.dtype) for a in runtime.pool.k_pages], shapes_before)
    for array in runtime.pool.k_pages + runtime.pool.v_pages:
      self.assertFalse(array.is_deleted(), "a donated pool array was never rebound")


if __name__ == "__main__":
  unittest.main()
