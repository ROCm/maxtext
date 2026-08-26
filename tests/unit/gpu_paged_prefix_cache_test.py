"""M5's exit criterion: a prefix cache hit must change the cost, not the answer.

The host-side tests in `kv_prefix_cache_test.py` prove that the right pages are
shared, that a namespace mismatch cannot hit, and that nothing leaks. None of
them can catch the failure that matters most here, because it is arithmetic
inside the model rather than bookkeeping around it.

**Position offset.** After a hit the step runs the prompt's *suffix*, and those
tokens sit at absolute positions `cached..prompt_len`. RoPE encodes absolute
position, so feeding the suffix at positions starting from zero produces K/V
rotated as though the suffix began the sequence. The pages would be laid out
correctly, the page table would be correct, nothing would leak, and the output
would be wrong. That is invisible to every host-side test and is precisely what
this file checks.

The claim is a comparison against the same engine with nothing cached: a warm
rollout must produce the identical token sequence to a cold one. Identical, not
close -- the two differ only in how much of the prompt was recomputed, so any
divergence at all is a bug rather than numerics. That also makes the test immune
to the tie-breaking flakiness the parity test has to work around, since both
sides here run the same kernels on the same pool.

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

from maxtext.common.common_types import MODEL_MODE_PREFILL
from maxtext.configs import pyconfig
from maxtext.inference.kv_common import CacheNamespace
from maxtext.utils import maxtext_utils, model_creation_utils

try:
  from maxtext.inference.maxengine import maxengine
except ModuleNotFoundError as _exc:  # pragma: no cover - environment dependent
  pytest.skip(
      f"MaxEngine needs JetStream ({_exc}). Install google-jetstream under a constraints file pinning "
      f"jax and jaxlib, or run with DECOUPLE_GCLOUD=TRUE to use the built-in stubs.",
      allow_module_level=True,
  )

from tests.utils.test_helpers import get_test_config_path  # pylint: disable=wrong-import-position

PAGE = 16
STEPS = 12

# 40 tokens is two and a half pages, so a hit shares whole pages and leaves a
# partial one to recompute. A prompt of one page or less would make the
# publication trivial and the suffix empty, and the position offset -- the thing
# most worth testing -- would never be exercised.
PROMPT = [3, 17, 42, 5, 9, 21, 33, 2, 11, 27] * 4

_COMMON = {
    "base_emb_dim": 512,
    "base_mlp_dim": 512,
    "base_num_query_heads": 4,
    "base_num_kv_heads": 4,
    "base_num_decoder_layers": 2,
    "head_dim": 128,
    "vocab_size": 256,
    "max_prefill_predict_length": 64,
    "max_target_length": 128,
    "per_device_batch_size": 1,
    "scan_layers": False,
    "sparse_matmul": False,
    "dtype": "bfloat16",
    "weight_dtype": "float32",
    "matmul_precision": "highest",
    "decode_sampling_strategy": "greedy",
    "enable_checkpointing": False,
    "skip_jax_distributed_system": True,
    "pure_nnx": True,
}

_PAGED = {
    "attention": "gpu_paged",
    "paged_page_size": PAGE,
    "paged_num_blocks": 64,
    "paged_enable_prefix_cache": True,
}

NAMESPACE = CacheNamespace(model_fingerprint="test-random-weights", tokenizer="test")


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
  """One device, for the same container reason as the parity test."""
  return jax.devices()[:1]


def _mesh(cfg):
  return jax.sharding.Mesh(maxtext_utils.create_device_mesh(config=cfg, devices=_devices()), cfg.mesh_axes)


@pytest.mark.gpu_only
class GpuPagedPrefixCacheTest(parameterized.TestCase):
  """Sharing a prefix must not change a single token."""

  def setUp(self):
    super().setUp()
    _require_kernels()
    self.cfg = _config(**_PAGED)
    mesh = _mesh(self.cfg)
    with nn_partitioning.axis_rules(self.cfg.logical_axis_rules), mesh:
      model = model_creation_utils.create_model(
          self.cfg, mesh, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(params=0, dropout=0)
      )
    _, self.params_state, _ = nnx.split(model, nnx.Param, ...)

  def _engine(self):
    engine = maxengine.MaxEngine(self.cfg, _devices())
    params = engine.load_params(params=self.params_state)
    runtime = engine.init_paged_runtime()
    return engine, params, runtime

  def _rollout(self, engine, params, request_id, prompt=None, namespace=NAMESPACE, steps=STEPS):
    """One greedy rollout, offering its context to the cache on the way out."""
    tokens = np.asarray(prompt if prompt is not None else PROMPT, dtype=np.int64)
    padded = jnp.asarray(
        list(tokens) + [0] * (self.cfg.max_prefill_predict_length - tokens.size), dtype=jnp.int32
    )
    handle, first = engine.prefill_paged(
        params=params,
        padded_tokens=padded,
        true_length=int(tokens.size),
        request_id=request_id,
        prompt_token_ids=tokens,
        namespace=namespace,
    )
    self.assertIsNotNone(handle, "the pool refused a single request, so it is mis-sized for this test")
    cached = engine.paged_runtime.cached_tokens(handle)

    generated = [int(first.data[0, 0])]
    for _ in range(steps):
      result, ok = engine.generate_paged(
          params, [handle], next_tokens=jnp.asarray([generated[-1]], jnp.int32)
      )
      self.assertTrue(ok, "the pool ran out of pages during a single-request decode")
      generated.append(int(result.data[0, 0]))

    context = np.concatenate([tokens, np.asarray(generated, dtype=np.int64)])
    engine.release(handle, context)
    return generated, cached

  def test_a_warm_rollout_produces_the_same_tokens_as_a_cold_one(self):
    """The exit criterion. A wrong position offset fails here and nowhere else.

    Both rollouts run in one engine so they share weights, kernels and pool
    exactly. The only difference is that the second one finds its prefix already
    computed, so an identical token sequence is the whole claim.
    """
    engine, params, runtime = self._engine()

    cold, cached_cold = self._rollout(engine, params, "cold")
    self.assertEqual(cached_cold, 0, "nothing was cached yet, so this rollout must have paid in full")
    self.assertGreater(runtime.control_plane.prefix_index.num_cached_pages, 0, "nothing was published")

    warm, cached_warm = self._rollout(engine, params, "warm")
    self.assertGreater(cached_warm, 0, "the repeated prompt did not hit the cache, so this proved nothing")
    self.assertEqual(
        cold,
        warm,
        f"a cache hit changed the output. {cached_warm} of {len(PROMPT)} prompt tokens were served from "
        f"the cache, and the suffix must be run at absolute positions {cached_warm}.. for RoPE to agree:"
        f"\n  cold = {cold}\n  warm = {warm}",
    )

  def test_a_namespace_mismatch_recomputes_and_still_agrees(self):
    """Two guarantees at once: the miss is real, and the miss path is unchanged."""
    engine, params, _ = self._engine()
    cold, _ = self._rollout(engine, params, "cold")

    other = CacheNamespace(model_fingerprint="test-random-weights", tokenizer="different")
    miss, cached = self._rollout(engine, params, "other-namespace", namespace=other)
    self.assertEqual(cached, 0, "a different tokenizer must not share pages")
    self.assertEqual(cold, miss)

  def test_a_shared_prefix_with_a_different_tail_agrees_with_a_cold_run(self):
    """The realistic case: a common system prompt and a divergent question.

    Stronger than the repeated-prompt test, because here the cache supplies part
    of the context and the step computes a genuine suffix, so the boundary
    between borrowed and freshly written pages falls mid-request.
    """
    engine, params, _ = self._engine()
    tail = [101, 202, 303, 44, 55, 66, 77, 88]

    baseline, cached_baseline = self._rollout(engine, params, "baseline", prompt=PROMPT + tail)
    self.assertEqual(cached_baseline, 0)

    self._rollout(engine, params, "publisher")
    shared, cached_shared = self._rollout(engine, params, "sharer", prompt=PROMPT + tail)

    self.assertGreater(cached_shared, 0, "the common prefix was not shared")
    self.assertLess(cached_shared, len(PROMPT) + len(tail), "the divergent tail must still be computed")
    self.assertEqual(baseline, shared, "sharing part of the context changed the output")

  def test_the_pool_is_accounted_for_once_the_cache_is_dropped(self):
    engine, params, runtime = self._engine()
    self._rollout(engine, params, "r0")
    self._rollout(engine, params, "r1")

    plane = runtime.control_plane
    self.assertEqual(plane.allocator.num_allocated_pages, plane.prefix_index.num_cached_pages)
    plane.evict_cached(plane.prefix_index.num_cached_pages)
    self.assertEqual(plane.allocator.num_allocated_pages, 0)


if __name__ == "__main__":
  unittest.main()
