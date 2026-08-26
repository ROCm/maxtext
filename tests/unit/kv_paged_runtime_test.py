"""Acceptance gates for the paged KV runtime, against the real kernels.

Two properties that cannot be established on the host alone.

**No recycled bytes are readable by a subsequent request.** A freed page keeps the
previous occupant's KV until something overwrites it, so this checks the whole
chain: the control plane refuses to describe a dirty page, the driver zeroes what
it recycled, and the page a new request receives is genuinely clean. The negative
control matters as much as the positive one -- a test that passes because the
sentinel was never written in the first place proves nothing.

**Mixed-length churn traces a bounded set of shapes.** Counted by incrementing a
counter inside the traced function, so it measures actual retraces rather than the
driver's own opinion of how many shapes it produced.

`gpu_only`, and additionally skipped when jax-aiter is absent or its shims are not
built, following the M3 pattern. Note that `PYTHONPATH` must include the jax-aiter
checkout and `JA_ROOT_DIR` must point at it, or the FFI shims will not be found.

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

import unittest

from absl.testing import parameterized
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from maxtext.inference.kv_common import KvStorageLayoutV1
from maxtext.inference.kv_control import DirtyPageError, NativeKvControlPlane, RequestDescriptor
from maxtext.inference.kv_control import metadata as kv_metadata
from maxtext.inference.kv_execution import allocate_pool, PagedDriver, PagedRequest
from maxtext.inference.kv_execution.pool_ops import POISON_SENTINEL, poison_pages, scrub_pages
from maxtext.inference.kv_execution.step_view import build_step_view

PAGE = 16
# The AOT pa_ragged configurations cover head_size 128, block_size 16 and
# gqa_ratio in {1, 4, 8}. Equal query and KV head counts put the ratio at 1, so
# these shapes are inside the prebuilt set.
HEAD_DIM = 128
NUM_HEADS = 8
DONOR_VALUE = 100.0
READER_VALUE = 3.0


def _require_kernels():
  """Skip unless jax-aiter is importable and its KV shims are built."""
  try:
    from jax_aiter.ffi.registry import standalone_symbol_available  # pylint: disable=import-outside-toplevel
  except ImportError as exc:
    raise unittest.SkipTest(
        "jax-aiter is not importable; set PYTHONPATH to the jax-aiter checkout"
    ) from exc
  for symbol in ("AppendKvJA", "PagedAttentionJA", "PagedPrefillJA"):
    if not standalone_symbol_available(symbol):
      raise unittest.SkipTest(
          f"{symbol} is not built; run 'make -f Makefile.kv ja_kv' in jax-aiter and set JA_ROOT_DIR"
      )


def _layout(num_pages, num_layers=1) -> KvStorageLayoutV1:
  return KvStorageLayoutV1(
      tokens_per_page=PAGE,
      num_pages=num_pages,
      num_layers=num_layers,
      num_kv_heads=NUM_HEADS,
      head_dim=HEAD_DIM,
      dtype="bfloat16",
  )


def _append(pool, layer, slot_mapping, value):
  """Write a constant K and V at every non-padded slot."""
  from jax_aiter.ops.append_kv import append_kv  # pylint: disable=import-outside-toplevel

  count = int(np.asarray(slot_mapping).shape[0])
  block = jnp.full((count, NUM_HEADS, HEAD_DIM), value, jnp.bfloat16)
  k, v = append_kv(block, block, jnp.asarray(slot_mapping, jnp.int32), pool.k_pages[layer], pool.v_pages[layer])
  pool.replace_layer(layer, k, v)


def _decode_attend(pool, layer, view):
  """One decode step over the pool, returning float32 output."""
  from jax_aiter.ops.paged_attention import paged_attention  # pylint: disable=import-outside-toplevel

  query = jnp.ones((view.shape.num_tokens, NUM_HEADS, HEAD_DIM), jnp.bfloat16)
  out = paged_attention(
      query,
      pool.k_pages[layer],
      pool.v_pages[layer],
      view.kv_indptr,
      view.kv_page_indices,
      view.kv_last_page_lens,
      max_seq_len=view.shape.max_seqlen_k,
      scale=1.0,
  )
  return np.asarray(out.astype(jnp.float32))


@pytest.mark.gpu_only
class PoisonedPageTest(parameterized.TestCase):
  """The acceptance gate: a recycled page must carry nothing forward."""

  def setUp(self):
    super().setUp()
    _require_kernels()
    # Exactly one allocatable page, so the second request is forced to take the
    # first one's. Any larger pool would hand out a fresh page and the test would
    # pass without ever exercising recycling.
    self.layout = _layout(num_pages=2)
    self.plane = NativeKvControlPlane(
        layout=self.layout, max_requests=2, max_context_len=PAGE, debug_mode=True
    )
    self.pool = allocate_pool(self.layout)

  def _admit(self, request_id, prompt_len):
    handle = self.plane.admit(
        RequestDescriptor(request_id=request_id, prompt_len=prompt_len, max_new_tokens=0)
    )
    self.assertIsNotNone(handle, "admission must succeed for this test to mean anything")
    self.assertTrue(self.plane.reserve([handle], [prompt_len]))
    return handle

  def _shape(self, num_requests, num_tokens, num_pages, is_decode=True):
    from maxtext.inference.kv_execution.bucketing import StepShape  # pylint: disable=import-outside-toplevel

    return StepShape(
        num_requests=num_requests,
        num_tokens=num_tokens,
        num_pages=num_pages,
        max_seqlen_k=PAGE,
        is_decode=is_decode,
    )

  def _run_donor(self):
    """Fill the single page with a distinctive value, then free it poisoned."""
    donor = self._admit("donor", PAGE)
    self.plane.confirm_scrubbed(self.plane.pending_scrub())
    table = self.plane.build_page_table([donor], [PAGE])
    _append(self.pool, 0, table.slot_mapping(PAGE), DONOR_VALUE)

    page = int(self.plane.page_map.pages(donor)[0])
    written = np.asarray(self.pool.k_pages[0].astype(jnp.float32))[page]
    self.assertTrue(bool((written == DONOR_VALUE).all()), "the donor did not fill its page")

    k, v = poison_pages(self.pool.k_pages[0], self.pool.v_pages[0], [page])
    self.pool.replace_layer(0, k, v)
    self.plane.release(donor)
    return page

  def test_the_sentinel_survives_the_pool_dtype_exactly(self):
    """Otherwise every comparison against it is approximate and proves little.

    bfloat16 carries eight bits of mantissa, so a round decimal sentinel is
    stored rounded and `== POISON_SENTINEL` is false even on a page that was
    definitely poisoned -- which reads as a passing scrub test.
    """
    stored = np.asarray(jnp.full((1,), POISON_SENTINEL, jnp.bfloat16).astype(jnp.float32))
    self.assertEqual(float(stored[0]), POISON_SENTINEL)

  def test_the_donor_really_did_contaminate_the_page(self):
    """Without this the other tests could pass vacuously."""
    page = self._run_donor()
    written = np.asarray(self.pool.k_pages[0].astype(jnp.float32))[page]
    self.assertTrue(bool((written == POISON_SENTINEL).all()))
    self.assertTrue(self.plane.allocator.is_dirty(page))

  def test_poisoning_does_not_count_as_scrubbing(self):
    """A sentinel makes a missed scrub loud; it does not make the page safe."""
    self._run_donor()
    self._admit("reader", 1)
    self.assertEqual(self.plane.pending_scrub().size, 1)

  def test_the_control_plane_refuses_to_describe_the_dirty_page(self):
    """The enforcement point, with the real pool behind it."""
    self._run_donor()
    reader = self._admit("reader", 1)
    with self.assertRaises(DirtyPageError):
      self.plane.build_page_table([reader], [1])

  def test_after_scrubbing_no_recycled_byte_remains_in_the_page(self):
    """The property the milestone asks for, stated over the whole page.

    Checked across the entire page rather than only the readable extent. Exact
    last-page lengths already keep a well-behaved kernel inside valid data; this
    is what makes an over-read harmless too, and over-reads are the failure the
    scrub actually exists for.
    """
    page = self._run_donor()
    reader = self._admit("reader", 1)

    pending = self.plane.pending_scrub()
    k, v = scrub_pages(self.pool.k_pages[0], self.pool.v_pages[0], pending)
    self.pool.replace_layer(0, k, v)
    self.plane.confirm_scrubbed(pending)

    table = self.plane.build_page_table([reader], [1])
    self.assertEqual(int(self.plane.page_map.pages(reader)[0]), page, "the page must have been recycled")
    _append(self.pool, 0, table.slot_mapping(PAGE), READER_VALUE)

    written = np.asarray(self.pool.k_pages[0].astype(jnp.float32))[page]
    self.assertTrue(bool((written[0] == READER_VALUE).all()), "the reader's own token must be present")
    self.assertFalse(bool((written == POISON_SENTINEL).any()), "the poison sentinel survived the scrub")
    self.assertFalse(bool((written == DONOR_VALUE).any()), "the donor's KV survived the scrub")
    self.assertTrue(bool((written[1:] == 0.0).all()), "everything the reader did not write must be zero")

  def test_attention_over_a_recycled_page_returns_the_readers_own_value(self):
    """A single-token context makes the correct answer exact rather than close.

    Softmax over one key is 1.0 whatever the query, so the output must equal that
    token's V exactly. Any contribution from the recycled page would show up as a
    mixture.
    """
    self._run_donor()
    reader = self._admit("reader", 1)

    pending = self.plane.pending_scrub()
    k, v = scrub_pages(self.pool.k_pages[0], self.pool.v_pages[0], pending)
    self.pool.replace_layer(0, k, v)
    self.plane.confirm_scrubbed(pending)

    table = self.plane.build_page_table([reader], [1])
    _append(self.pool, 0, table.slot_mapping(PAGE), READER_VALUE)

    view = build_step_view(table, self._shape(1, 1, 1), tokens_per_page=PAGE)
    out = _decode_attend(self.pool, 0, view)
    np.testing.assert_array_equal(out, np.full_like(out, READER_VALUE))

  def test_without_the_scrub_the_contamination_is_still_there(self):
    """The negative control: the assertions above have teeth.

    Bypasses the control plane's gate by building the table through the metadata
    builder directly, which is the one way to reach a dirty page. The sentinel is
    then still in the page, so the check that it is absent after a scrub is
    testing something real.
    """
    page = self._run_donor()
    reader = self._admit("reader", 1)

    table = kv_metadata.build_page_table(self.plane.page_map, [reader], [1])
    _append(self.pool, 0, table.slot_mapping(PAGE), READER_VALUE)

    written = np.asarray(self.pool.k_pages[0].astype(jnp.float32))[page]
    self.assertTrue(bool((written[0] == READER_VALUE).all()))
    self.assertTrue(
        bool((written[1:] == POISON_SENTINEL).all()),
        "the unscrubbed page should still hold the previous occupant's bytes",
    )


class _TracingStep:
  """Runs the real append-and-attend under `jit`, counting actual retraces.

  The counter increments in the traced body, which executes once per trace, so
  this measures compilation rather than the driver's own shape bookkeeping. The
  two agreeing is the result worth having.
  """

  def __init__(self):
    self.traces = 0
    # What the shapes would have been with no bucketing, so a test can show the
    # collapse rather than merely assert that a number came out small.
    self.raw_shapes: set[tuple[int, int, int, bool]] = set()

    def body(query, key, value, k_pool, v_pool, slot_mapping, kv_indptr, kv_page_indices,
             kv_last_page_lens, cu_seqlens_q, *, max_seqlen_q, max_seqlen_k, is_decode):
      self.traces += 1
      # pylint: disable=import-outside-toplevel
      from maxtext.layers.gpu_paged_attention import PagedPlan, paged_attention_step

      plan = PagedPlan(
          slot_mapping=slot_mapping,
          kv_indptr=kv_indptr,
          kv_page_indices=kv_page_indices,
          kv_last_page_lens=kv_last_page_lens,
          cu_seqlens_q=cu_seqlens_q,
          max_seqlen_q=max_seqlen_q,
          max_seqlen_k=max_seqlen_k,
          is_decode=is_decode,
      )
      out, pools = paged_attention_step(
          query, key, value, k_pool, v_pool, plan, backend="aiter", scale=1.0, causal=True
      )
      return out, pools[0], pools[1]

    self._jitted = jax.jit(body, static_argnames=("max_seqlen_q", "max_seqlen_k", "is_decode"))

  def __call__(self, view, pool):
    seq_lens = np.asarray(view.seq_lens)
    self.raw_shapes.add(
        (
            view.num_active_requests,
            view.num_active_tokens,
            int(seq_lens.max()) if seq_lens.size else 0,
            view.shape.is_decode,
        )
    )
    tokens = view.shape.num_tokens
    block = jnp.ones((tokens, NUM_HEADS, HEAD_DIM), jnp.bfloat16)
    _, k, v = self._jitted(
        block,
        block,
        block,
        pool.k_pages[0],
        pool.v_pages[0],
        view.slot_mapping,
        view.kv_indptr,
        view.kv_page_indices,
        view.kv_last_page_lens,
        view.cu_seqlens_q,
        max_seqlen_q=view.max_seqlen_q,
        max_seqlen_k=view.shape.max_seqlen_k,
        is_decode=view.shape.is_decode,
      )
    pool.replace_layer(0, k, v)
    return np.full((view.num_active_requests,), 7, dtype=np.int32)


@pytest.mark.gpu_only
class CompileCountTest(parameterized.TestCase):
  """Mixed-length churn must not retrace without bound."""

  def test_churn_traces_no_more_shapes_than_the_ladders_allow(self):
    _require_kernels()
    layout = _layout(num_pages=129)
    plane = NativeKvControlPlane(layout=layout, max_requests=4, max_context_len=64, debug_mode=True)
    step = _TracingStep()
    driver = PagedDriver(plane, allocate_pool(layout), step, max_batch=4, max_batched_tokens=64)

    rng = np.random.default_rng(0)
    driver.submit(
        [
            PagedRequest(
                request_id=f"r{i}",
                prompt_len=int(rng.integers(4, 48)),
                max_new_tokens=int(rng.integers(1, 12)),
            )
            for i in range(24)
        ]
    )
    done = driver.run()

    self.assertEqual(len(done), 24)
    self.assertGreater(step.traces, 0, "nothing was traced, so nothing was measured")
    self.assertEqual(
        step.traces,
        driver.num_distinct_shapes,
        "each bucketed shape must be traced exactly once; a mismatch means something "
        "outside the bucketing is varying",
    )
    self.assertLessEqual(step.traces, driver.planner.max_distinct_shapes())
    # The bucketing has to be doing the work, not the workload happening to be
    # uniform: many raw shapes must be collapsing onto few traced ones.
    self.assertGreater(
        len(step.raw_shapes),
        3 * step.traces,
        f"only {len(step.raw_shapes)} distinct raw shapes for {step.traces} traces, so this workload "
        f"would not have retraced much anyway and proves little",
    )
    # Every page returned, so the churn was real rather than a single long batch.
    self.assertEqual(plane.allocator.num_allocated_pages, 0)
    self.assertEqual(plane.allocator.available_pages, plane.allocator.capacity_pages)

  def test_repeating_a_shape_does_not_retrace(self):
    """The property bucketing exists for, isolated from the scheduling loop."""
    _require_kernels()
    layout = _layout(num_pages=65)
    plane = NativeKvControlPlane(layout=layout, max_requests=2, max_context_len=64, debug_mode=True)
    step = _TracingStep()
    driver = PagedDriver(plane, allocate_pool(layout), step, max_batch=2, max_batched_tokens=64)

    driver.submit([PagedRequest(request_id="a", prompt_len=20, max_new_tokens=10)])
    driver.run()
    after_first = step.traces

    driver.submit([PagedRequest(request_id="b", prompt_len=20, max_new_tokens=10)])
    driver.run()
    self.assertEqual(step.traces, after_first, "an identical second request retraced")


if __name__ == "__main__":
  unittest.main()
