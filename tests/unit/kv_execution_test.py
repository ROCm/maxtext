"""Tests for the paged KV execution layer: bucketing, step views, and the driver.

These need jax, but only on CPU: the step function is injected, so the scheduling
loop is exercised without a model or a kernel. The real-kernel checks -- the
poisoned-page acceptance gate and the compile-count bound -- live in
`kv_paged_runtime_test.py` and are `gpu_only`.

Same marking trap as the rest of this work: `tests/conftest.py` auto-marks
unmarked tests `cpu_only` and skips `cpu_only` on any accelerator testbed, so on a
GPU box this file skips rather than passes. Run it with `JAX_PLATFORMS=cpu` and
read the count.

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

import numpy as np

from maxtext.inference.kv_common import CacheNamespace, KvPageTableV1, KvStorageLayoutV1
from maxtext.inference.kv_control import NativeKvControlPlane, RequestHandle
from maxtext.inference.kv_execution import allocate_pool, PagedDriver, PagedRequest
from maxtext.inference.kv_execution.bucketing import (
    StepShape,
    StepShapePlanner,
    batch_ladder,
    bucket_up,
    seqlen_ladder,
    token_ladder,
)
from maxtext.inference.kv_execution.engine_adapter import PagedRuntime
from maxtext.inference.kv_execution.layout_builder import build_storage_layout, kv_head_shards
from maxtext.inference.kv_execution.pool_ops import (
    POISON_SENTINEL,
    poison_pages,
    scrub_pages,
    scrub_pages_all_layers,
)
from maxtext.inference.kv_execution.step_inputs import RequestSlice, build_step_inputs
from maxtext.inference.kv_execution.step_view import build_step_view

PAGE = 16


def _layout(**kw) -> KvStorageLayoutV1:
  """A small pool geometry, overridable field by field."""
  base = {
      "tokens_per_page": PAGE,
      "num_pages": 65,
      "num_layers": 2,
      "num_kv_heads": 4,
      "head_dim": 32,
      "dtype": "float32",
  }
  base.update(kw)
  return KvStorageLayoutV1(**base)


class _FakeMesh:
  def __init__(self, shape):
    self.shape = shape


class _FakeConfig:
  """The handful of attributes `build_storage_layout` reads."""

  def __init__(self, **kw):
    self.num_kv_heads = 4
    self.head_dim = 128
    self.num_decoder_layers = 8
    self.paged_page_size = PAGE
    self.paged_num_blocks = 256
    self.dtype = "bfloat16"
    for key, value in kw.items():
      setattr(self, key, value)


class LadderTest(unittest.TestCase):
  """The power-of-two rungs that decide which shapes can exist."""

  def test_bucket_up_finds_the_smallest_sufficient_rung(self):
    ladder = (1, 2, 4, 8)
    self.assertEqual(bucket_up(1, ladder), 1)
    self.assertEqual(bucket_up(3, ladder), 4)
    self.assertEqual(bucket_up(8, ladder), 8)

  def test_exceeding_the_ladder_raises_rather_than_clamping(self):
    """Clamping would silently drop the tokens that did not fit."""
    with self.assertRaisesRegex(ValueError, "mis-sized"):
      bucket_up(9, (1, 2, 4, 8))

  def test_batch_ladder_starts_at_one(self):
    self.assertEqual(batch_ladder(8), (1, 2, 4, 8))
    self.assertEqual(batch_ladder(5), (1, 2, 4, 8))

  def test_token_ladder_starts_at_sixty_four(self):
    """Matching MaxText's existing prefill bucketing; finer rungs buy nothing."""
    self.assertEqual(token_ladder(256), (64, 128, 256))
    self.assertEqual(token_ladder(10), (64,))

  def test_seqlen_ladder_starts_at_the_page_size(self):
    """A shorter context still occupies a whole page, so a finer rung cannot occur."""
    self.assertEqual(seqlen_ladder(16, 128), (16, 32, 64, 128))


class StepShapeTest(unittest.TestCase):
  """Which shapes the two phases produce, and how many there can be."""

  def _planner(self, max_batch=8, max_context_len=128, pool_pages=65, max_batched_tokens=None):
    return StepShapePlanner(
        tokens_per_page=PAGE,
        max_batch=max_batch,
        max_context_len=max_context_len,
        pool_pages=pool_pages,
        max_batched_tokens=max_batched_tokens,
    )

  def test_decode_ties_the_token_count_to_the_batch_bucket(self):
    """One token per request, so the token axis is not independently free."""
    shape = self._planner().decode_shape(num_requests=3, max_seq_len=40)
    self.assertEqual(shape.num_requests, 4)
    self.assertEqual(shape.num_tokens, 4)
    self.assertTrue(shape.is_decode)

  def test_extend_pins_the_batch_and_varies_only_tokens(self):
    planner = self._planner(max_batch=8, max_batched_tokens=512)
    first = planner.extend_shape(num_tokens=100, max_seq_len=100)
    second = planner.extend_shape(num_tokens=200, max_seq_len=100)
    self.assertEqual(first.num_requests, 8)
    self.assertEqual(second.num_requests, 8)
    self.assertEqual(first.num_tokens, 128)
    self.assertEqual(second.num_tokens, 256)

  def test_the_token_budget_is_a_batch_budget_not_a_request_length(self):
    """An extend step batches requests, so its total exceeds the longest one.

    Sizing the token ladder from max_context_len made a perfectly legal batch --
    four 100-token prompts against a 128-token cap -- impossible to bucket.
    """
    planner = self._planner(max_context_len=128, max_batched_tokens=1024)
    self.assertEqual(planner.extend_shape(num_tokens=400, max_seq_len=128).num_tokens, 512)

  def test_a_batch_past_the_token_budget_is_refused(self):
    planner = self._planner(max_context_len=128, max_batched_tokens=128)
    with self.assertRaisesRegex(ValueError, "mis-sized"):
      planner.extend_shape(num_tokens=400, max_seq_len=128)

  def test_the_gather_table_is_an_upper_bound_not_a_guess(self):
    """Derived from batch and length, so it cannot under-size the page list."""
    planner = self._planner(max_batch=4, max_context_len=64, pool_pages=1024)
    shape = planner.decode_shape(num_requests=4, max_seq_len=64)
    self.assertEqual(shape.num_pages, 4 * 4)

  def test_the_gather_table_is_clamped_to_the_pool(self):
    planner = self._planner(max_batch=256, max_context_len=4096, pool_pages=40)
    self.assertEqual(planner.decode_shape(256, 4096).num_pages, 40)

  def test_a_bucketed_shape_is_reached_by_a_range_of_batches(self):
    """The property that bounds compilation: many inputs, one shape."""
    planner = self._planner()
    shapes = {planner.decode_shape(n, 40) for n in (5, 6, 7, 8)}
    self.assertEqual(len(shapes), 1)

  def test_the_shape_count_has_a_stated_bound(self):
    planner = self._planner(max_batch=8, max_context_len=128)
    # 4 batch rungs + 2 token rungs, each against 4 seqlen rungs
    self.assertEqual(planner.max_distinct_shapes(), (4 + 2) * 4)


class StepViewTest(unittest.TestCase):
  """Padding a page table into a static shape, inertly."""

  def _table(self):
    return KvPageTableV1(
        page_ids=[[1, 2, 3], [4, 5]],
        seq_lens=np.array([33, 20], dtype=np.int32),
        query_lens=np.array([1, 1], dtype=np.int32),
        write_positions=np.array([32, 19], dtype=np.int32),
        request_order=np.array([0, 1], dtype=np.int32),
    )

  def _view(self, num_requests=4, num_tokens=4, num_pages=8, max_seqlen_k=64):
    """The two-request decode table above, padded to a given bucketed shape."""
    from maxtext.inference.kv_execution.bucketing import StepShape  # pylint: disable=import-outside-toplevel

    shape = StepShape(
        num_requests=num_requests,
        num_tokens=num_tokens,
        num_pages=num_pages,
        max_seqlen_k=max_seqlen_k,
        is_decode=True,
    )
    return build_step_view(self._table(), shape, tokens_per_page=PAGE)

  def test_the_arrays_have_exactly_the_bucketed_shapes(self):
    view = self._view()
    self.assertEqual(view.slot_mapping.shape, (4,))
    self.assertEqual(view.kv_indptr.shape, (5,))
    self.assertEqual(view.kv_page_indices.shape, (8,))
    self.assertEqual(view.kv_last_page_lens.shape, (4,))
    self.assertEqual(view.cu_seqlens_q.shape, (5,))
    self.assertEqual(view.num_active_requests, 2)
    self.assertEqual(view.num_active_tokens, 2)

  def test_padded_writes_go_to_the_skip_sentinel(self):
    """Not to a real slot, which would write garbage into a live page."""
    slots = np.asarray(self._view().slot_mapping)
    np.testing.assert_array_equal(slots[:2], [48, 83])
    np.testing.assert_array_equal(slots[2:], [-1, -1])

  def test_padded_requests_get_zero_length_page_ranges(self):
    """A repeated final indptr entry, so a kernel does no work for them."""
    view = self._view()
    np.testing.assert_array_equal(np.asarray(view.kv_indptr), [0, 3, 5, 5, 5])
    np.testing.assert_array_equal(np.asarray(view.cu_seqlens_q), [0, 1, 2, 2, 2])

  def test_padded_gather_entries_point_at_the_reserved_page(self):
    """Belt and braces behind the flat indptr: the reserved page reads as zeros."""
    view = self._view()
    indices = np.asarray(view.kv_page_indices)
    np.testing.assert_array_equal(indices[:5], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(indices[5:], [0, 0, 0])

  def test_padded_lengths_are_zero(self):
    view = self._view()
    np.testing.assert_array_equal(np.asarray(view.kv_last_page_lens), [1, 4, 0, 0])
    np.testing.assert_array_equal(np.asarray(view.seq_lens), [33, 20, 0, 0])

  def test_a_bucket_too_small_for_the_batch_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "kv_page_indices"):
      self._view(num_pages=2)

  def test_a_sequence_longer_than_the_configured_bucket_is_rejected(self):
    """Silently truncating max_seqlen_k would under-configure the kernel."""
    with self.assertRaisesRegex(ValueError, "ladder is mis-sized"):
      self._view(max_seqlen_k=32)

  def test_it_converts_to_the_attention_path_plan(self):
    plan = self._view().to_paged_plan()
    self.assertTrue(plan.is_decode)
    self.assertEqual(plan.max_seqlen_q, 1)
    self.assertEqual(plan.max_seqlen_k, 64)


class PoolTest(unittest.TestCase):
  """Allocation and the two hygiene operations."""

  def test_the_pool_starts_zeroed(self):
    """Load-bearing: the reserved page and every unscrubbed page rely on it."""
    pool = allocate_pool(_layout(num_pages=8, num_layers=3))
    self.assertEqual(pool.num_layers, 3)
    self.assertEqual(pool.page_shape, (8, PAGE, 4, 32))
    for layer in range(3):
      self.assertTrue(bool((np.asarray(pool.k_pages[layer]) == 0).all()))
      self.assertTrue(bool((np.asarray(pool.v_pages[layer]) == 0).all()))

  def test_scrubbing_zeroes_only_the_named_pages(self):
    pool = allocate_pool(_layout(num_pages=8, num_layers=1))
    k, v = poison_pages(pool.k_pages[0], pool.v_pages[0], [1, 2, 3])
    k, v = scrub_pages(k, v, [2])
    k, v = np.asarray(k), np.asarray(v)
    self.assertTrue(bool((k[2] == 0).all()))
    self.assertTrue(bool((k[1] == POISON_SENTINEL).all()))
    self.assertTrue(bool((v[3] == POISON_SENTINEL).all()))

  def test_scrubbing_nothing_is_a_no_op(self):
    pool = allocate_pool(_layout(num_pages=8, num_layers=1))
    k, v = scrub_pages(pool.k_pages[0], pool.v_pages[0], [])
    self.assertTrue(bool((np.asarray(k) == 0).all()))
    self.assertTrue(bool((np.asarray(v) == 0).all()))

  def test_scrubbing_every_layer_at_once_matches_the_per_layer_loop(self):
    """The batched scrub must be the per-layer loop's answer, not merely close.

    `scrub_pages_all_layers` exists for cost rather than for behaviour: the loop
    it replaces issued one dispatch per layer, which is eighty on a 70B model and
    lands on the critical path of every step that recycles a page. Cost changes
    are exactly where a behaviour change slips in unnoticed, so this pins the two
    against each other element by element rather than checking the batched one in
    isolation.
    """
    layers, pages = 5, 8
    batched = allocate_pool(_layout(num_pages=pages, num_layers=layers))
    looped = allocate_pool(_layout(num_pages=pages, num_layers=layers))
    for pool in (batched, looped):
      for layer in range(layers):
        k, v = poison_pages(pool.k_pages[layer], pool.v_pages[layer], [1, 2, 3, 4, 5])
        pool.replace_layer(layer, k, v)

    ks, vs = scrub_pages_all_layers(batched.k_pages, batched.v_pages, [2, 4, 5])
    for layer, (k, v) in enumerate(zip(ks, vs)):
      batched.replace_layer(layer, k, v)
    for layer in range(layers):
      k, v = scrub_pages(looped.k_pages[layer], looped.v_pages[layer], [2, 4, 5])
      looped.replace_layer(layer, k, v)

    for layer in range(layers):
      np.testing.assert_array_equal(
          np.asarray(batched.k_pages[layer]), np.asarray(looped.k_pages[layer])
      )
      np.testing.assert_array_equal(
          np.asarray(batched.v_pages[layer]), np.asarray(looped.v_pages[layer])
      )
    # And it really did scrub: named pages zeroed, unnamed ones left poisoned.
    scrubbed = np.asarray(batched.k_pages[layers - 1])
    self.assertTrue(bool((scrubbed[[2, 4, 5]] == 0).all()))
    self.assertTrue(bool((scrubbed[[1, 3]] == POISON_SENTINEL).all()))

  def test_scrubbing_every_layer_with_nothing_pending_is_a_no_op(self):
    """The common case on a pool that has not wrapped, and it must not dispatch."""
    pool = allocate_pool(_layout(num_pages=8, num_layers=3))
    ks, vs = scrub_pages_all_layers(pool.k_pages, pool.v_pages, [])
    self.assertEqual(len(ks), 3)
    self.assertEqual(len(vs), 3)
    for layer in range(3):
      self.assertTrue(bool((np.asarray(ks[layer]) == 0).all()))

  def test_an_odd_page_count_is_padded_idempotently(self):
    """Padding repeats a page already being written, so no mask is needed."""
    pool = allocate_pool(_layout(num_pages=8, num_layers=1))
    k, v = poison_pages(pool.k_pages[0], pool.v_pages[0], [1, 2, 3, 4, 5])
    k = np.asarray(k)
    for page in (1, 2, 3, 4, 5):
      self.assertTrue(bool((k[page] == POISON_SENTINEL).all()), f"page {page} not filled")
    for page in (0, 6, 7):
      self.assertTrue(bool((k[page] == 0).all()), f"page {page} was filled and should not have been")
    self.assertTrue(bool((np.asarray(v)[5] == POISON_SENTINEL).all()))


class StepInputsTest(unittest.TestCase):
  """Absolute positions, packing, and the ways a caller can present nonsense.

  This is where the position rule is locked down, because getting it wrong is not
  a crash: RoPE encodes absolute position, so a suffix rotated as though it began
  the sequence yields plausible text from a wrong computation. Cheap to pin here
  and expensive to notice anywhere else.
  """

  def _shape(self, *, num_requests=4, num_tokens=64, is_decode=False):
    return StepShape(
        num_requests=num_requests,
        num_tokens=num_tokens,
        num_pages=16,
        max_seqlen_k=128,
        is_decode=is_decode,
    )

  def test_a_fresh_prefill_starts_at_position_zero(self):
    context = np.arange(100, 110, dtype=np.int64)
    inputs = build_step_inputs(
        [RequestSlice(tokens=context[0:10], start=0, query_len=10)],
        self._shape(),
        is_decode=False,
    )
    self.assertEqual(inputs.tokens.shape, (1, 64))
    np.testing.assert_array_equal(inputs.tokens[0, :10], context)
    np.testing.assert_array_equal(inputs.positions[0, :10], np.arange(10))
    self.assertTrue(bool((inputs.segment_ids[0, :10] == 1).all()))
    # Sampling the last token it ran, indexed within the packed row.
    np.testing.assert_array_equal(inputs.sample_at, [9])
    np.testing.assert_array_equal(inputs.sample_rows, [0])

  def test_a_prefix_hit_feeds_the_suffix_at_absolute_positions(self):
    """The M5 hazard. Positions must not restart at zero for a cached prefix."""
    context = np.arange(200, 232, dtype=np.int64)  # 32 tokens
    inputs = build_step_inputs(
        [RequestSlice(tokens=context[16:32], start=16, query_len=16)],
        self._shape(),
        is_decode=False,
    )
    np.testing.assert_array_equal(inputs.tokens[0, :16], context[16:])
    np.testing.assert_array_equal(inputs.positions[0, :16], np.arange(16, 32))
    self.assertEqual(int(inputs.positions[0, 0]), 16, "the suffix must not be rotated as a prefix")

  def test_a_replay_after_preemption_feeds_the_longer_prompt(self):
    """Generated tokens are retained, so the replayed prompt is longer."""
    prompt, generated = np.arange(300, 310, dtype=np.int64), np.arange(400, 404, dtype=np.int64)
    context = np.concatenate([prompt, generated])
    inputs = build_step_inputs(
        [RequestSlice(tokens=context[0:14], start=0, query_len=14)],
        self._shape(),
        is_decode=False,
    )
    np.testing.assert_array_equal(inputs.tokens[0, :14], context)
    np.testing.assert_array_equal(inputs.positions[0, :14], np.arange(14))
    self.assertEqual(int(inputs.sample_at[0]), 13, "sampling follows the whole retained context")

  def test_batched_prefill_packs_requests_and_samples_each(self):
    """The capability the old single-row gather silently got wrong."""
    a = np.arange(10, 15, dtype=np.int64)  # 5 tokens
    b = np.arange(20, 27, dtype=np.int64)  # 7 tokens
    inputs = build_step_inputs(
        [
            RequestSlice(tokens=a, start=0, query_len=5),
            RequestSlice(tokens=b, start=0, query_len=7),
        ],
        self._shape(),
        is_decode=False,
    )
    np.testing.assert_array_equal(inputs.tokens[0, :5], a)
    np.testing.assert_array_equal(inputs.tokens[0, 5:12], b)
    # Each request's positions restart from its own start, not from the packed
    # offset: they are sequence positions, not row offsets.
    np.testing.assert_array_equal(inputs.positions[0, :5], np.arange(5))
    np.testing.assert_array_equal(inputs.positions[0, 5:12], np.arange(7))
    np.testing.assert_array_equal(inputs.sample_at, [4, 11], "one sample per request, within the row")
    np.testing.assert_array_equal(inputs.sample_rows, [0, 0], "a packed row samples from row zero")

  def test_prefill_pads_the_tail_inertly(self):
    inputs = build_step_inputs(
        [RequestSlice(tokens=np.arange(4, dtype=np.int64), start=0, query_len=4)],
        self._shape(num_tokens=64),
        is_decode=False,
    )
    self.assertTrue(bool((inputs.tokens[0, 4:] == 0).all()))
    self.assertTrue(bool((inputs.segment_ids[0, 4:] == 0).all()), "padded rows must be out of segment")

  def test_decode_batches_along_rows_and_carries_absolute_positions(self):
    # Two requests at *different* positions, which is the whole point of paging.
    short = np.arange(500, 512, dtype=np.int64)  # 12 tokens
    long = np.arange(600, 640, dtype=np.int64)  # 40 tokens
    inputs = build_step_inputs(
        [
            RequestSlice(tokens=short[11:12], start=11, query_len=1),
            RequestSlice(tokens=long[39:40], start=39, query_len=1),
        ],
        self._shape(num_requests=4, is_decode=True),
        is_decode=True,
    )
    self.assertEqual(inputs.tokens.shape, (4, 1))
    self.assertIsNone(inputs.segment_ids, "autoregressive mode refuses segment ids")
    self.assertEqual(int(inputs.tokens[0, 0]), 511, "decode feeds the last context token")
    self.assertEqual(int(inputs.tokens[1, 0]), 639)
    np.testing.assert_array_equal(inputs.positions[:2, 0], [11, 39])
    np.testing.assert_array_equal(inputs.sample_rows, np.arange(4))
    np.testing.assert_array_equal(inputs.sample_at, np.zeros(4))
    self.assertTrue(bool((inputs.tokens[2:] == 0).all()), "unused rows stay inert")

  def test_fewer_tokens_than_reserved_positions_is_rejected(self):
    """`query_len` is what the page table reserved; the tokens are what was assembled.

    Letting them disagree would leave the pool holding K/V for positions no query
    covered, so the two counts are required to match rather than the shorter one
    winning silently.
    """
    with self.assertRaises(ValueError):
      build_step_inputs(
          [RequestSlice(tokens=np.arange(5, dtype=np.int64), start=0, query_len=8)],
          self._shape(),
          is_decode=False,
      )

  def test_queries_exceeding_the_token_bucket_are_rejected(self):
    with self.assertRaises(ValueError):
      build_step_inputs(
          [RequestSlice(tokens=np.arange(100, dtype=np.int64), start=0, query_len=100)],
          self._shape(num_tokens=64),
          is_decode=False,
      )

  def test_more_requests_than_the_decode_bucket_are_rejected(self):
    slices = [
        RequestSlice(tokens=np.asarray([3], np.int64), start=3, query_len=1) for _ in range(5)
    ]
    with self.assertRaises(ValueError):
      build_step_inputs(slices, self._shape(num_requests=4, is_decode=True), is_decode=True)

  def test_a_multi_token_decode_is_rejected(self):
    with self.assertRaises(ValueError):
      build_step_inputs(
          [RequestSlice(tokens=np.arange(2, dtype=np.int64), start=4, query_len=2)],
          self._shape(is_decode=True),
          is_decode=True,
      )

  def test_an_empty_step_is_rejected(self):
    with self.assertRaises(ValueError):
      build_step_inputs([], self._shape(), is_decode=False)


class LayoutBuilderTest(unittest.TestCase):
  """Config and mesh into pool geometry."""

  def test_it_reads_the_model_dimensions(self):
    layout = build_storage_layout(_FakeConfig())
    self.assertEqual(layout.tokens_per_page, PAGE)
    self.assertEqual(layout.num_kv_heads, 4)
    self.assertEqual(layout.head_dim, 128)
    self.assertEqual(layout.num_layers, 8)

  def test_the_reserved_page_is_added_rather_than_taken_out_of_capacity(self):
    """paged_num_blocks is what the operator asked to be usable."""
    layout = build_storage_layout(_FakeConfig(paged_num_blocks=256))
    self.assertEqual(layout.num_pages, 257)
    self.assertEqual(layout.max_tokens(), 256 * PAGE)

  def test_shards_come_from_the_mesh_not_from_a_config_field(self):
    """Two sources of truth for TP width is how a pool ends up a factor too small."""
    mesh = _FakeMesh({"data": 2, "tensor": 4})
    self.assertEqual(kv_head_shards(mesh), 4)
    self.assertEqual(build_storage_layout(_FakeConfig(), mesh).kv_head_shards, 4)

  def test_multiple_tensor_axes_multiply(self):
    self.assertEqual(kv_head_shards(_FakeMesh({"tensor": 2, "tensor_sequence": 2})), 4)

  def test_no_mesh_means_unsharded(self):
    self.assertEqual(kv_head_shards(None), 1)

  def test_an_unset_pool_size_is_refused(self):
    with self.assertRaisesRegex(ValueError, "paged_num_blocks"):
      build_storage_layout(_FakeConfig(paged_num_blocks=0))


class _CountingStep:
  """A step function that records what it was handed."""

  def __init__(self, token=7):
    self.token = token
    self.views = []
    self.inputs = []

  def __call__(self, view, inputs, pool):
    del pool
    self.views.append(view)
    self.inputs.append(inputs)
    return np.full((view.num_active_requests,), self.token, dtype=np.int32)


class PagedDriverTest(unittest.TestCase):
  """Admission, scheduling, scrubbing and release, without a model."""

  def _driver(self, *, num_pages=65, max_requests=4, max_context_len=128, step=None, **kw):
    layout = _layout(num_pages=num_pages)
    plane = NativeKvControlPlane(
        layout=layout, max_requests=max_requests, max_context_len=max_context_len, debug_mode=True
    )
    pool = allocate_pool(layout)
    return PagedDriver(plane, pool, step or _CountingStep(), max_batch=max_requests, **kw)

  def _requests(self, count, prompt_len=20, max_new_tokens=8):
    # `prompt_tokens` is set on every request because the driver now needs them:
    # a step assembles the tokens it feeds the model, and inventing them would
    # turn a caller's omission into fluent output from garbage. Distinct ids per
    # request so nothing is accidentally a prefix of anything else.
    return [
        PagedRequest(
            request_id=f"r{i}",
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            prompt_tokens=np.arange(1000 * (i + 1), 1000 * (i + 1) + prompt_len, dtype=np.int64),
        )
        for i in range(count)
    ]

  def test_a_single_request_runs_to_its_length_cap(self):
    driver = self._driver()
    driver.submit(self._requests(1, prompt_len=20, max_new_tokens=5))
    done = driver.run()
    self.assertEqual(len(done), 1)
    self.assertEqual(len(done[0].generated), 5)
    self.assertEqual(done[0].finish_reason, "length")

  def test_an_eos_token_stops_generation_early(self):
    driver = self._driver(step=_CountingStep(token=99), eos_ids=(99,))
    driver.submit(self._requests(1, max_new_tokens=50))
    done = driver.run()
    self.assertEqual(len(done[0].generated), 1)
    self.assertEqual(done[0].finish_reason, "stop")

  def test_prefill_is_preferred_over_decode(self):
    """A deliberate policy, so it is worth pinning."""
    step = _CountingStep()
    driver = self._driver(step=step)
    driver.submit(self._requests(2))
    driver.step()
    self.assertFalse(step.views[0].shape.is_decode)

  def test_a_mixed_length_batch_completes_without_leaking(self):
    driver = self._driver(max_requests=4)
    lengths = [7, 19, 33, 48, 64, 5]
    driver.submit(
        [
            PagedRequest(
                request_id=f"r{i}",
                prompt_len=n,
                max_new_tokens=6,
                prompt_tokens=np.arange(1000 * (i + 1), 1000 * (i + 1) + n, dtype=np.int64),
            )
            for i, n in enumerate(lengths)
        ]
    )
    done = driver.run()

    self.assertEqual(len(done), len(lengths))
    self.assertTrue(all(len(r.generated) == 6 for r in done))
    self.assertEqual(driver.plane.allocator.num_allocated_pages, 0)
    self.assertEqual(driver.plane.allocator.available_pages, driver.plane.allocator.capacity_pages)
    self.assertEqual(driver.plane.num_live, 0)

  def test_the_driver_feeds_absolute_positions_in_both_phases(self):
    """The driver's half of the position rule, separated from assembly's half.

    `StepInputsTest` covers assembly given correct slices; this covers the driver
    producing correct slices. Split deliberately -- a test spanning both cannot
    say which half is wrong, and the two halves fail for different reasons.
    """
    step = _CountingStep(token=11)
    driver = self._driver(step=step)
    driver.submit(self._requests(1, prompt_len=20, max_new_tokens=3))
    driver.run()

    prefill, decodes = step.inputs[0], step.inputs[1:]
    context = np.arange(1000, 1020, dtype=np.int64)

    # Prefill runs the whole prompt from position zero and samples its last token.
    np.testing.assert_array_equal(prefill.tokens[0, :20], context)
    np.testing.assert_array_equal(prefill.positions[0, :20], np.arange(20))
    np.testing.assert_array_equal(prefill.sample_at, [19])

    # Each decode feeds the previous step's token at the next absolute position.
    # Position 20 is the first generated token's slot, so the first decode -- which
    # feeds that token back -- sits there, not at 21.
    for index, inputs in enumerate(decodes):
      self.assertEqual(int(inputs.tokens[0, 0]), 11, "decode feeds back what was generated")
      self.assertEqual(
          int(inputs.positions[0, 0]),
          20 + index,
          "decode positions must continue the sequence, not restart",
      )
      self.assertIsNone(inputs.segment_ids)

  def test_churn_traces_a_bounded_set_of_shapes(self):
    """The exit criterion, host side. Without bucketing this grows with the trace."""
    driver = self._driver(max_requests=8, max_context_len=256, num_pages=257)
    rng = np.random.default_rng(0)

    def churning(index):
      prompt_len = int(rng.integers(4, 120))
      return PagedRequest(
          request_id=f"r{index}",
          prompt_len=prompt_len,
          max_new_tokens=int(rng.integers(1, 30)),
          prompt_tokens=rng.integers(1, 30000, size=prompt_len, dtype=np.int64),
      )

    driver.submit([churning(i) for i in range(60)]
    )
    driver.run()
    self.assertLessEqual(driver.num_distinct_shapes, driver.planner.max_distinct_shapes())
    self.assertLessEqual(driver.num_distinct_shapes, 20, "many more shapes than rungs means bucketing broke")
    self.assertEqual(driver.plane.allocator.available_pages, driver.plane.allocator.capacity_pages)

  def test_a_pool_under_pressure_preempts_rather_than_deadlocking(self):
    """Recompute preemption: work is lost, but the loop always progresses."""
    # 8 usable pages of 16 tokens, four concurrent requests each wanting 55.
    driver = self._driver(num_pages=9, max_requests=4, max_context_len=64)
    driver.submit(
        [
            PagedRequest(
                request_id=f"r{i}",
                prompt_len=30,
                max_new_tokens=25,
                prompt_tokens=np.arange(1000 * (i + 1), 1000 * (i + 1) + 30, dtype=np.int64),
            )
            for i in range(5)
        ]
    )
    done = driver.run()

    self.assertEqual(len(done), 5)
    self.assertTrue(all(len(r.generated) == 25 for r in done))
    self.assertGreater(sum(r.preemptions for r in done), 0, "this pool is too small not to preempt")
    self.assertEqual(driver.plane.allocator.num_allocated_pages, 0)

  def test_a_preempted_request_keeps_the_tokens_it_generated(self):
    """Preemption replays a longer prompt; it does not discard output."""
    driver = self._driver(num_pages=9, max_requests=4, max_context_len=64)
    driver.submit(
        [
            PagedRequest(
                request_id=f"r{i}",
                prompt_len=30,
                max_new_tokens=25,
                prompt_tokens=np.arange(1000 * (i + 1), 1000 * (i + 1) + 30, dtype=np.int64),
            )
            for i in range(5)
        ]
    )
    done = driver.run()
    preempted = [r for r in done if r.preemptions]
    self.assertTrue(preempted)
    for request in preempted:
      self.assertEqual(len(request.generated), 25)

  def test_recycled_pages_are_scrubbed_before_the_step_reads_them(self):
    """The driver's half of the acceptance gate.

    A pool small enough to wrap forces recycling. If the driver failed to scrub,
    the control plane would refuse to build the table and this would raise.
    """
    driver = self._driver(num_pages=6, max_requests=2, max_context_len=64)
    driver.submit(self._requests(8, prompt_len=20, max_new_tokens=3))
    done = driver.run()
    self.assertEqual(len(done), 8)
    self.assertEqual(driver.plane.pending_scrub().size, 0)

  def test_poisoning_leaves_pages_dirty_so_they_must_still_be_scrubbed(self):
    """Poison is a detector, not a substitute for zeroing."""
    driver = self._driver(num_pages=6, max_requests=2, max_context_len=64, poison_on_free=True)
    driver.submit(self._requests(2, prompt_len=20, max_new_tokens=2))
    driver.run()
    self.assertGreater(driver.plane.allocator.num_dirty_pages, 0)

  def test_run_reports_a_stalled_loop_rather_than_returning_partial_output(self):
    driver = self._driver()
    driver.submit(self._requests(4, max_new_tokens=100))
    with self.assertRaisesRegex(RuntimeError, "no progress"):
      driver.run(max_steps=3)

  def test_an_empty_driver_does_nothing(self):
    self.assertIsNone(self._driver().step())
    self.assertEqual(self._driver().run(), [])


class DriverPrefixCacheTest(unittest.TestCase):
  """The driver's use of the prefix index: fewer prefill tokens, no lost pages.

  The index itself is covered in `kv_prefix_cache_test.py`. What is at stake here
  is the wiring -- that the saving reaches the step's query length, and that the
  two page-lifetime hazards sharing introduces are handled.
  """

  def _driver(self, *, num_pages=65, max_requests=4, max_context_len=128, step=None, **kw):
    layout = _layout(num_pages=num_pages)
    plane = NativeKvControlPlane(
        layout=layout,
        max_requests=max_requests,
        max_context_len=max_context_len,
        debug_mode=True,
        enable_prefix_cache=True,
    )
    return PagedDriver(plane, allocate_pool(layout), step or _CountingStep(), max_batch=max_requests, **kw)

  def _request(self, request_id, prompt, max_new_tokens=2):
    return PagedRequest(
        request_id=request_id,
        prompt_len=len(prompt),
        max_new_tokens=max_new_tokens,
        prompt_tokens=np.asarray(prompt, dtype=np.int64),
    )

  def test_a_repeated_prompt_prefills_fewer_tokens(self):
    """The milestone's whole point, measured as tokens the step actually computes.

    Not as the bucketed shape: the token ladder floors at 64, so a saving smaller
    than that is real but invisible there. The prompt here is long enough that
    both figures move, and both are checked -- the token count because it is the
    work avoided, the bucket because it is what the work costs.
    """
    step = _CountingStep()
    driver = self._driver(step=step, max_requests=1, max_context_len=512)
    prompt = list(range(256))

    driver.submit([self._request("first", prompt)])
    first = driver.step()
    driver.run()
    first_bucket = step.views[0].shape.num_tokens

    step.views.clear()
    driver.submit([self._request("second", prompt)])
    second = driver.step()

    self.assertEqual(first.num_tokens, 256)
    self.assertEqual(second.num_tokens, 16, "only the held-back final page should be recomputed")
    self.assertLess(step.views[0].shape.num_tokens, first_bucket)

  def test_a_diverging_prompt_shares_only_what_it_has_in_common(self):
    driver = self._driver(max_requests=1, max_context_len=512)
    shared = list(range(128))

    driver.submit([self._request("first", shared + list(range(900, 964)))])
    driver.run()

    driver.submit([self._request("second", shared + list(range(500, 564)))])
    outcome = driver.step()
    self.assertEqual(
        outcome.num_tokens, 64, "exactly the divergent tail should be computed, and the shared 128 skipped"
    )

  def test_submit_rejects_a_request_without_tokens(self):
    """Replaces an earlier test asserting the opposite, and the change is deliberate.

    That test read "supplying prompt ids is opt-in, so the cache must tolerate
    their absence", which was true when `prompt_tokens` existed only to offer the
    prefix cache something to match. The driver now assembles the tokens it feeds
    the model, so a request without them cannot run at all.

    Rejected at `submit` rather than at the step that needs them: by then the pool
    holds pages for the request and other requests have been scheduled around it,
    so the caller learns too late to do anything useful. Note this does not make
    the prefix cache mandatory -- that is still a control-plane switch.
    """
    driver = self._driver()
    with self.assertRaises(ValueError) as caught:
      driver.submit(
          [PagedRequest(request_id=f"r{i}", prompt_len=20, max_new_tokens=4) for i in range(3)]
      )
    self.assertIn("prompt_tokens", str(caught.exception))
    # Named, so a caller with a large batch can find the offenders.
    self.assertIn("r0", str(caught.exception))
    self.assertEqual(driver.num_waiting, 0, "a rejected batch must not be partially queued")

  def test_only_pages_with_computed_kv_are_published(self):
    """The final generated token has no K/V: its step never ran.

    Publishing it would cache a page whose tail is whatever the scrub left, and
    the next request to match that prefix would attend over zeros as though they
    were real keys.
    """
    driver = self._driver(max_requests=1)
    prompt = list(range(16))
    driver.submit([self._request("r0", prompt, max_new_tokens=5)])
    done = driver.run()

    written = len(prompt) + len(done[0].generated) - 1
    self.assertLessEqual(
        driver.plane.prefix_index.num_cached_pages * PAGE,
        written,
        "published more tokens than had their K/V computed",
    )

  def test_nothing_leaks_once_the_cache_is_dropped(self):
    driver = self._driver(max_requests=2)
    prompts = [list(range(i * 100, i * 100 + 48)) for i in range(6)]
    driver.submit([self._request(f"r{i}", p, max_new_tokens=3) for i, p in enumerate(prompts)])
    driver.run()

    driver.plane.evict_cached(driver.plane.prefix_index.num_cached_pages)
    self.assertEqual(driver.plane.prefix_index.num_cached_pages, 0)
    self.assertEqual(driver.plane.allocator.num_allocated_pages, 0)
    self.assertEqual(driver.plane.allocator.available_pages, driver.plane.allocator.capacity_pages)

  def test_the_cache_does_not_stop_a_tight_pool_making_progress(self):
    """Retained pages must be reclaimable, or sharing turns into a deadlock."""
    driver = self._driver(num_pages=9, max_requests=4, max_context_len=64)
    driver.submit(
        [self._request(f"r{i}", list(range(i * 100, i * 100 + 30)), max_new_tokens=25) for i in range(5)]
    )
    done = driver.run()
    self.assertEqual(len(done), 5)
    self.assertTrue(all(len(r.generated) == 25 for r in done))

  def test_poison_on_free_does_not_destroy_a_retained_page(self):
    """Poison must follow what was freed, not what the request held."""
    driver = self._driver(max_requests=1, poison_on_free=True)
    prompt = list(range(64))
    driver.submit([self._request("first", prompt)])
    driver.run()

    cached = driver.plane.prefix_index.match(prompt, CacheNamespace()).pages
    self.assertGreater(cached.size, 0)
    k = np.asarray(driver.pool.k_pages[0])[cached]
    self.assertFalse(np.any(k == POISON_SENTINEL), "poisoned a page the prefix cache had adopted")

  def test_a_preempted_request_does_not_retain_its_pages(self):
    """Preemption exists to free pages; publishing them would work against it.

    Checked at the first preemption rather than at the end of the run, because by
    then a completed request has published legitimately and the two sources of
    cached pages are no longer distinguishable.
    """
    driver = self._driver(num_pages=9, max_requests=4, max_context_len=64)
    requests = [
        self._request(f"r{i}", list(range(i * 100, i * 100 + 30)), max_new_tokens=25) for i in range(5)
    ]
    driver.submit(requests)
    while not any(r.preemptions for r in requests):
      if driver.step() is None:
        self.fail("this pool is too small not to preempt")

    self.assertFalse(any(r.is_finished for r in requests), "no request should have completed this early")
    self.assertEqual(
        driver.plane.prefix_index.num_cached_pages,
        0,
        "a preempted request published its pages, so the preemption reclaimed fewer than it should",
    )

  def test_the_only_pages_left_allocated_are_the_ones_the_cache_holds(self):
    """The leak invariant, restated for a runtime that deliberately retains pages."""
    driver = self._driver(num_pages=9, max_requests=4, max_context_len=64)
    driver.submit(
        [self._request(f"r{i}", list(range(i * 100, i * 100 + 30)), max_new_tokens=25) for i in range(5)]
    )
    driver.run()
    self.assertEqual(
        driver.plane.allocator.num_allocated_pages, driver.plane.prefix_index.num_cached_pages
    )


class PagedRuntimeTest(unittest.TestCase):
  """The engine adapter, and the slot shim over the request-based API."""

  def _runtime(self):
    layout = _layout()
    plane = NativeKvControlPlane(layout=layout, max_requests=4, max_context_len=128, debug_mode=True)
    return PagedRuntime(plane, allocate_pool(layout)), plane

  def _admit(self, plane, request_id="r0", prompt_len=20):
    from maxtext.inference.kv_control import RequestDescriptor  # pylint: disable=import-outside-toplevel

    handle = plane.admit(RequestDescriptor(request_id=request_id, prompt_len=prompt_len, max_new_tokens=4))
    plane.reserve([handle], [prompt_len])
    return handle

  def test_release_by_handle_reclaims_the_pages(self):
    runtime, plane = self._runtime()
    handle = self._admit(plane, prompt_len=33)
    runtime.track(handle)
    self.assertEqual(runtime.release(handle).size, 3)
    self.assertEqual(plane.allocator.num_allocated_pages, 0)

  def test_the_slot_shim_reaches_the_same_pages(self):
    """What keeps the three legacy release_pages call sites working."""
    runtime, plane = self._runtime()
    handle = self._admit(plane, prompt_len=33)
    runtime.track(handle, slot=5)
    self.assertEqual(runtime.release_slot(5).size, 3)
    self.assertEqual(plane.allocator.num_allocated_pages, 0)

  def test_an_unknown_slot_is_a_no_op(self):
    """The legacy call sites fire on termination and do not coordinate."""
    runtime, _ = self._runtime()
    self.assertEqual(runtime.release_slot(11).size, 0)

  def test_a_second_release_through_the_adapter_is_absorbed(self):
    runtime, plane = self._runtime()
    handle = self._admit(plane)
    runtime.track(handle, slot=0)
    runtime.release(handle)
    self.assertEqual(runtime.release(handle).size, 0)
    self.assertEqual(runtime.release_slot(0).size, 0)

  def test_a_handle_from_another_epoch_is_not_honoured(self):
    runtime, plane = self._runtime()
    handle = self._admit(plane)
    runtime.track(handle)
    stale = RequestHandle(request_id=handle.request_id, row=handle.row, epoch=handle.epoch + 1)
    self.assertEqual(runtime.release(stale).size, 0)

  def test_handles_are_findable_by_request_id(self):
    runtime, plane = self._runtime()
    handle = self._admit(plane, request_id="abc")
    runtime.track(handle, slot=2)
    self.assertEqual(runtime.handle_for_request("abc"), handle)
    self.assertEqual(runtime.handle_for_slot(2), handle)
    runtime.release(handle)
    self.assertIsNone(runtime.handle_for_request("abc"))
    self.assertIsNone(runtime.handle_for_slot(2))


if __name__ == "__main__":
  unittest.main()
