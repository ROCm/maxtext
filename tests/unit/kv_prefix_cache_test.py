"""Prefix sharing: the index, the namespace, and their use by the control plane.

Host-only and CPU-only, like the rest of `kv_control`. Nothing here allocates a
device array, so the whole file runs in under a second and can be a pre-commit
check rather than a nightly one.

The tests that matter most are the ones asserting a *negative*: that a namespace
mismatch cannot hit, that a shared page is never written, and that a page a live
request is reading cannot be evicted. Those are the failures which produce
plausible tokens instead of a crash, so they are the ones worth spending test
surface on.

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

import dataclasses
import unittest

import numpy as np

from maxtext.inference.kv_common import CacheNamespace, KvStorageLayoutV1
from maxtext.inference.kv_control import (
    NativeKvControlPlane,
    PrefixIndex,
    RequestDescriptor,
    SharedPageWriteError,
    block_hash,
)

PAGE = 4
NS = CacheNamespace(model_fingerprint="sha256:abc", tokenizer="llama3")


def _layout(**kw) -> KvStorageLayoutV1:
  base = {
      "tokens_per_page": PAGE,
      "num_pages": 64,
      "num_layers": 4,
      "num_kv_heads": 8,
      "head_dim": 128,
      "dtype": "bfloat16",
  }
  base.update(kw)
  return KvStorageLayoutV1(**base)


def _tokens(n, start=0):
  return list(range(start, start + n))


class CacheNamespaceTest(unittest.TestCase):
  """The identity that decides whether a hit is sound."""

  def test_the_digest_covers_every_field_without_being_told_them(self):
    """The property that survives someone adding a field and forgetting this test.

    Enumerating the dataclass rather than hard-coding names is what makes the
    digest exhaustive by construction. This asserts the mechanism, so a field
    added later is covered whether or not anyone writes a test for it.
    """
    baseline = CacheNamespace()
    for field in dataclasses.fields(baseline):
      value = 99 if field.name == "version" else f"changed-{field.name}"
      varied = dataclasses.replace(baseline, **{field.name: value})
      self.assertNotEqual(
          baseline.digest(),
          varied.digest(),
          f"changing {field.name} alone left the digest unchanged, so two configurations that "
          f"differ only in {field.name} would share cached K/V",
      )

  def test_the_digest_is_stable_across_instances(self):
    self.assertEqual(CacheNamespace(tenant="a").digest(), CacheNamespace(tenant="a").digest())

  def test_field_names_are_hashed_so_a_value_cannot_migrate_between_fields(self):
    """Two fields holding each other's values must not digest alike."""
    swapped = CacheNamespace(adapter="x", tenant="y")
    other = CacheNamespace(adapter="y", tenant="x")
    self.assertNotEqual(swapped.digest(), other.digest())

  def test_describe_names_the_fields_that_are_set(self):
    text = CacheNamespace(tenant="acme", adapter="lora-7").describe()
    self.assertIn("tenant=acme", text)
    self.assertIn("adapter=lora-7", text)
    self.assertNotIn("tokenizer", text)


class BlockHashTest(unittest.TestCase):
  """The chain that makes a page's key mean "after exactly this history"."""

  def test_the_same_page_after_a_different_prefix_hashes_differently(self):
    page = [7, 8, 9, 10]
    self.assertNotEqual(block_hash(b"parent-a", page), block_hash(b"parent-b", page))

  def test_the_chain_is_order_sensitive(self):
    self.assertNotEqual(
        block_hash(block_hash(b"", [1, 2]), [3, 4]),
        block_hash(block_hash(b"", [3, 4]), [1, 2]),
    )


class PrefixIndexTest(unittest.TestCase):
  """Matching, publication, refcounts and eviction, without a control plane."""

  def _index(self, enabled=True):
    return PrefixIndex(tokens_per_page=PAGE, enabled=enabled)

  def test_a_cold_index_matches_nothing(self):
    self.assertFalse(self._index().match(_tokens(16), NS))

  def test_publish_then_match_returns_the_same_pages(self):
    index = self._index()
    tokens = _tokens(16)
    result = index.publish(tokens, [10, 11, 12, 13], NS)
    self.assertEqual(result.adopted.tolist(), [10, 11, 12, 13])

    match = index.match(tokens, NS)
    # The last page is held back so the request has something left to compute.
    self.assertEqual(match.pages.tolist(), [10, 11, 12])
    self.assertEqual(match.num_tokens, 12)

  def test_a_fully_cached_prompt_still_has_a_page_left_to_compute(self):
    """Otherwise the step has no query tokens and nothing to predict from.

    The held-back page is the difference between "this prompt is cached" and
    "this prompt needs no work", and only the first of those is true.
    """
    index = self._index()
    tokens = _tokens(16)
    index.publish(tokens, [10, 11, 12, 13], NS)
    match = index.match(tokens, NS)
    self.assertEqual(match.num_tokens, 12)
    self.assertLess(match.num_tokens, len(tokens))

  def test_a_partial_page_is_never_published(self):
    """Sharing a page still being appended to would give two writers one page."""
    index = self._index()
    result = index.publish(_tokens(10), [10, 11, 12], NS)
    self.assertEqual(result.adopted.tolist(), [10, 11])

  def test_only_computed_tokens_are_published(self):
    index = self._index()
    result = index.publish(_tokens(16), [10, 11, 12, 13], NS, num_valid_tokens=9)
    self.assertEqual(result.adopted.tolist(), [10, 11])

  def test_a_longer_prompt_matches_the_shared_prefix_and_stops(self):
    index = self._index()
    index.publish(_tokens(16), [10, 11, 12, 13], NS)
    match = index.match(_tokens(12) + [900, 901, 902, 903] + _tokens(4), NS)
    self.assertEqual(match.pages.tolist(), [10, 11, 12])

  def test_a_divergent_prompt_matches_only_up_to_the_divergence(self):
    index = self._index()
    index.publish(_tokens(16), [10, 11, 12, 13], NS)
    diverged = _tokens(8) + [500, 501, 502, 503] + _tokens(4, 12)
    self.assertEqual(index.match(diverged, NS).pages.tolist(), [10, 11])

  def test_a_second_request_donating_the_same_content_keeps_the_original(self):
    index = self._index()
    index.publish(_tokens(16), [10, 11, 12, 13], NS)
    again = index.publish(_tokens(16), [20, 21, 22, 23], NS)
    self.assertEqual(again.adopted.size, 0)
    self.assertEqual(again.duplicate.tolist(), [20, 21, 22, 23])
    self.assertEqual(index.match(_tokens(16), NS).pages.tolist(), [10, 11, 12])

  def test_a_branch_shares_its_common_prefix(self):
    index = self._index()
    index.publish(_tokens(8) + _tokens(8, 100), [10, 11, 12, 13], NS)
    index.publish(_tokens(8) + _tokens(8, 200), [10, 11, 30, 31], NS)
    self.assertEqual(index.num_cached_pages, 6)

  def test_the_index_is_inert_when_disabled(self):
    index = self._index(enabled=False)
    self.assertEqual(index.publish(_tokens(16), [10, 11, 12, 13], NS).adopted.size, 0)
    self.assertFalse(index.match(_tokens(16), NS))

  def test_publishing_more_tokens_than_pages_is_refused(self):
    with self.assertRaises(ValueError):
      self._index().publish(_tokens(16), [10, 11], NS)


class NamespaceIsolationTest(unittest.TestCase):
  """Varying one namespace field alone must defeat the match.

  The plan asks for a negative test per field, and generating them from the
  dataclass rather than writing twelve near-identical methods means a field
  added later is covered on the day it is added.
  """

  def test_every_field_in_isolation_defeats_a_hit(self):
    tokens = _tokens(16)
    for field in dataclasses.fields(CacheNamespace):
      with self.subTest(field=field.name):
        index = PrefixIndex(tokens_per_page=PAGE)
        index.publish(tokens, [10, 11, 12, 13], NS)
        self.assertTrue(index.match(tokens, NS), "the control case should hit")

        value = 99 if field.name == "version" else f"other-{field.name}"
        varied = dataclasses.replace(NS, **{field.name: value})
        self.assertFalse(
            index.match(tokens, varied),
            f"a request differing only in {field.name} was served another configuration's K/V",
        )

  def test_two_namespaces_coexist_without_evicting_each_other(self):
    index = PrefixIndex(tokens_per_page=PAGE)
    other = dataclasses.replace(NS, tenant="second")
    index.publish(_tokens(16), [10, 11, 12, 13], NS)
    index.publish(_tokens(16), [20, 21, 22, 23], other)
    self.assertEqual(index.num_cached_pages, 8)
    self.assertEqual(index.match(_tokens(16), NS).pages.tolist(), [10, 11, 12])
    self.assertEqual(index.match(_tokens(16), other).pages.tolist(), [20, 21, 22])


class RefCountAndEvictionTest(unittest.TestCase):
  """What may be dropped, and what may not be dropped at any pressure."""

  def _loaded(self):
    """Two unrelated three-page sequences: pages 10-12 and 20-22."""
    index = PrefixIndex(tokens_per_page=PAGE)
    index.publish(_tokens(12), [10, 11, 12], NS)
    index.publish(_tokens(12, 100), [20, 21, 22], NS)
    return index

  def test_eviction_follows_recency_across_unrelated_sequences(self):
    """Using one sequence should cost the other its pages.

    Tail pages go first in both orderings, because a tail is never part of a
    match -- it is held back by design, so it really is the least useful page in
    the index rather than merely the coldest by accident.
    """
    index = self._loaded()
    index.match(_tokens(12), NS)
    self.assertEqual(index.evict(2).tolist(), [12, 22])

  def test_pressure_drains_the_untouched_sequence_in_full_first(self):
    """A page orphaned by eviction keeps its own recency, not its child's.

    So the cold sequence is reclaimed end to end -- deepest page first, since a
    parent only becomes eligible once its children are gone -- before the
    recently matched one gives up anything.
    """
    index = self._loaded()
    index.match(_tokens(12, 100), NS)
    self.assertEqual(index.evict(3).tolist(), [12, 11, 10])
    self.assertEqual(index.match(_tokens(12, 100), NS).num_pages, 2)

  def test_eviction_never_orphans_a_prefix(self):
    """A parent is only reachable for eviction once its children are gone."""
    index = PrefixIndex(tokens_per_page=PAGE)
    index.publish(_tokens(12), [10, 11, 12], NS)
    self.assertEqual(index.evict(2).tolist(), [12, 11])
    self.assertEqual(index.num_cached_pages, 1)

  def test_a_referenced_path_survives_maximum_pressure(self):
    index = self._loaded()
    index.acquire(index.match(_tokens(12), NS).node)
    self.assertEqual(index.protected_pages, 2)
    self.assertEqual(sorted(index.evict(100).tolist()), [12, 20, 21, 22])
    self.assertEqual(index.num_cached_pages, 2)

  def test_releasing_makes_the_path_evictable_again(self):
    index = self._loaded()
    node = index.match(_tokens(12), NS).node
    index.acquire(node)
    index.release(node)
    self.assertEqual(index.protected_pages, 0)
    self.assertEqual(index.evict(100).size, 6)

  def test_over_release_is_an_error_rather_than_a_silent_unprotect(self):
    index = self._loaded()
    node = index.match(_tokens(12), NS).node
    index.acquire(node)
    index.release(node)
    with self.assertRaises(RuntimeError):
      index.release(node)

  def test_two_readers_of_one_prefix_both_have_to_leave(self):
    index = self._loaded()
    node = index.match(_tokens(12), NS).node
    index.acquire(node)
    index.acquire(node)
    index.release(node)
    self.assertEqual(index.protected_pages, 2)
    self.assertEqual(index.num_cached_pages - index.evict(100).size, 2)

  def test_reset_refuses_while_a_request_still_reads(self):
    index = self._loaded()
    index.acquire(index.match(_tokens(12), NS).node)
    with self.assertRaises(RuntimeError):
      index.reset()

  def test_reset_hands_back_every_page(self):
    index = self._loaded()
    self.assertEqual(sorted(index.reset().tolist()), [10, 11, 12, 20, 21, 22])
    self.assertEqual(index.num_cached_pages, 0)


class ControlPlaneWithPrefixCacheTest(unittest.TestCase):
  """The end-to-end property: a repeated prompt does less work."""

  def _plane(self, num_pages=32, max_context_len=64, **kw):
    return NativeKvControlPlane(
        layout=_layout(num_pages=num_pages),
        max_requests=4,
        max_context_len=max_context_len,
        debug_mode=True,
        enable_prefix_cache=True,
        **kw,
    )

  def _run(self, plane, request_id, tokens, namespace=NS):
    """Admit, attach any cached prefix, prefill the rest, release.

    The whole point of the milestone in six lines: the step's query length is the
    prompt minus whatever the cache supplied, and everything downstream follows
    from that one subtraction.
    """
    handle = plane.admit(
        RequestDescriptor(request_id=request_id, prompt_len=len(tokens), max_new_tokens=0)
    )
    match = plane.attach_prefix(handle, tokens, namespace)
    to_prefill = len(tokens) - match.num_tokens
    self.assertTrue(plane.reserve([handle], [to_prefill]))
    plane.confirm_scrubbed(plane.pending_scrub())
    table = plane.build_page_table([handle], [to_prefill])
    plane.release(handle, tokens)
    return to_prefill, table

  def test_a_repeated_prompt_skips_the_prefill_it_already_paid_for(self):
    plane = self._plane()
    tokens = _tokens(16)
    first, _ = self._run(plane, "r0", tokens)
    second, _ = self._run(plane, "r1", tokens)
    self.assertEqual(first, 16)
    self.assertEqual(second, 4, "the cached 12 tokens should not have been prefilled again")

  def test_a_shared_page_never_appears_in_a_slot_mapping(self):
    """The M5 exit criterion, checked against the array a kernel actually writes."""
    plane = self._plane()
    tokens = _tokens(16)
    self._run(plane, "r0", tokens)

    handle = plane.admit(RequestDescriptor(request_id="r1", prompt_len=16, max_new_tokens=0))
    match = plane.attach_prefix(handle, tokens, NS)
    self.assertEqual(match.num_pages, 3)
    self.assertTrue(plane.reserve([handle], [16 - match.num_tokens]))
    plane.confirm_scrubbed(plane.pending_scrub())
    table = plane.build_page_table([handle], [16 - match.num_tokens])

    slots = table.slot_mapping(PAGE)
    written = set((slots[slots >= 0] // PAGE).tolist())
    self.assertTrue(
        written.isdisjoint(set(match.pages.tolist())),
        f"step wrote into shared pages {written & set(match.pages.tolist())}",
    )
    self.assertEqual(set(match.pages.tolist()) - set(table.flat_page_indices().tolist()), set())

  def test_the_debug_gate_catches_a_query_length_that_would_overwrite_a_prefix(self):
    """A wrong query length is the only way a shared page can be written."""
    plane = self._plane()
    tokens = _tokens(16)
    self._run(plane, "r0", tokens)

    handle = plane.admit(RequestDescriptor(request_id="r1", prompt_len=16, max_new_tokens=0))
    plane.attach_prefix(handle, tokens, NS)
    self.assertTrue(plane.reserve([handle], [4]))
    plane.confirm_scrubbed(plane.pending_scrub())
    with self.assertRaises(SharedPageWriteError):
      # Claiming the whole prompt as this step's query, as a caller would if it
      # ignored the match, walks write positions back over the shared pages.
      plane.build_page_table([handle], [16])

  def test_a_different_namespace_pays_full_price(self):
    plane = self._plane()
    tokens = _tokens(16)
    self._run(plane, "r0", tokens)
    other, _ = self._run(plane, "r1", tokens, namespace=dataclasses.replace(NS, adapter="lora-2"))
    self.assertEqual(other, 16)

  def test_cached_pages_stay_allocated_after_the_request_leaves(self):
    plane = self._plane()
    before = plane.allocator.available_pages
    self._run(plane, "r0", _tokens(16))
    self.assertEqual(plane.prefix_index.num_cached_pages, 4)
    self.assertEqual(plane.allocator.available_pages, before - 4)

  def test_pressure_evicts_the_cache_rather_than_refusing_to_serve(self):
    """A cache must never be the reason a request cannot run."""
    plane = self._plane(num_pages=10, max_context_len=32)
    self._run(plane, "r0", _tokens(16))
    self.assertEqual(plane.prefix_index.num_cached_pages, 4)

    handle = plane.admit(RequestDescriptor(request_id="big", prompt_len=32, max_new_tokens=0))
    self.assertTrue(plane.reserve([handle], [32]))
    self.assertLess(plane.prefix_index.num_cached_pages, 4)

  def test_a_borrowed_page_is_not_freed_when_the_borrower_leaves(self):
    plane = self._plane()
    tokens = _tokens(16)
    self._run(plane, "r0", tokens)
    cached = plane.prefix_index.match(tokens, NS).pages

    handle = plane.admit(RequestDescriptor(request_id="r1", prompt_len=16, max_new_tokens=0))
    plane.attach_prefix(handle, tokens, NS)
    self.assertTrue(plane.reserve([handle], [4]))
    freed = plane.release(handle, tokens)
    self.assertTrue(
        set(freed.tolist()).isdisjoint(set(cached.tolist())),
        "released a page the index still owns and other requests will read",
    )

  def test_no_pages_leak_across_repeated_churn(self):
    plane = self._plane()
    before = plane.allocator.available_pages
    for i in range(20):
      self._run(plane, f"r{i}", _tokens(16, start=i * 100))
    plane.evict_cached(plane.prefix_index.num_cached_pages)
    self.assertEqual(plane.allocator.available_pages, before)

  def test_attaching_a_prefix_to_a_started_request_is_refused(self):
    plane = self._plane()
    tokens = _tokens(16)
    self._run(plane, "r0", tokens)
    handle = plane.admit(RequestDescriptor(request_id="r1", prompt_len=16, max_new_tokens=0))
    self.assertTrue(plane.reserve([handle], [8]))
    with self.assertRaises(ValueError):
      plane.attach_prefix(handle, tokens, NS)

  def test_the_cache_is_off_unless_asked_for(self):
    plane = NativeKvControlPlane(
        layout=_layout(), max_requests=2, max_context_len=64, debug_mode=True
    )
    self.assertFalse(plane.prefix_cache_enabled)
    handle = plane.admit(RequestDescriptor(request_id="r0", prompt_len=16, max_new_tokens=0))
    self.assertTrue(plane.reserve([handle], [16]))
    freed = plane.release(handle, _tokens(16))
    self.assertEqual(freed.size, 4, "nothing should have been retained by a disabled cache")


if __name__ == "__main__":
  unittest.main()
