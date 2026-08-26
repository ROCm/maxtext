"""Sharing pages between requests that begin with the same tokens.

The K/V for a token depends on that token and everything before it, so two
requests with a common prefix compute bit-identical K/V for it. A shared system
prompt or a multi-turn conversation replays the same leading tokens on every
turn, and prefill is quadratic in prompt length, so not recomputing that prefix
is the largest single win available to a paged runtime.

The index is a trie of pages. Each node owns one page and is keyed by a hash
chained from its parent, so a node's key identifies not just its own tokens but
the entire path that produced them -- which is exactly the condition under which
its K/V is valid. The chain starts at the namespace digest rather than at a
constant, so two configurations do not share a root and a namespace mismatch is
structurally unable to produce a hit. See `kv_common/namespace.py`.

Three decisions differ from the sglang-jax radix cache this follows in outline:

**Nodes are one page, not a variable-length token run.** That removes node
splitting entirely, which is most of the reference's complexity, and costs
nothing here because only whole pages are ever published.

**Only full pages are published.** A partial page is still being appended to, so
sharing it would mean two requests writing different tokens into the same page.
The tail is left private and recomputed, which is at most `tokens_per_page - 1`
tokens of duplicated prefill.

**Recency is a counter, not a clock.** `time.monotonic()` ties at clock
resolution when many nodes are touched in one step, which makes eviction order
depend on how the heap happened to break the tie. A counter cannot tie, so
eviction order is reproducible and a test can assert against it.

Stored tokens are compared on a hash hit. A 256-bit chained hash makes collision
a non-argument, but the comparison is a page of integers against a cached page of
integers, and what it buys is that the one bug class here which produces wrong
output rather than a crash cannot occur at all.

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
import hashlib
import heapq
from typing import Iterable, Sequence

import numpy as np

from maxtext.inference.kv_common import CacheNamespace

_EMPTY_PAGES = np.empty((0,), dtype=np.int32)


def block_hash(parent_hash: bytes, token_ids: Sequence[int]) -> bytes:
  """Hash one page's tokens into the chain ending at `parent_hash`.

  Chaining rather than hashing the page alone is what makes a node's key mean
  "these tokens, at this position, after exactly this history". Hashing the page
  in isolation would let a page match at the wrong depth or after a different
  prefix, and its K/V would be wrong in both cases.
  """
  hasher = hashlib.blake2b(digest_size=32)
  hasher.update(parent_hash)
  hasher.update(np.asarray(token_ids, dtype=np.int64).tobytes())
  return hasher.digest()


@dataclasses.dataclass(eq=False)
class PrefixNode:
  """One cached page, and the path that produced it."""

  block_hash: bytes
  page_id: int
  tokens: np.ndarray
  depth: int
  parent: "PrefixNode | None"
  children: dict[bytes, "PrefixNode"] = dataclasses.field(default_factory=dict)
  ref_count: int = 0
  last_access: int = 0
  hit_count: int = 0

  @property
  def is_leaf(self) -> bool:
    return not self.children


@dataclasses.dataclass(frozen=True)
class PrefixMatch:
  """What a lookup found. `pages` are readable but must never be written."""

  pages: np.ndarray
  num_tokens: int
  node: PrefixNode | None

  @property
  def num_pages(self) -> int:
    return int(self.pages.size)

  def __bool__(self) -> bool:
    return self.pages.size > 0


@dataclasses.dataclass(frozen=True)
class PublishResult:
  """The outcome of offering a request's pages to the index.

  `adopted` are now owned by the index and must not be freed by the caller.
  `duplicate` held content some other request had already cached; the index kept
  what it had, so these are the caller's to free. Both are page ids the caller
  passed in, partitioned -- returning them rather than a count is what lets the
  caller free exactly the right set without tracking the split itself.
  """

  adopted: np.ndarray
  duplicate: np.ndarray

  @property
  def num_adopted(self) -> int:
    return int(self.adopted.size)


class PrefixIndex:
  """A page trie mapping token prefixes to the pages already holding their K/V.

  The index owns every page it holds. A page enters through `publish` and leaves
  only through `evict` or `reset`, both of which hand it back so the caller can
  return it to the allocator. Nothing here allocates or frees; this layer never
  touches a device or a free list.
  """

  def __init__(self, tokens_per_page: int, enabled: bool = True):
    if tokens_per_page < 1:
      raise ValueError(f"tokens_per_page must be at least 1, got {tokens_per_page}")
    self.tokens_per_page = int(tokens_per_page)
    self.enabled = bool(enabled)
    self._root = PrefixNode(block_hash=b"", page_id=-1, tokens=_EMPTY_PAGES, depth=0, parent=None)
    self._num_nodes = 0
    self._protected_nodes = 0
    self._clock = 0
    self.num_queries = 0
    self.num_hit_pages = 0
    self.num_queried_pages = 0

  # -- statistics -----------------------------------------------------------

  @property
  def num_cached_pages(self) -> int:
    return self._num_nodes

  @property
  def protected_pages(self) -> int:
    """Pages a live request depends on. Not evictable at any pressure."""
    return self._protected_nodes

  @property
  def evictable_pages(self) -> int:
    return self._num_nodes - self._protected_nodes

  @property
  def hit_rate(self) -> float:
    """Fraction of queried pages served from the cache, across all lookups."""
    if not self.num_queried_pages:
      return 0.0
    return self.num_hit_pages / self.num_queried_pages

  def stats(self) -> dict[str, float | int]:
    return {
        "cached_pages": self.num_cached_pages,
        "protected_pages": self.protected_pages,
        "evictable_pages": self.evictable_pages,
        "queries": self.num_queries,
        "queried_pages": self.num_queried_pages,
        "hit_pages": self.num_hit_pages,
        "hit_rate": self.hit_rate,
    }

  # -- lookup ---------------------------------------------------------------

  def match(self, token_ids: Sequence[int], namespace: CacheNamespace) -> PrefixMatch:
    """Find the longest cached prefix of `token_ids` under `namespace`.

    Matching stops one page short of the end even on a total match. A request
    whose every token is cached would have no query tokens left to run, and the
    step would have nothing to compute a next-token logit from; leaving the final
    page to be recomputed keeps every request with at least one token of work.
    """
    tokens = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    full_pages = int(tokens.size) // self.tokens_per_page
    # A prompt that is an exact multiple of the page size would otherwise match
    # to its own end; hold back the last page in every case.
    matchable = max(full_pages - 1, 0) if tokens.size % self.tokens_per_page == 0 else full_pages

    self.num_queries += 1
    self.num_queried_pages += matchable
    if not self.enabled or matchable == 0:
      return PrefixMatch(pages=_EMPTY_PAGES, num_tokens=0, node=None)

    matched: list[int] = []
    node = self._root
    parent_hash = namespace.digest()
    for page_index in range(matchable):
      start = page_index * self.tokens_per_page
      page_tokens = tokens[start : start + self.tokens_per_page]
      key = block_hash(parent_hash, page_tokens)
      child = node.children.get(key)
      if child is None or not np.array_equal(child.tokens, page_tokens):
        break
      node = child
      parent_hash = key
      matched.append(child.page_id)

    if not matched:
      return PrefixMatch(pages=_EMPTY_PAGES, num_tokens=0, node=None)

    self._clock += 1
    self.num_hit_pages += len(matched)
    walk: PrefixNode | None = node
    while walk is not None and walk is not self._root:
      walk.last_access = self._clock
      walk.hit_count += 1
      walk = walk.parent

    return PrefixMatch(
        pages=np.asarray(matched, dtype=np.int32),
        num_tokens=len(matched) * self.tokens_per_page,
        node=node,
    )

  # -- publication ----------------------------------------------------------

  def publish(
      self,
      token_ids: Sequence[int],
      page_ids: Sequence[int] | np.ndarray,
      namespace: CacheNamespace,
      num_valid_tokens: int | None = None,
  ) -> PublishResult:
    """Offer a request's computed pages to the index.

    Only pages fully covered by `num_valid_tokens` are taken: a page holding
    tokens the request has not written yet would be published with K/V that does
    not exist. `page_ids` is the request's page list in sequence order, and the
    caller must have written the K/V for every token it declares valid.
    """
    tokens = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    pages = np.asarray(page_ids, dtype=np.int32).reshape(-1)
    valid = int(tokens.size if num_valid_tokens is None else num_valid_tokens)
    if valid > tokens.size:
      raise ValueError(f"declared {valid} valid tokens but was given only {tokens.size} token ids")
    if valid < 0:
      raise ValueError(f"num_valid_tokens must be non-negative, got {valid}")

    publishable = valid // self.tokens_per_page
    if publishable > pages.size:
      raise ValueError(
          f"{valid} valid tokens span {publishable} pages but the request holds only {pages.size}; "
          "the page list and the token list disagree about the request's length"
      )
    if not self.enabled or publishable == 0:
      return PublishResult(adopted=_EMPTY_PAGES, duplicate=_EMPTY_PAGES)

    adopted: list[int] = []
    duplicate: list[int] = []
    node = self._root
    parent_hash = namespace.digest()
    self._clock += 1
    for page_index in range(publishable):
      start = page_index * self.tokens_per_page
      page_tokens = tokens[start : start + self.tokens_per_page]
      key = block_hash(parent_hash, page_tokens)
      child = node.children.get(key)
      if child is None:
        child = PrefixNode(
            block_hash=key,
            page_id=int(pages[page_index]),
            tokens=page_tokens.copy(),
            depth=page_index + 1,
            parent=node,
            last_access=self._clock,
        )
        node.children[key] = child
        self._num_nodes += 1
        adopted.append(int(pages[page_index]))
        # A new node under a protected parent inherits nothing: the request that
        # locked the parent never referenced this page, so it starts evictable.
      else:
        child.last_access = self._clock
        if int(pages[page_index]) != child.page_id:
          duplicate.append(int(pages[page_index]))
      node = child
      parent_hash = key

    return PublishResult(
        adopted=np.asarray(adopted, dtype=np.int32) if adopted else _EMPTY_PAGES,
        duplicate=np.asarray(duplicate, dtype=np.int32) if duplicate else _EMPTY_PAGES,
    )

  # -- reference counting ---------------------------------------------------

  def acquire(self, node: PrefixNode | None) -> None:
    """Protect `node` and its ancestors from eviction while a request reads them.

    Without this a page could be evicted, freed, reallocated and overwritten
    while a live request's page table still names it -- which is the same
    use-after-free the dirty-page gate exists to prevent, arriving by a different
    route.
    """
    while node is not None and node is not self._root:
      if node.ref_count == 0:
        self._protected_nodes += 1
      node.ref_count += 1
      node = node.parent

  def release(self, node: PrefixNode | None) -> None:
    """Undo one `acquire`, making the path evictable again once nothing holds it."""
    while node is not None and node is not self._root:
      if node.ref_count <= 0:
        raise RuntimeError(
            f"prefix node at depth {node.depth} (page {node.page_id}) was released more times than "
            "it was acquired, so some live request's pages are now evictable"
        )
      node.ref_count -= 1
      if node.ref_count == 0:
        self._protected_nodes -= 1
      node = node.parent

  # -- eviction -------------------------------------------------------------

  def evict(self, num_pages: int) -> np.ndarray:
    """Drop up to `num_pages` least-recently-used pages and return them.

    Only leaves are eligible, so a page is never dropped while a longer cached
    prefix still depends on it. Freeing the returned pages is the caller's job;
    the index has already forgotten them by the time this returns.
    """
    if num_pages <= 0:
      return _EMPTY_PAGES

    leaves = [(n.last_access, id(n), n) for n in self._collect_leaves() if n.ref_count == 0]
    heapq.heapify(leaves)

    evicted: list[int] = []
    while leaves and len(evicted) < num_pages:
      _, _, node = heapq.heappop(leaves)
      # A node reached earlier in this pass may have gained a reference or, more
      # commonly, is stale because we pushed its parent after deleting it.
      if node.ref_count > 0 or not node.is_leaf or node is self._root:
        continue
      parent = node.parent
      evicted.append(node.page_id)
      del parent.children[node.block_hash]
      node.parent = None
      self._num_nodes -= 1
      if parent is not self._root and parent.is_leaf and parent.ref_count == 0:
        heapq.heappush(leaves, (parent.last_access, id(parent), parent))

    return np.asarray(evicted, dtype=np.int32) if evicted else _EMPTY_PAGES

  def reset(self) -> np.ndarray:
    """Forget everything and hand back every page the index held.

    Refuses while any page is referenced: a live request's page table would
    still name pages the caller is about to free.
    """
    if self._protected_nodes:
      raise RuntimeError(
          f"{self._protected_nodes} cached pages are still referenced by live requests; "
          "release them before resetting the index"
      )
    pages = [node.page_id for node in self._walk(self._root) if node is not self._root]
    self._root = PrefixNode(block_hash=b"", page_id=-1, tokens=_EMPTY_PAGES, depth=0, parent=None)
    self._num_nodes = 0
    return np.asarray(pages, dtype=np.int32) if pages else _EMPTY_PAGES

  # -- internals ------------------------------------------------------------

  def _walk(self, node: PrefixNode) -> Iterable[PrefixNode]:
    stack = [node]
    while stack:
      current = stack.pop()
      yield current
      stack.extend(current.children.values())

  def _collect_leaves(self) -> list[PrefixNode]:
    return [n for n in self._walk(self._root) if n is not self._root and n.is_leaf]
