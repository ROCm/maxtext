"""Turn live page bookkeeping into a `KvPageTableV1` for one step.

The output is the neutral vocabulary, not a vendor shape. A backend converts it
into whatever flat arrays its kernels take, and that conversion is the only
place a vendor contract appears.

The one subtlety worth stating plainly, because getting it wrong is silent:
a request's page list is trimmed to exactly the pages its current length needs.
A page table is read together with `kv_last_page_lens`, and that occupancy is
applied to the *last* page in the list. Hand over one page more than the length
requires and the occupancy lands on the wrong page, so the kernel reads a full
page of whatever the previous occupant left behind and then one token of real
data. The page table type permits over-supply -- its own validation only checks
that enough pages are present -- so the trim has to happen here.

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

from typing import Sequence

import numpy as np

from maxtext.inference.kv_common import KvPageTableV1
from maxtext.inference.kv_control.logical_block import pages_for_tokens
from maxtext.inference.kv_control.page_map import PageMap
from maxtext.inference.kv_control.request import RequestHandle


def build_page_table(
    page_map: PageMap,
    handles: Sequence[RequestHandle],
    query_lens: Sequence[int] | np.ndarray,
) -> KvPageTableV1:
  """Build one step's page table.

  Args:
    page_map: the live bookkeeping. Sequence lengths are read from it and must
      already include this step's tokens, since the table describes the state
      the kernels will see rather than the state before the step.
    handles: the requests in this batch, in the order the batch presents them.
    query_lens: new tokens each request contributes this step. All ones for
      decode; the uncached suffix length for prefill.

  Returns:
    A validated `KvPageTableV1`.
  """
  query_lens = np.asarray(query_lens, dtype=np.int32).reshape(-1)
  if query_lens.size != len(handles):
    raise ValueError(f"got {len(handles)} handles but {query_lens.size} query lengths")

  tokens_per_page = page_map.tokens_per_page
  num_requests = len(handles)
  page_ids: list[list[int]] = []
  seq_lens = np.zeros((num_requests,), dtype=np.int32)
  request_order = np.zeros((num_requests,), dtype=np.int32)
  positions: list[np.ndarray] = []

  for i, handle in enumerate(handles):
    seq_len = page_map.seq_len(handle)
    query_len = int(query_lens[i])
    if query_len < 0:
      raise ValueError(f"request {handle.request_id!r} has negative query length {query_len}")
    if query_len > seq_len:
      raise ValueError(
          f"request {handle.request_id!r} contributes {query_len} tokens but its recorded length is "
          f"{seq_len}; advance the length as pages are reserved, not afterwards"
      )
    needed = pages_for_tokens(seq_len, tokens_per_page)
    page_ids.append(page_map.pages(handle)[:needed].tolist())
    seq_lens[i] = seq_len
    request_order[i] = handle.row
    positions.append(np.arange(seq_len - query_len, seq_len, dtype=np.int32))

  write_positions = (
      np.concatenate(positions).astype(np.int32) if positions else np.zeros((0,), dtype=np.int32)
  )
  table = KvPageTableV1(
      page_ids=page_ids,
      seq_lens=seq_lens,
      query_lens=query_lens,
      write_positions=write_positions,
      request_order=request_order,
  )
  table.validate(tokens_per_page)
  return table


def build_decode_table(page_map: PageMap, handles: Sequence[RequestHandle]) -> KvPageTableV1:
  """One token per request, which is what makes decode a fixed-shape step."""
  return build_page_table(page_map, handles, np.ones((len(handles),), dtype=np.int32))
