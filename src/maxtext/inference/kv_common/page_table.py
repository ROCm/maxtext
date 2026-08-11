"""Per-step page table: which pages each request holds, and where to write.

Purely semantic. No strides, no packing, no vendor shapes, and no device arrays --
everything here is host numpy, because every field is produced by data-dependent
irregular host logic that cannot live inside a traced computation.

A backend converts this into whatever flat arrays its kernels take. That
conversion is the only place a vendor contract appears.

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

import numpy as np

KV_PAGE_TABLE_VERSION = 1


@dataclasses.dataclass
class KvPageTableV1:
    """One step's worth of page bookkeeping.

    Attributes:
        page_ids: per request, the pages holding its context in sequence order.
        seq_lens: int32 [num_reqs] total context length after this step.
        query_lens: int32 [num_reqs] new tokens contributed this step. All ones
            for decode; the uncached suffix length for prefill.
        write_positions: int32 [num_tokens] absolute token index within its own
            sequence for each new token, flattened in request order.
        request_order: int32 [num_reqs] the order requests appear in the batch.
    """

    page_ids: list[list[int]] = dataclasses.field(default_factory=list)
    seq_lens: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )
    query_lens: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )
    write_positions: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )
    request_order: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )
    version: int = KV_PAGE_TABLE_VERSION

    @property
    def num_requests(self) -> int:
        return len(self.page_ids)

    @property
    def num_tokens(self) -> int:
        return int(self.write_positions.shape[0])

    def validate(self, tokens_per_page: int) -> None:
        """Check internal consistency before anything reaches a kernel.

        Cheap here, and the alternative is a silent out-of-bounds read inside an
        attention kernel.
        """
        n = self.num_requests
        for name, arr in (
            ("seq_lens", self.seq_lens),
            ("query_lens", self.query_lens),
            ("request_order", self.request_order),
        ):
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
            if arr.dtype != np.int32:
                raise ValueError(f"{name} must be int32, got {arr.dtype}")

        if int(self.query_lens.sum()) != self.num_tokens:
            raise ValueError(
                f"query_lens sums to {int(self.query_lens.sum())} but there are "
                f"{self.num_tokens} write positions"
            )

        for i, pages in enumerate(self.page_ids):
            needed = -(-int(self.seq_lens[i]) // tokens_per_page)  # ceil
            if len(pages) < needed:
                raise ValueError(
                    f"request {i} holds {len(pages)} pages but seq_len "
                    f"{int(self.seq_lens[i])} needs {needed} at "
                    f"{tokens_per_page} tokens per page"
                )

    def last_page_lens(self, tokens_per_page: int) -> np.ndarray:
        """Occupancy of each request's final page, in tokens.

        Kept exact rather than rounded up: an over-stated last-page length is how
        a kernel reads bytes belonging to a previous occupant of a recycled page.
        """
        lens = np.empty((self.num_requests,), dtype=np.int32)
        for i in range(self.num_requests):
            seq_len = int(self.seq_lens[i])
            if seq_len == 0:
                lens[i] = 0
                continue
            rem = seq_len % tokens_per_page
            lens[i] = rem if rem else tokens_per_page
        return lens

    def indptr(self) -> np.ndarray:
        """Exclusive prefix sum over per-request page counts, int32 [num_reqs+1]."""
        counts = np.array([len(p) for p in self.page_ids], dtype=np.int32)
        out = np.zeros((self.num_requests + 1,), dtype=np.int32)
        if self.num_requests:
            np.cumsum(counts, out=out[1:])
        return out

    def flat_page_indices(self) -> np.ndarray:
        """All page ids concatenated in request order, int32."""
        if not self.page_ids:
            return np.zeros((0,), dtype=np.int32)
        return np.concatenate(
            [np.asarray(p, dtype=np.int32) for p in self.page_ids]
        ).astype(np.int32)

    def slot_mapping(self, tokens_per_page: int, padding_page_id: int = 0) -> np.ndarray:
        """Absolute pool slot for each new token, int32 [num_tokens].

        A slot is ``page_id * tokens_per_page + offset_within_page``, which is
        what an append kernel scatters on. Tokens whose page is the padding
        sentinel map to -1 so the kernel skips them.
        """
        slots = np.empty((self.num_tokens,), dtype=np.int32)
        t = 0
        for i, pages in enumerate(self.page_ids):
            for _ in range(int(self.query_lens[i])):
                pos = int(self.write_positions[t])
                page_slot = pos // tokens_per_page
                if page_slot >= len(pages):
                    raise ValueError(
                        f"request {i} token at position {pos} needs page slot "
                        f"{page_slot} but only {len(pages)} pages are held"
                    )
                page_id = pages[page_slot]
                if page_id == padding_page_id:
                    slots[t] = -1
                else:
                    slots[t] = page_id * tokens_per_page + (pos % tokens_per_page)
                t += 1
        return slots
