"""Semantic control plane for the paged KV runtime: who owns which pages.

Host logic only, and deliberately so. Admission, page allocation, request-to-page
mapping and per-step metadata are all data-dependent irregular work over small
integer arrays -- the kind of thing that cannot live inside a traced computation
and does not want to. Keeping it here means the whole control plane is unit
testable on a machine with no accelerator.

Import rule, enforced by CI: this package may import the standard library,
``numpy``, and ``maxtext.inference.kv_common``. It must never import ``jax``,
``jax_aiter``, ``maxtext.layers``, or ``maxtext.models``. Vendor kernel ABIs are
reached only from ``kv_execution``, one layer up. Two things follow, and both are
worth the cost of policing the rule: the layer stays CPU-testable in ordinary CI,
and if a second consumer ever wants it, extraction is a directory move rather
than an archaeology exercise.

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

from maxtext.inference.kv_control.allocator import DoubleFreeError, PagedBlockAllocator
from maxtext.inference.kv_control.control_plane import DirtyPageError, NativeKvControlPlane
from maxtext.inference.kv_control.logical_block import (
    LogicalBlock,
    PageState,
    PageStateError,
    decode_needs_new_page,
    last_page_occupancy,
    new_pages_for_extend,
    pages_for_tokens,
    token_slot,
)
from maxtext.inference.kv_control.metadata import build_decode_table, build_page_table
from maxtext.inference.kv_control.page_map import PageCapacityError, PageMap, StaleRequestHandleError
from maxtext.inference.kv_control.protocols import KvControlPlane
from maxtext.inference.kv_control.request import RequestDescriptor, RequestHandle, RequestState

__all__ = [
    "DirtyPageError",
    "DoubleFreeError",
    "KvControlPlane",
    "LogicalBlock",
    "NativeKvControlPlane",
    "PageCapacityError",
    "PageMap",
    "PageState",
    "PageStateError",
    "PagedBlockAllocator",
    "RequestDescriptor",
    "RequestHandle",
    "RequestState",
    "StaleRequestHandleError",
    "build_decode_table",
    "build_page_table",
    "decode_needs_new_page",
    "last_page_occupancy",
    "new_pages_for_extend",
    "pages_for_tokens",
    "token_slot",
]
