"""MaxText-specific execution layer for the paged KV runtime.

Where the neutral vocabulary and the semantic control plane meet JAX, MaxText's
config, and a vendor's kernels. Unlike `kv_common` and `kv_control`, this package
is *not* extractable and is not meant to be: it exists precisely to hold the
couplings the other two refuse.

So the import rule inverts here. This package may import `jax`, MaxText config
and layers, and a vendor backend. What it must not do is let any of that leak
downwards -- `kv_control` calling into `kv_execution` would put jax back in the
control plane's dependency graph and undo the split. `kv_import_rule_test.py`
checks that direction statically.

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

from maxtext.inference.kv_execution.bucketing import (
    StepShape,
    batch_ladder,
    bucket_up,
    token_ladder,
)
from maxtext.inference.kv_execution.driver import PagedDriver, PagedRequest, StepOutcome
from maxtext.inference.kv_execution.layout_builder import build_storage_layout
from maxtext.inference.kv_execution.pool_factory import PagedKvPool, allocate_pool
from maxtext.inference.kv_execution.pool_ops import (
    POISON_SENTINEL,
    poison_pages,
    scrub_pages,
    scrub_pages_all_layers,
)
from maxtext.inference.kv_execution.step_inputs import RequestSlice, StepInputs, build_step_inputs
from maxtext.inference.kv_execution.step_view import StepView, build_step_view

__all__ = [
    "POISON_SENTINEL",
    "PagedDriver",
    "PagedKvPool",
    "PagedRequest",
    "RequestSlice",
    "StepInputs",
    "StepOutcome",
    "StepShape",
    "StepView",
    "allocate_pool",
    "batch_ladder",
    "bucket_up",
    "build_step_inputs",
    "build_step_view",
    "build_storage_layout",
    "poison_pages",
    "scrub_pages",
    "scrub_pages_all_layers",
    "token_ladder",
]
