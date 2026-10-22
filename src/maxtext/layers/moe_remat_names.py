# Copyright 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Names of MoE tensors that a remat policy can save instead of recomputing in the backward pass.

`remat_policy=minimal_with_context_moe` saves MOE_IO_NAMES, and `minimal_with_context_moe_routing` also saves
MOE_ROUTING_NAMES (see `Decoder.minimal_policy` in decoders.py and nnx_decoders.py, which share these lists).
A name only matters if the tensor is tagged with `checkpoint_name` somewhere; the tags are in `moe.py`.
"""

# Expert-parallel exchange results and the expert up-projection output, so backward does not replay the all-to-alls.
MOE_IO_NAMES = (
    "moe_mlpwi_1",  # expert up-projection output (tagged on the non-FlyDSL gmm path only)
    # `moe_dispatched` names different tensors on the two exchange paths:
    #   ragged all-to-all: the expert-sorted rows after `local_permute` (`route` in moe.py)
    #   DeepEP: `recv_x`, the rows as DeepEP delivered them, before any per-expert fan-out
    "moe_dispatched",
    "moe_dispatched_meta",  # DeepEP only: received expert ids, router weights and the dispatch handle
    "moe_dedup_plan",  # use_deepep_dedup_receive only: the row map and group sizes of the expert tier
    "moe_combined",  # the combine result (ragged return all-to-all, or DeepEP combine)
)

# Routing metadata and the MLA low-rank projections. They are cheap next to expert activations, and a recomputed top-k
# could differ from the forward one and misalign the saved expert rows.
MOE_ROUTING_NAMES = (
    "moe_topk_weights",
    "moe_topk_indices",
    "moe_sort_order",
    "moe_group_sizes",
    "moe_reshaped_group_sizes",
    "moe_all_shards_group_sizes",
    "moe_global_group_sizes",
    "moe_local_sort_order",
    "moe_local_group_sizes",
    "query_wa_proj",
    "kv_wa_proj",
    "mla_q",
    "mla_kv",
)
