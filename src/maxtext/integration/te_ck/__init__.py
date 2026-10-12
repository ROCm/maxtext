# SPDX-License-Identifier: Apache-2.0
"""Transformer Engine / Composable Kernel backend for MaxText routed MoE
(config-gated via use_te_ck_moe).

  moe_bridge.py - adapts MaxText's routing output to TE's grouped_dense

Requires the ROCm fork of `transformer_engine` built with CK grouped GEMM, and
NVTE_USE_CK_GROUPED_GEMM=1 in the environment to select CK over hipBLASLt.
"""
