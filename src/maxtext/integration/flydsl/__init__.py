# SPDX-License-Identifier: Apache-2.0
"""FlyDSL MXFP8 backend for MaxText routed MoE (config-gated via use_flydsl_moe).

  moe_bridge.py - adapts MaxText's routing output to the grouped MXFP8 GEMM

Requires the `jax_flydsl` package, which owns the kernels and the differentiable
ops, on PYTHONPATH.
"""
