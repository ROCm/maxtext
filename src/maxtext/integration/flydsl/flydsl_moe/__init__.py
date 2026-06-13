# SPDX-License-Identifier: Apache-2.0
"""JAX bindings for the FlyDSL 2-stage grouped-GEMM MoE kernel.

gemm: kernel calls; block: sort + 2-stage assembly; preshuffle: weight layout.
"""
