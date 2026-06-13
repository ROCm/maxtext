# SPDX-License-Identifier: Apache-2.0
"""FlyDSL ROCm MoE backend for MaxText (config-gated via use_flydsl_moe).

  flydsl_moe/  - JAX MoE op (sort + 2-stage GEMM + preshuffle)
  kernels/     - FlyDSL device-kernel builders, copied verbatim from the FlyDSL repo
  moe_bridge.py - adapts MaxText weights/routing to the op

pip deps: flydsl (compiler) + jax_flydsl (bridge).
"""

import os
import sys

# kernels/ and flydsl_moe/ import each other by bare name (from kernels.X import ...),
# so put this dir on sys.path ahead of any stale FlyDSL checkout.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
  sys.path.insert(0, _HERE)
