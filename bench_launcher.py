"""Runs MaxText pre-training with the fp8 qwix rule scoped to chosen ops.

MaxText's fp8_full rule covers ("dot_general", "gmm", "ragged_dot"), so it
quantizes attention and dense projections as well as the MoE grouped GEMMs.
MXFP8 only touches the three MoE GEMMs, so comparing against stock fp8_full is
not like-for-like. FP8_OPS overrides the op list, e.g.

  FP8_OPS=gmm,ragged_dot   fp8 on the MoE GEMMs only
  FP8_OPS=dot_general      fp8 everywhere except the MoE GEMMs

The latter pairs with use_flydsl_moe to isolate the MoE backend: both
sides then run fp8 attention and differ only in how the grouped GEMMs are done.
"""

import dataclasses
import os
import sys

# FlyDSL carries its own MLIR, and PyTorch -- pulled in by transformer_engine,
# which MaxText imports for its TE attention integration -- carries another
# LLVM. Whichever registers its command-line options second aborts the process
# with "Option 'print-inst-addrs' registered more than once", so the MXFP8 arm
# dies during tracing at flydsl._mlir.ir. Importing FlyDSL before anything that
# reaches torch lets it register first.
if os.environ.get("FLYDSL_FIRST"):
  import jax_flydsl  # noqa: F401

from absl import app  # noqa: E402  (import after the MLIR ordering fix above)

from maxtext.layers import quantizations


def _scope_fp8(op_names):
  original = quantizations.get_fp8_full_qwix_rule_w_sparsity

  def scoped(config):
    rules = [dataclasses.replace(r, op_names=op_names) for r in original(config)]
    print(f"[bench] fp8 scoped to {op_names} on {len(rules)} rule(s)", file=sys.stderr)
    return rules

  quantizations.get_fp8_full_qwix_rule_w_sparsity = scoped


_fp8_ops = os.environ.get("FP8_OPS", "")
if _fp8_ops:
  _scope_fp8(tuple(op.strip() for op in _fp8_ops.split(",") if op.strip()))

from maxtext.trainers.pre_train import train  # noqa: E402  (import after patching)

if __name__ == "__main__":
  app.run(train.main)
