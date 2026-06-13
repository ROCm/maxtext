# FlyDSL routed-MoE backend (Mixtral, ROCm)

Runs MaxText's routed-MoE matmul through the [FlyDSL](https://github.com/ROCm/FlyDSL)
2-stage grouped GEMM on AMD GPUs, as a config-gated first-class backend — no
monkeypatching. Validated for **Mixtral-8x7B**, **bf16**, inference.

## How it plugs in

`RoutedMoE.__call__` (`src/maxtext/layers/moe.py`) gains one branch:

```python
elif cfg.use_flydsl_moe:
    output, lb_loss, bias_updates = self.flydsl_matmul(
        inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel)
```

`flydsl_matmul` reuses MaxText routing (`get_topk`) and delegates the sort + GEMM to
the in-tree `flydsl_moe` package via `integration/flydsl/moe_bridge.py`.

## Layout (self-contained)

Everything needed at the JAX/kernel level lives here — no FlyDSL checkout or
`PYTHONPATH` juggling:

- `flydsl_moe/` — JAX MoE op: device-side sort, 2-stage grouped GEMM wrappers, and the
  weight-preshuffle helpers.
- `kernels/` — the FlyDSL device-kernel *builders* (JAX-free), copied verbatim from the
  [FlyDSL repo](https://github.com/ROCm/FlyDSL). The PyPI `flydsl` wheel ships the
  compiler but not these builders, and they import each other as `from kernels.X import`,
  so `integration/flydsl/__init__.py` front-inserts this dir on `sys.path` rather than
  rewriting them.

## Requirements

Two pip installs only:

- `flydsl` — the compiler wheel (bundles its own LLVM/MLIR).
- `jax_flydsl` — the generic `flydsl_call` JAX↔FlyDSL bridge.

Plus the ROCm runtime env (`LLVM_PATH`, `ROCM_PATH`, `XLA_FLAGS`) so XLA's AMDGPU
backend can find `ld.lld` — set by jax-flydsl's `setup_env.sh` (`.flydsl_env`).

## Which kernel?

Only one GPU kernel: the FlyDSL **2-stage grouped GEMM** (`compile_moe_gemm1` +
`compile_moe_gemm2`). Sort, weight preshuffle and block assembly are pure JAX in
`flydsl_moe`.

## Offline weight preshuffle

The kernels want a fused, MFMA-shuffled weight layout. Precompute it once from the
Hugging Face checkpoint instead of shuffling every step:

```bash
# From HF
python -m maxtext.integration.flydsl.preshuffle_mixtral_weights \
    --hf-repo mistralai/Mixtral-8x7B-v0.1 --out /data/mixtral_fly.npz

# Synthetic (CI / smoke; no download)
python -m maxtext.integration.flydsl.preshuffle_mixtral_weights \
    --synthetic --num-layers 2 --out /tmp/mixtral_fly_synth.npz
```

Output: `*.npz` with `layer_{i}/w1_shuffled` (`[E*2*M, D]`) and
`layer_{i}/w2_shuffled` (`[E*D, M]`) in bf16, plus a `*.meta.json`. The numerical test
proves these produce bit-identical output to inline shuffling.

> The live `flydsl_matmul` path currently shuffles inline so it runs against a stock
> Mixtral checkpoint with no extra step. Consuming the offline `.npz` to skip the
> per-step shuffle is a follow-up (wire `w1_shuffled`/`w2_shuffled` through
> `flydsl_routed_moe`, which already accepts them).

## Benchmark with `inference_microbenchmark`

No custom script — just toggle `use_flydsl_moe`:

```bash
python3 -m maxtext.inference.inference_microbenchmark \
    src/maxtext/configs/inference/inference.yml \
    model_name=mixtral-8x7b \
    use_flydsl_moe=true \
    dtype=bfloat16 weight_dtype=bfloat16 quantization="" \
    tokenizer_path=src/maxtext/assets/tokenizers/tokenizer.mistral-v1 \
    load_parameters_path=<path/to/mixtral/maxtext/ckpt> \
    hardware=gpu scan_layers=false async_checkpointing=false \
    per_device_batch_size=8 \
    max_prefill_predict_length=1024 max_target_length=2048 \
    inference_microbenchmark_stages=prefill,generate
```

Run the same command with `use_flydsl_moe=false` for the baseline A/B.

## Tests

```bash
pytest -m cpu_only tests/unit/flydsl_preshuffle_test.py       # layout/shape (no GPU)
pytest -m gpu_only tests/integration/flydsl_moe_test.py       # numerics vs fp32 ref (ROCm)
```

Validated on MI3xx: relative L2 error vs a float32 dense MoE reference ≈ 0.004 (bf16),
offline-preshuffled output bit-identical to inline.

## Limitations / notes

- **bf16 only** so far (matches `flydsl_moe.moe_block_fly_bf16_atomic`); FP8 is a follow-up.
- **Single-device / per-process GPU.** The FlyDSL runtime loads a kernel module bound to
  one device context, so single-process multi-GPU (`shard_map` across local devices) is
  not yet supported; use one process per GPU.
- Stage 1 passes `use_cshuffle_epilog=False` (in `flydsl_moe/block.py`) because FlyDSL
  0.2.0's stage-1 bf16 CShuffle epilogue mis-types its LDS store. The direct epilogue is
  numerically equivalent; remove the override once the pin fixes it.
