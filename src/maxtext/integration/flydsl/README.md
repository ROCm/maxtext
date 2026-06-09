# FlyDSL MoE integration for MaxText

This package adds FlyDSL GPU grouped-GEMM kernels as a routed-MoE backend,
selected with the native config flag `moe_backend=flydsl`. No monkeypatching:
`RoutedMoE.__call__` dispatches to `maxtext/kernels/flydsl_moe.py` when the flag
is set, so stock MaxText scripts work unchanged.

## Requirements

- the `jax_flydsl` bridge + `flydsl_moe` MoE kernel API (built via jax-flydsl `setup_env.sh`)
- FlyDSL `kernels.*` checkout on `PYTHONPATH` (the jax-flydsl `setup_env.sh` /
  `.flydsl_env` does this). Check with `python3 -c "import flydsl_moe; print(flydsl_moe.moe_gemm_available())"`.

## Usage

The MoE weights are preshuffled **once at load** (not per forward), which needs
a one-line fill between `load_params` and the timed loop. The stock
`inference_microbenchmark` has no seam for that, so use this package's thin
runner (it also works as a stock baseline with `moe_backend=default`):

```bash
cd $MAXTEXT_DIR
# baseline
python3 -m maxtext.integration.flydsl.run_inference configs/base.yml \
    model_name=mixtral-8x7b hardware=gpu attention=dot_product \
    sparse_matmul=True megablox=False per_device_batch_size=1 \
    moe_backend=default \
    inference_microbenchmark_prefill_lengths="256,512,1024,2048" \
    inference_microbenchmark_stages=prefill

# FlyDSL — same command, one flag
python3 -m maxtext.integration.flydsl.run_inference configs/base.yml \
    model_name=mixtral-8x7b hardware=gpu attention=dot_product \
    sparse_matmul=True megablox=False per_device_batch_size=1 \
    moe_backend=flydsl \
    inference_microbenchmark_prefill_lengths="256,512,1024,2048" \
    inference_microbenchmark_stages=prefill
```

The stock `maxtext.inference.inference_microbenchmark` also runs with
`moe_backend=flydsl`, but it would re-shuffle the weights every forward (no
load-time fill seam), so prefer the thin runner above for representative perf.

The preshuffle itself is one-time: `RoutedMoE.__init__` allocates
`fly_w1/w2_shuffled` placeholders (flydsl only) and
`inject_preshuffled_weights(params)` fills them once from the stock
`wi_0/wi_1/wo`.

## Real checkpoints (recommended): bake the preshuffle offline

Runtime injection only patches the local params dict, not the engine's internal
decode state, so restoring a *stock* checkpoint with `moe_backend=flydsl` leaves
`fly_w1/w2_shuffled` abstract and `init_decode_state` crashes. For real
checkpoints, convert once with `preshuffle_checkpoint.py`: it loads the stock
weights with the default backend, computes the shuffled `fly_w*` per MoE block,
and writes a new parameter checkpoint that restores ready-to-use at full speed.

```bash
cd $MAXTEXT_DIR
# 1) Convert once (use the SAME model config you'll run with, esp. scan_layers)
python3 -m maxtext.integration.flydsl.preshuffle_checkpoint configs/base.yml \
    model_name=mixtral-8x7b tokenizer_path=$TOKENIZER_PATH \
    load_parameters_path=$STOCK_CKPT/0/items \
    save_quantized_params_path=$FLY_CKPT/0/items \
    scan_layers=false weight_dtype=bfloat16 per_device_batch_size=1 \
    ici_fsdp_parallelism=1 ici_tensor_parallelism=1 async_checkpointing=false \
    checkpoint_storage_use_ocdbt=false checkpoint_storage_use_zarr3=false

# 2) Run any stock inference path against the converted checkpoint — no inject,
#    no inline shuffle, decode works.
python3 -m maxtext.inference.inference_microbenchmark configs/base.yml \
    model_name=mixtral-8x7b hardware=gpu attention=dot_product \
    sparse_matmul=True megablox=False per_device_batch_size=1 \
    moe_backend=flydsl load_parameters_path=$FLY_CKPT/0/items
```

The converted checkpoint keeps the original `wi_0/wi_1/wo` too, so it stays
valid for `moe_backend=default` as well.

## Diagnostics (this package)

- `compare_backends.py` - prefill correctness: ragged vs dense vs flydsl, same weights.
- `compare_decode.py` - greedy autoregressive decode comparison.
- `compare_reprefill.py` - re-prefill vs KV-cache decode (isolates upstream KV bugs).
- `verify_correctness_{mixtral,gemma}.py` - per-model prefill logit agreement gate.
- `metrics.py` - `apply_moe_correction` for MoE active-params throughput numbers.

All select the backend via `moe_backend` in the config (no class swapping).

## How it works

- `configs/types.py` / `configs/base.yml`: the `moe_backend` flag (`default` | `flydsl`).
- `layers/moe.py` `RoutedMoE.__call__`: dispatches to `flydsl_matmul` when `moe_backend=flydsl`.
- `kernels/flydsl_moe.py`: reuses MaxText's `get_topk` router, preshuffles the stock
  `wi_0/wi_1/wo` weights via `flydsl_moe.preshuffle`, and runs the FlyDSL 2-stage
  grouped GEMM via `flydsl_moe.block`. Checkpoints are unchanged.
