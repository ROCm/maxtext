# MXFP8 MoE training path in MaxText

Status as of 2026-08-07. Branch `flydsl-moe-mxfp8-training`, worktree
`/home/araganes/maxtext-mxfp8-train`, based on `eea99981`. Nothing committed yet.

This adds a *training* path for MXFP8 grouped GEMMs, parallel to and independent of
the existing inference path (`use_flydsl_moe`). It is gated behind a new config flag
`use_flydsl_moe` and is off by default.

## What changed

| file | lines | what |
| --- | --- | --- |
| `src/maxtext/integration/flydsl/moe_bridge.py` | +50 | `group_offsets_from_sizes`, `grouped_gemm_mxfp8_experts` |
| `src/maxtext/layers/moe.py` | +57/-10 | dispatch in `_moe_body`, guards in `sparse_matmul`, tokamax fix |
| `src/maxtext/configs/types.py` | +11 | `use_flydsl_moe` flag |
| `src/maxtext/configs/base.yml` | +4 | flag default and docs |

Untracked helper files, benchmarking only, not part of the feature:
`bench_moe_ep.sh`, `bench_launcher.py`.

The bridge transposes MaxText's `[E, K, N]` expert kernel into the operation's
`[G, N, K]`, converts `group_sizes` into `[G+1]` cumulative offsets, and zeroes rows
past `group_offsets[-1]`. That masking matters because the capacity-padded tail is
uninitialised allocator memory; without it the quantizer picks block scales off stale
data (observed as 1.7e38 in a reused buffer).

One change in `moe.py` is unrelated to MXFP8: `get_tokamax_group_sizes` was computed
unconditionally, which throws `TypeError: 'int' object is not iterable` against
tokamax 0.0.10. It is now guarded by `use_tokamax_gmm`. This is the same fix already
present in the uncommitted work on `flydsl-moe-consume-pkg`. Without it no MoE run
starts at all in this container.

### Constraints

Both expert dims must be multiples of 128 and at least 256, because K and N are each
contracted somewhere across the forward, dgrad and wgrad. The flag also rejects
combining with `use_flydsl_moe` or `num_moe_emb_chunks > 0`.

## Correctness

30 steps, 4 layers, 8 experts, top-2, emb 512, moe_mlp 1024, batch 4, seq 1024,
synthetic data, one MI355.

| step | bf16 (megablox) | MXFP8 | delta |
| --- | --- | --- | --- |
| 0 | 10.8510 | 10.8500 | -0.0010 |
| 10 | 10.4960 | 10.4960 | 0.0000 |
| 20 | 10.2190 | 10.2200 | +0.0010 |
| 29 | 10.1410 | 10.1430 | +0.0020 |

Max divergence over the run is 0.0020, roughly the resolution of the logged loss.
Gradients flow through the `custom_vjp`, so the backward dgrad and wgrad GEMMs are
what drive these updates.

The path was confirmed live, rather than silently falling back, by setting
`moe_mlp_dim=192`, which the operation must reject. It raised from
`moe_bridge.py:58` into `grouped_gemm_mxfp8.py:308` with
`N=192 must be a multiple of 128 and at least 256`.

## Performance

Median of steps 10-29, 2 layers, 8 experts, top-2, emb 1024, moe_mlp 4096, batch 8,
seq 1024. That is 16384 grouped-GEMM rows with K=1024, N=4096, so the MoE GEMMs
dominate rather than dispatch overhead.

| backend | median ms/step | min | max | loss@29 |
| --- | --- | --- | --- | --- |
| megablox bf16 | 397.0 | 394.0 | 412.0 | 9.9040 |
| megablox fp8 | 407.5 | 404.0 | 425.0 | 9.9130 |
| ragged_dot fp8 | 53.0 | 45.0 | 81.0 | 9.9130 |
| flydsl MXFP8 | 20.0 | 20.0 | 26.0 | 9.9070 |

All four start at loss 10.8900 and land within 0.009 of each other at step 29, so
every backend is doing correct work and the times are comparable.

**Headline: MXFP8 is about 2.7x the fp8 baseline** (20.0 vs 53.0 ms).

Do not quote the 19.9x against megablox bf16. Megablox is a Pallas kernel that is
pathologically slow on ROCm here. The tell is that fp8 on megablox is *slower* than
bf16 (407.5 vs 397.0): the step is bound by the Pallas kernel and the precision
change buys nothing. The honest comparison is the `ragged_dot` fp8 row.

### At Mixtral width, command buffers off (2026-08-10)

Everything above was measured on the shrunk stand-in, whose expert GEMMs are about
1/14 of Mixtral-8x7B's, and with command buffers enabled. Both distortions are now
gone. Real Mixtral width (emb 4096, moe_mlp 14336, 32/8 heads), 2 layers, batch 8,
seq 1024, one MI355, median of steps 10-19:

| backend | ms/step | TFLOP/s/device | loss@19 |
| --- | --- | --- | --- |
| ragged_dot bf16 | 302 | 151 | 7.040 |
| flydsl MXFP8 | 95 | 478 | 7.053 |

**3.18x over the bf16 `ragged_dot` baseline**, loss tracking to 0.013. The kernel
measured ~560 TFLOP/s standalone, so 478 end-to-end is most of what it can give.

Command buffers were also inflating the expert-parallel numbers, and not evenly:
at toy width, EP=4, turning them off moved MXFP8 from 219 to 64 ms and `ragged_dot`
bf16 from 165 to 80 ms. Any conclusion drawn from measurements with them enabled --
including the earlier reading that MXFP8 loses at EP=4 -- is void.

### Expert-parallel sweep, Mixtral width (2026-08-10)

Same config, `ici_expert_parallelism` swept, `per_device_batch_size=8` held so each
device keeps the same GEMM rows and global batch grows with the device count.
Median of steps 10-29. `fp8_moe` is qwix fp8 scoped to `gmm,ragged_dot`, which is
exactly what MXFP8 replaces; `fp8_full` additionally quantizes `dot_general`.

| arm | EP=1 | EP=2 | EP=4 | EP=8 |
| --- | --- | --- | --- | --- |
| bf16 | 301 ms | 634 ms | 463 ms | 259 ms |
| fp8_moe | 185 ms | 479 ms | 354 ms | 217 ms |
| fp8_full | 185 ms | died@3 | 362 ms | 225 ms |
| flydsl MXFP8 | 95 ms | 402 ms | 248 ms | 165 ms |
| MXFP8 vs bf16 | 3.17x | 1.58x | 1.87x | 1.57x |
| MXFP8 vs fp8_moe | 1.95x | 1.19x | 1.43x | 1.32x |

MXFP8 is the fastest arm at every degree. Final losses track bf16 within 0.02
throughout (4.994/4.973, 6.339/6.324, 7.423/7.412, 8.307/8.299) and are closer to
bf16 than `fp8_full` is at every degree, so MXFP8 beats plain fp8 on accuracy as
well as speed. Losses are comparable only within a column: global batch scales with
the device count, so each EP degree is a different trajectory.

The margin is widest at EP=1 and compresses to 1.2-1.4x once EP is on, because the
collectives and the capacity-padded buffer take a share of the step that a faster
GEMM cannot reach.

Two open anomalies, neither backend-specific:

- **EP=2 is slower than EP=4 for every arm** (634 vs 463 bf16) even though EP=4
  carries twice the receive buffer. It hits all four arms equally, so it is in the
  shared routing/all-to-all path.
- **`fp8_full` died at EP=2 only**, surviving EP=1, 4 and 8, while `fp8_moe` never
  failed. So fp8 on the MoE grouped GEMMs is fine under EP, the earlier
  "uninitialized ragged buffer poisons a per-tensor scale" theory is wrong, and the
  remaining failure looks flaky rather than structural.

### Caveat on scope, currently unresolved

MaxText's `fp8_full` qwix rule is
`op_names=("dot_general", "gmm", "ragged_dot")` (`quantizations.py:800`), so the fp8
runs quantize attention and dense projections too. MXFP8 only touches the three MoE
GEMMs and leaves everything else in bf16. The fp8 baseline is therefore getting
speedups in places MXFP8 is not, which makes 2.7x conservative for the MoE portion.

There is **no config option to scope quantization to the MoE layer**; the op tuple is
hardcoded. `bench_launcher.py` patches `get_fp8_full_qwix_rule_w_sparsity` to
override it, and the patch verifiably applies. But the obvious experiment, fp8 on
`("gmm", "ragged_dot")` only, segfaults, because that leaves attention in bf16 and
bf16 `ragged_dot` is one of the crashing combinations below.

The planned way around it, not yet run: give the MXFP8 run fp8 attention via
`FP8_OPS=dot_general`, so both sides have fp8 attention and differ only in the MoE
backend. `bench_launcher.py` already supports this through the `FP8_OPS` env var.

## Environment problems found, none in the new code

1. **tokamax vs gfx950.** `tokamax/_src/precision.py:131` does
   `float(compute_capability)`, which throws on `'gfx950'`. Hit through the default
   attention. Worked around with `attention=dot_product`; the real fix is upstream.

2. **tokamax 0.0.10 group sizes.** Fixed in `moe.py`, see above.

3. **`quantization=nanoo_fp8` cannot run MoE.** `get_quantization_dtypes` in
   `moe.py:1478` reads `self.quant.quant_dg`, and `NANOOFp8Quantization` has no such
   attribute. AMD's own fp8 mode is therefore unusable on the grouped-GEMM path.
   Worth reporting upstream. `fp8_full` with `use_qwix_quantization=true` works and is
   what the numbers above use.

4. **Segfault at first execution (SIGSEGV, exit 139) -- root-caused, fixed.**
   This capped every shape measured before 2026-08-10. It is XLA's *command
   buffer* path, not a size limit and nothing to do with MoE or quantization.

   Under gdb the faulting frames are `RocmCommandBuffer::LaunchGraph` ->
   `GpuCommandBuffer::Submit` -> `CommandBufferThunk::ExecuteOnStream`, and
   inside `libamdhip64.so.7` the same return address repeats six times: the HIP
   runtime recurses over the graph and overflows the stack. Consistent with
   `dmesg` showing no GPU memory-access fault -- the process dies host-side.

   The trigger is graph node count, not parameter count, which is why the
   earlier shape probes looked arbitrary (`emb=4096`/8 heads/head_dim 64 runs,
   `emb=2048`/16 heads/128 dies, `emb=512`/32 heads/128 runs) and why a fully
   quantized model survived where its bf16 twin died: different op counts
   produce different graphs. The "~3.1B params" threshold was a coincidence; a
   0.567B config reproduces it.

   Fix: `XLA_FLAGS=--xla_gpu_enable_command_buffer=`. Every benchmark arm must
   set it, both so the larger sizes run and so the arms stay comparable --
   command buffers were also distorting the timings badly (see below), and are
   the same feature that made decode 1.5-1.9x slower in the inference campaign.

## Reproducing

`bench_moe_ep.sh <ep> <steps> [toy|mixtral]` runs the bf16, scoped-fp8, full-fp8 and
MXFP8 arms at a chosen expert-parallel degree. From the container:

```
docker exec -e HIP_VISIBLE_DEVICES=0,1,2,3 -w /jax_dir/maxtext-mxfp8-train rocm_jax \
  bash /jax_dir/maxtext-mxfp8-train/bench_moe_ep.sh 4 30 mixtral
```

A single run needs `PYTHONPATH=/jax_dir/jax-flydsl:/jax_dir/maxtext-mxfp8-train/src`,
`attention=dot_product`, and `XLA_FLAGS=--xla_gpu_enable_command_buffer=`, plus
`use_flydsl_moe=true` to enable the path.

## Remaining work

Optimizations, roughly by expected value:

- Hoist weight quantization out of the per-call path. It was about 55% of the
  differentiated forward at benchmarked shapes and is a fixed cost per layer that is
  currently paid every microbatch. Needs an API change.
- Drop the per-call `jnp.swapaxes` of the expert kernel in the bridge, either by
  storing transposed or folding the transpose into quantization.
- Wrap the vendored `compile_qdual` dense quantizer and retire the JAX stopgap in
  `quantize_mxfp8_2d` / `quantize_mxfp8_weights`.
- Avoid or fuse the `[:total_m]` slice in the NT forward and dgrad. The kernel writes
  tight, so the padded tail is copied for nothing.
- Use the fused SwiGLU kernel instead of MaxText's activation; it folds the routing
  weight into the backward.
- Skip the tail mask when `sum(group_sizes)` already equals the buffer rows, so the
  extra pass per GEMM is only paid when `ragged_buffer_factor` demands it.
- Kernel tuning: forward measured ~560 TFLOP/s standalone, well under fp8 peak.

Note that at toy width the whole op -- forward, both gradients, quantization -- is
about 4 ms of a 219 ms step, so none of the above is worth doing on evidence from
that config. Measure at Mixtral width, where the GEMMs actually dominate.

Validation still owed:

- Finish the like-for-like fp8 comparison. `bench_moe_ep.sh` now runs fp8 scoped to
  `gmm,ragged_dot`, which matches MXFP8's scope exactly; the previous blocker for
  that arm was the command-buffer segfault, not the scoping.
- Re-run the EP sweep at Mixtral width with command buffers off. Every EP number
  taken before 2026-08-10 is void.
- Confirm whether the tail masking is load-bearing here or purely defensive, by
  checking whether `sum(group_sizes)` equals the buffer rows in these runs.
