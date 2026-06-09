# Inference microbenchmark: default vs FlyDSL (MI355X / gfx950)

`inference_microbenchmark` with `moe_backend=default` vs `moe_backend=flydsl`, on
the offline-preshuffled checkpoints, single device, bf16, `per_device_batch_size=1`,
`scan_layers=false`, prefill lengths 128/256/512/1024/2048, 10 timed iters.

Decode runs via the non-AOT generate path (the AOT generate executable has a
non-deterministic XLA auto-layout requirement on jaxlib 0.8.2+rocm; the AOT
prefill-insert step is likewise flaky and is guarded/skipped).

## Mixtral-8x7b — prefill step time (ms)

| seq | default | flydsl | speedup |
|----:|--------:|-------:|--------:|
| 128 | 70.3 | 26.1 | 2.7x |
| 256 | 72.6 | 30.0 | 2.4x |
| 512 | 97.3 | 40.4 | 2.4x |
| 1024 | 176.2 | 61.3 | 2.9x |
| 2048 | 292.7 | 115.6 | 2.5x |

Decode (AR): 69.8 -> 19.8 ms/step; 14.3 -> 50.6 tok/s (3.5x).

## Gemma4-26b — prefill step time (ms)

| seq | default | flydsl | speedup |
|----:|--------:|-------:|--------:|
| 128 | 149.4 | 14.6 | 10.2x |
| 256 | 260.8 | 20.0 | 13.0x |
| 512 | 349.4 | 23.7 | 14.7x |
| 1024 | 649.7 | 30.0 | 21.7x |
| 2048 | 1016.0 | 46.2 | 22.0x |

Decode (AR): 75.6 -> 17.1 ms/step; 13.2 -> 58.4 tok/s (4.4x).

## TFLOPs/sec/device (as reported - see caveat)

| seq | mixtral default | mixtral flydsl | gemma default | gemma flydsl |
|----:|----------------:|---------------:|--------------:|-------------:|
| 128 | 170.2 | 902.1 | 43.3 | 881.2 |
| 256 | 329.6 | 1567.4 | 49.6 | 1283.6 |
| 512 | 492.1 | 2326.6 | 74.1 | 2169.6 |
| 1024 | 544.5 | 3071.4 | 79.9 | 3431.7 |
| 2048 | 657.3 | 3261.3 | 102.7 | 4471.5 |

## Caveats

1. The TFLOPs figures for the flydsl path exceed the MI355X bf16 peak (~2.5 PFLOPs),
   so the FLOP accounting overcounts there. Use the **step times** for comparison.
2. Gemma's 10-22x reflects a pathologically slow default 128-expert `ragged_dot`
   path (43-103 TFLOPs/s), not only FlyDSL being fast.
3. Gemma decode output is currently garbage for **both** backends (upstream KV-cache
   bug); its AR timing is valid but the generated text is not. Mixtral decode is
   correct (fly == ragged, 10/10 prompts).
