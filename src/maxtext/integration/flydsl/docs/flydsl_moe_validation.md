# FlyDSL MoE in MaxText — Validation & Performance

Correctness and performance validation of the FlyDSL Mixture-of-Experts kernel
integrated into MaxText, for **Mixtral 8x7B** and **Gemma4-26B**, on AMD MI355X
(gfx950), ROCm, bf16, single device.

Full technical overview: [flydsl_maxtext_overview.md](flydsl_maxtext_overview.md).

## TL;DR

- **Correctness:** FlyDSL (`fly`) is numerically faithful to MaxText's exact
  sparse backend (`ragged_dot`). On **real trained weights**, fly matches ragged
  **10/10 top-1** on both models (prefill), cosine ≥ 0.9999.
- **Decode:** Mixtral **10/10** full greedy sequences exact (8 tokens × 10 prompts).
  Gemma decode fails for **both** backends (MaxText KV-cache bug, not fly).
- **Performance (bf16 prefill, end-to-end):** fly is **~2.5×** faster than ragged
  on Mixtral and **8–22×** faster on Gemma.

---

## 1. What is compared

Three MoE execution paths inside MaxText's `RoutedMoE`:

| backend | flag | description | exact? |
|---|---|---|---|
| **ragged** | `sparse_matmul=True` | stock `jax.lax.ragged_dot` | yes (dropless) |
| **dense** | `sparse_matmul=False` | capacity-padded batched matmul | no (drops on overflow) |
| **fly** | `fly_moe_backend()` context | tuned CDNA grouped GEMM | yes (dropless) |

`ragged` is the apples-to-apples baseline for `fly`. `dense` may drop tokens above
`capacity_factor` and is not bit-comparable when overflow occurs.

| | Mixtral-8x7B | Gemma4-26B |
|---|---|---|
| experts / topk | 8 / 2 | 128 / 8 |
| moe_mlp_dim | 14336 | 704 (fly pads → 768) |
| vocab | 32k | 262k |
| activation | silu (SwiGLU) | gelu (GeGLU) |
| shared expert | none | yes |

---

## 2. Correctness methodology

We do **not** require bit-identical logits — different kernels sum in different
orders. We measure agreement to the **bf16 precision floor** on identical
weights / seed:

- **top-1 match** — same argmax next-token (primary)
- **top-5 overlap**, **cosine**, **mean_abs** on the full last-position logit vector

### 2.1 Prefill (primary) — `compare_backends.py`

One prefill forward pass per prompt; logits at the **last position** = next-token
distribution. Not autoregressive.

```bash
python3 -m maxtext_moe.compare_backends \
  --model gemma \
  --backends ragged,fly \
  --load_parameters_path <ckpt>/0/items
```

Reports: `results/compare_{mixtral,gemma}_real/report.md`

**Results (real weights, bf16, 10 prompts):**

| model | top-1 (fly vs ragged) | cosine |
|---|---|---|
| Mixtral | **10/10** | 0.99998–1.00000 |
| Gemma | **10/10** | 0.99988–0.99998 |

Sensible identical predictions: `George`, `Jupiter`, ` plants`, ` lazy`, etc.

### 2.2 Decode — `compare_decode.py`

Real KV-cache path: `prefill → insert → generate()` loop. MoE runs at **tokens=1**
per decode step — different kernel regime than prefill.

```bash
python3 -m maxtext_moe.compare_decode \
  --model mixtral \
  --backends ragged,fly \
  --load_parameters_path <ckpt>/0/items \
  --gen_steps 8
```

Two metrics per run:

| metric | what it measures |
|---|---|
| **Free-running greedy** | Full `gen_steps` token sequences identical? |
| **Decode-step-1 logits** | After first `generate()`, same top-1 on identical context? |

**Mixtral (real weights, gen_steps=8):**

| metric | result |
|---|---|
| Full sequences exact | **10/10** (8/8 tokens each prompt) |
| Decode-step-1 top-1 | **10/10**, cosine 0.99989–1.0 |

Example matched text: `George Washington. He was born on February`, `100 degrees Celsius`,
`Jupiter. It is the fifth planet`.

Report: `results/decode_mixtral/report.md`

**Gemma (real weights):**

| metric | base ckpt | instruction-tuned ckpt |
|---|---|---|
| Full sequences exact | **0/10** (both backends degenerate) | 6/10 identical gibberish |
| Decode-step-1 top-1 | **5/10** | 7/10 |

Both backends fail the same way → **MaxText Gemma KV-cache decode bug**, not fly.

**Re-prefill diagnostic:** `prefill(prompt + token₁)` gives coherent token₂;
KV-cache `generate()` gives garbage. Prefill path is correct; cache path is not.

Reports: `results/decode_gemma_it/report.md` (and base ckpt runs in session logs).

### 2.3 Random-init Gemma (6/10) — not a fly bug

| check | result |
|---|---|
| ragged vs dense (no fly) | also diverges (~0.98 cosine) |
| fp32 | cosine 1.0, 4/4 top-1 |
| real trained weights | **10/10** top-1 |

Ruled out: gelu approximation, shared-expert handling, float32_weight_sum,
atomic-add nondeterminism. Conclusion: bf16 near-ties on random-init weights;
trained weights resolve it.

---

## 3. Performance (bf16 prefill, end-to-end, MI355X)

Baseline = stock MaxText `ragged_dot`. Single device, batch=1.

**Mixtral 8x7B**

| length | ragged (ms) | fly (ms) | speedup |
|---|---|---|---|
| 256 | 72.6 | 29.1 | 2.50× |
| 512 | 97.9 | 39.2 | 2.50× |
| 1024 | 174.9 | 61.3 | 2.85× |
| 2048 | 288.3 | 115.8 | 2.49× |

Decode (1 token): 69.8 ms → 19.7 ms = **3.54×**.

**Gemma4-26B (pretrained weights)**

| length | ragged (ms) | fly (ms) | speedup |
|---|---|---|---|
| 256 | 261.2 | 32.6 | 8.0× |
| 512 | 347.8 | 24.0 | 14.5× |
| 1024 | 645.2 | 29.6 | 21.8× |
| 2048 | 1013.5 | 45.8 | 22.1× |

Speedup scales with expert count (Mixtral E=8 vs Gemma E=128).

---

## 4. Why fly is faster than ragged on MI355

On JAX 0.8.2 / bf16 / gfx950, `ragged_dot` lowers to a **Triton masked batched dot**
over all E experts (`E×` FLOPs), not hipBLASLt grouped GEMM. FlyDSL runs a tuned
**grouped GEMM over exactly `tokens·topk` routed rows**, with fused activation +
expert-combine.

See overview §11 for HLO evidence and version/dtype caveats.

---

## 5. Reproduce

```bash
source /workspace/.flydsl_env
export MAXTEXT_DIR=/workspace/maxtext

# Prefill correctness
python3 -m maxtext_moe.compare_backends --model mixtral --backends ragged,fly \
  --load_parameters_path results/mixtral_ckpt/0/items

# Decode correctness
python3 -m maxtext_moe.compare_decode --model mixtral --backends ragged,fly \
  --load_parameters_path results/mixtral_ckpt/0/items --gen_steps 8

# End-to-end perf (via scripts)
BACKEND=fly bash scripts/run_mixtral_maxtext.sh
bash scripts/run_mixtral_maxtext.sh   # baseline
```
