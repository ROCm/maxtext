# MaxText Temp Memory Autoresearch (NVIDIA H100)

Autonomous experiment loop for reducing JAX "Temp size" in MaxText training
on NVIDIA H100 using upstream `main`.

Read `/work/cj-temp-mem-autoresearch.md` for full background, prior fix analysis,
and remaining hypotheses.

---

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `tmem-may5`).
   The branch `autoresearch/<tag>` must not already exist.

2. **Create the branch** from `main`:
   ```bash
   GIT_DIR=/work/maxtext/.git git checkout main
   GIT_DIR=/work/maxtext/.git git checkout -b autoresearch/<tag>
   ```

3. **Read these files for full context** (do not skip):
   - `/work/cj-temp-mem-autoresearch.md` — background, prior ROCm fixes, remaining hypotheses
   - `src/maxtext/layers/pipeline.py` — pipeline parallelism (largest source of temp)
   - `src/maxtext/layers/moe.py` — MoE GMM operations and tile tuples
   - `src/maxtext/layers/attention_op.py` — attention mask materialization
   - `src/maxtext/utils/sharding.py` — sharding constraint helpers
   - `src/maxtext/layers/decoders.py` — logits sharding, shared_embedding
   - `src/maxtext/layers/normalizations.py` — RMSNorm einsum vs multiply
   - `/work/jax-memory-bug-reproduce-2026-jan/ds-proxy-N1-ep2-pp4.yml` — the config under test

4. **Initialize results.tsv** in `/work/jax-memory-bug-reproduce-2026-jan/` with just the header row:
   ```
   commit	temp_gb	output_gb	status	description
   ```
   The baseline will be recorded after the first run.

5. **Confirm and go.**

---

## Repository layout

All code lives in a single repo at `/work/maxtext` with these branches:

| Branch | Purpose |
|---|---|
| `main` | Upstream MaxText — the starting point for experiments |
| `rocm-main` | ROCm fork baseline (before fixes) |
| `cj-fix-tmp-mem_rocm-main` | ROCm fork with temp-mem fixes — **reference only** |
| `autoresearch/<tag>` | Your experiment branch (created from `main`) |

Since git commands outside `/work/maxtext` won't find the repo, always use:
```bash
GIT_DIR=/work/maxtext/.git git <command>
```
Or `cd /work/maxtext && git <command>`.

The fix branch `cj-fix-tmp-mem_rocm-main` is the reference for what to port.
It targets ROCm but the temp-memory optimizations are hardware-agnostic.
Use `git diff` against it to see working implementations — do NOT copy blindly,
as some changes include ROCm-specific code:
```bash
GIT_DIR=/work/maxtext/.git git diff main..cj-fix-tmp-mem_rocm-main -- src/maxtext/layers/pipeline.py
```

---

## Running an experiment

The run script handles all env vars, XLA flags, and logging. Pass the maxtext
directory as the variant argument:

```bash
cd /work/jax-memory-bug-reproduce-2026-jan
bash single-node-ep2-pp4-maxtext.sh /work/maxtext > run.log 2>&1
```

The script runs `STEPS=10` by default (enough to trigger XLA compilation and get
the memory report). It tees output to `/work/log_ep2_pp4_maxtext.log`
and prints a memory summary at the end.

Extract the key metrics:

```bash
grep "Total memory size" run.log
# Expected format:
# Total memory size: 60.3 GB, Output size: 14.6 GB, Temp size: 45.7 GB, ...
```

If the grep is empty, the run crashed. Check:

```bash
tail -60 run.log
```

---

## What you CAN modify

Edit files only within `/work/maxtext/src/maxtext/`. All files are in scope.
Priority targets based on known temp memory sources (confirmed present in `main`):

| File | Known temp memory source | Already fixed in ROCm fix branch? |
|---|---|---|
| `layers/pipeline.py` | `shard_map`+`ppermute` rotation — hoisted into scan carry | Yes — port to main |
| `layers/moe.py` | 9-tuple GMM tile size includes unused backward-pass tiles | Yes — port to main |
| `layers/normalizations.py` | `einsum` vs `multiply` in RMSNorm | Yes — port to main |
| `layers/embeddings.py` | `out_sharding` on embedding lookup | Yes — port to main |
| `layers/decoders.py` | `with_logical_constraint` placement on logits | Yes — port to main |
| `utils/sharding.py` | No `skip_trivial_specs` guard yet | Yes — port to main |
| `configs/base.yml` | `float32_weight_sum: true` default | Yes — port to main |
| `trainers/pre_train/train.py` | Unconditional `nan_to_num` on grads, redundant dtype cast | Yes — port to main |
| `layers/attention_op.py` | Synthetic data mask shortcut | Yes — port to main |

To see the exact diff for any file:
```bash
GIT_DIR=/work/maxtext/.git git diff main..cj-fix-tmp-mem_rocm-main -- src/maxtext/layers/pipeline.py
```

**Important caveats when porting from the ROCm fix branch:**
- The ROCm branch diverges from `main` in ways unrelated to temp memory
  (ROCm-specific attention paths, different TE integration, etc.)
- Always diff against `main`, not `rocm-main`, to understand what `main` looks like
- Test each change independently — some fixes interact

## What you CANNOT modify

- `/work/jax-memory-bug-reproduce-2026-jan/ds-proxy-N1-ep2-pp4.yml` — the reference config
- `/work/jax-memory-bug-reproduce-2026-jan/single-node-ep2-pp4-maxtext.sh` — the run script
- The `main` branch — experiments go on the `autoresearch/<tag>` branch only

---

## The metric

**Goal: minimize `Temp size`** (GB) from `max_utils.py` compiled memory analysis.

**Baseline on `main`:** Run the baseline first and record it.

**Reference: ROCm fix branch achieved:** Temp size = 30.3 GB (same model, same config)
on ROCm MI300X. H100 numbers will differ due to XLA backend differences, but the
relative savings should be similar.

**Soft constraint:** `Output size` must not increase (that would indicate a weight
shape change or unexpected replication). Training must not crash.

---

## Output format

```
Total memory size: 60.3 GB, Output size: 14.6 GB, Temp size: 45.7 GB, Argument size: 14.6 GB, Host temp size: 0.0 GB.
```

Extract just temp and output:

```bash
grep "Total memory size" run.log | grep -oP "Temp size: \K[0-9.]+"
grep "Total memory size" run.log | grep -oP "Output size: \K[0-9.]+"
```

---

## Logging results

Log each experiment to `/work/jax-memory-bug-reproduce-2026-jan/results.tsv`
(tab-separated, NOT comma-separated).

```
commit	temp_gb	output_gb	status	description
```

1. `commit` — 7-char git hash
2. `temp_gb` — Temp size in GB (e.g. `45.7`) — use `0.0` for crashes
3. `output_gb` — Output size in GB (e.g. `14.6`) — use `0.0` for crashes
4. `status` — `keep`, `discard`, or `crash`
5. `description` — short text, no commas

Example:
```
commit	temp_gb	output_gb	status	description
a1b2c3d	45.7	14.6	keep	baseline main
b2c3d4e	40.1	14.6	keep	pipeline slice+concat replaces shard_map ppermute
c3d4e5f	45.6	14.6	discard	remat scope change had no effect
d4e5f6g	0.0	0.0	crash	pipeline carry quantization OOM
```

Do NOT commit `results.tsv`. Leave it untracked.

---

## The experiment loop

LOOP FOREVER:

1. Check git state: `GIT_DIR=/work/maxtext/.git git log --oneline -3`

2. Form a hypothesis about what is inflating temp size. Consult:
   - Section 5 of `/work/cj-temp-mem-autoresearch.md` (remaining opportunities)
   - The ROCm fix branch for reference implementations:
     `GIT_DIR=/work/maxtext/.git git diff main..cj-fix-tmp-mem_rocm-main -- src/maxtext/`
   - The HLO dumps in `/work/jax-memory-bug-reproduce-2026-jan/hlo_dump_rocm_main/`

3. Edit the relevant file(s) in `/work/maxtext/src/maxtext/`.

4. Commit:
   ```bash
   GIT_DIR=/work/maxtext/.git git add -A src/maxtext/
   GIT_DIR=/work/maxtext/.git git commit -m "short description"
   ```

5. Run:
   ```bash
   cd /work/jax-memory-bug-reproduce-2026-jan
   bash single-node-ep2-pp4-maxtext.sh /work/maxtext > run.log 2>&1
   ```
   Timeout: if the run exceeds 15 minutes, kill it and treat as crash.

6. Check results:
   ```bash
   grep "Total memory size" run.log
   ```
   If empty → crash → `tail -60 run.log` for the stack trace.

7. Record in `results.tsv`.

8. If `temp_gb` improved (lower) **and** `output_gb` did not increase: **keep** the commit.

9. If not improved: `GIT_DIR=/work/maxtext/.git git reset --hard HEAD~1` (discard).

---

## Hypotheses to try (priority order)

Work through these in order. Skip ones already in `results.tsv`.
All of these are confirmed present in `main` and fixed in the ROCm fix branch.

1. **Pipeline `shard_map`+`ppermute` → slice+concat** (`layers/pipeline.py`)
   The `_rotate_right` and `_shift_right` functions use `@jax.shard_map` with
   `jax.lax.ppermute`. XLA's `loop_broadcast_fusion` hoists the ppermute buffers
   into the `lax.scan` carry, inflating temp by ~10-15 GB for pp=4.
   Replace with `jax.lax.slice_in_dim` + `jnp.concatenate` (no shard_map needed).
   Also replace `_update_state_io` shard_map with slice-based equivalents.
   **Diff:** `GIT_DIR=/work/maxtext/.git git diff main..cj-fix-tmp-mem_rocm-main -- src/maxtext/layers/pipeline.py`

2. **GMM tile tuple: 9-element → 3-element** (`layers/moe.py`)
   `wi_tile_size` and `wo_tile_size` are 9-tuples including 6 backward-pass tile
   values. The ds-proxy config uses `megablox=False` (JAX ragged_dot path), so
   backward tiles are allocated but never used. Truncate to the 3-element forward
   tuple `(fwd_batch_seq, fwd_embed_dim, fwd_mlp_dim)`.
   Also remove unnecessary `.astype(self.dtype)` on `intermediate_layer` return.
   **Diff:** `GIT_DIR=/work/maxtext/.git git diff main..cj-fix-tmp-mem_rocm-main -- src/maxtext/layers/moe.py`

3. **Attention mask materialization shortcut** (`layers/attention_op.py`)
   For `dataset_type=synthetic`, all segment IDs are ones (single segment per
   sequence), so the segment mask degenerates to pure causal masking. The full
   mask computation creates large tensors that XLA hoists into the pipeline scan
   carry (+5 GB temp). Add a shortcut that sets `mask_type="causal"` and skips
   `generate_attention_mask` for synthetic data.
   **Diff:** `GIT_DIR=/work/maxtext/.git git diff main..cj-fix-tmp-mem_rocm-main -- src/maxtext/layers/attention_op.py`

4. **`float32_weight_sum: true → false`** (`configs/base.yml`)
   The default `true` forces MoE expert weight summation in fp32, adding ~2 GB
   f32 temp per device. Change to `false`.
   **Diff:** `GIT_DIR=/work/maxtext/.git git diff main..cj-fix-tmp-mem_rocm-main -- src/maxtext/configs/base.yml`

5. **`skip_trivial_specs` guard in sharding** (`utils/sharding.py`)
   Add a `skip_trivial_specs` parameter to `maybe_shard_with_logical`. When all
   axes resolve to `None`/`()`, return early without inserting a
   `with_sharding_constraint` copy. Then pass `skip_trivial_specs=True` at call
   sites in `pipeline.py` and `layers/attention_op.py`.
   **Diff:** `GIT_DIR=/work/maxtext/.git git diff main..cj-fix-tmp-mem_rocm-main -- src/maxtext/utils/sharding.py`

6. **RMSNorm `einsum → multiply`** (`layers/normalizations.py`)
   `jnp.einsum("...k,k->...k", y, effective_scale, out_sharding=...)` can generate
   larger XLA temporaries than a direct `y * effective_scale`. Replace with multiply
   and apply `with_sharding_constraint` explicitly only when `out_sharding` is set.
   **Diff:** `GIT_DIR=/work/maxtext/.git git diff main..cj-fix-tmp-mem_rocm-main -- src/maxtext/layers/normalizations.py`

7. **Logits sharding annotation** (`layers/decoders.py`)
   Add `nn.with_logical_constraint` on the logits output in `apply_output_head`
   with `("activation_embed_and_logits_batch", "activation_length", "activation_vocab")`.
   This gives XLA a stable sharding for the fp32 logits tensor
   `[batch, seq, vocab=102400]` and prevents unexpected replication.
   Also: move `shared_embedding` from a call-site argument to a module field.
   **Diff:** `GIT_DIR=/work/maxtext/.git git diff main..cj-fix-tmp-mem_rocm-main -- src/maxtext/layers/decoders.py`

8. **Unconditional `nan_to_num` and `grad_dtype` cast** (`trainers/pre_train/train.py`)
   Remove unconditional `jnp.nan_to_num` on all gradients (keep fp8 path only).
   Guard `grad_dtype` cast with `if config.grad_dtype != jnp.float32`.
   **Diff:** `GIT_DIR=/work/maxtext/.git git diff main..cj-fix-tmp-mem_rocm-main -- src/maxtext/trainers/pre_train/train.py`

9. **Embeddings sharding cleanup** (`layers/embeddings.py`)
   Add `nn.with_logical_constraint` after the embedding lookup. Make
   `out_pspec`/`out_sharding` conditional on `shard_mode == EXPLICIT`.
   **Diff:** `GIT_DIR=/work/maxtext/.git git diff main..cj-fix-tmp-mem_rocm-main -- src/maxtext/layers/embeddings.py`

10. **`scan_layers=False` diagnostic** (config override, not a real fix)
    Pass `scan_layers=False` as a CLI override to quantify how much temp comes from
    the `lax.scan` carry vs other sources. Do NOT keep this as a permanent change —
    it is slower. Record in tsv as `diagnostic`.

---

## If you get stuck

- Diff against the ROCm fix branch for a working reference:
  ```bash
  GIT_DIR=/work/maxtext/.git git diff main..cj-fix-tmp-mem_rocm-main -- src/maxtext/layers/pipeline.py
  ```
- Check the HLO dump for the largest temp buffers:
  ```bash
  ls /work/jax-memory-bug-reproduce-2026-jan/hlo_dump_rocm_main/
  ```
- Reduce model size for fast iteration: pass `base_num_decoder_layers=4` as a
  CLI override to confirm hypothesis direction before running the full 16-layer config.
- The `main` branch may have structural differences from `rocm-main` (different
  function signatures, moved code, etc). Always read the `main` version of a file
  before applying a fix — do not assume the fix branch patch applies cleanly.
- If a change crashes, check whether it's a merge conflict with a `main`-only change
  (e.g. new function parameters, renamed variables) before discarding the hypothesis.

**NEVER STOP.** Once the loop has begun, do not pause to ask if you should continue.
Run until manually interrupted.
