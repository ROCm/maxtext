# MaxText Tests Guide

This document summarizes conventions used across the test suite, especially around configuration decoupling and cloud resource references.

## Config Selection (Decoupling)
All tests that previously hard-coded `configs/base.yml` now use the helper `get_test_config_path()` from `tests/test_utils.py`.

Behavior:
- When the environment variable `DECOUPLE_GCLOUD=TRUE`, the helper returns `decoupled_base_test.yml`.
- Otherwise it returns `base.yml`.

Usage pattern:
```python
from maxtext.tests.test_utils import get_test_config_path
argv = [None, get_test_config_path(), "enable_checkpointing=False"]
```

## Cloud vs Local Paths
Some tests need to reference datasets or output directories. We are progressively decoupling these to allow local execution without GCS access.

Current patterns:
- Output directories often use `gs://runner-maxtext-logs` (or `gs://max-experiments/`).
- Dataset paths often use `gs://maxtext-dataset`.
- e.g. `decode_tests.py` shows the recommended conditional pattern: it picks local ROCm fixture data and a local logs directory if `DECOUPLE_GCLOUD=TRUE`.

Example conditional approach:
```python
decoupled = os.environ.get("DECOUPLE_GCLOUD", "").upper() == "TRUE"
_dataset_path = os.path.join(MAXTEXT_PKG_DIR, "..", "rocm", "c4_en_dataset_minimal") if decoupled else "gs://maxtext-dataset"
_base_output_directory = (
    os.path.join(MAXTEXT_PKG_DIR, "..", "rocm", "gcloud_decoupled_test_logs") if decoupled else "gs://runner-maxtext-logs"
)
```

## When Adding New Tests
1. Use `get_test_config_path()` instead of hard-coded `base.yml`.
2. Prefer conditional local fallbacks for cloud buckets if practical.
4. Avoid introducing new direct `gs://...` strings.

## Shell Scripts
Inference shell scripts (`test_llama2_7b_bf16.sh`, `test_llama2_7b_int8.sh`) manually select config paths and load parameters. They support decoupling by swapping the config if `DECOUPLE_GCLOUD=TRUE`. Checkpoints still require cloud access.

## Intentional Remaining Cloud References
Some tests require real checkpoints (e.g. vision encoder) or large datasets and are left pointing at `gs://` URIs. Will introduce local checkpoints/directories.

## Summary
- Config decoupling is centralized with `get_test_config_path()`.
- Local paths are used if DECOUPLE_GCLOUD=TRUE for both inputs and outputs.

