# MaxText Tests Guide

This document summarizes decoupling conventions used across the test suite.

## Config Selection (Decoupling)
All tests that previously hard-coded `configs/base.yml` now use the helper `get_test_config_path()` from `tests/test_utils.py`. This helper ensures usage of `decoupled_base_test.yml` if  `DECOUPLE_GCLOUD=TRUE`

## Cloud vs Local Paths
Some tests need to reference datasets or output directories. We decouple these to allow local execution without GCS access.

Current patterns:
- Output directories often use `gs://runner-maxtext-logs` (or `gs://max-experiments/`).
- Dataset paths often use `gs://maxtext-dataset`.
- If decouple mode, local mini dataset is accessed and logs are output to local logs directory.

Example conditional approach:
```python
from MaxText.decouple import is_decoupled
decoupled = is_decoupled()
_dataset_path = os.path.join(MAXTEXT_PKG_DIR, "..", "decoupled_datasets", "c4_en_dataset_minimal") if decoupled else "gs://maxtext-dataset"
_base_output_directory = (
    os.path.join(MAXTEXT_PKG_DIR, "..", "decoupled_datasets", "gcloud_decoupled_test_logs") if decoupled else "gs://runner-maxtext-logs"
)
```

## When Adding New Tests
1. Use `get_test_config_path()` instead of hard-coded `base.yml`.
2. Prefer conditional local fallbacks for cloud buckets if practical.
4. Avoid introducing direct `gs://...` paths.

## Shell Scripts
Inference shell scripts (`test_llama2_7b_bf16.sh`, `test_llama2_7b_int8.sh`) manually select config paths and load parameters. They support decoupling by swapping the config if `DECOUPLE_GCLOUD=TRUE`. 

## Remaining GCE-Dependent Elements
Some tests still require real checkpoints, so creating minimal testing checkpoints for local execution is a TODO.
Similarly, some tests require tokenizers from remote sources. Adding these to local execution destinations is a TODO (max 20 MB). 

## Pytest Markers
The following markers are used to skip tests in DECOUPLED_GCLOUD=TRUE:

* `external_serving` – JetStream / serving / decode server components AND tests that require pre-generated checkpoints (skipped when `DECOUPLE_GCLOUD=TRUE`).
* `external_training` – Tunix / SFT / goodput integrations (skipped when `DECOUPLE_GCLOUD=TRUE`).
* `tpu_only` – Requires TPU hardware; skipped automatically if no TPU devices are detected.
* `decoupled` – Auto-applied by `tests/conftest.py` to tests that are runnable in decoupled mode (i.e. not skipped for TPU or external markers).

Guidelines:
1. Only add `external_serving` or `external_training` where a test actually invokes JetStream/MaxEngine serving pathways or uses external SFT/Tunix/goodput operations (e.g. gsutil downloads, tokenizer cloud paths, goodput logging client).
2. Prefer the smallest scope instead of module-wide `pytestmark` when only a part of a file needs an external dependency.

To run only offline-safe tests in decoupled mode:
```bash
export DECOUPLE_GCLOUD=TRUE
pytest -m decoupled
```

