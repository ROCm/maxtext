"""Lightweight environment smoke test for MaxText development.

Run this after setting up your virtual environment to confirm core libs and GPU visibility.

Usage:
  python env_smoke_test.py

Optional env vars:
  DECOUPLE_GCLOUD=TRUE  # Will note JetStream disabled state.
"""
from __future__ import annotations
import os, sys, time, importlib

REPORT = []

def section(title: str):
    print(f"\n=== {title} ===")


def add(k: str, v):
    REPORT.append((k, v))


def check_import(name: str):
    t0 = time.time()
    try:
        mod = importlib.import_module(name)
        dt = time.time() - t0
        add(f"import:{name}", f"OK ({dt:.2f}s)")
        return mod
    except Exception as e:
        add(f"import:{name}", f"FAIL: {e}")
        return None


def main():
    section("Environment")
    add("python", sys.version.split()[0])
    add("platform", sys.platform)
    add("DECOUPLE_GCLOUD", os.environ.get("DECOUPLE_GCLOUD"))

    section("Core Imports")
    jax = check_import("jax")
    check_import("jax.numpy")
    flax = check_import("flax")
    transformers = check_import("transformers")
    numpy = check_import("numpy")

    section("MaxText Imports")
    mt = check_import("MaxText")
    check_import("MaxText.pyconfig")
    check_import("MaxText.maxengine")

    section("JAX Device Info")
    if jax:
        try:
            devices = jax.devices()
            add("jax.devices.count", len(devices))
            kinds = sorted({d.platform for d in devices})
            add("jax.platforms", ",".join(kinds))
            # Show first GPU/TPU device if present
            for d in devices[:3]:
                print(f" - {d.id}: {d.platform} {d.device_kind}")
        except Exception as e:
            add("jax.devices", f"ERROR: {e}")

    section("Decoupled Mode Status")
    decoupled = os.environ.get("DECOUPLE_GCLOUD", "").upper() == "TRUE"
    if decoupled:
        print("JetStream disabled (DECOUPLE_GCLOUD=TRUE). Serving-related tests will be skipped.")
    else:
        print("JetStream enabled. Ensure google-jetstream/tunix packages installed if needed.")

    section("Summary")
    width = max(len(k) for k, _ in REPORT) if REPORT else 10
    for k, v in REPORT:
        print(f"{k.ljust(width)} : {v}")


if __name__ == "main":  # typo guard if user copies snippet incorrectly
    main()

if __name__ == "__main__":
    unittest.main()

