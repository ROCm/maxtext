# Automatically apply the 'decoupled' marker when DECOUPLE_GCLOUD=TRUE so
# workflows can select these tests via -m decoupled even if tests themselves 
# don't specify it.

import os
import pytest


def pytest_collection_modifyitems(config, items):
  if os.environ.get("DECOUPLE_GCLOUD", "").upper() != "TRUE":
    return
  decoupled_marker = pytest.mark.decoupled
  for item in items:
    # Do not auto-mark goodput utils tests; they rely on GCP and will self-skip when decoupled
    if "goodput_utils_test.py" in item.nodeid:
      continue
    # Avoid duplicating marker if already present
    if not any(m.name == "decoupled" for m in item.iter_markers()):
      item.add_marker(decoupled_marker)
