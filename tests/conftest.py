# Automatically apply the 'decoupled' marker when DECOUPLE_GCLOUD=TRUE so
# workflows can select these tests via -m decoupled even if tests themselves 
# don't specify it.

import os
import pytest


JETSTREAM_HINT_FILENAMES = [
    "decode_tests.py",
    "offline_engine_test.py",
    "maxengine_server",
    "grpo_trainer_correctness_test.py",  # may exercise serving pathways
]

JETSTREAM_MARKERS = {"jetstream", "tunix", "serving", "decode_server"}


def pytest_collection_modifyitems(config, items):
  decoupled = os.environ.get("DECOUPLE_GCLOUD", "").upper() == "TRUE"
  if not decoupled:
    return

  decoupled_marker = pytest.mark.decoupled
  skip_jetstream = pytest.mark.skip(
      reason="Skipped: JetStream / Tunix disabled by DECOUPLE_GCLOUD=TRUE"
  )

  for item in items:
    nodeid = item.nodeid

    # Preserve existing behavior: mark decoupled tests (except goodput utils)
    if "goodput_utils_test.py" not in nodeid and not any(m.name == "decoupled" for m in item.iter_markers()):
      item.add_marker(decoupled_marker)

    # Determine if test should be skipped due to JetStream dependency
    markers = {m.name for m in item.iter_markers()}
    filename = str(getattr(item, "fspath", ""))
    needs_jetstream = False

    if markers & JETSTREAM_MARKERS:
      needs_jetstream = True
    else:
      for hint in JETSTREAM_HINT_FILENAMES:
        if hint in filename or hint in nodeid:
          needs_jetstream = True
          break

    if needs_jetstream:
      item.add_marker(skip_jetstream)


def pytest_configure(config):
  config.addinivalue_line("markers", "jetstream: tests requiring JetStream serving components")
  config.addinivalue_line("markers", "tunix: tests requiring tunix components")
  config.addinivalue_line("markers", "serving: tests invoking server mode")
  config.addinivalue_line("markers", "decode_server: tests invoking decode server")
  config.addinivalue_line("markers", "decoupled: auto-marked when DECOUPLE_GCLOUD=TRUE")
