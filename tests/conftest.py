# Automatically apply the 'decoupled' marker (when DECOUPLE_GCLOUD=TRUE) ONLY to
# tests that don't get skipped in decoupled mode. If a test is skipped because it
# requires TPU hardware (`tpu_only`) or any external integration
# (`external_serving`, `external_training`, `diagnostics`), we do NOT add the
# decoupled marker. This lets us select the decoupled tests using `-m decoupled` 
# easily.

import pytest
from MaxText.decouple import is_decoupled
try:
  import jax
  _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover
  _HAS_TPU = False


GCE_MARKERS = {"external_serving", "external_training"}

def pytest_collection_modifyitems(config, items):
  decoupled = is_decoupled()
  skip_no_tpu = None
  if not _HAS_TPU:
    skip_no_tpu = pytest.mark.skip(reason="Skipped: requires TPU hardware, none detected")
  # Adding the decoupled marker only to tests that aren't skipped.
  decoupled_marker = pytest.mark.decoupled if decoupled else None

  for item in items:
    cur_test_markers = {m.name for m in item.iter_markers()} # Iterate thru the markers of the cur test
    skip_flag = False
    if skip_no_tpu and "tpu_only" in cur_test_markers:
      item.add_marker(skip_no_tpu)
      skip_flag = True
    # Skip in decoupled mode if any external integration markers are present
    if decoupled:
      mutual_markers = cur_test_markers & GCE_MARKERS
      if mutual_markers:
        reason = (
            "Skipped: decoupled mode disables: "
            + ", ".join(sorted(mutual_markers))
        )
        item.add_marker(pytest.mark.skip(reason=reason))
        skip_flag = True
    # Add decoupled marker only if DECOUPLE_GCLOUD AND not skipped.
    if decoupled_marker and not skip_flag:
      item.add_marker(decoupled_marker)

def pytest_configure(config):
  for m in [
      "tpu_only: tests that require TPU hardware",
      "external_serving: JetStream / serving / decode server components",
      "external_training: SFT / tunix / goodput integrations",
      "decoupled: marked on tests that are not skipped due to GCE deps, when DECOUPLE_GCLOUD=TRUE",
  ]:
    config.addinivalue_line("markers", m)
