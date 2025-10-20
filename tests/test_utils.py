import os
from MaxText.globals import MAXTEXT_PKG_DIR

def get_test_config_path():
    """Centralized selection for returning absolute path to the test 
    config selected by DECOUPLE_GCLOUD env var (avoids code duplication).

    If DECOUPLE_GCLOUD=TRUE, use decoupled_base_test.yml else base.yml.
    """
    base_cfg = "base.yml"
    if os.environ.get("DECOUPLE_GCLOUD", "").upper() == "TRUE":
        base_cfg = "decoupled_base_test.yml"
    return os.path.join(MAXTEXT_PKG_DIR, "configs", base_cfg)

__all__ = ["get_test_config_path"]
