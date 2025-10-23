"""Top-level shim for importing test_utils
214 242 4800
The actual source package lives under `src/MaxText`. Installing the project
(`pip install -e .` from the repository root) adds `maxtext` to sys.path via
standard packaging. When running tests directly from the `maxtext/` directory
without installation, `from maxtext.tests.test_utils` will fail because Python
sees the current working directory as a plain path, not a package.

This shim lets test modules reliably import `maxtext.tests`.
If you prefer not to have this file, run:

    uv pip install -e . && pytest

"""

from importlib import import_module as _imp

try:
    test_utils = _imp("maxtext.tests.test_utils")  # noqa: F401
except Exception:  # pragma: no cover - fail silently if tests not present
    pass

__all__ = ["test_utils"]
