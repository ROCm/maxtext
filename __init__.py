<<<<<<< HEAD
"""Top-level shim for importing test_utils
The actual source package lives under `src/MaxText`. Installing the project
(`pip install -e .` from the repository root) adds `maxtext` to sys.path via
standard packaging. When running tests directly from the `maxtext/` directory
without installation, `from maxtext.tests.test_utils` will fail because Python
sees the current working directory as a plain path, not a package.

This shim lets test modules import `maxtext.tests`.

"""

from importlib import import_module as _imp

try:
    test_utils = _imp("maxtext.tests.test_utils")  # noqa: F401
except Exception:  # pragma: no cover - fail silently if tests not present
    pass

__all__ = ["test_utils"]
=======
"""Top-level shim for lowercase `maxtext` imports.

The actual source package lives under `src/MaxText`. Installing the project
(`pip install -e .` from the repository root) adds `maxtext` to sys.path via
standard packaging. When running tests directly from the `maxtext/` directory
without installation, `from maxtext.tests.test_utils` will fail because Python
sees the current working directory as a plain path, not a package.

This shim lets test modules reliably import `maxtext.tests` even in ad-hoc
execution contexts. If you prefer not to have this file, always run:

    cd ..  # repository root containing pyproject.toml
    uv pip install -e .
    pytest

"""

from importlib import import_module as _imp

# Re-export frequently used test utilities for convenience.
try:
    test_utils = _imp("maxtext.tests.test_utils")  # noqa: F401
except Exception:  # pragma: no cover - fail silently if tests not present
    pass

__all__ = ["test_utils"]
>>>>>>> 99eb0988 (layout_compat changes)
