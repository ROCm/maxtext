"""Centralized decoupling helpers for JetStream / Tunix / cloud diagnostics.

Set DECOUPLE_GCLOUD=TRUE in the environment to disable optional Google Cloud / JetStream / Tunix
integrations while still allowing local unit tests to import modules. This module provides:

- is_decoupled(): returns True if decoupled flag set.
- cloud_diagnostics(): tuple(diagnostic, debug_configuration, diagnostic_configuration, stack_trace_configuration)
  providing either real objects or lightweight stubs.
- jetstream(): returns a namespace-like object exposing Engine, Devices, ResultTokens etc. or stubs.
- tunix(): returns peft_trainer, DataHooks, TrainingHooks stubs or real imports if available and not decoupled.

All stubs raise RuntimeError only when actually invoked, not at import time, so test collection proceeds.
"""
from __future__ import annotations

from types import SimpleNamespace
import importlib.util
import os

_DECOUPLE = os.environ.get("DECOUPLE_GCLOUD", "").upper() == "TRUE"

def is_decoupled() -> bool:
    return _DECOUPLE

# ---------------- Cloud Diagnostics -----------------

def _cloud_diag_stubs():
    class _StubDiag:
        def run(self, *_a, **_k):
            return {"status": "skipped"}
        def diagnose(self, *_a, **_k):
            # Return a context manager that gracefully handles any errors
            import contextlib
            @contextlib.contextmanager
            def _graceful_diagnose():
                try:
                    yield
                except Exception as e:
                    # Log error but don't crash
                    print(f"Warning: Using stubs in decoupling mode for cloud_diagnostics replacement. This stub is for diagnose function: {e}")
            return _graceful_diagnose()
    class _StubDebugConfig:
        def __init__(self, *a, **k):
            pass
    class _StubStackTraceConfig:
        def __init__(self, *a, **k):
            pass
    class _StubDiagnosticConfig:
        def __init__(self, debug_config=None, *a, **k):
            self.debug_config = debug_config
    return (
        _StubDiag(),
        SimpleNamespace(DebugConfig=_StubDebugConfig, StackTraceConfig=_StubStackTraceConfig),
        SimpleNamespace(DiagnosticConfig=_StubDiagnosticConfig),
        SimpleNamespace(StackTraceConfig=_StubStackTraceConfig),
    )

def cloud_diagnostics():
    """Return cloud diagnostics modules if installed; otherwise lightweight stubs."""
    try:
        from cloud_tpu_diagnostics import diagnostic  # type: ignore
        from cloud_tpu_diagnostics.configuration import (  # type: ignore
            debug_configuration,
            diagnostic_configuration,
            stack_trace_configuration,
        )
        return diagnostic, debug_configuration, diagnostic_configuration, stack_trace_configuration
    except ModuleNotFoundError:
        return _cloud_diag_stubs()

# ---------------- JetStream -----------------

def _jetstream_stubs():
    class _MissingCallable:
        def __init__(self, name: str):
            self._name = name
        def __call__(self, *a, **k):
            raise RuntimeError(f"JetStream dependency '{self._name}' unavailable (DECOUPLE_GCLOUD=TRUE)")
    class _StubModule(SimpleNamespace):
        def __getattr__(self, item):  # pragma: no cover
            return _MissingCallable(item)
    return _StubModule(), _StubModule(), _StubModule(), _StubModule(), _StubModule()

def jetstream():
    if _DECOUPLE:
        return _jetstream_stubs()
    # Only import if modules are discoverable.
    needed = [
        "jetstream.core.config_lib",
        "jetstream.engine.engine_api",
        "jetstream.engine.token_utils",
        "jetstream.engine.tokenizer_api",
        "jetstream.engine.tokenizer_pb2",
    ]
    for mod in needed:
        if importlib.util.find_spec(mod) is None:
            return _jetstream_stubs()
    from jetstream.core import config_lib  # type: ignore
    from jetstream.engine import engine_api, token_utils, tokenizer_api  # type: ignore
    from jetstream.engine.tokenizer_pb2 import TokenizerParameters, TokenizerType  # type: ignore
    return config_lib, engine_api, token_utils, tokenizer_api, SimpleNamespace(TokenizerParameters=TokenizerParameters, TokenizerType=TokenizerType)

# ---------------- Tunix -----------------

def _tunix_stubs():
    class DataHooks:  # simple base type
        def __init__(self, *a, **k):
            pass
    class TrainingHooks:  # simple base type
        def __init__(self, *a, **k):
            pass
    class _StubPeftTrainer:
        def __init__(self, *a, **k):
            pass
    peft_trainer = SimpleNamespace(PeftTrainer=_StubPeftTrainer)
    hooks = SimpleNamespace(DataHooks=DataHooks, TrainingHooks=TrainingHooks)
    return peft_trainer, hooks

def tunix():
    if _DECOUPLE:
        return _tunix_stubs()
    if importlib.util.find_spec("tunix") is None:
        return _tunix_stubs()
    try:
        from tunix.sft import peft_trainer  # type: ignore
        from tunix.sft import hooks as tunix_hooks  # type: ignore
        return peft_trainer, tunix_hooks
    except ModuleNotFoundError:
        return _tunix_stubs()

__all__ = ["is_decoupled", "cloud_diagnostics", "jetstream", "tunix"]
