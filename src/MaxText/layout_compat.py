"""Minimal Layout/Format compatibility shim."""

import jax
from jax.experimental import layout as _layout  # type: ignore

Layout = _layout.Layout  # always present

# Parse version (major, minor, patch) ignoring local build metadata.
_ver_parts = jax.__version__.split('+')[0].split('.')
_ver = tuple(int(p) for p in _ver_parts[:3])

if _ver >= (0, 7, 0):  # JAX >= 0.7.0 -> real Format exists
    try:
        Format = _layout.Format  # type: ignore[attr-defined]
    except AttributeError:  # unexpected missing Format; fallback defensively
        Format = Layout  # type: ignore
else:  # jax <= 0.6.0
    Format = Layout  # type: ignore

# Device local layout provider; AUTO lives on Layout for newer versions.
if _ver >= (0, 7, 0):  # AUTO moved to Layout by >= (0, 7, 0)
    DLL = Layout
else:
    DLL = getattr(_layout, "DeviceLocalLayout", Layout)
