#!/usr/bin/env bash
# Installs jax-rocm10-pjrt/plugin at the installed jax's version, built for this container's ROCm SDK.
# AMD's index tags its wheels +rocm<sdk>; PyPI's carry no tag and name their ROCm build only in the
# package Summary. Either source is accepted, but the Summary must name `rocm-sdk version`, so a
# missing or mismatched build fails here rather than at device init.
#
# Usage: install_jax_rocm_plugin.sh <install command...>, e.g. `uv pip install [--system]`.
set -euo pipefail

INSTALL=("$@")
JAX_VER="$(python3 -c 'import importlib.metadata as m; print(m.version("jax"))')"
ROCM_SDK_VER="$(rocm-sdk version)"
PKGS=(jax-rocm10-pjrt jax-rocm10-plugin)

if ! "${INSTALL[@]}" --no-deps --force-reinstall --index-url https://stable.repo.amd.com/rocm/jax/whl-next/ \
    "${PKGS[@]/%/==${JAX_VER}+rocm${ROCM_SDK_VER}}"; then
  echo "No ${JAX_VER}+rocm${ROCM_SDK_VER} build on AMD's index; installing ${JAX_VER} from PyPI."
  "${INSTALL[@]}" --no-deps --force-reinstall "${PKGS[@]/%/==${JAX_VER}}"
fi

python3 - "${ROCM_SDK_VER}" <<'PY'
import importlib.metadata as m
import sys

want = f"ROCm:{sys.argv[1]}"
for pkg in ("jax-rocm10-pjrt", "jax-rocm10-plugin"):
  summary = m.metadata(pkg)["Summary"]
  if want not in summary:
    sys.exit(f"{pkg} {m.version(pkg)} is built for {summary!r}, but this container has {want}")
  print(f"{pkg} {m.version(pkg)}: {summary}")
PY
