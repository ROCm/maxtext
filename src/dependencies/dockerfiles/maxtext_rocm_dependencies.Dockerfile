# syntax=docker/dockerfile:1.7

# ROCm (TheRock) MaxText image. Downstream-only (ROCm/maxtext fork).
# Base image ships the ROCm SDK (rocm-sdk) system-wide; we add JAX 0.11.1, the jax-rocm
# pjrt/plugin, the MaxText package, and the ROCm Transformer Engine wheel.
ARG BASEIMAGE=ghcr.io/rocm/jax-dev-ubu24.therock-10.1:10.1
FROM $BASEIMAGE

ENV DEBIAN_FRONTEND=noninteractive
# TheRock runtime perf workarounds (see run_tests_against_package.yml / container-testing skill).
ENV DEBUG_HIP_DYNAMIC_QUEUES=0
ENV ROCPROFILER_QUEUE_INTERPOSITION=0
ENV NVTE_FUSED_ATTN_AOTRITON=0

ENV MAXTEXT_REPO_ROOT=/deps
ENV MAXTEXT_ASSETS_ROOT=/deps/src/maxtext/assets
ENV MAXTEXT_TEST_ASSETS_ROOT=/deps/tests/assets
ENV MAXTEXT_PKG_DIR=/deps/src/maxtext
ENV PYTHONPATH="/deps/src${PYTHONPATH:+:${PYTHONPATH}}"
WORKDIR /deps

# jax-rocm pjrt/plugin version, from AMD's index; overridable at build time. Must equal the jax
# version in rocm-requirements.txt. The +rocm local version is BASEIMAGE's `rocm-sdk version`.
ARG ROCM_JAX_VERSION=0.11.1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install -U uv

# Copy the (already upstream-synced) repo into the image.
COPY . /deps

# Install the ROCm/JAX stack into the image's system Python (which already has rocm-sdk):
#   1. jax/jaxlib (PyPI, pinned) + MaxText runtime/test deps from the rocm requirements file
#   2. the MaxText package itself (--no-deps; deps came from the requirements file)
#   3. jax-rocm pjrt + plugin built for the image's ROCm SDK (AMD index)
#   4. the ROCm Transformer Engine wheel, if the workflow staged one into the build context
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eux; \
    export UV_LINK_MODE=copy; \
    uv pip install --system -r src/dependencies/requirements/base_requirements/rocm-requirements.txt; \
    uv pip install --system . --no-deps; \
    ROCM_SDK_VER="$(rocm-sdk version)"; \
    uv pip install --system --no-deps --index-url https://stable.repo.amd.com/rocm/jax/whl-next/ \
      "jax-rocm10-pjrt==${ROCM_JAX_VERSION}+rocm${ROCM_SDK_VER}" \
      "jax-rocm10-plugin==${ROCM_JAX_VERSION}+rocm${ROCM_SDK_VER}"; \
    TE_WHL="$(ls /deps/transformer_engine-*.whl 2>/dev/null | sort | tail -n1 || true)"; \
    if [ -n "${TE_WHL}" ]; then uv pip install --system --no-deps --force-reinstall "${TE_WHL}"; fi

# Batch training/test image: no long-running service to probe.
HEALTHCHECK NONE

# Stays root like its ROCm base image: GPU access (/dev/kfd, /dev/dri) and in-container
# installs assume it.
# trivy:ignore:DS-0002
USER root
