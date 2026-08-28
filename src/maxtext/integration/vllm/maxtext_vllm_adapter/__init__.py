# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MaxText vLLM adapter package.

This package is platform-agnostic: it names no particular vLLM hardware plugin.
It previously imported `tpu_inference` at module scope for a logger and a model
registry, which made the adapter importable only on TPU even though the model it
wraps is not TPU-specific.
"""

import logging

from .adapter import MaxTextForCausalLM


logger = logging.getLogger(__name__)

MODEL_NAME = "MaxTextForCausalLM"

__all__ = ["MaxTextForCausalLM", "MODEL_NAME", "register"]


def register(register_model=None):
  """Register MaxTextForCausalLM with a vLLM platform plugin's model registry.

  Note, this function is invoked directly by the vLLM engine during startup. As
  such, it leverages vLLM logging to report its status.

  Args:
    register_model: The registry callable, taking (name, cls). Injected so the
      same adapter serves either platform. When omitted, the TPU plugin's
      registry is imported, preserving the previous behaviour for TPU callers.
  """
  using_tpu_registry = register_model is None
  if using_tpu_registry:
    # pylint: disable=import-outside-toplevel
    from tpu_inference.models.common.model_loader import register_model

  logger.info("Registering %s.", MODEL_NAME)
  register_model(MODEL_NAME, MaxTextForCausalLM)

  # The patch targets tpu_inference's KVCacheManager, so it is only meaningful
  # for that platform. It degrades gracefully elsewhere, but skipping it keeps
  # a GPU registration from logging a failure that is not one.
  if using_tpu_registry:
    # pylint: disable=import-outside-toplevel
    from .adapter import patch_kv_cache_manager

    patch_kv_cache_manager()

  logger.info("Successfully registered %s.", MODEL_NAME)
