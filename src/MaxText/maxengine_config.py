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

"""Configure MaxText For JetStream"""
import functools
from typing import Any, Type

import jax

import os
DECOUPLE_GCLOUD = os.environ.get("DECOUPLE_GCLOUD", "").upper() == "TRUE"
if not DECOUPLE_GCLOUD:
  from jetstream.core import config_lib  # type: ignore
  from jetstream.engine import engine_api  # type: ignore
else:
  class _Stub:
    def __getattr__(self, name):
      raise RuntimeError("JetStream config unavailable (DECOUPLE_GCLOUD TRUE)")
  config_lib = _Stub()  # type: ignore
  engine_api = _Stub()  # type: ignore

from MaxText import maxengine


# TODO: merge it with the above create_maxengine().
def create_exp_maxengine(devices: Any, config: Any):
  if DECOUPLE_GCLOUD:
    raise RuntimeError("JetStream disabled by DECOUPLE_GCLOUD=TRUE; experimental MaxEngine unsupported.")
  return maxengine.MaxEngine(config=config, devices=devices)


def create_maxengine(devices: Any, config: Any):
  del devices
  return maxengine.MaxEngine(config)


def get_server_config(config_str: str, config: Any):
  """Gets the Server Config Required by JetStream (disabled when DECOUPLE_GCLOUD=TRUE)."""
  if DECOUPLE_GCLOUD:
    raise RuntimeError("JetStream disabled by DECOUPLE_GCLOUD=TRUE; server config unsupported.")
  match config_str:
    case "MaxtextInterleavedServer":
      server_config = config_lib.ServerConfig(
          prefill_slices=(),
          generate_slices=(),
          interleaved_slices=("tpu=" + str(jax.device_count()),),
          prefill_engine_create_fns=(),
          generate_engine_create_fns=(),
          interleaved_engine_create_fns=(functools.partial(create_maxengine, config=config),),
      )
    case "ExperimentalMaxtextDisaggregatedServer":
      # ExperimentalMaxtextDisaggregatedServer is still under development.
      # Its dependencies IFRT Proxy and other components are not publicly available
      # either.
      server_config = config_lib.ServerConfig(
          prefill_slices=(config.prefill_slice,),
          generate_slices=(config.generate_slice,),
          interleaved_slices=(),
          prefill_engine_create_fns=(functools.partial(create_exp_maxengine, config=config),),
          generate_engine_create_fns=(functools.partial(create_exp_maxengine, config=config),),
          interleaved_engine_create_fns=(),
      )
    case _:
      raise NotImplementedError
  return server_config
