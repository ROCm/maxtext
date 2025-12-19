# Copyright 2023–2025 Google LLC
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# pytype: skip-file
"""Pydantic-based configuration management for MaxText."""
import logging
import os
import sys
from typing import Any

import jax
import jax.numpy as jnp

import omegaconf

from MaxText import max_utils
from MaxText import pyconfig_deprecated
from MaxText.common_types import DecoderBlockType, ShardMode
from MaxText.configs import types
from MaxText.inference_utils import str2bool

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

_BASE_CONFIG_ATTR = "base_config"
_MAX_PREFIX = "M_"
_yaml_types_to_parser = {str: str, int: int, float: float, bool: str2bool}


def yaml_key_to_env_key(s: str) -> str:
  return _MAX_PREFIX + s.upper()


def resolve_config_path(param: str) -> str:
  """Resolve config path to auto rewrite to use new src folder."""
  return param if os.path.isfile(param) else os.path.join("src", param)


def _merge_logical_axis_rules(base_rules, new_rules):
  """Merges two lists of logical_axis_rules. Rules in new_rules override all rules
  with the same name in base_rules."""
  if not new_rules:
    return base_rules

  new_rule_keys = {rule[0] for rule in new_rules}

  # Filter old rules to exclude any that will be replaced.
  updated_rules = [rule for rule in base_rules if rule[0] not in new_rule_keys]

  # Add all the new rules.
  updated_rules.extend(new_rules)
  return updated_rules


def _load_config(config_name: str) -> omegaconf.DictConfig:
  """Loads a YAML file and its base_configs recursively using OmegaConf."""
  cfg = omegaconf.OmegaConf.load(config_name)
  if _BASE_CONFIG_ATTR in cfg:
    base_path = cfg[_BASE_CONFIG_ATTR]
    if not os.path.isabs(base_path):
      # Search relative to current config, then in the default configs folder
      loaded_parent_config_filename = os.path.join(os.path.dirname(config_name), base_path)
      if not os.path.isfile(loaded_parent_config_filename):
        dir_path = os.path.dirname(os.path.realpath(__file__))
<<<<<<< HEAD
        loaded_parent_config_filename = os.path.join(dir_path, "configs", base_path)
=======
        file_path = os.path.join(dir_path, "configs", "models", f"{model_name}.yml")
      # Use omegaconf.OmegaConf to load the model-specific configuration.
      model_vars = omegaconf.OmegaConf.load(file_path)
      model_vars = omegaconf.OmegaConf.to_container(model_vars, resolve=True)
      if raw_keys["override_model_config"]:
        model_vars = {key: value for key, value in model_vars.items() if key not in keys_from_env_and_command_line}
      updated_keys = list(model_vars.keys())
      raw_keys = validate_and_update_keys(raw_keys, model_vars, config_name)
    return updated_keys


def create_parallelisms_list(raw_keys):
  ici_parallelism = [
      raw_keys["ici_data_parallelism"],
      raw_keys["ici_pipeline_parallelism"],
      raw_keys["ici_fsdp_parallelism"],
      raw_keys["ici_fsdp_transpose_parallelism"],
      raw_keys["ici_sequence_parallelism"],
      raw_keys["ici_context_parallelism"],
      raw_keys["ici_context_autoregressive_parallelism"],
      raw_keys["ici_tensor_parallelism"],
      raw_keys["ici_tensor_transpose_parallelism"],
      raw_keys["ici_tensor_sequence_parallelism"],
      raw_keys["ici_expert_parallelism"],
      raw_keys["ici_autoregressive_parallelism"],
  ]
  dcn_parallelism = [
      raw_keys["dcn_data_parallelism"],
      raw_keys["dcn_pipeline_parallelism"],
      raw_keys["dcn_fsdp_parallelism"],
      raw_keys["dcn_fsdp_transpose_parallelism"],
      raw_keys["dcn_sequence_parallelism"],
      raw_keys["dcn_context_parallelism"],
      raw_keys["dcn_context_autoregressive_parallelism"],
      raw_keys["dcn_tensor_parallelism"],
      raw_keys["dcn_tensor_transpose_parallelism"],
      raw_keys["dcn_tensor_sequence_parallelism"],
      raw_keys["dcn_expert_parallelism"],
      raw_keys["dcn_autoregressive_parallelism"],
  ]
  raw_keys["ici_parallelism"] = ici_parallelism
  raw_keys["dcn_parallelism"] = dcn_parallelism
  return raw_keys


def set_mu_dtype(raw_keys):
  # Default mu_dtype to weight_dtype if unset
  if raw_keys["mu_dtype"]:
    assert raw_keys["opt_type"] != "adam_pax", "opt_type adam_pax doesn't support explicitly setting mu_dtype"

  if raw_keys["mu_dtype"] == "":
    return raw_keys["weight_dtype"]
  else:
    return jax.numpy.dtype(raw_keys["mu_dtype"])


def validate_and_set_hlo_dump_defaults(raw_keys):
  if not raw_keys["dump_hlo"]:
    return raw_keys
  if os.environ.get("XLA_FLAGS") and raw_keys["dump_hlo_xla_flags"]:
    raise ValueError("You must set either XLA_FLAGS or dump_hlo_xla_flags to dump HLO, but not both.")
  if not os.environ.get("XLA_FLAGS") and not raw_keys["dump_hlo_xla_flags"]:
    raw_keys["dump_hlo_xla_flags"] = f"--xla_dump_to={raw_keys['dump_hlo_local_dir']} --xla_dump_large_constants"
    if raw_keys["dump_hlo_local_module_name"]:
      raw_keys["dump_hlo_xla_flags"] = (
          f"{raw_keys['dump_hlo_xla_flags']} --xla_dump_hlo_module_re={raw_keys['dump_hlo_local_module_name']}"
      )
  if not raw_keys["dump_hlo_gcs_dir"]:
    raw_keys["dump_hlo_gcs_dir"] = os.path.join(raw_keys["base_output_directory"], raw_keys["run_name"], "xla_dump")
  else:
    raw_keys["dump_hlo_gcs_dir"] = gcs_utils.add_trailing_slash(raw_keys["dump_hlo_gcs_dir"])
  if not os.environ.get("XLA_FLAGS"):
    os.environ["XLA_FLAGS"] = raw_keys["dump_hlo_xla_flags"]
  return raw_keys


def validate_multiple_slices(raw_keys):
  if (
      math.fabs(
          math.prod(
              [
                  raw_keys["dcn_data_parallelism"],
                  raw_keys["dcn_pipeline_parallelism"],
                  raw_keys["dcn_fsdp_parallelism"],
                  raw_keys["dcn_fsdp_transpose_parallelism"],
                  raw_keys["dcn_sequence_parallelism"],
                  raw_keys["dcn_context_parallelism"],
                  raw_keys["dcn_tensor_parallelism"],
                  raw_keys["dcn_tensor_sequence_parallelism"],
                  raw_keys["dcn_expert_parallelism"],
                  raw_keys["dcn_context_autoregressive_parallelism"],
                  raw_keys["dcn_autoregressive_parallelism"],
              ]
          )
      )
      > 1
  ):
    assert raw_keys["num_slices"] > 1, "DCN parallelism requested but only one slice available."


def set_and_validate_pipeline_config(raw_keys):
  if using_pipeline_parallelism(raw_keys):
    # For pipeline parallelism, model_fsdp_ag_once should be False, and pipeline_fsdp_ag_once is typically True.
    if raw_keys["model_fsdp_ag_once"]:
      raise ValueError(
          "You should only set pipeline_fsdp_once=True, leave model_fsdp_ag_once=False with pipeline parallelism."
      )

    def modify_activation_embed_and_logits_batch(logical_axis_rules):
      for idx, logical_rule in enumerate(logical_axis_rules):
        if logical_rule[0] == "activation_embed_and_logits_batch":
          # For pipeline parallelism the pre and post decoder layer tensors' batch dimension is sharded by stages.
          # Microbatches are sharded by stage, so moving out of and into this sharding should be a local reshape.
          # The "stage" needs to be listed first since the microbatch dimension is first before the reshape.
          logical_axis_rules[idx] = [
              "activation_embed_and_logits_batch",
              ["stage", "data", "fsdp", "fsdp_transpose", "expert"] if not raw_keys["use_jaxpp"] else
              ["stage", "fsdp", "fsdp_transpose", "expert"],
          ]
          break  # Exit the loop after modifying the list
      return logical_axis_rules

    def pipeline_first_axis(raw_keys):
      # We have seen better performance when axes used for DCN are earlier in this list than ICI, see (b/339009148) for details
      ici_parallelism = [
          raw_keys["ici_pipeline_parallelism"],
          raw_keys["ici_data_parallelism"],
          raw_keys["ici_fsdp_parallelism"],
          raw_keys["ici_fsdp_transpose_parallelism"],
          raw_keys["ici_sequence_parallelism"],
          raw_keys["ici_context_parallelism"],
          raw_keys["ici_context_autoregressive_parallelism"],
          raw_keys["ici_tensor_parallelism"],
          raw_keys["ici_tensor_transpose_parallelism"],
          raw_keys["ici_tensor_sequence_parallelism"],
          raw_keys["ici_expert_parallelism"],
          raw_keys["ici_autoregressive_parallelism"],
      ]
      dcn_parallelism = [
          raw_keys["dcn_pipeline_parallelism"],
          raw_keys["dcn_data_parallelism"],
          raw_keys["dcn_fsdp_parallelism"],
          raw_keys["dcn_fsdp_transpose_parallelism"],
          raw_keys["dcn_sequence_parallelism"],
          raw_keys["dcn_context_parallelism"],
          raw_keys["dcn_context_autoregressive_parallelism"],
          raw_keys["dcn_tensor_parallelism"],
          raw_keys["dcn_tensor_transpose_parallelism"],
          raw_keys["dcn_tensor_sequence_parallelism"],
          raw_keys["dcn_expert_parallelism"],
          raw_keys["dcn_autoregressive_parallelism"],
      ]
      mesh_axes = [
          "stage",
          "data",
          "fsdp",
          "fsdp_transpose",
          "sequence",
          "context",
          "context_autoregressive",
          "tensor",
          "tensor_transpose",
          "tensor_sequence",
          "expert",
          "autoregressive",
      ]
      data_sharding = [
          [
              "stage",
              "data",
              "fsdp",
              "fsdp_transpose",
              "sequence",
              "context",
              "context_autoregressive",
              "tensor",
              "tensor_transpose",
              "tensor_sequence",
              "expert",
              "autoregressive",
          ]
      ]

      raw_keys["ici_parallelism"] = ici_parallelism
      raw_keys["dcn_parallelism"] = dcn_parallelism
      raw_keys["mesh_axes"] = mesh_axes
      raw_keys["data_sharding"] = data_sharding
      return raw_keys

    raw_keys["using_pipeline_parallelism"] = True
    raw_keys["logical_axis_rules"] = modify_activation_embed_and_logits_batch(raw_keys["logical_axis_rules"])
    raw_keys = pipeline_first_axis(raw_keys)
    num_stages = int(raw_keys["ici_pipeline_parallelism"] * raw_keys["dcn_pipeline_parallelism"])
    if raw_keys["use_jaxpp"]:
      assert raw_keys["pipeline_delay_activation_forwarding"] is False
      assert raw_keys["num_pipeline_repeats"] >= 1
      assert raw_keys["num_pipeline_microbatches"] >= 1
      return raw_keys

    if raw_keys["pipeline_parallel_layers"] == -1:
      if raw_keys["decoder_block"] == "deepseek":
        moe_layers = raw_keys["num_decoder_layers"] - raw_keys["first_num_dense_layers"]
        raw_keys["pipeline_parallel_layers"] = moe_layers
      else:
        raw_keys["pipeline_parallel_layers"] = raw_keys["num_decoder_layers"]
>>>>>>> jaxpp/main
    else:
      loaded_parent_config_filename = base_path

<<<<<<< HEAD
    base_cfg = _load_config(loaded_parent_config_filename)
    cfg = omegaconf.OmegaConf.merge(base_cfg, cfg)
  return cfg
=======
    if raw_keys["num_pipeline_repeats"] == -1:
      num_pipeline_repeats, remainder = divmod(
          raw_keys["pipeline_parallel_layers"], num_stages * raw_keys["num_layers_per_pipeline_stage"]
      )
      assert (
          not remainder
      ), f"The number of layers per stage ({raw_keys['num_layers_per_pipeline_stage']}) times the number of stages ({num_stages}) must divide the number of pipeline_parallel_layers which defaults to decoder layers  ({raw_keys['pipeline_parallel_layers']}) "
      raw_keys["num_pipeline_repeats"] = num_pipeline_repeats
    assert (
        num_stages * raw_keys["num_pipeline_repeats"] * raw_keys["num_layers_per_pipeline_stage"]
        == raw_keys["pipeline_parallel_layers"]
    ), f"The product of pipeline stages ({num_stages}), repeats ({raw_keys['num_pipeline_repeats']}), and layers per stage ({raw_keys['num_layers_per_pipeline_stage']}) must be equal to pipeline_parallel_layers which defaults to decoder layers ({raw_keys['pipeline_parallel_layers']})"
    if raw_keys["num_pipeline_microbatches"] == -1:
      if raw_keys["pipeline_delay_activation_forwarding"]:
        raw_keys["num_pipeline_microbatches"] = 2 * num_stages
      else:
        raw_keys["num_pipeline_microbatches"] = num_stages
    assert (
        raw_keys["num_pipeline_microbatches"] % num_stages == 0 or raw_keys["use_jaxpp"]
    ), f"The number of microbatches ({raw_keys['num_pipeline_microbatches']}) must be divisible by the number of stages ({num_stages})"
    assert (
        raw_keys["micro_batch_size_to_train_on"] % raw_keys["num_pipeline_microbatches"] == 0
    ), f"The batch size ({raw_keys['micro_batch_size_to_train_on']}) must be divisible by the number of microbatches ({raw_keys['num_pipeline_microbatches']})"
    if raw_keys["pipeline_delay_activation_forwarding"]:
      assert (
          raw_keys["num_pipeline_microbatches"] >= 2 * num_stages
      ), f"Delayed activation forwarding requires at least 2 * num_stages microbatches, but {num_stages} stages are used with {raw_keys['num_pipeline_microbatches']} microbatches"
  else:
    raw_keys["using_pipeline_parallelism"] = False
  return raw_keys
>>>>>>> jaxpp/main


def _tuples_to_lists(l: list | tuple | Any) -> list | Any:
  """Recursively converts nested tuples to lists for Pydantic compatibility."""
  return [_tuples_to_lists(x) for x in l] if isinstance(l, (list, tuple)) else l


def _lists_to_tuples(l: list | Any) -> tuple | Any:
  """Recursively converts nested lists to tuples for JAX compatibility."""
  return tuple(_lists_to_tuples(x) for x in l) if isinstance(l, list) else l


def _prepare_for_pydantic(raw_keys: dict[str, Any]) -> dict[str, Any]:
  """Prepares the raw dictionary for Pydantic model instantiation."""
  pydantic_kwargs = {}
  valid_fields = types.MaxTextConfig.model_fields.keys()

  # This is a workaround for tests that use `dataset_type='hf'` but do not
  # specify `tokenizer_type='huggingface'`, which they should.
  if raw_keys.get("dataset_type") == "hf" and "tokenizer_type" not in raw_keys:
    raw_keys["tokenizer_type"] = "huggingface"

<<<<<<< HEAD
  for key, value in raw_keys.items():
    if key not in valid_fields:
      logger.warning("Ignoring invalid/unsupported field from YAML/CLI: %s", repr(key))
=======
def validate_sparse_matmul_parallelism(raw_keys):
  # TODO: remove once b/434699033 resolved
  if raw_keys["sparse_matmul"] and (using_expert_parallelism(raw_keys) and (not raw_keys["use_jaxpp"] and using_pipeline_parallelism(raw_keys))):
    raise ValueError("Sparse matmul doesn't support using expert and pipeline parallelism together.")

  # TODO: remove once b/435539039 resolved
  if raw_keys["sparse_matmul"] and (
      using_fsdp_and_transpose_parallelism(raw_keys)
      and using_expert_parallelism(raw_keys)
      and using_tensor_parallelism(raw_keys)
  ):
    raise ValueError("Sparse matmul doesn't support using fsdp, expert, and tensor parallelism together.")
  tensor_parallelism = (
      raw_keys["ici_tensor_parallelism"]
      * raw_keys["dcn_tensor_parallelism"]
      * raw_keys["ici_tensor_sequence_parallelism"]
      * raw_keys["dcn_tensor_sequence_parallelism"]
      * raw_keys["ici_tensor_transpose_parallelism"]
      * raw_keys["dcn_tensor_transpose_parallelism"]
  )
  if raw_keys["sparse_matmul"] and using_tensor_parallelism(raw_keys) and (raw_keys["emb_dim"] % tensor_parallelism):
    raise ValueError(
        f"The embedding dimension {raw_keys['emb_dim']} is not divisible by tensor parallelism setting {tensor_parallelism}."
    )
  expert_parallelism = raw_keys["ici_expert_parallelism"] * raw_keys["dcn_expert_parallelism"]
  if raw_keys["sparse_matmul"] and using_expert_parallelism(raw_keys) and (raw_keys["num_experts"] % expert_parallelism):
    raise ValueError(
        f"The expert dimension {raw_keys['num_experts']} is not divisible by expert parallelism setting {expert_parallelism}."
    )


def validate_ring_of_experts_parallelism(raw_keys):
  if raw_keys["use_ring_of_experts"] and not using_expert_parallelism(raw_keys):
    raise ValueError("Ring-of-experts requires expert-parallelism to be enabled.")


def validate_shard_fsdp_on_expert_parallelism(raw_keys):
  if raw_keys["fsdp_shard_on_exp"] and raw_keys["num_experts"] % raw_keys["ici_fsdp_parallelism"] != 0:
    raise ValueError("fsdp_shard_on_exp requires num_experts is divisiable by ici_fsdp_parallelism.")
  if raw_keys["fsdp_shard_on_exp"] and (using_tensor_parallelism(raw_keys) or using_expert_parallelism(raw_keys)):
    raise ValueError(
        "fsdp_shard_on_exp requires ici_expert_parallelism = 1 and ici_tensor_parallelism/ici_tensor_transpose_parallelism = 1."
    )


def validate_ragged_dot(raw_keys):
  if raw_keys["sparse_matmul"] and not raw_keys["megablox"]:
    config_flag = "jax_ragged_dot_use_ragged_dot_instruction"
    try:
      jax.config.update(config_flag, True)
    except AttributeError:
      max_logging.log(f"JAX config {config_flag} not found, possibly due to old JAX version.")


def create_new_logical_axis_rules(old_logical_axis_rules, new_logical_axis_rules):
  new_logical_axis = set()
  replacements = []
  for logical_axis, mesh_axes in new_logical_axis_rules:
    logical_axis_exists = any(rule for rule in old_logical_axis_rules if rule[0] == logical_axis)
    if not logical_axis_exists:
>>>>>>> jaxpp/main
      continue

    new_value = value
    if isinstance(new_value, str) and new_value.lower() == "none":
      new_value = None

    # Pydantic validates enums from their values, so string is fine.
    # It also handles type coercion for simple types.
    if key in ("logical_axis_rules", "data_sharding"):
      if isinstance(new_value, tuple):
        new_value = _tuples_to_lists(new_value)
      if key == "data_sharding" and isinstance(new_value, list) and new_value and isinstance(new_value[0], str):
        new_value = [new_value]

    if key in ("run_name", "hf_train_files", "hf_eval_files") and new_value is None:
      new_value = ""

    pydantic_kwargs[key] = new_value

  return pydantic_kwargs


<<<<<<< HEAD
=======
def update_model_keys(raw_keys, model_keys, key):
  """Update `key` value in `raw_keys` from the value in `model_keys`."""
  assert key in model_keys and key in raw_keys
  if key == "logical_axis_rules":
    raw_keys[key] = create_new_logical_axis_rules(
        old_logical_axis_rules=raw_keys[key], new_logical_axis_rules=model_keys[key]
    )
    return
  raw_keys[key] = model_keys[key]


def validate_and_update_keys(raw_keys, model_keys, config_name: str):
  """Validate and update model specific config keys"""
  max_logging.log("Updating following parameters in config\n")

  for k in model_keys:
    max_logging.log(f"{k}: {model_keys[k]}")
    if k not in raw_keys:
      raise ValueError(f"Key {k} does not exist in config {config_name}.")
    elif not isinstance(raw_keys[k], type(model_keys[k])):
      raise ValueError(f"Type of key:{k} does not match with {type(model_keys[k])}")
    else:
      update_model_keys(raw_keys, model_keys, k)
  return raw_keys


def get_individual_scales(scale):
  """Choose appropriate scales for individual dimensions based on global scale
  We choose to rotate between doubling:
    num_head and mlp_dim
    embed_dim
    num_layers
  Any one of these steps is not a perfect doubling, although going through a cycle
  of three is a near perfect 8x scaling except for the linear -> softmax -> output step"""

  log_2_scale = math.floor((math.log2(scale)))
  if 2**log_2_scale != scale:
    raise ValueError(
        "Global parameter scale should be a power of 2. If you want finer grained control of the model sizes "
        "then you can explicitly set base_embed_dim, base_num_heads, base_mlp_dim, base_num_decoder_layers and/or head_dim."
    )
  base_scale, rem = divmod(log_2_scale, 3)
  num_head_scale = base_scale + int(rem > 0)
  mlp_dim_scale = num_head_scale
  emb_scale = base_scale + int(rem > 1)
  layer_scale = base_scale
  return emb_scale, num_head_scale, mlp_dim_scale, layer_scale


def calculate_global_batch_sizes(
    per_device_batch_size, expansion_factor_real_data, num_devices, gradient_accumulation_steps
):
  """Calculates target global batch size from target devices and per_device_batch"""
  if per_device_batch_size < 1.0:
    # For per_device_batch_size<1, we load the data as if per_device_batch_size=1
    if expansion_factor_real_data != -1:
      micro_batch_size_to_load = num_devices * expansion_factor_real_data
    else:
      micro_batch_size_to_load = num_devices
  else:
    if expansion_factor_real_data != -1:
      micro_batch_size_to_load = int(num_devices * per_device_batch_size * expansion_factor_real_data)
    else:
      micro_batch_size_to_load = int(num_devices * per_device_batch_size)

  micro_batch_size_to_train_on = int(num_devices * per_device_batch_size)
  global_batch_size_to_load = int(micro_batch_size_to_load * gradient_accumulation_steps)
  global_batch_size_to_train_on = int(micro_batch_size_to_train_on * gradient_accumulation_steps)
  return global_batch_size_to_load, global_batch_size_to_train_on, micro_batch_size_to_train_on


def get_num_target_devices(raw_keys):
  # In AOT case compile_topology is set (e.g. is not the empty string), and we determine the
  # number of devices from the compile_topology. In non-AOT settings we simply can use jax.devices().
  if raw_keys.get("compile_topology"):
    compile_topology = accelerator_to_spec_map.get_system_characteristics(raw_keys["compile_topology"])
    devices_per_slice = compile_topology.devices_per_slice
    return int(devices_per_slice * raw_keys["compile_topology_num_slices"])
  elif raw_keys.get("subslice_shape") and raw_keys.get("enable_single_controller"):
    subslice_shape = tuple(int(x) for x in raw_keys["subslice_shape"].split(","))
    return prod(subslice_shape)
  else:
    return len(jax.devices())


def get_quantization_local_shard_count(raw_keys):
  if raw_keys["quantization_local_shard_count"] == -1:
    return raw_keys["num_slices"]
  else:
    return raw_keys["quantization_local_shard_count"]


def get_context_parallel_size(raw_keys):
  cp_size = raw_keys["ici_context_parallelism"] * raw_keys["dcn_context_parallelism"]
  # ep acts as cp in attention
  if raw_keys["expert_shard_attention_option"] == "context":
    cp_size = cp_size * raw_keys["ici_expert_parallelism"] * raw_keys["dcn_expert_parallelism"]
  return cp_size


def using_pipeline_parallelism(raw_keys) -> bool:
  return raw_keys["use_jaxpp"] or int(raw_keys["ici_pipeline_parallelism"]) > 1 or int(raw_keys["dcn_pipeline_parallelism"]) > 1

def using_tensor_parallelism(raw_keys) -> bool:
  return (
      int(raw_keys["ici_tensor_parallelism"]) > 1
      or int(raw_keys["dcn_tensor_parallelism"]) > 1
      or int(raw_keys["ici_tensor_sequence_parallelism"]) > 1
      or int(raw_keys["dcn_tensor_sequence_parallelism"]) > 1
  )


def using_sequence_parallelism(raw_keys) -> bool:
  return int(raw_keys["ici_sequence_parallelism"]) > 1 or int(raw_keys["dcn_sequence_parallelism"]) > 1


def using_expert_parallelism(raw_keys) -> bool:
  if int(raw_keys["ici_expert_parallelism"]) > 1 and int(raw_keys["dcn_expert_parallelism"]) > 1:
    raise ValueError("Expert parallelism can only be enabled on ICI or DCN, not both.")
  return int(raw_keys["ici_expert_parallelism"]) > 1 or int(raw_keys["dcn_expert_parallelism"]) > 1


def using_fsdp_and_transpose_parallelism(raw_keys) -> bool:
  return (
      int(raw_keys["ici_fsdp_parallelism"]) > 1
      or int(raw_keys["dcn_fsdp_parallelism"]) > 1
      or int(raw_keys["ici_fsdp_transpose_parallelism"]) > 1
      or int(raw_keys["dcn_fsdp_transpose_parallelism"]) > 1
  )


@register_pytree_node_class
>>>>>>> jaxpp/main
class HyperParameters:
  """
  Wrapper class to expose the configuration in a read-only manner,
  maintaining backward compatibility with attribute-style access and JAX object types.
  """

  def __init__(self, pydantic_config: types.MaxTextConfig):
    object.__setattr__(self, "_pydantic_config", pydantic_config)

    final_dict = pydantic_config.model_dump()
    final_dict["dtype"] = jnp.dtype(final_dict["dtype"])
    final_dict["grad_dtype"] = jnp.dtype(final_dict["grad_dtype"])
    final_dict["weight_dtype"] = jnp.dtype(final_dict["weight_dtype"])
    final_dict["mu_dtype"] = (
        final_dict["weight_dtype"] if not final_dict["mu_dtype"] else jnp.dtype(final_dict["mu_dtype"])
    )

    final_dict["logical_axis_rules"] = _lists_to_tuples(final_dict["logical_axis_rules"])
    final_dict["data_sharding"] = _lists_to_tuples(final_dict["data_sharding"])

    final_dict["decoder_block"] = DecoderBlockType(final_dict["decoder_block"])
    final_dict["shard_mode"] = ShardMode(final_dict["shard_mode"])

    object.__setattr__(self, "_flat_config", final_dict)

  def __getattr__(self, attr: str) -> Any:
    """Provides attribute-style access to the final configuration dictionary."""
    if attr in self._flat_config:
      return self._flat_config[attr]
    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

  def __setattr__(self, attr: str, value: Any) -> None:
    """Makes the configuration object read-only."""
    raise ValueError("Configuration is read-only and cannot be modified after initialization.")

  def get_keys(self) -> dict[str, Any]:
    """Returns the configuration as a flat dictionary for backward compatibility."""
    return self._flat_config


def initialize(argv: list[str], **kwargs) -> HyperParameters:
  """Initializes the configuration by loading YAML files, and applying CLI, env, and kwarg overrides."""
  # 1. Load base and inherited configs from file(s)
  config_path = resolve_config_path(argv[1])
  base_yml_config = _load_config(config_path)

  # 2. Get overrides from CLI and kwargs
  cli_cfg = omegaconf.OmegaConf.from_cli(argv[2:])
  kwargs_cfg = omegaconf.OmegaConf.create(kwargs)
  overrides_cfg = omegaconf.OmegaConf.merge(cli_cfg, kwargs_cfg)

  # 3. Handle model-specific config
  temp_cfg = omegaconf.OmegaConf.merge(base_yml_config, overrides_cfg)
  model_name = temp_cfg.get("model_name", "default")
  model_cfg = {}
  if model_name != "default":
    # First try relative to base config path
    model_config_path = os.path.join(os.path.dirname(config_path), "models", f"{model_name}.yml")
    if not os.path.isfile(model_config_path):
      # Fallback to default location within package
      dir_path = os.path.dirname(os.path.realpath(__file__))
      model_config_path = os.path.join(dir_path, "configs", "models", f"{model_name}.yml")

    if os.path.exists(model_config_path):
      model_loaded_cfg = omegaconf.OmegaConf.load(model_config_path)
      # if override_model_config=True, only apply model configs for keys not present in overrides.
      if temp_cfg.get("override_model_config"):
        model_cfg = {k: v for k, v in model_loaded_cfg.items() if k not in overrides_cfg}
      else:
        model_cfg = model_loaded_cfg
    else:
      logger.warning("Model config for '%s' not found at %s", model_name, model_config_path)

      # 4. Final merge (base, model, then overrides)
  model_cfg_oc = omegaconf.OmegaConf.create(model_cfg)

  # 4. Manually merge logical_axis_rules to avoid OmegaConf's list replacement behavior.
  base_rules_oc = base_yml_config.get("logical_axis_rules", [])
  model_rules_oc = model_cfg_oc.get("logical_axis_rules", [])
  overrides_rules_oc = overrides_cfg.get("logical_axis_rules", [])

  base_rules = omegaconf.OmegaConf.to_container(base_rules_oc, resolve=True) if base_rules_oc else []
  model_rules = omegaconf.OmegaConf.to_container(model_rules_oc, resolve=True) if model_rules_oc else []
  overrides_rules = omegaconf.OmegaConf.to_container(overrides_rules_oc, resolve=True) if overrides_rules_oc else []

  merged_rules = _merge_logical_axis_rules(base_rules, model_rules)
  merged_rules = _merge_logical_axis_rules(merged_rules, overrides_rules)

  # Remove the rules from the original configs before the main merge
  if "logical_axis_rules" in base_yml_config:
    del base_yml_config["logical_axis_rules"]
  if "logical_axis_rules" in model_cfg_oc:
    del model_cfg_oc["logical_axis_rules"]
  if "logical_axis_rules" in overrides_cfg:
    del overrides_cfg["logical_axis_rules"]

  # 5. Final merge for all other keys
  final_config = omegaconf.OmegaConf.merge(base_yml_config, model_cfg_oc, overrides_cfg)
  final_config["logical_axis_rules"] = merged_rules

  raw_keys_dict = omegaconf.OmegaConf.to_container(final_config, resolve=True)

  # 6. Handle environment variable overrides
  cli_keys = frozenset(omegaconf.OmegaConf.to_container(cli_cfg, resolve=True).keys())
  kwargs_keys = frozenset(kwargs.keys())
  for k in tuple(raw_keys_dict.keys()):
    env_key = yaml_key_to_env_key(k)
    if env_key in os.environ:
      if k in cli_keys or k in kwargs_keys:
        raise ValueError(
            f"Key '{k}' is overridden by both CLI/kwargs and environment variable '{env_key}'. This is not allowed."
        )

      new_proposal = os.environ.get(env_key)
      original_value = raw_keys_dict.get(k)
      parser = None
      if isinstance(original_value, bool):
        parser = _yaml_types_to_parser[bool]
      elif isinstance(original_value, (str, int, float)):
        parser = type(original_value)

      if parser is None:
        raise TypeError(f"Type {type(original_value)} for key '{k}' not supported for ENV override.")

      try:
        raw_keys_dict[k] = parser(new_proposal)
      except (ValueError, KeyError) as e:
        raise ValueError(f"Couldn't parse value from ENV '{new_proposal}' for key '{k}'") from e

  pydantic_kwargs = _prepare_for_pydantic(raw_keys_dict)

  # Initialize JAX distributed system before device backend is initialized.
  if pydantic_kwargs.get("jax_debug_log_modules"):
    jax.config.update("jax_debug_log_modules", pydantic_kwargs["jax_debug_log_modules"])
  # Do not initialize jax distributed system during pytest runs.
  if "pytest" not in sys.modules:
    max_utils.maybe_initialize_jax_distributed_system(pydantic_kwargs)
  if pydantic_kwargs.get("jax_cache_dir"):
    from jax.experimental.compilation_cache import compilation_cache  # pylint: disable=import-outside-toplevel

    compilation_cache.set_cache_dir(os.path.expanduser(pydantic_kwargs["jax_cache_dir"]))

  pydantic_config = types.MaxTextConfig(**pydantic_kwargs)
  config = HyperParameters(pydantic_config)

  if config.log_config:
    for k, v in sorted(config.get_keys().items()):
      if k != "hf_access_token":
        logger.info("Config param %s: %s", k, v)

  return config


# Shim for backward compatibility with pyconfig_deprecated_test.py
validate_and_update_keys = pyconfig_deprecated.validate_and_update_keys
__all__ = ["initialize"]
