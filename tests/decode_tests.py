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

"""Tests for decode with various configs."""

import os
from MaxText.gcloud_stub import is_decoupled
import unittest

import pytest

from absl.testing import absltest

pytestmark = [pytest.mark.tpu_only, pytest.mark.external_serving]

from MaxText.decode import main as decode_main
from MaxText.globals import MAXTEXT_PKG_DIR, MAXTEXT_ASSETS_ROOT
from maxtext.tests.test_utils import get_test_config_path


class DecodeTests(unittest.TestCase):
  """Tests decode with various configs."""
  decoupled = is_decoupled()
  _dataset_path = (
    os.path.join(MAXTEXT_PKG_DIR, "..", "datasets", "c4_en_dataset_minimal") if decoupled else "gs://maxtext-dataset"
  )
  _base_output_directory = (
      os.path.join(MAXTEXT_PKG_DIR, "..", "datasets", "gcloud_decoupled_test_logs")
      if decoupled
      else "gs://runner-maxtext-logs"
  )

  CONFIGS = {
      "base": [  # tests decode
          None,
          get_test_config_path(),
          f"base_output_directory={_base_output_directory}",
          "run_name=runner_test",
          f"dataset_path={_dataset_path}",
          "steps=2",
          "enable_checkpointing=False",
          "ici_tensor_parallelism=4",
          "max_target_length=128",
          "per_device_batch_size=1",
          rf"tokenizer_path={os.path.join('src', MAXTEXT_ASSETS_ROOT, 'tokenizer.llama2')}",
      ],
      "int8": [  # tests decode with int8 quantization
          None,
          get_test_config_path(),
          f"base_output_directory={_base_output_directory}",
          "run_name=runner_test",
          f"dataset_path={_dataset_path}",
          "steps=2",
          "enable_checkpointing=False",
          "ici_tensor_parallelism=4",
          "max_target_length=128",
          "per_device_batch_size=1",
          "quantization=int8",
          "quantize_kvcache=True",
          rf"tokenizer_path={os.path.join('src', MAXTEXT_ASSETS_ROOT, 'tokenizer.llama2')}",
      ],
      "pdb_lt_1": [  # tests decode with per_device_batch_size < 1
          None,
          get_test_config_path(),
          f"base_output_directory={_base_output_directory}",
          "run_name=runner_test",
          f"dataset_path={_dataset_path}",
          "steps=2",
          "enable_checkpointing=False",
          "ici_tensor_parallelism=4",
          "max_target_length=128",
          "per_device_batch_size=.25",
          rf"tokenizer_path={os.path.join('src', MAXTEXT_ASSETS_ROOT, 'tokenizer.llama2')}",
      ],
  }

  @pytest.mark.tpu_only
  def test_tpu_base(self):
    decode_main(DecodeTests.CONFIGS["base"])

  @pytest.mark.gpu_only
  def test_gpu_base(self):
    decode_main(DecodeTests.CONFIGS["base"] + ["attention=dot_product"])

  @pytest.mark.tpu_only
  def test_tpu_int8(self):
    decode_main(DecodeTests.CONFIGS["int8"] + ["attention=dot_product"])

  @pytest.mark.gpu_only
  def test_gpu_int8(self):
    decode_main(DecodeTests.CONFIGS["int8"] + ["attention=dot_product"])

  @pytest.mark.tpu_only
  def test_tpu_pdb_lt_1(self):
    decode_main(DecodeTests.CONFIGS["pdb_lt_1"])

  @pytest.mark.gpu_only
  def test_gpu_pdb_lt_1(self):
    decode_main(DecodeTests.CONFIGS["pdb_lt_1"] + ["attention=dot_product"])


if __name__ == "__main__":
  absltest.main()
