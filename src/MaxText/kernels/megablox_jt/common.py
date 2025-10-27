# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common utilities for GMM kernels."""

import re

import jax
import jax.numpy as jnp
import triton
from jax._src.lib import gpu_triton as triton_kernel_call_lib

#def is_tpu() -> bool:
#  return "TPU" in jax.devices()[0].device_kind


#def tpu_kind() -> str:
#  """Query identification string for the currently attached TPU."""
#  return jax.devices()[0].device_kind


#_TPU_KIND_PATTERN = re.compile(r"TPU v(\d+)")


#def tpu_generation() -> int:
#  """Generation number of the currently attached TPU."""
#  if version := _TPU_KIND_PATTERN.match(tpu_kind()):
#    return int(version[1])
#  raise NotImplementedError("only TPU devices are supported")
def is_power_of_2(x: int) -> bool:
    return (x > 0) and (x & (x - 1) == 0)

def check_tiling(
    M: int,
    K: int,
    N: int,
    tiling: tuple[int, int, int],
    group_sizes: jnp.ndarray | None = None,
) -> tuple[int, int, int]:
    assert M > 0, f"Number of lhs rows M must be positive (M = {M})."
    assert K > 0, f"Number of lhs columns / rhs rows K must be positive (K = {K})."
    assert N > 0, f"Number of rhs columns N must be positive (N = {N})."
    assert len(tiling) == 3, f"tiling must have 3 dimensions (it's = {len(tiling)})."
#    if group_sizes is not None:
#        max_group_size = jnp.max(group_sizes)
#        assert (
#            jnp.all(max_group_size)
#        ), f"The size of the largest group must be positive (it's {max_group_size})."
#        M = min(M, max_group_size)

    block_size_m, block_size_k, block_size_n = tiling

    # Pick smaller block sizes for toy shapes.
    block_size_m = min(triton.next_power_of_2(M), block_size_m)
    block_size_k = min(triton.next_power_of_2(K), block_size_k)
    block_size_n = min(triton.next_power_of_2(N), block_size_n)

    assert is_power_of_2(
        block_size_m
    ), f"M-dimension tile size must be a power of 2 (it's {block_size_m})."
    assert is_power_of_2(
        block_size_k
    ), f"K-dimension tile size must be a power of 2 (it's {block_size_k})."
    assert is_power_of_2(
        block_size_n
    ), f"N-dimension tile size must be a power of 2 (it's {block_size_n})."

    return block_size_m, block_size_k, block_size_n



def supports_bfloat16_matmul() -> bool:
  """Does the currently attached CPU support bfloat16 inputs?"""
  return True


def assert_is_supported_dtype(dtype: jnp.dtype) -> None:
  if dtype not in (jnp.bfloat16, jnp.float32):
    raise ValueError(f"Expected bfloat16 or float32 array but got {dtype}.")


def select_input_dtype(lhs: jnp.ndarray, rhs: jnp.ndarray) -> jnp.dtype:
  """A type to which both input should be adapted to before dot product."""
  # bf16xbf16 matmul is only supported since TPUv4 generation. In case of mixed
  # input precision, we need to convert bf16 argument to fp32 beforehand.
  if supports_bfloat16_matmul() and lhs.dtype == jnp.bfloat16 and rhs.dtype == jnp.bfloat16:
    return jnp.bfloat16
  else:
    return jnp.float32
def get_arch():
    arch = triton_kernel_call_lib.get_arch_details("0")
    arch = arch.split(":")[0]
    return arch

def num_sms( ) -> int:
    """
    #    available_devices = jax.devices()
    # Filter for GPU devices and extract information.
    #    gpu_devices = [d for d in available_devices if d.platform == 'gpu']
    #    num_sms = -1;
    #    if not gpu_devices:
    #        print("No GPU devices found.")
    #    else:
    #        gpu_name = gpu_devices[0].device_kind
    #        if "MI35" in gpu_name:
    #            print("AMD MI35X GPU")
    #            num_sms = 256
    #        if "MI300X" in gpu_name or "MI325X" in gpu_name:
    #            print("AMD MI300X or MI325 GPU")
    #            num_sms = 304
    #    assert num_sms, f"Number of SMs must be positive (it's {num_sms}).
    """
    arch = get_arch()
    match arch:
        case "gfx950":
            num_sms = 256
        case  "gfx942":
            num_sms = 304
        case _:
            num_sms = -1   
    return num_sms
