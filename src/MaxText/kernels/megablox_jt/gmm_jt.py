import jax
import jax.numpy as jnp
import triton
import triton.language as tl
import jax_triton as jt
from collections.abc import Callable
from typing import Any, Optional, Literal
from MaxText.kernels.megablox_jt.triton_gmm_kernel import triton_gmm_kernel_core
#tgmm and ptgmm  kernels
from MaxText.kernels.megablox_jt.triton_tgmm_kernel import (
    triton_tgmm_persistent_kernel_core,
    triton_tgmm_non_persistent_kernel_core,
)
from MaxText.kernels.megablox_jt.common import (
        is_power_of_2,
        check_tiling,
)
from MaxText.kernels.megablox_jt.best_config import (
    pick_best_gmm_config,    
    pick_best_persistent_tgmm_config,
    pick_best_non_persistent_tgmm_config,
)

#GRID_DIM = 304
#NUM_WARPS = 8
#NUM_STAGS =  2
#GROUP_SIZE = 2
LutFn = Callable[[int, int, int], Optional[tuple[int, int, int]]]
'''
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
    if group_sizes is not None:
        max_group_size = jnp.max(group_sizes)
        assert (
            max_group_size > 0
        ), f"The size of the largest group must be positive (it's {max_group_size})."
        M = min(M, max_group_size)

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
'''    
def compute_ptgmm_grid(
    K: int,
    N: int,
    G: int,
    block_size_k: int,
    block_size_n: int,
    grid_dim: int,
) -> tuple[int]:
    assert K > 0, f"K must be positive, it's {K}."
    assert N > 0, f"N must be positive, it's {N}."
    assert G > 0, f"G must be positive, it's {G}."
    assert is_power_of_2(
        block_size_k
    ), f"K-dimension tile size must be a power of 2 (it's {block_size_k})."
    assert is_power_of_2(
        block_size_n
    ), f"N-dimension tile size must be a power of 2 (it's {block_size_n})."
    assert grid_dim > 0, f"Grid dimension must be positive (it's {grid_dim})."
    num_k_tiles = triton.cdiv(K, block_size_k)
    assert num_k_tiles > 0, f"num_k_tiles must be positive, it's {num_k_tiles}."
    num_n_tiles = triton.cdiv(N, block_size_n)
    assert num_n_tiles > 0, f"num_n_tiles must be positive, it's {num_n_tiles}."
    num_tiles = G * num_k_tiles * num_n_tiles
    assert num_tiles > 0, f"num_tiles must be positive, it's {num_tiles}."
    num_programs = min(grid_dim, num_tiles)
    assert num_programs > 0, f"num_programs must be positive, it's {num_programs}."
    return (num_programs,)

def compute_tgmm_grid(
    K: int,
    N: int,
    G: int,
    block_size_k: int,
    block_size_n: int,
) -> tuple[int, int]:
    assert K > 0, f"K must be positive, it's {K}."
    assert N > 0, f"N must be positive, it's {N}."
    assert G > 0, f"G must be positive, it's {G}."
    assert is_power_of_2(
        block_size_k
    ), f"K-dimension tile size must be a power of 2 (it's {block_size_k})."
    assert is_power_of_2(
        block_size_n
    ), f"N-dimension tile size must be a power of 2 (it's {block_size_n})."
    num_k_tiles = triton.cdiv(K, block_size_k)
    assert num_k_tiles > 0, f"num_k_tiles must be positive, it's {num_k_tiles}."
    num_n_tiles = triton.cdiv(N, block_size_n)
    assert num_n_tiles > 0, f"num_n_tiles must be positive, it's {num_n_tiles}."
    num_tiles_per_mm = num_k_tiles * num_n_tiles
    assert (
        num_tiles_per_mm > 0
    ), f"num_tiles_per_mm must be positive, it's {num_tiles_per_mm}."
    return (G, num_tiles_per_mm)

def compute_gmm_grid(
    N: int,
    block_size_m: int,
    block_size_n: int,
    group_sizes: jnp.ndarray,
    grid_dim: int,
) -> tuple[int]:
    assert N > 0, f"N must be positive, it's {N}."
    assert is_power_of_2(
        block_size_m
    ), f"M-dimension tile size must be a power of 2 (it's {block_size_m})."
    assert is_power_of_2(
        block_size_n
    ), f"N-dimension tile size must be a power of 2 (it's {block_size_n})."
#    print("debug: ", N, block_size_m, block_size_n, grid_dim)
#    print("debug group_sizes: ", group_sizes)
#    assert jnp.all(group_sizes >= 0).item(), "All group_sizes must be non-negative."
    assert grid_dim > 0, f"Grid dimension must be positive (it's {grid_dim})."
    num_m_tiles = (group_sizes + block_size_m - 1) // block_size_m
#    assert jnp.all(num_m_tiles >= 0).item(), "All num_m_tiles must be non-negative."
    num_n_tiles = triton.cdiv(N, block_size_n)
    assert num_n_tiles > 0, f"num_n_tiles must be positive, it's {num_n_tiles}."
    num_tiles = jnp.max(num_m_tiles * num_n_tiles)
#    print("debug num_tiles: ",num_tiles)
#    jax.debug.print("debug: num_tiles {}", num_tiles)
#    assert num_tiles > 0, f"num_tiles must be positive, it's {num_tiles}."
    num_programs = grid_dim #int(min(grid_dim, num_tiles))
    assert num_programs > 0, f"num_programs must be positive, it's {num_programs}."
    return (num_programs,)


def gmm(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128),
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
    lhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None = None,
    rhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None = None,
) -> jnp.ndarray:
  """Compute lhs[sizes[i-1]:sizes[i], :] @ rhs for each group 'i'.

  Args:
    lhs: A 2d, jnp.ndarray with shape [m, k].
    rhs: A 3d, jnp.ndarray with shape [num_groups, k, n].
    group_sizes: A 1d, jnp.ndarray with shape [num_groups] and jnp.int32 dtype.
    preferred_element_type: jnp.dtype, the element type for the output matrix.
    tiling: 3-tuple of ints. The m, k and n-dimension tile sizes.
    group_offset: The group in group sizes to start computing from. This is
      particularly useful for when rhs num_groups is sharded.
    existing_out: Existing output to write to.
    transpose_rhs: True if the rhs needs to be transposed.
    interpret: Whether or not to run the kernel in interpret mode, helpful for
      testing and debugging.

  Returns:
    A 2d, jnp.ndarray with shape [m, n].
  """
  if lhs_quantize_dtype != None or rhs_quantize_dtype != None:
      raise ValueError("quantized unsupport yet")
  m, k, n, g = (lhs.shape[0], lhs.shape[1], rhs.shape[2], group_sizes.shape[0])
  if transpose_rhs:
    n = rhs.shape[1]
#  block_size_m, block_size_k, block_size_n = check_tiling(m, k, n, tiling)
  out_shape = jax.ShapeDtypeStruct(shape=(m, n), dtype=preferred_element_type)
  m_config =  pick_best_gmm_config(M = m, K = k, N = n, G = g,
          group_sizes = None,  input_type = jnp.bfloat16, output_type = jnp.bfloat16, trans_rhs = transpose_rhs)
  grid = compute_gmm_grid(n, m_config.block_size_m, m_config.block_size_n, group_sizes, m_config.grid_dim)
  print("gmm config grid: ", transpose_rhs, m, k, n, m_config, grid)
  is_k_divisible_by_block_k = (k%m_config.block_size_k)==0
  return  jt.triton_call(
        lhs,
        rhs,
        group_sizes,
        kernel=triton_gmm_kernel_core,
        out_shape = out_shape,
        grid = grid,
        num_warps = m_config.num_warps,
        num_stages = m_config.num_stages,
        #  shapes:
        M=m,
        K=k,
        N=n,
        G=g,
        # Meta-parameters:
        TRANS_RHS =  transpose_rhs,
        BLOCK_SIZE_M=m_config.block_size_m,
        BLOCK_SIZE_K=m_config.block_size_k,
        BLOCK_SIZE_N=m_config.block_size_n,
        K_DIVISIBLE_BY_BLOCK_SIZE_K=is_k_divisible_by_block_k,
        GROUP_SIZE = m_config.group_size,
        GRID_DIM = m_config.grid_dim,
        debug=False
    )

def ptgmm(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128),
    group_offset: jnp.ndarray | None = None,
    num_actual_groups: int | None = None,
    existing_out: jnp.ndarray | None = None,
    interpret: bool = False,
) -> jnp.ndarray:
  """Compute lhs[:, sizes[i-1]:sizes[i]] @ rhs[sizes[i-1]:sizes[i], :].

  Args:
    lhs: A 2d, jnp.ndarray with shape [k, m].
    rhs: A 2d, jnp.ndarray with shape [m, n].
    group_sizes: A 1d, jnp.ndarray with shape [num_groups] and jnp.int32 dtype.
    preferred_element_type: jnp.dtype, the element type for the output matrix.
    tiling: 3-tuple of ints. The m, k and n-dimension tile sizes.
    group_offset: The group in group sizes to start computing from. This is
      particularly useful for when rhs num_groups is sharded.
    num_actual_groups: For when num_groups is sharded and we should only compute
      the groups that are local, starting from group_offset.
    existing_out: Existing output to write to.
    interpret: Whether or not to run the kernel in interpret mode, helpful for
      testing and debugging.

  Returns:
    A  3d, jnp.ndarray with shape [num_groups, k, n].
  """

  # Gather shape information.
  trans_lhs = False
  if lhs.shape[1] != rhs.shape[0] and  lhs.shape[0] == rhs.shape[0]: 
      trans_lhs = True
      k, m, n = (lhs.shape[1], lhs.shape[0], rhs.shape[1])
  elif lhs.shape[1] == rhs.shape[0]:
      k, m, n = (lhs.shape[0], lhs.shape[1], rhs.shape[1])    
  else:
      raise ValueError("dims of lha and rhs does not match for gemm")
  num_group = group_sizes.shape[0]
#  num_actual_groups = (
#      num_actual_groups if num_actual_groups is not None else num_groups
#  )
#  block_size_m, block_size_k, block_size_n = check_tiling(m, k, n, tiling)
  out_shape = jax.ShapeDtypeStruct(shape=(num_group, k, n), dtype=preferred_element_type)
  m_config =  pick_best_persistent_tgmm_config(M = m, K = k, N = n, G = num_group,
          group_sizes = group_sizes,  input_type = jnp.bfloat16, output_type = jnp.bfloat16, 
          trans_lhs = trans_lhs,)
  grid = compute_ptgmm_grid( k, n, num_group, m_config.block_size_k, m_config.block_size_n, m_config.grid_dim,)
  print("ptgmm config grid: ",trans_lhs, m, k, n, num_group, m_config, grid)
  return  jt.triton_call(
        lhs,
        rhs,
        group_sizes,
        kernel=triton_tgmm_persistent_kernel_core,
        out_shape = out_shape,
        grid = grid,
        num_warps = m_config.num_warps,
        num_stages = m_config.num_stages,
        #  shapes:
        M = m,
        K = k,
        N = n,
        G = num_group,
        # Meta-parameters:
        TRANS_LHS = trans_lhs,
        BLOCK_SIZE_M = m_config.block_size_m,
        BLOCK_SIZE_K = m_config.block_size_k,
        BLOCK_SIZE_N = m_config.block_size_n,
        GROUP_SIZE = m_config.group_size,
        GRID_DIM = m_config.grid_dim,
        debug=False
    )


def nptgmm(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128),
    group_offset: jnp.ndarray | None = None,
    num_actual_groups: int | None = None,
    existing_out: jnp.ndarray | None = None,
    interpret: bool = False,
) -> jnp.ndarray:
  """Compute lhs[:, sizes[i-1]:sizes[i]] @ rhs[sizes[i-1]:sizes[i], :].

  Args:
    lhs: A 2d, jnp.ndarray with shape [k, m].
    rhs: A 2d, jnp.ndarray with shape [m, n].
    group_sizes: A 1d, jnp.ndarray with shape [num_groups] and jnp.int32 dtype.
    preferred_element_type: jnp.dtype, the element type for the output matrix.
    tiling: 3-tuple of ints. The m, k and n-dimension tile sizes.
    group_offset: The group in group sizes to start computing from. This is
      particularly useful for when rhs num_groups is sharded.
    num_actual_groups: For when num_groups is sharded and we should only compute
      the groups that are local, starting from group_offset.
    existing_out: Existing output to write to.
    interpret: Whether or not to run the kernel in interpret mode, helpful for
      testing and debugging.

  Returns:
    A  3d, jnp.ndarray with shape [num_groups, k, n].
  """

  # Gather shape information.
  trans_lhs = False
  if lhs.shape[1] != rhs.shape[0] and  lhs.shape[0] == rhs.shape[0]:
      trans_lhs = True
      k, m, n = (lhs.shape[1], lhs.shape[0], rhs.shape[1])
  elif  lhs.shape[1] == rhs.shape[0]:
      k, m, n = (lhs.shape[0], lhs.shape[1], rhs.shape[1])
  else:
       raise ValueError("Number must be positive.")
  num_group = group_sizes.shape[0]
#  num_actual_groups = (
#      num_actual_groups if num_actual_groups is not None else num_groups
#  )
#  block_size_m, block_size_k, block_size_n = check_tiling(m, k, n, tiling)
  out_shape = jax.ShapeDtypeStruct(shape=(num_group, k, n), dtype=preferred_element_type)
  m_config =  pick_best_non_persistent_tgmm_config(M = m, K = k, N = n, G = num_group,
          group_sizes = group_sizes,  input_type = jnp.bfloat16, output_type = jnp.bfloat16)
  grid = compute_tgmm_grid( k, n, num_group, m_config.block_size_k, m_config.block_size_n)
  block_size_g = triton.next_power_of_2(num_group)
  print("nptgmm config grid: ", trans_lhs, m, k, n, num_group, m_config, grid,num_group)
  return  jt.triton_call(
        lhs,
        rhs,
        group_sizes,
        kernel=triton_tgmm_non_persistent_kernel_core,
        out_shape = out_shape,
        grid = grid,
        num_warps = m_config.num_warps,
        num_stages = m_config.num_stages,
        #  shapes:
        M = m,
        K = k,
        N = n,
        G = num_group,
        # Meta-parameters:
        TRANS_LHS = trans_lhs,
        BLOCK_SIZE_G = num_group,
        BLOCK_SIZE_M = m_config.block_size_m,
        BLOCK_SIZE_K = m_config.block_size_k,
        BLOCK_SIZE_N = m_config.block_size_n,
        GROUP_SIZE = m_config.group_size,
        debug=False
    )  
    
