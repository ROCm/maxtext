# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------

from dataclasses import dataclass
from functools import partial
import logging

# PyTorch
#import torch

# Triton
import triton
import jax
import jax.numpy as jnp
from MaxText.kernels.megablox_jt.common import is_power_of_2, check_tiling
#from common import is_power_of_2, check_tiling
# Types module
#from dtypes import SUPPORTED_DTYPES, DTYPE

# Kernel configuration classes.
# ------------------------------------------------------------------------------


@dataclass(frozen=True, eq=True)
class ConfigKey:
    M: int
    K: int
    N: int
    G: int
    input_type:  jnp.dtype = jnp.bfloat16
    output_type: jnp.dtype = jnp.bfloat16
    trans_lhs: bool = False
    trans_rhs: bool = False

    def __post_init__(self):
        assert self.M > 0, f"Number of lhs rows M must be positive (M = {self.M})."
        assert (
            self.K > 0
        ), f"Number of lhs columns / rhs rows K must be positive (K = {self.K})."
        assert self.N > 0, f"Number of rhs columns N must be positive (N = {self.N})."
        assert self.G > 0, f"Number of groups G must be positive (G = {self.G})."
        assert (
            self.input_type == jnp.bfloat16 or self.input_type == jnp.float32
        ), "Input type must be supported by the kernel."
        assert (
            self.output_type == jnp.bfloat16 or self.output_type == jnp.float32
        ), "Output type must be supported by the kernel."


@dataclass(frozen=True, eq=True)
class Config:
    block_size_m: int = 128
    block_size_k: int = 128
    block_size_n: int = 128
    group_size: int = 4
    grid_dim: int | None = 304 #ToDo get cu number from hip_python or jax num_sms()
    num_warps: int = 8
    num_stages: int = 2

    def __post_init__(self):
        assert is_power_of_2(
            self.block_size_m
        ), f"M-dimension tile size must be a power of 2 (it's {self.block_size_m})."
        assert is_power_of_2(
            self.block_size_k
        ), f"K-dimension tile size must be a power of 2 (it's {self.block_size_k})."
        assert is_power_of_2(
            self.block_size_n
        ), f"N-dimension tile size must be a power of 2 (it's {self.block_size_n})."
        assert (
            self.group_size > 0
        ), f"Group size must be positive (it's {self.group_size})."
        if self.grid_dim is not None:
            assert (
                self.grid_dim > 0
            ), f"Grid dimension must be positive when given (it's {self.grid_dim})."
        assert (
            self.num_warps > 0
        ), f"Number of warps must be positive (it's {self.num_warps})."
        assert (
            self.num_stages >= 0
        ), f"Number of software pipeliner stages must be non-negative (it's {self.num_stages})."


# Default configurations.
# ------------------------------------------------------------------------------


OUTER_BLOCK_SIZE: int = 256
INNER_BLOCK_SIZE: int = 64

DEFAULT_GMM_CONFIG: Config = Config(
    block_size_m=OUTER_BLOCK_SIZE,
    block_size_k=INNER_BLOCK_SIZE,
    block_size_n=OUTER_BLOCK_SIZE,
)

DEFAULT_PERSISTENT_TGMM_CONFIG: Config = Config(
    block_size_m=INNER_BLOCK_SIZE,
    block_size_k=OUTER_BLOCK_SIZE,
    block_size_n=OUTER_BLOCK_SIZE,
)

DEFAULT_NON_PERSISTENT_TGMM_CONFIG: Config = Config(
    block_size_m=INNER_BLOCK_SIZE,
    block_size_k=OUTER_BLOCK_SIZE,
    block_size_n=OUTER_BLOCK_SIZE,
    grid_dim=None,
)


# Database of best kernel configurations.
# ------------------------------------------------------------------------------


# GMM tuning database for gfx942.
# fmt: off
BEST_GMM_CONFIGS: dict[ConfigKey, Config] = {
    # NN
    ConfigKey(M=  49152, K= 1408, N= 2048, G=64): Config(block_size_m=128, block_size_k=64, block_size_n=128, group_size=4, grid_dim=1216, num_warps=8, num_stages=2),
    ConfigKey(M=3145728, K= 2048, N= 1408, G= 8): Config(block_size_m=256, block_size_k=32, block_size_n=256, group_size=1, grid_dim= 304, num_warps=8, num_stages=2),
#
    ConfigKey(M= 393216, K= 2048, N= 1408, G=64): Config(block_size_m=256, block_size_k=32, block_size_n=256, group_size=1, grid_dim= 304, num_warps=8, num_stages=2),
    ConfigKey(M= 393216, K= 1408, N= 2048, G=64): Config(block_size_m=256, block_size_k=32, block_size_n=256, group_size=1, grid_dim= 304, num_warps=8, num_stages=2),
#
    ConfigKey(M= 163840, K= 4096, N= 14336, G=8): Config(block_size_m=256, block_size_k=32, block_size_n=256, group_size=4, grid_dim= 912, num_warps=8, num_stages=2),
    ConfigKey(M= 163840, K= 14336, N= 4096, G=8): Config(block_size_m=256, block_size_k=32, block_size_n=256, group_size=1, grid_dim= 1216, num_warps=8, num_stages=2),
#
    ConfigKey(M= 491520, K= 2048, N= 1408, G=64): Config(block_size_m=256, block_size_k=32, block_size_n=256, group_size=1, grid_dim= 304, num_warps=8, num_stages=2),
    ConfigKey(M= 491520, K= 1408, N= 2048, G=64): Config(block_size_m=256, block_size_k=32, block_size_n=256, group_size=1, grid_dim= 304, num_warps=8, num_stages=2),
#
    ConfigKey(M=  32768, K= 6144, N=16384, G= 8): Config(block_size_m=256, block_size_k=32, block_size_n=256, group_size=4, grid_dim= 608, num_warps=8, num_stages=2),
    ConfigKey(M=  32768, K=16384, N= 6144, G= 8): Config(block_size_m=256, block_size_k=32, block_size_n=256, group_size=4, grid_dim= 304, num_warps=8, num_stages=2),
#
    # NT
    ConfigKey(M=  49152, K= 1408, N= 2048, G=64, trans_rhs=True): Config(block_size_m=128, block_size_k= 64, block_size_n=256, group_size=4, grid_dim= 304, num_warps=8, num_stages=2),
    ConfigKey(M=3145728, K= 2048, N= 1408, G= 8, trans_rhs=True): Config(block_size_m=256, block_size_k= 64, block_size_n=256, group_size=2, grid_dim= 912, num_warps=8, num_stages=2),
#
    ConfigKey(M= 393216, K= 2048, N= 1408, G=64, trans_rhs=True): Config(block_size_m=128, block_size_k=128, block_size_n=128, group_size=4, grid_dim= 304, num_warps=8, num_stages=2),
    ConfigKey(M= 393216, K= 1408, N= 2048, G=64, trans_rhs=True): Config(block_size_m=256, block_size_k=32, block_size_n=256, group_size=4, grid_dim= 304, num_warps=8, num_stages=2),
#
    ConfigKey(M= 163840, K= 4096, N= 14336, G=8, trans_rhs=True): Config(block_size_m=256, block_size_k=64, block_size_n=256, group_size=4, grid_dim= 1216, num_warps=8, num_stages=2),
    ConfigKey(M= 163840, K= 14336, N= 4096, G=8, trans_rhs=True): Config(block_size_m=256, block_size_k=64, block_size_n=256, group_size=2, grid_dim= 304, num_warps=8, num_stages=2),
#
    ConfigKey(M= 491520, K= 2048, N= 1408, G=64, trans_rhs=True): Config(block_size_m=256, block_size_k=64, block_size_n=256, group_size=2, grid_dim= 304, num_warps=8, num_stages=2),
    ConfigKey(M= 491526, K= 1408, N= 2048, G=64, trans_rhs=True): Config(block_size_m=256, block_size_k=32, block_size_n=256, group_size=1, grid_dim= 304, num_warps=8, num_stages=2),
#
    ConfigKey(M=  32768, K= 6144, N=16384, G= 8, trans_rhs=True): Config(block_size_m=256, block_size_k= 64, block_size_n=256, group_size=8, grid_dim=1216, num_warps=8, num_stages=2),
    ConfigKey(M=  32768, K=16384, N= 6144, G= 8, trans_rhs=True): Config(block_size_m=256, block_size_k= 64, block_size_n=256, group_size=4, grid_dim=1216, num_warps=8, num_stages=2),
}
# fmt: on


# Persistent TGMM tuning database for gfx942.
# fmt: off
BEST_PERSISTENT_TGMM_CONFIGS: dict[ConfigKey, Config] = {
    # NN
    ConfigKey(M=  49152, K= 1408, N= 2048, G=64): Config(block_size_m=32, block_size_k=128, block_size_n=256, group_size=4, grid_dim=304, num_warps=8, num_stages=2),
    ConfigKey(M=3145728, K= 2048, N= 1408, G= 8): Config(block_size_m=64, block_size_k=128, block_size_n=256, group_size=4, grid_dim=912, num_warps=4, num_stages=1),
    ConfigKey(M= 393216, K= 2048, N= 1408, G=64): Config(block_size_m=32, block_size_k= 64, block_size_n=256, group_size=8, grid_dim=304, num_warps=8, num_stages=2),
    ConfigKey(M=  32768, K= 6144, N=16384, G= 8): Config(block_size_m=32, block_size_k=128, block_size_n=256, group_size=4, grid_dim=304, num_warps=8, num_stages=2),
    ConfigKey(M=  32768, K=16384, N= 6144, G= 8): Config(block_size_m=32, block_size_k=128, block_size_n=256, group_size=4, grid_dim=304, num_warps=8, num_stages=2),
    # TN
    ConfigKey(M=  49152, K= 1408, N= 2048, G=64, trans_lhs=True): Config(block_size_m=32, block_size_k=128, block_size_n=256, group_size=4, grid_dim=304, num_warps=8, num_stages=2),
    ConfigKey(M=3145728, K= 2048, N= 1408, G= 8, trans_lhs=True): Config(block_size_m=64, block_size_k=128, block_size_n=128, group_size=4, grid_dim=608, num_warps=2, num_stages=1),
#    
    ConfigKey(M= 393216, K= 2048, N= 1408, G=64, trans_lhs=True): Config(block_size_m=32, block_size_k=256, block_size_n=256, group_size=2, grid_dim=912, num_warps=8, num_stages=2),
    ConfigKey(M= 393216, K= 1408, N= 2048, G=64, trans_lhs=True): Config(block_size_m=64, block_size_k=128, block_size_n=256, group_size=4, grid_dim=912, num_warps=4, num_stages=2),
#    
    ConfigKey(M= 163840, K= 4096, N= 14336, G=8, trans_lhs=True): Config(block_size_m=32, block_size_k=256, block_size_n=64, group_size=8, grid_dim= 1216, num_warps=2, num_stages=2),
    ConfigKey(M= 163840, K= 14336, N= 4096, G=8, trans_lhs=True): Config(block_size_m=32, block_size_k=256, block_size_n=256, group_size=1, grid_dim= 608, num_warps=8, num_stages=2),
#
    ConfigKey(M= 491520, K= 2048, N= 1408, G=64, trans_lhs=True): Config(block_size_m=32, block_size_k=256, block_size_n=128, group_size=1, grid_dim=608, num_warps=8, num_stages=2),
    ConfigKey(M= 491520, K= 1408, N= 2048, G=64, trans_lhs=True): Config(block_size_m=64, block_size_k=128, block_size_n=256, group_size=4, grid_dim=1216, num_warps=4, num_stages=2),    
#
    ConfigKey(M=  32768, K= 6144, N=16384, G= 8, trans_lhs=True): Config(block_size_m=32, block_size_k=256, block_size_n=256, group_size=4, grid_dim=912, num_warps=8, num_stages=2),
    ConfigKey(M=  32768, K=16384, N= 6144, G= 8, trans_lhs=True): Config(block_size_m=32, block_size_k=256, block_size_n=256, group_size=4, grid_dim=304, num_warps=8, num_stages=2),
}
# fmt: on


# Non-persistent TGMM tuning database for gfx942.
# fmt: off
BEST_NON_PERSISTENT_TGMM_CONFIGS: dict[ConfigKey, Config] = {
    # NN
    ConfigKey(M=  49152, K= 1408, N= 2048, G=64): Config(grid_dim=None, block_size_m=32, block_size_k=128, block_size_n=256, group_size=2, num_warps=8, num_stages=2),
    ConfigKey(M=3145728, K= 2048, N= 1408, G= 8): Config(grid_dim=None, block_size_m=64, block_size_k=128, block_size_n=256, group_size=1, num_warps=8, num_stages=1),
    ConfigKey(M= 393216, K= 2048, N= 1408, G=64): Config(grid_dim=None, block_size_m=64, block_size_k=128, block_size_n=256, group_size=1, num_warps=4, num_stages=2),
    ConfigKey(M=  32768, K= 6144, N=16384, G= 8): Config(grid_dim=None, block_size_m=32, block_size_k=128, block_size_n=256, group_size=2, num_warps=8, num_stages=2),
    ConfigKey(M=  32768, K=16384, N= 6144, G= 8): Config(grid_dim=None, block_size_m=32, block_size_k=256, block_size_n=256, group_size=1, num_warps=8, num_stages=2),
    # TN
    ConfigKey(M=  49152, K= 1408, N= 2048, G=64, trans_lhs=True): Config(grid_dim=None, block_size_m= 32, block_size_k=256, block_size_n=256, group_size=8, num_warps=8, num_stages=2),
    ConfigKey(M=3145728, K= 2048, N= 1408, G= 8, trans_lhs=True): Config(grid_dim=None, block_size_m=128, block_size_k=256, block_size_n=128, group_size=1, num_warps=8, num_stages=1),
    ConfigKey(M= 393216, K= 2048, N= 1408, G=64, trans_lhs=True): Config(grid_dim=None, block_size_m= 32, block_size_k=256, block_size_n=256, group_size=2, num_warps=8, num_stages=2),
    ConfigKey(M=  32768, K= 6144, N=16384, G= 8, trans_lhs=True): Config(grid_dim=None, block_size_m= 32, block_size_k=256, block_size_n=256, group_size=1, num_warps=8, num_stages=2),
    ConfigKey(M=  32768, K=16384, N= 6144, G= 8, trans_lhs=True): Config(grid_dim=None, block_size_m= 32, block_size_k=256, block_size_n=256, group_size=1, num_warps=8, num_stages=2),
}
# fmt: on


# Selection of best kernel configuration.
# ------------------------------------------------------------------------------


def _pick_best_config(
    desc: str,
    best_configs: dict[ConfigKey, Config],
    default_config: Config,
    M: int,
    K: int,
    N: int,
    G: int,
    group_sizes: jnp.ndarray | None,
    input_type:  jnp.dtype = jnp.bfloat16,
    output_type: jnp.dtype = jnp.float16,
    trans_lhs: bool = False,
    trans_rhs: bool = False,
) -> Config:
    config_key = ConfigKey(M, K, N, G, input_type, output_type, trans_lhs, trans_rhs)
    logging.debug("Querying best %s config for %s.", desc, config_key)
    try:
        best_config = best_configs[config_key]
    except KeyError:
        logging.debug(
            "Could not find best %s config for %s, picking default one + block size heuristics.",
            desc,
            config_key,
        )
        block_size_m, block_size_k, block_size_n = check_tiling(
            M,
            K,
            N,
            (
                default_config.block_size_m,
                default_config.block_size_k,
                default_config.block_size_n,
            ),
            group_sizes=group_sizes,
        )
        best_config = Config(
            block_size_m=block_size_m,
            block_size_k=block_size_k,
            block_size_n=block_size_n,
            grid_dim=default_config.grid_dim,
        )
    logging.debug("Best %s config for %s is %s.", desc, config_key, best_config)
    return best_config


pick_best_gmm_config = partial(
    _pick_best_config,
    "GMM",
    BEST_GMM_CONFIGS,
    DEFAULT_GMM_CONFIG,
)

pick_best_persistent_tgmm_config = partial(
    _pick_best_config,
    "persistent TGMM",
    BEST_PERSISTENT_TGMM_CONFIGS,
    DEFAULT_PERSISTENT_TGMM_CONFIG,
)

pick_best_non_persistent_tgmm_config = partial(
    _pick_best_config,
    "non-persistent TGMM",
    BEST_NON_PERSISTENT_TGMM_CONFIGS,
    DEFAULT_NON_PERSISTENT_TGMM_CONFIG,
)


# Get unique configurations from a given tuning database.
# (for Triton autotuning purposes)
# ------------------------------------------------------------------------------


def _unique_triton_configs(
    best_configs: dict[ConfigKey, Config],
    default_config: Config,
) -> list[triton.Config]:
    configs = [
        triton.Config(
            (
                {
                    "BLOCK_SIZE_M": config.block_size_m,
                    "BLOCK_SIZE_K": config.block_size_k,
                    "BLOCK_SIZE_N": config.block_size_n,
                    "GROUP_SIZE": config.group_size,
                }
                | ({"GRID_DIM": config.grid_dim} if config.grid_dim is not None else {})
            ),
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        for config in set(best_configs.values()) | {default_config}
    ]
    assert all("GRID_DIM" in config.kwargs for config in configs) or all(
        "GRID_DIM" not in config.kwargs for config in configs
    ), "All configs must have GRID_DIM or all configs must not have GRID_DIM."
    return configs


unique_triton_gmm_configs = partial(
    _unique_triton_configs,
    BEST_GMM_CONFIGS,
    DEFAULT_GMM_CONFIG,
)

unique_triton_persistent_tgmm_configs = partial(
    _unique_triton_configs,
    BEST_PERSISTENT_TGMM_CONFIGS,
    DEFAULT_PERSISTENT_TGMM_CONFIG,
)

unique_triton_non_persistent_tgmm_configs = partial(
    _unique_triton_configs,
    BEST_NON_PERSISTENT_TGMM_CONFIGS,
    DEFAULT_NON_PERSISTENT_TGMM_CONFIG,
)
if __name__ == "__main__":
  M , K, N, G = 393216, 2048, 1408, 64

  print(pick_best_gmm_config(
        M,
        K,
        N,
        G,
        group_sizes = None,
        input_type = jnp.bfloat16,
        output_type = jnp.bfloat16,
        trans_rhs = False,
    )
  )
  print(pick_best_gmm_config(
        M,
        K,
        N,
        G,
        group_sizes = None,
        input_type = jnp.bfloat16,
        output_type = jnp.bfloat16,
        trans_rhs = True,
    )
  )

  print(pick_best_gmm_config(
        M,
        N,
        K,
        G,
        group_sizes = None,
        input_type = jnp.bfloat16,
        output_type = jnp.bfloat16,
        trans_rhs = False,
    )
  )
  print(pick_best_gmm_config(
        M,
        N,
        K,
        G,
        group_sizes = None,
        input_type = jnp.bfloat16,
        output_type = jnp.bfloat16,
        trans_rhs = True,
    )
  )
  

#  print(pick_best_persistent_tgmm_config(
#        M,
#        K,
#        N,
#        G,
#        group_sizes = None,
#        input_type = jnp.bfloat16,
#        output_type = jnp.bfloat16,
#        trans_lhs = False,
#     )
#  )
  print(pick_best_persistent_tgmm_config(
        M,
        K,
        N,
        G,
        group_sizes = None,
        input_type = jnp.bfloat16,
        output_type = jnp.bfloat16,
        trans_lhs = True,
     )
  )
  print(pick_best_persistent_tgmm_config(
        M,
        N,
        K,
        G,
        group_sizes = None,
        input_type = jnp.bfloat16,
        output_type = jnp.bfloat16,
        trans_lhs = True,
     )
  )

#  print(pick_best_non_persistent_tgmm_config(
#        M,
#        K,
#        N,
#        G,
#        group_sizes = None,
#        input_type = jnp.bfloat16,
#        output_type = jnp.bfloat16,
#        trans_lhs = False,
#     )
#  )
#  print(pick_best_non_persistent_tgmm_config(
#        M,
#        K,
#        N,
#        G,
#        group_sizes = None,
#        input_type  = jnp.bfloat16,
#        output_type = jnp.bfloat16,
#        trans_lhs = True,
#     )
#  )
