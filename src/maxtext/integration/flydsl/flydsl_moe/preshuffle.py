# SPDX-License-Identifier: Apache-2.0
"""FlyDSL MoE weight-layout helpers: fuse gate/up, pad inter_dim, MFMA shuffle.

stage-1: wi_0/wi_1 [E,D,M] -> fuse, pad M to 128, shuffle -> [E*2*M_pad, D].
stage-2: wo [E,M,D] -> transpose, pad M, shuffle -> [E*D, M_pad].
Zero-padding M to a multiple of 128 is numerically exact (no-op for Mixtral 14336).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

PAD_TILE = 128


def pad_inter_dim(M: int, tile: int = PAD_TILE) -> int:
    """Round ``M`` up to the smallest multiple of ``tile``.

    No-op when ``M`` is already a multiple of ``tile`` (e.g. Mixtral 14336).
    Bumps gemma4's 704 to 768.
    """
    return ((int(M) + tile - 1) // tile) * tile


def shuffle_weight_jax(b: jax.Array, layout: tuple[int, int] = (16, 16)) -> jax.Array:
    """B[N, K] -> B'[N, K] in FlyDSL's MFMA-friendly tile layout."""
    N, K = b.shape
    BN, BK_half = layout
    BK = BK_half * 2
    K_inner = 16 // b.itemsize
    BK_K = BK // K_inner
    return jnp.transpose(
        b.reshape(N // BN, BN, K // BK, BK_K, K_inner),
        (0, 2, 3, 1, 4),
    ).reshape(N, K)


def fuse_gate_up(
    wi_0: jax.Array, wi_1: jax.Array, *, pad_tile: int = PAD_TILE
) -> jax.Array:
    """(wi_0, wi_1) [E,D,M] -> [E*2*M_pad, D]; gate then up per expert.

    Order matters: FlyDSL stage-1 computes ``act(gate) * up``, so gate must
    come before up. M is zero-padded to a multiple of ``pad_tile``.
    """
    E, D, M = wi_0.shape
    M_pad = pad_inter_dim(M, pad_tile)
    if M_pad > M:
        pad = M_pad - M
        wi_0 = jnp.pad(wi_0, ((0, 0), (0, 0), (0, pad)))
        wi_1 = jnp.pad(wi_1, ((0, 0), (0, 0), (0, pad)))
    fused = jnp.concatenate(
        [jnp.transpose(wi_0, (0, 2, 1)), jnp.transpose(wi_1, (0, 2, 1))], axis=1
    )
    return fused.reshape(E * 2 * M_pad, D)


def transpose_down(wo: jax.Array, *, pad_tile: int = PAD_TILE) -> jax.Array:
    """wo [E,M,D] -> [E*D, M_pad]. Same inter_dim padding as the stage-1 fuse."""
    E, M, D = wo.shape
    M_pad = pad_inter_dim(M, pad_tile)
    if M_pad > M:
        wo = jnp.pad(wo, ((0, 0), (0, M_pad - M), (0, 0)))
    return jnp.transpose(wo, (0, 2, 1)).reshape(E * D, M_pad)


def make_w1_shuffled(
    wi_0: jax.Array, wi_1: jax.Array, *, pad_tile: int = PAD_TILE
) -> jax.Array:
    """Full stage-1 preshuffle: fuse gate|up, pad, MFMA-shuffle -> [E*2*M_pad, D]."""
    return shuffle_weight_jax(fuse_gate_up(wi_0, wi_1, pad_tile=pad_tile))


def make_w2_shuffled(wo: jax.Array, *, pad_tile: int = PAD_TILE) -> jax.Array:
    """Full stage-2 preshuffle: transpose, pad, MFMA-shuffle -> [E*D, M_pad]."""
    return shuffle_weight_jax(transpose_down(wo, pad_tile=pad_tile))
