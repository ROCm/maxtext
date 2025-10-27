# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------

# Python standard library
import typing

# Triton
import triton
import triton.language as tl

# Common matrix multiplication tiling
from MaxText.kernels.megablox_jt.triton_mm_tiling import remap_xcd_tile_grid


# Triton persistent TGMM kernel.
# ------------------------------------------------------------------------------


@triton.jit
@typing.no_type_check
def triton_tgmm_persistent_kernel_core(
    # Tensor pointers:
    lhs_ptr,
    rhs_ptr,
    group_sizes_ptr,
    out_ptr,
    # Tensor shapes:
    M: int,
    K: int,
    N: int,
    G: int,
    # Meta-parameters:
    TRANS_LHS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GRID_DIM: tl.constexpr,
):
    tl.assume(M > 0)
    tl.assume(K > 0)
    tl.assume(N > 0)
    tl.assume(G > 0)

    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    tl.device_assert(num_k_tiles > 0, "num_k_tiles <= 0")

    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    tl.device_assert(num_n_tiles > 0, "num_n_tiles <= 0")

    num_tiles = num_k_tiles * num_n_tiles
    tl.device_assert(num_tiles > 0, "num_tiles <= 0")

    # Current tile. Each program computes multiple tiles of each group.
    tile = tl.program_id(0)
    tl.device_assert(tile >= 0, "tile < 0 (at initialization)")

    # Tile limit of last MM problem (inclusive).
    last_mm_tile = 0

    # Last input column of lhs and input row of rhs. Each group reads some
    # columns of lhs and some rows of rhs.
    last_m = 0

    # Loop through all (K, m, N) MM problems:
    #   (K, m) x (m, N) = (K, N)
    #   sum(m) = M
    for g in range(G):
        # Get m dimension of current MM problem.
        m = tl.load(group_sizes_ptr + g)
        # m can be zero if group is empty
        tl.device_assert(m >= 0, "m < 0")

        # Loop through tiles of current MM problem.
        while tile >= last_mm_tile and tile < last_mm_tile + num_tiles:
            # Figure out tile coordinates in current MM problem.
            tile_in_mm = tile - last_mm_tile
            tl.device_assert(tile_in_mm >= 0, "tile_in_mm < 0")

            tile_k, tile_n = remap_xcd_tile_grid(
                tile_in_mm, num_k_tiles, num_n_tiles, GROUP_SIZE=GROUP_SIZE
            )

            # Do regular MM:

            tl.device_assert(tile_k * BLOCK_SIZE_K >= 0, "tile_k * BLOCK_SIZE_K < 0")
            tl.device_assert(tile_n * BLOCK_SIZE_N >= 0, "tile_n * BLOCK_SIZE_N < 0")

            offs_lhs_k = (
                tile_k.to(tl.int64) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            ) % K
            offs_rhs_n = (
                tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            ) % N
            offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)

            if TRANS_LHS:
                lhs_ptrs = (
                    lhs_ptr + offs_lhs_k[:, None] + (last_m + offs_m[None, :]) * K
                )
            else:
                lhs_ptrs = (
                    lhs_ptr + offs_lhs_k[:, None] * M + (last_m + offs_m[None, :])
                )

            rhs_ptrs = rhs_ptr + (last_m + offs_m[:, None]) * N + offs_rhs_n[None, :]

            loop_m = tl.cdiv(m, BLOCK_SIZE_M)
            m_divisible_by_block_m = m % BLOCK_SIZE_M == 0
            if not m_divisible_by_block_m:
                loop_m -= 1

            acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)

            for _ in range(0, loop_m):
                lhs = tl.load(lhs_ptrs)
                rhs = tl.load(rhs_ptrs)

                acc += tl.dot(lhs, rhs, input_precision="ieee")

                if TRANS_LHS:
                    lhs_ptrs += BLOCK_SIZE_M * K
                else:
                    lhs_ptrs += BLOCK_SIZE_M

                rhs_ptrs += BLOCK_SIZE_M * N

            if not m_divisible_by_block_m:
                offs_lhs_k = (
                    tile_k.to(tl.int64) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                ) % K
                offs_rhs_n = (
                    tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                ) % N
                offs_m = loop_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                lhs = tl.load(lhs_ptrs, mask=offs_m[None, :] < m, other=0)
                rhs = tl.load(rhs_ptrs, mask=offs_m[:, None] < m, other=0)
                acc += tl.dot(lhs, rhs, input_precision="ieee")

            acc = acc.to(out_ptr.type.element_ty)

            offs_out_k = tile_k.to(tl.int64) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            offs_out_n = tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            out_ptrs = (
                out_ptr
                + g.to(tl.int64) * K * N
                + offs_out_k[:, None] * N
                + offs_out_n[None, :]
            )

            tl.store(
                out_ptrs,
                acc,
                mask=(offs_out_k[:, None] < K) & (offs_out_n[None, :] < N),
            )

            # Go to the next tile by advancing number of programs.
            tile += GRID_DIM
            tl.device_assert(tile > 0, "tile <= 0 (at update)")

        # Get ready to go to the next MM problem.

        last_mm_tile += num_tiles
        # last_mm_tile can be zero if group 0 is skipped
        tl.device_assert(last_mm_tile >= 0, "last_mm_tile < 0 (at update)")

        last_m += m
        # last_m can be zero if group 0 is skipped
        tl.device_assert(last_m >= 0, "last_m < 0 (at update)")
        tl.device_assert(last_m <= M, "last_m > M (at update)")


# Triton non-persistent TGMM kernel.
# ------------------------------------------------------------------------------


@triton.jit
@typing.no_type_check
def triton_tgmm_non_persistent_kernel_core(
    # Tensor pointers:
    lhs_ptr,
    rhs_ptr,
    group_sizes_ptr,
    out_ptr,
    # Tensor shapes:
    M: int,
    K: int,
    N: int,
    G: int,
    # Meta-parameters:
    TRANS_LHS: tl.constexpr,
    BLOCK_SIZE_G: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    tl.assume(M > 0)
    tl.assume(K > 0)
    tl.assume(N > 0)
    tl.assume(G > 0)

    # Get group ID from grid.
    g = tl.program_id(0)
    tl.device_assert(g >= 0, "g < 0")
    tl.device_assert(g < G, "g >= G")

    # Get m dimension of current MM group.
    m = tl.load(group_sizes_ptr + g)
    # m can be zero if group is empty.
    tl.device_assert(m >= 0, "m < 0")

    # Skip empty groups.
    if m == 0:
        return

    # Compute sum(group_sizes) until current group g.
    # It's the starting column of lhs and starting row of rhs.
    offs_g = tl.arange(0, BLOCK_SIZE_G)
    group_sizes = tl.load(group_sizes_ptr + offs_g, mask=offs_g < g, other=0)
    start_m = tl.sum(group_sizes)

    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    tl.device_assert(num_k_tiles > 0, "num_k_tiles <= 0")

    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    tl.device_assert(num_n_tiles > 0, "num_n_tiles <= 0")

    # Get MM tile from grid.
    tile_in_mm = tl.program_id(1)
    tl.device_assert(tile_in_mm >= 0, "tile_in_mm < 0")

    tile_k, tile_n = remap_xcd_tile_grid(
        tile_in_mm, num_k_tiles, num_n_tiles, GROUP_SIZE=GROUP_SIZE
    )

    tl.device_assert(tile_k * BLOCK_SIZE_K >= 0, "tile_k * BLOCK_SIZE_K < 0")
    tl.device_assert(tile_n * BLOCK_SIZE_N >= 0, "tile_n * BLOCK_SIZE_N < 0")

    offs_lhs_k = (tile_k.to(tl.int64) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) % K
    offs_rhs_n = (tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)

    if TRANS_LHS:
        lhs_ptrs = lhs_ptr + offs_lhs_k[:, None] + (start_m + offs_m[None, :]) * K
    else:
        lhs_ptrs = lhs_ptr + offs_lhs_k[:, None] * M + (start_m + offs_m[None, :])

    rhs_ptrs = rhs_ptr + (start_m + offs_m[:, None]) * N + offs_rhs_n[None, :]

    loop_m = tl.cdiv(m, BLOCK_SIZE_M)
    m_divisible_by_block_m = m % BLOCK_SIZE_M == 0
    if not m_divisible_by_block_m:
        loop_m -= 1

    acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, loop_m):
        lhs = tl.load(lhs_ptrs)
        rhs = tl.load(rhs_ptrs)

        acc += tl.dot(lhs, rhs, input_precision="ieee")

        if TRANS_LHS:
            lhs_ptrs += BLOCK_SIZE_M * K
        else:
            lhs_ptrs += BLOCK_SIZE_M

        rhs_ptrs += BLOCK_SIZE_M * N

    if not m_divisible_by_block_m:
        offs_lhs_k = (
            tile_k.to(tl.int64) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        ) % K
        offs_rhs_n = (
            tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        ) % N
        offs_m = loop_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        lhs = tl.load(lhs_ptrs, mask=offs_m[None, :] < m, other=0)
        rhs = tl.load(rhs_ptrs, mask=offs_m[:, None] < m, other=0)
        acc += tl.dot(lhs, rhs, input_precision="ieee")

    acc = acc.to(out_ptr.type.element_ty)

    offs_out_k = tile_k.to(tl.int64) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_out_n = tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    out_ptrs = (
        out_ptr + g.to(tl.int64) * K * N + offs_out_k[:, None] * N + offs_out_n[None, :]
    )

    tl.store(
        out_ptrs,
        acc,
        mask=(offs_out_k[:, None] < K) & (offs_out_n[None, :] < N),
    )
