# -*- coding: utf-8 -*-


# Imports.
# ------------------------------------------------------------------------------


# Triton
import triton
import triton.language as tl


# General matrix multiplication tiling functions.
# ------------------------------------------------------------------------------


DEFAULT_NUM_XCDS: tl.constexpr = 8

DEFAULT_GROUP_SIZE: tl.constexpr = 1


# Tile remapping on XCDs.
@triton.jit
def remap_xcd(tile_in_mm, grid_row_col, NUM_XCDS: tl.constexpr = DEFAULT_NUM_XCDS):
    # Number of tiles per XCD in the new arrangement.
    tiles_per_xcd = (grid_row_col + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some XCDs will have tiles_per_xcd tiles,
    # the other will have tiles_per_xcd - 1 tiles. We calculate the number of XCDs
    # that have tiles_per_xcd tiles as tall_xcds.
    tall_xcds = grid_row_col % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local tile within the XCD.
    xcd = tile_in_mm % NUM_XCDS
    local_tile = tile_in_mm // NUM_XCDS
    # Calculate new tile based on the new grouping.
    # Note that we need to consider the following two cases:
    # 1. the current tile is on a tall xcd
    # 2. the current tile is on a short xcd
    if xcd < tall_xcds:
        tile = xcd * tiles_per_xcd + local_tile
    else:
        tile = (
            tall_xcds * tiles_per_xcd
            + (xcd - tall_xcds) * (tiles_per_xcd - 1)
            + local_tile
        )
    return tile


# Re-order tile ID for better L2 performance.
@triton.jit
def tile_grid(
    tile_in_mm,
    num_row_tiles,
    num_col_tiles,
    GROUP_SIZE: tl.constexpr = DEFAULT_GROUP_SIZE,
):
    if GROUP_SIZE == 1:
        row_tile = tile_in_mm // num_col_tiles
        col_tile = tile_in_mm % num_col_tiles
    else:
        num_tiles_in_group = GROUP_SIZE * num_col_tiles
        group_id = tile_in_mm // num_tiles_in_group
        first_row_tile = group_id * GROUP_SIZE
        group_row_size = min(num_row_tiles - first_row_tile, GROUP_SIZE)
        row_tile = first_row_tile + (tile_in_mm % group_row_size)
        col_tile = (tile_in_mm % num_tiles_in_group) // group_row_size

    tl.device_assert(row_tile >= 0, "row_tile < 0")
    tl.device_assert(row_tile < num_row_tiles, "row_tile >= num_row_tiles")

    tl.device_assert(col_tile >= 0, "col_tile < 0")
    tl.device_assert(col_tile < num_col_tiles, "col_tile >= num_col_tiles")

    return row_tile, col_tile


# Combines XCD remapping with L2 tile performance.
@triton.jit
def remap_xcd_tile_grid(
    tile_in_mm,
    num_row_tiles,
    num_col_tiles,
    GROUP_SIZE: tl.constexpr = DEFAULT_GROUP_SIZE,
    NUM_XCDS: tl.constexpr = DEFAULT_NUM_XCDS,
):
    return tile_grid(
        remap_xcd(tile_in_mm, num_row_tiles * num_col_tiles, NUM_XCDS=NUM_XCDS),
        num_row_tiles,
        num_col_tiles,
        GROUP_SIZE=GROUP_SIZE,
    )
