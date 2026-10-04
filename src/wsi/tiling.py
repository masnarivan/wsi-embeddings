"""Whole-slide image tiling: tissue detection + a grid of fixed-magnification tiles.

Coordinates convention: every (x, y) stored in the tile table is in LEVEL-0 pixels,
because that is what OpenSlide.read_region expects regardless of the level read.
"""
from dataclasses import dataclass

import numpy as np
import openslide
import pandas as pd
from PIL import Image
from skimage import color, filters, morphology


@dataclass
class TissueMask:
    mask: np.ndarray        # bool, H x W at `downsample` w.r.t. level 0
    downsample: float
    thumbnail: np.ndarray   # RGB uint8, same H x W as mask


def slide_mpp(slide: openslide.OpenSlide) -> float:
    """Microns per pixel at level 0."""
    return float(slide.properties[openslide.PROPERTY_NAME_MPP_X])


def compute_tissue_mask(slide: openslide.OpenSlide, target_downsample: float = 64,
                        min_object_px: int = 200) -> TissueMask:
    """Separate tissue from background on a low-resolution level.

    Glass background is bright and grey (low saturation); H&E-stained tissue is
    pink/purple (high saturation). Otsu picks the saturation threshold automatically.
    """
    level = slide.get_best_level_for_downsample(target_downsample)
    downsample = slide.level_downsamples[level]
    thumb = np.asarray(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB"))

    saturation = color.rgb2hsv(thumb)[..., 1]
    mask = saturation > filters.threshold_otsu(saturation)
    # Clean up: fill small holes inside tissue, drop specks of dust / noise.
    mask = morphology.closing(mask, morphology.disk(2))
    mask = morphology.remove_small_holes(mask, max_size=min_object_px)
    mask = morphology.remove_small_objects(mask, max_size=min_object_px)
    return TissueMask(mask=mask, downsample=downsample, thumbnail=thumb)


def make_tile_grid(slide: openslide.OpenSlide, tissue: TissueMask, target_mpp: float = 0.5,
                   tile_px: int = 224, min_tissue: float = 0.5) -> pd.DataFrame:
    """Non-overlapping grid of tiles covering the tissue at `target_mpp` (0.5 um/px ~ 20x).

    Returns one row per kept tile with level-0 coords, the pyramid level to read from,
    the size to read at that level (resized to tile_px afterwards) and its tissue fraction.
    """
    target_downsample = target_mpp / slide_mpp(slide)               # ~2.06 for CAMELYON16
    level = slide.get_best_level_for_downsample(target_downsample)  # never upsample
    read_px = round(tile_px * target_downsample / slide.level_downsamples[level])
    step = tile_px * target_downsample                               # tile side in level-0 px

    W, H = slide.dimensions
    xs = np.arange(0, W - step, step)
    ys = np.arange(0, H - step, step)

    # Tissue fraction of each tile = mean of the mask over the tile's footprint.
    # Vectorised with an integral image so we don't loop over ~100k tiles in Python.
    m = tissue.mask.astype(np.float64)
    integral = np.pad(m.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ds = tissue.downsample
    x0 = np.clip((xs / ds).astype(int), 0, m.shape[1])
    x1 = np.clip(((xs + step) / ds).astype(int), 0, m.shape[1])
    y0 = np.clip((ys / ds).astype(int), 0, m.shape[0])
    y1 = np.clip(((ys + step) / ds).astype(int), 0, m.shape[0])
    X0, Y0 = np.meshgrid(x0, y0)
    X1, Y1 = np.meshgrid(x1, y1)
    area = np.maximum((X1 - X0) * (Y1 - Y0), 1)
    tissue_sum = integral[Y1, X1] - integral[Y0, X1] - integral[Y1, X0] + integral[Y0, X0]
    frac = tissue_sum / area

    gx, gy = np.meshgrid(xs, ys)
    keep = frac >= min_tissue
    return pd.DataFrame({
        "x": gx[keep].round().astype(int),
        "y": gy[keep].round().astype(int),
        "level": level,
        "read_px": read_px,
        "tile_px": tile_px,
        "size_l0": step,
        "tissue_frac": frac[keep],
    })


def read_tile(slide: openslide.OpenSlide, row) -> Image.Image:
    """Read one tile (a row of the grid) as a tile_px x tile_px RGB image."""
    img = slide.read_region((int(row.x), int(row.y)), int(row.level), (int(row.read_px),) * 2)
    img = img.convert("RGB")
    if row.read_px != row.tile_px:
        img = img.resize((int(row.tile_px),) * 2, Image.Resampling.BILINEAR)
    return img
