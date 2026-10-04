"""Step 2: tissue mask + 20x tile grid for one slide, with QC figures.

Usage: uv run scripts/tile_slide.py tumor_031
"""
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openslide
from matplotlib.patches import Rectangle

from wsi.tiling import compute_tissue_mask, make_tile_grid, read_tile, slide_mpp

ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("slide_id")
args = parser.parse_args()

out_dir = ROOT / "outputs" / args.slide_id
fig_dir = ROOT / "outputs" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

slide = openslide.OpenSlide(ROOT / "data/raw" / f"{args.slide_id}.tif")
t0 = time.time()
tissue = compute_tissue_mask(slide)
tiles = make_tile_grid(slide, tissue)
tiles.to_parquet(out_dir / "tiles.parquet")

row = tiles.iloc[0]
print(f"mpp {slide_mpp(slide):.3f} -> read level {row.level:.0f} at {row.read_px:.0f}px, resize to {row.tile_px:.0f}px")
print(f"tile side = {row.size_l0:.0f} level-0 px = {row.size_l0 * slide_mpp(slide):.0f} um")
print(f"tissue covers {tissue.mask.mean():.1%} of the slide")
print(f"{len(tiles)} tiles kept (>=50% tissue)  [{time.time() - t0:.1f}s]")

# --- Figure 1: thumbnail | tissue mask | tile grid -------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 9))
axes[0].imshow(tissue.thumbnail); axes[0].set_title(f"{args.slide_id} thumbnail")
axes[1].imshow(tissue.mask, cmap="gray"); axes[1].set_title("tissue mask (Otsu on saturation)")
axes[2].imshow(tissue.thumbnail); axes[2].set_title(f"{len(tiles)} tiles @ 20x")
s = row.size_l0 / tissue.downsample
for x, y in zip(tiles.x / tissue.downsample, tiles.y / tissue.downsample):
    axes[2].add_patch(Rectangle((x, y), s, s, fill=False, lw=0.2, ec="tab:blue"))
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig(fig_dir / f"{args.slide_id}_tiling.png", dpi=150)

# --- Figure 2: a few random tiles at full resolution -----------------------------
rng = np.random.default_rng(0)
sample = tiles.iloc[rng.choice(len(tiles), 16, replace=False)]
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for ax, (_, r) in zip(axes.flat, sample.iterrows()):
    ax.imshow(read_tile(slide, r))
    ax.set_title(f"tissue {r.tissue_frac:.0%}", fontsize=8)
    ax.axis("off")
fig.tight_layout()
fig.savefig(fig_dir / f"{args.slide_id}_sample_tiles.png", dpi=120)
print(f"figures -> {fig_dir}")
