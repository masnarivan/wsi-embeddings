"""Step 3: label each tile tumor / normal / boundary from the expert annotations.

Usage: uv run scripts/label_tiles.py tumor_031   (run tile_slide.py first)
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from matplotlib.patches import Patch

from wsi.annotations import label_tiles, load_tumor_geometry, tumor_fraction
from wsi.tiling import compute_tissue_mask, read_tile
from wsi.viz import LABEL_COLORS as COLORS, draw_tiles, show_tissue

ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("slide_id")
args = parser.parse_args()

out_dir = ROOT / "outputs" / args.slide_id
fig_dir = ROOT / "outputs" / "figures"
tiles = pd.read_parquet(out_dir / "tiles.parquet")
tumor = load_tumor_geometry(ROOT / "data/raw" / f"{args.slide_id}.xml")

tiles["tumor_frac"] = tumor_fraction(tiles, tumor)
tiles["label"] = label_tiles(tiles.tumor_frac.to_numpy())
tiles.to_parquet(out_dir / "tiles.parquet")
print(tiles.label.value_counts().to_string())

# --- Figure 1: annotation outline | tiles colored by label -----------------------
slide = openslide.OpenSlide(ROOT / "data/raw" / f"{args.slide_id}.tif")
tissue = compute_tissue_mask(slide)
ds = tissue.downsample

fig, axes = plt.subplots(1, 2, figsize=(10, 9))
x_off, y_off = show_tissue(axes[0], tissue)
show_tissue(axes[1], tissue)
for poly in getattr(tumor, "geoms", [tumor]):
    for ring in [poly.exterior, *poly.interiors]:
        x, y = np.asarray(ring.coords).T
        axes[0].plot(x / ds - x_off, y / ds - y_off, lw=0.8, c="lime")
axes[0].set_title(f"{args.slide_id}: expert tumor annotation")

draw_tiles(axes[1], tiles, tissue, (x_off, y_off), tiles.label.map(COLORS))
axes[1].legend(handles=[Patch(color=c, label=f"{k} ({(tiles.label == k).sum()})") for k, c in COLORS.items()],
               loc="lower left")
axes[1].set_title("tile labels")
fig.tight_layout()
fig.savefig(fig_dir / f"{args.slide_id}_labels.png", dpi=150)

# --- Figure 2: 8 tumor vs 8 normal tiles -----------------------------------------
rng = np.random.default_rng(0)
fig, axes = plt.subplots(4, 4, figsize=(8, 8.5))
for i, lab in enumerate(["tumor", "normal"]):
    sub = tiles[tiles.label == lab]
    for ax, (_, r) in zip(axes[2 * i:2 * i + 2].flat, sub.iloc[rng.choice(len(sub), 8, replace=False)].iterrows()):
        ax.imshow(read_tile(slide, r))
        ax.set_title(lab, fontsize=9, color=COLORS[lab])
        ax.axis("off")
fig.tight_layout()
fig.savefig(fig_dir / f"{args.slide_id}_tumor_vs_normal.png", dpi=120)
print(f"figures -> {fig_dir}")
