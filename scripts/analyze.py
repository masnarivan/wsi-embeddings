"""Step 5: what did Phikon learn? UMAP, k-means back on the slide, linear probe.

Usage: uv run scripts/analyze.py tumor_031   (run embed_slide.py first)
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import umap
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wsi.tiling import compute_tissue_mask, read_tile
from wsi.viz import LABEL_COLORS, draw_tiles, show_tissue

ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("slide_id")
parser.add_argument("--k", type=int, default=6, help="number of k-means clusters")
parser.add_argument("--block-tiles", type=int, default=8, help="spatial CV block side, in tiles")
args = parser.parse_args()

out_dir = ROOT / "outputs" / args.slide_id
fig_dir = ROOT / "outputs" / "figures"
emb = np.load(out_dir / "embeddings.npy")
tiles = pd.read_parquet(out_dir / "tiles_embedded.parquet")
assert len(emb) == len(tiles), "embeddings and tile table are not aligned"
slide = openslide.OpenSlide(ROOT / "data/raw" / f"{args.slide_id}.tif")
tissue = compute_tissue_mask(slide)

# --- 1. UMAP: 768-d -> 2-d, preserving local neighbourhoods ------------------------
xy = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(emb)
tiles["umap_x"], tiles["umap_y"] = xy[:, 0], xy[:, 1]

fig, ax = plt.subplots(figsize=(7, 6))
for lab in ["normal", "boundary", "tumor"]:  # tumor drawn last, on top
    sub = tiles[tiles.label == lab]
    ax.scatter(sub.umap_x, sub.umap_y, s=4, c=LABEL_COLORS[lab], label=f"{lab} ({len(sub)})",
               alpha=0.4 if lab == "boundary" else 0.8, lw=0)
ax.legend(markerscale=3)
ax.set(title=f"{args.slide_id}: Phikon embeddings (UMAP), colored by expert annotation",
       xticks=[], yticks=[], xlabel="UMAP 1", ylabel="UMAP 2")
fig.tight_layout()
fig.savefig(fig_dir / f"{args.slide_id}_umap.png", dpi=150)

# --- 2. k-means: unsupervised clusters, drawn back on the slide ---------------------
# PCA to 50 dims first: removes noise dimensions and makes k-means distances more meaningful.
z = PCA(n_components=50, random_state=0).fit_transform(StandardScaler().fit_transform(emb))
tiles["cluster"] = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit_predict(z)
palette = plt.get_cmap("Dark2").colors  # no red/blue: those mean tumor/normal elsewhere
cluster_colors = tiles.cluster.map(lambda c: palette[c])

purity = pd.crosstab(tiles.cluster, tiles.label, normalize="index")[["tumor", "boundary", "normal"]]
print("\ncluster composition (fraction of tiles per expert label):")
print(purity.assign(n=tiles.cluster.value_counts().sort_index()).round(2).to_string())

fig, axes = plt.subplots(1, 3, figsize=(16, 7), gridspec_kw={"width_ratios": [1.1, 1, 1]})
axes[0].scatter(tiles.umap_x, tiles.umap_y, s=4, c=cluster_colors, lw=0)
axes[0].set(title=f"k-means (k={args.k}) on embeddings", xticks=[], yticks=[])
off = show_tissue(axes[1], tissue)
draw_tiles(axes[1], tiles, tissue, off, cluster_colors, alpha=0.6)
axes[1].set_title("clusters on the slide (no labels used)")
axes[1].legend(handles=[Patch(color=palette[c], label=f"cluster {c}: {purity.loc[c, 'tumor']:.0%} tumor")
                        for c in range(args.k)], loc="lower left", fontsize=8)
off = show_tissue(axes[2], tissue)
draw_tiles(axes[2], tiles, tissue, off, tiles.label.map(LABEL_COLORS))
axes[2].set_title("expert annotation")
fig.tight_layout()
fig.savefig(fig_dir / f"{args.slide_id}_clusters.png", dpi=150)

# Example tiles per cluster: what does each cluster look like?
rng = np.random.default_rng(0)
fig, axes = plt.subplots(args.k, 6, figsize=(9, 1.6 * args.k))
for c in range(args.k):
    sub = tiles[tiles.cluster == c]
    for j, (_, r) in enumerate(sub.iloc[rng.choice(len(sub), 6, replace=False)].iterrows()):
        axes[c, j].imshow(read_tile(slide, r))
        axes[c, j].set_xticks([]), axes[c, j].set_yticks([])
        for spine in axes[c, j].spines.values():
            spine.set(color=palette[c], lw=3)
    axes[c, 0].set_ylabel(f"cluster {c}\n{purity.loc[c, 'tumor']:.0%} tumor", fontsize=8)
fig.tight_layout()
fig.savefig(fig_dir / f"{args.slide_id}_cluster_tiles.png", dpi=110)

# --- 3. Linear probe: how linearly separable is tumor vs normal in embedding space? ---
# Boundary tiles are ambiguous by construction -> excluded.
# Spatial CV: neighbouring tiles look alike, so a random split leaks information
# (train on a tile, test on its twin next door). Grouping tiles into blocks of
# block_tiles x block_tiles and keeping each block entirely in train or in test avoids it.
probe = tiles.label.isin(["tumor", "normal"]).to_numpy()
X, y = emb[probe], (tiles.label[probe] == "tumor").to_numpy().astype(int)
block = (tiles.size_l0.iloc[0] * args.block_tiles)
groups = (tiles.x[probe] // block).astype(str) + "_" + (tiles.y[probe] // block).astype(str)

clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
p_random = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                             method="predict_proba")[:, 1]
p_spatial = cross_val_predict(clf, X, y, cv=GroupKFold(5), groups=groups, method="predict_proba")[:, 1]
auc_random, auc_spatial = roc_auc_score(y, p_random), roc_auc_score(y, p_spatial)
mm = block * float(slide.properties["openslide.mpp-x"]) / 1000
print(f"\nlinear probe, tumor vs normal ({y.sum()} vs {len(y) - y.sum()} tiles, 5-fold):")
print(f"  random split            AUC = {auc_random:.3f}")
print(f"  spatial split ({mm:.1f} mm blocks, {groups.nunique()} blocks) AUC = {auc_spatial:.3f}")

# Controls. Negative: permuted labels must give ~0.5, otherwise the pipeline leaks.
# Harder task: keep boundary tiles, positive = any tumor in the tile.
y_perm = np.random.default_rng(0).permutation(y)
p_perm = cross_val_predict(clf, X, y_perm, cv=GroupKFold(5), groups=groups, method="predict_proba")[:, 1]
y_any = (tiles.tumor_frac > 0).to_numpy().astype(int)
groups_all = (tiles.x // block).astype(str) + "_" + (tiles.y // block).astype(str)
p_any = cross_val_predict(clf, emb, y_any, cv=GroupKFold(5), groups=groups_all, method="predict_proba")[:, 1]
print(f"  control, permuted labels AUC = {roc_auc_score(y_perm, p_perm):.3f}  (should be ~0.5)")
print(f"  harder, incl. boundary (any tumor) AUC = {roc_auc_score(y_any, p_any):.3f}")

# Where does the probe disagree with the annotation? (spatial-CV predictions)
tiles["p_tumor"] = np.nan
tiles.loc[probe, "p_tumor"] = p_spatial
fig, axes = plt.subplots(1, 2, figsize=(10, 7))
off = show_tissue(axes[0], tissue)
draw_tiles(axes[0], tiles[probe], tissue, off, plt.get_cmap("coolwarm")(p_spatial), alpha=0.7)
axes[0].set_title(f"linear probe P(tumor), spatial CV\nAUC {auc_spatial:.3f}")
off = show_tissue(axes[1], tissue)
draw_tiles(axes[1], tiles, tissue, off, tiles.label.map(LABEL_COLORS))
axes[1].set_title("expert annotation")
fig.tight_layout()
fig.savefig(fig_dir / f"{args.slide_id}_probe.png", dpi=150)

tiles.to_parquet(out_dir / "tiles_analyzed.parquet")
print(f"\nfigures -> {fig_dir}")
