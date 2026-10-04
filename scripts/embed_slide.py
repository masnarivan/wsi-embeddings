"""Step 4: extract Phikon embeddings for every tile of a slide.

Usage: uv run scripts/embed_slide.py tumor_031 [--max-tiles 256] [--check]
Outputs (aligned row by row): outputs/<slide>/embeddings.npy, outputs/<slide>/tiles_embedded.parquet
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from wsi.dataset import TileDataset, phikon_transform
from wsi.embed import embed_tiles, get_device, load_phikon

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slide_id")
    parser.add_argument("--max-tiles", type=int, default=None, help="random subset, for quick tests")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--check", action="store_true", help="embed twice and compare (determinism)")
    args = parser.parse_args()

    out_dir = ROOT / "outputs" / args.slide_id
    tiles = pd.read_parquet(out_dir / "tiles.parquet")
    if args.max_tiles:
        tiles = tiles.sample(min(args.max_tiles, len(tiles)), random_state=0)

    dataset = TileDataset(ROOT / "data/raw" / f"{args.slide_id}.tif", tiles, phikon_transform())
    # No shuffling: inference only, and a fixed order keeps runs reproducible.
    # persistent_workers keeps the workers alive for the second pass of --check.
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    device = get_device()
    model = load_phikon(device)

    x, idx = dataset[0]
    print(f"device {device} | one sample: {tuple(x.shape)} {x.dtype}, mean {x.mean():.2f}, std {x.std():.2f}")

    t0 = time.time()
    emb = embed_tiles(model, loader, device)
    dt = time.time() - t0
    print(f"embeddings {emb.shape} in {dt:.1f}s ({len(dataset) / dt:.0f} tiles/s), NaN: {np.isnan(emb).any()}")

    if args.check:
        emb2 = embed_tiles(model, loader, device)
        print(f"determinism: max |run1 - run2| = {np.abs(emb - emb2).max():.2e}")

    np.save(out_dir / "embeddings.npy", emb)
    dataset.tiles.to_parquet(out_dir / "tiles_embedded.parquet")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":  # required on macOS: DataLoader workers re-import this file
    main()
