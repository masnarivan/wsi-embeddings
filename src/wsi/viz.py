"""Plot helpers: tiles drawn back onto the slide thumbnail."""
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from wsi.tiling import TissueMask

LABEL_COLORS = {"tumor": "tab:red", "normal": "tab:blue", "boundary": "0.6"}


def show_tissue(ax: Axes, tissue: TissueMask, pad: int = 20) -> tuple[int, int]:
    """Show the thumbnail cropped to the tissue bounding box; return the crop offset (x, y)."""
    ys, xs = np.nonzero(tissue.mask)
    y0, x0 = max(ys.min() - pad, 0), max(xs.min() - pad, 0)
    ax.imshow(tissue.thumbnail[y0:ys.max() + pad, x0:xs.max() + pad])
    ax.axis("off")
    return x0, y0


def draw_tiles(ax: Axes, tiles: pd.DataFrame, tissue: TissueMask, offset: tuple[int, int],
               colors, alpha: float = 0.5) -> None:
    """Fill each tile's footprint on a thumbnail shown with show_tissue()."""
    ds = tissue.downsample
    s = tiles.size_l0.iloc[0] / ds
    rects = [Rectangle((x / ds - offset[0], y / ds - offset[1]), s, s) for x, y in zip(tiles.x, tiles.y)]
    ax.add_collection(PatchCollection(rects, facecolor=colors, alpha=alpha, lw=0))
