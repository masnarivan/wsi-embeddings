"""CAMELYON16 annotations (ASAP XML) -> tumor geometry -> per-tile labels.

The XML lists polygons in level-0 pixel coordinates, each tagged with a group:
  - "Tumor" (or "_0" / "_1" in some files): metastasis outlines
  - "Exclusion" (or "_2"): non-tumor holes drawn inside a tumor region
Tumor area = union(tumor polygons) - union(exclusion polygons).
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

TUMOR_GROUPS = {"Tumor", "_0", "_1"}
EXCLUSION_GROUPS = {"Exclusion", "_2"}


def load_tumor_geometry(xml_path: Path) -> shapely.Geometry:
    tumor, exclusion = [], []
    for ann in ET.parse(xml_path).getroot().iter("Annotation"):
        coords = sorted(ann.iter("Coordinate"), key=lambda c: int(c.get("Order")))
        pts = [(float(c.get("X")), float(c.get("Y"))) for c in coords]
        if len(pts) < 3:
            continue
        # Hand-drawn outlines often self-intersect; make_valid repairs them.
        poly = shapely.make_valid(Polygon(pts))
        group = ann.get("PartOfGroup")
        if group in TUMOR_GROUPS:
            tumor.append(poly)
        elif group in EXCLUSION_GROUPS:
            exclusion.append(poly)
    geom = unary_union(tumor).difference(unary_union(exclusion))
    # make_valid can leave stray zero-area lines/points; keep only the polygons.
    polys = [p for p in shapely.get_parts(geom) if p.geom_type in ("Polygon", "MultiPolygon")]
    return unary_union(polys)


def tumor_fraction(tiles: pd.DataFrame, tumor: shapely.Geometry) -> np.ndarray:
    """Fraction of each tile's area covered by tumor (vectorised over all tiles)."""
    boxes = shapely.box(tiles.x, tiles.y, tiles.x + tiles.size_l0, tiles.y + tiles.size_l0)
    shapely.prepare(tumor)
    frac = np.zeros(len(tiles))
    touching = np.flatnonzero(shapely.intersects(boxes, tumor))  # cheap test first
    # clip_by_rect is a specialised axis-aligned clip: same areas as shapely.intersection
    # here, but ~400x faster (0.2 s vs 90 s on tumor_031's ~56k-vertex outline).
    for i in touching:
        frac[i] = shapely.clip_by_rect(tumor, *boxes[i].bounds).area / boxes[i].area
    return frac


def label_tiles(frac: np.ndarray, tumor_min: float = 0.5) -> np.ndarray:
    """tumor (>tumor_min), normal (no tumor at all), boundary (anything in between)."""
    return np.where(frac > tumor_min, "tumor", np.where(frac == 0, "normal", "boundary"))
