"""Step 1 sanity check: is the environment ready (PyTorch + MPS, OpenSlide)?"""
import sys
from pathlib import Path

import openslide
import torch

print(f"python   {sys.version.split()[0]}")
print(f"torch    {torch.__version__}  | mps available = {torch.backends.mps.is_available()}")
print(f"openslide library {openslide.__library_version__}")

slide_path = Path(__file__).parent.parent / "data/raw/tumor_031.tif"
if slide_path.exists():
    slide = openslide.OpenSlide(slide_path)
    mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    print(f"\n{slide_path.name}: level-0 size {slide.dimensions}, mpp {mpp:.3f} (~{10 / mpp:.0f}x)")
    for lvl, (dims, ds) in enumerate(zip(slide.level_dimensions, slide.level_downsamples)):
        print(f"  level {lvl}: {dims[0]:>6} x {dims[1]:<6}  downsample {ds:.0f}")
else:
    print(f"\n{slide_path.name} not downloaded yet")
