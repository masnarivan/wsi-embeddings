"""PyTorch Dataset that serves tiles straight from the whole-slide image.

Key idea: a Dataset only needs __len__ and __getitem__(i). The DataLoader then calls
__getitem__ from several worker processes in parallel and stacks the results into batches.
"""
from pathlib import Path

import openslide
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from transformers import AutoImageProcessor

from wsi.tiling import read_tile

MODEL_ID = "owkin/phikon"


def phikon_transform() -> T.Compose:
    """PIL image -> normalised float tensor (3, 224, 224), as Phikon saw during pretraining."""
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    return T.Compose([
        T.ToTensor(),                                           # uint8 HWC [0,255] -> float CHW [0,1]
        T.Normalize(mean=proc.image_mean, std=proc.image_std),  # ImageNet statistics
    ])


class TileDataset(Dataset):
    def __init__(self, slide_path: Path, tiles: pd.DataFrame, transform):
        self.slide_path = slide_path
        self.tiles = tiles.reset_index(drop=True)
        self.transform = transform
        # Do NOT open the slide here: an OpenSlide handle cannot be pickled, and on macOS
        # each DataLoader worker receives a pickled copy of this object (spawn start method).
        self._slide = None

    @property
    def slide(self) -> openslide.OpenSlide:
        if self._slide is None:  # opened lazily, once per worker process
            self._slide = openslide.OpenSlide(self.slide_path)
        return self._slide

    def __getstate__(self):
        # Called when the dataset is pickled for a worker: never ship an open handle,
        # even if the main process already used it (e.g. dataset[0] for a sanity check).
        return {**self.__dict__, "_slide": None}

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # idx is returned with the tensor so embed_tiles can write each embedding to the
        # right row whatever order the batches arrive in (e.g. if shuffling is turned on).
        row = self.tiles.iloc[idx]
        img = read_tile(self.slide, row)
        return self.transform(img), idx
