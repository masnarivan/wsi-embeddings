"""Phikon feature extraction: batches of tiles -> 768-d embeddings (ViT CLS token)."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import ViTModel

from wsi.dataset import MODEL_ID


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_phikon(device: torch.device) -> ViTModel:
    model = ViTModel.from_pretrained(MODEL_ID)  # its pooler head is loaded but unused
    return model.eval().to(device)  # eval(): disable dropout etc. -> deterministic outputs


def embed_tiles(model: ViTModel, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Run the model over every batch and return an (n_tiles, 768) float32 array,
    in the same order as the dataset (row i <-> dataset index i)."""
    out = np.zeros((len(loader.dataset), model.config.hidden_size), dtype=np.float32)
    # last_hidden_state is (B, 197, 768): 1 CLS token + 14 x 14 patch tokens (224 / 16).
    # The CLS token summarises the whole tile and is used as its embedding.
    with torch.inference_mode():  # no autograd graph: less memory, faster
        for images, idx in loader:
            images = images.to(device)
            cls_tokens = model(pixel_values=images).last_hidden_state[:, 0]  # (B, 768)
            out[idx.numpy()] = cls_tokens.cpu().numpy()
    return out
