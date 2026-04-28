"""MedMNIST -> PyTorch DataLoader helpers.

Wraps MedMNIST's dataset classes so we can:
- ensure the data root directory exists (MedMNIST won't create nested dirs)
- transform images to (1, 28, 28) tensors normalised to [0, 1]
- expose train / val / test splits as PyTorch DataLoaders
- expose the underlying labels array for FL partitioning
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


_DATASET_REGISTRY: dict[str, str] = {
    "breastmnist": "BreastMNIST",
    "pneumoniamnist": "PneumoniaMNIST",
    "dermamnist": "DermaMNIST",
}

# Number of input channels per dataset (1 = grayscale, 3 = RGB)
_DATASET_CHANNELS: dict[str, int] = {
    "breastmnist": 1,
    "pneumoniamnist": 1,
    "dermamnist": 3,
}

# Number of output classes per dataset
_DATASET_NUM_CLASSES: dict[str, int] = {
    "breastmnist": 2,
    "pneumoniamnist": 2,
    "dermamnist": 7,
}


def get_dataset_meta(name: str) -> dict:
    """Return {channels, num_classes} for a registered MedMNIST dataset."""
    key = name.lower()
    return {
        "in_channels": _DATASET_CHANNELS[key],
        "num_classes": _DATASET_NUM_CLASSES[key],
    }


def _get_medmnist_class(name: str):
    import medmnist

    key = name.lower()
    if key not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Known: {list(_DATASET_REGISTRY)}")
    return getattr(medmnist, _DATASET_REGISTRY[key])


class _MedMNISTTensor(Dataset):
    """Thin Dataset wrapper that yields (FloatTensor[1,H,W], LongTensor[]) pairs."""

    def __init__(self, base) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, label = self.base[idx]
        # MedMNIST returns PIL images by default; convert to tensor [0,1].
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        if img.ndim == 2:
            img = img[None, ...]            # (H, W)   -> (1, H, W)
        elif img.ndim == 3 and img.shape[-1] in (1, 3):
            img = img.transpose(2, 0, 1)    # (H, W, C) -> (C, H, W)
        x = torch.from_numpy(img.astype("float32") / 255.0)
        y = torch.tensor(int(np.asarray(label).flatten()[0]), dtype=torch.long)
        return x, y


def get_dataset(
    name: str,
    split: str = "train",
    root: str | Path = "data/raw",
) -> _MedMNISTTensor:
    Path(root).mkdir(parents=True, exist_ok=True)
    cls = _get_medmnist_class(name)
    base = cls(split=split, download=True, root=str(root))
    return _MedMNISTTensor(base)


def get_dataloaders(
    name: str,
    batch_size: int = 32,
    root: str | Path = "data/raw",
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = get_dataset(name, split="train", root=root)
    val_ds = get_dataset(name, split="val", root=root)
    test_ds = get_dataset(name, split="test", root=root)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def get_labels(name: str, split: str = "train", root: str | Path = "data/raw") -> np.ndarray:
    """Return the full label array for a split — used by FL partitioners."""
    cls = _get_medmnist_class(name)
    base = cls(split=split, download=True, root=str(root))
    return np.asarray(base.labels).flatten().astype(int)


if __name__ == "__main__":
    tr, va, te = get_dataloaders("breastmnist", batch_size=8)
    x, y = next(iter(tr))
    print(f"breastmnist train batch: x={tuple(x.shape)} y={tuple(y.shape)} dtype={x.dtype}")
    print(f"sizes: train={len(tr.dataset)} val={len(va.dataset)} test={len(te.dataset)}")
