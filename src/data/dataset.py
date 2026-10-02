"""
Datasets. Transforms are injected, so colour augmentation plugs in without editing this file.

Two classes, because training and evaluation want genuinely different things:

  SareeDataset       one row to one tensor. Used for evaluation and inference, where an
                     image must be read exactly as the manifest describes it.
  TrainViewDataset   one design to K augmented views. Used for SupCon, which needs several
                     positives of the same identity in a batch.

Colour normalisation happens here and nowhere else. Every image is converted to RGB on load.
The corpus is RGB today, but palette (P), grayscale (L) and alpha (RGBA) files are routine in
scraped textile data, and an alpha channel silently composited differently by another library
is a reproducibility bug that is painful to find. One conversion point, one comment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# ImageNet statistics, which is what the DINOv2 checkpoints were trained with.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_rgb(path: Path) -> np.ndarray:
    """THE single place images become RGB. See module docstring for why."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def to_tensor(rgb: np.ndarray, size: int | None = None,
              grayscale: bool = False) -> torch.Tensor:
    """HWC uint8 to normalised CHW float tensor.

    `grayscale` replicates the single channel back to 3, so a grayscale baseline can reuse
    the same backbone without changing its first layer.
    """
    import cv2

    if size is not None and (rgb.shape[0] != size or rgb.shape[1] != size):
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    if grayscale:
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        rgb = np.stack([g, g, g], axis=-1)
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class SareeDataset(Dataset):
    """One manifest row to one sample.

    Returns {"image", "label", "design_id", "image_id"}. `label` is -1 when the design is not
    in the label map, which is the correct value for val and test: those designs are unseen
    by construction, so no class index exists for them.
    """

    def __init__(self, manifest_df: pd.DataFrame, data_root: Path,
                 transform: Callable[[np.ndarray], np.ndarray] | None = None,
                 label_map: dict[str, int] | None = None,
                 img_size: int = 224, grayscale: bool = False):
        self.df = manifest_df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform
        self.label_map = label_map or {}
        self.img_size = img_size
        self.grayscale = grayscale

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        rgb = read_rgb(self.data_root / row["path"])
        if self.transform is not None:
            rgb = self.transform(rgb)
        return {
            "image": to_tensor(rgb, self.img_size, self.grayscale),
            "label": int(self.label_map.get(row["design_id"], -1)),
            "design_id": str(row["design_id"]),
            "image_id": str(row["image_id"]),
        }


class TrainViewDataset(Dataset):
    """One design to K augmented views, for SupCon.

    Indexed by DESIGN, not by image, because the loss needs several positives of one identity
    together. Each view independently samples a source image for that design (any member of
    any of its duplicate groups, so non-representative extras are used as free positives),
    then applies the injected view transform.

    `same_palette_group` lets the sampler force a set of different designs into one palette,
    producing hard negatives where colour is uninformative and only the motif separates them.
    """

    def __init__(self, manifest_df: pd.DataFrame, data_root: Path,
                 label_map: dict[str, int], views_per_design: int,
                 view_transform: Callable[[np.ndarray, int, str | None, str], np.ndarray],
                 img_size: int = 224, seed: int = 42, grayscale: bool = False):
        self.data_root = Path(data_root)
        self.grayscale = grayscale
        self.label_map = label_map
        self.k = views_per_design
        self.view_transform = view_transform
        self.img_size = img_size
        self.seed = seed

        df = manifest_df[manifest_df["split"] == "train"]
        self.designs: list[str] = sorted(df["design_id"].unique())
        self.paths_by_design: dict[str, list[str]] = {
            d: sorted(g["path"].tolist()) for d, g in df.groupby("design_id")
        }

    def __len__(self) -> int:
        return len(self.designs)

    def __getitem__(self, key) -> dict:
        # The sampler passes (design index, forced palette or None). The palette rides inside
        # the key because it is the only channel guaranteed to reach a DataLoader worker; see
        # PKDesignSampler. A bare int is accepted too, meaning "no forced palette".
        if isinstance(key, (tuple, list)):
            i, palette_id = int(key[0]), key[1]
        else:
            i, palette_id = int(key), None
        design = self.designs[i]
        pool = self.paths_by_design[design]
        # Fresh randomness every call, so the same design yields different views each time it
        # is drawn. Reproducibility comes from seeding the worker processes, not from reusing
        # one fixed stream per design, which would show the model identical views every epoch.
        # The worker id is part of the seed because each worker keeps its OWN call counter;
        # without it, two workers could reach the same (design, count) and emit identical views.
        self._calls = getattr(self, "_calls", 0) + 1
        info = torch.utils.data.get_worker_info()
        worker = info.id if info is not None else 0
        rng = np.random.default_rng((self.seed, i, worker, self._calls))

        images = []
        for v in range(self.k):
            rel = pool[int(rng.integers(len(pool)))]
            rgb = read_rgb(self.data_root / rel)
            seed_v = int(rng.integers(0, 2 ** 31 - 1))
            # The relative path is passed so the transform can look up this image's CACHED
            # k-means cluster map instead of rerunning k-means, which was the bottleneck.
            rgb = self.view_transform(rgb, seed_v, palette_id, rel)
            images.append(to_tensor(rgb, self.img_size, self.grayscale))

        return {
            "images": torch.stack(images),                # K, C, H, W
            "label": int(self.label_map.get(design, -1)),
            "design_id": design,
        }



def collate_views(batch: list[dict]) -> dict:
    """Flatten per-design view stacks into one batch.

    SupCon wants a flat tensor plus a label per row, where rows sharing a label are
    positives. So (P designs, K views) becomes (P*K) with each label repeated K times.
    """
    images = torch.cat([b["images"] for b in batch], dim=0)
    k = batch[0]["images"].shape[0]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long).repeat_interleave(k)
    design_ids = [d for b in batch for d in [b["design_id"]] * k]
    return {"image": images, "label": labels, "design_id": design_ids}
