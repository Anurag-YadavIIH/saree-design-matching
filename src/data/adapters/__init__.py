"""Adapter registry.

Adding a source is one new module plus one entry here plus one config block. Nothing
downstream changes, which is the whole point of the data contract.
"""

from __future__ import annotations

from pathlib import Path

from .base import BaseAdapter, Record, UnmatchedFile, stable_image_id
from .deeplure import DeepLureAdapter
from .kaggle import KaggleAdapter

REGISTRY: dict[str, type[BaseAdapter]] = {
    DeepLureAdapter.name: DeepLureAdapter,
    KaggleAdapter.name: KaggleAdapter,
}


def get_adapter(source_key: str, cfg: dict, data_root: Path) -> BaseAdapter:
    key = cfg.get("adapter")
    if key not in REGISTRY:
        raise KeyError(f"unknown adapter '{key}' for source '{source_key}'. "
                       f"Known adapters: {sorted(REGISTRY)}")
    return REGISTRY[key](source_key, cfg, data_root)


__all__ = ["BaseAdapter", "Record", "UnmatchedFile", "stable_image_id",
           "REGISTRY", "get_adapter"]
