"""
The ONE way downstream code reads the manifest.

Everything after Phase 1 (augmentation, sampler, training, eval, inference) goes through
`load_manifest`. Nothing downstream globs `data/`, parses a filename, or knows that Roboflow
exports look like `<stem>_jpg.rf.<hash>.jpg`. That knowledge lives in the adapters and the
config, which is what makes it safe to change how labels are derived without touching
training code.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from src.data.schema import COLUMNS, DTYPES

DEFAULT_CONFIG = "configs/data.yaml"

# Columns that pandas would happily mangle. `keep_default_na=False` stops an empty string
# becoming NaN, which matters because "" is a MEANINGFUL value here: it marks a real image
# with no colourway, no parent and no recolour provenance.
_READ_KWARGS = dict(keep_default_na=False, na_values=[])


@lru_cache(maxsize=4)
def _load_raw(config_path: str) -> tuple[pd.DataFrame, dict]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    path = Path(cfg["paths"]["manifest"])
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first: python -m src.data.build_manifest")
    df = pd.read_csv(path, **_READ_KWARGS)

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest is missing contract columns {missing}; "
                         f"rebuild with the current src/data/build_manifest.py")
    df = df[COLUMNS]

    # CSV has no booleans, so restore them explicitly rather than letting truthiness of the
    # string "False" silently make every row True.
    for col in ("is_representative", "is_synthetic"):
        if df[col].dtype == object:
            df[col] = df[col].map({"True": True, "False": False,
                                   True: True, False: False}).astype(bool)
    for col, dt in DTYPES.items():
        if dt in ("int64",):
            df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")
    return df, cfg


def load_manifest(split: str | None = None, role: str | None = None,
                  source: str | None = None, query_kind: str | None = None,
                  include_synthetic: bool | None = None,
                  representatives_only: bool = False,
                  config_path: str = DEFAULT_CONFIG) -> pd.DataFrame:
    """Return a filtered copy of the manifest.

    Every filter is a column equality on the contract, so a caller never needs to know how a
    value was derived.

    split              train / val / test
    role               train / gallery / query
    source             e.g. deeplure_drive
    query_kind         real_view / real_view_recolored / synthetic_recolor
    include_synthetic  None keeps both, True only generated, False only real
    representatives_only  drop non-representative duplicates (train-only extra positives)
    """
    df, _ = _load_raw(config_path)
    out = df
    if split is not None:
        out = out[out["split"] == split]
    if role is not None:
        out = out[out["role"] == role]
    if source is not None:
        out = out[out["source"] == source]
    if query_kind is not None:
        out = out[out["query_kind"] == query_kind]
    if include_synthetic is not None:
        out = out[out["is_synthetic"] == bool(include_synthetic)]
    if representatives_only:
        out = out[out["is_representative"]]
    return out.reset_index(drop=True).copy()


def load_label_map(config_path: str = DEFAULT_CONFIG) -> dict[str, int]:
    """design_id to contiguous class index, train split only."""
    _, cfg = _load_raw(config_path)
    blob = json.loads(Path(cfg["paths"]["label_map"]).read_text(encoding="utf-8"))
    return blob["label_map"]


def data_root(config_path: str = DEFAULT_CONFIG) -> Path:
    _, cfg = _load_raw(config_path)
    return Path(cfg["paths"]["data_root"])


def load_config(config_path: str = DEFAULT_CONFIG) -> dict:
    _, cfg = _load_raw(config_path)
    return cfg
