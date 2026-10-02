"""
The manifest schema in one place, imported by the builder, the validator and the loader.

Defined here rather than repeated so the three can never disagree about column order, dtype
or allowed values. See docs/DATA_CONTRACT.md for what each column means and why it exists.
"""

from __future__ import annotations

SCHEMA_VERSION = 3

# Column order is part of the contract, so a diff of two manifests stays readable.
COLUMNS: list[str] = [
    # identity
    "image_id",
    "path",
    "source",
    "design_id",
    "colorway_id",
    "dup_group",
    "is_representative",
    "parent_image_id",
    "craft_family",
    # file facts
    "width",
    "height",
    "phash",
    # split and role
    "split",
    "role",
    "query_kind",
    "is_synthetic",
    # synthesis provenance
    "recolor_method",
    "palette_id",
    "recolor_seed",
    "geo_params",
    # quality
    "label_confidence",
    "notes",
]

DTYPES: dict[str, str] = {
    "image_id": "string",
    "path": "string",
    "source": "string",
    "design_id": "string",
    "colorway_id": "string",
    "dup_group": "string",
    "is_representative": "bool",
    "parent_image_id": "string",
    "craft_family": "string",
    "width": "int64",
    "height": "int64",
    "phash": "string",
    "split": "string",
    "role": "string",
    "query_kind": "string",
    "is_synthetic": "bool",
    "recolor_method": "string",
    "palette_id": "string",
    "recolor_seed": "int64",
    "geo_params": "string",
    "label_confidence": "string",
    "notes": "string",
}

SPLITS = ("train", "val", "test")
ROLES = ("train", "gallery", "query")
QUERY_KINDS = ("", "real_view", "real_view_recolored", "synthetic_recolor",
               "synthetic_tonal")
CONFIDENCES = ("high", "medium", "low")

# Sentinel for "this row is real, so it has no recolour seed".
NO_SEED = -1

# A verified pair whose aligned overlap is at least this large is the SAME PIXELS, so the
# two images are duplicates rather than different views of one design. Below it, the pair is
# the same design seen in a different region, which is a genuine positive pair.
DUP_OVERLAP_MIN = 0.85


def empty_row() -> dict:
    """A row with every column present, so no caller can omit one by accident."""
    return {
        "image_id": "", "path": "", "source": "", "design_id": "", "colorway_id": "",
        "dup_group": "", "is_representative": False, "parent_image_id": "",
        "craft_family": "", "width": 0, "height": 0, "phash": "", "split": "",
        "role": "", "query_kind": "", "is_synthetic": False, "recolor_method": "",
        "palette_id": "", "recolor_seed": NO_SEED, "geo_params": "",
        "label_confidence": "medium", "notes": "",
    }
