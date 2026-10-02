"""
Tests for the data contract, runnable without the proprietary corpus.

A tiny fake corpus is generated in a temp directory, mimicking both real layouts:

  Drive-like   a flat folder of `img_<id>.jpg`, including TWO OVERLAPPING CROPS of one large
               texture. That reproduces the central Phase 1 finding: re-crops that pHash cannot
               see but geometric verification must merge into a single design.
  Kaggle-like  `<vendor split>/<CraftFamily>/<stem>_jpg.rf.<hash>.jpg`, including a noise
               triplet that must collapse into one duplicate group.

The most important tests are the NEGATIVE ones. A validator that passes on a good manifest
proves little; what matters is that it FAILS on a leaky one. Each of those tests corrupts a
valid manifest in one specific way and asserts the validator catches it.

Run:  python -m pytest tests/ -q
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build_manifest import build  # noqa: E402
from src.data.validate import validate_invariants  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _texture(seed: int, size: int) -> np.ndarray:
    """A random but structured texture: smooth blobs plus fine grain.

    Solid colours would give ORB no keypoints at all, so the re-crop test would pass for the
    wrong reason. A texture with real structure exercises the actual matching path.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, size=(size // 16, size // 16, 3), dtype=np.uint8)
    img = np.asarray(Image.fromarray(coarse).resize((size, size), Image.BICUBIC), dtype=np.int16)
    grain = rng.integers(-40, 41, size=(size, size, 3))
    return np.clip(img + grain, 0, 255).astype(np.uint8)


def _save(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, quality=95)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("fake")
    data = root / "data"
    drive = data / "Google_drive_data" / "handloom_sarees"

    # Two overlapping crops of ONE large texture: the same design at different offsets.
    big = _texture(seed=1, size=900)
    _save(big[0:600, 0:600], drive / "img_100001.jpg")
    _save(big[300:900, 300:900], drive / "img_100002.jpg")
    # Distinct single-image Drive designs.
    for i in range(10):
        _save(_texture(seed=100 + i, size=500), drive / f"img_2000{i:02d}.jpg")

    # Kaggle-like layout, including a noise triplet of one source image.
    fams = ["Banarasi", "Ikat"]
    for f_i, fam in enumerate(fams):
        for d in range(8):
            base = _texture(seed=500 + 50 * f_i + d, size=400)
            stem = f"{fam.lower()}_motif{d}"
            n_copies = 3 if d == 0 else 1
            for c in range(n_copies):
                noisy = base.copy()
                if c:
                    # Sparse salt-and-pepper, as Roboflow adds: same framing, same pixels
                    # otherwise, so this must collapse into ONE duplicate group.
                    rng = np.random.default_rng(c)
                    idx = rng.random(noisy.shape[:2]) < 0.001
                    noisy[idx] = 255
                h = hashlib.md5(f"{stem}{c}".encode()).hexdigest()
                split = ["train", "valid", "test"][d % 3]
                _save(noisy, data / "Kaggle_data" / split / fam / f"{stem}_jpg.rf.{h}.jpg")

    cfg = yaml.safe_load((REPO / "configs" / "data.yaml").read_text(encoding="utf-8"))
    cfg["paths"] = {
        "data_root": str(data),
        "synthetic_dir": str(data / "synthetic"),
        "manifest": str(root / "out" / "manifest.csv"),
        "label_map": str(root / "out" / "label_map.json"),
        "summary": str(root / "out" / "data_summary.md"),
        "unmatched": str(root / "out" / "unmatched.csv"),
        "review": str(root / "out" / "review_flags.csv"),
    }
    cfg["sources"]["deeplure_drive"]["root"] = str(data / "Google_drive_data")
    cfg["sources"]["kaggle_fabric"]["root"] = str(data / "Kaggle_data")
    # Keep the test offline and fast: no DINOv2 download. The corpus is small enough for
    # all-pairs verification, so the prefilter is never needed anyway.
    cfg["dedup"]["use_dinov2_prefilter"] = False
    cfg["split"]["extra_grouping"]["enabled"] = False
    cfg["eval"]["queries_per_design"] = 2

    cfg_path = root / "data.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    df = build(str(cfg_path))
    label_map = json.loads(Path(cfg["paths"]["label_map"]).read_text())["label_map"]
    return {"df": df, "cfg": cfg, "label_map": label_map}


# ------------------------------------------------------------------- positive behaviour

def test_valid_manifest_passes(corpus):
    f = validate_invariants(corpus["df"], corpus["cfg"], corpus["label_map"])
    assert not f.items, f.items


def test_recrops_merge_into_one_design(corpus):
    """The core Phase 1 finding: overlapping crops are one design even though pHash
    disagrees. If this regresses, the pipeline would leak re-crops across splits."""
    df = corpus["df"]
    real = df[~df["is_synthetic"]]
    a = real[real["path"].str.endswith("img_100001.jpg")]["design_id"].iloc[0]
    b = real[real["path"].str.endswith("img_100002.jpg")]["design_id"].iloc[0]
    assert a == b, "overlapping crops of one texture were not merged into one design"


def test_noise_triplet_is_one_dup_group_with_one_representative(corpus):
    df = corpus["df"]
    real = df[~df["is_synthetic"]]
    trip = real[real["path"].str.contains("banarasi_motif0_jpg")]
    assert len(trip) == 3
    assert trip["dup_group"].nunique() == 1
    assert int(trip["is_representative"].sum()) == 1


def test_unrelated_designs_stay_separate(corpus):
    real = corpus["df"][~corpus["df"]["is_synthetic"]]
    singles = real[real["path"].str.contains("img_2000")]
    assert singles["design_id"].nunique() == 10


def test_synthetic_rows_inherit_parent_identity(corpus):
    df = corpus["df"]
    parent = df.set_index("image_id")
    for _, r in df[df["is_synthetic"]].iterrows():
        p = parent.loc[r["parent_image_id"]]
        assert r["design_id"] == p["design_id"]
        assert r["source"] == p["source"]
        assert r["split"] == p["split"]


def test_eval_never_uses_training_recolor_method(corpus):
    df = corpus["df"]
    train_method = corpus["cfg"]["recolor"]["methods"]["train"]
    ev = df[df["split"].isin(["val", "test"]) & df["is_synthetic"]]
    assert not (ev["recolor_method"] == train_method).any()


def test_build_is_deterministic(corpus, tmp_path):
    """Same config and seed must give an identical manifest. Guards against seeding with
    Python's per-process randomised hash(), which once slipped into the builder."""
    df1 = corpus["df"]
    cfg = copy.deepcopy(corpus["cfg"])
    for k in ("manifest", "label_map", "summary", "unmatched", "review"):
        cfg["paths"][k] = str(tmp_path / Path(cfg["paths"][k]).name)
    # Must stay under data_root: the contract stores every path relative to it. A separate
    # subfolder keeps this rebuild from overwriting the first build's images.
    cfg["paths"]["synthetic_dir"] = str(Path(cfg["paths"]["data_root"]) / "synthetic_rebuild")
    p = tmp_path / "data.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    df2 = build(str(p))
    cols = ["design_id", "dup_group", "split", "role", "query_kind", "palette_id",
            "recolor_seed"]
    real1 = df1[~df1["is_synthetic"]].sort_values("path")[["path", *cols]].reset_index(drop=True)
    real2 = df2[~df2["is_synthetic"]].sort_values("path")[["path", *cols]].reset_index(drop=True)
    pd.testing.assert_frame_equal(real1, real2)
    s1 = sorted(zip(df1.loc[df1["is_synthetic"], "parent_image_id"],
                    df1.loc[df1["is_synthetic"], "recolor_seed"]))
    s2 = sorted(zip(df2.loc[df2["is_synthetic"], "parent_image_id"],
                    df2.loc[df2["is_synthetic"], "recolor_seed"]))
    assert s1 == s2, "synthetic recolour seeds differ between identical builds"


# ------------------------------------------------- negative behaviour: must FAIL when leaky

def _fails(df, corpus, needle: str) -> bool:
    f = validate_invariants(df, corpus["cfg"], corpus["label_map"])
    return any(needle in m for m in f.items)


def test_catches_design_leaking_across_splits(corpus):
    df = corpus["df"].copy()
    real = df[~df["is_synthetic"]]
    victim = real[real["split"] == "test"].iloc[0]
    # Move ONE image of a test design into train: the classic leak.
    sib = real[(real["design_id"] == victim["design_id"])].index[0]
    df.loc[sib, "split"] = "train"
    assert _fails(df, corpus, "design_id(s) appear in more than one split")


def test_catches_training_method_in_test(corpus):
    df = corpus["df"].copy()
    i = df[(df["split"] == "test") & df["is_synthetic"]].index[0]
    df.loc[i, "recolor_method"] = corpus["cfg"]["recolor"]["methods"]["train"]
    assert _fails(df, corpus, "TRAIN recolor method")


def test_catches_synthetic_gallery(corpus):
    df = corpus["df"].copy()
    i = df[(df["role"] == "query") & df["is_synthetic"]].index[0]
    df.loc[i, "role"] = "gallery"
    df.loc[i, "query_kind"] = ""
    assert _fails(df, corpus, "gallery rows are synthetic")


def test_catches_palette_shared_across_splits(corpus):
    df = corpus["df"].copy()
    val_pal = df[(df["split"] == "val") & df["is_synthetic"]]["palette_id"].iloc[0]
    i = df[(df["split"] == "test") & df["is_synthetic"]].index[0]
    df.loc[i, "palette_id"] = val_pal
    assert _fails(df, corpus, "palette_id(s) used in more than one split")


def test_catches_missing_representative(corpus):
    df = corpus["df"].copy()
    i = df[df["is_representative"] & ~df["is_synthetic"]].index[0]
    df.loc[i, "is_representative"] = False
    assert _fails(df, corpus, "exactly one representative")


def test_catches_stale_label_map(corpus):
    stale = dict(corpus["label_map"])
    stale["not_a_real_design"] = 999
    f = validate_invariants(corpus["df"], corpus["cfg"], stale)
    assert any("label_map covers" in m for m in f.items)
