"""
Hard validation of outputs/manifest.csv. The build fails rather than shipping a bad manifest.

Two layers:

  **Invariants** (sections in docs/DATA_CONTRACT.md). Mechanical assertions about leakage,
  referential integrity, evaluability, schema and files. Any failure is fatal.

  **Leakage audit.** Invariants prove no design_id or dup_group crosses a split, but that
  only guarantees what our identity grouping could SEE. The corpus taught us the hard way
  that identity can hide from a weak method: pHash missed 27 Drive designs entirely. So this
  independently asks, for every test image, which train image is its nearest neighbour in
  embedding space, and surfaces the closest cases for human review. It is a detector for the
  failure mode we already know we have, not a formality.

Run:  python -m src.data.validate [--config configs/data.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.schema import (  # noqa: E402
    COLUMNS,
    CONFIDENCES,
    NO_SEED,
    QUERY_KINDS,
    ROLES,
    SPLITS,
)


class Failures:
    """Collects every problem so one run reports all of them, not just the first."""

    def __init__(self) -> None:
        self.items: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        if not condition:
            self.items.append(message)

    def report(self) -> bool:
        if not self.items:
            print("[validate] all invariants hold")
            return True
        print(f"[validate] {len(self.items)} FAILURE(S):")
        for m in self.items:
            print(f"  - {m}")
        return False


def validate_invariants(df: pd.DataFrame, cfg: dict, label_map: dict) -> Failures:
    f = Failures()
    data_root = Path(cfg["paths"]["data_root"])
    real = df[~df["is_synthetic"]]
    synth = df[df["is_synthetic"]]

    # ---- leakage
    for col in ("design_id", "dup_group"):
        spans = df.groupby(col)["split"].nunique()
        bad = spans[spans > 1]
        f.check(bad.empty, f"{len(bad)} {col}(s) appear in more than one split: "
                           f"{list(bad.index[:5])}")

    if not synth.empty:
        parent_split = df.set_index("image_id")["split"]
        mismatched = [r["image_id"] for _, r in synth.iterrows()
                      if r["parent_image_id"] in parent_split.index
                      and parent_split[r["parent_image_id"]] != r["split"]]
        f.check(not mismatched,
                f"{len(mismatched)} synthetic rows sit in a different split from their parent")

    rc = cfg["recolor"]
    train_method = rc["methods"]["train"]
    leaked_method = synth[(synth["split"].isin(["val", "test"]))
                          & (synth["recolor_method"] == train_method)]
    f.check(leaked_method.empty,
            f"{len(leaked_method)} val/test rows use the TRAIN recolor method "
            f"'{train_method}', which would make the held-out eval meaningless")

    # Palette pools must be disjoint across splits.
    pal_by_split = (synth[synth["palette_id"] != ""]
                    .groupby("palette_id")["split"].nunique())
    shared = pal_by_split[pal_by_split > 1]
    f.check(shared.empty, f"{len(shared)} palette_id(s) used in more than one split: "
                          f"{list(shared.index[:5])}")

    # ---- referential integrity
    ids = set(df["image_id"])
    orphans = [p for p in synth["parent_image_id"] if p not in ids]
    f.check(not orphans, f"{len(orphans)} synthetic rows reference a missing parent")

    if not synth.empty:
        parent = df.set_index("image_id")
        for col in ("source", "design_id", "craft_family"):
            bad = [r["image_id"] for _, r in synth.iterrows()
                   if r["parent_image_id"] in parent.index
                   and parent.loc[r["parent_image_id"], col] != r[col]]
            f.check(not bad, f"{len(bad)} synthetic rows disagree with their parent on {col}")

    f.check(df["image_id"].is_unique, "image_id is not unique")
    f.check(df["path"].is_unique, "path is not unique")

    rep_counts = real.groupby("dup_group")["is_representative"].sum()
    bad_reps = rep_counts[rep_counts != 1]
    f.check(bad_reps.empty,
            f"{len(bad_reps)} dup_group(s) do not have exactly one representative")

    # ---- evaluability
    for split in ("val", "test"):
        sub = df[df["split"] == split]
        if sub.empty:
            f.check(False, f"split '{split}' is empty")
            continue
        per_design = sub.groupby("design_id")["role"].agg(
            gallery=lambda s: (s == "gallery").sum(), query=lambda s: (s == "query").sum())
        no_gal = per_design[per_design["gallery"] < 1]
        no_q = per_design[per_design["query"] < 1]
        f.check(no_gal.empty, f"{len(no_gal)} {split} designs have no gallery row")
        f.check(no_q.empty, f"{len(no_q)} {split} designs have no query row")

    gal = df[df["role"] == "gallery"]
    f.check(not gal["is_synthetic"].any(),
            f"{int(gal['is_synthetic'].sum())} gallery rows are synthetic; a synthetic "
            f"gallery would measure our generator rather than the designs")
    f.check(bool(gal["is_representative"].all()),
            f"{int((~gal['is_representative']).sum())} gallery rows are not representatives")

    non_rep_eval = df[(~df["is_representative"]) & (~df["is_synthetic"])
                      & df["role"].isin(["gallery", "query"])]
    f.check(non_rep_eval.empty,
            f"{len(non_rep_eval)} non-representative real rows are used for evaluation")

    train_designs = set(df.loc[df["split"] == "train", "design_id"])
    f.check(set(label_map) == train_designs,
            f"label_map covers {len(label_map)} designs but train has {len(train_designs)}")

    q = df[df["role"] == "query"]
    f.check(bool((q["query_kind"] != "").all()),
            f"{int((q['query_kind'] == '').sum())} query rows have no query_kind")
    nonq = df[df["role"] != "query"]
    f.check(bool((nonq["query_kind"] == "").all()),
            f"{int((nonq['query_kind'] != '').sum())} non-query rows carry a query_kind")

    # ---- schema and enums
    f.check(list(df.columns) == COLUMNS,
            f"column order differs from the contract: {list(df.columns)}")
    for col, allowed in (("split", SPLITS), ("role", ROLES),
                         ("query_kind", QUERY_KINDS), ("label_confidence", CONFIDENCES)):
        bad = sorted(set(df[col].dropna().unique()) - set(allowed))
        f.check(not bad, f"column '{col}' has values outside the contract: {bad}")

    # ---- synthesis provenance consistency
    f.check(bool((real["recolor_seed"] == NO_SEED).all()),
            f"{int((real['recolor_seed'] != NO_SEED).sum())} real rows carry a recolor_seed")
    f.check(bool((real["recolor_method"] == "").all()),
            f"{int((real['recolor_method'] != '').sum())} real rows carry a recolor_method")
    if not synth.empty:
        f.check(bool((synth["recolor_seed"] >= 0).all()),
                "some synthetic rows have no recolor_seed")
        f.check(bool((synth["recolor_method"] != "").all()),
                "some synthetic rows have no recolor_method")
        f.check(bool((synth["parent_image_id"] != "").all()),
                "some synthetic rows have no parent_image_id")

    # ---- files exist and open
    missing = []
    for p in df["path"]:
        if not (data_root / p).exists():
            missing.append(p)
            if len(missing) > 20:
                break
    f.check(not missing, f"{len(missing)}+ manifest paths do not exist, e.g. {missing[:3]}")

    return f


def leakage_audit(df: pd.DataFrame, cfg: dict, top_n: int = 20) -> pd.DataFrame | None:
    """For every test image, its nearest TRAIN image by embedding cosine.

    Independent of the grouping that produced the splits, so it can catch same-design pairs
    our verifier missed. Uses zero-shot DINOv2 when torch is available and falls back to
    translation-invariant histogram descriptors otherwise.
    """
    from src.data.geometric_id import dinov2_embeddings, histogram_descriptors

    data_root = Path(cfg["paths"]["data_root"])
    real = df[~df["is_synthetic"]]
    test = real[real["split"] == "test"]
    train = real[real["split"] == "train"]
    if test.empty or train.empty:
        print("[validate] leakage audit skipped: test or train is empty")
        return None

    test_paths = [data_root / p for p in test["path"]]
    train_paths = [data_root / p for p in train["path"]]
    try:
        emb_test = dinov2_embeddings(test_paths)
        emb_train = dinov2_embeddings(train_paths)
        method = "zero-shot DINOv2 cosine"
    except Exception as exc:
        print(f"[validate] DINOv2 unavailable ({type(exc).__name__}), using histogram "
              f"descriptors for the leakage audit")
        emb_test = histogram_descriptors(test_paths)
        emb_train = histogram_descriptors(train_paths)
        method = "histogram descriptor cosine (DINOv2 unavailable)"

    sims = emb_test @ emb_train.T
    best = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)

    audit = pd.DataFrame({
        "test_path": test["path"].values,
        "test_design": test["design_id"].values,
        "nearest_train_path": [train.iloc[j]["path"] for j in best],
        "nearest_train_design": [train.iloc[j]["design_id"] for j in best],
        "cosine": np.round(best_sim, 4),
        "method": method,
    }).sort_values("cosine", ascending=False)

    out_path = Path("outputs/leakage_audit.csv")
    audit.to_csv(out_path, index=False)

    qs = np.percentile(best_sim, [50, 90, 99])
    print(f"[validate] leakage audit ({method}) over {len(test)} test images:")
    print(f"           nearest-train cosine: median {qs[0]:.3f}, p90 {qs[1]:.3f}, "
          f"p99 {qs[2]:.3f}, max {best_sim.max():.3f}")
    print(f"           top {top_n} written to {out_path} FOR YOUR REVIEW")
    for _, r in audit.head(min(5, top_n)).iterrows():
        print(f"             {r['cosine']:.3f}  {r['test_path']}  vs  {r['nearest_train_path']}")
    print("           Non-overlapping tiles of one saree cannot be detected by any content "
          "method, so this bounds rather than eliminates the risk. Stated in the README.")
    return audit


def validate_all(config_path: str = "configs/data.yaml", run_audit: bool = True) -> bool:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    df = pd.read_csv(Path(cfg["paths"]["manifest"]), keep_default_na=False)
    for col in ("is_representative", "is_synthetic"):
        if df[col].dtype == object:
            df[col] = df[col].map({"True": True, "False": False, True: True, False: False})
    label_map = json.loads(Path(cfg["paths"]["label_map"]).read_text(encoding="utf-8"))["label_map"]

    f = validate_invariants(df, cfg, label_map)
    ok = f.report()

    if run_audit:
        try:
            leakage_audit(df, cfg)
        except Exception as exc:
            print(f"[validate] leakage audit could not run: {type(exc).__name__}: {exc}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--no-audit", action="store_true")
    args = ap.parse_args()
    return 0 if validate_all(args.config, run_audit=not args.no_audit) else 1


if __name__ == "__main__":
    raise SystemExit(main())
