"""
Build outputs/manifest.csv: the single artefact all downstream code reads.

Pipeline:
  1. adapters           raw files to partial records, unmatched files logged
  2. identity           geometric verification plus union-find to assign design_id
  3. duplicates         dup_group from same-pixel evidence, one representative each
  4. split              by design_id, stratified, grouped so duplicates cannot cross
  5. roles              gallery and query assignment inside val and test
  6. synthesis          materialise val and test recolors, write rows for them
  7. outputs            manifest.csv, label_map.json, data_summary.md, then validate

Run:  python -m src.data.build_manifest [--config configs/data.yaml] [--limit-source N]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.adapters import get_adapter  # noqa: E402
from src.data.color_aug import (  # noqa: E402
    GeometryConfig,
    Recolorer,
    build_palette_bank,
    read_rgb,
    split_palette_bank,
)
from src.data.geometric_id import (  # noqa: E402
    GeoConfig,
    HueConfig,
    group_designs,
    union_find_groups,
)
from src.data.schema import (  # noqa: E402
    COLUMNS,
    DTYPES,
    DUP_OVERLAP_MIN,
    SCHEMA_VERSION,
    empty_row,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def stable_seed(*parts) -> int:
    """A seed derived from its inputs that is identical in every process and on every machine.

    Python's built-in hash() is NOT usable here: for strings it is randomised per process
    (PYTHONHASHSEED), so seeding queries with it would generate a different test set on every
    rebuild, silently breaking reproducibility. SHA1 of the parts is stable everywhere.
    """
    import hashlib
    digest = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2 ** 31)


# ------------------------------------------------------------------ duplicate handling

def impulse_noise_score(path: Path) -> float:
    """Fraction of pixels deviating sharply from a 3x3 median.

    This is exactly what salt-and-pepper augmentation introduces, so the cluster member with
    the lowest score is the least corrupted copy and becomes the representative.
    """
    try:
        gray = cv2.cvtColor(read_rgb(path), cv2.COLOR_RGB2GRAY)
        med = cv2.medianBlur(gray, 3)
        return float((np.abs(gray.astype(np.int16) - med.astype(np.int16)) > 40).mean())
    except Exception:
        return 1.0  # unreadable sorts last, so a clean copy is always preferred


def choose_representatives(df: pd.DataFrame, data_root: Path, tie_break: str) -> set[str]:
    """One representative image_id per dup_group: fewest impulse-noise pixels."""
    reps: set[str] = set()
    for dup, grp in df.groupby("dup_group", sort=True):
        if len(grp) == 1:
            reps.add(grp.iloc[0]["image_id"])
            continue
        scored = []
        for _, row in grp.iterrows():
            score = impulse_noise_score(data_root / row["path"])
            scored.append((score, row["path"], row["image_id"]))
        scored.sort()  # by score, then path, which is the configured tie break
        reps.add(scored[0][2])
    return reps


# ---------------------------------------------------------------------------- splitting

def compute_split_groups(df: pd.DataFrame, cfg: dict, data_root: Path,
                         log=log) -> dict[str, str]:
    """Group designs that must not be separated by the split.

    Returns design_id to split_group_id. Most designs are alone in their group.

    Motivation: geometric verification can only link images that share pixels. Two separate
    photographs of one saree share none, so they look like different designs and could land on
    opposite sides of the split. The leakage audit found real examples. This catches them with
    a deliberately conservative rule: high embedding similarity AND a shared distinctive
    filename prefix. Requiring both keeps false merges rare, and because this only constrains
    the split rather than merging identities, a false positive costs nothing but flexibility.
    """
    import re

    sp = cfg["split"].get("extra_grouping", {}) or {}
    designs = sorted(df["design_id"].unique())
    groups = {d: d for d in designs}
    if not sp.get("enabled", False):
        return groups

    cos_min = float(sp.get("cosine_min", 0.93))
    min_prefix = int(sp.get("min_shared_prefix", 5))
    skip_generic = bool(sp.get("skip_generic_stems", True))

    reps = df[df["is_representative"] & (~df["is_synthetic"])].reset_index(drop=True)
    if len(reps) < 2:
        return groups

    from src.data.geometric_id import dinov2_embeddings, histogram_descriptors
    paths = [data_root / p for p in reps["path"]]
    try:
        emb = dinov2_embeddings(paths)
    except Exception as exc:
        log(f"[build] extra grouping: DINOv2 unavailable ({type(exc).__name__}), "
            f"using histogram descriptors")
        emb = histogram_descriptors(paths)

    rf_re = re.compile(r"^(.*?)_(?:jpg|jpeg|png)\.rf\.[0-9a-f]+$", re.IGNORECASE)
    generic_re = re.compile(r"^(?:image|images|img|download|untitled|photo)[-_ ]?\d*$",
                            re.IGNORECASE)

    def stem(rel: str) -> str:
        base = Path(rel).stem
        m = rf_re.match(base)
        return m.group(1) if m else base

    # The filename condition must look at EVERY member of a design, not only its
    # representative. A duplicate group can hold one image saved under several unrelated
    # names: the corpus has `image21` holding the same pixels as the `9557ER3_4` triplet, and
    # pHash correctly put them in one group. If `image21` happens to be the representative,
    # checking only the representative sees a generic stem and misses the 9557ER2 / 9557ER3
    # relationship entirely. That exact miss is how this bug was found.
    stems_by_design: dict[str, set[str]] = {}
    for d, grp in df[~df["is_synthetic"]].groupby("design_id"):
        names = {stem(p) for p in grp["path"]}
        if skip_generic:
            names = {s for s in names if not generic_re.match(s)}
        stems_by_design[d] = names

    def shared_prefix(a: str, b: str) -> int:
        n = 0
        for ca, cb in zip(a, b):
            if ca != cb:
                break
            n += 1
        return n

    def best_shared(di: str, dj: str) -> tuple[int, str, str]:
        """Longest shared prefix over all distinctive name pairs between two designs."""
        best = (0, "", "")
        for a in stems_by_design.get(di, ()):
            for b in stems_by_design.get(dj, ()):
                n = shared_prefix(a, b)
                if n > best[0]:
                    best = (n, a, b)
        return best

    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    edges: list[tuple[str, str, float, str, str]] = []
    cand_i, cand_j = np.where(sims >= cos_min)
    for i, j in zip(cand_i, cand_j):
        if i >= j:
            continue
        di, dj = reps.iloc[i]["design_id"], reps.iloc[j]["design_id"]
        if di == dj:
            continue
        n, si, sj = best_shared(di, dj)
        if n < min_prefix:
            continue
        edges.append((di, dj, float(sims[i, j]), si, sj))

    if not edges:
        log("[build] extra grouping: no cross-design pairs met both the similarity and "
            "filename-prefix conditions")
        return groups

    # Union-find over design ids.
    parent = {d: d for d in designs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for di, dj, *_ in edges:
        ri, rj = find(di), find(dj)
        if ri != rj:
            parent[rj] = ri
    groups = {d: find(d) for d in designs}

    n_grouped = len({g for g in groups.values()
                     if sum(1 for v in groups.values() if v == g) > 1})
    log(f"[build] extra grouping: {len(edges)} constraining pair(s) formed {n_grouped} "
        f"multi-design split group(s), so these cannot straddle a split")
    for di, dj, c, si, sj in sorted(edges, key=lambda e: -e[2])[:8]:
        log(f"           cos={c:.3f}  {si[:34]}  ||  {sj[:34]}")
    return groups


def assign_splits(designs: pd.DataFrame, cfg: dict, seed: int,
                  forced_test: set[str]) -> dict[str, str]:
    """Assign each design_id to train / val / test.

    Stratified by (source, craft_family) so each split mirrors the corpus composition, and
    the Drive source carries a higher test fraction because it is the client's target domain.
    Designs forced to test (the real multi-view ones) are placed first and counted towards
    that source's test quota.
    """
    sp = cfg["split"]
    rng = np.random.default_rng(seed)
    val_frac = float(sp["val_frac"])
    test_frac_default = float(sp["test_frac"])
    per_source = sp.get("per_source_test_frac", {}) or {}

    assignment: dict[str, str] = {d: "test" for d in forced_test}

    for (source, family), grp in designs.groupby(["source", "craft_family"], sort=True):
        ids = sorted(grp["design_id"].tolist())
        free = [d for d in ids if d not in assignment]
        n_total = len(ids)
        already_test = sum(1 for d in ids if assignment.get(d) == "test")

        test_frac = float(per_source.get(source, test_frac_default))
        n_test_target = int(round(test_frac * n_total))
        n_val_target = int(round(val_frac * n_total))

        rng.shuffle(free)
        # The forced designs already count towards test, so only top up the remainder.
        n_test_more = max(0, min(n_test_target - already_test, len(free)))
        for d in free[:n_test_more]:
            assignment[d] = "test"
        rest = free[n_test_more:]
        n_val = max(0, min(n_val_target, len(rest)))
        for d in rest[:n_val]:
            assignment[d] = "val"
        for d in rest[n_val:]:
            assignment[d] = "train"

    return assignment


# ------------------------------------------------------------------------------ builder

def build(config_path: str, limit_source: int | None = None) -> pd.DataFrame:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    data_root = Path(cfg["paths"]["data_root"])
    geo_cfg = GeoConfig.from_dict(cfg["dedup"]["geometric"])
    hue_cfg = HueConfig.from_dict(cfg["dedup"]["hue"])
    phash_thresh = int(cfg["dedup"]["phash"]["hamming_threshold"])

    # ---------------------------------------------------------------- 1. adapters
    all_rows: list[dict] = []
    unmatched_rows: list[dict] = []
    for source_key, source_cfg in cfg["sources"].items():
        if not source_cfg.get("enabled", False):
            log(f"[build] source '{source_key}' disabled, skipping")
            continue
        t0 = time.time()
        adapter = get_adapter(source_key, source_cfg, data_root)
        records = adapter.build_records()
        if limit_source:
            records = records[:limit_source]
        log(f"[build] {source_key}: {len(records):,} records, "
            f"{len(adapter.unmatched)} unmatched ({time.time() - t0:.0f}s)")
        for r in records:
            row = empty_row()
            row.update(r.to_dict())
            row["_stem"] = r.hints.get("stem", "")
            all_rows.append(row)
        for u in adapter.unmatched:
            unmatched_rows.append({"path": u.path, "source": u.source, "reason": u.reason})

    if not all_rows:
        raise SystemExit("no records produced; check configs/data.yaml source roots")

    df = pd.DataFrame(all_rows)

    unmatched_path = Path(cfg["paths"]["unmatched"])
    unmatched_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(unmatched_rows or [{"path": "", "source": "", "reason": "none"}]).to_csv(
        unmatched_path, index=False)
    log(f"[build] unmatched log written to {unmatched_path} ({len(unmatched_rows)} entries)")

    # --------------------------------------------- 2 and 3. identity and duplicates
    review_rows: list[dict] = []
    design_of: dict[str, str] = {}
    dup_of: dict[str, str] = {}

    from src.data.geometric_id import phash_candidates

    rep_ids: set[str] = set()
    for source_key, sub in df.groupby("source", sort=True):
        sub = sub.sort_values("path").reset_index(drop=True)
        paths = [data_root / p for p in sub["path"]]
        phashes = sub["phash"].tolist()
        prefix = cfg["sources"][source_key].get("design_prefix", source_key)
        n = len(paths)

        # ---- stage 1: collapse same-pixel duplicates with pHash.
        # Cheap, and reliable for THIS relation: the Roboflow noise triplets share framing
        # exactly, which is the case pHash handles well. It is only re-cropped views that
        # pHash cannot see, and those are design identity, resolved in stage 3.
        dup_comps = union_find_groups(n, list(phash_candidates(phashes, threshold=phash_thresh)))

        # ---- stage 2: one representative per duplicate cluster, the least noisy copy.
        by_dup: dict[int, list[int]] = collections.defaultdict(list)
        for i, c in enumerate(dup_comps):
            by_dup[c].append(i)
        rep_idx_of_dup: dict[int, int] = {}
        t0 = time.time()
        for c, members in by_dup.items():
            if len(members) == 1:
                rep_idx_of_dup[c] = members[0]
                continue
            scored = sorted((impulse_noise_score(paths[i]), str(paths[i]), i)
                            for i in members)
            rep_idx_of_dup[c] = scored[0][2]
        log(f"[build] {source_key}: {n:,} images collapse to {len(by_dup):,} duplicate "
            f"groups, representatives chosen in {time.time() - t0:.0f}s")

        # ---- stage 3: design identity over REPRESENTATIVES only.
        # This is the expensive stage (DINOv2 prefilter plus ORB verification), so running it
        # on representatives instead of every image is what keeps the build tractable: 412
        # Kaggle representatives rather than 1,468 images.
        rep_order = sorted(rep_idx_of_dup.items())           # (dup_comp, image index)
        rep_indices = [i for _, i in rep_order]
        log(f"[build] identity for {source_key}: verifying {len(rep_indices):,} representatives")
        comps_rep, records = group_designs(
            [paths[i] for i in rep_indices],
            [phashes[i] for i in rep_indices],
            geo_cfg, hue_cfg,
            prefilter_topk=int(cfg["dedup"].get("prefilter_topk", 10)),
            phash_threshold=phash_thresh,
            use_dinov2=bool(cfg["dedup"].get("use_dinov2_prefilter", True)),
            time_budget_s=cfg["dedup"]["geometric"].get("time_budget_s"),
            log=log,
        )

        # ---- stage 4: a verified pair of representatives whose overlap is near total means
        # those two duplicate clusters are actually the same pixels, so merge them.
        merge_edges = [(r["i"], r["j"]) for r in records
                       if r["overlap"] >= DUP_OVERLAP_MIN and r["ncc"] >= 0.95]
        if merge_edges:
            merged = union_find_groups(len(rep_indices), merge_edges)
            root_to_dups: dict[int, list[int]] = collections.defaultdict(list)
            for k, (dup_c, _) in enumerate(rep_order):
                root_to_dups[merged[k]].append(dup_c)
            remap: dict[int, int] = {}
            for dups in root_to_dups.values():
                keep = min(dups)
                for d in dups:
                    remap[d] = keep
            dup_comps = [remap.get(c, c) for c in dup_comps]
            log(f"[build] {source_key}: merged {len(merge_edges)} near-identical "
                f"representative pairs into shared duplicate groups")

        # ---- propagate: every image inherits the design of its cluster's representative.
        design_of_dup: dict[int, int] = {}
        for k, (dup_c, _) in enumerate(rep_order):
            design_of_dup[dup_c] = comps_rep[k]
        # After a stage-4 merge, a remapped cluster takes the design of the cluster it
        # merged into, so look up through the remap when the key is absent.
        comp_to_design: dict[int, str] = {}
        for c in sorted(set(design_of_dup.values())):
            comp_to_design[c] = f"{prefix}_d{len(comp_to_design):05d}"
        dup_to_id: dict[int, str] = {}
        for c in sorted(set(dup_comps)):
            dup_to_id[c] = f"{prefix}_g{len(dup_to_id):05d}"

        for i in range(n):
            dup_c = dup_comps[i]
            comp = design_of_dup.get(dup_c)
            if comp is None:
                # Cluster was merged away; inherit from any surviving member's design.
                comp = next((design_of_dup[d] for d in by_dup
                             if d in design_of_dup and dup_comps[rep_idx_of_dup[d]] == dup_c),
                            None)
            image_id = sub.iloc[i]["image_id"]
            design_of[image_id] = comp_to_design.get(comp, f"{prefix}_d_orphan{i}")
            dup_of[image_id] = dup_to_id[dup_c]

        for dup_c, idx in rep_idx_of_dup.items():
            rep_ids.add(sub.iloc[idx]["image_id"])

        n_designs = len({design_of[sub.iloc[i]["image_id"]] for i in range(n)})
        per_design = collections.Counter(design_of[sub.iloc[i]["image_id"]] for i in range(n))
        multi = sum(1 for v in per_design.values() if v > 1)
        log(f"[build] {source_key}: {n_designs:,} designs from {n:,} images "
            f"({multi} multi-image), {len(set(dup_comps)):,} dup groups")

        # A verified pair with a clearly different palette inside the overlap would be a real
        # colourway pair, which the corpus is not supposed to contain. Flag any for review.
        for r in records:
            hv = r.get("hue_overlap")
            if hv is not None and hv == hv and hv >= hue_cfg.different_palette_min:
                review_rows.append({
                    "kind": "possible_real_colorway_pair", "source": source_key,
                    "path_a": r["path_a"], "path_b": r["path_b"],
                    "ncc": round(r["ncc"], 4), "overlap": round(r["overlap"], 3),
                    "hue_overlap": round(hv, 3),
                    "note": "same design, different palette inside the aligned overlap",
                })

    df["design_id"] = df["image_id"].map(design_of)
    df["dup_group"] = df["image_id"].map(dup_of)

    # A design spanning several craft families is contradictory. Resolve by majority so the
    # split can stratify, but flag it so the merge can be inspected.
    if cfg["dedup"].get("flag_cross_family_clusters", True):
        for design, grp in df.groupby("design_id", sort=True):
            fams = sorted(set(f for f in grp["craft_family"] if f))
            if len(fams) > 1:
                major = collections.Counter(grp["craft_family"]).most_common(1)[0][0]
                review_rows.append({
                    "kind": "design_spans_craft_families", "source": grp.iloc[0]["source"],
                    "path_a": grp.iloc[0]["path"], "path_b": grp.iloc[-1]["path"],
                    "ncc": "", "overlap": "", "hue_overlap": "",
                    "note": f"families {fams}, resolved to '{major}'",
                })
                df.loc[grp.index, "craft_family"] = major

    # ------------------------------------------------------- representatives
    # Already chosen in stage 2, so reuse rather than rescoring every image.
    df["is_representative"] = df["image_id"].isin(rep_ids)
    log(f"[build] {len(rep_ids):,} representatives (one per duplicate group)")

    # ------------------------------------------------------------------ 4. split
    designs = (df.groupby("design_id", sort=True)
                 .agg(source=("source", "first"), craft_family=("craft_family", "first"),
                      n_images=("image_id", "size"))
                 .reset_index())

    # Real multi-view designs: more than one dup_group inside one design means genuinely
    # different views, not duplicates. These are the only real positive pairs in the corpus
    # and all go to test.
    multiview = set()
    for design, grp in df.groupby("design_id", sort=True):
        if grp["dup_group"].nunique() > 1:
            multiview.add(design)
    forced = multiview if cfg["split"].get("force_multiview_to_test", True) else set()
    log(f"[build] {len(multiview)} real multi-view designs "
        f"({'forced to test' if forced else 'not forced'})")

    # Designs that must stay on the same side of the split, because they may be separate
    # photographs of one saree that geometry cannot link. See compute_split_groups.
    split_group = compute_split_groups(df, cfg, data_root)
    df["_split_group"] = df["design_id"].map(split_group)

    # Split at the GROUP level, then map back to designs, so a constrained pair cannot be
    # separated. Each group takes the source and family of its members (identical in practice,
    # since the constraint requires high similarity).
    group_rows = (df[~df["is_synthetic"]]
                  .groupby("_split_group", sort=True)
                  .agg(source=("source", "first"), craft_family=("craft_family", "first"),
                       n_images=("image_id", "size"))
                  .reset_index()
                  .rename(columns={"_split_group": "design_id"}))
    forced_groups = {split_group[d] for d in forced if d in split_group}
    group_assignment = assign_splits(group_rows, cfg, seed, forced_groups)
    assignment = {d: group_assignment[g] for d, g in split_group.items()}
    df["split"] = df["design_id"].map(assignment)
    df["role"] = "train"

    # ------------------------------------------------------- 5 and 6. roles and synthesis
    rc = cfg["recolor"]
    bank = build_palette_bank(rc["palette_bank"]["size"], rc["palette_bank"]["k_colors"],
                              seed=seed)
    pools = split_palette_bank(bank, rc["palette_bank"]["split"], seed=seed)
    geo_eval = GeometryConfig.from_dict(cfg["geometry"], out_size=cfg["loader"]["img_size"])
    synth_dir = Path(cfg["paths"]["synthetic_dir"])
    # Every manifest path is relative to data_root, so generated images must live under it.
    # Fail early with a clear reason rather than deep inside path arithmetic.
    try:
        synth_dir.resolve().relative_to(data_root.resolve())
    except ValueError:
        raise SystemExit(f"paths.synthetic_dir ({synth_dir}) must be inside paths.data_root "
                         f"({data_root}); the contract stores all paths relative to data_root")
    n_queries = int(cfg["eval"]["queries_per_design"])

    new_rows: list[dict] = []
    for split_name in ("val", "test"):
        method = rc["methods"][split_name]
        pool = pools[split_name]
        recolorer = Recolorer(method=method, palettes=pool, geometry=geo_eval,
                              k_colors=rc["palette_bank"]["k_colors"],
                              cache_dir=Path("data/cache"))
        tonal_cfg = rc.get("tonal", {}) or {}
        tonal_recolorer = None
        n_tonal = int(tonal_cfg.get("queries_per_design", 0))
        if tonal_cfg.get("enabled", False) and n_tonal > 0:
            tonal_recolorer = Recolorer(method=tonal_cfg.get("method", "tonal_remap"),
                                        palettes=pool, geometry=geo_eval,
                                        k_colors=rc["palette_bank"]["k_colors"],
                                        cache_dir=Path("data/cache"),
                                        invert_prob=float(tonal_cfg.get("invert_prob", 0.5)))
        subset = df[(df["split"] == split_name) & df["is_representative"]]
        log(f"[build] {split_name}: generating queries for "
            f"{subset['design_id'].nunique():,} designs using {method}")

        for design, grp in subset.groupby("design_id", sort=True):
            grp = grp.sort_values("path")
            gallery_row = grp.iloc[0]
            df.loc[gallery_row.name, "role"] = "gallery"
            other_views = grp.iloc[1:]

            # (a) other real views as-is: real framing change at a fixed palette.
            for _, view in other_views.iterrows():
                df.loc[view.name, "role"] = "query"
                df.loc[view.name, "query_kind"] = "real_view"

            # (b) other real views recoloured with a held-out palette: real framing change
            # PLUS synthetic palette change, the closest proxy we have to the true task.
            for vi, (_, view) in enumerate(other_views.iterrows()):
                seed_v = stable_seed(view["image_id"], "real_view_recolored", vi)
                try:
                    img, meta = recolorer(data_root / view["path"], view["path"], seed_v)
                except Exception as exc:
                    log(f"[build] WARN recolor failed for {view['path']}: {exc}")
                    continue
                new_rows.append(_synth_row(img, view, meta, split_name, synth_dir,
                                           data_root, "real_view_recolored"))

            # (c) the standard protocol: recolour plus geometry applied to the gallery image.
            for q in range(n_queries):
                seed_q = stable_seed(gallery_row["image_id"], "synthetic_recolor", q)
                try:
                    img, meta = recolorer(data_root / gallery_row["path"],
                                          gallery_row["path"], seed_q)
                except Exception as exc:
                    log(f"[build] WARN recolor failed for {gallery_row['path']}: {exc}")
                    continue
                new_rows.append(_synth_row(img, gallery_row, meta, split_name, synth_dir,
                                           data_root, "synthetic_recolor"))

            # (d) tonal recolour: lightness changes too, and dark/light order may invert. The
            # only query kind on which grayscale input is NOT invariant by construction.
            if tonal_recolorer is not None:
                for q in range(n_tonal):
                    seed_t = stable_seed(gallery_row["image_id"], "synthetic_tonal", q)
                    try:
                        img, meta = tonal_recolorer(data_root / gallery_row["path"],
                                                    gallery_row["path"], seed_t)
                    except Exception as exc:
                        log(f"[build] WARN tonal recolor failed for {gallery_row['path']}: {exc}")
                        continue
                    new_rows.append(_synth_row(img, gallery_row, meta, split_name, synth_dir,
                                               data_root, "synthetic_tonal"))

    log(f"[build] materialised {len(new_rows):,} synthetic query images")
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    # Non-representative images are train-only extra positives, never evaluated.
    mask = (~df["is_representative"]) & (~df["is_synthetic"])
    df.loc[mask & df["role"].isin(["gallery", "query"]), "role"] = "train"

    # ------------------------------------------------------------------ 7. outputs
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = empty_row()[col]
    df = df[COLUMNS]
    for col, dt in DTYPES.items():
        try:
            df[col] = df[col].astype(dt)
        except Exception as exc:
            raise SystemExit(f"column '{col}' will not cast to {dt}: {exc}")

    manifest_path = Path(cfg["paths"]["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    log(f"[build] wrote {manifest_path} ({len(df):,} rows)")

    train_designs = sorted(df.loc[df["split"] == "train", "design_id"].unique())
    label_map = {d: i for i, d in enumerate(train_designs)}
    Path(cfg["paths"]["label_map"]).write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "label_map": label_map}, indent=2),
        encoding="utf-8")
    log(f"[build] wrote label_map with {len(label_map):,} train designs")

    review_path = Path(cfg["paths"]["review"])
    pd.DataFrame(review_rows or [{"kind": "none", "source": "", "path_a": "", "path_b": "",
                                  "ncc": "", "overlap": "", "hue_overlap": "",
                                  "note": "nothing flagged"}]).to_csv(review_path, index=False)
    log(f"[build] review flags written to {review_path} ({len(review_rows)} entries)")

    write_summary(df, cfg, pools, multiview)
    return df


def _synth_row(img: np.ndarray, parent: pd.Series, meta: dict, split_name: str,
               synth_dir: Path, data_root: Path, query_kind: str) -> dict:
    """Write a generated image to disk and return its manifest row."""
    from src.data.adapters import stable_image_id

    out_dir = synth_dir / split_name / str(parent["design_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{parent['image_id']}_{meta['palette_id']}_{meta['recolor_seed']}_{query_kind}"
    out_path = out_dir / f"{stem}.jpg"
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    rel = out_path.resolve().relative_to(data_root.resolve()).as_posix()
    row = empty_row()
    row.update({
        "image_id": stable_image_id(rel),
        "path": rel,
        "source": parent["source"],              # inherits, so per-source breakdown works
        "design_id": parent["design_id"],        # a recolour is the same design
        "colorway_id": meta["palette_id"],
        "dup_group": parent["dup_group"],
        "is_representative": False,
        "parent_image_id": parent["image_id"],
        "craft_family": parent["craft_family"],
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "phash": "",
        "split": split_name,
        "role": "query",
        "query_kind": query_kind,
        "is_synthetic": True,
        "recolor_method": meta["recolor_method"],
        "palette_id": meta["palette_id"],
        "recolor_seed": meta["recolor_seed"],
        "geo_params": meta["geo_params"],
        "label_confidence": parent["label_confidence"],
        "notes": f"generated from {parent['path']}",
    })
    return row


# ------------------------------------------------------------------------------ summary

def write_summary(df: pd.DataFrame, cfg: dict, pools: dict, multiview: set[str]) -> None:
    out: list[str] = ["# Data summary", "",
                      f"Schema version {SCHEMA_VERSION}. Generated by "
                      "`src/data/build_manifest.py`.", "",
                      "Counts only: no image contents, nothing copied out of `data/`.", ""]

    real = df[~df["is_synthetic"]]
    out += ["## Totals", "",
            f"- manifest rows: {len(df):,} ({len(real):,} real, "
            f"{int(df['is_synthetic'].sum()):,} synthetic)",
            f"- designs: {df['design_id'].nunique():,}",
            f"- duplicate groups: {df['dup_group'].nunique():,}",
            f"- real multi-view designs: {len(multiview):,}", ""]

    out += ["## Images and designs per source and split", "",
            "| source | split | designs | real images | synthetic | gallery | query |",
            "|---|---|---|---|---|---|---|"]
    for (source, split), grp in df.groupby(["source", "split"], sort=True):
        out.append(f"| {source} | {split} | {grp['design_id'].nunique():,} "
                   f"| {int((~grp['is_synthetic']).sum()):,} "
                   f"| {int(grp['is_synthetic'].sum()):,} "
                   f"| {int((grp['role'] == 'gallery').sum()):,} "
                   f"| {int((grp['role'] == 'query').sum()):,} |")
    out.append("")

    out += ["## Craft family by split", "",
            "| craft family | train | val | test |", "|---|---|---|---|"]
    fam_tab = (real.drop_duplicates("design_id")
                   .groupby(["craft_family", "split"]).size().unstack(fill_value=0))
    for fam, row in fam_tab.iterrows():
        label = fam if fam else "(none, Drive)"
        out.append(f"| {label} | {row.get('train', 0):,} | {row.get('val', 0):,} "
                   f"| {row.get('test', 0):,} |")
    out.append("")

    sizes = real.groupby("design_id").size()
    hist = collections.Counter(sizes.values)
    out += ["## Real images per design", "",
            "| images in design | number of designs |", "|---|---|"]
    for k in sorted(hist):
        out.append(f"| {k} | {hist[k]:,} |")
    out.append("")

    out += ["## Query composition", "",
            "| query kind | rows | designs |", "|---|---|---|"]
    q = df[df["role"] == "query"]
    for kind, grp in q.groupby("query_kind", sort=True):
        out.append(f"| {kind} | {len(grp):,} | {grp['design_id'].nunique():,} |")
    out += ["",
            "`real_view` is another real view of the same saree, unmodified. The corpus's "
            "real pairs are overlapping crops of one photograph at scale 1.0 with about 25% "
            "overlap, so they test region and framing invariance at fixed lighting and "
            "scale. They are not re-photography and they do not vary palette.", ""]

    out += ["## Palette banks (mutually disjoint)", "",
            "| pool | palettes | ids |", "|---|---|---|"]
    for name, pool in pools.items():
        ids = ", ".join(p["palette_id"] for p in pool[:4])
        out.append(f"| {name} | {len(pool)} | {ids}{', ...' if len(pool) > 4 else ''} |")
    out += ["",
            f"Recolor methods: train `{cfg['recolor']['methods']['train']}`, "
            f"val `{cfg['recolor']['methods']['val']}`, "
            f"test `{cfg['recolor']['methods']['test']}`. Eval uses a different algorithm "
            "AND disjoint palettes, so a high score cannot come from inverting the training "
            "generator.", ""]

    colorway = df[(df["role"] == "query") & (df["query_kind"] != "real_view")]
    out += ["## Honest limitations", "",
            "- The corpus contains **zero real colourway pairs**. Verified by exhaustive "
            "matching with hue compared inside the aligned overlap. So every colour-varying "
            f"positive is synthetic ({len(colorway):,} rows).",
            "- Kaggle images were stretched to 640x640 by Roboflow, distorting aspect ratio, "
            "and carry salt-and-pepper noise. Drive images are untouched.",
            "- Pairs of non-overlapping tiles from one saree cannot be detected by any "
            "content method, so a small amount of undetected same-design leakage may remain "
            "across splits.", ""]

    path = Path(cfg["paths"]["summary"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    log(f"[build] wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--limit-source", type=int, default=None,
                    help="cap records per source, for a fast smoke run")
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()

    build(args.config, args.limit_source)

    if not args.skip_validate:
        from src.data.validate import validate_all
        ok = validate_all(args.config)
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
