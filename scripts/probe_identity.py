"""
Phase 1 / Step 1b: does this corpus actually contain design identity and colourways?

scripts/recon.py reports structure. This script answers the two questions that decide
the whole project:

  Q1  How many UNIQUE pictures are there, once near-duplicates are collapsed?
  Q2  Do real "same design, different palette" pairs exist?

Method for Q1: perceptual hash on grayscale, union-find clustering at several Hamming
thresholds. If the cluster count plateaus across a range of thresholds, that plateau is
the true number of unique pictures and the threshold choice is not load bearing.

Method for Q2: grayscale pHash encodes structure and ignores hue. So a genuine colourway
pair looks like LOW grayscale pHash distance plus a LARGE hue-histogram difference, while
an augmentation duplicate is low on both. We count pairs in each category.

Privacy: metadata and hashes only. No pixels printed, nothing copied out of data/.

Run:  python scripts/probe_identity.py [--data-root data] [--out outputs/identity_probe.md]
"""

from __future__ import annotations

import argparse
import collections
import itertools
import re
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Roboflow exports encode the original filename plus an export hash:
#   <original stem>_<ext>.rf.<32 hex>.jpg
ROBOFLOW_RE = re.compile(r"^(.*?)_(jpg|jpeg|png)\.rf\.[0-9a-f]+$", re.IGNORECASE)

# Stems so generic that different collectors reuse them for unrelated pictures.
GENERIC_STEM_RE = re.compile(r"^(image|images|img|download|untitled|photo)[-_ ]?\d*$", re.IGNORECASE)

DUP_THRESHOLDS = (0, 2, 4, 6, 8)
STRUCT_MAX = 12    # grayscale pHash distance below which two images are "structurally close"
HIST_DUP_MAX = 0.15   # hue-histogram L1 below which the palette is effectively identical
HIST_DIFF_MIN = 0.45  # hue-histogram L1 above which the palette is clearly different


def source_stem(path: Path) -> str:
    """The original filename before any export suffix."""
    m = ROBOFLOW_RE.match(path.stem)
    return m.group(1) if m else path.stem


def collect(data_root: Path) -> list[Path]:
    return sorted(p for p in data_root.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                  and not any(part.startswith(".") for part in p.parts))


def features(path: Path) -> tuple[imagehash.ImageHash, np.ndarray]:
    """Grayscale pHash (structure) and a coarse hue histogram (palette)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        gray_hash = imagehash.phash(im.convert("L"))
        hsv = np.asarray(im.resize((64, 64)).convert("HSV"), dtype=np.float32)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Near-grey and near-black pixels have meaningless hue, so they are excluded.
    mask = (sat > 40) & (val > 30)
    hist = np.histogram(hue[mask], bins=18, range=(0, 256))[0].astype(np.float32)
    return gray_hash, hist / max(hist.sum(), 1.0)


def cluster(paths: list[Path], hashes: dict[Path, imagehash.ImageHash],
            threshold: int) -> dict[Path, list[Path]]:
    """Union-find over pairs within `threshold` Hamming distance.

    Comparing all pairs is quadratic, so candidates are bucketed by several slices of the
    hash string: two hashes within a small distance almost always share at least one slice.
    """
    parent = {p: p for p in paths}

    def find(x: Path) -> Path:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: Path, b: Path) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    buckets: dict[tuple[int, str], list[Path]] = collections.defaultdict(list)
    for p in paths:
        s = str(hashes[p])
        for i in range(0, 16, 4):
            buckets[(i, s[i:i + 4])].append(p)

    for bucket in buckets.values():
        for a, b in itertools.combinations(bucket, 2):
            if hashes[a] - hashes[b] <= threshold:
                union(a, b)

    clusters: dict[Path, list[Path]] = collections.defaultdict(list)
    for p in paths:
        clusters[find(p)].append(p)
    return clusters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="outputs/identity_probe.md")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    paths = collect(data_root)
    if not paths:
        print(f"no images under {data_root}")
        return 1

    out: list[str] = [
        "# Identity and colourway probe (Phase 1, Step 1b)",
        "",
        "Hashes and metadata only: no pixels printed, nothing copied out of `data/`.",
        "",
        f"Images scanned: {len(paths):,}",
        "",
    ]

    gray: dict[Path, imagehash.ImageHash] = {}
    hists: dict[Path, np.ndarray] = {}
    for p in paths:
        try:
            gray[p], hists[p] = features(p)
        except Exception as exc:
            out.append(f"- could not read `{p.relative_to(data_root).as_posix()}`: {type(exc).__name__}")
    usable = [p for p in paths if p in gray]

    # ---- Q1: how many unique pictures
    out += ["## Q1. Unique pictures after near-duplicate collapse", "",
            "| pHash threshold | clusters | multi-image clusters | largest |",
            "|---|---|---|---|"]
    cluster_cache: dict[int, dict[Path, list[Path]]] = {}
    for t in DUP_THRESHOLDS:
        cl = cluster(usable, gray, t)
        cluster_cache[t] = cl
        sizes = [len(v) for v in cl.values()]
        out.append(f"| {t} | {len(cl):,} | {sum(1 for s in sizes if s > 1):,} | {max(sizes)} |")
    out += ["",
            "A plateau across thresholds means the duplicate structure is unambiguous and "
            "the threshold is not a load-bearing choice.",
            ""]

    # Duplicate clusters that straddle any top-level folder: evidence about a provided split.
    chosen = 4 if 4 in cluster_cache else DUP_THRESHOLDS[0]
    cl = cluster_cache[chosen]
    out += [f"At threshold {chosen}:", ""]
    sizes = collections.Counter(len(v) for v in cl.values())
    out.append(f"- cluster size histogram: {dict(sorted(sizes.items()))}")

    def top_level(p: Path) -> str:
        rp = p.relative_to(data_root)
        return rp.parts[0] if len(rp.parts) > 1 else "<root>"

    def second_level(p: Path) -> str:
        rp = p.relative_to(data_root)
        return rp.parts[1] if len(rp.parts) > 2 else "<none>"

    span_top = sum(1 for v in cl.values() if len({top_level(p) for p in v}) > 1)
    span_second = sum(1 for v in cl.values() if len({second_level(p) for p in v}) > 1)
    out += [
        f"- duplicate clusters spanning more than one top-level folder: {span_top:,} "
        f"({100 * span_top / len(cl):.1f}% of clusters)",
        f"- duplicate clusters spanning more than one second-level folder: {span_second:,}",
        "",
        "Clusters spanning top-level folders mean any split provided by the dataset leaks "
        "duplicates across its own boundaries and cannot be reused.",
        "",
    ]

    # ---- filename stems: are they a safe duplicate key?
    by_stem: dict[str, list[Path]] = collections.defaultdict(list)
    for p in usable:
        by_stem[source_stem(p)].append(p)
    generic = {k: v for k, v in by_stem.items() if GENERIC_STEM_RE.match(k)}
    distinctive = {k: v for k, v in by_stem.items() if not GENERIC_STEM_RE.match(k)}

    def stem_spread(groups: dict[str, list[Path]], label: str) -> list[str]:
        d: list[int] = []
        for v in groups.values():
            if len(v) < 2:
                continue
            for a, b in itertools.combinations(v, 2):
                d.append(gray[a] - gray[b])
        if not d:
            return [f"- {label}: no multi-image groups"]
        d.sort()
        return [f"- {label}: {len(d):,} pairs, median distance {d[len(d) // 2]}, "
                f"{sum(x <= 2 for x in d) / len(d):.0%} within distance 2"]

    out += ["## Q1b. Is the filename stem a safe duplicate key?", "",
            f"- distinctive stems: {len(distinctive):,} ({sum(len(v) for v in distinctive.values()):,} images)",
            f"- generic stems (image, images, img, ...): {len(generic):,} "
            f"({sum(len(v) for v in generic.values()):,} images)"]
    out += stem_spread(distinctive, "within distinctive stems")
    out += stem_spread(generic, "within generic stems   ")
    out += ["",
            "If generic stems show a median distance comparable to unrelated images, the stem "
            "is a filename collision rather than an identity, so duplicates must be found by "
            "hash and not by name.",
            ""]

    # ---- Q2: real colourway pairs
    buckets: dict[str, list[Path]] = collections.defaultdict(list)
    for p in usable:
        buckets[str(gray[p])[:4]].append(p)

    close_pairs = 0
    same_palette = 0
    diff_palette: list[tuple[int, float, Path, Path]] = []
    for bucket in buckets.values():
        for a, b in itertools.combinations(bucket, 2):
            d = gray[a] - gray[b]
            if d > STRUCT_MAX:
                continue
            close_pairs += 1
            l1 = float(np.abs(hists[a] - hists[b]).sum())
            if l1 < HIST_DUP_MAX:
                same_palette += 1
            elif l1 >= HIST_DIFF_MIN:
                diff_palette.append((d, l1, a, b))

    # Collapse to unique duplicate-cluster pairs: the 3x export inflates every real pair.
    root_of = {p: r for r, v in cl.items() for p in v}
    unique_cw = {tuple(sorted((str(root_of[a]), str(root_of[b])))) for _, _, a, b in diff_palette}

    out += ["## Q2. Do real colourway pairs exist?", "",
            f"- structurally close pairs (grayscale pHash <= {STRUCT_MAX}): {close_pairs:,}",
            f"- of those, palette effectively identical (hue L1 < {HIST_DUP_MAX}): {same_palette:,} "
            f"({100 * same_palette / max(close_pairs, 1):.0f}%) -> these are duplicates",
            f"- of those, palette clearly different (hue L1 >= {HIST_DIFF_MIN}): {len(diff_palette):,}",
            f"- after collapsing duplicate clusters, distinct candidate colourway pairs: "
            f"{len(unique_cw):,}",
            ""]
    if diff_palette:
        diff_palette.sort(key=lambda t: (t[0], -t[1]))
        out += ["Closest candidates by structure (lower pHash distance is a stronger match):", ""]
        for d, l1, a, b in diff_palette[:8]:
            out.append(f"- pHash distance {d}, hue L1 {l1:.2f}")
            out.append(f"  - `{a.relative_to(data_root).as_posix()}`")
            out.append(f"  - `{b.relative_to(data_root).as_posix()}`")
        out += ["",
                "A candidate only counts as a real colourway pair if its structural distance is "
                "small. Matches near the threshold are usually two unrelated fabric close-ups "
                "that happen to share coarse layout.",
                ""]

    text = "\n".join(out) + "\n"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"[probe] written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
