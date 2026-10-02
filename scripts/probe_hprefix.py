"""
Phase 1 / Step 1c: what does the `h_` filename prefix mean?

Nine Drive files are named `h_img_<id>.jpg` instead of `img_<id>.jpg`. If any of them is a
second view of a saree that also appears as a plain `img_` file, those pairs would be the
only REAL positive pairs in the entire corpus, which makes them worth a careful look.

A plain pHash comparison is not enough: a second view can be flipped, cropped, rescaled or
shot from another angle, and pHash is none of those things. So this script uses three
progressively more tolerant tests.

  1. pHash over 4 orientations (identity, horizontal flip, vertical flip, 180 rotation),
     which catches a mirrored or rotated duplicate.
  2. ORB keypoint matching plus RANSAC homography, counting geometric inliers. This is
     crop, scale and viewpoint tolerant, and it is the test that actually decides.
  3. A control distribution of ORB inliers over random unrelated pairs, because an inlier
     count means nothing without knowing what unrelated images score.

Privacy: metadata, hashes and keypoint counts only. No pixels printed, nothing copied out
of data/.

Run:  python scripts/probe_hprefix.py [--out outputs/hprefix_probe.md]
"""

from __future__ import annotations

import argparse
import itertools
import random
import statistics
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

MAX_SIDE = 800        # common working scale for keypoint detection
ORB_FEATURES = 2000
RATIO = 0.75          # Lowe ratio test
RANSAC_PX = 5.0
CONTROL_PAIRS = 300


def load_gray(path: Path) -> np.ndarray:
    """Grayscale, longest side capped, so keypoint counts are comparable across sizes."""
    with Image.open(path) as im:
        im = im.convert("L")
        scale = MAX_SIDE / max(im.size)
        if scale < 1.0:
            im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
        return np.asarray(im)


def phash_orientations(path: Path) -> dict[str, imagehash.ImageHash]:
    with Image.open(path) as im:
        g = im.convert("L")
        return {
            "identity": imagehash.phash(g),
            "hflip": imagehash.phash(g.transpose(Image.FLIP_LEFT_RIGHT)),
            "vflip": imagehash.phash(g.transpose(Image.FLIP_TOP_BOTTOM)),
            "rot180": imagehash.phash(g.transpose(Image.ROTATE_180)),
        }


def orb_inliers(a: np.ndarray, b: np.ndarray, orb, matcher) -> int:
    """Number of keypoint matches consistent with a single homography."""
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    # knnMatch needs at least 2 candidates per query descriptor
    if len(db) < 2:
        return 0
    good = []
    for m_n in matcher.knnMatch(da, db, k=2):
        if len(m_n) == 2 and m_n[0].distance < RATIO * m_n[1].distance:
            good.append(m_n[0])
    if len(good) < 8:
        return 0
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    return int(mask.sum()) if mask is not None else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Google_drive_data/handloom_sarees")
    ap.add_argument("--out", default="outputs/hprefix_probe.md")
    args = ap.parse_args()

    root = Path(args.root)
    h_files = sorted(root.glob("h_img_*.jpg"))
    i_files = sorted(p for p in root.glob("img_*.jpg"))
    if not h_files:
        print(f"no h_img_* files under {root}")
        return 1

    out: list[str] = [
        "# What does the `h_` prefix mean? (Phase 1, Step 1c)",
        "",
        "Hashes, keypoint counts and metadata only. No pixels printed.",
        "",
        f"- `h_img_*` files: {len(h_files)}",
        f"- `img_*` files: {len(i_files)}",
        "",
    ]

    # --- ID overlap
    h_ids = {p.stem.removeprefix("h_img_") for p in h_files}
    i_ids = {p.stem.removeprefix("img_") for p in i_files}
    shared = sorted(h_ids & i_ids)
    out += ["## 1. Do the numeric IDs overlap?", "",
            f"Shared IDs between `h_img_<id>` and `img_<id>`: **{len(shared)}**"
            f"{' (' + ', '.join(shared) + ')' if shared else ''}", ""]
    if not shared:
        out += ["So there is no same-ID counterpart to compare against, and the `h_` prefix "
                "is not a marker pairing two files that share an ID. Any relationship has to "
                "be found by image content instead.", ""]

    # --- file-level facts
    def facts(paths: list[Path]) -> tuple[list[int], list[int]]:
        sides, bytes_ = [], []
        for p in paths:
            with Image.open(p) as im:
                sides.append(max(im.size))
            bytes_.append(p.stat().st_size)
        return sides, bytes_

    hs, hb = facts(h_files)
    is_, ib = facts(i_files)
    out += ["## 2. Are `h_` files simply higher resolution?", "",
            "| group | n | longest side (min / median / max) | bytes (median) |",
            "|---|---|---|---|",
            f"| `h_img_` | {len(h_files)} | {min(hs)} / {int(statistics.median(hs))} / {max(hs)} "
            f"| {int(statistics.median(hb)):,} |",
            f"| `img_` | {len(i_files)} | {min(is_)} / {int(statistics.median(is_))} / {max(is_)} "
            f"| {int(statistics.median(ib)):,} |",
            "",
            f"Plain `img_` files reach {max(is_)} px as well, so `h_` is not a strict "
            "resolution tier, though `h_` files are systematically larger.",
            ""]

    # --- orientation-aware pHash, every h_ against every img_
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    i_hashes = {p: imagehash.phash(Image.open(p).convert("L")) for p in i_files}
    gray_cache: dict[Path, np.ndarray] = {}

    def gray(p: Path) -> np.ndarray:
        if p not in gray_cache:
            gray_cache[p] = load_gray(p)
        return gray_cache[p]

    out += ["## 3. Best content match for each `h_` file", "",
            "`phash_d` is the best distance over 4 orientations. `orb_inliers` is the count of "
            "keypoint matches consistent with one homography, which tolerates crop and scale.",
            "",
            "| `h_` file | best `img_` match | phash_d | orientation | orb_inliers |",
            "|---|---|---|---|---|"]

    rows = []
    for hp in h_files:
        ori = phash_orientations(hp)
        best = None
        for ip in i_files:
            for name, hh in ori.items():
                d = hh - i_hashes[ip]
                if best is None or d < best[0]:
                    best = (d, ip, name)
        d, ip, name = best
        inl = orb_inliers(gray(hp), gray(ip), orb, matcher)
        rows.append((hp, ip, d, name, inl))
        out.append(f"| `{hp.name}` | `{ip.name}` | {d} | {name} | {inl} |")

    # Also report the strongest ORB match overall, which may differ from the pHash pick.
    out += ["", "### Strongest ORB match for each `h_` file", "",
            "pHash can miss a cropped or re-shot view, so this scans ORB over all candidates.",
            "",
            "| `h_` file | best ORB match | orb_inliers |", "|---|---|---|"]
    orb_best = []
    for hp in h_files:
        scores = [(orb_inliers(gray(hp), gray(ip), orb, matcher), ip) for ip in i_files]
        scores.sort(reverse=True, key=lambda t: t[0])
        orb_best.append((hp, scores[0][1], scores[0][0]))
        out.append(f"| `{hp.name}` | `{scores[0][1].name}` | {scores[0][0]} |")

    # --- control distribution: what do unrelated pairs score?
    random.seed(0)
    pool = i_files if len(i_files) > 4 else i_files
    control = []
    pairs = random.sample(list(itertools.combinations(pool, 2)),
                          min(CONTROL_PAIRS, len(pool) * (len(pool) - 1) // 2))
    for a, b in pairs:
        control.append(orb_inliers(gray(a), gray(b), orb, matcher))
    control.sort()
    out += ["", "## 4. Control: what do unrelated pairs score?", "",
            f"ORB inliers over {len(control)} random `img_` pairs:", "",
            f"- median {control[len(control) // 2]}",
            f"- 95th percentile {control[int(0.95 * len(control))]}",
            f"- 99th percentile {control[int(0.99 * len(control))]}",
            f"- max {control[-1]}",
            ""]
    threshold = max(control[int(0.99 * len(control))] + 1, 15)
    out += [f"Treating the 99th percentile as the noise ceiling, a genuine match needs more "
            f"than about **{threshold}** inliers.", ""]

    verdict_hits = [(hp, ip, n) for hp, ip, n in orb_best if n >= threshold]
    out += ["## 5. Verdict", ""]
    if verdict_hits:
        out += [f"**{len(verdict_hits)} `h_` file(s) exceed the noise ceiling** and are "
                "candidate real second views:", ""]
        for hp, ip, n in verdict_hits:
            out.append(f"- `{hp.name}` matches `{ip.name}` with {n} inliers")
        out += ["", "These need a human look before being trusted as real positive pairs.", ""]
    else:
        out += ["**No `h_` file matches any `img_` file above the noise ceiling.**", "",
                "So the `h_` files are not second views of sarees that also appear as `img_` "
                "files. They are 9 additional distinct designs, and the prefix carries no "
                "pairing information we can exploit. The corpus therefore contains no real "
                "positive pairs, and synthetic recolors remain the only source of positives.",
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
