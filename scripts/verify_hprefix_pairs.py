"""
Phase 1 / Step 1d: are the ORB candidate pairs real, or lattice false positives?

scripts/probe_hprefix.py flagged 5 candidate `h_`/`img_` pairs on ORB inlier count, but its
own control showed unrelated pairs reaching a 99th percentile of 309 inliers. That is a
warning sign: textile motifs repeat, so RANSAC can fit a homography to a grid of similar
motifs belonging to two completely different fabrics. Inlier count alone cannot be trusted.

This script applies the decisive test. For each pair it estimates the homography, warps one
image into the other's frame, and measures whether the PIXELS actually line up in the
overlap region using normalised cross-correlation. A genuine second view of the same saree
aligns. A lattice false match does not, however many inliers it produced.

Two controls make the NCC number interpretable:
  positive control  an image against a cropped, rescaled, slightly rotated copy of itself,
                    which is the strongest possible "same object, different view" signal
  negative control  random unrelated pairs

Privacy: scores and metadata only. No pixels printed, nothing copied out of data/.

Run:  python scripts/verify_hprefix_pairs.py [--out outputs/hprefix_verify.md]
"""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MAX_SIDE = 800
ORB_FEATURES = 2000
RATIO = 0.75
RANSAC_PX = 5.0

# Candidates flagged by probe_hprefix.py, as (h_ file, img_ file).
CANDIDATES = [
    ("h_img_132981.jpg", "img_447852.jpg"),
    ("h_img_31665.jpg", "img_120749.jpg"),
    ("h_img_35259.jpg", "img_90832.jpg"),
    ("h_img_44440.jpg", "img_65263.jpg"),
    ("h_img_593526.jpg", "img_257108.jpg"),
]


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("L")
        scale = MAX_SIDE / max(im.size)
        if scale < 1.0:
            im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
        return np.asarray(im)


def align_score(a: np.ndarray, b: np.ndarray, orb, matcher) -> dict:
    """Estimate a homography from b to a, then measure pixel agreement in the overlap.

    Returns inlier count, the NCC of the aligned overlap, the overlap fraction, and a
    degeneracy flag for homographies that collapse or wildly distort the image.
    """
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    result = {"inliers": 0, "ncc": float("nan"), "overlap": 0.0, "degenerate": True}
    if da is None or db is None or len(db) < 2 or len(ka) < 8 or len(kb) < 8:
        return result

    good = []
    for m_n in matcher.knnMatch(db, da, k=2):
        if len(m_n) == 2 and m_n[0].distance < RATIO * m_n[1].distance:
            good.append(m_n[0])
    if len(good) < 8:
        return result

    src = np.float32([kb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ka[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    if H is None or mask is None:
        return result
    result["inliers"] = int(mask.sum())

    # A sane same-object homography is close to a similarity transform. Check the implied
    # scale is not absurd and the matrix is not near singular.
    det = float(np.linalg.det(H[:2, :2]))
    result["degenerate"] = not (0.05 < abs(det) < 20.0)

    warped = cv2.warpPerspective(b, H, (a.shape[1], a.shape[0]))
    valid = cv2.warpPerspective(np.ones_like(b, dtype=np.uint8), H,
                                (a.shape[1], a.shape[0])) > 0
    result["overlap"] = float(valid.mean())
    if valid.sum() < 0.05 * a.size:
        return result

    x = a[valid].astype(np.float64)
    y = warped[valid].astype(np.float64)
    xs, ys = x.std(), y.std()
    if xs < 1e-6 or ys < 1e-6:
        return result
    result["ncc"] = float(((x - x.mean()) * (y - y.mean())).mean() / (xs * ys))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Google_drive_data/handloom_sarees")
    ap.add_argument("--out", default="outputs/hprefix_verify.md")
    args = ap.parse_args()

    root = Path(args.root)
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    cache: dict[Path, np.ndarray] = {}

    def gray(p: Path) -> np.ndarray:
        if p not in cache:
            cache[p] = load_gray(p)
        return cache[p]

    out: list[str] = [
        "# Are the `h_` ORB candidates real pairs? (Phase 1, Step 1d)",
        "",
        "Scores and metadata only. No pixels printed.",
        "",
        "ORB inlier count is unreliable on repeating textile motifs, so each candidate is "
        "re-tested by warping one image onto the other and measuring whether the pixels "
        "actually agree (normalised cross-correlation over the overlap).",
        "",
    ]

    # --- positive control: an image against a cropped, rescaled, rotated copy of itself
    random.seed(0)
    all_imgs = sorted(root.glob("img_*.jpg"))
    pos_scores = []
    for p in random.sample(all_imgs, 12):
        g = gray(p)
        h, w = g.shape
        crop = g[int(0.1 * h):int(0.9 * h), int(0.1 * w):int(0.9 * w)]
        M = cv2.getRotationMatrix2D((crop.shape[1] / 2, crop.shape[0] / 2), 7, 1.0)
        warped = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]))
        view = cv2.resize(warped, (int(crop.shape[1] * 0.85), int(crop.shape[0] * 0.85)))
        r = align_score(g, view, orb, matcher)
        if not np.isnan(r["ncc"]):
            pos_scores.append(r["ncc"])

    # --- negative control: random unrelated pairs
    neg_scores = []
    for a, b in random.sample(list(itertools.combinations(all_imgs, 2)), 120):
        r = align_score(gray(a), gray(b), orb, matcher)
        if not np.isnan(r["ncc"]):
            neg_scores.append(r["ncc"])

    def describe(v: list[float], label: str) -> str:
        if not v:
            return f"- {label}: no valid alignments"
        s = sorted(v)
        return (f"- {label}: n={len(s)} median NCC {s[len(s) // 2]:.3f}, "
                f"5th pct {s[int(0.05 * len(s))]:.3f}, 95th pct {s[int(0.95 * len(s))]:.3f}")

    out += ["## Controls", "",
            describe(pos_scores, "positive control (same image, cropped + rotated + rescaled)"),
            describe(neg_scores, "negative control (random unrelated pairs)"),
            ""]
    if pos_scores and neg_scores:
        cut = (np.median(pos_scores) + np.percentile(neg_scores, 95)) / 2
        out += [f"A reasonable decision boundary sits near NCC **{cut:.3f}**, between the "
                "negative tail and the positive median.", ""]
    else:
        cut = 0.5

    out += ["## Candidates", "",
            "| `h_` file | `img_` file | inliers | overlap | NCC | degenerate H | verdict |",
            "|---|---|---|---|---|---|---|"]

    verdicts = []
    for hn, inm in CANDIDATES:
        hp, ip = root / hn, root / inm
        if not hp.exists() or not ip.exists():
            out.append(f"| `{hn}` | `{inm}` | missing | | | | skipped |")
            continue
        r = align_score(gray(hp), gray(ip), orb, matcher)
        ncc = r["ncc"]
        real = (not r["degenerate"]) and (not np.isnan(ncc)) and ncc >= cut
        verdicts.append((hn, inm, real, ncc))
        ncc_s = "n/a" if np.isnan(ncc) else f"{ncc:.3f}"
        out.append(f"| `{hn}` | `{inm}` | {r['inliers']} | {r['overlap']:.2f} | {ncc_s} | "
                   f"{'yes' if r['degenerate'] else 'no'} | "
                   f"{'REAL PAIR' if real else 'false positive'} |")

    n_real = sum(1 for *_, real, _ in [(a, b, c, d) for a, b, c, d in verdicts] if real)
    out += ["", "## Verdict", ""]
    if n_real:
        out += [f"**{n_real} of {len(verdicts)} candidates survive pixel verification.**", ""]
        for hn, inm, real, ncc in verdicts:
            if real:
                out.append(f"- `{hn}` and `{inm}` align with NCC {ncc:.3f}")
        out += ["", "These are genuine real positive pairs and should be kept together, placed "
                "in the test split, and reported as a separate real-pairs evaluation.", ""]
    else:
        out += ["**None of the candidates survives pixel verification.**", "",
                "Every one was a lattice false positive: ORB found many matches among "
                "repeating motifs, RANSAC fitted a homography to that repetition, but the "
                "warped images do not actually agree pixel for pixel. This is consistent with "
                "the exhaustive pHash result, where the minimum distance over all 13,530 Drive "
                "pairs was 10.",
                "",
                "Conclusion: the `h_` prefix does not mark a second view. The 9 `h_` files are "
                "9 further distinct designs. The corpus contains no real positive pairs, so "
                "synthetic recolors remain the only source of positives.",
                ""]

    text = "\n".join(out) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"[verify] written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
