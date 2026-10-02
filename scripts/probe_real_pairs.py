"""
Phase 1 / Step 1e: translation-invariant search for real same-design pairs.

Why this exists. The earlier colourway probe filtered candidate pairs on grayscale pHash
distance <= 10. That filter has a blind spot. Textile images are often crops of a seamless
repeating pattern, so two images of the same design can sit at different translation
offsets. A half-period shift leaves local pixels almost unchanged but completely changes
the DCT that pHash is computed from, pushing the distance up to random levels (28 to 38
was observed). Those pairs were therefore invisible to the earlier search.

This script matches every pair with ORB keypoints plus a RANSAC homography, which is
invariant to translation, crop and scale, then verifies the match at the pixel level with
normalised cross-correlation over the aligned overlap. Inlier count alone is not trusted,
because repeating motifs let RANSAC fit a homography between unrelated fabrics.

Each verified same-design pair is then classified by palette, which is the question that
actually matters for this project:

  NCC high + hue distance LOW   same design, same palette -> a duplicate or a re-crop
  NCC high + hue distance HIGH  same design, DIFFERENT palette -> a real colourway pair
  NCC low                       unrelated

Privacy: scores and metadata only. No pixels printed, nothing copied out of data/.

Run:  python scripts/probe_real_pairs.py [--root <dir>] [--out outputs/real_pairs.md]
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MAX_SIDE = 800
ORB_FEATURES = 1500
RATIO = 0.75
RANSAC_PX = 5.0

MIN_INLIERS = 25        # below this, no homography is worth verifying
MIN_OVERLAP = 0.10      # an overlap smaller than this makes NCC meaningless
MIN_STD = 8.0           # both overlap regions must carry real texture, not flat background
NCC_SAME_DESIGN = 0.90  # pixel agreement required to call it the same design
HUE_SAME = 0.15         # hue histogram L1 below this means the palette is unchanged
HUE_DIFFERENT = 0.45    # above this the palette is clearly different


def load(path: Path) -> np.ndarray:
    """RGB working image with the longest side capped, so scores are size comparable."""
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        scale = MAX_SIDE / max(rgb.size)
        if scale < 1.0:
            rgb = rgb.resize((max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))))
        return np.asarray(rgb)


def hue_hist(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Normalised hue histogram over `mask`, ignoring pixels whose hue is meaningless.

    Returns None when too few saturated pixels remain, because a histogram built from a
    handful of pixels is noise and will invent colour differences that are not there.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = mask & (s > 40) & (v > 30)
    if m.sum() < 500:
        return None
    hist = np.histogram(h[m], bins=18, range=(0, 180))[0].astype(np.float32)
    return hist / hist.sum()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Google_drive_data/handloom_sarees")
    ap.add_argument("--out", default="outputs/real_pairs.md")
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--paths-file", default=None,
                    help="newline separated image paths to scan instead of globbing root; "
                         "used to scan one representative per duplicate cluster")
    args = ap.parse_args()

    root = Path(args.root)
    if args.paths_file:
        paths = sorted(Path(line.strip()) for line in
                       Path(args.paths_file).read_text(encoding="utf-8").splitlines()
                       if line.strip())
    else:
        paths = sorted(p for p in root.glob(args.glob))
    if len(paths) < 2:
        print(f"need at least 2 images under {root}")
        return 1

    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Detect once per image; the pairwise loop then only matches descriptors.
    rgbs: dict[Path, np.ndarray] = {}
    grays: dict[Path, np.ndarray] = {}
    kps: dict[Path, tuple] = {}
    for p in paths:
        rgb = load(p)
        rgbs[p] = rgb
        grays[p] = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kps[p] = orb.detectAndCompute(grays[p], None)

    def verify(a: Path, b: Path) -> dict | None:
        ka, da = kps[a]
        kb, db = kps[b]
        if da is None or db is None or len(da) < 8 or len(db) < 2:
            return None
        good = []
        for m_n in matcher.knnMatch(db, da, k=2):
            if len(m_n) == 2 and m_n[0].distance < RATIO * m_n[1].distance:
                good.append(m_n[0])
        if len(good) < 8:
            return None
        src = np.float32([kb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([ka[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
        if H is None or mask is None:
            return None
        inl = int(mask.sum())
        if inl < MIN_INLIERS:
            return None
        det = abs(float(np.linalg.det(H[:2, :2])))
        if not (0.25 < det < 4.0):   # reject wild distortions; a real view is near-similarity
            return None

        ga, gb = grays[a], grays[b]
        ra, rb = rgbs[a], rgbs[b]
        h_a, w_a = ga.shape
        warped = cv2.warpPerspective(gb, H, (w_a, h_a))
        warped_rgb = cv2.warpPerspective(rb, H, (w_a, h_a))
        valid = cv2.warpPerspective(np.ones(gb.shape, dtype=np.uint8), H, (w_a, h_a)) > 0
        overlap = float(valid.mean())
        if overlap < MIN_OVERLAP or valid.sum() < 2000:
            return None
        x = ga[valid].astype(np.float64)
        y = warped[valid].astype(np.float64)
        if x.std() < MIN_STD or y.std() < MIN_STD:
            return None
        ncc = float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std()))

        # Hue must be compared INSIDE the aligned overlap. Comparing whole images conflates
        # a recolour with a different crop: a saree's body, border and pallu carry different
        # colours, so two crops of one fabric show a large whole-image hue gap while the
        # shared region is identical. Measured on this corpus: whole-image L1 up to 0.86 for
        # pairs whose in-overlap L1 was 0.01 to 0.04.
        ha_full = hue_hist(ra, np.ones(ga.shape, dtype=bool))
        hb_full = hue_hist(rb, np.ones(gb.shape, dtype=bool))
        hue_whole = (float(np.abs(ha_full - hb_full).sum())
                     if ha_full is not None and hb_full is not None else float("nan"))
        ha_ov = hue_hist(ra, valid)
        hb_ov = hue_hist(warped_rgb, valid)
        hue_overlap = (float(np.abs(ha_ov - hb_ov).sum())
                       if ha_ov is not None and hb_ov is not None else float("nan"))

        return {"inliers": inl, "overlap": overlap, "ncc": ncc, "scale": det ** 0.5,
                "hue_overlap": hue_overlap, "hue_whole": hue_whole}

    verified: list[tuple[Path, Path, dict]] = []
    total = len(paths) * (len(paths) - 1) // 2
    for n, (a, b) in enumerate(itertools.combinations(paths, 2), 1):
        if n % 2000 == 0:
            print(f"  ... {n}/{total} pairs")
        r = verify(a, b)
        if r and r["ncc"] >= NCC_SAME_DESIGN:
            verified.append((a, b, r))

    # Group verified pairs into connected components: one component is one design.
    parent: dict[Path, Path] = {p: p for p in paths}

    def find(x: Path) -> Path:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in verified:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    comps: dict[Path, list[Path]] = {}
    for p in paths:
        comps.setdefault(find(p), []).append(p)
    multi = {k: v for k, v in comps.items() if len(v) > 1}

    # Classification uses the in-overlap hue distance only.
    def ov_hue(t) -> float:
        return t[2]["hue_overlap"]

    measurable = [t for t in verified if ov_hue(t) == ov_hue(t)]  # drop NaN
    unmeasurable = len(verified) - len(measurable)
    same_palette = [t for t in measurable if ov_hue(t) < HUE_SAME]
    diff_palette = [t for t in measurable if ov_hue(t) >= HUE_DIFFERENT]
    middle = [t for t in measurable if HUE_SAME <= ov_hue(t) < HUE_DIFFERENT]

    out: list[str] = [
        "# Real same-design pairs, found translation-invariantly (Phase 1, Step 1e)",
        "",
        "Scores and metadata only. No pixels printed.",
        "",
        f"- source: `{root.as_posix()}`",
        f"- images: {len(paths)}, pairs tested: {total:,}",
        f"- pairs verified as the same design (ORB homography plus NCC >= {NCC_SAME_DESIGN}): "
        f"**{len(verified)}**",
        f"- designs with more than one image: **{len(multi)}** "
        f"(covering {sum(len(v) for v in multi.values())} images)",
        "",
        "## Why pHash missed these",
        "",
        "A pair can be the same seamless pattern at a different translation offset. Local "
        "pixels then agree almost perfectly while the global DCT, and therefore pHash, "
        "changes completely. Any dedup or identity rule for this corpus must be "
        "translation-invariant, not pHash alone.",
        "",
        "## Palette classification of the verified same-design pairs",
        "",
        f"| category | criterion | count |",
        "|---|---|---|",
        f"| same palette | in-overlap hue L1 < {HUE_SAME} | {len(same_palette)} |",
        f"| ambiguous | between the thresholds | {len(middle)} |",
        f"| **different palette** | in-overlap hue L1 >= {HUE_DIFFERENT} "
        f"| **{len(diff_palette)}** |",
        f"| not measurable | too few saturated pixels in overlap | {unmeasurable} |",
        "",
    ]

    if diff_palette:
        out += ["### Real colourway pairs (same design, different palette)", "",
                "These are the pairs this project most needs. Each should be reviewed by eye "
                "before being trusted.", "",
                "| image A | image B | NCC | overlap | hue (overlap) | hue (whole) | inliers |",
                "|---|---|---|---|---|---|---|"]
        for a, b, r in sorted(diff_palette, key=lambda t: -t[2]["hue_overlap"]):
            out.append(f"| `{a.name}` | `{b.name}` | {r['ncc']:.3f} | {r['overlap']:.2f} | "
                       f"{r['hue_overlap']:.2f} | {r['hue_whole']:.2f} | {r['inliers']} |")
        out.append("")

    if multi:
        out += ["## Multi-image designs", "",
                "| design (component) | images |", "|---|---|"]
        for k, v in sorted(multi.items(), key=lambda t: -len(t[1])):
            out.append(f"| `{k.name}` | {', '.join('`' + p.name + '`' for p in sorted(v))} |")
        out.append("")

    if same_palette:
        out += ["## Same-design, same-palette pairs (duplicates or re-crops)", "",
                "These must be collapsed into one `dup_group`, and they cannot serve as "
                "colourway positives.", "",
                "| image A | image B | NCC | hue (overlap) | hue (whole) |",
                "|---|---|---|---|---|"]
        for a, b, r in sorted(same_palette, key=lambda t: -t[2]["ncc"])[:40]:
            out.append(f"| `{a.name}` | `{b.name}` | {r['ncc']:.3f} | "
                       f"{r['hue_overlap']:.2f} | {r['hue_whole']:.2f} |")
        out.append("")

    text = "\n".join(out) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"[probe] written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
