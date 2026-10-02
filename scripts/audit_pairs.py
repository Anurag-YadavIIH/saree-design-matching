"""
Verdicts for the leakage audit's top test/train pairs.

The audit (src/data/validate.py) ranks each test image's nearest TRAIN image by zero-shot
DINOv2 cosine. High cosine alone does not mean "same saree": two different sarees of one craft
family can look alike. So each pair is re-tested with the pixel-level verifier used to build
design identity, over several orientations, because a second photo may be mirrored or rotated:

  SAME            ORB homography plus NCC >= 0.90 in some orientation. Same pixels: a leak.
  NEEDS EYES      not pixel-verifiable but cosine is very high. Could be a separate photo of
                  one saree, which no content method can prove. Decide from the grid.
  DIFFERENT       neither.

Writes outputs/leakage_verdicts.csv and, for human review only, outputs/leakage_pairs.jpg
(gitignored: it shows proprietary images and must never be committed or shared).

Run:  python scripts/audit_pairs.py [--top 20]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.geometric_id import GeoConfig, GeometricVerifier, HueConfig  # noqa: E402

ORIENTATIONS = {
    "identity": lambda a: a,
    "hflip": lambda a: a[:, ::-1],
    "vflip": lambda a: a[::-1],
    "rot90": lambda a: np.rot90(a, 1),
    "rot180": lambda a: np.rot90(a, 2),
    "rot270": lambda a: np.rot90(a, 3),
}
NEEDS_EYES_COSINE = 0.93


def verify_oriented(pa: Path, pb: Path, geo: GeoConfig, hue: HueConfig) -> tuple[float, str, int]:
    """Best NCC over orientations of image B. Returns (ncc, orientation, inliers)."""
    best = (0.0, "", 0)
    for name, fn in ORIENTATIONS.items():
        v = GeometricVerifier([pa, pb], geo, hue)
        v._load(0)
        v._load(1)
        rgb = np.ascontiguousarray(fn(v._rgb[1]))
        v._rgb[1] = rgb
        v._gray[1] = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        v._kp[1] = v._orb.detectAndCompute(v._gray[1], None)
        r = v.verify(0, 1)
        if r is not None and r["ncc"] > best[0]:
            best = (r["ncc"], name, r["inliers"])
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--audit", default="outputs/leakage_audit.csv")
    args = ap.parse_args()

    a = pd.read_csv(args.audit)
    # Roboflow noise copies of one image repeat the same pair; collapse to design pairs.
    a = a.sort_values("cosine", ascending=False).drop_duplicates(
        ["test_design", "nearest_train_design"]).head(args.top).reset_index(drop=True)

    # Verification uses a looser NCC floor so near misses are visible; the verdict uses 0.90.
    geo = GeoConfig(ncc_same_design=0.0)
    hue = HueConfig()
    root = Path("data")
    rows, tiles = [], []
    for r in a.itertuples():
        pa, pb = root / r.test_path, root / r.nearest_train_path
        ncc, ori, inl = verify_oriented(pa, pb, geo, hue)
        if ncc >= 0.90:
            verdict = "SAME"
        elif r.cosine >= NEEDS_EYES_COSINE:
            verdict = "NEEDS EYES"
        else:
            verdict = "DIFFERENT"
        rows.append({"rank": r.Index + 1, "cosine": r.cosine, "best_ncc": round(ncc, 3),
                     "orientation": ori or "-", "inliers": inl, "verdict": verdict,
                     "test_design": r.test_design, "train_design": r.nearest_train_design,
                     "test_path": r.test_path, "train_path": r.nearest_train_path})

        # Side-by-side tile for the human reviewer.
        def thumb(p):
            im = cv2.imread(str(p))
            return cv2.resize(im, (256, 256)) if im is not None else np.zeros((256, 256, 3), np.uint8)
        tile = np.hstack([thumb(pa), np.full((256, 6, 3), 255, np.uint8), thumb(pb)])
        bar = np.full((28, tile.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, f"#{r.Index + 1} {verdict}  cos={r.cosine:.3f} ncc={ncc:.2f} {ori}",
                    (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, tile]))

    out = pd.DataFrame(rows)
    out.to_csv("outputs/leakage_verdicts.csv", index=False)
    grid_rows = [np.hstack(tiles[i:i + 2]) if i + 1 < len(tiles)
                 else np.hstack([tiles[i], np.full_like(tiles[i], 255)])
                 for i in range(0, len(tiles), 2)]
    cv2.imwrite("outputs/leakage_pairs.jpg", np.vstack(grid_rows),
                [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    print(out[["rank", "cosine", "best_ncc", "orientation", "inliers", "verdict",
               "test_design", "train_design"]].to_string(index=False))
    print(f"\nverdicts: {out['verdict'].value_counts().to_dict()}")
    print("grid for YOUR review only: outputs/leakage_pairs.jpg (left = test, right = train)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
