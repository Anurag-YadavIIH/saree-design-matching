"""
Mine REAL colourway pairs from the Kaggle catalogue: same print, different colours.

Why an earlier probe found none. It required low grayscale pHash distance first, then a large
hue difference. A real colourway is usually a SEPARATE photograph (often a swirl or drape shot),
so its pHash distance is high even when the print is identical; that filter discarded every true
pair before hue was ever examined. Here the structural signal is zero-shot DINOv2 on GRAYSCALE
input, which tolerates reframing and ignores colour by construction.

Steps: embed every Kaggle representative in grayscale, take each one's top-k neighbours from a
different design, keep pairs whose hue histograms differ strongly, and render contact sheets
for a human (or Claude) judgement. Only judged pairs are used for evaluation.

Kaggle only: the source is public (MIT), so contact sheets may be viewed. Never Drive images.

Run:  python scripts/mine_colourway_pairs.py [--topk 3] [--min-hue 0.6]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.geometric_id import dinov2_embeddings  # noqa: E402
from src.data.manifest import load_manifest  # noqa: E402

THUMB = 200


def hue_hist(path: Path) -> tuple[np.ndarray | None, float]:
    """Whole-image hue histogram over saturated pixels, plus the saturated fraction."""
    bgr = cv2.imread(str(path))
    hsv = cv2.cvtColor(cv2.resize(bgr, (128, 128)), cv2.COLOR_BGR2HSV)
    m = (hsv[..., 1] > 40) & (hsv[..., 2] > 30)
    if m.sum() < 200:
        return None, float(m.mean())
    h = np.histogram(hsv[..., 0][m], bins=18, range=(0, 180))[0].astype(np.float32)
    return h / h.sum(), float(m.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--min-hue", type=float, default=0.6)
    ap.add_argument("--min-sat", type=float, default=0.15)
    ap.add_argument("--per-sheet", type=int, default=20)
    args = ap.parse_args()

    reps = load_manifest(source="kaggle_fabric", include_synthetic=False,
                         representatives_only=True).reset_index(drop=True)
    root = Path("data")
    paths = [root / p for p in reps["path"]]
    print(f"[mine] {len(reps)} Kaggle representatives, embedding in grayscale")
    emb = dinov2_embeddings(paths, grayscale=True)
    sims = emb @ emb.T
    np.fill_diagonal(sims, -1)

    hists = [hue_hist(p) for p in paths]
    designs = reps["design_id"].to_numpy()
    cands = {}
    for i in range(len(reps)):
        for j in np.argsort(-sims[i])[:args.topk]:
            j = int(j)
            if designs[i] == designs[j]:
                continue
            a, b = min(i, j), max(i, j)
            (ha, sa), (hb, sb) = hists[a], hists[b]
            if ha is None or hb is None or min(sa, sb) < args.min_sat:
                continue
            hue = float(np.abs(ha - hb).sum())
            if hue >= args.min_hue:
                cands[(a, b)] = (float(sims[a, b]), hue)

    rows = []
    for (a, b), (cos, hue) in sorted(cands.items(), key=lambda kv: -kv[1][0]):
        rows.append({"idx": len(rows) + 1, "cosine_gray": round(cos, 4), "hue_l1": round(hue, 3),
                     "design_a": designs[a], "design_b": designs[b],
                     "family_a": reps.loc[a, "craft_family"], "family_b": reps.loc[b, "craft_family"],
                     "split_a": reps.loc[a, "split"], "split_b": reps.loc[b, "split"],
                     "path_a": reps.loc[a, "path"], "path_b": reps.loc[b, "path"]})
    cand_df = pd.DataFrame(rows)
    cand_df.to_csv("outputs/colourway_candidates.csv", index=False)
    print(f"[mine] {len(cand_df)} candidates (top-{args.topk} gray neighbours, hue L1 >= "
          f"{args.min_hue}, saturated >= {args.min_sat})")

    def thumb(p):
        return cv2.resize(cv2.imread(str(root / p)), (THUMB, THUMB))

    for s in range(0, len(cand_df), args.per_sheet):
        chunk = cand_df.iloc[s:s + args.per_sheet]
        tiles = []
        for r in chunk.itertuples():
            pair = np.hstack([thumb(r.path_a), np.full((THUMB, 4, 3), 255, np.uint8),
                              thumb(r.path_b)])
            bar = np.full((24, pair.shape[1], 3), 255, np.uint8)
            cv2.putText(bar, f"#{r.idx} cos={r.cosine_gray:.2f} hue={r.hue_l1:.2f} {r.family_a[:4]}/{r.family_b[:4]}",
                        (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(np.vstack([bar, pair, np.full((6, pair.shape[1], 3), 255, np.uint8)]))
        while len(tiles) % 2:
            tiles.append(np.full_like(tiles[0], 255))
        grid = np.vstack([np.hstack([tiles[i], np.full((tiles[i].shape[0], 12, 3), 200, np.uint8),
                                     tiles[i + 1]]) for i in range(0, len(tiles), 2)])
        out = f"outputs/colourway_sheet_{s // args.per_sheet + 1}.jpg"
        cv2.imwrite(out, grid, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        print(f"[mine] wrote {out} (pairs {chunk.idx.min()} to {chunk.idx.max()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
