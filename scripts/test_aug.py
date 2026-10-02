"""
Visual check on the colour augmentation. For Anurag's eyes only.

Numbers cannot tell us whether a recoloured saree still looks like a saree. If the palette
swap posterises the weave into flat blocks, or the geometry crops away the motif entirely,
the training signal is broken in a way no assertion catches. So this writes a preview grid.

The output lands in outputs/, which is gitignored, because it contains derivatives of the
proprietary corpus. It must never be committed or uploaded.

Run:  python scripts/test_aug.py [--n-images 6] [--n-variants 4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.color_aug import (  # noqa: E402
    GeometryConfig,
    Recolorer,
    build_palette_bank,
    read_rgb,
    split_palette_bank,
)

TILE = 224
LABEL_H = 22


def label_tile(img: np.ndarray, text: str) -> np.ndarray:
    """Caption a tile so the grid is readable without counting rows."""
    out = np.full((TILE + LABEL_H, TILE, 3), 255, np.uint8)
    out[LABEL_H:] = cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA)
    cv2.putText(out, text[:30], (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1,
                cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--n-images", type=int, default=6)
    ap.add_argument("--n-variants", type=int, default=4)
    ap.add_argument("--out", default="outputs/aug_preview.jpg")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rc = cfg["recolor"]
    bank = build_palette_bank(rc["palette_bank"]["size"], rc["palette_bank"]["k_colors"],
                              seed=cfg["seed"])
    pools = split_palette_bank(bank, rc["palette_bank"]["split"], seed=cfg["seed"])
    geo = GeometryConfig.from_dict(cfg["geometry"], out_size=TILE)

    data_root = Path(cfg["paths"]["data_root"])
    # Prefer the Drive corpus: it is the client's target domain and is not stretched.
    drive_root = Path(cfg["sources"]["deeplure_drive"]["root"])
    pool = sorted(drive_root.rglob("*.jpg"))
    if len(pool) < args.n_images:
        pool = sorted(data_root.rglob("*.jpg"))
    if not pool:
        print(f"no images found under {data_root}")
        return 1

    rng = np.random.default_rng(cfg["seed"])
    picks = [pool[i] for i in rng.choice(len(pool), size=min(args.n_images, len(pool)),
                                        replace=False)]

    train_rc = Recolorer(method=rc["methods"]["train"], palettes=pools["train"],
                         geometry=geo, k_colors=rc["palette_bank"]["k_colors"],
                         cache_dir=Path("data/cache"))
    test_rc = Recolorer(method=rc["methods"]["test"], palettes=pools["test"],
                        geometry=geo, k_colors=rc["palette_bank"]["k_colors"],
                        cache_dir=Path("data/cache"))

    rows = []
    for img_path in picks:
        rel = img_path.relative_to(data_root).as_posix()
        tiles = [label_tile(read_rgb(img_path), "ORIGINAL")]
        for v in range(args.n_variants):
            out, meta = train_rc(img_path, rel, seed=1000 + v)
            tiles.append(label_tile(out, f"train kmeans {meta['palette_id']}"))
        for v in range(args.n_variants):
            out, meta = test_rc(img_path, rel, seed=2000 + v)
            tiles.append(label_tile(out, f"TEST lab {meta['palette_id']}"))
        rows.append(np.hstack(tiles))

    grid = np.vstack(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    print(f"[aug] wrote {out_path}  ({grid.shape[1]}x{grid.shape[0]})")
    print(f"[aug] {len(picks)} images x (1 original + {args.n_variants} train + "
          f"{args.n_variants} test variants)")
    print("[aug] CHECK BY EYE: does the weave texture survive the recolour, and is the "
          "motif still visible after cropping?")
    print("[aug] train and TEST rows must look like different recolouring styles. If they "
          "look identical, the held-out generator is not actually held out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
