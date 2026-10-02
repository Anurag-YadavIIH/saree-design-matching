"""
Precompute the k-means cluster map for every real TRAIN image, once.

Why this is a separate step. k-means was the training bottleneck: running it per image per
epoch would spend most of the time budget on CPU work instead of gradient steps. Each image's
cluster assignment is computed once here at 256 px and cached as a uint8 index map under
data/cache/. A train-time recolour is then just `palette[cluster_map]` plus a luminance restore.

The cache lives under data/ (gitignored) because the maps are derived from the proprietary
corpus. Cache filenames are keyed by path, k and resolution, so changing k cannot silently
reuse stale maps.

Run:  python scripts/build_cache.py [--config configs/data.yaml] [--workers 4]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.color_aug import cache_path, load_or_build_cluster_map  # noqa: E402
from src.data.manifest import load_manifest  # noqa: E402


def _one(args: tuple[str, str, str, int]) -> tuple[str, bool]:
    data_root, rel, cache_dir, k = args
    p = cache_path(Path(cache_dir), rel, k)
    if p.exists():
        return rel, False
    load_or_build_cluster_map(Path(data_root) / rel, rel, Path(cache_dir), k, seed=0)
    return rel, True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    k = int(cfg["recolor"]["palette_bank"]["k_colors"])
    data_root = cfg["paths"]["data_root"]

    # Train images, plus val GALLERY images. Val and test queries were materialised at build
    # time with a different algorithm and never need a cluster map, but the in-distribution
    # diagnostic deliberately applies the TRAINING k-means method to the val gallery, so those
    # images need maps too. Without them the first validation would run k-means inline.
    train = load_manifest(split="train", include_synthetic=False, config_path=args.config)
    val_gallery = load_manifest(split="val", role="gallery", config_path=args.config)
    rels = sorted(set(train["path"]) | set(val_gallery["path"]))
    print(f"[cache] {len(train):,} train + {len(val_gallery):,} val gallery images "
          f"({len(rels):,} unique), k={k}, cache at {args.cache_dir}")

    t0 = time.time()
    built = 0
    jobs = [(data_root, r, args.cache_dir, k) for r in rels]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_one, j) for j in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            _, did_build = fut.result()
            built += int(did_build)
            if n % 200 == 0 or n == len(futures):
                print(f"[cache] {n:,}/{len(futures):,} ({built:,} newly built, "
                      f"{time.time() - t0:.0f}s)")

    print(f"[cache] done: {built:,} built, {len(rels) - built:,} already cached, "
          f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
