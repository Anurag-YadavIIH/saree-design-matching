"""
Design identity by geometric verification, because pHash cannot do this job.

The central finding of Phase 1. Textile images are often crops of a repeating pattern, so
two images of the same design can sit at different translation offsets. A half-period shift
leaves local pixels almost unchanged but completely changes the DCT that pHash is built
from. Measured on the Drive corpus: pairs at pHash distance **28 to 38** (random level)
whose pixels agree at **NCC 0.97 to 0.99** once aligned. Relying on pHash would have split
27 Drive designs across train and test, putting the same physical saree on both sides.

So identity is established in three stages:

  1. **Prefilter.** All-pairs ORB does not scale: 412 Kaggle representatives is ~85k pairs.
     Zero-shot DINOv2 cosine gives each image its top-k nearest neighbours cheaply, and only
     those candidate pairs go to stage 2. pHash-identical pairs are added as candidates too,
     so the cheap signal is not wasted.
  2. **Verify.** ORB keypoints plus a RANSAC homography, then normalised cross-correlation
     over the aligned overlap. Inlier count alone is NOT trusted: repeating motifs let RANSAC
     fit a homography between unrelated fabrics, and the control distribution reached a 99th
     percentile of 309 inliers. Pixel agreement is what decides.
  3. **Group.** Union-find over the verified-pair graph, so a chain of overlapping tiles
     (1 overlaps 2, 2 overlaps 3) merges into ONE design even when 1 and 3 share no pixels.

Hue is compared inside the aligned overlap only. Comparing whole images conflates a recolour
with a different crop and produced 8 false colourway pairs on the first pass.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class GeoConfig:
    max_side: int = 800
    orb_features: int = 1500
    lowe_ratio: float = 0.75
    ransac_px: float = 5.0
    min_inliers: int = 25
    max_det_ratio: tuple[float, float] = (0.25, 4.0)
    min_overlap: float = 0.10
    min_region_std: float = 8.0
    ncc_same_design: float = 0.90

    @classmethod
    def from_dict(cls, d: dict) -> "GeoConfig":
        return cls(
            max_side=int(d.get("max_side", 800)),
            orb_features=int(d.get("orb_features", 1500)),
            lowe_ratio=float(d.get("lowe_ratio", 0.75)),
            ransac_px=float(d.get("ransac_px", 5.0)),
            min_inliers=int(d.get("min_inliers", 25)),
            max_det_ratio=tuple(d.get("max_det_ratio", (0.25, 4.0))),
            min_overlap=float(d.get("min_overlap", 0.10)),
            min_region_std=float(d.get("min_region_std", 8.0)),
            ncc_same_design=float(d.get("ncc_same_design", 0.90)),
        )


@dataclass
class HueConfig:
    bins: int = 18
    min_saturation: int = 40
    min_value: int = 30
    min_pixels: int = 500
    same_palette_max: float = 0.15
    different_palette_min: float = 0.45

    @classmethod
    def from_dict(cls, d: dict) -> "HueConfig":
        return cls(**{k: type(getattr(cls(), k))(v) for k, v in d.items()
                      if hasattr(cls(), k)})


# ------------------------------------------------------------------- DINOv2 prefilter

def dinov2_embeddings(paths: list[Path], batch_size: int = 16, image_size: int = 224,
                      device: str | None = None, grayscale: bool = False) -> np.ndarray:
    """Zero-shot DINOv2 ViT-S/14 CLS embeddings, L2 normalised.

    Imported lazily so the rest of the data layer works without torch installed. Used both
    as the candidate prefilter here and as a baseline in Phase 3, so one implementation
    serves both and they cannot disagree.
    """
    import torch
    from PIL import Image

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    model.eval().to(device)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            chunk = paths[i:i + batch_size]
            arrs = []
            for p in chunk:
                with Image.open(p) as im:
                    im = im.convert("L").convert("RGB") if grayscale else im.convert("RGB")
                    im = im.resize((image_size, image_size), Image.BICUBIC)
                    arrs.append(np.asarray(im, dtype=np.float32) / 255.0)
            x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).to(device)
            x = (x - mean) / std
            feats = model(x)
            feats = torch.nn.functional.normalize(feats, dim=1)
            out.append(feats.cpu().numpy())
    return np.concatenate(out, axis=0)


def histogram_descriptors(paths: list[Path], size: int = 128) -> np.ndarray:
    """Torch-free fallback prefilter descriptor, L2 normalised.

    Used when DINOv2 is unavailable. The design constraint that matters: the pairs we must
    catch are overlapping CROPS of one fabric, so the descriptor has to be translation
    invariant. Global histograms are invariant by construction, which a downscaled-image
    descriptor would not be:

      - hue histogram            which colours the fabric uses
      - gradient orientation     which directions the weave and motif run in
      - gradient magnitude       how fine or coarse the texture is
      - intensity histogram      the tonal distribution

    Weaker than DINOv2, so it is only a candidate generator. Every pair it proposes is still
    pixel-verified, so a mediocre prefilter costs recall, never precision.
    """
    from PIL import Image

    out = []
    for p in paths:
        with Image.open(p) as im:
            rgb = np.asarray(im.convert("RGB").resize((size, size), Image.BICUBIC))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        sat_mask = (s > 40) & (v > 30)
        hue_h = np.histogram(h[sat_mask] if sat_mask.any() else np.zeros(1),
                             bins=18, range=(0, 180))[0].astype(np.float32)

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        ang = (np.arctan2(gy, gx) % np.pi) * (180.0 / np.pi)
        ori_h = np.histogram(ang, bins=18, range=(0, 180), weights=mag)[0].astype(np.float32)
        mag_h = np.histogram(mag, bins=16, range=(0, 512))[0].astype(np.float32)
        int_h = np.histogram(gray, bins=16, range=(0, 256))[0].astype(np.float32)

        # Each block is normalised separately so one block cannot dominate the cosine.
        blocks = []
        for b in (hue_h, ori_h, mag_h, int_h):
            n = np.linalg.norm(b)
            blocks.append(b / n if n > 0 else b)
        d = np.concatenate(blocks)
        out.append(d / max(np.linalg.norm(d), 1e-8))
    return np.stack(out)


def topk_candidates(embeddings: np.ndarray, k: int = 10) -> set[tuple[int, int]]:
    """Each image's k nearest neighbours by cosine, as an undirected candidate set.

    Chunked so a large corpus does not need an N x N matrix in memory at once.
    """
    n = embeddings.shape[0]
    pairs: set[tuple[int, int]] = set()
    chunk = 512
    for start in range(0, n, chunk):
        sims = embeddings[start:start + chunk] @ embeddings.T
        for local, row in enumerate(sims):
            i = start + local
            row[i] = -np.inf                      # never pair an image with itself
            for j in np.argpartition(-row, min(k, n - 1))[:k]:
                pairs.add((min(i, int(j)), max(i, int(j))))
    return pairs


def phash_candidates(phashes: list[str], threshold: int = 4,
                     slice_width: int = 4) -> set[tuple[int, int]]:
    """Pairs within a Hamming threshold, found by bucketing on hash slices.

    Catches same-framing duplicates (the Roboflow noise triplets) that DINOv2 might rank
    below some other neighbour. Cheap, so it is simply added to the candidate set.
    """
    def to_int(h: str) -> int:
        return int(h, 16)

    buckets: dict[tuple[int, str], list[int]] = {}
    for i, h in enumerate(phashes):
        if not h:
            continue
        for s in range(0, len(h), slice_width):
            buckets.setdefault((s, h[s:s + slice_width]), []).append(i)

    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) > 200:      # a degenerate bucket is not informative
            continue
        for a, b in itertools.combinations(members, 2):
            if bin(to_int(phashes[a]) ^ to_int(phashes[b])).count("1") <= threshold:
                pairs.add((min(a, b), max(a, b)))
    return pairs


# ------------------------------------------------------------------------- verification

class GeometricVerifier:
    """Holds per-image keypoints so each image is only processed once."""

    def __init__(self, paths: list[Path], cfg: GeoConfig, hue_cfg: HueConfig):
        self.paths = paths
        self.cfg = cfg
        self.hue_cfg = hue_cfg
        self._orb = cv2.ORB_create(nfeatures=cfg.orb_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._rgb: dict[int, np.ndarray] = {}
        self._gray: dict[int, np.ndarray] = {}
        self._kp: dict[int, tuple] = {}

    def _load(self, i: int) -> None:
        if i in self._rgb:
            return
        from PIL import Image
        with Image.open(self.paths[i]) as im:
            rgb = im.convert("RGB")
            scale = self.cfg.max_side / max(rgb.size)
            if scale < 1.0:
                rgb = rgb.resize((max(1, int(rgb.width * scale)),
                                  max(1, int(rgb.height * scale))))
            arr = np.asarray(rgb)
        self._rgb[i] = arr
        self._gray[i] = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        self._kp[i] = self._orb.detectAndCompute(self._gray[i], None)

    def _hue_hist(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        c = self.hue_cfg
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        m = mask & (s > c.min_saturation) & (v > c.min_value)
        if m.sum() < c.min_pixels:
            return None
        hist = np.histogram(h[m], bins=c.bins, range=(0, 180))[0].astype(np.float32)
        return hist / hist.sum()

    def verify(self, i: int, j: int) -> dict | None:
        """Return match stats if i and j are the same design, else None."""
        cfg = self.cfg
        self._load(i)
        self._load(j)
        ka, da = self._kp[i]
        kb, db = self._kp[j]
        if da is None or db is None or len(da) < 8 or len(db) < 2:
            return None

        good = []
        for m_n in self._matcher.knnMatch(db, da, k=2):
            if len(m_n) == 2 and m_n[0].distance < cfg.lowe_ratio * m_n[1].distance:
                good.append(m_n[0])
        if len(good) < 8:
            return None

        src = np.float32([kb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([ka[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, cfg.ransac_px)
        if H is None or mask is None:
            return None
        inliers = int(mask.sum())
        if inliers < cfg.min_inliers:
            return None

        det = abs(float(np.linalg.det(H[:2, :2])))
        if not (cfg.max_det_ratio[0] < det < cfg.max_det_ratio[1]):
            return None

        ga, gb = self._gray[i], self._gray[j]
        h_a, w_a = ga.shape
        warped = cv2.warpPerspective(gb, H, (w_a, h_a))
        valid = cv2.warpPerspective(np.ones(gb.shape, np.uint8), H, (w_a, h_a)) > 0
        overlap = float(valid.mean())
        if overlap < cfg.min_overlap or valid.sum() < 2000:
            return None

        x = ga[valid].astype(np.float64)
        y = warped[valid].astype(np.float64)
        if x.std() < cfg.min_region_std or y.std() < cfg.min_region_std:
            return None
        ncc = float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std()))
        if ncc < cfg.ncc_same_design:
            return None

        # Hue inside the overlap only. See module docstring.
        warped_rgb = cv2.warpPerspective(self._rgb[j], H, (w_a, h_a))
        ha = self._hue_hist(self._rgb[i], valid)
        hb = self._hue_hist(warped_rgb, valid)
        hue_overlap = (float(np.abs(ha - hb).sum())
                       if ha is not None and hb is not None else float("nan"))

        return {"inliers": inliers, "overlap": overlap, "ncc": ncc,
                "scale": det ** 0.5, "hue_overlap": hue_overlap}

    def release(self) -> None:
        self._rgb.clear()
        self._gray.clear()
        self._kp.clear()


# ------------------------------------------------------------------------------ grouping

def union_find_groups(n: int, edges: list[tuple[int, int]]) -> list[int]:
    """Connected components, so chains of overlapping tiles merge into one design.

    This is why union-find rather than pairwise grouping: if tile 1 overlaps tile 2 and tile
    2 overlaps tile 3, all three are the same saree even though 1 and 3 may share no pixels
    at all.
    """
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return [find(i) for i in range(n)]


def group_designs(paths: list[Path], phashes: list[str], geo_cfg: GeoConfig,
                  hue_cfg: HueConfig, prefilter_topk: int = 10,
                  phash_threshold: int = 4, use_dinov2: bool = True,
                  time_budget_s: float | None = None,
                  log=print) -> tuple[list[int], list[dict]]:
    """Full pipeline. Returns a component label per image and the verified edge records.

    `time_budget_s` time-boxes verification: candidates are checked in descending prefilter
    similarity so the most promising pairs are resolved first if the budget runs out.
    """
    n = len(paths)
    if n < 2:
        return list(range(n)), []

    candidates: set[tuple[int, int]] = set()
    sims: dict[tuple[int, int], float] = {}

    # All-pairs is affordable for a small source and needs no prefilter at all, which avoids
    # any recall loss. 165 images is 13,530 pairs; 412 images would be 84,666.
    all_pairs_budget = 20_000
    total_pairs = n * (n - 1) // 2
    if total_pairs <= all_pairs_budget:
        candidates = set(itertools.combinations(range(n), 2))
        log(f"[geo] {total_pairs:,} pairs is within the all-pairs budget, so no prefilter "
            f"is used and no candidate can be missed")
    else:
        emb = None
        if use_dinov2:
            try:
                t0 = time.time()
                log(f"[geo] embedding {n} images with zero-shot DINOv2 for the prefilter")
                emb = dinov2_embeddings(paths)
                log(f"[geo] DINOv2 embeddings in {time.time() - t0:.0f}s")
            except Exception as exc:
                log(f"[geo] DINOv2 unavailable ({type(exc).__name__}: {exc}); "
                    f"falling back to translation-invariant histogram descriptors")
                emb = None
        if emb is None:
            t0 = time.time()
            emb = histogram_descriptors(paths)
            log(f"[geo] histogram descriptors in {time.time() - t0:.0f}s")

        pre_pairs = topk_candidates(emb, k=prefilter_topk)
        for a, b in pre_pairs:
            sims[(a, b)] = float(emb[a] @ emb[b])
        candidates |= pre_pairs
        log(f"[geo] prefilter: {len(pre_pairs):,} candidate pairs "
            f"(versus {total_pairs:,} all-pairs)")

    ph_pairs = phash_candidates(phashes, threshold=phash_threshold)
    new = ph_pairs - candidates
    candidates |= ph_pairs
    for p in ph_pairs:
        sims.setdefault(p, 1.0)        # a pHash match is a strong prior, check it early
    log(f"[geo] pHash added {len(new):,} further candidates "
        f"({len(candidates):,} total, versus {n * (n - 1) // 2:,} all-pairs)")

    verifier = GeometricVerifier(paths, geo_cfg, hue_cfg)
    ordered = sorted(candidates, key=lambda p: -sims.get(p, 0.0))

    edges: list[tuple[int, int]] = []
    records: list[dict] = []
    t0 = time.time()
    checked = 0
    for a, b in ordered:
        if time_budget_s is not None and (time.time() - t0) > time_budget_s:
            log(f"[geo] TIME BUDGET reached after {checked:,}/{len(ordered):,} candidates. "
                f"Remaining pairs are unverified, which can only MISS merges, never invent "
                f"them, so splits stay safe but may contain avoidable duplicates.")
            break
        checked += 1
        r = verifier.verify(a, b)
        if r is not None:
            edges.append((a, b))
            records.append({"i": a, "j": b, "path_a": paths[a].as_posix(),
                            "path_b": paths[b].as_posix(), **r})
    verifier.release()
    log(f"[geo] verified {len(edges):,} same-design pairs from {checked:,} candidates "
        f"in {time.time() - t0:.0f}s")

    return union_find_groups(n, edges), records
