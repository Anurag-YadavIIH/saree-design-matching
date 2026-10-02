"""
Colour augmentation: the core of how this model learns colour invariance.

The project has zero real colourway pairs (see docs/DATA_CONTRACT.md), so every positive
pair is produced here. Two recolouring algorithms exist on purpose:

  kmeans_palette_swap   used for TRAINING only
  lab_hue_chroma_remap  used for VAL and TEST only

Holding the algorithm out matters. If eval used the same generator as training, a high score
would only prove the model learned to invert our own k-means swap, not that it understands
design identity. The palette banks are disjoint too.

Speed note, which drives the whole design here. Running k-means per image per epoch would
dominate training time. Instead the cluster assignment is computed ONCE per real image and
cached as a uint8 index map. A train-time recolour is then a lookup:

    new_rgb = palette[cluster_map]

followed by restoring the original luminance, so weave texture and shading survive the
swap instead of being flattened into blocks of solid colour. That last step is what keeps
the recoloured image a plausible textile rather than a posterised cartoon.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Cluster maps are cached at this resolution. Large enough to preserve motif structure,
# small enough that the cache stays modest and the lookup is fast.
CACHE_SIZE = 256


# --------------------------------------------------------------------------- palettes

def build_palette_bank(size: int, k_colors: int, seed: int) -> list[dict]:
    """Generate a bank of palettes, each a list of k RGB colours.

    Palettes are sampled in HSV so that each one is coherent rather than a random jumble:
    a base hue plus controlled spread, with a spread of lightness. Real saree colourways
    behave this way, built around one or two dominant hues.
    """
    rng = np.random.default_rng(seed)
    bank = []
    for i in range(size):
        base_hue = rng.uniform(0, 180)                 # OpenCV hue range is 0 to 180
        spread = rng.choice([8.0, 20.0, 45.0, 90.0])   # monochrome through complementary
        sat_lo, sat_hi = rng.uniform(60, 120), rng.uniform(150, 255)
        colors = []
        for j in range(k_colors):
            # Spread hues around the base, and walk lightness from dark to light so the
            # palette can represent both ground and motif.
            hue = (base_hue + rng.normal(0, spread)) % 180
            sat = rng.uniform(sat_lo, sat_hi)
            val = 30 + (205 * j / max(k_colors - 1, 1)) + rng.uniform(-20, 20)
            val = float(np.clip(val, 15, 250))
            hsv = np.uint8([[[hue, np.clip(sat, 0, 255), val]]])
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
            colors.append([int(c) for c in rgb])
        bank.append({"palette_id": f"pal_{i:03d}", "colors": colors})
    return bank


def split_palette_bank(bank: list[dict], counts: dict[str, int], seed: int) -> dict[str, list[dict]]:
    """Partition the bank into mutually disjoint train / val / test pools.

    Disjointness is the guarantee that an eval palette was never seen in training, so it
    is asserted here rather than left to convention.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(bank))
    out: dict[str, list[dict]] = {}
    cursor = 0
    for name in ("train", "val", "test"):
        n = counts[name]
        out[name] = [bank[i] for i in order[cursor:cursor + n]]
        cursor += n
    ids = [p["palette_id"] for pool in out.values() for p in pool]
    assert len(ids) == len(set(ids)), "palette banks overlap, which would leak eval palettes"
    return out


# ----------------------------------------------------------------------- cluster cache

def cache_path(cache_dir: Path, image_path: str, k: int) -> Path:
    """Cache filename keyed by path and k, so changing k cannot silently reuse stale maps."""
    h = hashlib.sha1(f"{image_path}|k{k}|{CACHE_SIZE}".encode()).hexdigest()[:16]
    return cache_dir / f"{h}.npz"


def build_cluster_map(rgb: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Run k-means once on an image's colours.

    Returns the per-pixel cluster index map (uint8) and the cluster centres in RGB. This is
    the expensive step, which is why it is cached and never repeated during training.
    """
    small = cv2.resize(rgb, (CACHE_SIZE, CACHE_SIZE), interpolation=cv2.INTER_AREA)
    # Cluster in LAB: Euclidean distance there tracks perceived colour difference far
    # better than it does in RGB, so the clusters follow what a human would call a colour.
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    cv2.setRNGSeed(int(seed))
    _, labels, centers_lab = cv2.kmeans(lab, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers_rgb = cv2.cvtColor(centers_lab.astype(np.uint8).reshape(-1, 1, 3),
                               cv2.COLOR_LAB2RGB).reshape(-1, 3)
    return labels.reshape(CACHE_SIZE, CACHE_SIZE).astype(np.uint8), centers_rgb


def load_or_build_cluster_map(image_path: Path, rel_path: str, cache_dir: Path,
                              k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    p = cache_path(cache_dir, rel_path, k)
    if p.exists():
        try:
            d = np.load(p)
            return d["labels"], d["centers"]
        except Exception:
            pass  # a corrupt cache entry is simply rebuilt
    rgb = read_rgb(image_path)
    labels, centers = build_cluster_map(rgb, k, seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, labels=labels, centers=centers)
    return labels, centers


def read_rgb(path: Path) -> np.ndarray:
    """Read an image as RGB. Single conversion point, see docs/DATA_CONTRACT.md."""
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


# ------------------------------------------------------------------------- recolouring

def _restore_luminance(recolored: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Give `recolored` the lightness channel of `original`.

    Without this, a palette swap replaces every pixel of a cluster with one flat colour and
    destroys the weave texture, which is precisely the signal the model must learn. Keeping
    the original L channel means only chroma changes, so the design survives intact.
    """
    lab_new = cv2.cvtColor(recolored, cv2.COLOR_RGB2LAB)
    lab_old = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
    lab_new[..., 0] = lab_old[..., 0]
    return cv2.cvtColor(lab_new, cv2.COLOR_LAB2RGB)


def kmeans_palette_swap(rgb: np.ndarray, cluster_map: np.ndarray, palette: list[list[int]],
                        rng: np.random.Generator) -> np.ndarray:
    """TRAIN-time recolour: map each colour cluster to a palette entry, then restore luminance.

    The cluster map is cached at CACHE_SIZE and resized with nearest-neighbour so cluster
    boundaries stay hard; interpolating indices would invent clusters that do not exist.
    """
    h, w = rgb.shape[:2]
    cmap = cv2.resize(cluster_map, (w, h), interpolation=cv2.INTER_NEAREST)
    pal = np.asarray(palette, dtype=np.uint8)
    k = int(cmap.max()) + 1
    # Shuffle which cluster receives which palette colour, so one image plus one palette
    # yields several distinct colourways instead of a single deterministic one.
    assign = rng.permutation(len(pal))[:k]
    if len(assign) < k:                      # palette smaller than cluster count
        assign = np.resize(assign, k)
    lut = pal[assign]
    recolored = lut[cmap]
    return _restore_luminance(recolored, rgb)


def lab_hue_chroma_remap(rgb: np.ndarray, palette: list[list[int]],
                         rng: np.random.Generator) -> np.ndarray:
    """EVAL-time recolour, deliberately a different algorithm from the training one.

    Rather than quantising to clusters, this rotates hue globally and rescales chroma in LAB
    towards the palette's dominant hue. It produces smooth, continuous colour shifts, so a
    model that merely learned to undo k-means blocking will not generalise to it. Luminance
    is untouched, so texture survives here too.
    """
    pal = np.asarray(palette, dtype=np.uint8).reshape(-1, 1, 3)
    pal_lab = cv2.cvtColor(pal, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    target_a = float(pal_lab[:, 1].mean())
    target_b = float(pal_lab[:, 2].mean())

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1] - 128.0, lab[..., 2] - 128.0

    theta = rng.uniform(0, 2 * np.pi)
    chroma_gain = rng.uniform(0.7, 1.4)
    pull = rng.uniform(0.35, 0.75)      # how strongly colours move towards the palette

    a_rot = (a * np.cos(theta) - b * np.sin(theta)) * chroma_gain
    b_rot = (a * np.sin(theta) + b * np.cos(theta)) * chroma_gain
    a_new = (1 - pull) * a_rot + pull * (target_a - 128.0)
    b_new = (1 - pull) * b_rot + pull * (target_b - 128.0)

    lab[..., 1] = np.clip(a_new + 128.0, 0, 255)
    lab[..., 2] = np.clip(b_new + 128.0, 0, 255)
    lab[..., 0] = L
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def tonal_remap(rgb: np.ndarray, cluster_map: np.ndarray, palette: list[list[int]],
                rng: np.random.Generator, invert_prob: float = 0.5) -> tuple[np.ndarray, bool]:
    """Recolour that changes LIGHTNESS too, not only chroma.

    Why it exists. Both other generators restore the original L channel, so a grayscale image
    is identical before and after recolouring: grayscale input is invariant to them BY
    CONSTRUCTION. That is why zero-shot DINOv2 on grayscale scored 0.913 Rank-1 and why the
    RGB-versus-gray gap was only 2 points. Real colourways often change or invert light and
    dark (dark motif on a light ground becomes light on dark), which this generator models.

    Per k-means cluster c, the new colour takes the palette entry's own L, a and b, and texture
    survives as the deviation from the cluster mean:

        new_L = palette_L[c] + (L - mean_L[c])

    Clusters are matched to palette entries by lightness rank, and with probability
    `invert_prob` that order is reversed, so the darkest region becomes the lightest.
    Returns the image and whether the order was inverted.
    """
    h, w = rgb.shape[:2]
    cmap = cv2.resize(cluster_map, (w, h), interpolation=cv2.INTER_NEAREST)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]

    present = [c for c in range(int(cmap.max()) + 1) if (cmap == c).any()]
    mean_L = {c: float(L[cmap == c].mean()) for c in present}

    pal_lab = cv2.cvtColor(np.asarray(palette, dtype=np.uint8).reshape(-1, 1, 3),
                           cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    idx = rng.choice(len(pal_lab), size=len(present), replace=len(present) > len(pal_lab))
    chosen = pal_lab[idx]
    chosen = chosen[np.argsort(chosen[:, 0])]               # palette entries, dark to light
    clusters_dark_to_light = sorted(present, key=lambda c: mean_L[c])
    inverted = bool(rng.random() < invert_prob)
    if inverted:
        chosen = chosen[::-1]                                # darkest region gets lightest colour

    out = np.empty_like(lab)
    for c, col in zip(clusters_dark_to_light, chosen):
        m = cmap == c
        out[..., 0][m] = col[0] + (L[m] - mean_L[c])
        out[..., 1][m] = col[1]
        out[..., 2][m] = col[2]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB), inverted


# -------------------------------------------------------------------------- geometry

@dataclass
class GeometryConfig:
    """Realistic capture variation. No flips: many saree motifs are directional, and a
    mirrored motif is arguably a different design, so flipping would teach a false
    invariance."""
    crop_scale: tuple[float, float] = (0.70, 1.00)
    rotation_deg: float = 15.0
    perspective: float = 0.05
    scale: tuple[float, float] = (0.8, 1.2)
    blur_sigma: tuple[float, float] = (0.0, 1.0)
    jpeg_quality: tuple[int, int] = (60, 90)
    hflip: bool = False
    vflip: bool = False
    out_size: int = 224

    @classmethod
    def from_dict(cls, d: dict, out_size: int = 224) -> "GeometryConfig":
        return cls(
            crop_scale=tuple(d.get("crop_scale", (0.70, 1.00))),
            rotation_deg=float(d.get("rotation_deg", 15.0)),
            perspective=float(d.get("perspective", 0.05)),
            scale=tuple(d.get("scale", (0.8, 1.2))),
            blur_sigma=tuple(d.get("blur_sigma", (0.0, 1.0))),
            jpeg_quality=tuple(d.get("jpeg_quality", (60, 90))),
            hflip=bool(d.get("hflip", False)),
            vflip=bool(d.get("vflip", False)),
            out_size=out_size,
        )


def apply_geometry(rgb: np.ndarray, cfg: GeometryConfig,
                   rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Apply crop, rotation, perspective, scale, blur and JPEG recompression.

    Returns the image and the exact parameters used, which go into the manifest's
    `geo_params` so a surprising eval result can be traced back to its perturbation.
    """
    h, w = rgb.shape[:2]
    params: dict = {}

    # Random resized crop. The scale range reaches down to 0.70 for eval (and lower for
    # training) because real queries may show a different REGION of a saree, which the
    # corpus demonstrates: its real pairs are overlapping crops of one photograph.
    area = rng.uniform(*cfg.crop_scale)
    aspect = rng.uniform(0.85, 1.18)
    ch = int(round(np.sqrt(area / aspect) * h))
    cw = int(round(np.sqrt(area * aspect) * w))
    ch, cw = min(ch, h), min(cw, w)
    y0 = int(rng.integers(0, max(h - ch, 0) + 1))
    x0 = int(rng.integers(0, max(w - cw, 0) + 1))
    img = rgb[y0:y0 + ch, x0:x0 + cw]
    params["crop"] = [x0, y0, cw, ch]

    # Rotation about the centre, with reflection padding so no black corners appear. Black
    # corners would be a trivial cue the model could latch onto instead of the motif.
    ang = float(rng.uniform(-cfg.rotation_deg, cfg.rotation_deg))
    sc = float(rng.uniform(*cfg.scale))
    hh, ww = img.shape[:2]
    M = cv2.getRotationMatrix2D((ww / 2, hh / 2), ang, sc)
    img = cv2.warpAffine(img, M, (ww, hh), borderMode=cv2.BORDER_REFLECT_101)
    params["rotation_deg"] = round(ang, 3)
    params["scale"] = round(sc, 3)

    if cfg.perspective > 0:
        d = cfg.perspective
        src = np.float32([[0, 0], [ww, 0], [ww, hh], [0, hh]])
        jitter = rng.uniform(-d, d, size=(4, 2)) * np.float32([ww, hh])
        P = cv2.getPerspectiveTransform(src, (src + jitter).astype(np.float32))
        img = cv2.warpPerspective(img, P, (ww, hh), borderMode=cv2.BORDER_REFLECT_101)
        params["perspective"] = [[round(float(v), 4) for v in row] for row in jitter]

    if cfg.hflip and rng.random() < 0.5:
        img = img[:, ::-1]
        params["hflip"] = True
    if cfg.vflip and rng.random() < 0.5:
        img = img[::-1]
        params["vflip"] = True

    img = cv2.resize(img, (cfg.out_size, cfg.out_size), interpolation=cv2.INTER_AREA)

    sigma = float(rng.uniform(*cfg.blur_sigma))
    if sigma > 0.05:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
        params["blur_sigma"] = round(sigma, 3)

    q = int(rng.integers(cfg.jpeg_quality[0], cfg.jpeg_quality[1] + 1))
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        img = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        params["jpeg_quality"] = q

    return np.ascontiguousarray(img), params


# ------------------------------------------------------------------- unified entry point

@dataclass
class Recolorer:
    """Applies a named recolour method plus geometry, and reports what it did.

    One entry point for both the build-time eval generator and the train-time transform, so
    the two cannot drift apart.
    """
    method: str
    palettes: list[dict]
    geometry: GeometryConfig
    k_colors: int = 6
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    invert_prob: float = 0.5           # used by tonal_remap only

    def __call__(self, image_path: Path, rel_path: str, seed: int,
                 palette_id: str | None = None) -> tuple[np.ndarray, dict]:
        rng = np.random.default_rng(seed)
        rgb = read_rgb(image_path)

        if palette_id is None:
            pal = self.palettes[int(rng.integers(len(self.palettes)))]
        else:
            matches = [p for p in self.palettes if p["palette_id"] == palette_id]
            if not matches:
                raise KeyError(f"palette {palette_id} is not in this pool")
            pal = matches[0]

        if self.method == "kmeans_palette_swap":
            cmap, _ = load_or_build_cluster_map(image_path, rel_path, self.cache_dir,
                                                self.k_colors, seed=0)
            recolored = kmeans_palette_swap(rgb, cmap, pal["colors"], rng)
        elif self.method == "lab_hue_chroma_remap":
            recolored = lab_hue_chroma_remap(rgb, pal["colors"], rng)
        elif self.method == "tonal_remap":
            cmap, _ = load_or_build_cluster_map(image_path, rel_path, self.cache_dir,
                                                self.k_colors, seed=0)
            recolored, inverted = tonal_remap(rgb, cmap, pal["colors"], rng,
                                              invert_prob=self.invert_prob)
        else:
            raise ValueError(f"unknown recolor method: {self.method}")

        out, geo = apply_geometry(recolored, self.geometry, rng)
        if self.method == "tonal_remap":
            geo["tonal_inverted"] = inverted
        meta = {
            "recolor_method": self.method,
            "palette_id": pal["palette_id"],
            "recolor_seed": int(seed),
            "geo_params": json.dumps(geo, separators=(",", ":")),
        }
        return out, meta
