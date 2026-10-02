# Color-Invariant Saree Design Recognition

DeepLure AIE-CASE submission by Anurag Yadav.

Identify a saree by its surface design regardless of colour palette: "face recognition for
textiles". The same motif in a different palette must match; different motifs in an identical
palette must not.

> Results are filled in after the Kaggle run. See `outputs/results.md` once generated.

## Problem framing

Two tasks over one embedding:

* **Identification**: rank a gallery of designs by similarity to a query.
* **Verification**: decide whether two images carry the same design.

The core requirement has two halves, measured separately:

* **Stress A**, same design in a new palette, **must** match.
* **Stress B**, different designs in the same palette, **must not** match.

## Data and splits

### What the data actually contains

Nothing in either source labels design identity. **The Drive corpus has no real colourway
pairs.** The Kaggle catalogue does: at least 33 separately photographed same-print,
different-colour pairs, found late by mining (D-40, D-42) after two earlier probes missed them
because they assumed a same-design pair shares pixels or layout. Training and the main eval
protocol use synthetic colour positives; the real Kaggle pairs give a separate, smaller
verification eval.

| Source | Files | Designs | Notes |
|---|---|---|---|
| DeepLure Drive corpus | 165 | 95 | Proprietary. 28 designs appear as several overlapping crops |
| Indian Fabric Patterns v1 | 1,468 | 411 | Public, MIT. ~3.6x inflated by augmentation duplicates |

### How identity was established

Design identity is constructed from image content, in four stages: collapse same-pixel
duplicates with a perceptual hash, choose the least noisy copy as representative, verify
same-design pairs geometrically (ORB keypoints, RANSAC homography, then normalised
cross-correlation over the aligned overlap), and take connected components so chains of
overlapping tiles merge into one design.

The key finding: **perceptual hashing is blind to translation.** Two crops of the same saree
at different offsets scored pHash distances of 28 to 38 (random level) while their pixels
agreed at NCC 0.97 to 0.99 once aligned. A pHash-only pipeline would have split 28 Drive
designs across train and test, putting the same physical saree on both sides.

### Splits

By design, never by image, stratified by source and craft family: 0.15 val and 0.20 test
overall, with the Drive test fraction raised to 0.40 because Drive is the client's target
domain. All 28 real multi-view Drive designs are in test. The vendor's own train/valid/test
split was discarded because 88 duplicate clusters straddle its boundaries.

A post-split leakage audit embeds every test image and reports its nearest train neighbour,
written to `outputs/leakage_audit.csv` for review. Designs that are probably separate
photographs of one saree (high similarity plus a shared distinctive filename prefix) are
constrained to the same side of the split.

## Approach

* **Backbone**: DINOv2 ViT-S/14, last 4 blocks fine-tuned, the rest frozen.
* **Head**: `Linear(384,512) - BN - ReLU - Linear(512,128)`, L2-normalised 128-d embedding.
* **Loss**: supervised contrastive (SupCon, multi-positive), temperature 0.07.
* **Batches**: 32 designs x 4 views. With probability 0.3, a group of 4 different designs is
  forced onto one palette, so colour is useless inside that group and only the motif separates
  them.
* **Colour augmentation**: k-means palette swap with the original luminance restored, so the
  weave texture survives. Cluster maps are precomputed once per image.

## Evaluation protocol

* **Gallery**: one real representative per test design.
* **Queries**: 5 per design, recoloured with a **held-out palette bank AND a held-out recolour
  algorithm** (LAB hue and chroma remap, never seen in training), plus realistic geometry:
  crop 70 to 100%, rotation +/-15 degrees, mild perspective, scale, blur, JPEG recompression.
  No flips, because many saree motifs are directional.
* **Identification**: Rank-1, Rank-5, mAP. 95% confidence intervals from 1000 bootstrap
  resamples over **designs**, not queries, because sibling queries are correlated.
* **Verification**: ROC-AUC, EER, TAR at FAR=1e-3, with the threshold chosen on val and
  frozen for test.
* **Stress B**: tier 1 same palette; tier 2 same palette and same craft family.
* **Baselines**: grayscale pHash, RGB colour histogram, zero-shot DINOv2 on RGB and on
  grayscale input.

### The real pairs, stated precisely

The corpus's only real positive pairs are **overlapping crops of one photograph** (scale 1.0,
about 25% overlap). They therefore test **region and framing invariance at fixed lighting and
scale**. They are **not** re-photography of a physical saree, and **they do not vary
palette**. They are reported as a separate table with three query kinds, of which
`real_view_recolored` (a real framing change plus a held-out palette) is the closest proxy
to the true task this corpus permits.

## Results

### Baselines, before and after post-hoc leakage exclusion

Gray zero-shot DINOv2 is the bar to beat. Checkpoints are judged ONLY against the
post-exclusion bars (`DECISIONS.md` D-34 for the rule, D-43 for the numbers). 95% CIs from 1000
bootstrap resamples over designs.

| method | test set | main Rank-1 | main TAR@1e-3 | tonal Rank-1 | real_view_recolored Rank-1 |
|---|---|---|---|---|---|
| grayscale pHash | pre, 120 designs | 0.448 [0.402, 0.492] | 0.353 [0.312, 0.397] | 0.190 [0.157, 0.222] | 0.028 [0.000, 0.066] |
| grayscale pHash | post, 93 designs | 0.454 [0.404, 0.503] | 0.342 [0.299, 0.391] | 0.209 [0.176, 0.243] | 0.023 [0.000, 0.073] |
| RGB colour histogram | pre, 120 | 0.090 [0.063, 0.118] | 0.053 [0.032, 0.077] | 0.010 [0.002, 0.022] | 0.127 [0.045, 0.203] |
| RGB colour histogram | post, 93 | 0.099 [0.067, 0.131] | 0.052 [0.030, 0.073] | 0.013 [0.002, 0.030] | 0.159 [0.062, 0.270] |
| DINOv2 zero-shot RGB | pre, 120 | 0.892 [0.853, 0.925] | 0.637 [0.583, 0.690] | 0.573 [0.517, 0.632] | 0.746 [0.623, 0.855] |
| DINOv2 zero-shot RGB | post, 93 | 0.901 [0.862, 0.940] | 0.583 [0.518, 0.647] | 0.559 [0.495, 0.628] | 0.750 [0.579, 0.877] |
| DINOv2 zero-shot gray | pre, 120 | 0.913 [0.878, 0.943] | 0.667 [0.618, 0.718] | 0.638 [0.583, 0.692] | 0.761 [0.657, 0.855] |
| **DINOv2 zero-shot gray** | **post, 93** | **0.912 [0.873, 0.948]** | **0.609 [0.546, 0.673]** | **0.645 [0.581, 0.714]** | **0.864 [0.744, 0.944]** |

Post-exclusion success bars: tonal Rank-1 > 0.714, or main TAR > 0.673, or
real_view_recolored > 0.944; and main Rank-1 >= 0.873. The real-pairs set shrank to 15 designs
after exclusion, which makes its bar close to unreachable.

### Real colourway verification (Kaggle, 33 judged pairs)

| method | ROC-AUC | TAR@FAR=1e-2 |
|---|---|---|
| grayscale pHash | 0.686 [0.562, 0.820] | 0.273 [0.121, 0.424] |
| RGB colour histogram | 0.597 [0.473, 0.710] | 0.030 [0.000, 0.091] |
| DINOv2 zero-shot RGB | 0.998 [0.996, 0.999] | 0.939 [0.848, 1.000] |
| DINOv2 zero-shot gray | 0.999 [0.999, 1.000] | 1.000 [1.000, 1.000] |

**Read with care.** Pairs were mined from gray zero-shot DINOv2's own nearest neighbours, so
that model scores them highly by construction; this eval cannot rank it against correlated
models. It does show pHash and colour histograms failing on real colourways. 10 of 33 pairs
were trained as negatives, so fine-tuned models are underestimated here (D-40).

### Trained models

To be filled from the Kaggle checkpoints, judged against the post-exclusion bars.

## Efficiency

Architecture numbers (device independent; latency is measured on the Kaggle T4 by
`scripts/profile_model.py` and reported in `outputs/profile.md`):

| quantity | value |
|---|---|
| parameters | 22.3M total, 7.4M fine-tuned (last 4 blocks, final norm, head), 0.26M head |
| compute | **11.0 GFLOPs = 5.5 GMACs** per 224x224 image |
| FLOP convention | 1 multiply-add = 2 FLOPs, as counted by torch `FlopCounterMode` (measured on a probe layer). fvcore counts multiply-adds, so it would print about 5.5 for the same model |
| tokens | **257 = 256 patches (16x16 grid of 14 px) + 1 CLS**, no register tokens |
| embedding | 128-d: 512 B fp32, 256 B fp16; 488 MB per million gallery designs at fp32, 3x smaller than the 384-d DINOv2 CLS vector |

Gray input costs the same as RGB: the single channel is replicated to three.

Latency at batch 1 and 32 (fp32 and fp16): to be copied from the T4 run.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r requirements.txt
python -m src.data.build_manifest                   # manifest, synthetic queries, validation
python scripts/build_cache.py                       # k-means cluster maps, once
python scripts/train.py --smoke                     # 2 minutes, loss must decrease
python scripts/train.py                             # time boxed run
python scripts/evaluate.py                          # baselines plus trained model
python scripts/profile_model.py                     # params, FLOPs, latency (run on the T4)
```

On Kaggle, use `notebooks/kaggle_train.ipynb`; see "Running on Kaggle" below.

### Running on Kaggle

1. **Data dataset.** Kaggle, Datasets, New Dataset. Upload a folder containing
   `Google_drive_data/` and `Kaggle_data/` at its top level, exactly as under `data/` here.
   Set visibility to **Private** before creating it. Name it `saree-reid-data`.
2. **Code dataset.** A second dataset, `saree-reid-code`, containing this repo **without**
   `data/`, `outputs/` and `.venv/`. Alternatively push the repo to GitHub and set `REPO_URL`
   in the notebook instead.
3. **Notebook.** Import `notebooks/kaggle_train.ipynb`. Settings: Accelerator **GPU T4 x1**,
   Internet **On** (DINOv2 downloads from `torch.hub`). Add both datasets as inputs.
4. Run all. Download only `outputs/results.md`, `results.json`, `train_log.csv` and
   `run/train_summary.json`.

## Pretrained checkpoints and external data used

| Item | Source | Licence |
|---|---|---|
| DINOv2 ViT-S/14 | `torch.hub` `facebookresearch/dinov2`, weights `dinov2_vits14_pretrain.pth` | Apache 2.0 |
| Indian Fabric Patterns v1 | Roboflow Universe, `div-szivu/indian-fabric-patterns`, exported 2024-05-26 | MIT |

**Preprocessing applied by Roboflow, not by us:** auto-orientation with EXIF stripping, resize
of every image to **640x640 by stretching** (so aspect ratio is distorted and does not reflect
the true drape), and three salt-and-pepper noise variants per source image. Drive images are
untouched, 226 to 1512 px and mostly square. Cross-source comparisons must account for this
asymmetry.

## Limitations and extensions

* **Few real colourway pairs.** None in Drive; 33 judged in Kaggle (top 60 of 788 mined
  candidates reviewed). Training colour positives are synthetic. Holding out both the palette
  bank and the recolour algorithm guards against the model simply learning to invert our own
  generator, but it cannot fully substitute for real dyed variants of one design. The single
  most valuable addition would be the client's catalogue colourway groupings.
* **Real pairs vary framing, not palette** (see above).
* **Undetectable leakage.** Two non-overlapping tiles of one saree share no pixels and no
  keypoints, so no content-based method can link them. Some same-design leakage across splits
  may therefore remain undetected. The leakage audit bounds this risk but cannot eliminate it.
* **Leakage audit outcome.** Of the 20 test designs most similar to a train design, none is pixel-verified as the same saree; 5 have very high similarity but cannot be verified by content and were reviewed by eye. Any judged to be the same saree are excluded post hoc from test (`configs/eval_exclude.yaml`), and the results state how many.
* **Training labels contain undetected duplicates.** Real colourway twins in Kaggle and
  separate photographs of one saree in Drive carry different design_ids, so SupCon saw some
  same-design pairs as NEGATIVES. This works directly against the colour-invariance objective
  and likely understates what the method could achieve with clean labels.
* **Post-hoc test exclusions.** 27 test designs (19 of 38 Drive, 8 Kaggle) were judged to
  leak and are excluded from every reported test number (`configs/eval_exclude.yaml`).
* **Small test set.** 119 test designs, 38 of them from Drive, so confidence intervals are
  wide. They are reported rather than hidden.
* **Source asymmetry.** Kaggle images are stretched and noisy; Drive images are not.
