# Color-Invariant Saree Design Recognition

DeepLure AIE-CASE submission by Anurag Yadav.

Developed with Claude Code as a coding assistant; every design decision is mine and logged in DECISIONS.md, with AI workflow files in ai_workflow/.

## Problem framing

Identify a saree by its surface design regardless of colour palette: face recognition for
textiles. One embedding serves two tasks:

* **Identification**: rank a gallery of designs by similarity to a query image.
* **Verification**: decide whether two images carry the same design.

The core requirement has two halves, measured separately:

* **Stress A**: the same design in a new palette **must** match.
* **Stress B**: different designs in the same palette **must not** match.

## Data and provenance

| Source | Files | Designs | Licence and handling |
|---|---|---|---|
| DeepLure Drive corpus | 165 | 95 | Proprietary. **Not redistributed**: never committed, uploaded publicly, or copied out of `data/` |
| Indian Fabric Patterns v1 | 1,468 | 411 | **MIT.** Downloaded from Kaggle; originally a Roboflow Universe export, `div-szivu/indian-fabric-patterns` (https://universe.roboflow.com/div-szivu/indian-fabric-patterns), exported 2024-05-26 |

Neither source labels design identity. The Kaggle folders are craft families (Banarasi,
Bandhani, Ikat, Pichwai), which stratify the split but are not identities. The Drive corpus is
a flat folder of opaque numeric filenames with no metadata.

**Preprocessing applied by Roboflow, not by us:** auto-orientation with EXIF stripping, every
image **stretched to 640x640** (aspect ratio distorted), and three salt-and-pepper noise
variants per source image. Drive images are untouched: 226 to 1512 px, mostly square.

## What the data taught us

Four findings shaped everything downstream. Each is logged with evidence in `DECISIONS.md`.

**1. Re-cropped views that perceptual hashing cannot see (D-03).** 27 Drive designs appear as
several overlapping crops of one photograph. A translation offset leaves local pixels intact
but completely changes the DCT that pHash is built from: these pairs score pHash distance 28
to 38 (random level) while their pixels agree at NCC 0.97 to 0.99 once aligned. A pHash-only
pipeline would have put the same saree in train and test. Identity is therefore established
geometrically (ORB keypoints, RANSAC homography, normalised cross-correlation over the aligned
overlap), and union-find merges chains of overlapping tiles. Drive: 165 files, 95 designs.

**2. Roboflow duplicates (D-04, D-08, D-09).** The Kaggle export is inflated 3.6x: 1,468 files
collapse to 412 duplicate groups and 411 designs. The cluster count plateaus across pHash
thresholds 4 to 8, so the threshold is not load-bearing. Filenames are not a safe key: generic
stems such as `image21` collide across unrelated images, and one image was saved under two
unrelated names. The vendor's own train/valid/test split was discarded because 88 duplicate
clusters straddle its boundaries.

**3. Same-saree leakage (D-38, D-41).** Separate photographs of one saree share no pixels, so
no content method links them. After splitting, a leakage audit ranked every test image's
nearest train image; a human review judged **19 of the 38 Drive test designs** to be the same
saree as a train design, plus **8 Kaggle test designs** with a real colourway twin in train.
These 27 designs are excluded from every reported test number (`configs/eval_exclude.yaml`),
leaving **93 test designs**. No retraining was needed: removing the test side removes the leak.

**4. Real colourway pairs exist in Kaggle (D-40, D-42).** The Drive corpus has none. The
Kaggle catalogue does: separately photographed products in the same print and different
colours. Two early probes missed them because each assumed a same-design pair shares pixels or
layout. Mining with grayscale DINOv2 neighbours plus a large hue difference, then judging
contact sheets, gave **33 confirmed pairs** (top 60 of 788 candidates reviewed). The most
common false match was different Ikat prints photographed on the same mannequin.

### Splits

By design, never by image, stratified by source and craft family, with the Drive test fraction
raised to 0.40 because Drive is the client's target domain. Designs that may be separate photos
of one saree are constrained to the same side of the split.

| split | designs | Drive | Kaggle |
|---|---|---|---|
| train | 310 | 43 | 267 |
| val | 76 | 14 | 62 |
| test (as built) | 120 | 38 | 82 |
| test (after post-hoc exclusion) | **93** | **19** | **74** |

## Approach

* **Backbone**: DINOv2 ViT-S/14, last 4 transformer blocks and the final norm fine-tuned, the
  rest frozen. Self-supervised features encode texture and structure rather than ImageNet
  category boundaries, a good prior for textiles.
* **Head**: `Linear(384, 512) - BatchNorm - ReLU - Linear(512, 128)`, L2-normalised to a 128-d
  embedding. Cosine similarity is then a dot product.
* **Input**: 224x224, ImageNet normalisation. Two variants trained: RGB input and grayscale input
  (one channel replicated to three, same cost).
* **Loss**: supervised contrastive (SupCon, multi-positive), temperature 0.07. Chosen over
  ArcFace because most designs have one real image: positives come from augmentation, and
  SupCon learns from in-batch pairs with no per-class centre to estimate.
* **Sampling**: PK batches of 32 designs x 4 views, so every anchor has 3 positives. With
  probability 0.3 a group of 4 different designs is forced onto one palette: inside that group
  colour is useless and only the motif separates them. This attacks the colour shortcut directly.
* **Colour augmentation**: k-means palette swap with the original luminance restored, so weave
  texture survives (grayscale correlation with the original 0.98). With probability 0.5 a tonal
  remap instead changes lightness too and inverts dark and light half the time, because real
  colourways often do. Cluster maps are computed once per image and cached, so a recolour is a
  table lookup.
* **Geometry**: random resized crop at scale 0.5 to 1.0 (queries can show a different region of
  a saree), rotation up to 15 degrees, mild perspective, scale, blur, JPEG recompression. No
  flips, because many saree motifs are directional.
* **Optimisation**: AdamW, learning rate 1e-3 for the head and 2e-5 for the backbone, weight
  decay 0.05, cosine schedule with 5% warmup, mixed precision, gradient clipping at 1.0, seed 42,
  time boxed per run.

## Evaluation protocol

* **Gallery**: one real representative image per test design.
* **Queries**, all generated with palettes from a bank never used in training:
  * `synthetic_recolor` (5 per design): a **different recolour algorithm** from training (LAB
    hue and chroma remap) plus realistic geometry. The main protocol.
  * `synthetic_tonal` (5 per design): lightness changes and dark and light may invert. The only
    synthetic queries on which grayscale input is not invariant by construction.
  * `real_view` and `real_view_recolored`: for designs with several real crops, another real crop
    as-is, and recoloured. These are **overlapping crops of one photograph** (scale 1.0, about
    25% overlap), so they test region and framing invariance at fixed lighting and scale. They
    are not re-photography and they do not vary palette.
* **Identification**: Rank-1, Rank-5, mAP.
* **Verification**: ROC-AUC, EER, TAR at FAR = 1e-3, with the threshold chosen on val and frozen
  for test.
* **Stress B**: tier 1 same palette; tier 2 same palette and same craft family; tier 3 same
  tonal palette.
* **Real colourway verification**: the 33 mined Kaggle pairs against 21,120 same-family
  negatives, ROC-AUC and TAR at FAR = 1e-2.
* **Confidence intervals**: 95%, 1000 bootstrap resamples over **designs**, not queries,
  because the queries of one design are correlated.
* **Baselines**: grayscale pHash, RGB colour histogram (expected to fail, which shows the
  benchmark punishes colour shortcuts), zero-shot DINOv2 on RGB input and on grayscale input.

## Pre-registered success criteria

Fixed in `DECISIONS.md` before any checkpoint was evaluated. The rule is D-34; the numeric bars
are D-43, derived on the post-exclusion test set.

**Reference**: zero-shot DINOv2 with grayscale input. It beats RGB input on every metric, so
feeding grayscale is the trivial colour-invariance fix that any fine-tuning must improve on.

**Success requires both:**

1. Beat gray zero-shot outside its 95% CI on at least one of: tonal Rank-1 > **0.714**; main
   TAR@FAR=1e-3 > **0.673**; `real_view_recolored` Rank-1 > **0.944**.
2. Main Rank-1 at or above **0.873**, the lower bound of gray zero-shot's CI.

**If it fails**: report that plainly and recommend gray zero-shot DINOv2 plus a projection.

"Outside the CI" rather than "higher" because with 93 test designs the intervals are 6 to 9
points wide. After exclusion the real-pairs set holds only 15 designs, so its bar is close to
unreachable in practice.

### Baselines, before and after post-hoc exclusion

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

The colour histogram scores below chance on stress B tier 3 (ROC-AUC 0.21): it prefers a
different design in the same palette over the true match, the exact failure the brief warns
about. Gray zero-shot main TAR fell from 0.667 to 0.609 after exclusion, because the leaked
designs were easy cases with a near-twin in train.

**Real colourway verification** (33 Kaggle pairs): pHash ROC-AUC 0.686, colour histogram 0.597,
DINOv2 RGB 0.998, DINOv2 gray 0.999. Read with care: the pairs were mined from gray DINOv2's own
nearest neighbours, so that model scores them highly by construction, and 10 of the 33 pairs were
trained as negatives, so fine-tuned models are underestimated here.

## Results

<!-- RESULTS -->

## Efficiency

Hardware-independent numbers, measured by `scripts/profile_model.py`:

| quantity | value |
|---|---|
| parameters | 22.3M total; 7.4M fine-tuned (last 4 blocks, final norm, head); 0.26M head |
| compute | **11.0 GFLOPs = 5.5 GMACs** per 224x224 image |
| FLOP convention | 1 multiply-add = 2 FLOPs, as counted by torch `FlopCounterMode` (measured on a probe layer). fvcore counts multiply-adds, so it would report about 5.5 for the same model |
| tokens | **257 = 256 patches (16x16 grid of 14 px) + 1 CLS**, no register tokens |
| embedding | 128-d: 512 B fp32, 256 B fp16; 488 MB per million gallery designs at fp32, 3x smaller than the 384-d DINOv2 CLS vector |

Gray input costs the same as RGB, since the single channel is replicated to three.

Latency on the Kaggle T4, batch 1 and 32, fp32 and fp16:

<!-- EFFICIENCY -->

## Reproduce

### On Kaggle (the reported run)

1. Create a **private** Kaggle dataset from this repository without `data/`, `outputs/` and
   `.venv/`.
2. Create a **private** dataset containing `Kaggle_data/`, and attach your own copy of
   `Google_drive_data/` if you hold it. The Drive corpus is not distributed with this repository.
3. Import `notebooks/kaggle_train.ipynb`. Settings: accelerator **GPU T4 x1**, internet **on**
   (DINOv2 downloads from torch.hub). Attach both datasets; the notebook finds the folders
   automatically at any depth.
4. Run all, about 70 minutes: build the manifest, cache k-means maps, smoke test, train the RGB
   and gray variants (time boxed), evaluate, profile on the T4, then a cleanup cell deletes
   every derivative of the Drive corpus and verifies the deletion. Keep the notebook private:
   the checkpoints are themselves derived from Drive images.

### Locally

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r requirements.txt
python -m src.data.build_manifest                   # manifest, synthetic queries, validation
python scripts/build_cache.py                       # k-means cluster maps, once
python scripts/train.py --smoke                     # 2 minutes, loss must decrease
python scripts/train.py --input gray --run-name run_gray
python scripts/evaluate.py --checkpoints outputs/run_rgb/best.pt outputs/run_gray/best.pt
python scripts/eval_colourway.py                    # real colourway verification
python scripts/profile_model.py                     # params, FLOPs, latency (run on the T4)
python -m pytest tests/ -q                          # 13 tests, offline, synthetic fixture
```

Builds are deterministic: a test builds the manifest twice and asserts identical identities,
splits and recolour seeds, so a checkpoint trained on Kaggle can be evaluated locally against
the same test set.

## Pretrained checkpoints and external data

| item | source | licence |
|---|---|---|
| DINOv2 ViT-S/14 (`dinov2_vits14`) | https://github.com/facebookresearch/dinov2, weights https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth, loaded via `torch.hub` | Apache 2.0 |
| Indian Fabric Patterns v1 | https://universe.roboflow.com/div-szivu/indian-fabric-patterns | MIT |

No other pretrained weights or external data are used.

## Limitations

* **Mostly synthetic colour positives.** Training uses synthetic recolours; the 33 real
  colourway pairs are an evaluation set only. Holding out both the palettes and the recolour
  algorithm guards against learning to invert our own generator, but cannot model dye bleed,
  yarn sheen, lighting or a different weaver's rendering of a motif.
* **Training labels contain undetected duplicates.** Real colourway twins in Kaggle and separate
  photographs of one saree in Drive carry different design ids, so SupCon saw some same-design
  pairs as negatives. This works against the colour-invariance objective and likely understates
  what the method can reach with clean labels.
* **Post-hoc test exclusions.** 27 test designs were judged to leak and are excluded. The
  judgement was visual, so a few same-saree pairs may remain undetected or a few distinct designs
  may have been removed.
* **Small test set.** 93 designs after exclusion, 19 of them from Drive, so intervals are wide and
  the Drive-only numbers especially so.
* **Real pairs vary framing, not palette.** They are crops of one photograph.
* **Source asymmetry.** Kaggle images are stretched and noisy; Drive images are not.

## Extension to other garments

The pipeline separates what is textile-specific from what is general.

* **Unchanged**: the data contract and adapters (a new source is one adapter plus one config
  block), geometric identity grouping, the leakage audit, the held-out recolour protocol, and
  the pre-registered evaluation. All of these apply to any printed or woven fabric.
* **Changes for stitched garments** (kurtas, dupattas, lehengas): the design occupies only part
  of the image and is distorted by cut, seams, folds and the wearer. Segment the fabric regions
  first, embed several patches per garment, and aggregate patch embeddings for retrieval, so a
  sleeve close-up can still match the full garment.
* **Changes for structure-defined designs** (embroidery, appliqué, woven borders): colour
  invariance still matters, but relief and stitch texture carry identity, so lighting and
  specular augmentation matter more than palette swaps, and grayscale input loses less.
* **Data to ask for first**: catalogue SKUs and colourway groupings. The biggest weaknesses here
  (synthetic positives, duplicate labels, post-hoc leakage) all come from missing identity labels.
