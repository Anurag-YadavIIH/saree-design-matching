# Data contract

**Schema version: 2**

This document is the frozen interface between the raw corpus and every piece of
downstream code. The rule it exists to enforce:

> Downstream code (augmentation, sampler, training, eval, inference) reads **only**
> `outputs/manifest.csv`, through `src/data/manifest.py`. Raw folder structure, filename
> quirks and labeling rules live **only** in source adapters and `configs/data.yaml`.

If we learn something new about the data, we change an adapter or a config block. We do not
change downstream code, and we do not change this schema without bumping the version.

Changelog: **v1 to v2** replaced pHash-based `dup_group` with translation-invariant
matching after discovering that pHash cannot see re-cropped views of the same design, added
`is_representative`, `craft_family` and `query_kind`, and added the real-pairs evaluation.

## What recon established, and why the schema looks like this

Evidence lives in `outputs/recon_report.md`, `identity_probe.md`, `hprefix_probe.md`,
`hprefix_verify.md` and `real_pairs.md`. Five facts drive every decision below.

**1. No design-identity labels exist anywhere.** The Kaggle folders are craft families, not
designs. The Drive corpus has opaque numeric filenames and no metadata. Identity must be
*constructed*.

**2. pHash cannot find duplicate designs, and relying on it would have leaked.** Textile
images are often crops of a repeating pattern. A translation offset leaves local pixels
almost unchanged but completely changes the DCT that pHash is built from. Measured on the
Drive corpus: pairs with pHash distance **28 to 38** (random level) whose pixels agree at
**NCC 0.98** once aligned. The Drive corpus is therefore **95 designs, not 165**, with 27
multi-image designs covering 97 images. A pHash-based `dup_group` would have split those 27
designs across train and test, putting the same physical saree on both sides.

**3. There are zero real colourway pairs.** An earlier pass appeared to find 8, but that was
a measurement error: hue was compared over the whole image while alignment only verified the
overlapping quarter, and a saree's body, border and pallu differ in colour, so different
framing alone inflates the hue gap. Comparing hue *inside the aligned overlap* collapses all
8 (whole-image L1 0.42 to 0.86, in-overlap L1 **0.01 to 0.04**). Final tally over 72 verified
same-design pairs: 69 same palette, 3 ambiguous, **0 different palette**.

**4. We do have 27 real same-design groups**, varying framing, region, scale and lighting at
a fixed palette. These are the only real positive pairs in the corpus and are reserved
entirely for evaluation.

**5. The Kaggle set is inflated ~3.6x by augmentation duplicates**, and 88 duplicate clusters
straddle the vendor's own train/valid/test boundary, so that split is discarded.

Consequence: **every colour positive is synthetic.** Synthetic rows are first-class citizens
of the manifest, not an afterthought.

## Identity hierarchy

Three nested levels, which must not be confused:

```
design_id          the identity label; what retrieval must match
  dup_group        images that are the same pixels (noise variants, re-encodings)
    image          one file on disk
```

A `design_id` may span several `dup_group`s (different views of one design). A `dup_group`
holds images that carry no new information. Exactly one image per `dup_group` is the
representative; the rest are train-only extra positives and are never evaluation queries.

## `outputs/manifest.csv`

One row per image, real or synthetic. Primary key is `image_id`.

### Identity

| Column | Type | Purpose |
|---|---|---|
| `image_id` | str | Stable 16-hex SHA1 of `path`; the primary key, so a row survives a rebuild unchanged. |
| `path` | str | Posix path relative to `data/`; the single source of truth for where bytes live. |
| `source` | str | Which corpus this row ultimately came from, so every metric can be broken down Drive vs Kaggle. A synthetic row **inherits its parent's source**, which is what makes that breakdown possible. |
| `design_id` | str | The identity label, globally unique and source-prefixed. A synthetic row **shares its parent's `design_id`**, because a recolor is the same design. |
| `colorway_id` | str | Which palette this image carries; `""` for real images (genuinely unknown), `palette_id` for synthetic ones. |
| `dup_group` | str | Same-pixel cluster id, source-prefixed. Built translation-invariantly, not from pHash alone. The unit that splitting respects. |
| `is_representative` | bool | Whether this image is the canonical member of its `dup_group`. Exactly one per group. Non-representatives are train-only and never evaluation queries. |
| `parent_image_id` | str | `""` for real rows; the parent's `image_id` for synthetic rows, so each recolor links back to the image it derives from. |
| `craft_family` | str | Weave or craft category (`Banarasi`, `Bandhani`, `Ikat`, `Pichwai`); `""` for Drive, which has none. Worthless as identity, but it stratifies the split and defines the hardest tier of stress test B. |

### File facts

| Column | Type | Purpose |
|---|---|---|
| `width` | int | Pixel width of the file on disk, for resize sanity checks and the efficiency report. |
| `height` | int | Pixel height of the file on disk. |
| `phash` | str | 16-hex grayscale perceptual hash. Retained as a cheap duplicate prefilter and as a zero-cost retrieval baseline, but **not** the authority on `dup_group`. |

### Split and role

| Column | Type | Purpose |
|---|---|---|
| `split` | enum | `train` / `val` / `test`; assigned by `design_id`, stratified by source and craft family, deterministic under the seed. |
| `role` | enum | `train` / `gallery` / `query`; what this row is used as inside its split. |
| `query_kind` | enum | `""` for non-queries, else which evaluation row this is: `real_view`, `real_view_recolored`, `synthetic_recolor`. Lets the three query types be reported as separate rows without separate files. |
| `is_synthetic` | bool | Whether these pixels were generated by us. Any headline number must be read knowing this is `True` for all colour-varying queries. |

### Synthesis provenance

Empty or `-1` on real rows. Present so a materialised eval set is reproducible and
inspectable without rerunning the generator.

| Column | Type | Purpose |
|---|---|---|
| `recolor_method` | str | Which recoloring algorithm produced this row (`kmeans_palette_swap`, `lab_hue_chroma_remap`); the axis held out between train and eval. |
| `palette_id` | str | Which palette from the bank was applied; **the key that makes stress test B computable**, since rows sharing a `palette_id` across different `design_id`s are same-palette hard negatives. |
| `recolor_seed` | int | Seed for this specific recolor, so the exact image can be regenerated. |
| `geo_params` | str | JSON of the geometric transform actually applied, so a surprising result can be traced to its perturbation. |

### Quality

| Column | Type | Purpose |
|---|---|---|
| `label_confidence` | enum | `high` / `medium` / `low`; how much to trust this row's `design_id`. |
| `notes` | str | Free text, including any review flag raised during the build. |

## `outputs/label_map.json`

```json
{"schema_version": 2, "label_map": {"drive_img_132981": 0, "kaggle_c0007": 1}}
```

Maps `design_id` to a contiguous integer class index, built from the **train split only**,
so val and test designs are genuinely unseen and no classifier head can index them.
Training recolors inherit their parent's label, so no synthetic entries are needed.

## Labeling rule per source

| Source key | `design_id` | `dup_group` | `label_confidence` |
|---|---|---|---|
| `deeplure_drive` | ORB homography plus in-overlap NCC >= 0.90 connected component | one per image (no same-pixel duplicates exist; minimum pHash distance is 10) | `high`: singletons are proven distinct, groups are pixel-verified |
| `kaggle_fabric` | ORB-verified component over `dup_group` representatives | grayscale pHash cluster at Hamming <= 4 (the vendor's noise triplets share framing, so pHash is reliable here) | `medium`: identity rests on clustering, so a wrong merge produces a wrong design |

Both rules are expressed in `configs/data.yaml`. Changing one is a config edit.

**Kaggle representative choice:** the cluster member with the fewest impulse-noise pixels,
ties broken by sorted path. Noise is estimated by counting pixels that deviate sharply from
a 3x3 median, which is exactly what salt-and-pepper augmentation adds.

## Where synthetic images live

```
data/synthetic/<split>/<design_id>/<image_id>.jpg
```

Under `data/`, which is gitignored, because they derive from the proprietary corpus and
inherit its restrictions. **Never under `outputs/`.**

## Which recolors are materialised, and which are not

| Split | Materialised? | Method | Palette bank |
|---|---|---|---|
| `train` | No, generated on the fly in the Dataset transform | `kmeans_palette_swap` | `bank_train` (24) |
| `val` | Yes, at build time | `lab_hue_chroma_remap` | `bank_val` (8) |
| `test` | Yes, at build time | `lab_hue_chroma_remap` | `bank_test` (8) |
| `val` diagnostic | No, on the fly with a fixed seed | `kmeans_palette_swap` | `bank_train` |

**The held-out recolor distribution is the point.** The three banks are mutually disjoint,
and the eval splits use a *different algorithm* from training. Without this, a high score
would only prove the model learned to invert our own k-means palette swap. Val uses the same
method as test so that model selection is an honest proxy rather than an optimistic one.

**The val diagnostic** applies the *training* method and palette bank to val designs, on the
fly under a fixed seed. It exists solely to report the in-distribution vs held-out
generalisation gap. It is **never** used for checkpoint selection, and the validator asserts
no row of it is written to the manifest.

## Evaluation protocol encoded by this schema

**Queries per eval design: 5.** Gallery is 1 real representative per design.

For the 27 real multi-image Drive designs (all in test), three query kinds:

| `query_kind` | What it is | What it measures |
|---|---|---|
| `real_view` | another real view, unmodified | real framing change, same palette |
| `real_view_recolored` | another real view, recolored with a held-out test palette | real framing change plus synthetic palette change; **the closest proxy we have to the true task** |
| `synthetic_recolor` | recolor plus geometry applied to the gallery image | the standard protocol, as for every other design |

Every other design has `synthetic_recolor` queries only.

### Stress tests, as queries over the manifest

Neither needs a new file. Both fall out of the columns:

- **Stress A, same design / new palette.** Each `query` row against the `gallery` row sharing
  its `design_id`. True by construction.
- **Stress B, different design / same palette.** Rows sharing a `palette_id` but differing in
  `design_id`. Reported in two tiers:
  - *tier 1*: same palette, any design.
  - *tier 2*: same palette **and** same `craft_family` (Kaggle only). Hardest, because colour
    and craft are both controlled, leaving only the motif to discriminate.

### Confidence intervals

Bootstrap over **designs**, not queries: 1000 resamples, 95% interval. Queries sharing a
parent are correlated, so resampling queries would understate the interval.

### Baselines (wired in Phase 3)

| Baseline | Expectation |
|---|---|
| grayscale pHash | Non-trivial. Colour-blind by construction, so it sets the floor a learned model must clear |
| RGB colour histogram | **Expected to fail.** Its failure is the evidence that the benchmark punishes colour shortcuts rather than rewarding them |
| zero-shot DINOv2, RGB input | The pretrained-embedding baseline from the brief |
| zero-shot DINOv2, grayscale input | Isolates how much of DINOv2's performance is colour versus structure |

All four report in the same table as the trained model, with per-source breakdown and the
real-pairs table beside it.

## Invariants enforced by `src/data/validate.py`

The build fails rather than writing a manifest that violates any of these.

**Leakage**
1. No `design_id` appears in more than one `split`.
2. No `dup_group` appears in more than one `split`.
3. Every synthetic row sits in the same `split` as its parent.
4. `bank_train`, `bank_val` and `bank_test` share no `palette_id`.
5. No row uses the training recolor method inside `val` or `test`.

**Referential integrity**
6. Every `parent_image_id` resolves to a real row in the manifest.
7. A synthetic row's `source`, `design_id` and `craft_family` equal its parent's.
8. `image_id` and `path` are each unique.
9. Exactly one `is_representative` row per `dup_group`.

**Evaluability**
10. Every `test` and `val` design has at least one `gallery` row and at least one `query` row.
11. Every `gallery` row is real and representative.
12. No non-representative row has `role` of `gallery` or `query`.
13. `label_map` covers exactly the distinct `design_id`s with `split == train`.
14. Every `query` row has a non-empty `query_kind`; no non-query row has one.

**Schema and files**
15. Columns, order and dtypes match this document; enums contain only declared values.
16. Every `path` exists and opens.
17. Real rows have empty synthesis provenance; synthetic rows have it populated with
    `recolor_seed >= 0`.

## Colour normalisation happens in exactly one place

`src/data/dataset.py` converts every image to RGB on load. Reason: the corpus is RGB today,
but palette (`P`), grayscale (`L`) and alpha (`RGBA`) files appear routinely in scraped
textile data, and an alpha channel silently composited by a different library is a
reproducibility bug that is painful to find. One conversion point, one comment.

## Provenance to disclose in the README

| Source | Detail |
|---|---|
| DeepLure Drive corpus | Proprietary. Never committed, uploaded or copied out of `data/`. 165 files, 95 designs. |
| Kaggle / Roboflow | **Indian Fabric Patterns v1**, Roboflow Universe, `div-szivu/indian-fabric-patterns`, exported 2024-05-26. **MIT licence.** 1,468 images. |

**Kaggle preprocessing to disclose:** Roboflow applied auto-orientation with EXIF stripping
and **resized every image to 640x640 by stretching**, so aspect ratio is distorted and does
not reflect the true drape of the fabric. It also added salt-and-pepper noise to 0.1% of
pixels across 3 generated versions per source image. Drive images are untouched, 226 to
1512 px, mostly square. Any cross-source comparison must account for this asymmetry.
