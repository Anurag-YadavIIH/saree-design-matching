# Decision log

Study guide for the live follow-up. One entry per decision.

## Template
### D-XX: <short title>
- **Decision:**
- **Why:**
- **Alternatives rejected (and why):**
- **Evidence:** (metric, plot, or test that supports it)
- **What I would do with more time:**

---

### D-01: Framework and compute
- **Decision:** PyTorch, Kaggle free GPU, training budget under ~1.5 h.
- **Why:** Required by the brief; keeps the notebook reproducible by reviewers.
- **Alternatives rejected:** None (hard constraint).
- **Evidence:** Brief, section 5.

---

### D-02: The corpus has no design labels, so identity must be constructed
- **Decision:** `design_id` is derived from image content, not read from any label.
- **Why:** Neither source carries design identity. The Kaggle folders are craft families
  (Banarasi, Bandhani, Ikat, Pichwai), which are weave and technique categories: 432
  Banarasi images are 432 different designs. The Drive corpus is a flat folder of
  `img_<random id>.jpg` with no metadata file at all.
- **Alternatives rejected:**
  - *Craft family as the label.* Would make retrieval a 4-way classification task and would
    not test design identity at all.
  - *Filename stem as the label.* Rejected on evidence, see D-04.
- **Evidence:** `outputs/recon_report.md`. Kaggle: 4 class folders, 489/316/342/321 images.
  Drive: 165 files, 1 folder, 0 label files.
- **What I would do with more time:** Ask the client for catalogue SKUs, which is what a
  real deployment would key on.

---

### D-03: pHash is blind to translation, which nearly caused silent leakage
- **Decision:** `dup_group` and `design_id` are established by geometric verification (ORB
  keypoints, RANSAC homography, normalised cross-correlation over the aligned overlap), not
  by perceptual hash. pHash is retained only as a cheap prefilter, as the authority on
  same-framing duplicates, and as a baseline.
- **Why:** Textile images are frequently crops of a repeating pattern. A translation offset
  leaves local pixels almost unchanged, but pHash is a DCT of a 32x32 downsample, and
  shifting the content by half a period changes those coefficients completely. So pHash
  reports "unrelated" for two crops of the same saree.
- **Evidence, and this is the strongest number in the project:** Drive pairs with grayscale
  **pHash distance 28 to 38** (the random-pair median is 32) whose pixels agree at
  **NCC 0.97 to 0.99** once aligned. Consequence: the Drive corpus is **95 designs, not
  165**, with 28 multi-image designs covering 98 images. A pHash-based `dup_group` would
  have split those 28 designs across train and test, placing the same physical saree on both
  sides of the evaluation.
- **Alternatives rejected:**
  - *pHash alone.* Demonstrably misses 28 designs.
  - *ORB inlier count alone.* Not trustworthy on textiles, see D-05.
- **What I would do with more time:** Replace ORB with LoFTR or SuperGlue, which handle
  repetitive texture better than hand-crafted keypoints.
- **Known tech debt:** `scripts/recon.py` has a flawed candidate-key table: a key that
  collapses everything into a few broad groups scores *best* on "percent of groups with >= 2
  images". It did not affect any conclusion, because neither source encodes identity in
  filenames and the dedicated probes were used instead, but it would mislead on a source
  that does. Deliberately not fixed, for time.

---

### D-04: The filename stem is not an identity key
- **Decision:** The Roboflow source stem is emitted as a hint and flagged when generic; it is
  never used as a label.
- **Why:** Roboflow encodes names as `<stem>_<ext>.rf.<32 hex>`, and genuine augmentation
  triplets do share a stem. But many stems are generic and collide across unrelated images.
- **Evidence:** 354 distinctive stems show a within-stem median pHash distance of **0** (true
  triplets, 99% within distance 2). 314 generic stems (`image21`, `images4`) show a median of
  **28**, statistically indistinguishable from unrelated images, and 53 of them span multiple
  craft families. They are filename collisions, not identities.
- **Alternatives rejected:** *Trust the stem.* Would have merged unrelated designs.

---

### D-05: ORB inlier count is not sufficient; pixel agreement decides
- **Decision:** A pair is only the same design if, after the RANSAC homography, normalised
  cross-correlation over the aligned overlap is >= 0.90, with guards on overlap fraction,
  region texture (std) and homography determinant.
- **Why:** Repeating motifs let RANSAC fit a homography between two *different* fabrics by
  aligning a lattice of similar motifs. Inlier count then looks convincing while the images
  are unrelated.
- **Evidence:** A control distribution over 300 random unrelated Drive pairs reached a 99th
  percentile of **309 inliers** and a max of 396. Candidate "matches" sat in the same range.
  Only pixel-level agreement separated the real pairs from the lattice artefacts.
- **What I would do with more time:** Calibrate the NCC threshold against a hand-labelled set
  rather than against controls.

---

### D-06: There are zero real colourway pairs (and my first measurement was wrong)
- **Decision:** All colour-varying positives are synthetic. Stated plainly in the README.
- **Why:** Confirmed empirically after correcting a measurement error of my own.
- **The error, worth remembering:** I first reported 8 real colourway pairs. I had compared
  hue over the **whole image** while the homography only verified the overlapping quarter. A
  saree's body, border and pallu carry different colours, so two crops of one fabric show a
  large whole-image hue gap purely from framing. Comparing hue **inside the aligned overlap**
  collapsed all 8.
- **Evidence:** For those 8 pairs, whole-image hue L1 was **0.42 to 0.86** while in-overlap
  hue L1 was **0.01 to 0.04**. Final tally over 72 verified same-design pairs: 69 same
  palette, 3 ambiguous, **0 different palette**.
- **Alternatives rejected:** *Report the 8 as real colourway pairs.* Would have put a false
  claim at the centre of the submission.
- **What I would do with more time:** Ask the client for catalogue colourway groupings, which
  is the single most valuable missing piece of data.

---

### D-07: The `h_` filename prefix carries no pairing information
- **Decision:** Strip `h_` for readability; never use it as a label.
- **Why:** Investigated because, if `h_img_<id>` paired with `img_<id>`, those would have been
  real positive pairs.
- **Evidence:** **Zero** of the 9 `h_` files share a numeric id with any `img_` file. Content
  matching found no systematic relationship. `h_` files are larger on average (median 246 KB
  vs 44 KB, median longest side 1259 vs 656 px) but plain `img_` files also reach 1512 px, so
  it is not a strict resolution tier either.
- **Evidence file:** `outputs/hprefix_probe.md`, `outputs/hprefix_verify.md`.

---

### D-08: The vendor's train/valid/test split is discarded
- **Decision:** Re-split from scratch by `design_id`.
- **Why:** The Roboflow split leaks across its own boundaries.
- **Evidence:** **88 of 412** duplicate clusters straddle the vendor's split boundaries. In a
  500-image sample, 9 pHash-identical clusters spanned more than one vendor split.
- **Alternatives rejected:** *Reuse the vendor split.* Would have inflated every metric.

---

### D-09: Duplicates collapse first, identity is verified on representatives only
- **Decision:** Stage 1 collapses same-pixel duplicates with pHash. Stage 2 picks one
  representative per cluster (fewest impulse-noise pixels, ties by sorted path). Stage 3 runs
  the expensive geometric identity check on representatives only. Stage 4 merges clusters
  whose representatives turn out to be near-identical.
- **Why:** Two reasons. Correctness: pHash is genuinely reliable for the Roboflow noise
  triplets, which share framing exactly, so using it there is appropriate rather than a
  shortcut. Cost: the geometric stage is expensive, and this reduces Kaggle from 1,468 images
  (1,077,278 pairs) to 412 representatives.
- **Evidence:** Kaggle yielded **411 designs from 412 pHash clusters**, so the re-crop
  phenomenon is specific to the Drive corpus and Kaggle's duplication is purely the noise
  augmentation. Drive, by contrast, went 165 images to 95 designs.
- **Alternatives rejected:** *Geometric verification on every image.* Over a million pairs,
  with no benefit, since duplicates are already resolvable more cheaply.

---

### D-10: Union-find over the verified-pair graph
- **Decision:** `design_id` is a connected component, not a pairwise grouping.
- **Why:** Overlapping tiles form chains. If tile 1 overlaps tile 2 and tile 2 overlaps tile
  3, all three are the same saree even when tiles 1 and 3 share no pixels and could never be
  matched directly. Pairwise grouping would split that design and leak it across splits.
- **Evidence:** Drive multi-image designs reach 5 and 6 images each, which only a transitive
  grouping recovers.

---

### D-11: Held-out recolour distribution, by palette AND by algorithm
- **Decision:** Training uses `kmeans_palette_swap` with a 24-palette bank. Val and test use
  `lab_hue_chroma_remap` with disjoint 8-palette banks. All three banks are mutually
  disjoint, asserted at build time.
- **Why:** If evaluation used the same generator as training, a high score would only prove
  the model learned to invert our own k-means palette swap. Holding out both the palettes and
  the algorithm makes the number mean something closer to genuine colour invariance.
- **Why val matches test rather than train:** so model selection is an honest proxy for the
  test number instead of an optimistic one. The cost is that we lose an in-distribution val
  signal, which is why D-12 exists.
- **Evidence:** The two methods are measurably different: hue L1 between a train-method and a
  test-method recolour of the same image is **1.96 of a maximum 2.0**.
- **Alternatives rejected:** *One recolour method everywhere.* Measures generator inversion,
  not invariance.

---

### D-12: In-distribution val diagnostic, report-only
- **Decision:** Also recolour val designs with the *training* method and bank, on the fly
  under a fixed seed. Report the gap. Never use it for checkpoint selection.
- **Why:** The difference between in-distribution and held-out val mAP is the direct measure
  of how much the model overfits our generator. That number is worth reporting and is
  dangerous to optimise.
- **Evidence:** Enforced by a validator invariant that no manifest row in val or test carries
  the training recolour method.

---

### D-13: Luminance is restored after recolouring
- **Decision:** After a palette swap, copy the original LAB L channel back.
- **Why:** A raw palette swap replaces every pixel of a cluster with one flat colour, which
  destroys the weave texture: precisely the signal the model must learn. Restoring luminance
  means only chroma changes, so the design survives.
- **Evidence:** Grayscale correlation with the original is **0.981** (train method) and
  **0.997** (test method), while hue L1 from the original is **1.96 to 1.99 of 2.0**. So
  structure is preserved while colour is genuinely replaced. Mean absolute L difference is
  under 1 unit on a 0 to 255 scale.

---

### D-14: Cached cluster maps, because k-means was the bottleneck
- **Decision:** Compute the k-means cluster assignment once per real image at 256 px and
  cache it as a uint8 index map. A train-time recolour is then `palette[cluster_map]` plus a
  luminance restore.
- **Why:** Running k-means per image per epoch would dominate training time and would waste
  most of the 40-minute budget on CPU work instead of gradient steps.
- **Implementation note:** The cached map is upsampled with nearest-neighbour, never
  interpolation, because interpolating cluster indices invents clusters that do not exist.
- **Alternatives rejected:** *k-means per batch.* Far too slow for the time budget.

---

### D-15: No horizontal or vertical flips in augmentation
- **Decision:** Geometry includes crop, rotation (+/- 15 deg), mild perspective, scale,
  blur and JPEG recompression, but no flips.
- **Why:** Many saree motifs are directional (paisley, figurative pallu scenes, directional
  weaves). A mirrored motif is arguably a *different* design, so training flip invariance
  would teach the model to ignore a real distinction.
- **What I would do with more time:** Measure it, by training one variant with flips enabled
  and comparing stress test B.

---

### D-16: Reflection padding on rotation, not black fill
- **Decision:** `BORDER_REFLECT_101` when rotating and warping.
- **Why:** Black corners are a trivial cue correlated with the augmentation itself. The model
  could learn to detect "this is a rotated query" instead of learning the design.

---

### D-17: Split stratified by source and craft family, with a higher Drive test fraction
- **Decision:** 0.15 val / 0.20 test overall, stratified by `(source, craft_family)`, with
  the Drive test fraction raised to 0.40. All 28 real multi-view designs are forced into test
  and count towards the Drive quota.
- **Why:** Drive is the client's target domain, so it deserves more test mass. The 28 real
  multi-view designs are the only real positive pairs in the corpus and are worth more as
  evaluation evidence than as training signal: synthetic views already supply thousands of
  training positives, so roughly 14 real pairs would add little there.
- **Evidence:** Result is 38 Drive test designs out of 95, and 81 Kaggle test designs out of
  411. See `outputs/data_summary.md`.
- **Trade-off accepted:** 38 test designs is a small sample, so confidence intervals are wide.
  Reporting them honestly is better than hiding the uncertainty. This is why D-18 exists.
- **Alternatives rejected:** *Raising the Drive test fraction to 0.68 to reach 65 test
  designs.* Would leave only ~30 Drive designs for training, starving the domain that matters
  most.

---

### D-18: Bootstrap confidence intervals resample designs, not queries
- **Decision:** 1000 resamples, 95% percentile interval, resampling **designs**.
- **Why:** Each design contributes 5 or more queries derived from the same parent image, so
  sibling queries are strongly correlated: fail one and you likely fail its siblings.
  Resampling queries would treat them as independent, understate the variance, and report an
  interval that is too narrow.
- **Alternatives rejected:** *Resampling queries.* Statistically wrong here, and it would
  flatter the result.

---

### D-19: Stress test B falls out of `palette_id`, in two tiers
- **Decision:** No separate file or role. Rows sharing a `palette_id` but differing in
  `design_id` are same-palette hard negatives. Tier 1 is any design; tier 2 additionally
  requires the same `craft_family` (Kaggle only).
- **Why:** Because eval designs draw from a shared 8-palette bank, many designs receive the
  same palette, so these pairs exist in quantity without generating anything extra. Tier 2 is
  the hardest case available: colour and craft are both controlled, leaving only the motif.

---

### D-20: Three query kinds for the 28 real multi-view designs
- **Decision:** `real_view` (another real view, unmodified), `real_view_recolored` (another
  real view plus a held-out palette), `synthetic_recolor` (the standard protocol).
- **Why:** `real_view_recolored` is the closest proxy to the true task that this corpus
  permits: a real framing and region change combined with a palette change.
- **Precise framing, which must not be overstated:** the real pairs are **overlapping crops
  of one photograph** (scale 1.0, about 25% overlap). They therefore test **region and
  framing invariance at fixed lighting and scale**. They are *not* re-photography of a
  physical saree, and they do not vary palette.
- **Evidence:** 71 `real_view` and 71 `real_view_recolored` rows over 28 designs.

---

### D-21: Queries may show a different region, so training crops aggressively
- **Decision:** Keep a wide `RandomResizedCrop` (scale 0.5 to 1.0) in training augmentation,
  wider than the 0.70 to 1.00 used when generating eval queries.
- **Why:** The corpus demonstrates that in practice two images of one saree can be different
  regions of it. Training on aggressive crops matches that deployment reality; a model trained
  only on near-full views would degrade when handed a close-up of a border.

---

### D-22: Post-split leakage audit, because our detector has a known blind spot
- **Decision:** After splitting, embed every test and train real image and report, for each
  test image, its nearest train neighbour by cosine, writing the top matches for human review.
- **Why:** The invariants prove no `design_id` or `dup_group` crosses a split, but only for
  identity our grouping could *see*. pHash already fooled us once (D-03). An independent
  check on a different representation bounds the residual risk.
- **Stated limitation:** Two **non-overlapping** tiles of one saree share no pixels and no
  keypoints, so no content-based method can link them. Some undetected same-design leakage may
  remain. This is disclosed in the README rather than quietly ignored.

---

### D-23: Gallery rows must be real and representative
- **Decision:** Validator asserts no gallery row is synthetic and every gallery row is its
  duplicate cluster's representative.
- **Why:** A synthetic gallery would mean comparing one of our generations against another,
  measuring the generator rather than the designs. A noisy representative would handicap the
  gallery for no reason.

---

### D-24: Non-representative duplicates are train-only
- **Decision:** They may be sampled as extra training positives but are never gallery or
  query rows. Enforced by an invariant.
- **Why:** A Roboflow noise triplet member differs from its representative only by
  salt-and-pepper noise. As a test query it would be trivially retrievable and would inflate
  Rank-1 without measuring anything. As a training positive it is mildly useful free variation.

---

### D-25: One RGB conversion point
- **Decision:** `src/data/dataset.py` converts every image to RGB on load, and nowhere else
  does.
- **Why:** The corpus is RGB today, but palette (P), grayscale (L) and alpha (RGBA) files are
  routine in scraped textile data. An alpha channel composited differently by another library
  is a reproducibility bug that is painful to find.

---

### D-26: PK sampling with forced same-palette groups
- **Decision:** Batches are P=32 designs x K=4 views. With probability 0.3, a group of 4
  different designs in the batch is forced onto a single palette.
- **Why:** PK is required because SupCon learns from in-batch structure: a uniformly shuffled
  batch drawn from ~400 designs would be almost all singletons, giving no positive pairs. The
  forced same-palette group attacks the colour shortcut directly: inside that group colour is
  uninformative, so only the motif can separate the designs. That is exactly the brief's
  requirement that different motifs in identical palettes must not match.

---

### D-27: Local training is not viable; the real run goes to Kaggle
- **Decision:** CPU-only torch locally, for the smoke test and baselines. Real training on
  Kaggle's T4.
- **Why:** Two independent blockers. The local GPU is a GTX 1050 with 4 GB, which cannot hold
  P=32 x K=4 at 224 px, and Pascal has no fp16 tensor cores so AMP gains little. Separately,
  the CUDA wheel was downloading at roughly 100 KB/s against a 2.4 GB payload, which would
  have consumed hours of a deadline-bound session.
- **Alternatives rejected:** *Shrinking the batch to fit 4 GB.* Would change the loss
  dynamics that SupCon depends on, making the local result non-comparable to the Kaggle run.

---

### D-28: Leakage safety net is a split constraint, not an identity merge
- **Decision:** Designs with embedding cosine >= 0.93 AND a shared distinctive filename prefix
  of >= 5 characters are forced onto the same side of the split. Their `design_id`s are NOT
  merged.
- **Why:** Geometric verification can only link images that share pixels. Two separate
  photographs of one saree share none, so they look like different designs. The post-split
  leakage audit found real cases. A constraint is the asymmetric-risk choice: if the pair is
  the same design, no leak; if it is not, we lose a little split flexibility and nothing else.
  A merge would assert identity we have not proven, and would corrupt retrieval labels if
  wrong.
- **Evidence:** Caught `9557ER2_4_grande` vs `9557ER3_4` (cosine 0.935) and two
  `whatsapp-image-2023-09-19-at-10-28` files sent in the same minute (cosine 0.955).
- **Alternatives rejected:** *Merge the designs.* Unproven identity claim. *Ignore and only
  document.* These cases are detectable, so leaving them would knowingly inflate the metric.
- **Known miss:** `dsc00117` vs `dsc00106` (sequential camera frames) sits at cosine 0.92,
  just under the threshold. Lowering the threshold would catch it but raise false constraints.
  Left as is and reported via the leakage audit.

---

### D-29: A bug found in my own safety net: one image, two unrelated names
- **What happened:** The first version checked the filename prefix on each design's
  REPRESENTATIVE only, and missed the 9557ER pair entirely.
- **Root cause:** pHash had correctly placed a file named `image21` in the same duplicate group
  as the `9557ER3_4` noise triplet, because they hold the same pixels. The source collection
  saved one image twice under unrelated names. `image21` won the representative choice (fewest
  noise pixels), so the check saw a generic stem and skipped the group.
- **Fix:** The prefix condition now considers every distinctive filename in a design.
- **Lesson worth stating in the interview:** this is the third independent piece of evidence
  that filenames are not identity in this corpus (after generic-stem collisions in D-04 and the
  meaningless `h_` prefix in D-07). Content decides; names are at most a hint.

---

### D-30: Seeds come from SHA1, never from Python's hash()
- **Decision:** Synthetic query seeds are `stable_seed(image_id, kind, index)`, a SHA1 digest.
- **Why:** Python's built-in `hash()` of a string is randomised per process (PYTHONHASHSEED).
  The first builder used it, which meant every rebuild generated a DIFFERENT test set: a silent
  reproducibility failure that no metric would reveal.
- **Evidence:** `tests/test_data_contract.py::test_build_is_deterministic` builds twice and
  asserts identical identities, splits, roles and recolour seeds. It passes.

---

### D-31: The validator is tested by breaking it
- **Decision:** Six negative tests each corrupt a valid manifest in one specific way and
  assert the validator fails: a design leaking across splits, the training recolour method in
  test, a synthetic gallery row, a palette shared across splits, a missing representative, and
  a stale label map.
- **Why:** A validator that passes on a good manifest proves almost nothing. The question is
  whether it catches a bad one.
- **Evidence:** 13 of 13 tests pass, offline, on a generated fixture, in about 40 seconds. The
  fixture also reproduces the core finding: two overlapping crops of one texture merge into a
  single design.

---

### D-32: The bar to beat is zero-shot DINOv2 on GRAYSCALE input
- **Decision:** The trained model is judged against zero-shot DINOv2 with grayscale input, not
  against pHash or RGB DINOv2.
- **Why:** Gray input beats RGB on every metric, so the trivial colour-invariance fix is simply
  to feed grayscale. A fine-tuned model that only matches it has not earned its training cost.
- **Evidence (test, 120 designs):** colour histogram Rank-1 0.090, pHash 0.448, DINOv2 RGB
  0.892, DINOv2 gray **0.913** (mAP 0.943, TAR@FAR=1e-3 0.630).
- **Benchmark validity evidence:** the colour histogram scores 0.52 Rank-1 on `real_view`
  (palette unchanged, colour is a free clue) and falls to 0.13 on `real_view_recolored`. On
  stress B it sits at chance (AUC 0.54). The protocol punishes colour shortcuts.
- **Where the headroom is:** verification at strict FAR (TAR 0.63), real-pair framing
  invariance (`real_view` 0.80, `real_view_recolored` 0.76 versus 0.96 synthetic on the same
  designs), and stress B tier 2 (0.976).
- **Honest caveat:** the held-out LAB generator separates RGB from gray DINOv2 by only about 2
  points, so the synthetic headline under-stresses colour reliance. The real-pairs table is
  the stronger test and is reported prominently.

---

### D-33: Tonal queries, because grayscale was invariant to every query BY CONSTRUCTION
- **Decision:** Add a `tonal_remap` generator and a `synthetic_tonal` query kind (5 per eval
  design). Per k-means cluster the new colour takes the palette entry's own lightness, with
  texture kept as `new_L = palette_L[c] + (L - mean_L[c])`, and with probability 0.5 the
  dark-to-light order of clusters is inverted. Stress B gains a tier 3: different designs in
  the same tonal palette.
- **Why:** Both earlier generators restore the original L channel, so a grayscale image is
  unchanged by recolouring. That, not model quality, explains zero-shot gray DINOv2 at 0.913
  and the 2-point RGB/gray gap. Real colourways often change or invert light and dark.
- **Evidence:** Grayscale correlation with the original: old LAB remap **0.997**; tonal with
  order kept **0.93**; tonal with order inverted **-0.70**.
- **Honest caveat:** For tonal queries only the PALETTES are held out; training uses the same
  tonal rule (p=0.5, train bank). For the LAB remap both palettes and algorithm are held out.
- **Credit:** Flagged by Anurag in review of the baseline table.

---

### D-34: Success criteria, written BEFORE the training runs
Recorded before either Kaggle run, so the bar cannot be moved to fit the result.

- **Reference:** zero-shot DINOv2 with grayscale input, rerun on the new manifest.
- **Success** requires BOTH:
  1. The trained model beats gray zero-shot **outside gray zero-shot's 95% CI** on at least
     one of: tonal Rank-1, TAR@FAR=1e-3 (main queries), `real_view_recolored` Rank-1.
  2. The trained model's main (`synthetic_recolor`) Rank-1 does **not** fall below the lower
     bound of gray zero-shot's CI.
- **If it fails:** report that plainly, and ship **gray zero-shot DINOv2 plus a projection**
  as the recommended model. The negative result stays in the README.
- **Runs:** (a) RGB input, (b) gray input, identical config otherwise, about 20 minutes each.
  Checkpoint selection: mean of val mAP on held-out recolour and tonal queries (never the
  in-distribution diagnostic).
- **Why "outside the CI" and not "higher":** with 120 test designs the CIs are 6 to 9 points
  wide. A point estimate that is higher but inside the interval is not evidence of improvement.

---

### D-35: Baselines on tonal queries (the reference for D-34)
- **Evidence (test, 120 designs, 95% CIs over designs):**

  | method | main Rank-1 | tonal Rank-1 | tonal TAR@1e-3 | stress B tier 3 AUC |
  |---|---|---|---|---|
  | RGB colour histogram | 0.090 | 0.010 | 0.002 | 0.212 |
  | grayscale pHash | 0.448 | 0.190 | 0.130 | 0.525 |
  | DINOv2 zero-shot RGB | 0.892 | 0.573 [0.517, 0.632] | 0.195 | 0.941 |
  | DINOv2 zero-shot gray | 0.913 [0.878, 0.943] | 0.638 [0.583, 0.692] | 0.278 [0.235, 0.325] | 0.952 |

- **Reading:** tonal queries remove the grayscale-invariance advantage (gray 0.913 to 0.638,
  pHash 0.448 to 0.190), creating real headroom. Gray input still beats RGB even on tonal
  queries. The colour histogram scores BELOW chance on tier 3: it prefers a different design in
  the same palette over the true match, the exact failure the brief describes.
- **Bars for D-34:** tonal Rank-1 > 0.692, or main TAR > 0.718, or real_view_recolored above
  gray's CI upper bound; and main Rank-1 >= 0.878.
- **Note:** main-table TAR for gray moved 0.630 to 0.667 because the val threshold is now chosen
  over val pairs that include tonal queries. Same model, different frozen threshold; every
  method uses the same rule.

---

### D-36: Efficiency is measured on the T4, and the script is not called profile.py
- **Decision:** `scripts/profile_model.py` reports params, FLOPs, latency (batch 1 and 32,
  fp32 and fp16), peak memory and embedding bytes. The notebook cell asserts the GPU is a T4
  before running it, and the report stamps the device name and flags non-target hardware.
- **Why the name:** running a script puts its folder first on `sys.path`, so `profile.py`
  shadowed the standard-library `profile` module. torch imports `cProfile`, which imports
  `profile`, and crashed. Found by running it, not by reasoning.
- **Architecture numbers (device independent):** 22.3M parameters total, 7.4M fine-tuned,
  0.26M in the head; 11.0 GFLOPs per 224x224 image; 128-d embedding = 512 B fp32 / 256 B fp16,
  3x smaller than the raw 384-d DINOv2 CLS vector (488 MB per million designs at fp32).

---

### D-37: Proprietary-derivative cleanup on Kaggle
- **Decision:** A final notebook cell deletes everything derived from the Drive corpus from
  `/kaggle/working` and VERIFIES the deletion: no image or `.npz` files remain, `data/` is gone,
  and no kept text file contains a Drive filename. Kept: results, efficiency report, per-run
  train log, summary and best checkpoint.
- **Why:** Kaggle notebook outputs become public if the notebook is made public, and synthetic
  recolours and k-means maps are still derivatives of proprietary images.
- **Second leak path, closed separately:** printed cell output is stored INSIDE the notebook,
  where no file deletion reaches it. The build log names Drive files, so it is written to a
  file (deleted at cleanup) and the cell prints only path-free summary lines.
- **Symlink safety:** the raw source folders are symlinks into read-only `/kaggle/input`.
  `is_symlink()` is checked before `is_dir()` and links are `unlink()`ed, never `rmtree`d.
- **Residual risk stated plainly:** `best.pt` is itself a derivative, since it was fine-tuned
  partly on Drive designs. It is kept because it is needed, so the notebook must stay private.

---

### D-38: Leakage audit verdicts, and post-hoc exclusion instead of retraining
- **Method:** `scripts/audit_pairs.py` re-tests the audit's top 20 test/train design pairs with
  the pixel-level verifier (ORB, RANSAC homography, NCC over the aligned overlap) across six
  orientations (identity, both flips, three rotations), since a second photo may be mirrored
  or rotated. SAME needs NCC >= 0.90. High cosine alone is not evidence: two sarees of one
  craft family can look alike.
- **Result:** **0 of 20 pixel-verified as the same saree.** 15 DIFFERENT. 5 NEEDS EYES
  (cosine >= 0.93, not verifiable): the strongest is #4, drive_d00003 vs drive_d00083, NCC
  0.853 after a 90 degree rotation with 64 inliers. Suggestive, since a crop-and-rotate
  positive control had median NCC 0.68, but a symmetric motif rotated 90 degrees can also
  match a different fabric. A human decides from `outputs/leakage_pairs.jpg`.
- **Mechanism:** test designs judged to be the same saree go in `configs/eval_exclude.yaml`;
  `evaluate.py` drops them from test gallery AND queries, logs the count, and states the
  exclusion at the top of the results. No retrain: removing the test side removes the
  possible leak from every reported number. Tested: excluding drive_d00003 drops 1 design and
  14 queries.
- **Status at time of writing:** list empty, pending Anurag's visual review.
- **Why evaluate locally rather than re-run on Kaggle:** the Kaggle cleanup cell deletes the
  manifest and synthetic queries, so the evaluate cell cannot be re-run there. Builds are
  deterministic (`test_build_is_deterministic`), so the local manifest is identical to Kaggle's
  and accuracy metrics do not depend on hardware. Downloaded checkpoints are evaluated locally.

---

### D-39: FLOPs stated with their convention, and tokens counted from the model
- **Numbers:** 11.03 GFLOPs = 5.52 GMACs per 224x224 image; 257 tokens = 256 patches (16x16 of
  14 px) + 1 CLS, no register tokens.
- **Why it matters:** "GFLOPs" is ambiguous. torch `FlopCounterMode` counts 2 per multiply-add;
  fvcore counts multiply-adds and calls them flops, so the same model reads 11.0 or 5.5.
- **Evidence, not memory:** the profiler measures the convention on a bias-free
  Linear(384, 512) probe (196,608 multiply-adds, reported as 393,216) and reads the token count
  from `prepare_tokens_with_masks`. Trap avoided: `patch_embed.num_patches` reports 1369, the
  37x37 grid of the 518 px pretraining size; at 224 the position embeddings are interpolated.

---

### D-40: Real colourway verification on the Kaggle catalogue
- **Method:** `scripts/mine_colourway_pairs.py` embeds all 412 Kaggle representatives with
  zero-shot DINOv2 on GRAYSCALE input, takes each image's top-3 neighbours from a different
  design, and keeps pairs whose hue histograms differ strongly (L1 >= 0.6): 788 candidates.
  The top 60 by similarity were judged on contact sheets (Kaggle images only, public MIT
  data): **33 accepted** as same print in a different colourway, 27 rejected with reasons in
  `outputs/real_colourway_pairs.csv`. Acceptance fell 70% -> 60% -> 35% across the three
  sheets, so review stopped at 60; 727 candidates are unreviewed.
- **Most common rejection:** different Ikat motifs photographed on the same mannequin. The
  similarity was the photo setup, not the print: a staging confound worth naming.
- **Eval:** verification, positives = the 33 pairs, negatives = 21,120 same-craft-family
  pairs not linked through accepted pairs and not among unreviewed candidates. ROC-AUC and
  TAR at FAR = 1e-2 (1e-3 would rest on a handful of negatives), with bootstrap CIs.
- **Baselines:** pHash AUC 0.686 [0.562, 0.820], TAR 0.273; colour histogram AUC 0.597, TAR
  0.030; DINOv2 RGB AUC 0.998, TAR 0.939; DINOv2 gray AUC 0.999, TAR 1.000.
- **Caveats, stated with the numbers:**
  1. **Selection bias, and it is large.** Positives were mined as gray zero-shot DINOv2's own
     top-3 neighbours, so that model scores them highly by construction. Its perfect TAR is
     not evidence; this eval cannot rank gray zero-shot against anything correlated with it.
     It IS informative that pHash and the colour histogram fail on real colourways.
  2. 10 of 33 pairs have both sides in train with different design_ids, so SupCon was trained
     to push them apart: the fine-tuned models are underestimated here.
  3. 21 of 33 pairs straddle splits: one colourway in train, another in val or test.
  4. TAR threshold is set on the same negatives; there is no separate validation set.
- **What I would do with more time:** mine with an independent signal (e.g. ORB on
  edge maps, or a different backbone) so the eval does not favour the mining model.

---

### D-41: Drive same-saree leakage: 19 of 38 Drive test designs excluded post hoc
- **Finding:** Anurag's visual audit judged 19 of the 38 Drive test designs to be separate
  photographs of a saree that also appears in train. Plus 8 Kaggle test designs with a real
  colourway twin in train. 27 test designs in `configs/eval_exclude.yaml`.
- **Cause:** separate photographs of one saree share no pixels, so the geometric identity
  check (which only links overlapping crops) cannot connect them, and the leakage safety net
  needs a shared distinctive filename, which Drive's opaque numeric names never provide. My
  pixel-level re-check of the top 20 audit pairs verified none (best NCC 0.853, rotated 90
  degrees). Human judgement found what content matching could not.
- **Consequence:** half the Drive test set was leaked. Excluding it leaves 19 Drive test
  designs, so Drive-only confidence intervals are roughly twice as wide.
- **Fix chosen:** post-hoc exclusion from test, no retrain (D-38). The model has seen the
  train side, which was never a test target.
- **What I would do with more time:** re-split with these same-saree links as split
  constraints and retrain, so the full Drive test set is usable.

---

### D-42: Real colourway pairs DO exist in Kaggle; D-06 is superseded for that source
- **Correction:** D-06 concluded zero real colourway pairs in the corpus. That holds for the
  Drive corpus. It is wrong for Kaggle: at least 33 exist (D-40).
- **Why the earlier probe missed them:** it required LOW grayscale pHash distance before
  checking hue. A real colourway in this catalogue is a separately photographed product shot,
  often a swirl or drape, so its pHash distance is high even when the print is identical. The
  filter discarded every true pair before hue was examined. The later geometric probe had the
  same blind spot for a different reason: it only links images that share pixels.
- **Lesson:** both of my "no pairs" conclusions inherited the assumption that a same-design
  pair shares pixels or layout. Re-photography breaks that assumption.
- **Consequence for training:** these pairs carry different design_ids, so SupCon treated them
  as negatives. The training labels contain undetected duplicates of the most valuable kind.

---

### D-43: Success bars re-derived AFTER post-hoc exclusion (supersedes the numbers in D-35)
- **Rule unchanged (D-34).** Only the numeric bars change: D-35's were computed on 120 test
  designs; the checkpoints are judged ONLY against bars from the 93-design post-exclusion set.
  Both were fixed before any checkpoint was evaluated.
- **Post-exclusion gray zero-shot (93 designs, 1018 queries):** main Rank-1 0.912
  [0.873, 0.948]; main TAR@1e-3 0.609 [0.546, 0.673]; tonal Rank-1 0.645 [0.581, 0.714];
  real_view_recolored Rank-1 0.864 [0.744, 0.944] on 15 designs, 44 queries.
- **Bars.** Success needs at least one of: tonal Rank-1 > **0.714**; main TAR@FAR=1e-3 >
  **0.673**; real_view_recolored Rank-1 > **0.944**. AND main Rank-1 >= **0.873**.
- **Reading the change:**
  - 13 of the 28 real multi-view designs were among the excluded leaks, leaving 15. Gray
    zero-shot rose to 0.864 there and the 0.944 bar is near-unreachable, so in practice the
    test rests on tonal Rank-1 or main TAR.
  - Gray main TAR fell 0.667 -> 0.609 after exclusion: the leaked designs were easy cases with
    a near-twin in train. Mild evidence the exclusions removed real inflation.
