# Phase 2: Color augmentations and synthetic colorways (target: ~1 h)

Before coding, explain to me in plain language:
- Why plain ColorJitter is not enough for this task.
- How k-means palette swap works and why it creates both hard positives (same design, new palette) and hard negatives (different designs, same palette).
- The risk that the model learns "grayscale structure" shortcuts, and how we check for it.

Then implement src/data/color_aug.py with:
1. hue_rotate(img, degrees)
2. channel_permute(img)
3. palette_swap(img, k=6, target_palette=None): k-means on pixel colors, remap clusters to a target palette (random or given). Keep it fast (subsample pixels for k-means, cache if needed).
4. to_structure(img): grayscale + Sobel magnitude as a 2 or 3 channel input option.
5. A unit test script (scripts/test_aug.py) that runs on 4 images and saves a grid to outputs/aug_preview.png for MY eyes only (gitignored).

Then src/data/dataset.py: a Dataset reading outputs/splits.csv, and src/data/sampler.py: a PK batch sampler (P designs x K images), with an option to force some K slots to be synthetic recolors and to inject "same palette, different design" negatives.

Log decisions. Quiz me with 3 questions.
