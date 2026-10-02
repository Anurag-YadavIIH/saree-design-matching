# Phase 5: Efficiency, inference, packaging (target: ~1.5 h)

1. scripts/profile.py: parameter count (total and trainable), FLOPs at 224x224 (fvcore or thop), latency (batch 1 and batch 32, GPU and CPU, median of 100 runs after warmup), embedding size in bytes (float32 and float16).
2. scripts/infer.py: given a query image and a gallery folder (or precomputed gallery embeddings), print top-5 matches with scores; given two images, print same/different with the VAL-chosen threshold.
3. notebooks/saree_reid_kaggle.ipynb: thin wrapper. Clones/attaches the repo, installs requirements, runs train -> evaluate -> profile -> infer demo. Must run top to bottom on Kaggle with only the dataset paths changed in one config cell.
4. README.md: problem framing, data handling (including that the Drive corpus is not redistributed), approach, evaluation protocol, results table (baselines vs ours vs ablations), efficiency table, how to reproduce, all pretrained checkpoints disclosed, limitations, extension to other garments.
5. Help me draft the 500-character approach note. I write the final wording; you check the character count and that it covers architecture, pre/post-processing, loss, sampling, augmentation. No em dashes.

Final check: fresh-clone dry run, make sure no data files are tracked by git, all links public.
