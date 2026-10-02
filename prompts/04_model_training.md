# Phase 4: Model, loss, training (target: ~2.5 h incl. a Kaggle run)

Before coding, give me a short comparison and wait for my pick:
- Backbone: DINOv2 ViT-S/14 vs EfficientNet-B0 vs MobileNetV3 (accuracy vs params/FLOPs vs Kaggle time).
- Loss: ArcFace vs SupCon vs Triplet with batch-hard mining (what each needs from the sampler, stability, how they behave with few images per design).
- Freeze strategy: frozen backbone + head, vs partial unfreeze (last N blocks), vs full fine-tune.

Then implement:
1. src/models/embedder.py: backbone + projection head (Linear -> BN -> 128-d, L2-normalized). Config-driven.
2. src/losses/: the chosen loss (and optionally one alternative for an ablation).
3. scripts/train.py: AMP, cosine LR with warmup, gradient clipping, seed control, checkpoint best on VAL mAP, logs to outputs/train_log.csv. Config from configs/default.yaml.
4. A 2-minute smoke test on a tiny subset (CPU OK) before the real run.
5. Tell me exactly what to run on Kaggle and expected time.

After the run: compare against Phase 3 baselines. Plan at most 2 ablations (e.g. with vs without palette swap; RGB vs structure input) that fit the time left.

Log decisions. Quiz me with 3 questions.
