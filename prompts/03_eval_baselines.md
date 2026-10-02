# Phase 3: Baselines and evaluation harness FIRST (target: ~1 h)

Rationale: build the ruler before building the model, so every later change is measured.

1. Implement src/eval/retrieval.py: given query and gallery embeddings + labels, compute Rank-1, Rank-5, mAP (cosine similarity).
2. Implement src/eval/verification.py: build positive/negative pairs from the test split, compute ROC-AUC, EER, TAR@FAR=1e-3, and pick a threshold on VAL (not test).
3. Implement src/eval/stress.py:
   - Test A: same design, synthetic new palette (must match).
   - Test B: different designs, forced into the same palette (must NOT match).
   Report the similarity distributions and pass rates.
4. scripts/evaluate.py: takes a checkpoint OR a "--zero-shot" backbone name, writes outputs/results_<name>.json and a markdown table.
5. Run zero-shot baselines (no training): ImageNet ResNet-50, DINOv2 ViT-S/14 on RGB, DINOv2 on grayscale. These numbers go in the README as the bar to beat.

Explain each metric to me in one or two sentences I could repeat in the interview. Log decisions. Quiz me with 3 questions.
