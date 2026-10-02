# CLAUDE.md: Color-Invariant Saree Design Recognition (DeepLure AIE-CASE)

## Context
Take-home assessment for DeepLure. Deadline: **11:59 AM IST, 2 Oct 2026**.
Goal: identify a saree by its surface design regardless of color palette ("face recognition for textiles").
- Identification: rank a gallery by similarity to a query.
- Verification: decide if two images carry the same design.
- Core requirement: same motif in different palettes MUST match; different motifs in identical palettes MUST NOT match.

Deliverables: 500-char approach note, end-to-end PyTorch code (train + inference, runnable as-is on Kaggle free GPU), evaluation protocol + results, efficiency report (params, FLOPs, latency, embedding size).

## How to work with me (Anurag)
The brief says all work must be my own and I will defend every decision live, from memory. So:
1. **Explain before you write.** For every non-trivial choice, give me the options, the trade-off, and your recommendation in plain language. Wait for my go-ahead on architecture, loss, split, and metrics.
2. **Log every decision** in `DECISIONS.md` (what, why, alternatives rejected, evidence). This file is my study guide for the interview.
3. **Small steps.** One module at a time, each runnable and tested before moving on.
4. **No magic.** Prefer readable code over clever code. Comment the *why*, not the *what*.
5. **Quiz me** at the end of each phase with 3 questions an interviewer might ask.
6. Never use em dashes in any writing (docs, comments, the approach note).

## Hard constraints
- Framework: PyTorch only. Pretrained backbones allowed; disclose every checkpoint used in README.
- Must run on Kaggle free GPU (T4/P100, ~16 GB VRAM, ~9-12 h session limit). Keep training under ~1.5 h.
- The DeepLure Drive corpus is proprietary: NEVER commit it, upload it, or copy images into outputs. `data/` is gitignored.
- Seed everything; results must be reproducible.

## Planned approach (to be confirmed with me)
- Backbone: DINOv2 ViT-S/14 (or EfficientNet-B0 for the lean variant) + projection head to 128-d L2-normalized embedding.
- Color invariance: (a) input-level: grayscale + gradient/edge channel option; (b) augmentation: hue rotation, channel permutation, k-means palette swap.
- Loss: ArcFace or SupCon with design ID as class; PK sampling (P designs x K images).
- Hard negatives: different designs recolored into the same palette.
- Synthetic colorways via palette swap if real colorway pairs are scarce (disclose).

## Evaluation protocol
- Split by design ID (test designs never seen in training).
- Gallery: one colorway per design; queries: remaining colorways (+ synthetic recolors).
- Identification: Rank-1, Rank-5, mAP. Verification: ROC-AUC, EER, TAR@FAR=1e-3.
- Stress tests: (A) same design / new palette, (B) different design / same palette.
- Baselines to beat: raw ImageNet/DINOv2 embeddings, RGB vs grayscale.

## Layout
```
configs/      YAML configs
src/data/     dataset, splits, samplers, color augmentations
src/models/   backbone + embedding head
src/losses/   ArcFace / SupCon
src/eval/     retrieval + verification metrics, stress tests
src/utils/    seeding, logging, efficiency profiling
scripts/      train.py, evaluate.py, infer.py, profile.py
notebooks/    final Kaggle notebook (thin wrapper over src/)
prompts/      phase prompts for Claude Code
```
