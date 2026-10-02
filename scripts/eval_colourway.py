"""
Verification on REAL colourway pairs mined from the Kaggle catalogue.

Positives: pairs judged "same print, different colourway" in outputs/real_colourway_pairs.csv.
Negatives: every pair of Kaggle representatives in the SAME craft family that is not linked by
an accepted pair. Same family makes the negatives hard: similar technique, different print.

Excluded from negatives, so that likely-true pairs are never scored as impostors:
  - pairs connected through accepted pairs (if A~B and B~C, A-C is probably the same print)
  - unreviewed mined candidates (high-similarity pairs nobody has judged yet)

Reported: ROC-AUC and TAR at FAR = 1e-2, with 95% bootstrap CIs (positives and negatives
resampled independently). With about 30 positives, FAR = 1e-3 would rest on a handful of
negative scores, so 1e-2 is the honest operating point.

Caveats printed with the results:
  - SELECTION BIAS: candidates were mined as gray zero-shot DINOv2's own top neighbours, so
    every positive is one that model already ranks highly. This inflates gray zero-shot and
    anything correlated with it.
  - Pairs with both sides in train were trained as NEGATIVES (different design_ids), which
    pushes the fine-tuned model to separate exactly these pairs: it is underestimated.
  - The threshold for TAR is chosen on these same pairs (no separate validation set).

Run:  python scripts/eval_colourway.py [--checkpoints outputs/run_rgb/best.pt ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import DinoZeroShot, embed_color_hist, embed_model, embed_phash  # noqa: E402
from src.data.manifest import load_manifest  # noqa: E402

FAR = 1e-2


def tar_at_far(pos: np.ndarray, neg: np.ndarray, far: float) -> float:
    thr = np.quantile(neg, 1.0 - far)
    return float((pos >= thr).mean())


def with_ci(pos, neg, fn, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    point = fn(pos, neg)
    draws = []
    for _ in range(n_boot):
        draws.append(fn(pos[rng.integers(0, len(pos), len(pos))],
                        neg[rng.integers(0, len(neg), len(neg))]))
    return point, (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def auc_fn(pos, neg):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="outputs/real_colourway_pairs.csv")
    ap.add_argument("--candidates", default="outputs/colourway_candidates.csv")
    ap.add_argument("--checkpoints", nargs="*",
                    default=["outputs/run_rgb/best.pt", "outputs/run_gray/best.pt"])
    ap.add_argument("--out", default="outputs/results_colourway")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = pd.read_csv(args.pairs)
    acc = pairs[pairs.decision == "accept"]
    reviewed = set(zip(pairs.design_a, pairs.design_b))
    unreviewed = {(a, b) for a, b in zip(*pd.read_csv(args.candidates)[["design_a", "design_b"]]
                                          .values.T) if (a, b) not in reviewed}

    reps = load_manifest(source="kaggle_fabric", include_synthetic=False,
                         representatives_only=True).reset_index(drop=True)
    idx = {d: i for i, d in enumerate(reps.design_id)}

    # Connected components of accepted pairs: never score two members of one as impostors.
    parent = list(range(len(reps)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip(acc.design_a, acc.design_b):
        parent[find(idx[a])] = find(idx[b])

    pos = [(idx[a], idx[b]) for a, b in zip(acc.design_a, acc.design_b)]
    fam = reps.craft_family.to_numpy()
    neg = []
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            if fam[i] != fam[j] or find(i) == find(j):
                continue
            key = tuple(sorted((reps.design_id[i], reps.design_id[j])))
            if key in unreviewed:
                continue
            neg.append((i, j))
    pos, neg = np.array(pos), np.array(neg)
    both_train = int(((acc.split_a == "train") & (acc.split_b == "train")).sum())
    print(f"[colourway] {len(pos)} positive pairs ({both_train} with both sides in train), "
          f"{len(neg):,} same-family negatives, {len(unreviewed)} unreviewed candidates excluded")

    paths = [Path("data") / p for p in reps.path]
    methods = [("grayscale pHash", embed_phash), ("RGB colour histogram", embed_color_hist)]
    dino = DinoZeroShot().to(device)
    methods += [("DINOv2 zero-shot RGB", lambda ps: embed_model(ps, dino, device)),
                ("DINOv2 zero-shot gray", lambda ps: embed_model(ps, dino, device, grayscale=True))]
    from src.models.embedder import SareeEmbedder
    for ck in args.checkpoints:
        if not Path(ck).exists():
            print(f"[colourway] checkpoint {ck} not found; skipping")
            continue
        st = torch.load(ck, map_location=device, weights_only=False)
        mc = st["config"]["model"]
        m = SareeEmbedder(mc["backbone"], int(mc["embed_dim"]), int(mc["hidden_dim"]),
                          int(mc["unfreeze_last_n_blocks"])).to(device)
        m.load_state_dict(st["model"])
        g = mc.get("input_mode", "rgb") == "gray"
        methods.append((f"trained {'gray' if g else 'RGB'}",
                        lambda ps, m=m, g=g: embed_model(ps, m, device, grayscale=g)))

    results = []
    for name, fn in methods:
        e = np.asarray(fn(paths), dtype=np.float32)
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        sp = np.sum(e[pos[:, 0]] * e[pos[:, 1]], axis=1)
        sn = np.sum(e[neg[:, 0]] * e[neg[:, 1]], axis=1)
        auc, auc_ci = with_ci(sp, sn, auc_fn)
        tar, tar_ci = with_ci(sp, sn, lambda p, n: tar_at_far(p, n, FAR))
        results.append({"method": name, "roc_auc": auc, "roc_auc_ci": auc_ci,
                        f"tar_at_far_{FAR:g}": tar, "tar_ci": tar_ci})
        print(f"[colourway] {name:24s} AUC {auc:.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]  "
              f"TAR@FAR={FAR:g} {tar:.3f} [{tar_ci[0]:.3f}, {tar_ci[1]:.3f}]")

    md = ["# Real colourway verification (Kaggle)", "",
          f"{len(pos)} positive pairs judged same print / different colourway; {len(neg):,} "
          "same-craft-family negatives. 95% CIs from 1000 bootstrap resamples.", "",
          f"| method | ROC-AUC | TAR@FAR={FAR:g} |", "|---|---|---|"]
    for r in results:
        md.append(f"| {r['method']} | {r['roc_auc']:.3f} [{r['roc_auc_ci'][0]:.3f}, "
                  f"{r['roc_auc_ci'][1]:.3f}] | {r[f'tar_at_far_{FAR:g}']:.3f} "
                  f"[{r['tar_ci'][0]:.3f}, {r['tar_ci'][1]:.3f}] |")
    md += ["", "**Caveats.**",
           "- Selection bias: candidates were mined as gray zero-shot DINOv2's top neighbours, "
           "so positives favour that model and anything correlated with it.",
           f"- {both_train} of {len(pos)} pairs had both sides in train with different design_ids, "
           "so SupCon was trained to push them APART: the fine-tuned models are underestimated.",
           "- The TAR threshold is set on these same negatives; there is no separate validation set.",
           f"- Only the top 60 of {len(pairs) + len(unreviewed)} mined candidates were reviewed."]
    Path(args.out).with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    Path(args.out).with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[colourway] wrote {args.out}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
