"""
Stress tests: the two halves of the brief's core requirement, measured directly.

  Stress A   same design, new palette     MUST match
  Stress B   different design, same palette  MUST NOT match

Both are expressed as queries over the manifest rather than as separate data files, which is
why `palette_id` is a manifest column.

Stress A is the ordinary retrieval protocol, since every query is a held-out recolour of its
design. It is reported here for completeness and broken down by query kind.

Stress B is where a colour-reliant model is exposed. For each query we keep its genuine
gallery pair plus ONLY the impostors that share its palette. A model that matches on colour
will score those impostors highly, so stress B separability collapses for it even when overall
retrieval looks fine. Two tiers:

  tier 1  same palette, any design
  tier 2  same palette AND same craft family (Kaggle only). The hardest available case:
          colour and craft are both held fixed, so only the motif can separate the pair.

Implementation note. The gallery holds REAL images, which have no palette_id. Same-palette
impostors therefore come from comparing a query against the OTHER QUERIES that wear its
palette, using their parent designs as identity. That is the faithful version of "two
different designs rendered in identical palettes".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.eval.verification import equal_error_rate


def _normalise(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def stress_b(query_emb: np.ndarray, query_df: pd.DataFrame, tier: int = 1,
             far_target: float = 1e-3, threshold: float | None = None) -> dict:
    """Same-palette impostor discrimination among queries.

    Genuine pairs: two queries of the SAME design (any palettes).
    Impostor pairs: two queries of DIFFERENT designs wearing the SAME palette_id, and for
    tier 2 also the same craft_family.
    """
    df = query_df.reset_index(drop=True)
    has_pal = df["palette_id"].astype(str) != ""
    df = df[has_pal].reset_index(drop=True)
    emb = _normalise(query_emb[has_pal.values])
    if len(df) < 4:
        return {"tier": tier, "note": "too few recoloured queries"}

    sims = emb @ emb.T
    design = df["design_id"].to_numpy()
    pal = df["palette_id"].to_numpy()
    fam = df["craft_family"].to_numpy()

    iu = np.triu_indices(len(df), k=1)
    same_design = design[iu[0]] == design[iu[1]]
    same_pal = pal[iu[0]] == pal[iu[1]]
    impostor = (~same_design) & same_pal
    if tier == 2:
        same_fam = (fam[iu[0]] == fam[iu[1]]) & (fam[iu[0]] != "")
        impostor &= same_fam

    keep = same_design | impostor
    s = sims[iu][keep]
    g = same_design[keep]

    n_imp, n_gen = int((~g).sum()), int(g.sum())
    if n_imp == 0 or n_gen == 0:
        return {"tier": tier, "genuine": n_gen, "impostor": n_imp,
                "note": "no impostor pairs at this tier"}

    auc = float(roc_auc_score(g.astype(int), s))
    eer, _ = equal_error_rate(s, g)
    # Of the same-palette impostors, how many would be wrongly ACCEPTED at the operating
    # threshold? This is the number the brief cares about most.
    false_accept = float((s[~g] >= threshold).mean()) if threshold is not None else float("nan")
    # Mean score gap: how far apart, on average, genuine and same-palette impostor pairs sit.
    margin = float(s[g].mean() - s[~g].mean())
    return {
        "tier": tier,
        "ROC-AUC": round(auc, 4),
        "EER": round(eer, 4),
        "same_palette_false_accept_rate": (round(false_accept, 4)
                                           if false_accept == false_accept else None),
        "genuine_minus_impostor_margin": round(margin, 4),
        "genuine": n_gen,
        "impostor": n_imp,
    }
