"""
Identification metrics: Rank-1, Rank-5, mAP, with bootstrap confidence intervals.

Protocol. The gallery holds one real representative per evaluation design. Queries are
recolored and geometrically perturbed views. A query is correct when the retrieved gallery
entry carries the same `design_id`.

Why the bootstrap resamples DESIGNS and not queries. Each design contributes 5 or more
queries derived from the SAME parent image, so those queries are strongly correlated: if the
model fails on one, it very likely fails on its siblings. Resampling queries would treat them
as independent, understate the variance and produce an interval that is too narrow. Resampling
designs respects the correlation. With roughly 120 test designs the interval is wide, and
reporting it honestly is the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RetrievalResult:
    rank1: float
    rank5: float
    map: float
    n_queries: int
    n_gallery: int
    n_designs: int
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)

    def as_row(self) -> dict:
        def fmt(k: str, v: float) -> str:
            if k in self.ci:
                lo, hi = self.ci[k]
                return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"
            return f"{v:.3f}"
        return {
            "Rank-1": fmt("rank1", self.rank1),
            "Rank-5": fmt("rank5", self.rank5),
            "mAP": fmt("map", self.map),
            "queries": self.n_queries,
            "gallery": self.n_gallery,
            "designs": self.n_designs,
        }


def cosine_similarity(queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Queries x gallery cosine. Inputs are expected L2 normalised; normalise defensively.

    Defensive because a baseline embedding (a colour histogram, say) may not arrive
    normalised, and silently comparing unnormalised vectors would measure magnitude rather
    than direction.
    """
    q = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    g = gallery / np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12)
    return q @ g.T


def _per_query_metrics(sims: np.ndarray, q_designs: np.ndarray,
                       g_designs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-query hit@1, hit@5 and average precision.

    Average precision is computed over all gallery entries sharing the query's design. With
    one gallery entry per design that reduces to reciprocal rank, but the general form is
    used so the code stays correct if a multi-entry gallery is ever introduced.
    """
    order = np.argsort(-sims, axis=1)
    ranked = g_designs[order]
    match = ranked == q_designs[:, None]

    hit1 = match[:, 0].astype(np.float64)
    hit5 = match[:, :5].any(axis=1).astype(np.float64)

    aps = np.zeros(len(sims), dtype=np.float64)
    for i in range(len(sims)):
        rel = match[i]
        n_rel = int(rel.sum())
        if n_rel == 0:
            continue              # design absent from the gallery; AP is 0 by definition
        ranks = np.flatnonzero(rel) + 1
        precision_at_hits = np.arange(1, n_rel + 1) / ranks
        aps[i] = precision_at_hits.mean()
    return hit1, hit5, aps


def evaluate_retrieval(query_emb: np.ndarray, query_designs: list[str],
                       gallery_emb: np.ndarray, gallery_designs: list[str],
                       n_bootstrap: int = 1000, confidence: float = 0.95,
                       seed: int = 42) -> RetrievalResult:
    q_designs = np.asarray(query_designs)
    g_designs = np.asarray(gallery_designs)
    sims = cosine_similarity(query_emb, gallery_emb)
    hit1, hit5, aps = _per_query_metrics(sims, q_designs, g_designs)

    result = RetrievalResult(
        rank1=float(hit1.mean()), rank5=float(hit5.mean()), map=float(aps.mean()),
        n_queries=len(q_designs), n_gallery=len(g_designs),
        n_designs=int(len(np.unique(q_designs))),
    )

    if n_bootstrap > 0:
        result.ci = bootstrap_ci_over_designs(
            {"rank1": hit1, "rank5": hit5, "map": aps},
            q_designs, n_bootstrap=n_bootstrap, confidence=confidence, seed=seed)
    return result


def bootstrap_ci_over_designs(per_query: dict[str, np.ndarray], query_designs: np.ndarray,
                              n_bootstrap: int = 1000, confidence: float = 0.95,
                              seed: int = 42) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap, resampling DESIGNS with replacement.

    Each resample draws designs, then takes ALL queries belonging to the drawn designs. That
    keeps sibling queries together, which is exactly the correlation we must not pretend away.
    """
    rng = np.random.default_rng(seed)
    designs = np.unique(query_designs)
    idx_by_design = {d: np.flatnonzero(query_designs == d) for d in designs}

    draws: dict[str, list[float]] = {k: [] for k in per_query}
    for _ in range(n_bootstrap):
        picked = rng.choice(len(designs), size=len(designs), replace=True)
        idx = np.concatenate([idx_by_design[designs[i]] for i in picked])
        for k, vals in per_query.items():
            draws[k].append(float(vals[idx].mean()))

    alpha = (1.0 - confidence) / 2.0
    return {k: (float(np.percentile(v, 100 * alpha)),
                float(np.percentile(v, 100 * (1 - alpha))))
            for k, v in draws.items()}
