"""
Verification metrics: decide whether two images carry the same design.

Pairs are scored by cosine similarity. Genuine pairs share a design_id; impostor pairs do not.

  ROC-AUC         threshold-free separability of genuine versus impostor scores
  EER             the operating point where false accepts equal false rejects
  TAR @ FAR=1e-3  how many genuine pairs we accept when only 1 in 1000 impostors gets through

Why the threshold is chosen on VAL and only applied on test. Picking the threshold on the
same pairs it is evaluated on is a form of test-set tuning: it finds the best possible cut for
those specific scores and reports an optimistic number. A deployed system must commit to a
threshold before it sees new data, so we do the same.

On TAR @ FAR=1e-3 and sample size. Estimating a 1-in-1000 false accept rate needs on the order
of thousands of impostor pairs, otherwise the "threshold" is set by one or two outlying scores.
This module reports how many impostor pairs underlie the estimate so a reader can judge it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class PairSet:
    scores: np.ndarray        # cosine similarity per pair
    genuine: np.ndarray       # bool: same design
    q_design: np.ndarray      # query design per pair, so the bootstrap can resample designs


def build_pairs(query_emb: np.ndarray, query_designs: list[str], gallery_emb: np.ndarray,
                gallery_designs: list[str],
                restrict: np.ndarray | None = None) -> PairSet:
    """Every query against every gallery entry.

    `restrict` is an optional (n_query, n_gallery) boolean mask of pairs to keep. Stress test B
    uses it to keep only the genuine pairs plus impostors sharing the query's palette.
    """
    q = query_emb / np.maximum(np.linalg.norm(query_emb, axis=1, keepdims=True), 1e-12)
    g = gallery_emb / np.maximum(np.linalg.norm(gallery_emb, axis=1, keepdims=True), 1e-12)
    sims = q @ g.T
    qd = np.asarray(query_designs)
    gd = np.asarray(gallery_designs)
    genuine = qd[:, None] == gd[None, :]
    mask = np.ones_like(genuine, dtype=bool) if restrict is None else restrict
    return PairSet(scores=sims[mask], genuine=genuine[mask],
                   q_design=np.broadcast_to(qd[:, None], sims.shape)[mask])


def equal_error_rate(scores: np.ndarray, genuine: np.ndarray) -> tuple[float, float]:
    """EER and the threshold achieving it."""
    fpr, tpr, thr = roc_curve(genuine.astype(int), scores)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2.0), float(thr[i])


def threshold_at_far(scores: np.ndarray, genuine: np.ndarray, far: float) -> float:
    """Smallest threshold whose false accept rate does not exceed `far`.

    Taken directly from the impostor score distribution: the (1 - far) quantile of impostor
    scores, so that at most a `far` fraction of impostors land above it.
    """
    imp = np.sort(scores[~genuine])
    if len(imp) == 0:
        return float("inf")
    k = int(np.ceil((1.0 - far) * len(imp))) - 1
    k = min(max(k, 0), len(imp) - 1)
    return float(imp[k])


@dataclass
class VerificationResult:
    roc_auc: float
    eer: float
    tar_at_far: float
    far_target: float
    threshold: float
    achieved_far: float
    n_genuine: int
    n_impostor: int

    def as_row(self) -> dict:
        return {
            "ROC-AUC": f"{self.roc_auc:.4f}",
            "EER": f"{self.eer:.4f}",
            f"TAR@FAR={self.far_target:g}": f"{self.tar_at_far:.4f}",
            "threshold": f"{self.threshold:.4f}",
            "genuine": self.n_genuine,
            "impostor": self.n_impostor,
        }


def evaluate_verification(pairs: PairSet, far_target: float = 1e-3,
                          threshold: float | None = None) -> VerificationResult:
    """Compute verification metrics.

    Pass `threshold` (chosen on val) to evaluate test honestly. If omitted, the threshold is
    derived from these same pairs, which is only appropriate when these ARE the val pairs.
    """
    g = pairs.genuine.astype(bool)
    s = pairs.scores
    if g.all() or (~g).all():
        nan = float("nan")
        return VerificationResult(nan, nan, nan, far_target, nan, nan, int(g.sum()),
                                  int((~g).sum()))

    auc = float(roc_auc_score(g.astype(int), s))
    eer, _ = equal_error_rate(s, g)
    thr = threshold if threshold is not None else threshold_at_far(s, g, far_target)
    tar = float((s[g] >= thr).mean())
    achieved_far = float((s[~g] >= thr).mean())
    return VerificationResult(auc, eer, tar, far_target, float(thr), achieved_far,
                              int(g.sum()), int((~g).sum()))
