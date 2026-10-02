"""
Supervised contrastive loss (Khosla et al., 2020), multi-positive form.

For each anchor row i in a batch of N L2-normalised embeddings:

    loss_i = - 1/|P(i)| * sum_{p in P(i)} log( exp(z_i.z_p / T) / sum_{a != i} exp(z_i.z_a / T) )

where P(i) is every OTHER row with the same label. With PK sampling (P designs, K views) each
anchor has exactly K-1 positives and everything else in the batch is a negative.

Why SupCon rather than ArcFace here. ArcFace learns one class centre per training design,
which needs several real images per class to place that centre well. This corpus has one real
image per design for most designs: positives come from augmentation, not from a population of
real images. SupCon learns directly from pairs inside the batch, with no per-class centre, so
it uses augmented positives naturally. It also never builds a classifier over the training
designs, which suits a task whose test designs are unseen by construction.

Why temperature 0.07. Small temperatures sharpen the softmax, so the loss concentrates on the
hardest negatives: the ones nearest the anchor. Our forced same-palette groups exist to supply
exactly those, so a sharp temperature is what lets them carry weight.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """embeddings: (N, D), expected L2-normalised. labels: (N,) int."""
        if embeddings.ndim != 2:
            raise ValueError(f"expected (N, D) embeddings, got shape {tuple(embeddings.shape)}")
        n = embeddings.shape[0]
        device = embeddings.device

        # Compute in float32 even under AMP. exp(sim / 0.07) spans e^-14 to e^+14, and the
        # log-sum-exp below is the classic place fp16 overflows or loses precision.
        z = embeddings.float()
        logits = (z @ z.T) / self.temperature

        self_mask = torch.eye(n, dtype=torch.bool, device=device)
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T) & ~self_mask

        # Numerical stability: subtracting the row max leaves the softmax unchanged but keeps
        # exp() in range. Detached because it is a constant shift, not a learnable quantity.
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        # The denominator excludes the anchor itself, as in the paper.
        exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        n_pos = pos_mask.sum(dim=1)
        has_pos = n_pos > 0
        if not has_pos.any():
            # No anchor has a positive: the batch carries no contrastive signal at all. Return
            # a zero that still participates in autograd, so the step is a clean no-op.
            return embeddings.sum() * 0.0

        mean_log_prob_pos = (log_prob * pos_mask).sum(dim=1)[has_pos] / n_pos[has_pos]
        return -mean_log_prob_pos.mean()
