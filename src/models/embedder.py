"""
Embedding model: DINOv2 ViT-S/14 backbone plus a projection head to a 128-d unit vector.

Why DINOv2. It is self-supervised, so its features were learned without class labels and
encode structure and texture rather than ImageNet category boundaries. That is a good prior for
textiles, where "which design" matters and "which object" does not.

Why freeze all but the last 4 blocks. Early ViT blocks learn generic edge and texture filters
that transfer well. Only the late blocks need to specialise to "which motif". Freezing the
rest protects the pretrained representation from being damaged by roughly 300 training
designs, which is far too few to retrain a ViT from scratch, and it cuts memory and time.

Why a 128-d L2-normalised output. Normalising puts every embedding on the unit sphere, so
cosine similarity is a dot product and the SupCon temperature has a stable meaning. 128 is
small enough for a cheap gallery index and large enough for a few hundred designs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONE_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
}


class ProjectionHead(nn.Module):
    """Linear(in, hidden) - BN - ReLU - Linear(hidden, out).

    A non-linear head is standard for contrastive learning: the loss is applied after the
    head, which lets the head absorb invariances specific to the training objective while the
    backbone keeps a richer, more general representation.
    """

    def __init__(self, in_dim: int = 384, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SareeEmbedder(nn.Module):
    def __init__(self, backbone: str = "dinov2_vits14", embed_dim: int = 128,
                 hidden_dim: int = 512, unfreeze_last_n_blocks: int = 4,
                 pretrained: bool = True):
        super().__init__()
        if backbone not in BACKBONE_DIMS:
            raise ValueError(f"unsupported backbone '{backbone}'; known: {sorted(BACKBONE_DIMS)}")
        self.backbone_name = backbone
        if pretrained:
            self.backbone = torch.hub.load("facebookresearch/dinov2", backbone, verbose=False)
        else:
            # Architecture only, for profiling and tests that must not hit the network.
            self.backbone = torch.hub.load("facebookresearch/dinov2", backbone,
                                           pretrained=False, verbose=False)
        self.feat_dim = BACKBONE_DIMS[backbone]
        self.head = ProjectionHead(self.feat_dim, hidden_dim, embed_dim)
        self.embed_dim = embed_dim
        self.set_trainable_blocks(unfreeze_last_n_blocks)

    def set_trainable_blocks(self, n: int) -> None:
        """Freeze everything, then unfreeze the last n transformer blocks and the final norm.

        The final norm is unfrozen with the late blocks because it normalises their output; a
        frozen norm over retrained blocks would apply statistics computed for the old features.
        """
        for p in self.backbone.parameters():
            p.requires_grad = False
        blocks = list(self.backbone.blocks)
        if n > 0:
            for blk in blocks[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.backbone.norm.parameters():
                p.requires_grad = True
        self.n_trainable_blocks = n

    def backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """The CLS token after the final norm: DINOv2's global image descriptor."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone_features(x)
        return F.normalize(self.head(feats), dim=1)

    def param_groups(self, lr_head: float, lr_backbone: float, weight_decay: float) -> list[dict]:
        """Separate learning rates: the head learns from scratch, the backbone only adapts.

        A fresh head needs a large learning rate to move at all, while pretrained blocks need a
        small one so they are refined rather than overwritten. Biases and norm parameters skip
        weight decay, since decaying them towards zero is not a sensible regulariser.
        """
        def split(params):
            decay, no_decay = [], []
            for name, p in params:
                if not p.requires_grad:
                    continue
                (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
            return decay, no_decay

        bb_decay, bb_nodecay = split(self.backbone.named_parameters())
        hd_decay, hd_nodecay = split(self.head.named_parameters())
        groups = [
            {"params": hd_decay, "lr": lr_head, "weight_decay": weight_decay, "name": "head"},
            {"params": hd_nodecay, "lr": lr_head, "weight_decay": 0.0, "name": "head_nd"},
            {"params": bb_decay, "lr": lr_backbone, "weight_decay": weight_decay,
             "name": "backbone"},
            {"params": bb_nodecay, "lr": lr_backbone, "weight_decay": 0.0,
             "name": "backbone_nd"},
        ]
        return [g for g in groups if g["params"]]

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        head = sum(p.numel() for p in self.head.parameters())
        return {"total": total, "trainable": trainable, "head": head,
                "backbone_frozen": total - trainable}
