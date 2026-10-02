"""
PK sampler: P designs per batch, K views each, plus optional same-palette hard negatives.

Why PK rather than plain shuffling. SupCon learns from the structure inside a batch: rows
sharing a label pull together, rows with different labels push apart. A uniformly shuffled
batch of 128 images drawn from ~400 designs would contain mostly singletons, so there would
be almost no positive pairs to learn from. Sampling P designs and K views of each guarantees
exactly K-1 positives per row.

The same-palette option is the other half of the core requirement. The brief demands that
different motifs in identical palettes must NOT match. If every design in a batch wears a
different random palette, the easiest way to separate them is colour, and the model will take
that shortcut. Forcing a group of different designs onto ONE palette removes the shortcut: in
that group colour is uninformative, so only the motif can tell them apart.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class PKDesignSampler(Sampler[list[tuple[int, str | None]]]):
    """Yields batches of P (design index, forced palette or None) pairs.

    The dataset expands each pair into K views. This is a BATCH sampler, so it goes to
    `DataLoader(batch_sampler=...)`, not `sampler=...`.

    Why the palette travels inside the index. The obvious design, where the training loop sets
    a "forced palette" attribute on the dataset before each batch, is silently broken with
    `num_workers > 0`: every DataLoader worker holds its own copy of the dataset, so an
    attribute set in the main process never reaches them, and the hard negatives would simply
    never happen. Carrying the palette in the sampled index is the only route guaranteed to
    reach the worker that builds the view.
    """

    def __init__(self, design_ids: list[str], p_designs: int, batches_per_epoch: int,
                 seed: int = 42, assigner: "SamePaletteGroupAssigner | None" = None):
        self.design_ids = list(design_ids)
        self.n_designs = len(self.design_ids)
        self.p = min(p_designs, self.n_designs)
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.assigner = assigner
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Reseeds so each epoch draws different design combinations, reproducibly."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        for _ in range(self.batches_per_epoch):
            idx = [int(i) for i in rng.choice(self.n_designs, size=self.p, replace=False)]
            forced: dict[str, str] = {}
            if self.assigner is not None:
                forced = self.assigner.assign([self.design_ids[i] for i in idx])
            yield [(i, forced.get(self.design_ids[i])) for i in idx]


class SamePaletteGroupAssigner:
    """Decides, per batch, which designs are forced to share a palette.

    Applied by the training loop before the batch is fetched, because the palette must be
    known at augmentation time. With probability `prob`, one group of `group_size` designs in
    the batch is pinned to a single palette drawn from the TRAIN bank.
    """

    def __init__(self, palette_ids: list[str], prob: float = 0.3, group_size: int = 4,
                 seed: int = 42):
        self.palette_ids = list(palette_ids)
        self.prob = prob
        self.group_size = group_size
        self.rng = np.random.default_rng(seed)

    def assign(self, design_ids: list[str]) -> dict[str, str]:
        """Return design_id to palette_id for the forced subset. Empty means no forcing.

        Only the forced designs appear in the mapping. Everything else keeps its default
        random palette, so the batch still contains ordinary easy negatives alongside the
        hard same-palette group.
        """
        if not self.palette_ids or self.rng.random() >= self.prob:
            return {}
        n = min(self.group_size, len(design_ids))
        if n < 2:
            return {}
        chosen = self.rng.choice(len(design_ids), size=n, replace=False)
        palette = self.palette_ids[int(self.rng.integers(len(self.palette_ids)))]
        return {design_ids[int(i)]: palette for i in chosen}
