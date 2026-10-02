"""
Train the colour-invariant saree embedder. Config driven: configs/train.yaml.

Loop in one sentence: draw P designs x K views, recolour each view with the TRAIN palette bank
(sometimes forcing a group of designs onto one palette), apply realistic geometry, embed, apply
SupCon, and select the checkpoint on held-out-method val mAP.

Run:
  python scripts/train.py                               # full run
  python scripts/train.py --smoke                       # 2 minute check on 20 designs
  python scripts/train.py --max-minutes 40 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.color_aug import (  # noqa: E402
    GeometryConfig,
    apply_geometry,
    build_palette_bank,
    kmeans_palette_swap,
    load_or_build_cluster_map,
    split_palette_bank,
    tonal_remap,
)
from src.data.dataset import (  # noqa: E402
    SareeDataset,
    TrainViewDataset,
    collate_views,
    to_tensor,
)
from src.data.manifest import load_label_map, load_manifest  # noqa: E402
from src.data.sampler import PKDesignSampler, SamePaletteGroupAssigner  # noqa: E402
from src.eval.retrieval import evaluate_retrieval  # noqa: E402
from src.losses.supcon import SupConLoss  # noqa: E402
from src.models.embedder import SareeEmbedder  # noqa: E402


def seed_everything(seed: int) -> None:
    """Seed every RNG that touches training, so a run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init(worker_id: int) -> None:
    """Give each DataLoader worker its own deterministic numpy and python seed.

    Without this, forked workers inherit identical RNG state and produce correlated
    augmentations, which quietly shrinks the effective diversity of every batch.
    """
    base = torch.initial_seed() % (2 ** 31)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


# ---------------------------------------------------------------------------- transforms

class TrainViewTransform:
    """Recolour with the TRAIN bank, then geometry. Picklable, so it survives worker fork."""

    def __init__(self, palettes: list[dict], geometry: GeometryConfig, cache_dir: str,
                 data_root: str, k_colors: int, tonal_prob: float = 0.0,
                 tonal_invert_prob: float = 0.5):
        self.tonal_prob = tonal_prob
        self.tonal_invert_prob = tonal_invert_prob
        self.palettes = palettes
        self.by_id = {p["palette_id"]: p for p in palettes}
        self.geometry = geometry
        self.cache_dir = cache_dir
        self.data_root = data_root
        self.k_colors = k_colors

    def __call__(self, rgb: np.ndarray, seed: int, palette_id: str | None,
                 rel_path: str) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if palette_id is not None and palette_id in self.by_id:
            pal = self.by_id[palette_id]                  # forced same-palette hard negative
        else:
            pal = self.palettes[int(rng.integers(len(self.palettes)))]
        cmap, _ = load_or_build_cluster_map(Path(self.data_root) / rel_path, rel_path,
                                            Path(self.cache_dir), self.k_colors, seed=0)
        if rng.random() < self.tonal_prob:
            # Lightness changes too, and dark/light order may invert: the only training views
            # on which a grayscale shortcut fails.
            recolored, _ = tonal_remap(rgb, cmap, pal["colors"], rng,
                                       invert_prob=self.tonal_invert_prob)
        else:
            recolored = kmeans_palette_swap(rgb, cmap, pal["colors"], rng)
        out, _ = apply_geometry(recolored, self.geometry, rng)
        return out


# ---------------------------------------------------------------------------- evaluation

@torch.no_grad()
def embed_dataset(model: torch.nn.Module, ds, device: str, batch_size: int,
                  num_workers: int) -> tuple[np.ndarray, list[str]]:
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    embs, designs = [], []
    for b in loader:
        x = b["image"].to(device, non_blocking=True)
        embs.append(model(x).float().cpu().numpy())
        designs.extend(b["design_id"])
    return np.concatenate(embs) if embs else np.zeros((0, 1)), designs


class InDistributionQueries(torch.utils.data.Dataset):
    """Val gallery images recoloured with the TRAINING method and bank, fixed seed.

    Exists only to measure the generalisation gap: in-distribution mAP minus held-out mAP is
    how much the model leans on our own generator. Never used to select a checkpoint.
    """

    def __init__(self, gallery_df, data_root: Path, palettes, geometry, cache_dir,
                 k_colors, n_per_design: int, img_size: int, seed: int,
                 grayscale: bool = False, tonal_prob: float = 0.0):
        self.items = []
        for _, row in gallery_df.iterrows():
            for q in range(n_per_design):
                self.items.append((row["path"], row["design_id"], seed + q))
        self.data_root = data_root
        # Same mixture as training, so this really is the in-distribution reference.
        self.tf = TrainViewTransform(palettes, geometry, cache_dir, str(data_root), k_colors,
                                     tonal_prob=tonal_prob)
        self.img_size = img_size
        self.grayscale = grayscale

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        from src.data.dataset import read_rgb
        rel, design, s = self.items[i]
        rgb = self.tf(read_rgb(self.data_root / rel), s, None, rel)
        return {"image": to_tensor(rgb, self.img_size, self.grayscale), "label": -1,
                "design_id": design, "image_id": f"{rel}#{s}"}


def run_validation(model, val_sets: dict, device: str, cfg: dict) -> dict[str, float]:
    bs = int(cfg["eval"]["batch_size"])
    nw = int(cfg["loader"]["num_workers"])
    g_emb, g_des = embed_dataset(model, val_sets["gallery"], device, bs, nw)
    out: dict[str, float] = {}
    for name in ("heldout", "tonal", "indist"):
        if name not in val_sets:
            continue
        q_emb, q_des = embed_dataset(model, val_sets[name], device, bs, nw)
        if len(q_des) == 0:
            continue
        r = evaluate_retrieval(q_emb, q_des, g_emb, g_des, n_bootstrap=0)
        out[f"val_map_{name}"] = r.map
        out[f"val_rank1_{name}"] = r.rank1
    parts = [out[k] for k in ("val_map_heldout", "val_map_tonal") if k in out]
    out["val_map_select"] = float(np.mean(parts)) if parts else -1.0
    model.train()
    return out


# ------------------------------------------------------------------------------ schedule

def lr_lambda_factory(total_steps: int, warmup_frac: float):
    """Linear warmup then cosine decay to zero.

    Warmup matters here specifically: the head starts random, and its large early gradients
    would otherwise flow straight into the freshly unfrozen backbone blocks and damage the
    pretrained features before the head has learned anything useful.
    """
    warmup = max(1, int(total_steps * warmup_frac))

    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return f


# ---------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="2 minute run on 20 designs; must show the loss decreasing")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--input", choices=["rgb", "gray"], default=None,
                    help="override model.input_mode")
    ap.add_argument("--run-name", default=None,
                    help="outputs go to outputs/<run-name>/, so two runs never overwrite")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg = yaml.safe_load(Path(cfg["data_config"]).read_text(encoding="utf-8"))
    if args.input is not None:
        cfg["model"]["input_mode"] = args.input
    if args.run_name:
        cfg["output"]["dir"] = f"outputs/{args.run_name}"
        cfg["output"]["log_csv"] = f"outputs/{args.run_name}/train_log.csv"
        cfg["output"]["checkpoint"] = f"outputs/{args.run_name}/best.pt"
    gray = cfg["model"].get("input_mode", "rgb") == "gray"
    tonal_prob = float(cfg["augment"].get("tonal_prob", 0.0))
    seed = int(cfg["seed"])
    seed_everything(seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["optim"]["amp"]) and device == "cuda"
    if args.num_workers is not None:
        cfg["loader"]["num_workers"] = args.num_workers

    max_minutes = args.max_minutes if args.max_minutes is not None else cfg["budget"]["max_minutes"]
    max_steps = args.max_steps if args.max_steps is not None else cfg["budget"]["max_steps"]
    eval_every = int(cfg["eval"]["every_steps"])
    p_designs = int(cfg["batch"]["p_designs"])
    k_views = int(cfg["batch"]["k_views"])

    if args.smoke:
        # Small enough for a CPU in about two minutes, large enough that the loss has
        # somewhere to go.
        max_minutes, max_steps, eval_every = 2.0, 40, 8
        p_designs, k_views = 8, 4

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_cfg["paths"]["data_root"])
    img_size = int(cfg["model"]["img_size"])

    # ---- palettes. Training sees ONLY the train bank.
    rc = data_cfg["recolor"]
    k_colors = int(rc["palette_bank"]["k_colors"])
    bank = build_palette_bank(rc["palette_bank"]["size"], k_colors, seed=data_cfg["seed"])
    pools = split_palette_bank(bank, rc["palette_bank"]["split"], seed=data_cfg["seed"])

    geo_train = GeometryConfig.from_dict(
        {**data_cfg["geometry"], "crop_scale": cfg["augment"]["train_crop_scale"]},
        out_size=img_size)
    geo_eval = GeometryConfig.from_dict(data_cfg["geometry"], out_size=img_size)

    # ---- data, read ONLY through the manifest loader.
    label_map = load_label_map(cfg["data_config"])
    train_df = load_manifest(split="train", include_synthetic=False,
                             config_path=cfg["data_config"])
    if args.smoke:
        keep = sorted(train_df["design_id"].unique())[:20]
        train_df = train_df[train_df["design_id"].isin(keep)]

    view_tf = TrainViewTransform(pools["train"], geo_train, "data/cache", str(data_root),
                                 k_colors, tonal_prob=tonal_prob,
                                 tonal_invert_prob=float(cfg["augment"].get(
                                     "tonal_invert_prob", 0.5)))
    train_ds = TrainViewDataset(train_df, data_root, label_map, k_views, view_tf,
                                img_size=img_size, seed=seed, grayscale=gray)
    assigner = SamePaletteGroupAssigner(
        [p["palette_id"] for p in pools["train"]],
        prob=float(cfg["batch"]["same_palette_prob"]),
        group_size=int(cfg["batch"]["same_palette_group_size"]), seed=seed)
    sampler = PKDesignSampler(train_ds.designs, p_designs, batches_per_epoch=10 ** 9,
                              seed=seed, assigner=assigner)
    nw = int(cfg["loader"]["num_workers"])
    train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate_views,
                              num_workers=nw, worker_init_fn=worker_init,
                              persistent_workers=nw > 0, pin_memory=device == "cuda")

    # ---- validation: held-out method (selection) and in-distribution (diagnostic only).
    val_gallery_df = load_manifest(split="val", role="gallery", config_path=cfg["data_config"])
    val_query_df = load_manifest(split="val", role="query", config_path=cfg["data_config"])
    if args.smoke:
        keep_v = sorted(val_gallery_df["design_id"].unique())[:12]
        val_gallery_df = val_gallery_df[val_gallery_df["design_id"].isin(keep_v)]
        val_query_df = val_query_df[val_query_df["design_id"].isin(keep_v)]
    recolor_q = val_query_df[val_query_df["query_kind"] == "synthetic_recolor"]
    tonal_q = val_query_df[val_query_df["query_kind"] == "synthetic_tonal"]
    val_sets = {
        "gallery": SareeDataset(val_gallery_df, data_root, img_size=img_size, grayscale=gray),
        "heldout": SareeDataset(recolor_q, data_root, img_size=img_size, grayscale=gray),
        "indist": InDistributionQueries(
            val_gallery_df, data_root, pools["train"], geo_eval, "data/cache", k_colors,
            n_per_design=int(data_cfg["eval"]["queries_per_design"]), img_size=img_size,
            seed=int(rc["val_in_distribution_diagnostic"]["seed"]),
            grayscale=gray, tonal_prob=tonal_prob),
    }
    if len(tonal_q):
        val_sets["tonal"] = SareeDataset(tonal_q, data_root, img_size=img_size, grayscale=gray)
    print(f"[train] input={'gray' if gray else 'rgb'} tonal_prob={tonal_prob} "
          f"select={cfg['eval']['select_metric']}")
    print(f"[train] device={device} amp={use_amp} | train designs={len(train_ds.designs)} "
          f"| val gallery={len(val_gallery_df)} heldout queries={len(val_query_df)} "
          f"| batch={p_designs}x{k_views}={p_designs * k_views}")

    # ---- model, loss, optimiser
    model = SareeEmbedder(cfg["model"]["backbone"], int(cfg["model"]["embed_dim"]),
                          int(cfg["model"]["hidden_dim"]),
                          int(cfg["model"]["unfreeze_last_n_blocks"])).to(device)
    counts = model.count_parameters()
    print(f"[train] params total={counts['total']:,} trainable={counts['trainable']:,} "
          f"head={counts['head']:,}")

    criterion = SupConLoss(float(cfg["loss"]["temperature"]))
    optimizer = torch.optim.AdamW(model.param_groups(
        float(cfg["optim"]["lr_head"]), float(cfg["optim"]["lr_backbone"]),
        float(cfg["optim"]["weight_decay"])))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(max_steps, float(cfg["optim"]["warmup_frac"])))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_path = Path(cfg["output"]["log_csv"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_fh)
    writer.writerow(["step", "elapsed_s", "loss", "lr_head", "lr_backbone",
                     "val_map_heldout", "val_rank1_heldout", "val_map_indist",
                     "val_rank1_indist", "generalisation_gap", "val_map_tonal",
                     "val_map_select"])

    best = -1.0
    best_path = Path(cfg["output"]["checkpoint"])
    t0 = time.time()
    step = 0
    last_eval_step = 0
    losses: list[float] = []
    model.train()

    for batch in train_loader:
        if step >= max_steps or (time.time() - t0) / 60.0 >= max_minutes:
            break
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            z = model(x)
        loss = criterion(z, y)                     # computed in fp32 inside the loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        # Unscale before clipping, otherwise the clip threshold is applied to scaled grads.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["optim"]["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        losses.append(float(loss.item()))

        if step % 10 == 0:
            recent = np.mean(losses[-10:])
            print(f"[train] step {step:5d}  loss {recent:.4f}  "
                  f"{time.time() - t0:6.0f}s", flush=True)

        if step % eval_every == 0 or step == max_steps:
            best = _validate_and_maybe_save(model, val_sets, device, cfg, optimizer, writer,
                                            log_fh, step, t0, losses, eval_every, best,
                                            best_path)
            last_eval_step = step

    # The time box can end the loop between evaluations. Without a final validation, the last
    # stretch of training would be discarded, and if the box hit before the FIRST evaluation
    # no checkpoint would exist at all, leaving evaluate.py with nothing to load.
    if step > 0 and last_eval_step != step:
        best = _validate_and_maybe_save(model, val_sets, device, cfg, optimizer, writer,
                                        log_fh, step, t0, losses, eval_every, best, best_path)

    log_fh.close()
    first = float(np.mean(losses[:5])) if len(losses) >= 5 else float("nan")
    last = float(np.mean(losses[-5:])) if len(losses) >= 5 else float("nan")
    summary = {"steps": step, "minutes": round((time.time() - t0) / 60, 2),
               "loss_first5": first, "loss_last5": last, "best_select_metric": best,
               "input_mode": "gray" if gray else "rgb",
               "device": device}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[train] done: {json.dumps(summary)}")

    if args.smoke:
        ok = last < first and best_path.exists()
        print(f"[smoke] loss {first:.4f} -> {last:.4f}, checkpoint saved: {best_path.exists()}: "
              f"{'PASSED' if ok else 'FAILED, investigate'}")
        return 0 if ok else 1
    return 0


def _validate_and_maybe_save(model, val_sets, device, cfg, optimizer, writer, log_fh, step,
                             t0, losses, eval_every, best, best_path) -> float:
    """Run validation, log it, and save the checkpoint if the selection metric improved.

    Returns the (possibly updated) best value of the selection metric.
    """
    metrics = run_validation(model, val_sets, device, cfg)
    gap = metrics.get("val_map_indist", float("nan")) - metrics.get(
        "val_map_heldout", float("nan"))
    lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}
    writer.writerow([step, round(time.time() - t0, 1),
                     round(float(np.mean(losses[-eval_every:])), 5),
                     lrs.get("head"), lrs.get("backbone"),
                     metrics.get("val_map_heldout"), metrics.get("val_rank1_heldout"),
                     metrics.get("val_map_indist"), metrics.get("val_rank1_indist"), gap,
                     metrics.get("val_map_tonal"), metrics.get("val_map_select")])
    log_fh.flush()
    sel = metrics.get(cfg["eval"]["select_metric"], -1.0)
    print(f"[val]   step {step}  heldout {metrics.get('val_map_heldout', 0):.4f}  "
          f"tonal {metrics.get('val_map_tonal', 0):.4f}  "
          f"select {metrics.get('val_map_select', 0):.4f}  "
          f"indist {metrics.get('val_map_indist', 0):.4f}  gap {gap:+.4f}", flush=True)
    if sel > best:
        best = sel
        torch.save({"model": model.state_dict(), "step": step, "metric": sel,
                    "config": cfg}, best_path)
        print(f"[val]   new best {cfg['eval']['select_metric']}={sel:.4f} "
              f"saved to {best_path}", flush=True)
    return best


if __name__ == "__main__":
    raise SystemExit(main())
