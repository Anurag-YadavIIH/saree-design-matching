"""
Evaluate baselines and the trained model on the test split. One table, markdown and JSON.

Methods, all scored identically:
  grayscale pHash          colour-blind by construction; the floor a learned model must clear
  RGB colour histogram     EXPECTED TO FAIL. Its failure is the evidence that the benchmark
                           punishes colour shortcuts instead of rewarding them
  DINOv2 zero-shot, RGB    the pretrained-embedding baseline from the brief
  DINOv2 zero-shot, gray   isolates how much DINOv2 relies on colour versus structure
  trained model            ours

For each: identification (Rank-1, Rank-5, mAP with bootstrap CIs over designs), verification
(ROC-AUC, EER, TAR@FAR=1e-3 with the threshold chosen on VAL), stress test B in two tiers, a
per-source breakdown, and the real-pairs table broken down by query kind.

Run:  python scripts/evaluate.py [--checkpoint outputs/run/best.pt] [--skip-baselines]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import SareeDataset, read_rgb  # noqa: E402
from src.data.manifest import load_config, load_manifest  # noqa: E402
from src.eval.retrieval import bootstrap_ci_over_designs, evaluate_retrieval  # noqa: E402
from src.eval.stress import stress_b  # noqa: E402
from src.eval.verification import (  # noqa: E402
    build_pairs,
    evaluate_verification,
    threshold_at_far,
)


# --------------------------------------------------------------------------- embedders

def embed_phash(paths: list[Path]) -> np.ndarray:
    """64-bit grayscale pHash as a +/-1 vector.

    Cosine between two such vectors is 1 - 2 * hamming / 64, so ranking by cosine is exactly
    ranking by Hamming distance, and pHash slots into the same metric code as every other
    method.
    """
    import imagehash
    from PIL import Image
    out = []
    for p in paths:
        with Image.open(p) as im:
            bits = imagehash.phash(im.convert("L")).hash.flatten()
        out.append(np.where(bits, 1.0, -1.0))
    return np.asarray(out, dtype=np.float32)


def embed_color_hist(paths: list[Path], bins: int = 8) -> np.ndarray:
    """Joint RGB histogram, 8x8x8 = 512 dims, L2 normalised.

    Pure colour, no spatial information whatsoever. On a benchmark where every query is
    recoloured into a held-out palette, this SHOULD fail, and it should be fooled worst of all
    by stress test B, where different designs share one palette.
    """
    out = []
    for p in paths:
        rgb = read_rgb(p)
        h, _ = np.histogramdd(rgb.reshape(-1, 3), bins=(bins, bins, bins),
                              range=((0, 256), (0, 256), (0, 256)))
        v = h.flatten().astype(np.float32)
        out.append(v / max(np.linalg.norm(v), 1e-12))
    return np.asarray(out)


def embed_model(paths: list[Path], model: torch.nn.Module, device: str,
                img_size: int = 224, grayscale: bool = False, batch_size: int = 32) -> np.ndarray:
    """Shared embedding loop for both zero-shot DINOv2 and the trained model."""
    from src.data.dataset import to_tensor
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            x = torch.stack([to_tensor(read_rgb(p), img_size, grayscale)
                             for p in paths[i:i + batch_size]]).to(device)
            z = model(x)
            z = torch.nn.functional.normalize(z.float(), dim=1)
            out.append(z.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)


class DinoZeroShot(torch.nn.Module):
    """DINOv2 CLS features, no training. The baseline the brief asks us to beat."""

    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                       verbose=False)

    def forward(self, x):
        return self.backbone(x)


# --------------------------------------------------------------------------- evaluation

def tar_with_ci(q_emb, q_designs, g_emb, g_designs, thr, n_boot):
    """TAR at a fixed threshold, with a bootstrap CI over designs.

    Each query has exactly one genuine gallery entry (its design's representative), so TAR is
    the mean of a per-query indicator "genuine score >= threshold". That per-query form lets the
    same design-level bootstrap used for Rank-1 produce an interval for TAR, which the success
    criteria need ("beat gray zero-shot outside the CI").
    """
    q = q_emb / np.maximum(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-12)
    g = g_emb / np.maximum(np.linalg.norm(g_emb, axis=1, keepdims=True), 1e-12)
    sims = q @ g.T
    col = {d: i for i, d in enumerate(g_designs)}
    genuine = np.array([sims[i, col[d]] if d in col else -np.inf
                        for i, d in enumerate(q_designs)])
    hit = (genuine >= thr).astype(np.float64)
    ci = bootstrap_ci_over_designs({"tar": hit}, np.asarray(q_designs), n_bootstrap=n_boot)
    return float(hit.mean()), ci["tar"]


def resolve_exclusions(path: Path, data_config: str) -> tuple[list[str], str]:
    """Turn the audit's image_id LINKS into test design_ids for the manifest being evaluated.

    Design ids are renumbered whenever a borderline geometric match flips between machines, and
    the split can reshuffle with them (D-45). So the audit is stored as links between two
    images, keyed by image_id (a stable hash of the path), and resolved here:

      one side in TEST, the other in TRAIN  -> exclude the test-side design (a real leak)
      both in test, or partner in val       -> not a train leak; kept, and logged
      neither side in test                  -> nothing to do

    Also reports how many of the current test designs the human audit actually reviewed, since
    a different split can put never-audited designs into test.
    """
    if not path.exists():
        return [], ""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if cfg.get("exclude_test_designs"):
        raise SystemExit(f"{path} is the old design_id-keyed format, which breaks on renumbering; "
                         "regenerate it with scripts/make_exclusion_links.py")
    links = cfg.get("links") or []
    real = load_manifest(include_synthetic=False, config_path=data_config)
    by_id = real.set_index("image_id")
    missing = [l for l in links for k in ("test_image_id", "other_image_id")
               if l[k] not in by_id.index]
    if missing:
        raise SystemExit(f"{len(missing)} link image_id(s) not in this manifest; refusing to guess")

    excluded: list[str] = []
    print("[eval] exclusion links resolved against this manifest:")
    print(f"       {'audit id':14s} {'kind':26s} {'test-side design (split)':30s} "
          f"{'partner design (split)':30s} action")
    for l in links:
        a, b = by_id.loc[l["test_image_id"]], by_id.loc[l["other_image_id"]]
        sa, sb = a["split"], b["split"]
        if sa == "test" and sb == "train":
            action, target = "EXCLUDE test side", a["design_id"]
        elif sb == "test" and sa == "train":
            action, target = "EXCLUDE partner (now in test)", b["design_id"]
        elif a["design_id"] == b["design_id"]:
            action, target = "same design now; no leak", None
        elif "test" in (sa, sb):
            action, target = f"kept: partner in {sb if sa == 'test' else sa}, not train", None
        else:
            action, target = "no test side", None
        if target and target not in excluded:
            excluded.append(target)
        print(f"       {l.get('audit_numbering', ''):14s} {l['kind']:26s} "
              f"{a['design_id'] + ' (' + sa + ')':30s} {b['design_id'] + ' (' + sb + ')':30s} "
              f"{action}")

    audited = set(cfg.get("audited_test_image_ids") or [])
    test_designs = set(real.loc[real["split"] == "test", "design_id"])
    covered = set(real.loc[real["image_id"].isin(audited), "design_id"]) & test_designs
    coverage = (f"{len(covered)} of {len(test_designs)} current test designs were in the audited "
                f"test set; {len(test_designs - covered)} were never reviewed by a human")
    return excluded, coverage


def evaluate_method(name: str, embed_fn, test_g: pd.DataFrame, test_q: pd.DataFrame,
                    val_g: pd.DataFrame, val_q: pd.DataFrame, data_root: Path,
                    far: float, n_boot: int) -> dict:
    t0 = time.time()
    paths = lambda df: [data_root / p for p in df["path"]]  # noqa: E731

    tg = embed_fn(paths(test_g))
    tq = embed_fn(paths(test_q))
    vg = embed_fn(paths(val_g))
    vq = embed_fn(paths(val_q))

    res: dict = {"method": name}

    # ---- verification threshold is chosen on VAL, then frozen for test.
    val_pairs = build_pairs(vq, val_q["design_id"].tolist(), vg, val_g["design_id"].tolist())
    thr = threshold_at_far(val_pairs.scores, val_pairs.genuine.astype(bool), far)
    res["val_threshold"] = thr

    # ---- identification, overall, on the standard protocol queries
    std = test_q["query_kind"] == "synthetic_recolor"
    r = evaluate_retrieval(tq[std.values], test_q.loc[std, "design_id"].tolist(),
                           tg, test_g["design_id"].tolist(), n_bootstrap=n_boot)
    res["identification"] = r.as_row()
    res["identification_raw"] = {"rank1": r.rank1, "rank5": r.rank5, "map": r.map, "ci": r.ci}

    # ---- verification on test, at the val-chosen threshold
    test_pairs = build_pairs(tq[std.values], test_q.loc[std, "design_id"].tolist(),
                             tg, test_g["design_id"].tolist())
    v = evaluate_verification(test_pairs, far_target=far, threshold=thr)
    res["verification"] = v.as_row()
    res["verification"]["achieved_FAR_on_test"] = f"{v.achieved_far:.4f}"
    tar, tar_ci = tar_with_ci(tq[std.values], test_q.loc[std, "design_id"].to_numpy(),
                              tg, test_g["design_id"].to_numpy(), thr, n_boot)
    res["tar_ci"] = {"tar": tar, "ci": tar_ci}

    # ---- tonal queries: lightness changes too, so grayscale input is NOT invariant here
    ton = test_q["query_kind"] == "synthetic_tonal"
    if ton.any():
        rt = evaluate_retrieval(tq[ton.values], test_q.loc[ton, "design_id"].tolist(),
                                tg, test_g["design_id"].tolist(), n_bootstrap=n_boot)
        res["tonal"] = rt.as_row()
        res["tonal_raw"] = {"rank1": rt.rank1, "map": rt.map, "ci": rt.ci}
        t_tar, t_ci = tar_with_ci(tq[ton.values], test_q.loc[ton, "design_id"].to_numpy(),
                                  tg, test_g["design_id"].to_numpy(), thr, n_boot)
        res["tonal_tar_ci"] = {"tar": t_tar, "ci": t_ci}
        q_ton = test_q[ton].reset_index(drop=True)
        # Tier 3: different designs rendered in the SAME tonal palette.
        res["stress_b_tier3"] = stress_b(tq[ton.values], q_ton, tier=1, threshold=thr)

    # ---- stress B, both tiers, at the same frozen threshold
    q_std = test_q[std].reset_index(drop=True)
    res["stress_b_tier1"] = stress_b(tq[std.values], q_std, tier=1, threshold=thr)
    res["stress_b_tier2"] = stress_b(tq[std.values], q_std, tier=2, threshold=thr)

    # ---- per-source breakdown (the free source-held-out signal)
    per_source = {}
    for src in sorted(test_q["source"].unique()):
        m = (test_q["source"] == src) & std
        gm = test_g["source"] == src
        if m.sum() == 0 or gm.sum() == 0:
            continue
        # Gallery stays FULL: restricting it would make retrieval artificially easy.
        rs = evaluate_retrieval(tq[m.values], test_q.loc[m, "design_id"].tolist(),
                                tg, test_g["design_id"].tolist(), n_bootstrap=n_boot)
        per_source[src] = rs.as_row()
    res["per_source"] = per_source

    # ---- real pairs: the 28 multi-view Drive designs, by query kind
    real_designs = set(test_q.loc[test_q["query_kind"] == "real_view", "design_id"])
    real_pairs = {}
    for kind in ("real_view", "real_view_recolored", "synthetic_recolor",
                 "synthetic_tonal"):
        m = (test_q["query_kind"] == kind) & test_q["design_id"].isin(real_designs)
        if m.sum() == 0:
            continue
        rk = evaluate_retrieval(tq[m.values], test_q.loc[m, "design_id"].tolist(),
                                tg, test_g["design_id"].tolist(), n_bootstrap=n_boot)
        real_pairs[kind] = rk.as_row()
    res["real_pairs"] = real_pairs
    res["seconds"] = round(time.time() - t0, 1)
    print(f"[eval] {name:28s} Rank-1 {r.rank1:.3f}  mAP {r.map:.3f}  "
          f"AUC {v.roc_auc:.3f}  ({res['seconds']}s)", flush=True)
    return res


def to_markdown(results: list[dict], far: float) -> str:
    L: list[str] = ["# Results", "",
                    "Test split. Identification uses the standard protocol queries "
                    "(`synthetic_recolor`: held-out palette AND held-out recolour algorithm, "
                    "plus realistic geometry). 95% confidence intervals from 1000 bootstrap "
                    "resamples over **designs**, not queries, because sibling queries of one "
                    "design are correlated.", "",
                    "Verification thresholds are chosen on **val** at the target FAR and then "
                    "frozen for test.", ""]

    L += ["## Identification and verification", "",
          f"| method | Rank-1 | Rank-5 | mAP | ROC-AUC | EER | TAR@FAR={far:g} |",
          "|---|---|---|---|---|---|---|"]
    def tar_fmt(d):
        lo, hi = d["ci"]
        return f"{d['tar']:.3f} [{lo:.3f}, {hi:.3f}]"

    for r in results:
        i, v = r["identification"], r["verification"]
        L.append(f"| {r['method']} | {i['Rank-1']} | {i['Rank-5']} | {i['mAP']} | "
                 f"{v['ROC-AUC']} | {v['EER']} | {tar_fmt(r['tar_ci'])} |")
    L.append("")

    if any("tonal" in r for r in results):
        L += ["## Tonal queries: lightness changes, dark and light may invert", "",
              "The standard generators restore the original L channel, so grayscale input is "
              "invariant to them by construction. Tonal queries change lightness and invert the "
              "dark/light order of colour clusters half the time, so a colour-blind shortcut no "
              "longer works. Stress B tier 3: different designs in the same tonal palette.", "",
              f"| method | tonal Rank-1 | tonal mAP | tonal TAR@FAR={far:g} | "
              "stress B tier 3 AUC | tier 3 false accept |",
              "|---|---|---|---|---|---|"]
        for r in results:
            if "tonal" not in r:
                continue
            t, sb = r["tonal"], r.get("stress_b_tier3", {})
            fa = sb.get("same_palette_false_accept_rate")
            L.append(f"| {r['method']} | {t['Rank-1']} | {t['mAP']} | "
                     f"{tar_fmt(r['tonal_tar_ci'])} | {sb.get('ROC-AUC', 'n/a')} | "
                     f"{'n/a' if fa is None else fa} |")
        L.append("")

    L += ["## Stress test B: different design, same palette (must NOT match)", "",
          "Only genuine pairs and same-palette impostors are kept. A colour-reliant method "
          "collapses here. `false accept` is the share of same-palette impostors accepted at "
          "the val-chosen threshold: lower is better.", "",
          "| method | tier | ROC-AUC | EER | false accept | margin | impostor pairs |",
          "|---|---|---|---|---|---|---|"]
    for r in results:
        for key, label in (("stress_b_tier1", "1: same palette"),
                           ("stress_b_tier2", "2: + same craft family")):
            s = r[key]
            if "ROC-AUC" not in s:
                L.append(f"| {r['method']} | {label} | {s.get('note', 'n/a')} | | | | |")
                continue
            fa = s["same_palette_false_accept_rate"]
            L.append(f"| {r['method']} | {label} | {s['ROC-AUC']} | {s['EER']} | "
                     f"{'n/a' if fa is None else fa} | {s['genuine_minus_impostor_margin']} | "
                     f"{s['impostor']:,} |")
    L.append("")

    L += ["## Per-source breakdown", "",
          "Gallery is the FULL test gallery in every row; only queries are filtered.", "",
          "| method | source | Rank-1 | mAP | designs |", "|---|---|---|---|---|"]
    for r in results:
        for src, row in r["per_source"].items():
            L.append(f"| {r['method']} | {src} | {row['Rank-1']} | {row['mAP']} | "
                     f"{row['designs']} |")
    L.append("")

    L += ["## Real pairs (28 multi-view Drive designs)", "",
          "These are the only real positive pairs in the corpus. They are **overlapping crops "
          "of one photograph** (scale 1.0, about 25% overlap), so they test region and framing "
          "invariance at fixed lighting and scale. They are not re-photography, and they do "
          "not vary palette.", "",
          "- `real_view`: another real view, unmodified",
          "- `real_view_recolored`: another real view plus a held-out palette; the closest "
          "proxy this corpus permits to the true task",
          "- `synthetic_recolor`: the standard protocol on these same designs", "",
          "| method | query kind | Rank-1 | Rank-5 | mAP | queries |",
          "|---|---|---|---|---|---|"]
    for r in results:
        for kind, row in r["real_pairs"].items():
            L.append(f"| {r['method']} | {kind} | {row['Rank-1']} | {row['Rank-5']} | "
                     f"{row['mAP']} | {row['queries']} |")
    L.append("")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--checkpoint", default="outputs/run/best.pt")
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="several checkpoints, e.g. the RGB and gray runs")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of: phash colorhist dino_rgb dino_gray trained")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--exclude", default="configs/eval_exclude.yaml",
                    help="test designs dropped post hoc as suspected leaks")
    ap.add_argument("--out", default="outputs/results")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.data_config)
    data_root = Path(cfg["paths"]["data_root"])
    far = float(yaml.safe_load(Path("configs/data.yaml").read_text(encoding="utf-8"))
                .get("eval", {}).get("far_target", 1e-3))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    test_g = load_manifest(split="test", role="gallery", config_path=args.data_config)
    test_q = load_manifest(split="test", role="query", config_path=args.data_config)

    # Post-hoc leakage exclusion, resolved against THIS manifest (configs/eval_exclude.yaml).
    excluded, coverage = resolve_exclusions(Path(args.exclude), args.data_config)
    if excluded:
        n_q = int(test_q["design_id"].isin(excluded).sum())
        test_g = test_g[~test_g["design_id"].isin(excluded)].reset_index(drop=True)
        test_q = test_q[~test_q["design_id"].isin(excluded)].reset_index(drop=True)
        print(f"[eval] POST-HOC EXCLUSION: dropped {len(excluded)} test design(s), {n_q} queries")
    else:
        print(f"[eval] no post-hoc exclusions ({args.exclude})")
    if coverage:
        print(f"[eval] audit coverage: {coverage}")
    val_g = load_manifest(split="val", role="gallery", config_path=args.data_config)
    val_q = load_manifest(split="val", role="query", config_path=args.data_config)
    print(f"[eval] test: {len(test_g)} gallery, {len(test_q)} queries | "
          f"val: {len(val_g)} gallery, {len(val_q)} queries | device={device}")

    wanted = set(args.only) if args.only else {"phash", "colorhist", "dino_rgb",
                                               "dino_gray", "trained"}
    if args.skip_baselines:
        wanted = {"trained"}

    methods = []
    if "phash" in wanted:
        methods.append(("grayscale pHash", embed_phash))
    if "colorhist" in wanted:
        methods.append(("RGB colour histogram", embed_color_hist))
    if wanted & {"dino_rgb", "dino_gray"}:
        dino = DinoZeroShot().to(device)
        if "dino_rgb" in wanted:
            methods.append(("DINOv2 zero-shot RGB",
                            lambda ps: embed_model(ps, dino, device)))
        if "dino_gray" in wanted:
            methods.append(("DINOv2 zero-shot gray",
                            lambda ps: embed_model(ps, dino, device, grayscale=True)))
    if "trained" in wanted:
        from src.models.embedder import SareeEmbedder
        for ck in [Path(c) for c in (args.checkpoints or [args.checkpoint])]:
            if not ck.exists():
                print(f"[eval] no checkpoint at {ck}; skipping")
                continue
            state = torch.load(ck, map_location=device, weights_only=False)
            mc = state["config"]["model"]
            model = SareeEmbedder(mc["backbone"], int(mc["embed_dim"]), int(mc["hidden_dim"]),
                                  int(mc["unfreeze_last_n_blocks"])).to(device)
            model.load_state_dict(state["model"])
            # The checkpoint records its own input mode, so a gray-trained model is always
            # evaluated on gray input. Getting this wrong would silently mis-score it.
            gray = mc.get("input_mode", "rgb") == "gray"
            label = f"trained {'gray' if gray else 'RGB'} (step {state.get('step', '?')})"
            methods.append((label, lambda ps, m=model, g=gray: embed_model(
                ps, m, device, grayscale=g)))

    results = [evaluate_method(n, fn, test_g, test_q, val_g, val_q, data_root, far,
                               args.n_boot) for n, fn in methods]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    md = to_markdown(results, far)
    note = (f"**Post-hoc leakage exclusion:** {len(excluded)} test design(s) removed from "
            f"gallery and queries ({', '.join(excluded)}); see DECISIONS.md D-38."
            if excluded else "No post-hoc leakage exclusions applied.")
    head = "# Results"
    md = md.replace(head, head + chr(10) + chr(10) + note, 1)
    out.with_suffix(".md").write_text(md, encoding="utf-8")

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    out.with_suffix(".json").write_text(json.dumps(clean(results), indent=2), encoding="utf-8")
    print(f"[eval] wrote {out.with_suffix('.md')} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
