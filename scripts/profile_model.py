"""
Efficiency report: parameters, FLOPs, latency, embedding size.

Named profile_model.py, NOT profile.py: running a script puts its folder first on sys.path,
so a file called profile.py shadows the standard library "profile" module. torch imports
cProfile, which imports profile, and crashes. That happened; the name is deliberate.

Latency only means something on the target hardware, so this script records the device it
ran on and stamps it into every output. Run it on the Kaggle T4 for the reported numbers; a
laptop CPU run is useful only to check that the script works.

Measured:
  params      total, trainable (fine-tuned), head only
  FLOPs       forward pass per image at 224x224, counted with torch's FlopCounterMode
              (1 multiply-add = 2 FLOPs, the usual convention)
  latency     batch 1 and batch 32, fp32 and fp16 autocast, median and p90 over timed
              iterations after warm-up, with CUDA synchronisation so GPU time is real
  memory      peak GPU memory at batch 32
  embedding   bytes per image at fp32 and fp16, and per million gallery entries, against the
              384-d zero-shot DINOv2 CLS vector for comparison

Input mode does not change cost: gray input replicates one channel to three, so the network
sees the same tensor shape either way.

Run:  python scripts/profile_model.py [--checkpoint outputs/run_rgb/best.pt] [--quick]
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.embedder import SareeEmbedder  # noqa: E402


def count_flops(model: torch.nn.Module, img_size: int, device: str) -> int:
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.inference_mode(), FlopCounterMode(display=False) as fc:
        model(x)
    return int(fc.get_total_flops())


def time_forward(model: torch.nn.Module, batch: int, img_size: int, device: str,
                 fp16: bool, warmup: int, iters: int) -> dict:
    x = torch.randn(batch, 3, img_size, img_size, device=device)
    use_amp = fp16 and device == "cuda"
    times = []
    with torch.inference_mode():
        for i in range(warmup + iters):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                model(x)
            if device == "cuda":
                # Without this, perf_counter measures only kernel LAUNCH time, not GPU work.
                torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    med = statistics.median(times)
    return {
        "batch": batch,
        "precision": "fp16 autocast" if use_amp else "fp32",
        "median_ms": round(med, 3),
        "p90_ms": round(times[int(0.9 * (len(times) - 1))], 3),
        "ms_per_image": round(med / batch, 3),
        "images_per_s": round(1000.0 * batch / med, 1),
        "iters": iters,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="optional; architecture is identical with or without trained weights")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--quick", action="store_true", help="few iterations, for a CPU sanity run")
    ap.add_argument("--out", default="outputs/profile")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or "CPU"

    model = SareeEmbedder("dinov2_vits14", 128, 512, unfreeze_last_n_blocks=4).to(device)
    if args.checkpoint and Path(args.checkpoint).exists():
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    model.eval()

    counts = model.count_parameters()
    flops_full = count_flops(model, args.img_size, device)
    flops_backbone = count_flops(model.backbone, args.img_size, device)

    # Measure the counter's convention instead of asserting it: a bias-free Linear(384, 512) on
    # one row is exactly 384*512 multiply-adds. torch's FlopCounterMode reports 2 per
    # multiply-add; fvcore reports 1 (it counts MACs and calls them "flops").
    probe = torch.nn.Linear(384, 512, bias=False).to(device)
    from torch.utils.flop_counter import FlopCounterMode
    with torch.inference_mode(), FlopCounterMode(display=False) as fc:
        probe(torch.randn(1, 384, device=device))
    flops_per_mac = fc.get_total_flops() / (384 * 512)

    # Tokens actually entering the transformer at this input size. Read from the model, not
    # from patch_embed.num_patches, which reports 1369 (the 518 px pretraining grid); at 224
    # the position embeddings are interpolated to a 16 x 16 grid.
    with torch.inference_mode():
        tokens = model.backbone.prepare_tokens_with_masks(
            torch.randn(1, 3, args.img_size, args.img_size, device=device))
    n_tokens = int(tokens.shape[1])
    n_reg = int(getattr(model.backbone, "num_register_tokens", 0))
    n_patches = n_tokens - 1 - n_reg

    warmup, iters_b1, iters_b32 = (3, 5, 3) if args.quick else (20, 100, 30)
    latency = []
    for fp16 in ([False, True] if device == "cuda" else [False]):
        latency.append(time_forward(model, 1, args.img_size, device, fp16, warmup, iters_b1))
        latency.append(time_forward(model, 32, args.img_size, device, fp16, warmup, iters_b32))

    peak_mem_mb = None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        time_forward(model, 32, args.img_size, device, True, 2, 3)
        peak_mem_mb = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)

    dim, zs_dim = model.embed_dim, model.feat_dim
    embedding = {
        "dim": dim,
        "bytes_fp32": dim * 4,
        "bytes_fp16": dim * 2,
        "gallery_1M_MB_fp32": round(dim * 4 * 1e6 / 2 ** 20, 1),
        "gallery_1M_MB_fp16": round(dim * 2 * 1e6 / 2 ** 20, 1),
        "zero_shot_dinov2_dim": zs_dim,
        "zero_shot_bytes_fp32": zs_dim * 4,
        "size_reduction_vs_zero_shot": f"{zs_dim / dim:.1f}x",
    }

    report = {
        "device": device,
        "device_name": device_name,
        "torch": torch.__version__,
        "is_target_hardware": device == "cuda" and "T4" in device_name,
        "input": f"3x{args.img_size}x{args.img_size} (gray input is replicated to 3 channels: same cost)",
        "params": counts,
        "flop_convention": f"{flops_per_mac:.0f} FLOPs per multiply-add (torch FlopCounterMode); "
                           f"GMACs = GFLOPs / {flops_per_mac:.0f}. fvcore would report GMACs.",
        "gmacs_per_image": round(flops_full / flops_per_mac / 1e9, 3),
        "tokens": {"total": n_tokens, "patches": n_patches, "cls": 1, "registers": n_reg,
                   "grid": f"{int(n_patches ** 0.5)}x{int(n_patches ** 0.5)}",
                   "patch_size": int(model.backbone.patch_size)},
        "gflops_per_image": round(flops_full / 1e9, 3),
        "gflops_backbone": round(flops_backbone / 1e9, 3),
        "gflops_head": round((flops_full - flops_backbone) / 1e9, 5),
        "latency": latency,
        "peak_gpu_memory_MB_batch32_fp16": peak_mem_mb,
        "embedding": embedding,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    p = counts
    md = [
        "# Efficiency report", "",
        f"Measured on **{device_name}** ({device}), torch {torch.__version__}."
        + ("" if report["is_target_hardware"]
           else "  **Not the target hardware: latency here is NOT the reported number.**"),
        "",
        "| quantity | value |", "|---|---|",
        f"| parameters, total | {p['total']:,} |",
        f"| parameters, fine-tuned (last 4 blocks, final norm, head) | {p['trainable']:,} |",
        f"| parameters, projection head | {p['head']:,} |",
        f"| GFLOPs per image ({args.img_size}x{args.img_size}) | {report['gflops_per_image']} "
        f"(backbone {report['gflops_backbone']}, head {report['gflops_head']}) |",
        f"| GMACs per image | {report['gmacs_per_image']} |",
        f"| FLOP convention | {report['flop_convention']} |",
        f"| tokens per image | {n_tokens} = {n_patches} patches ({report['tokens']['grid']} of "
        f"{report['tokens']['patch_size']}px) + 1 CLS + {n_reg} registers |",
        f"| embedding | {dim}-d, {embedding['bytes_fp32']} B fp32, {embedding['bytes_fp16']} B fp16 |",
        f"| gallery of 1M designs | {embedding['gallery_1M_MB_fp32']} MB fp32, "
        f"{embedding['gallery_1M_MB_fp16']} MB fp16 |",
        f"| vs zero-shot DINOv2 CLS | {zs_dim}-d, {embedding['zero_shot_bytes_fp32']} B fp32 "
        f"({embedding['size_reduction_vs_zero_shot']} larger) |",
    ]
    if peak_mem_mb is not None:
        md.append(f"| peak GPU memory, batch 32, fp16 | {peak_mem_mb} MB |")
    md += ["", "| batch | precision | median ms | p90 ms | ms / image | images / s |",
           "|---|---|---|---|---|---|"]
    for r in latency:
        md.append(f"| {r['batch']} | {r['precision']} | {r['median_ms']} | {r['p90_ms']} | "
                  f"{r['ms_per_image']} | {r['images_per_s']} |")
    md.append("")
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"[profile] wrote {out.with_suffix('.md')} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
