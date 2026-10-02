"""
Generate notebooks/kaggle_train.ipynb.

The notebook is a THIN wrapper: every cell calls a script in this repo, so there is one source
of truth and the notebook cannot drift from the code. Generated from Python rather than
hand-edited JSON so that it is always valid and its contents are reviewable in a diff.

Run:  python scripts/make_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


CELLS = [
    md("""
# Colour-invariant saree design recognition: train and evaluate

Thin wrapper over the repo. Every cell calls a script, so the notebook cannot drift from the
code. Run top to bottom on a **GPU T4 x1** accelerator.

Pipeline: build manifest, build k-means cache, smoke test, train (time boxed), evaluate.
"""),
    md("""
## 1. Configuration

The only cell you should need to edit.

* Data folders `Google_drive_data/` and `Kaggle_data/` are found automatically in whatever
  datasets you attached, at any nesting depth. They may live in separate datasets.
* The code is found automatically as the attached folder containing `src/`, `configs/` and
  `scripts/`, or set `REPO_URL` to clone it instead.
* `INCLUDE_DRIVE = False` trains and evaluates on the public Kaggle source only, which avoids
  placing the proprietary corpus on Kaggle at all. See the README for that trade-off.
"""),
    code("""
# Leave as "auto" to search every attached dataset for these folder names, whatever the
# dataset slugs or nesting. Set an explicit path only if auto-discovery picks the wrong one.
DRIVE_DIR = "auto"                               # folder named Google_drive_data
KAGGLE_DIR = "auto"                              # folder named Kaggle_data
CODE_INPUT = "auto"                              # folder containing src/ and configs/
REPO_URL = ""                                    # alternatively, a git URL to clone
INCLUDE_DRIVE = True                             # proprietary corpus, private dataset

WORK = "/kaggle/working/saree-reid"
TRAIN_MINUTES = 20                               # time box PER run (two runs)
"""),
    md("## 2. Code and environment"),
    code("""
import os, shutil, subprocess, sys
from pathlib import Path

def find_dir(name, must_contain=None):
    # First directory under /kaggle/input called `name`, or containing all of `must_contain`.
    for p in sorted(Path("/kaggle/input").rglob("*")):
        if not p.is_dir():
            continue
        if must_contain is not None:
            if all((p / m).exists() for m in must_contain):
                return p
        elif p.name == name:
            return p
    raise FileNotFoundError(f"no '{name or must_contain}' under /kaggle/input; attach the dataset")

if CODE_INPUT == "auto" and not REPO_URL:
    CODE_INPUT = str(find_dir(None, must_contain=["src", "configs", "scripts"]))
    print("code found at", CODE_INPUT)

if Path(WORK).exists():
    shutil.rmtree(WORK)
if REPO_URL:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, WORK], check=True)
else:
    # Copy, not symlink: the build writes into the repo tree, and /kaggle/input is read-only.
    shutil.copytree(CODE_INPUT, WORK, ignore=shutil.ignore_patterns("data", "outputs", ".venv"))
os.chdir(WORK)
print("working directory:", os.getcwd())
"""),
    code("""
# imagehash is not preinstalled on Kaggle; torch, torchvision, opencv and sklearn are.
!pip install -q imagehash
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU ONLY")
assert torch.cuda.is_available(), "Enable a GPU: Settings > Accelerator > GPU T4 x1"
"""),
    md("""
## 3. Wire the data in

`/kaggle/input` is read-only, but the build writes synthetic queries and a k-means cache
under `data/`. So `data/` is a real, writable folder here, and only the two raw source
folders are symlinked into it. Every relative path in the code then works unchanged.
"""),
    code("""
data_dir = Path(WORK) / "data"
data_dir.mkdir(exist_ok=True)
sources = {
    "Google_drive_data": DRIVE_DIR if DRIVE_DIR != "auto" else None,
    "Kaggle_data": KAGGLE_DIR if KAGGLE_DIR != "auto" else None,
}
for name, explicit in sources.items():
    if name == "Google_drive_data" and not INCLUDE_DRIVE:
        print(f"skipping {name} (INCLUDE_DRIVE=False)")
        continue
    src = Path(explicit) if explicit else find_dir(name)
    dst = data_dir / name
    assert src.exists(), f"missing {src}: check DRIVE_DIR / KAGGLE_DIR"
    if not dst.exists():
        dst.symlink_to(src, target_is_directory=True)
    print(f"linked {dst} -> {src}")

if not INCLUDE_DRIVE:
    # Disable the Drive source in the data config so the adapters never look for it.
    import yaml
    cfg_path = Path("configs/data.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["sources"]["deeplure_drive"]["enabled"] = False
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("Drive source disabled in configs/data.yaml")
"""),
    md("""
## 4. Build the manifest

Adapters, geometric identity grouping, split, synthetic val and test queries, then validation
and the leakage audit. Deterministic under the seed, so it reproduces the local manifest.
"""),
    code("""
# The full log names Drive files (leakage audit pairs, split-constraint stems). Printed cell
# output is saved INSIDE the notebook, where no file cleanup can reach it, so the full log goes
# to a file that the cleanup cell deletes, and only path-free summary lines are shown here.
import subprocess
with open("outputs_build.log", "w") as fh:
    rc = subprocess.run([sys.executable, "-m", "src.data.build_manifest"],
                        stdout=fh, stderr=subprocess.STDOUT).returncode
SAFE = ("designs from", "duplicate groups", "invariants", "FAILURE", "materialised",
        "wrote outputs", "label_map", "nearest-train cosine", "constraining pair")
for line in open("outputs_build.log", encoding="utf-8", errors="replace"):
    if any(k in line for k in SAFE) and "/" not in line and "_data" not in line:
        print(line.rstrip())
assert rc == 0, "manifest build failed: inspect outputs_build.log (do not paste it publicly)"
"""),
    code("""
print(open("outputs/data_summary.md").read())
"""),
    md("""
## 5. Precompute the k-means cache

k-means was the training bottleneck, so each train image's cluster map is computed once here.
A train-time recolour is then a table lookup.
"""),
    code("""
!python scripts/build_cache.py --workers 4
"""),
    md("""
## 6. Smoke test

About two minutes on 20 designs. It must report the loss decreasing before the real run.
"""),
    code("""
!python scripts/train.py --smoke --num-workers 2
"""),
    md("""
## 7. Train: two runs, RGB input and gray input

Identical config except input mode, each time boxed. Checkpoints are selected on the mean of
val mAP over held-out recolour and tonal queries; the in-distribution gap is logged but never
used for selection. Success criteria were fixed in DECISIONS.md (D-34) before these runs.
"""),
    code("""
!python scripts/train.py --input rgb --run-name run_rgb --max-minutes {TRAIN_MINUTES} --num-workers 2
"""),
    code("""
!python scripts/train.py --input gray --run-name run_gray --max-minutes {TRAIN_MINUTES} --num-workers 2
"""),
    code("""
import pandas as pd
for run in ["run_rgb", "run_gray"]:
    print("=====", run)
    display(pd.read_csv(f"outputs/{run}/train_log.csv").tail(6))
    print(open(f"outputs/{run}/train_summary.json").read())
"""),
    md("""
## 8. Evaluate

Baselines and the trained model, one table: identification with bootstrap CIs over designs,
verification at a val-chosen threshold, stress test B in two tiers, per-source and real-pairs
breakdowns.
"""),
    code("""
!python scripts/evaluate.py --checkpoints outputs/run_rgb/best.pt outputs/run_gray/best.pt
"""),
    code("""
from IPython.display import Markdown
Markdown(open("outputs/results.md").read())
"""),
    md("""
## 9. Efficiency, measured on THIS GPU

Parameters, FLOPs, latency at batch 1 and 32 in fp32 and fp16, peak memory, embedding bytes.
Latency is only meaningful on the target hardware, so this cell refuses to run off a T4.
Weights do not affect cost, so the RGB checkpoint stands in for both runs.
"""),
    code("""
import torch
assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0), (
    f"latency must come from a T4, got {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
!python scripts/profile_model.py --checkpoint outputs/run_rgb/best.pt --out outputs/profile
"""),
    md("""
## 10. Cleanup: delete everything derived from the Drive corpus

Synthetic query images and k-means maps are derivatives of proprietary images, and the
manifest, label map, leakage audit and build log all name Drive files. Kaggle notebook outputs
become public if the notebook is ever made public, so everything not on the allowlist below is
deleted, then the deletion is verified.

Kept: results, efficiency report, per-run train log, summary and best checkpoint.

Note that `best.pt` holds weights fine-tuned partly on Drive designs, so it is itself a
derivative. Download it and keep this notebook private.
"""),
    code("""
import shutil, re
from pathlib import Path

work = Path(WORK)
out = work / "outputs"
KEEP = {
    "results.md", "results.json", "profile.md", "profile.json",
    *[f"{run}/{name}" for run in ("run_rgb", "run_gray")
      for name in ("train_summary.json", "train_log.csv", "best.pt")],
}

# 1. data/: the two raw source folders are SYMLINKS into read-only /kaggle/input. unlink()
#    removes only the link and never touches the target; rmtree would try to walk into it.
data = work / "data"
if data.exists():
    for p in data.iterdir():
        if p.is_symlink():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)          # data/synthetic and data/cache: real derived files
        else:
            p.unlink()
    data.rmdir()

# 2. outputs/: delete everything not on the allowlist (manifest, label map, audit, summaries,
#    the smoke-test run, review flags), deepest paths first so emptied folders can go too.
for p in sorted(out.rglob("*"), key=lambda q: len(q.parts), reverse=True):
    rel = p.relative_to(out).as_posix()
    if p.is_file() and rel not in KEEP:
        p.unlink()
    elif p.is_dir() and not any(p.iterdir()):
        p.rmdir()
for stray in [work / "outputs_build.log"]:
    stray.unlink(missing_ok=True)

# 3. Verify, rather than assume.
leftover_images = [p for p in Path("/kaggle/working").rglob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".npz"}]
assert not leftover_images, f"derived images remain: {leftover_images[:5]}"
assert not (work / "data").exists(), "data/ still exists"
drive_name = re.compile(r"Google_drive_data|handloom_sarees|\\bh?_?img_\\d+")
for p in out.rglob("*"):
    if p.is_file() and p.suffix in {".md", ".json", ".csv"}:
        assert not drive_name.search(p.read_text(errors="replace")), f"Drive filename in {p}"
remaining = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
assert set(remaining) <= KEEP, f"unexpected files kept: {set(remaining) - KEEP}"
print("cleanup verified. Remaining outputs:")
for r in remaining:
    print("  ", r)
print("Download these, then keep the notebook PRIVATE (best.pt is a Drive derivative).")
"""),
]


def main() -> int:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path("notebooks/kaggle_train.ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
