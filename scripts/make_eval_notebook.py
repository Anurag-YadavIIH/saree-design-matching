"""
Generate notebooks/kaggle_eval.ipynb: EVALUATION ONLY, on Kaggle, for existing checkpoints.

Why evaluation must run on Kaggle (D-45). The manifest is deterministic within one environment
but not across them: several Drive geometric matches sit within 0.01 of the NCC acceptance
gate, OpenCV builds differ, and one flipped match renumbers designs and reshuffles the split.
The checkpoints were trained on the manifest Kaggle builds, so only a Kaggle rebuild reproduces
their exact train and test sets. This notebook rebuilds it, ASSERTS it matches the training
run's numbers, computes the success bars from the baselines, and only then scores checkpoints.

Run:  python scripts/make_eval_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path


def md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}


def code(t: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip("\n").splitlines(True)}


CELLS = [
    md("""
# Evaluate trained checkpoints (Kaggle, GPU T4)

Evaluation only, no training. Rebuilds the manifest in the SAME environment the checkpoints
were trained in, verifies it is the training manifest, computes the pre-registered success
bars from the baselines, then scores the checkpoints against them (D-34, D-45).
"""),
    code("""
# Training-run fingerprint, copied from the training notebook's build output. The rebuild below
# must reproduce these exactly, or the test set is not the one the checkpoints were trained for.
EXPECT = {"drive_designs": 96, "drive_multi": 27, "kaggle_designs": 411, "kaggle_dups": 412,
          "constraining_pairs": 2, "synthetic": 2030, "train_designs": 311}
REPO_URL = "https://github.com/Anurag-YadavIIH/saree-design-matching"
WORK = "/kaggle/working/saree-reid"
SAFE = "/kaggle/working/ckpts"   # outside WORK, so re-cloning cannot delete it
"""),
    md("## 1. Secure the checkpoints BEFORE anything is deleted"),
    code("""
import os, shutil, subprocess, sys, re
from pathlib import Path

def find_ckpt(run):
    # The training notebook's output, attached via Add Input > Your Work, lands somewhere under
    # /kaggle/input; the exact nesting depends on the notebook slug, so search everywhere.
    hits = sorted(p for root in ("/kaggle/input", "/kaggle/working")
                  for p in Path(root).rglob(f"{run}/best.pt") if SAFE not in str(p))
    if len(hits) > 1:
        print(f"note: {len(hits)} candidates for {run}; using the first")
    return hits[0] if hits else None

for run in ("run_rgb", "run_gray"):
    dst = Path(SAFE) / run / "best.pt"
    if dst.exists():
        print("already safe:", dst); continue
    src = find_ckpt(run)
    assert src is not None, (f"no {run}/best.pt under /kaggle/input: attach the training "
                             "notebook via Add Input > Your Work")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"copied {src} -> {dst}")
"""),
    md("""
## 2. Code from GitHub, data COPIED into data/

The data is copied, not symlinked, exactly as in the training run. The adapters compute each
path with `Path.resolve()`, which follows a symlink out of `data/` into `/kaggle/input`, so a
symlinked folder breaks the relative-path step. Copies make the build identical to training.
"""),
    code("""
def find_data(name, min_images=100):
    # A folder called `name` under /kaggle/input that really holds images, so an empty or
    # partial folder (for example inside a notebook output) can never be picked by mistake.
    for p in sorted(Path("/kaggle/input").rglob(name)):
        if p.is_dir() and sum(1 for _ in p.rglob("*.jpg")) >= min_images:
            return p
    raise FileNotFoundError(f"no '{name}' folder with >= {min_images} images under /kaggle/input")

if Path(WORK).exists():
    shutil.rmtree(WORK)
subprocess.run(["git", "clone", "--depth", "1", REPO_URL, WORK], check=True)
os.chdir(WORK)
print("code commit:", subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                     capture_output=True, text=True).stdout.strip())
assert Path("scripts/judge_d34.py").exists(), "cloned code is too old: push the eval changes first"

data = Path(WORK) / "data"
data.mkdir(exist_ok=True)
for name in ("Google_drive_data", "Kaggle_data"):
    dst = data / name
    if dst.is_symlink():
        os.unlink(dst)                       # never rmtree a link: that would walk the target
    if not dst.exists():
        shutil.copytree(find_data(name), dst)
    # Counts only: printing folder or file names here would put Drive paths in the notebook.
    print(f"{name}: {sum(1 for _ in dst.rglob('*.jpg'))} images copied")

!pip install -q imagehash
import torch
assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0)
print("GPU:", torch.cuda.get_device_name(0))
"""),
    md("""
## 3. Rebuild the manifest and PROVE it is the training manifest

The full log names Drive files, so it goes to a file that cleanup deletes; only the
fingerprint numbers are printed.
"""),
    code("""
with open("outputs_build.log", "w") as fh:
    rc = subprocess.run([sys.executable, "-m", "src.data.build_manifest", "--skip-validate"],
                        stdout=fh, stderr=subprocess.STDOUT).returncode
log = open("outputs_build.log", encoding="utf-8", errors="replace").read()
assert rc == 0, "build failed: inspect outputs_build.log privately"

def grab(pattern):
    m = re.search(pattern, log)
    return int(m.group(1).replace(",", "")) if m else None

got = {
    "drive_designs": grab(r"deeplure_drive: ([\\d,]+) designs"),
    "drive_multi": grab(r"deeplure_drive: [\\d,]+ designs from [\\d,]+ images \\((\\d+) multi"),
    "kaggle_designs": grab(r"kaggle_fabric: ([\\d,]+) designs"),
    "kaggle_dups": grab(r"kaggle_fabric: [\\d,]+ designs .*?, ([\\d,]+) dup groups"),
    "constraining_pairs": grab(r"extra grouping: (\\d+) constraining"),
    "synthetic": grab(r"materialised ([\\d,]+) synthetic"),
    "train_designs": grab(r"label_map with ([\\d,]+) train"),
}
for k in EXPECT:
    print(f"{k:20s} expected {EXPECT[k]:5d}  got {got[k]}  {'ok' if got[k] == EXPECT[k] else 'MISMATCH'}")
assert got == EXPECT, "rebuilt manifest differs from the training run: do NOT evaluate"
!python -m src.data.validate --no-audit
"""),
    md("""
## 4. Baselines, then the success bars, BEFORE any checkpoint is scored

Exclusions are resolved from image_id links against this manifest, with audit coverage reported.
"""),
    code("""
!python scripts/evaluate.py --only phash colorhist dino_rgb dino_gray --out outputs/results_baselines
!python scripts/judge_d34.py bars --baselines outputs/results_baselines.json --out outputs/bars.json
"""),
    md("## 5. Score the checkpoints and apply the D-34 rule mechanically"),
    code("""
!python scripts/evaluate.py --skip-baselines --checkpoints {SAFE}/run_rgb/best.pt {SAFE}/run_gray/best.pt --out outputs/results_checkpoints
!python scripts/judge_d34.py verdict --checkpoints outputs/results_checkpoints.json --bars outputs/bars.json --out outputs/verdict.json
"""),
    md("## 6. Real colourway verification (Kaggle pairs, keyed by image_id)"),
    code("""
!python scripts/eval_colourway.py --checkpoints {SAFE}/run_rgb/best.pt {SAFE}/run_gray/best.pt --out outputs/results_colourway
"""),
    code("""
from IPython.display import Markdown, display
for f in ("results_baselines", "results_checkpoints", "results_colourway"):
    display(Markdown(open(f"outputs/{f}.md").read()))
print(open("outputs/verdict.json").read())
"""),
    md("""
## 7. Cleanup: delete every Drive derivative, keep results and checkpoints

Download `outputs/*.md`, `outputs/*.json` and the checkpoints, and keep this notebook private.
"""),
    code("""
work, out = Path(WORK), Path(WORK) / "outputs"
KEEP = {f"{n}.{e}" for n in ("results_baselines", "results_checkpoints", "results_colourway")
        for e in ("md", "json")} | {"bars.json", "verdict.json"}
data = work / "data"
for p in data.iterdir():
    if p.is_symlink():
        p.unlink()            # removes the link only; /kaggle/input is untouched
    elif p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()
data.rmdir()
for p in sorted(out.rglob("*"), key=lambda q: len(q.parts), reverse=True):
    if p.is_file() and p.relative_to(out).as_posix() not in KEEP:
        p.unlink()
    elif p.is_dir() and not any(p.iterdir()):
        p.rmdir()
(work / "outputs_build.log").unlink(missing_ok=True)
left = [p for p in Path("/kaggle/working").rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".npz"}]
assert not left, f"derived images remain: {left[:5]}"
pat = re.compile(r"Google_drive_data|handloom_sarees|\\bh?_?img_\\d+")
for p in out.rglob("*"):
    assert not pat.search(p.read_text(errors="replace")), f"Drive filename in {p}"
print("cleanup verified; kept:", sorted(p.name for p in out.iterdir()), "+ checkpoints in", SAFE)
"""),
]


def main() -> int:
    nb = {"cells": CELLS,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    out = Path("notebooks/kaggle_eval.ipynb")
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
