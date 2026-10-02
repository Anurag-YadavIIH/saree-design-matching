"""
Phase 1 / Step 1: read-only recon of whatever sits under data/.

Why this exists: we must not design a labeling rule from imagination. This script
reports the raw facts (tree, extensions, label files, image metadata) and then tests
several candidate "design identity" keys side by side, so the labeling rule is chosen
from evidence instead of a guess.

Privacy: this script reads image headers only. It never prints pixel data, never writes
an image, and never copies anything out of data/. The report goes to outputs/, which is
gitignored.

Run:  python scripts/recon.py [--data-root data] [--out outputs/recon_report.md]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import statistics
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install pillow")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".avif"}
LABEL_EXTS = {".csv", ".json", ".txt", ".tsv", ".xlsx", ".jsonl", ".yaml", ".yml"}

# Used only as evidence that a filename or folder encodes a colorway. Not a final rule.
COLOR_WORDS = [
    "red", "blue", "green", "yellow", "pink", "purple", "violet", "orange", "brown",
    "black", "white", "grey", "gray", "gold", "golden", "silver", "maroon", "navy",
    "teal", "turquoise", "cyan", "magenta", "beige", "cream", "ivory", "peach",
    "mustard", "olive", "lavender", "lilac", "coral", "rust", "wine", "rani",
    "firozi", "mehendi", "sandal", "copper", "bronze", "multicolor", "multi",
]

# Verifying every file end to end is slow on a large corpus, so a full decode runs on a
# seeded sample while cheap header reads cover every file.
VERIFY_SAMPLE_PER_SOURCE = 300
SEED = 42


# ----------------------------------------------------------------------------- helpers

def human(n: int) -> str:
    return f"{n:,}"


def rel(path: Path, root: Path) -> str:
    """Posix-style path relative to data/, which is what the manifest will store."""
    return path.relative_to(root).as_posix()


def walk_files(root: Path) -> list[Path]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip archive and notebook noise that is never part of the corpus.
        dirnames[:] = [d for d in dirnames
                       if d not in {"__MACOSX", ".ipynb_checkpoints"} and not d.startswith(".")]
        for fn in filenames:
            if fn.startswith("."):
                continue
            out.append(Path(dirpath) / fn)
    return sorted(out)


def tree_summary(src_root: Path, max_depth: int = 3) -> list[str]:
    """Directory tree to a fixed depth, with image counts per directory."""
    lines = []
    for dirpath, dirnames, filenames in os.walk(src_root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        d = Path(dirpath)
        depth = len(d.relative_to(src_root).parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        n_img = sum(1 for f in filenames if Path(f).suffix.lower() in IMAGE_EXTS)
        indent = "  " * depth
        name = src_root.name if depth == 0 else d.name
        lines.append(f"{indent}{name}/  [{human(n_img)} images, {human(len(dirnames))} subdirs]")
        if depth == max_depth and dirnames:
            lines.append(f"{indent}  ... {human(len(dirnames))} subdirs not expanded")
    return lines


def peek_label_file(path: Path) -> list[str]:
    """Report the shape of a candidate label file without dumping its contents."""
    lines = [f"- `{path.name}` ({human(path.stat().st_size)} bytes)"]
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                header = fh.readline().rstrip("\n")
                n_rows = sum(1 for _ in fh)
            lines.append(f"  - columns: {header.split(sep)}")
            lines.append(f"  - data rows: {human(n_rows)}")
        elif suffix in {".json", ".jsonl"}:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                if suffix == ".jsonl":
                    obj = json.loads(fh.readline())
                    keys = sorted(obj)[:20] if isinstance(obj, dict) else type(obj).__name__
                    lines.append(f"  - first record keys: {keys}")
                else:
                    obj = json.load(fh)
                    if isinstance(obj, dict):
                        lines.append(f"  - top-level keys ({len(obj)}): {sorted(obj)[:20]}")
                    elif isinstance(obj, list):
                        first = obj[0] if obj else None
                        keys = sorted(first)[:20] if isinstance(first, dict) else type(first).__name__
                        lines.append(f"  - list of {human(len(obj))}; first item keys: {keys}")
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                head = [next(fh, "").rstrip("\n") for _ in range(3)]
            lines.append(f"  - first lines: {[h for h in head if h]}")
    except Exception as exc:  # a malformed label file is itself a finding
        lines.append(f"  - could not parse: {type(exc).__name__}: {exc}")
    return lines


# ------------------------------------------------------- candidate design-identity keys
# Each keyer maps a relative image path to a candidate design_id. The report shows the
# resulting images-per-design distribution for all of them. A rule that produces many
# groups of size >= 2 is a plausible design key; a rule where almost every group has
# size 1 cannot support query/gallery pairs at all.

TRAILING_INDEX = re.compile(r"[\s._\-(]*\d+\)?$")
COLOR_RE = re.compile(
    r"(?<![a-z])(" + "|".join(sorted(COLOR_WORDS, key=len, reverse=True)) + r")(?![a-z])",
    re.IGNORECASE,
)


def key_parent_dir(relpath: str) -> str:
    """K1: the immediate parent folder. Correct when each design is its own folder."""
    parent = str(Path(relpath).parent)
    return parent if parent != "." else "<root>"


def key_top_dir(relpath: str) -> str:
    """K2: the first folder level. Correct when folders are broad categories, not designs."""
    parts = Path(relpath).parts
    return parts[0] if len(parts) > 1 else "<root>"


def key_stem_no_index(relpath: str) -> str:
    """K3: filename with a trailing counter stripped (saree_12_3.jpg -> saree_12)."""
    stem = Path(relpath).stem
    return TRAILING_INDEX.sub("", stem) or stem


def key_stem_no_color(relpath: str) -> str:
    """K4: filename with colour words and trailing counters stripped. Correct when one
    design is stored as <design>_<colour>.jpg, which is exactly the multi-colorway case
    this project needs."""
    stem = Path(relpath).stem
    stripped = COLOR_RE.sub("", stem)
    stripped = re.sub(r"[\s._\-]{2,}", "_", stripped).strip("._- ")
    return TRAILING_INDEX.sub("", stripped) or stem


KEYERS = {
    "K1 parent dir": key_parent_dir,
    "K2 top dir": key_top_dir,
    "K3 stem minus trailing index": key_stem_no_index,
    "K4 stem minus colour word and index": key_stem_no_color,
}


def group_report(name: str, groups: dict[str, list[str]], total: int) -> list[str]:
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    multi = [s for s in sizes if s >= 2]
    hist = collections.Counter(sizes)
    hist_str = ", ".join(f"{k} img x{human(v)}" for k, v in sorted(hist.items())[:12])
    lines = [
        f"**{name}**",
        f"  - groups: {human(len(groups))} for {human(total)} images "
        f"(mean {total / max(len(groups), 1):.2f} images/group)",
        f"  - groups with >= 2 images: {human(len(multi))} "
        f"({100 * len(multi) / max(len(groups), 1):.1f}% of groups, "
        f"covering {human(sum(multi))} images)",
        f"  - size histogram: {hist_str}",
    ]
    # A couple of example multi-image groups make the rule concrete without dumping data.
    examples = [(k, v) for k, v in groups.items() if len(v) >= 2][:3]
    for k, v in examples:
        lines.append(f"  - example group `{k}` -> {[Path(p).name for p in sorted(v)[:4]]}")
    return lines


# ------------------------------------------------------------------------------- recon

def recon_source(src_root: Path, data_root: Path) -> list[str]:
    rng = random.Random(SEED)
    out: list[str] = [f"\n## Source: `{src_root.name}`\n"]

    files = walk_files(src_root)
    if not files:
        out.append("_No files found._")
        return out

    images = [f for f in files if f.suffix.lower() in IMAGE_EXTS]
    label_files = [f for f in files if f.suffix.lower() in LABEL_EXTS]
    others = [f for f in files
              if f.suffix.lower() not in IMAGE_EXTS and f.suffix.lower() not in LABEL_EXTS]

    out.append(f"Total files: {human(len(files))} "
               f"({human(len(images))} images, {human(len(label_files))} label-ish, "
               f"{human(len(others))} other)\n")

    out.append("### Folder tree (depth 3)\n```")
    out.extend(tree_summary(src_root))
    out.append("```\n")

    out.append("### File counts by extension\n```")
    for ext, n in collections.Counter(f.suffix.lower() for f in files).most_common():
        out.append(f"{ext or '<none>':10s} {human(n)}")
    out.append("```\n")

    out.append("### Candidate label files\n")
    if label_files:
        for f in label_files[:20]:
            out.extend(peek_label_file(f))
    else:
        out.append("_None found. Labels must come from folder names or filenames._")
    out.append("")

    if others:
        out.append("### Non-image, non-label files (first 10)\n```")
        out.extend(rel(f, data_root) for f in others[:10])
        out.append("```\n")

    if not images:
        out.append("_No images in this source._")
        return out

    out.append("### 15 sample relative paths\n```")
    sample_paths = rng.sample(images, min(15, len(images)))
    out.extend(rel(p, data_root) for p in sorted(sample_paths))
    out.append("```\n")

    # ---- image metadata: cheap header read for every file
    widths: list[int] = []
    heights: list[int] = []
    ratios: list[float] = []
    modes: collections.Counter = collections.Counter()
    formats: collections.Counter = collections.Counter()
    corrupt: list[tuple[str, str]] = []
    for p in images:
        try:
            with Image.open(p) as im:
                w, h = im.size
                widths.append(w)
                heights.append(h)
                ratios.append(w / h if h else 0.0)
                modes[im.mode] += 1
                formats[im.format or "?"] += 1
        except Exception as exc:
            corrupt.append((rel(p, data_root), type(exc).__name__))

    out.append("### Image size stats\n```")
    if widths:
        for label, vals in (("width", widths), ("height", heights)):
            out.append(f"{label:7s} min={min(vals)}  median={int(statistics.median(vals))}  "
                       f"mean={int(statistics.mean(vals))}  max={max(vals)}")
        out.append(f"aspect  min={min(ratios):.2f}  median={statistics.median(ratios):.2f}  "
                   f"max={max(ratios):.2f}")
        small = sum(1 for w, h in zip(widths, heights) if min(w, h) < 224)
        out.append(f"images with min side < 224 (our train size): {human(small)}")
    out.append(f"modes:   {dict(modes)}")
    out.append(f"formats: {dict(formats)}")
    out.append("```\n")

    out.append("### Integrity\n")
    out.append(f"- unreadable headers: {human(len(corrupt))}")
    for path, err in corrupt[:10]:
        out.append(f"  - `{path}`: {err}")
    # A full decode on a seeded sample catches truncated files that header reads do not.
    verify_set = rng.sample(images, min(VERIFY_SAMPLE_PER_SOURCE, len(images)))
    truncated: list[tuple[str, str]] = []
    for p in verify_set:
        try:
            with Image.open(p) as im:
                im.load()
        except Exception as exc:
            truncated.append((rel(p, data_root), type(exc).__name__))
    out.append(f"- full decode check on {human(len(verify_set))} sampled images: "
               f"{human(len(truncated))} failed")
    for path, err in truncated[:10]:
        out.append(f"  - `{path}`: {err}")
    out.append("")

    # ---- structural evidence about how identity could be encoded
    relpaths = [rel(p, data_root) for p in images]
    # Paths relative to the source root, so folder depth is comparable across sources.
    inner = [Path(r).relative_to(src_root.name).as_posix() for r in relpaths]
    depths = collections.Counter(len(Path(r).parts) - 1 for r in inner)
    out.append("### Path structure\n```")
    out.append(f"directory depth below {src_root.name}/ : {dict(sorted(depths.items()))}")
    out.append(f"distinct parent directories: {human(len({str(Path(r).parent) for r in inner}))}")
    n_color_named = sum(1 for r in inner if COLOR_RE.search(Path(r).stem))
    out.append(f"filenames containing a colour word: {human(n_color_named)} "
               f"({100 * n_color_named / len(inner):.1f}%)")
    dir_color = sum(1 for r in inner if COLOR_RE.search(str(Path(r).parent)))
    out.append(f"paths whose directory contains a colour word: {human(dir_color)} "
               f"({100 * dir_color / len(inner):.1f}%)")
    seps: collections.Counter = collections.Counter()
    for r in inner:
        for ch in "_-. ":
            if ch in Path(r).stem:
                seps[ch] += 1
    out.append(f"filename separators present: {dict(seps)}")
    out.append("```\n")

    out.append("### Candidate design-identity keys\n")
    out.append("Each block applies one labeling rule and reports the resulting grouping.\n")
    for name, fn in KEYERS.items():
        groups: dict[str, list[str]] = collections.defaultdict(list)
        for r in inner:
            groups[fn(r)].append(r)
        out.extend(group_report(name, groups, len(inner)))
        out.append("")

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="outputs/recon_report.md")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    if not data_root.is_dir():
        print(f"ERROR: {data_root} is not a directory")
        return 1

    sources = sorted(p for p in data_root.iterdir() if p.is_dir() and not p.name.startswith("."))
    report = [
        "# Recon report (Phase 1, Step 1)",
        "",
        "Read-only. Image metadata only: no pixels printed, nothing copied out of `data/`.",
        "",
        f"Data root: `{data_root}`",
        f"Sources discovered: {len(sources)} {[s.name for s in sources]}",
    ]

    if not sources:
        loose = [p for p in data_root.iterdir() if p.is_file() and not p.name.startswith(".")]
        report += [
            "",
            "## No source directories found",
            "",
            f"`{data_root}` contains no subdirectories. Loose files present: "
            f"{[p.name for p in loose] or 'none'}.",
            "",
            "Expected layout (per START_HERE.md):",
            "```",
            "data/deeplure/   DeepLure Drive corpus (proprietary)",
            "data/kaggle/     Indian Saree Patterns",
            "```",
        ]
    else:
        for src in sources:
            report += recon_source(src, data_root)

    text = "\n".join(report) + "\n"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"[recon] written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
