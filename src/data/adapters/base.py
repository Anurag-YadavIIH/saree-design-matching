"""
Adapter interface: the only place that is allowed to know what raw files look like.

Everything downstream reads `outputs/manifest.csv`. An adapter's single job is to turn one
raw source into a list of partially filled contract records. It deliberately does NOT assign
design_id, dup_group, split or role: those need a global view across sources and are decided
in src/data/build_manifest.py.

Adding a source is one new file plus one config block, with no edits anywhere else.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def stable_image_id(rel_path: str) -> str:
    """Primary key: a hash of the relative path.

    Derived from the path rather than the bytes so a row keeps its identity across rebuilds
    and so computing it never requires opening the file. The cost is that moving a file
    changes its id, which is acceptable because the corpus is read-only.
    """
    return hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:16]


@dataclass
class Record:
    """One image, as much of the contract as an adapter can know on its own."""
    image_id: str
    path: str                  # posix, relative to data_root
    source: str
    craft_family: str = ""
    width: int = 0
    height: int = 0
    phash: str = ""
    label_confidence: str = "medium"
    notes: str = ""
    # Adapter-local grouping hints, consumed by build_manifest and then discarded. The
    # Roboflow source stem goes here, for example: it is a useful hint but, because generic
    # stems collide across unrelated images, it is never trusted as an identity.
    hints: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "image_id": self.image_id,
            "path": self.path,
            "source": self.source,
            "craft_family": self.craft_family,
            "width": self.width,
            "height": self.height,
            "phash": self.phash,
            "label_confidence": self.label_confidence,
            "notes": self.notes,
        }
        return d


@dataclass
class UnmatchedFile:
    """A file no rule claimed. Logged, never silently dropped."""
    path: str
    source: str
    reason: str


class BaseAdapter(ABC):
    """One adapter per raw source."""

    #: registry key, matched against `adapter:` in configs/data.yaml
    name: str = "base"

    def __init__(self, source_key: str, cfg: dict, data_root: Path):
        self.source_key = source_key
        self.cfg = cfg
        self.data_root = data_root
        self.root = Path(cfg["root"])
        self.unmatched: list[UnmatchedFile] = []

    @abstractmethod
    def build_records(self) -> list[Record]:
        """Return one Record per usable image, appending to self.unmatched for the rest."""

    # ---------------------------------------------------------------- shared helpers

    def iter_files(self) -> list[Path]:
        pattern = self.cfg.get("glob", "**/*")
        files = sorted(p for p in self.root.glob(pattern) if p.is_file())
        kept = []
        for p in files:
            if p.name.startswith(".") or any(part.startswith(".") for part in p.parts):
                continue
            if p.suffix.lower() not in IMAGE_EXTS:
                self.unmatched.append(UnmatchedFile(
                    self.rel(p), self.source_key, f"not an image extension ({p.suffix})"))
                continue
            kept.append(p)
        return kept

    def rel(self, path: Path) -> str:
        return path.resolve().relative_to(self.data_root.resolve()).as_posix()

    def read_meta(self, path: Path) -> tuple[int, int, str] | None:
        """Size plus grayscale pHash. Returns None when the file cannot be read.

        pHash is kept as a cheap prefilter and as a baseline, but it is NOT the authority on
        duplicates: a translation offset leaves pixels intact while changing the DCT
        completely, which is how 27 Drive designs were initially missed.
        """
        try:
            import imagehash
            with Image.open(path) as im:
                w, h = im.size
                ph = str(imagehash.phash(im.convert("L")))
            return w, h, ph
        except Exception as exc:
            self.unmatched.append(UnmatchedFile(
                self.rel(path), self.source_key, f"unreadable: {type(exc).__name__}"))
            return None
