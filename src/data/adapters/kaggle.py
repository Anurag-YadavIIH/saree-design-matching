"""
Adapter for the Kaggle / Roboflow source: Indian Fabric Patterns v1, MIT licence.

Raw layout:

    data/Kaggle_data/<vendor split>/<CraftFamily>/<original stem>_<ext>.rf.<32 hex>.jpg

Three recon findings shape this adapter:

1. **The vendor's train/valid/test split is discarded.** 88 duplicate clusters straddle its
   boundaries, so reusing it would leak. The split directory is read only to confirm it is
   being ignored deliberately, and recorded in notes.

2. **The craft family is not identity.** Banarasi, Bandhani, Ikat and Pichwai are weave and
   craft categories. 432 Banarasi images are 432 different designs. The family is kept
   because it stratifies the split and defines the hardest tier of stress test B, but it
   must never be used as a design label.

3. **The filename stem is not a safe key.** Roboflow encodes the original name as
   `<stem>_<ext>.rf.<hash>`, and genuine augmentation triplets do share a stem. But 314
   stems are generic (`image21`, `images4`): within those, median pHash distance is 28,
   statistically indistinguishable from unrelated images, and 53 span multiple craft
   families. They are filename collisions. So the stem is emitted as a HINT only, flagged
   when generic, and duplicate detection is done by content downstream.
"""

from __future__ import annotations

import re

from .base import BaseAdapter, Record, stable_image_id


class KaggleAdapter(BaseAdapter):
    name = "kaggle"

    def build_records(self) -> list[Record]:
        rf_re = re.compile(self.cfg["roboflow_filename_re"], re.IGNORECASE)
        generic_re = re.compile(self.cfg["generic_stem_re"], re.IGNORECASE)
        family_idx = int(self.cfg.get("craft_family_from_path_index", 1))
        vendor_dirs = set(self.cfg.get("ignore_vendor_split_dirs", []))
        confidence = self.cfg.get("label_confidence", "medium")

        records: list[Record] = []
        for path in self.iter_files():
            meta = self.read_meta(path)
            if meta is None:
                continue
            w, h, ph = meta

            rel = self.rel(path)
            # Path components below the source root: <vendor split>/<CraftFamily>/<file>
            try:
                inner = path.resolve().relative_to(self.root.resolve()).parts
            except ValueError:
                self.unmatched.append(self._unmatched(rel, "path is outside the source root"))
                continue

            if len(inner) <= family_idx:
                self.unmatched.append(self._unmatched(
                    rel, f"path too shallow to contain a craft family at index {family_idx}"))
                continue

            vendor_split = inner[0] if inner[0] in vendor_dirs else ""
            craft_family = inner[family_idx]

            notes = []
            if vendor_split:
                # Stated explicitly so a reader cannot mistake the discard for an oversight.
                notes.append(f"vendor split '{vendor_split}' ignored (88 dup clusters "
                             f"straddle its boundaries)")

            m = rf_re.match(path.stem)
            if m:
                stem = m.group("stem")
            else:
                stem = path.stem
                notes.append("filename does not match the Roboflow export pattern")

            is_generic = bool(generic_re.match(stem))
            if is_generic:
                notes.append(f"generic stem '{stem}' is a filename collision, not an identity")

            records.append(Record(
                image_id=stable_image_id(rel),
                path=rel,
                source=self.source_key,
                craft_family=craft_family,
                width=w,
                height=h,
                phash=ph,
                label_confidence=confidence,
                notes="; ".join(notes),
                hints={"stem": stem, "generic_stem": is_generic,
                       "vendor_split": vendor_split},
            ))

        return records

    def _unmatched(self, rel: str, reason: str):
        from .base import UnmatchedFile
        return UnmatchedFile(rel, self.source_key, reason)
