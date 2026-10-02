"""
Adapter for the proprietary DeepLure Drive corpus.

Raw layout is as flat as it gets:

    data/Google_drive_data/handloom_sarees/img_<id>.jpg
    data/Google_drive_data/handloom_sarees/h_img_<id>.jpg

There are no labels, no metadata file, and no folder structure to exploit. Identity is
therefore established downstream by geometric verification, not here.

Two findings from recon shape this adapter:

1. The `h_` prefix carries NO pairing information. Zero of the 9 `h_` files share a numeric
   id with any plain `img_` file, and content matching found no systematic relationship
   (outputs/hprefix_probe.md). It is stripped for readability only, never used as a label.

2. The corpus holds 95 designs, not 165, because 27 designs appear as several overlapping
   crops of one photograph. pHash cannot see this, so grouping happens in build_manifest
   via geometric verification. This adapter emits one record per file and stays dumb.
"""

from __future__ import annotations

from .base import BaseAdapter, Record, stable_image_id


class DeepLureAdapter(BaseAdapter):
    name = "deeplure"

    def build_records(self) -> list[Record]:
        prefixes = self.cfg.get("strip_filename_prefixes", [])
        confidence = self.cfg.get("label_confidence", "high")
        records: list[Record] = []

        for path in self.iter_files():
            meta = self.read_meta(path)
            if meta is None:
                continue
            w, h, ph = meta

            stem = path.stem
            note_bits = []
            for pre in prefixes:
                if stem.startswith(pre):
                    stem = stem[len(pre):]
                    # Recorded because it is provenance we do not understand, and a reader
                    # should not assume it was meaningless just because we could not use it.
                    note_bits.append(f"had filename prefix '{pre}' (no pairing meaning found)")
                    break

            records.append(Record(
                image_id=stable_image_id(self.rel(path)),
                path=self.rel(path),
                source=self.source_key,
                craft_family=self.cfg.get("craft_family", ""),
                width=w,
                height=h,
                phash=ph,
                label_confidence=confidence,
                notes="; ".join(note_bits),
                hints={"stem": stem},
            ))

        return records
