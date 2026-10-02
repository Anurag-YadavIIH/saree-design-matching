"""
Apply the pre-registered success rule (DECISIONS.md D-34) mechanically.

Two steps, run in this order and never the other way round:

  python scripts/judge_d34.py bars    --baselines outputs/results_baselines.json
      Reads the gray zero-shot DINOv2 row and writes outputs/bars.json: the upper bounds of its
      95% CIs on tonal Rank-1, main TAR@1e-3 and real_view_recolored Rank-1, plus the lower
      bound of its main Rank-1 CI as the guardrail. No checkpoint has been evaluated yet.

  python scripts/judge_d34.py verdict --checkpoints outputs/results_checkpoints.json
      Reads bars.json and scores each trained model. SUCCESS needs BOTH:
        1. a point estimate ABOVE gray's CI upper bound on at least one of the three metrics
        2. main Rank-1 at or above gray's CI lower bound

Writing the bars to a file before scoring means the bars cannot be chosen after seeing results.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REFERENCE = "DINOv2 zero-shot gray"


def parse(cell: str) -> tuple[float, float, float]:
    """'0.913 [0.878, 0.943]' -> (0.913, 0.878, 0.943)."""
    m = re.match(r"\s*([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]", cell)
    if not m:
        raise ValueError(f"cannot parse a value with CI from {cell!r}")
    return tuple(float(x) for x in m.groups())


def metrics(row: dict) -> dict:
    tar = row["tar_ci"]
    return {
        "main_rank1": parse(row["identification"]["Rank-1"]),
        "main_tar": (tar["tar"], *tar["ci"]),
        "tonal_rank1": parse(row["tonal"]["Rank-1"]),
        "real_view_recolored_rank1": (parse(row["real_pairs"]["real_view_recolored"]["Rank-1"])
                                      if "real_view_recolored" in row.get("real_pairs", {})
                                      else None),
    }


def cmd_bars(args) -> int:
    rows = {r["method"]: r for r in json.loads(Path(args.baselines).read_text(encoding="utf-8"))}
    m = metrics(rows[REFERENCE])
    bars = {
        "reference": REFERENCE,
        "source": args.baselines,
        "beat_any": {
            "tonal_rank1": m["tonal_rank1"][2],
            "main_tar": m["main_tar"][2],
            "real_view_recolored_rank1": (m["real_view_recolored_rank1"][2]
                                          if m["real_view_recolored_rank1"] else None),
        },
        "guardrail_main_rank1_min": m["main_rank1"][1],
        "reference_values": {k: v for k, v in m.items()},
    }
    Path(args.out).write_text(json.dumps(bars, indent=2), encoding="utf-8")
    print(f"[judge] bars from {REFERENCE} (written to {args.out} BEFORE any checkpoint is scored):")
    for k, v in bars["beat_any"].items():
        print(f"         {k:28s} must exceed {v}")
    print(f"         {'main_rank1 guardrail':28s} must be >= {bars['guardrail_main_rank1_min']}")
    return 0


def cmd_verdict(args) -> int:
    bars = json.loads(Path(args.bars).read_text(encoding="utf-8"))
    rows = json.loads(Path(args.checkpoints).read_text(encoding="utf-8"))
    out = []
    for r in rows:
        if not r["method"].startswith("trained"):
            continue
        m = metrics(r)
        beats = {}
        for k, bar in bars["beat_any"].items():
            if bar is None or m.get(k) is None:
                continue
            beats[k] = {"value": m[k][0], "bar": bar, "beats": m[k][0] > bar}
        guard_val = m["main_rank1"][0]
        guard_ok = guard_val >= bars["guardrail_main_rank1_min"]
        success = any(b["beats"] for b in beats.values()) and guard_ok
        out.append({"method": r["method"], "criteria": beats,
                    "guardrail": {"value": guard_val, "min": bars["guardrail_main_rank1_min"],
                                  "ok": guard_ok},
                    "verdict": "SUCCESS" if success else "FAIL"})
        print(f"[judge] {r['method']}: {'SUCCESS' if success else 'FAIL'}")
        for k, b in beats.items():
            print(f"         {k:28s} {b['value']:.3f} vs bar {b['bar']:.3f}  "
                  f"{'beats' if b['beats'] else 'does not beat'}")
        print(f"         {'main_rank1 guardrail':28s} {guard_val:.3f} vs min "
              f"{bars['guardrail_main_rank1_min']:.3f}  {'ok' if guard_ok else 'FAILS'}")
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    any_success = any(o["verdict"] == "SUCCESS" for o in out)
    print(f"[judge] D-34 outcome: {'at least one checkpoint succeeds' if any_success else 'no checkpoint succeeds: ship gray zero-shot DINOv2 plus projection (fallback note)'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bars")
    b.add_argument("--baselines", default="outputs/results_baselines.json")
    b.add_argument("--out", default="outputs/bars.json")
    v = sub.add_parser("verdict")
    v.add_argument("--checkpoints", default="outputs/results_checkpoints.json")
    v.add_argument("--bars", default="outputs/bars.json")
    v.add_argument("--out", default="outputs/verdict.json")
    args = ap.parse_args()
    return cmd_bars(args) if args.cmd == "bars" else cmd_verdict(args)


if __name__ == "__main__":
    raise SystemExit(main())
