#!/usr/bin/env python3
"""
Compare TS raw ordering vs rule-based repaired chain (baseline).

Outputs:
  - outputs/evaluation/ts_raw_vs_repaired.md
  - outputs/evaluation/ts_raw_vs_repaired.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spine_point_cloud_assembly.utils.anatomy_transitions import nvc_transition, strict_nvc_transition  # noqa: E402
from spine_point_cloud_assembly.utils.soed import soed  # noqa: E402


def _parse_int_list(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(x) for x in s.split()]


def _summarize(records: list[dict], key: str) -> dict:
    vals = [float(r[key]) for r in records if r.get(key) is not None and str(r[key]) != "nan"]
    if not vals:
        return {"mean": float("nan"), "median": float("nan")}
    return {"mean": float(np.mean(vals)), "median": float(np.median(vals))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ts-report",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/spine_point_cloud_assembly/outputs/evaluation/ts_failure_report.csv"),
    )
    parser.add_argument(
        "--repaired-report",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/spine_point_cloud_assembly/outputs/evaluation/ts_repaired_report.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/spine_point_cloud_assembly/outputs/evaluation"),
    )
    args = parser.parse_args()

    # Load TS raw per-subject info
    ts: dict[str, dict] = {}
    with args.ts_report.open() as f:
        r = csv.DictReader(f)
        for row in r:
            ts[row["subject_id"]] = row

    # Load repaired per-subject
    repaired: dict[str, dict] = {}
    with args.repaired_report.open() as f:
        r = csv.DictReader(f)
        for row in r:
            repaired[row["subject_id"]] = row

    per_case: list[dict] = []
    for sid, tr in ts.items():
        gt = _parse_int_list(tr.get("gt_order", ""))
        pred_raw = _parse_int_list(tr.get("pred_order", ""))
        pred_rep = _parse_int_list(repaired.get(sid, {}).get("repaired_pred_order", ""))

        rec = {
            "subject_id": sid,
            "raw_len": len(pred_raw),
            "repaired_len": len(pred_rep),
            "raw_nvc": nvc_transition(pred_raw, consider_direction=False) if pred_raw else 0.0,
            "repaired_nvc": nvc_transition(pred_rep, consider_direction=False) if pred_rep else 0.0,
            "raw_strict_nvc": strict_nvc_transition(pred_raw, consider_direction=False) if pred_raw else 0.0,
            "repaired_strict_nvc": strict_nvc_transition(pred_rep, consider_direction=False) if pred_rep else 0.0,
            "raw_soed": soed(gt, pred_raw, normalize_by="gt") if gt and pred_raw else float("nan"),
            "repaired_soed": soed(gt, pred_rep, normalize_by="gt") if gt and pred_rep else float("nan"),
            "extra_vs_gt_labels": tr.get("extra_vs_gt_labels", ""),
            "spurious_non_spine_proxy_count": int(float(tr.get("spurious_non_spine_proxy_count", 0) or 0)),
            "removed_spurious_count": int(float(repaired.get(sid, {}).get("removed_spurious_count", 0) or 0)),
        }
        per_case.append(rec)

    summary = {
        "n_subjects": len(per_case),
        "raw": {
            "nvc": _summarize(per_case, "raw_nvc"),
            "strict_nvc": _summarize(per_case, "raw_strict_nvc"),
            "soed": _summarize(per_case, "raw_soed"),
            "len": _summarize(per_case, "raw_len"),
        },
        "repaired": {
            "nvc": _summarize(per_case, "repaired_nvc"),
            "strict_nvc": _summarize(per_case, "repaired_strict_nvc"),
            "soed": _summarize(per_case, "repaired_soed"),
            "len": _summarize(per_case, "repaired_len"),
        },
        "spurious": {
            "raw_spurious_count": _summarize(per_case, "spurious_non_spine_proxy_count"),
            "removed_spurious_count": _summarize(per_case, "removed_spurious_count"),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ts_raw_vs_repaired.json").write_text(json.dumps({"summary": summary, "per_case": per_case}, indent=2))

    md = []
    md.append("# TS raw vs TS+repair baseline\n\n")
    md.append(f"- N subjects: **{summary['n_subjects']}**\n\n")
    md.append("## Aggregate metrics\n\n")
    md.append("| Metric | TS raw (mean/median) | TS+repair (mean/median) |\n")
    md.append("|---|---:|---:|\n")
    md.append(
        f"| NVC(validTransition) | {summary['raw']['nvc']['mean']:.3f}/{summary['raw']['nvc']['median']:.3f} | "
        f"{summary['repaired']['nvc']['mean']:.3f}/{summary['repaired']['nvc']['median']:.3f} |\n"
    )
    md.append(
        f"| strict NVC | {summary['raw']['strict_nvc']['mean']:.3f}/{summary['raw']['strict_nvc']['median']:.3f} | "
        f"{summary['repaired']['strict_nvc']['mean']:.3f}/{summary['repaired']['strict_nvc']['median']:.3f} |\n"
    )
    md.append(
        f"| SOED (vs GT order) | {summary['raw']['soed']['mean']:.3f}/{summary['raw']['soed']['median']:.3f} | "
        f"{summary['repaired']['soed']['mean']:.3f}/{summary['repaired']['soed']['median']:.3f} |\n"
    )
    md.append(
        f"| Seq length | {summary['raw']['len']['mean']:.2f}/{summary['raw']['len']['median']:.2f} | "
        f"{summary['repaired']['len']['mean']:.2f}/{summary['repaired']['len']['median']:.2f} |\n"
    )
    md.append(
        f"| Spurious(non-spine) count | {summary['spurious']['raw_spurious_count']['mean']:.2f}/{summary['spurious']['raw_spurious_count']['median']:.2f} | "
        f"removed {summary['spurious']['removed_spurious_count']['mean']:.2f}/{summary['spurious']['removed_spurious_count']['median']:.2f} |\n"
    )
    md.append("\n## Notes\n")
    md.append("- Repair baseline removes **spurious_non_spine_proxy** (HU non-bone / axis-outlier) and keeps the remaining labels in the original TS order (no GT used).\n")
    md.append("- The key takeaway is the **trade-off**: removing spurious instances alone can create **gaps** (missing intermediate levels), which hurts continuity (NVC/strict-NVC) and alignment to GT (SOED).\n")
    md.append("- This is a **heuristic**; Spinal Field aims to solve this as structured inference (and also correct ID confusion).\n")

    (args.out_dir / "ts_raw_vs_repaired.md").write_text("".join(md))
    print(f"✓ Wrote: {args.out_dir / 'ts_raw_vs_repaired.md'}")
    print(f"✓ Wrote: {args.out_dir / 'ts_raw_vs_repaired.json'}")


if __name__ == "__main__":
    main()


