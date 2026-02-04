#!/usr/bin/env python3
"""
Rule-based repair baseline for TS mask-based reconstruction.

Goal: show that segmentation-only outputs require *structural inference* to become usable.

This baseline is intentionally conservative:
1) remove evidence-based spurious non-spine instances:
   - non_bone (HU quantile low) OR axis_outlier
2) keep the remaining labels in the original TS-predicted order (no GT used)

This keeps coverage/levels (e.g., S1) intact when they look anatomically plausible,
and only removes masks that look like non-spine.

This is NOT the final method (Spinal Field), but a strong "TS + heuristics" baseline.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _parse_int_list(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(x) for x in s.split()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/spine_point_cloud_assembly/outputs/evaluation/ts_failure_report.csv"),
        help="Subject-level report produced by ts_failure_report.py",
    )
    parser.add_argument(
        "--instance-csv",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/spine_point_cloud_assembly/outputs/evaluation/failure_types.csv"),
        help="Per-instance report produced by ts_failure_report.py",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/spine_point_cloud_assembly/outputs/evaluation/ts_repaired_report.csv"),
    )
    args = parser.parse_args()

    # Load per-instance spurious flags
    inst: dict[str, dict[int, dict]] = {}
    with args.instance_csv.open() as f:
        r = csv.DictReader(f)
        for row in r:
            sid = row["subject_id"]
            lab = int(row["label"])
            inst.setdefault(sid, {})[lab] = row

    out_rows: list[dict] = []
    with args.report_csv.open() as f:
        r = csv.DictReader(f)
        for row in r:
            sid = row["subject_id"]
            pred_order = _parse_int_list(row.get("pred_order", ""))
            if not pred_order:
                pred_order = _parse_int_list(row.get("extra_vs_gt_labels", ""))

            # Remove spurious evidence-based instances
            kept_order: list[int] = []
            removed: list[int] = []
            for lab in pred_order:
                info = inst.get(sid, {}).get(lab)
                is_spurious = False
                if info is not None:
                    is_spurious = int(info.get("is_spurious_non_spine_proxy", "0")) == 1
                if is_spurious:
                    removed.append(lab)
                else:
                    kept_order.append(lab)

            out_rows.append(
                {
                    "subject_id": sid,
                    "raw_pred_len": len(pred_order),
                    "kept_len": len(kept_order),
                    "removed_spurious_count": len(removed),
                    "removed_spurious_labels": " ".join(map(str, removed)),
                    "repaired_pred_order": " ".join(map(str, kept_order)),
                    # keep original summary columns for convenience
                    "mean_dice": row.get("mean_dice", ""),
                    "extra_vs_gt_labels": row.get("extra_vs_gt_labels", ""),
                    "transition_violations_count": row.get("transition_violations_count", ""),
                    "pred_order": row.get("pred_order", ""),
                    "gt_order": row.get("gt_order", ""),
                }
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [])
        w.writeheader()
        w.writerows(out_rows)

    print(f"✓ Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()


