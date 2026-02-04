#!/usr/bin/env python3
"""
Summarize TS mask-based reconstruction limitations into paper-ready tables.

Input:
  - outputs/evaluation/ts_failure_report.csv
Output:
  - outputs/evaluation/ts_failure_summary.json
  - outputs/evaluation/ts_failure_summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/spine_point_cloud_assembly/outputs/evaluation/ts_failure_report.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/spine_point_cloud_assembly/outputs/evaluation"),
    )
    args = parser.parse_args()

    import pandas as pd

    df = pd.read_csv(args.report_csv)
    n = len(df)

    def frac(col: str) -> float:
        return float((df[col] > 0).mean()) if n else 0.0

    summary = {
        "n_subjects": int(n),
        "mean_dice_mean": float(df["mean_dice"].mean()),
        "mean_dice_median": float(df["mean_dice"].median()),
        "extra_vs_gt_any_frac": frac("extra_vs_gt_count"),
        "extra_vs_gt_mean": float(df["extra_vs_gt_count"].mean()),
        "transition_violations_any_frac": frac("transition_violations_count"),
        "transition_violations_mean": float(df["transition_violations_count"].mean()),
        "axis_outliers_any_frac": frac("axis_outliers_count"),
        "axis_outliers_mean": float(df["axis_outliers_count"].mean()),
        "non_bone_any_frac": frac("non_bone_count"),
        "non_bone_mean": float(df["non_bone_count"].mean()),
        "spurious_non_spine_any_frac": frac("spurious_non_spine_proxy_count"),
        "spurious_non_spine_mean": float(df["spurious_non_spine_proxy_count"].mean()),
    }

    # Top-k examples for each failure kind
    top_spurious = (
        df.sort_values(
            ["spurious_non_spine_proxy_count", "non_bone_count", "axis_outliers_count", "mean_dice"],
            ascending=[False, False, False, True],
        )
        .head(10)[
            [
                "subject_id",
                "mean_dice",
                "spurious_non_spine_proxy_count",
                "non_bone_count",
                "axis_outliers_count",
                "extra_vs_gt_labels",
            ]
        ]
        .to_dict(orient="records")
    )

    top_transitions = (
        df.sort_values(["transition_violations_count", "mean_dice"], ascending=[False, True])
        .head(10)[
            [
                "subject_id",
                "mean_dice",
                "transition_violations_count",
                "pred_order",
                "transition_violations",
            ]
        ]
        .to_dict(orient="records")
    )

    out = {
        "summary": summary,
        "top_spurious_non_spine": top_spurious,
        "top_transition_violations": top_transitions,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ts_failure_summary.json").write_text(json.dumps(out, indent=2))

    # Markdown table
    md = []
    md.append("# TS mask-based reconstruction: limitation summary\n")
    md.append(f"- **N subjects**: {n}\n")
    md.append("\n## Key rates (what breaks)\n")
    md.append("| Signal | Meaning | Rate |\n")
    md.append("|---|---|---:|\n")
    md.append(f"| extra_vs_gt | GT coverage mismatch (not necessarily error) | {_pct(summary['extra_vs_gt_any_frac'])} |\n")
    md.append(f"| transition violations | violates validTransition in predicted order | {_pct(summary['transition_violations_any_frac'])} |\n")
    md.append(f"| axis outliers | centroid far from estimated spine axis | {_pct(summary['axis_outliers_any_frac'])} |\n")
    md.append(f"| non-bone | HU p90 inside mask below threshold | {_pct(summary['non_bone_any_frac'])} |\n")
    md.append(f"| spurious_non_spine_proxy | (axis outlier OR non-bone) | {_pct(summary['spurious_non_spine_any_frac'])} |\n")
    md.append("\n## Averages (severity)\n")
    md.append("| Metric | Mean | Median |\n")
    md.append("|---|---:|---:|\n")
    md.append(f"| mean_dice | {summary['mean_dice_mean']:.3f} | {summary['mean_dice_median']:.3f} |\n")
    md.append(f"| extra_vs_gt_count | {summary['extra_vs_gt_mean']:.2f} | {float(df['extra_vs_gt_count'].median()):.2f} |\n")
    md.append(f"| transition_violations_count | {summary['transition_violations_mean']:.2f} | {float(df['transition_violations_count'].median()):.2f} |\n")
    md.append(f"| spurious_non_spine_proxy_count | {summary['spurious_non_spine_mean']:.2f} | {float(df['spurious_non_spine_proxy_count'].median()):.2f} |\n")
    md.append("\n## Top examples (spurious non-spine proxy)\n")
    md.append("| subject | mean_dice | spurious | non_bone | axis_outliers | extra_vs_gt |\n")
    md.append("|---|---:|---:|---:|---:|---|\n")
    for r in top_spurious[:5]:
        md.append(
            f"| {r['subject_id']} | {r['mean_dice']:.3f} | {int(r['spurious_non_spine_proxy_count'])} | "
            f"{int(r['non_bone_count'])} | {int(r['axis_outliers_count'])} | {str(r.get('extra_vs_gt_labels', '')).strip()} |\n"
        )

    (args.out_dir / "ts_failure_summary.md").write_text("".join(md))
    print(f"✓ Wrote: {args.out_dir / 'ts_failure_summary.json'}")
    print(f"✓ Wrote: {args.out_dir / 'ts_failure_summary.md'}")


if __name__ == "__main__":
    main()

