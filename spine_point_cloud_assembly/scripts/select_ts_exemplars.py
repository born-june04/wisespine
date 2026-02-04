#!/usr/bin/env python3
"""
Select representative TS failure exemplars by category and generate supporting visuals.

Outputs under:
  outputs/evaluation/exemplars/<category>/<subject_id>/
    - overlays: sagittal/axial combined panels (RAW|RAW+GT|RAW+TS)
    - chain.png: predicted ID chain with spurious/extra/violations highlighted
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

# Ensure imports work when executed as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spine_point_cloud_assembly.utils.anatomy_transitions import transition_violations  # noqa: E402


def _parse_int_list(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(x) for x in s.split()]


def _ensure_overlays(subject_id: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = out_dir / f"{subject_id}_sagittal_combined.png"
    if expected.exists():
        return
    cmd = [
        sys.executable,
        "-m",
        "totalseg_eval.visualize_overlays",
        "--subject-id",
        subject_id,
        "--views",
        "sagittal,axial",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)


def _save_chain_plot(row: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    pred_order = _parse_int_list(row.get("pred_order", ""))
    extra_vs_gt = set(_parse_int_list(row.get("extra_vs_gt_labels", "")))
    axis_outliers = set(_parse_int_list(row.get("axis_outlier_labels", "")))
    non_bone = set(_parse_int_list(row.get("non_bone_labels", "")))
    spurious = set(axis_outliers).union(non_bone)
    viol = transition_violations(pred_order, consider_direction=False) if pred_order else []

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    if not pred_order:
        ax.text(0.5, 0.5, "No pred_order", ha="center", va="center")
        ax.axis("off")
    else:
        xs = np.arange(len(pred_order))
        ys = np.array(pred_order)
        colors = []
        for y in ys.tolist():
            if y in spurious:
                colors.append("red")
            elif y in extra_vs_gt:
                colors.append("magenta")
            else:
                colors.append("dodgerblue")
        ax.plot(xs, ys, color="black", alpha=0.3)
        ax.scatter(xs, ys, c=colors, s=50)
        for v in viol:
            ax.axvspan(v.index - 0.2, v.index + 0.2, color="orange", alpha=0.25)
        ax.set_xlabel("Position along predicted order")
        ax.set_ylabel("Label ID")
        ax.grid(True, alpha=0.2)

    sid = row.get("subject_id", "")
    title = (
        f"{sid} | spurious={row.get('spurious_non_spine_proxy_count','?')} "
        f"(non_bone={row.get('non_bone_count','?')}, axis_outliers={row.get('axis_outliers_count','?')}) "
        f"| extra_vs_gt={row.get('extra_vs_gt_labels','').strip()} "
        f"| viol={row.get('transition_violations_count','?')}"
    )
    ax.set_title(title, fontweight="bold", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/spine_point_cloud_assembly/outputs/evaluation/ts_failure_report.csv"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/spine_point_cloud_assembly/outputs/evaluation/exemplars"),
    )
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    rows: list[dict] = []
    with args.report_csv.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    def top_by(key: str) -> list[dict]:
        return sorted(rows, key=lambda r: (int(r.get(key, 0)), -float(r.get("mean_dice", 0.0) or 0.0)), reverse=True)[: args.k]

    categories = {
        "non_bone_spurious": top_by("non_bone_count"),
        "axis_outlier_spurious": top_by("axis_outliers_count"),
        "transition_violations": top_by("transition_violations_count"),
        "extra_vs_gt": top_by("extra_vs_gt_count"),
    }

    for cat, chosen in categories.items():
        for row in chosen:
            sid = row["subject_id"]
            out_dir = args.out_root / cat / sid
            overlay_dir = out_dir / "overlays"
            _ensure_overlays(sid, overlay_dir)
            _save_chain_plot(row, out_dir / "chain.png")

    print(f"✓ Saved exemplars under: {args.out_root}")


if __name__ == "__main__":
    main()


