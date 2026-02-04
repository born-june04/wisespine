#!/usr/bin/env python3
"""
Figure 1 generator (B3): "Segmentation vs Structure mismatch".

Creates a multi-row figure:
  - Left: CT + TS overlay (sagittal combined panel from totalseg_eval.visualize_overlays)
  - Right: Predicted ID chain + violations + extra labels annotation

Inputs:
  - spine_point_cloud_assembly/outputs/evaluation/ts_failure_report.csv

Output:
  - spine_point_cloud_assembly/outputs/evaluation/fig1_ts_structure_mismatch.png
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


def _ensure_overlay(subject_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = out_dir / f"{subject_id}_sagittal_combined.png"
    if combined.exists():
        return combined
    cmd = [
        sys.executable,
        "-m",
        "totalseg_eval.visualize_overlays",
        "--subject-id",
        subject_id,
        "--views",
        "sagittal",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    if not combined.exists():
        raise FileNotFoundError(f"Expected overlay not found: {combined}")
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/spine_point_cloud_assembly/outputs/evaluation/ts_failure_report.csv"),
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path(
            "/gscratch/scrubbed/june0604/vindr/spine_point_cloud_assembly/outputs/evaluation/fig1_ts_structure_mismatch.png"
        ),
    )
    parser.add_argument(
        "--overlay-cache-dir",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/spine_point_cloud_assembly/outputs/evaluation/fig1_overlays"),
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    rows: list[dict] = []
    with args.report_csv.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    def score(row: dict) -> tuple:
        # prioritize "non-spine/outlier" first, then extra labels, then transition violations
        return (
            int(row.get("axis_outliers_count", 0)),
            int(row.get("non_bone_count", 0)),
            int(row.get("extra_vs_gt_count", 0)),
            int(row.get("transition_violations_count", 0)),
            -float(row.get("mean_dice", 0.0) or 0.0),
        )

    rows.sort(key=score, reverse=True)
    chosen = rows[: max(1, args.top_k)]

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    n = len(chosen)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(18, 6 * n))
    if n == 1:
        axes = np.array([axes])

    for i, row in enumerate(chosen):
        sid = row["subject_id"]
        overlay_path = _ensure_overlay(sid, args.overlay_cache_dir / sid)
        img = mpimg.imread(str(overlay_path))

        ax_img = axes[i, 0]
        ax_chain = axes[i, 1]

        ax_img.imshow(img)
        ax_img.set_title(f"{sid}: CT | CT+GT | CT+TS (sagittal)", fontweight="bold")
        ax_img.axis("off")

        pred_order = _parse_int_list(row.get("pred_order", ""))
        extra_vs_gt = set(_parse_int_list(row.get("extra_vs_gt_labels", "")))
        axis_outliers = set(_parse_int_list(row.get("axis_outlier_labels", "")))
        non_bone = set(_parse_int_list(row.get("non_bone_labels", "")))
        spurious = set(axis_outliers).union(non_bone)

        if pred_order:
            xs = np.arange(len(pred_order))
            ys = np.array(pred_order)
            ax_chain.plot(xs, ys, color="white", linewidth=2.0, alpha=0.7)
            # Color meaning:
            # - red: spurious non-spine proxy (HU non-bone OR axis outlier)
            # - magenta: extra vs GT (coverage mismatch; not necessarily spurious)
            # - blue: normal
            colors = []
            for y in ys.tolist():
                if y in spurious:
                    colors.append("red")
                elif y in extra_vs_gt:
                    colors.append("magenta")
                else:
                    colors.append("dodgerblue")
            ax_chain.scatter(xs, ys, s=40, c=colors)

            viol = transition_violations(pred_order, consider_direction=False)
            for v in viol:
                ax_chain.axvspan(v.index - 0.2, v.index + 0.2, color="orange", alpha=0.25)

            ax_chain.set_title(
                "Predicted ID chain (red=spurious(non-spine), magenta=extra-vs-GT, orange=transition violation)",
                fontweight="bold",
            )
            ax_chain.set_xlabel("Position along predicted order")
            ax_chain.set_ylabel("Label ID")
            ax_chain.grid(True, alpha=0.2)
        else:
            ax_chain.text(0.5, 0.5, "No pred_order available", ha="center", va="center")
            ax_chain.axis("off")

        # Metrics text box
        metrics_txt = (
            f"mean_dice={float(row.get('mean_dice', 'nan')):.3f}\n"
            f"extra_vs_gt={row.get('extra_vs_gt_labels', '').strip()}\n"
            f"spurious_non_bone={row.get('non_bone_labels', '').strip()}\n"
            f"spurious_axis_outlier={row.get('axis_outlier_labels', '').strip()}\n"
            f"violations={row.get('transition_violations_count', '0')}\n"
            f"non_bone={row.get('non_bone_count', '0')}\n"
            f"SOED(ts)={float(row.get('soed_totalseg', 'nan')):.3f}\n"
            f"NVC(ts)={float(row.get('nvc_totalseg', 'nan')):.3f}"
        )
        ax_chain.text(
            0.02,
            0.98,
            metrics_txt,
            transform=ax_chain.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
            color="white",
        )

    fig.suptitle("Figure 1: Segmentation vs Structure Mismatch (TS)", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=180)
    plt.close(fig)
    print(f"✓ Saved Figure 1 to: {args.out_path}")


if __name__ == "__main__":
    main()


