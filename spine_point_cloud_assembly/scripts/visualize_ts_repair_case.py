#!/usr/bin/env python3
"""
Visualize TS raw vs repaired ordering for a single subject.

Outputs:
  outputs/evaluation/repair_cases/<subject_id>/chain_raw_vs_repaired.png
  outputs/evaluation/repair_cases/<subject_id>/overlays/* (reuse totalseg overlay)
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from spine_point_cloud_assembly.utils.anatomy_transitions import transition_violations  # noqa: E402


def _parse_int_list(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(x) for x in s.split()]


def _ensure_overlays(subject_id: str, out_dir: Path) -> Path:
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
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-id", required=True)
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
        default=Path("/gscratch/scrubbed/june0604/vindr/spine_point_cloud_assembly/outputs/evaluation/repair_cases"),
    )
    args = parser.parse_args()

    sid = args.subject_id
    ts_row = None
    with args.ts_report.open() as f:
        r = csv.DictReader(f)
        for row in r:
            if row["subject_id"] == sid:
                ts_row = row
                break
    if ts_row is None:
        raise ValueError(f"subject not found in ts report: {sid}")

    rep_row = None
    with args.repaired_report.open() as f:
        r = csv.DictReader(f)
        for row in r:
            if row["subject_id"] == sid:
                rep_row = row
                break
    if rep_row is None:
        raise ValueError(f"subject not found in repaired report: {sid}")

    raw = _parse_int_list(ts_row.get("pred_order", ""))
    repaired = _parse_int_list(rep_row.get("repaired_pred_order", ""))
    extra_vs_gt = set(_parse_int_list(ts_row.get("extra_vs_gt_labels", "")))
    non_bone = set(_parse_int_list(ts_row.get("non_bone_labels", "")))
    axis_outliers = set(_parse_int_list(ts_row.get("axis_outlier_labels", "")))
    spurious = set(non_bone).union(axis_outliers)

    out_case = args.out_dir / sid
    overlay_dir = out_case / "overlays"
    _ensure_overlays(sid, overlay_dir)

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), gridspec_kw={"width_ratios": [1.3, 1.0]})

    # Overlay panel
    img = mpimg.imread(str(overlay_dir / f"{sid}_sagittal_combined.png"))
    axes[0, 0].imshow(img)
    axes[0, 0].axis("off")
    axes[0, 0].set_title("CT | CT+GT | CT+TS (sagittal)", fontweight="bold")

    # Raw chain
    def plot_chain(ax, seq, title):
        if not seq:
            ax.text(0.5, 0.5, "empty", ha="center", va="center")
            ax.axis("off")
            return
        xs = np.arange(len(seq))
        ys = np.array(seq)
        colors = []
        for y in ys.tolist():
            if y in spurious:
                colors.append("red")
            elif y in extra_vs_gt:
                colors.append("magenta")
            else:
                colors.append("dodgerblue")
        ax.plot(xs, ys, color="black", alpha=0.3)
        ax.scatter(xs, ys, s=55, c=colors)
        viol = transition_violations(seq, consider_direction=False)
        for v in viol:
            ax.axvspan(v.index - 0.2, v.index + 0.2, color="orange", alpha=0.25)
        ax.set_xlabel("pos")
        ax.set_ylabel("label id")
        ax.grid(True, alpha=0.2)
        ax.set_title(title, fontweight="bold")

    plot_chain(axes[0, 1], raw, "Raw TS predicted order")
    plot_chain(axes[1, 1], repaired, "Repaired (spurious removed, order preserved)")

    # Text summary
    axes[1, 0].axis("off")
    txt = (
        f"subject={sid}\n"
        f"extra_vs_gt={ts_row.get('extra_vs_gt_labels','').strip()}\n"
        f"non_bone={ts_row.get('non_bone_labels','').strip()}\n"
        f"axis_outliers={ts_row.get('axis_outlier_labels','').strip()}\n"
        f"raw_len={len(raw)} → repaired_len={len(repaired)}\n"
        f"raw spurious_non_spine_proxy={ts_row.get('spurious_non_spine_proxy_count','?')} "
        f"(non_bone={ts_row.get('non_bone_count','?')}, axis_outliers={ts_row.get('axis_outliers_count','?')})\n"
        f"removed_spurious={rep_row.get('removed_spurious_labels','').strip()}\n"
        "\nInterpretation:\n"
        "- Removing spurious masks alone can create gaps (missing intermediate levels),\n"
        "  which motivates structured inference (Spinal Field) rather than segmentation-only postprocessing.\n"
    )
    axes[1, 0].text(0.02, 0.98, txt, va="top", ha="left", fontsize=11)

    fig.tight_layout()
    out_case.mkdir(parents=True, exist_ok=True)
    out_path = out_case / "chain_raw_vs_repaired.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"✓ Saved: {out_path}")


if __name__ == "__main__":
    main()


