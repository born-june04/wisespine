#!/usr/bin/env python3
"""
Update summary.json from existing per-case metrics JSON files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def update_summary(results_dir: Path, output_path: Path | None = None) -> dict:
    files = sorted(p for p in results_dir.glob("*.json") if p.name != "summary.json")
    results = []
    for path in files:
        with open(path, "r") as f:
            results.append(json.load(f))

    dice_vals = np.array([r.get("overall", {}).get("mean_dice", 0.0) for r in results], dtype=float)
    iou_vals = np.array([r.get("overall", {}).get("mean_iou", 0.0) for r in results], dtype=float)

    mean_dice = float(np.mean(dice_vals)) if results else 0.0
    mean_iou = float(np.mean(iou_vals)) if results else 0.0
    median_dice = float(np.median(dice_vals)) if results else 0.0
    median_iou = float(np.median(iou_vals)) if results else 0.0
    std_dice = float(np.std(dice_vals)) if results else 0.0
    std_iou = float(np.std(iou_vals)) if results else 0.0

    structural_vals = {"nvc": [], "vla": [], "soed": [], "strict_nvc": []}
    structural_available = 0
    extra_pred_cases = []
    extra_pred_total = 0
    extra_pred_rate_vals = []

    for r in results:
        labels = set(r.get("labels", []))
        pred_labels = set(r.get("pred_labels", []))
        extra = sorted(pred_labels - labels)
        if extra:
            extra_pred_cases.append({"subject": Path(r.get("pred_path", "")).name, "extra_labels": extra})
            extra_pred_total += len(extra)
        if pred_labels:
            extra_pred_rate_vals.append(len(extra) / len(pred_labels))
        else:
            extra_pred_rate_vals.append(0.0)

        structural = r.get("structural", {})
        if structural.get("available"):
            structural_available += 1
            structural_vals["nvc"].append(structural.get("nvc", 0.0))
            structural_vals["vla"].append(structural.get("vla", 0.0))
            structural_vals["soed"].append(structural.get("soed", 0.0))
            gt_order = structural.get("gt_order", [])
            pred_order = structural.get("pred_order", [])
            if gt_order and pred_order:
                gt_adj = set(tuple(gt_order[i : i + 2]) for i in range(len(gt_order) - 1))
                pred_adj = set(tuple(pred_order[i : i + 2]) for i in range(len(pred_order) - 1))
                union = gt_adj | pred_adj
                inter = gt_adj & pred_adj
                strict_nvc = len(inter) / len(union) if union else 0.0
                structural_vals["strict_nvc"].append(strict_nvc)

    def _mean_or_zero(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    summary = {
        "num_cases": len(results),
        "mean_dice": mean_dice,
        "median_dice": median_dice,
        "std_dice": std_dice,
        "mean_iou": mean_iou,
        "median_iou": median_iou,
        "std_iou": std_iou,
        "structural_available": structural_available,
        "structural_mean_nvc": _mean_or_zero(structural_vals["nvc"]),
        "structural_mean_vla": _mean_or_zero(structural_vals["vla"]),
        "structural_mean_soed": _mean_or_zero(structural_vals["soed"]),
        "structural_mean_strict_nvc": _mean_or_zero(structural_vals["strict_nvc"]),
        "extra_pred_case_rate": float(len(extra_pred_cases) / len(results)) if results else 0.0,
        "extra_pred_rate_mean": float(np.mean(extra_pred_rate_vals)) if results else 0.0,
        "extra_pred_cases": extra_pred_cases,
        "extra_pred_total": extra_pred_total,
    }

    output_path = output_path or (results_dir / "summary.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Update summary.json from per-case metrics JSON.")
    parser.add_argument("--results_dir", required=True, help="Directory with per-case JSON files")
    parser.add_argument("--out", default=None, help="Output summary.json path")
    args = parser.parse_args()

    update_summary(Path(args.results_dir), Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()


