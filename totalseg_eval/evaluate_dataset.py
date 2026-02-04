#!/usr/bin/env python3
"""
Evaluate TotalSegmentator predictions for a dataset split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from totalseg_eval.evaluate_case import evaluate_case


def _find_gt_mask(derivatives_root: Path, subject_id: str) -> Path | None:
    subj_dir = derivatives_root / subject_id
    if not subj_dir.exists():
        return None
    candidates = sorted(subj_dir.glob("*_seg-vert_msk.nii.gz"))
    return candidates[0] if candidates else None


def _resolve_pred_input(pred_path: Path) -> Path | None:
    if pred_path.is_dir():
        body_file = pred_path / "vertebrae_body.nii.gz"
        if body_file.exists():
            return body_file
        has_verts = any(pred_path.glob("vertebrae_*.nii.gz"))
        if has_verts:
            return pred_path
        return None
    return pred_path if pred_path.exists() else None


def _eval_one(args: tuple) -> tuple[str, dict | None, str | None]:
    subject_id, pred_root, derivatives_root, output_dir, resample_gt, use_torch, device = args
    pred_path = Path(pred_root) / subject_id
    gt_path = _find_gt_mask(Path(derivatives_root), subject_id)
    if gt_path is None:
        return subject_id, None, "missing_gt"

    pred_input = _resolve_pred_input(pred_path)
    if pred_input is None:
        return subject_id, None, "missing_pred"

    out_path = Path(output_dir) / f"{subject_id}.json"
    res = evaluate_case(
        pred_path=pred_input,
        gt_path=gt_path,
        output_path=out_path,
        resample_gt=resample_gt,
        use_torch=use_torch,
        device=device,
    )
    return subject_id, res, None


def evaluate_dataset(
    pred_root: Path,
    derivatives_root: Path,
    output_dir: Path,
    resample_gt: bool = False,
    use_torch: bool = False,
    device: str = "cuda",
    num_workers: int = 0,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    missing = {"missing_gt": [], "missing_pred": []}

    subjects = sorted([p.name for p in pred_root.iterdir() if p.is_dir()])
    if use_torch and device.startswith("cuda") and num_workers > 0:
        raise ValueError("Multiprocessing with CUDA metrics is not supported. Use num_workers=0 or device=cpu.")

    if num_workers > 0:
        import multiprocessing as mp

        tasks = [
            (sid, pred_root, derivatives_root, output_dir, resample_gt, use_torch, device)
            for sid in subjects
        ]
        from tqdm import tqdm

        with mp.Pool(processes=num_workers) as pool:
            for sid, res, miss in tqdm(
                pool.imap_unordered(_eval_one, tasks),
                total=len(tasks),
                desc="Evaluating",
            ):
                if miss:
                    missing[miss].append(sid)
                elif res is not None:
                    results.append(res)
    else:
        from tqdm import tqdm

        for subject_id in tqdm(subjects, desc="Evaluating"):
            sid, res, miss = _eval_one(
                (subject_id, pred_root, derivatives_root, output_dir, resample_gt, use_torch, device)
            )
            if miss:
                missing[miss].append(sid)
            elif res is not None:
                results.append(res)

    dice_vals = np.array([r["overall"]["mean_dice"] for r in results], dtype=float)
    iou_vals = np.array([r["overall"]["mean_iou"] for r in results], dtype=float)

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
            extra_pred_cases.append({"subject": Path(r["pred_path"]).name, "extra_labels": extra})
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
        "missing": missing,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dataset predictions.")
    parser.add_argument("--pred_root", required=True, help="Predictions root (per-subject dirs)")
    parser.add_argument("--derivatives_root", required=True, help="Derivatives root with GT masks")
    parser.add_argument("--out_dir", required=True, help="Output directory for metrics")
    parser.add_argument("--resample_gt", action="store_true", help="Resample GT to pred space")
    parser.add_argument("--use_torch", action="store_true", help="Use torch for metric computation")
    parser.add_argument("--device", default="cuda", help="Torch device (e.g., cuda, cpu)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes")
    args = parser.parse_args()

    evaluate_dataset(
        pred_root=Path(args.pred_root),
        derivatives_root=Path(args.derivatives_root),
        output_dir=Path(args.out_dir),
        resample_gt=args.resample_gt,
        use_torch=args.use_torch,
        device=args.device,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()


