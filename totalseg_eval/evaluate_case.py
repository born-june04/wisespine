#!/usr/bin/env python3
"""
Evaluate a single TotalSegmentator prediction against VerSe GT.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from totalseg_eval.metrics.ordering_metrics import (
    neighbor_vertebra_consistency,
    spine_ordering_edit_distance,
    vertebra_level_accuracy,
)
from totalseg_eval.metrics.segmentation_metrics import dice_score, iou_score, mean_iou


def _load_nifti(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.uint16)
    return data, img


def _resample_to(gt_img: nib.Nifti1Image, pred_img: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.processing.resample_from_to(gt_img, pred_img, order=0)


def _vertebra_label_map() -> dict[str, int]:
    labels = {}
    for i in range(1, 8):
        labels[f"C{i}"] = i
    for i in range(1, 13):
        labels[f"T{i}"] = 7 + i
    for i in range(1, 6):
        labels[f"L{i}"] = 19 + i
    labels["S1"] = 25
    return labels


def _merge_vertebrae_dir(pred_dir: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    label_map = _vertebra_label_map()
    candidates = sorted(pred_dir.glob("vertebrae_*.nii.gz"))
    if not candidates:
        raise FileNotFoundError(f"No vertebrae_*.nii.gz in {pred_dir}")

    base_img = nib.load(str(candidates[0]))
    merged = np.zeros(base_img.shape, dtype=np.uint16)

    for path in candidates:
        name = path.name.replace("vertebrae_", "")
        if name.endswith(".nii.gz"):
            name = name[: -len(".nii.gz")]
        elif name.endswith(".nii"):
            name = name[: -len(".nii")]
        label = label_map.get(name)
        if label is None:
            continue
        mask = nib.load(str(path)).get_fdata() > 0
        merged[mask] = label

    merged_img = nib.Nifti1Image(merged, base_img.affine, base_img.header)
    return merged, merged_img


def _dice_iou(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    use_torch: bool,
    device: str,
) -> tuple[float, float]:
    if not use_torch:
        return dice_score(pred_bin, gt_bin), iou_score(pred_bin, gt_bin)

    import torch

    pred_t = torch.from_numpy(pred_bin.astype(np.float32, copy=False)).to(device)
    gt_t = torch.from_numpy(gt_bin.astype(np.float32, copy=False)).to(device)
    intersection = (pred_t * gt_t).sum()
    dice = (2.0 * intersection + 1e-6) / (pred_t.sum() + gt_t.sum() + 1e-6)
    union = (pred_t + gt_t).clamp(0.0, 1.0).sum()
    iou = (intersection + 1e-6) / (union + 1e-6)
    return float(dice.item()), float(iou.item())


def _label_centroids(mask: np.ndarray, labels: list[int]) -> dict[int, np.ndarray]:
    centroids = {}
    for label in labels:
        coords = np.argwhere(mask == label)
        if coords.size == 0:
            continue
        centroids[int(label)] = coords.mean(axis=0)
    return centroids


def _ordering_axis(centroids: dict[int, np.ndarray]) -> int:
    if not centroids:
        return 0
    coords = np.stack(list(centroids.values()), axis=0)
    variances = coords.var(axis=0)
    return int(np.argmax(variances))


def _order_labels_by_axis(centroids: dict[int, np.ndarray], axis: int) -> list[int]:
    items = [(label, c[axis]) for label, c in centroids.items()]
    items.sort(key=lambda x: x[1])
    return [label for label, _ in items]


def _vla_from_overlap(
    pred: np.ndarray,
    gt: np.ndarray,
    labels: list[int],
    pred_labels: list[int],
    iou_threshold: float = 0.1,
) -> float:
    if not labels:
        return 0.0
    if not pred_labels:
        return 0.0
    correct = 0
    total = 0
    for label in labels:
        gt_bin = gt == label
        if gt_bin.sum() == 0:
            continue
        total += 1
        best_iou = 0.0
        best_label = None
        for pl in pred_labels:
            pred_bin = pred == pl
            if pred_bin.sum() == 0:
                continue
            inter = np.logical_and(pred_bin, gt_bin).sum()
            union = np.logical_or(pred_bin, gt_bin).sum()
            iou = inter / (union + 1e-6)
            if iou > best_iou:
                best_iou = iou
                best_label = pl
        if best_label == label and best_iou >= iou_threshold:
            correct += 1
    return correct / total if total > 0 else 0.0


def evaluate_case(
    pred_path: Path,
    gt_path: Path,
    output_path: Path,
    resample_gt: bool = False,
    use_torch: bool = False,
    device: str = "cuda",
) -> dict:
    if pred_path.is_dir():
        pred, pred_img = _merge_vertebrae_dir(pred_path)
    else:
        pred, pred_img = _load_nifti(pred_path)
    gt, gt_img = _load_nifti(gt_path)

    if pred.shape != gt.shape:
        if not resample_gt:
            raise ValueError(
                f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}. "
                "Use --resample_gt to align."
            )
        gt_img = _resample_to(gt_img, pred_img)
        gt = gt_img.get_fdata().astype(np.uint16)

    labels = sorted(set(np.unique(gt)) - {0})
    pred_labels = sorted(set(np.unique(pred)) - {0})
    is_binary_pred = len(pred_labels) <= 1

    per_label = {}
    if is_binary_pred:
        pred_bin = (pred > 0).astype(np.uint8)
        gt_bin = (gt > 0).astype(np.uint8)
        binary_dice, binary_iou = _dice_iou(pred_bin, gt_bin, use_torch, device)

        for label in labels:
            gt_label = (gt == label).astype(np.uint8)
            if gt_label.sum() == 0:
                continue
            dice, iou = _dice_iou(pred_bin, gt_label, use_torch, device)
            per_label[int(label)] = {
                "dice": dice,
                "iou": iou,
            }

        overall = {
            "mode": "binary_pred",
            "binary_dice": binary_dice,
            "binary_iou": binary_iou,
            "mean_iou": binary_iou,
            "mean_dice": binary_dice,
        }
    else:
        for label in labels:
            pred_bin = (pred == label).astype(np.uint8)
            gt_bin = (gt == label).astype(np.uint8)
            if gt_bin.sum() == 0:
                continue
            dice, iou = _dice_iou(pred_bin, gt_bin, use_torch, device)
            per_label[int(label)] = {
                "dice": dice,
                "iou": iou,
            }

        overall = {
            "mode": "multiclass_pred",
            "mean_iou": float(np.mean([v["iou"] for v in per_label.values()])) if per_label else 0.0,
            "mean_dice": float(np.mean([v["dice"] for v in per_label.values()])) if per_label else 0.0,
        }

    structural = {
        "available": False,
        "reason": "prediction is binary or unlabeled",
    }
    if not is_binary_pred:
        gt_centroids = _label_centroids(gt, labels)
        pred_centroids = _label_centroids(pred, pred_labels)
        axis = _ordering_axis(gt_centroids)
        gt_order = _order_labels_by_axis(gt_centroids, axis)
        pred_order = _order_labels_by_axis(pred_centroids, axis)
        structural = {
            "available": True,
            "axis": axis,
            "gt_order": gt_order,
            "pred_order": pred_order,
            "nvc": neighbor_vertebra_consistency(gt_order, pred_order),
            "soed": spine_ordering_edit_distance(gt_order, pred_order),
            "vla": _vla_from_overlap(pred, gt, labels, pred_labels, iou_threshold=0.1),
            "vla_iou_threshold": 0.1,
        }

    results = {
        "pred_path": str(pred_path),
        "gt_path": str(gt_path),
        "labels": [int(l) for l in labels],
        "pred_labels": [int(l) for l in pred_labels],
        "overall": overall,
        "per_label": per_label,
        "structural": structural,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a TotalSegmentator case.")
    parser.add_argument("--pred", required=True, help="Path to predicted mask NIfTI")
    parser.add_argument("--gt", required=True, help="Path to GT mask NIfTI")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--resample_gt", action="store_true", help="Resample GT to pred space")
    parser.add_argument("--use_torch", action="store_true", help="Use torch for metric computation")
    parser.add_argument("--device", default="cuda", help="Torch device (e.g., cuda, cpu)")
    args = parser.parse_args()

    evaluate_case(
        pred_path=Path(args.pred),
        gt_path=Path(args.gt),
        output_path=Path(args.out),
        resample_gt=args.resample_gt,
        use_torch=args.use_torch,
        device=args.device,
    )


if __name__ == "__main__":
    main()


