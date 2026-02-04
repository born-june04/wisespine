#!/usr/bin/env python3
"""
Phase 0: minimal baselines for the ablation plan.

What it does (single subject for now):
  - Load GT multi-label vertebra segmentation (VerSe labels 1..25)
  - Load TotalSeg predictions (per-vertebra NIfTI) and merge into one label map
  - Compute mask-level metrics:
      * per-label Dice / IoU
      * overall mean Dice / mean IoU across labels present in GT
  - Apply simple random corruptions to the TS label map (baseline augmentation)
      * erosion / dilation on selected labels
    and re-evaluate metrics.

Outputs:
  spine-rl-sim/ablation_outputs/<date>/phase0_<subject>/*.json

Example:
  python spine-rl-sim/ablation/run_phase0_baselines.py \
    --subject sub-verse563 \
    --out_dir spine-rl-sim/ablation_outputs/2026-01-28/phase0_sub-verse563
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import nibabel as nib
except Exception as e:  # pragma: no cover
    raise RuntimeError("nibabel is required (pip install nibabel)") from e

try:
    from scipy import ndimage as ndi
except Exception as e:  # pragma: no cover
    raise RuntimeError("scipy is required (pip install scipy)") from e


REPO_ROOT = Path(__file__).resolve().parents[2]


def _vertebra_label_map() -> Dict[str, int]:
    labels: Dict[str, int] = {}
    for i in range(1, 8):
        labels[f"C{i}"] = i
    for i in range(1, 13):
        labels[f"T{i}"] = 7 + i
    for i in range(1, 6):
        labels[f"L{i}"] = 19 + i
    labels["S1"] = 25
    return labels


def gt_path_for(subject: str) -> Path:
    p = REPO_ROOT / "VerSe" / "dataset-03test" / "derivatives" / subject / f"{subject}_dir-iso_seg-vert_msk.nii.gz"
    if not p.exists():
        raise FileNotFoundError(f"GT segmentation not found: {p}")
    return p


def totalseg_dir_for(subject: str) -> Path:
    p = REPO_ROOT / "totalseg_eval" / "predictions_total" / subject
    if not p.exists():
        raise FileNotFoundError(f"TotalSeg predictions dir not found: {p}")
    return p


def load_nifti_u16(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.uint16)
    return data, img


def merge_totalseg_dir(pred_dir: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
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
        lab = label_map.get(name)
        if lab is None:
            continue
        mask = nib.load(str(path)).get_fdata() > 0
        merged[mask] = lab
    merged_img = nib.Nifti1Image(merged, base_img.affine, base_img.header)
    return merged, merged_img


def merge_totalseg_dir_into(gt_shape: Tuple[int, int, int], pred_dir: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    Memory-safe merge:
      - Allocate label map with GT shape
      - Load each vertebrae_*.nii.gz one at a time and write into the map
    Assumes shapes match (we validate).
    """
    label_map = _vertebra_label_map()
    candidates = sorted(pred_dir.glob("vertebrae_*.nii.gz"))
    if not candidates:
        raise FileNotFoundError(f"No vertebrae_*.nii.gz in {pred_dir}")
    base_img = nib.load(str(candidates[0]))
    if tuple(base_img.shape) != tuple(gt_shape):
        raise ValueError(f"Shape mismatch: GT {gt_shape} vs pred {base_img.shape}")
    merged = np.zeros(gt_shape, dtype=np.uint16)
    for path in candidates:
        name = path.name.replace("vertebrae_", "")
        if name.endswith(".nii.gz"):
            name = name[: -len(".nii.gz")]
        elif name.endswith(".nii"):
            name = name[: -len(".nii")]
        lab = label_map.get(name)
        if lab is None:
            continue
        mask = nib.load(str(path)).get_fdata() > 0
        merged[mask] = lab
    merged_img = nib.Nifti1Image(merged, base_img.affine, base_img.header)
    return merged, merged_img


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred = pred.astype(bool, copy=False)
    gt = gt.astype(bool, copy=False)
    inter = np.logical_and(pred, gt).sum()
    ps = pred.sum()
    gs = gt.sum()
    dice = (2.0 * inter + 1e-6) / (ps + gs + 1e-6)
    union = np.logical_or(pred, gt).sum()
    iou = (inter + 1e-6) / (union + 1e-6)
    return float(dice), float(iou)


@dataclass
class MaskMetrics:
    mode: str
    labels: List[int]
    per_label: Dict[int, Dict[str, float]]
    mean_dice: float
    mean_iou: float


def compute_multilabel_metrics(pred: np.ndarray, gt: np.ndarray) -> MaskMetrics:
    labels = sorted(set(np.unique(gt).tolist()) - {0})
    per: Dict[int, Dict[str, float]] = {}
    ds: List[float] = []
    is_: List[float] = []
    for lab in labels:
        gt_bin = gt == lab
        if gt_bin.sum() == 0:
            continue
        pred_bin = pred == lab
        d, i = dice_iou(pred_bin, gt_bin)
        per[int(lab)] = {"dice": d, "iou": i, "gt_voxels": float(gt_bin.sum()), "pred_voxels": float(pred_bin.sum())}
        ds.append(d)
        is_.append(i)
    return MaskMetrics(
        mode="multilabel",
        labels=[int(l) for l in labels],
        per_label=per,
        mean_dice=float(np.mean(ds)) if ds else 0.0,
        mean_iou=float(np.mean(is_)) if is_ else 0.0,
    )


def corrupt_labels_morphology(
    label_map: np.ndarray,
    labels: List[int],
    op: str,
    radius: int,
    rng: np.random.Generator,
    p_apply: float = 0.5,
) -> np.ndarray:
    """
    Random baseline corruption: per-label binary erosion/dilation with a small radius.
    Keeps label identities (no swapping) and doesn't touch background.
    """
    if radius <= 0:
        return label_map
    struct = ndi.generate_binary_structure(3, 1)
    struct = ndi.iterate_structure(struct, radius)
    out = label_map.copy()

    # Memory-safe: operate on tight per-label bounding boxes rather than full volume.
    for lab in labels:
        if rng.random() > p_apply:
            continue
        coords = np.argwhere(label_map == lab)
        if coords.size == 0:
            continue
        x0, y0, z0 = coords.min(axis=0).tolist()
        x1, y1, z1 = coords.max(axis=0).tolist()
        # Expand bbox by radius to give morphology room.
        x0 = max(0, x0 - radius)
        y0 = max(0, y0 - radius)
        z0 = max(0, z0 - radius)
        x1 = min(label_map.shape[0] - 1, x1 + radius)
        y1 = min(label_map.shape[1] - 1, y1 + radius)
        z1 = min(label_map.shape[2] - 1, z1 + radius)

        sub = label_map[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1]
        m = (sub == lab)
        if op == "erode":
            m2 = ndi.binary_erosion(m, structure=struct)
        elif op == "dilate":
            m2 = ndi.binary_dilation(m, structure=struct)
        else:
            raise ValueError(f"Unknown op: {op}")

        # Update only inside bbox region to avoid scanning/allocating full volume.
        sub_out = out[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1]
        sub_out[sub_out == lab] = 0
        sub_out[m2] = lab
        out[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] = sub_out

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="sub-verse563")
    ap.add_argument("--out_dir", default=None, help="Default: spine-rl-sim/ablation_outputs/<today>/phase0_<subject>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corrupt_op", choices=["erode", "dilate"], default="erode")
    ap.add_argument("--corrupt_radius", type=int, default=1)
    ap.add_argument("--corrupt_p_apply", type=float, default=0.5)
    args = ap.parse_args()

    subject = args.subject
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir) if args.out_dir else (REPO_ROOT / "spine-rl-sim" / "ablation_outputs" / str(date.today()) / f"phase0_{subject}")
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = gt_path_for(subject)
    pred_dir = totalseg_dir_for(subject)

    gt, _gt_img = load_nifti_u16(gt_path)
    # Avoid holding multiple large arrays at once: allocate pred based on GT shape.
    pred, _pred_img = merge_totalseg_dir_into(gt.shape, pred_dir)

    # Baseline: clean
    clean_metrics = compute_multilabel_metrics(pred, gt)

    # Corruption baseline on TS label map
    labels = clean_metrics.labels
    pred_corrupt = corrupt_labels_morphology(
        pred,
        labels=labels,
        op=args.corrupt_op,
        radius=args.corrupt_radius,
        rng=rng,
        p_apply=args.corrupt_p_apply,
    )
    corrupt_metrics = compute_multilabel_metrics(pred_corrupt, gt)

    payload = {
        "subject": subject,
        "gt_path": str(gt_path),
        "pred_dir": str(pred_dir),
        "seed": args.seed,
        "clean": asdict(clean_metrics),
        "corrupt": {
            "op": args.corrupt_op,
            "radius": args.corrupt_radius,
            "p_apply": args.corrupt_p_apply,
            "metrics": asdict(corrupt_metrics),
        },
    }

    out_path = out_dir / f"phase0_{subject}_mask_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")
    print("Clean mean_dice:", clean_metrics.mean_dice, "mean_iou:", clean_metrics.mean_iou)
    print("Corrupt mean_dice:", corrupt_metrics.mean_dice, "mean_iou:", corrupt_metrics.mean_iou)


if __name__ == "__main__":
    main()


