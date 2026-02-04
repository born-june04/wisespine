"""
Segmentation metrics (voxel-level).
"""

from __future__ import annotations

import numpy as np


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Dice score for binary masks.
    """
    pred = pred.astype(np.float32).reshape(-1)
    target = target.astype(np.float32).reshape(-1)
    intersection = np.sum(pred * target)
    return float((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    IoU score for binary masks.
    """
    pred = pred.astype(np.float32).reshape(-1)
    target = target.astype(np.float32).reshape(-1)
    intersection = np.sum(pred * target)
    union = np.sum(np.clip(pred + target, 0.0, 1.0))
    return float((intersection + smooth) / (union + smooth))


def mean_iou(
    pred: np.ndarray,
    target: np.ndarray,
    labels: list[int] | None = None,
    smooth: float = 1e-6,
) -> float:
    """
    Mean IoU across label set (integer label masks).
    """
    if labels is None:
        labels = sorted(set(np.unique(target)) - {0})
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        pred_bin = (pred == label).astype(np.uint8)
        target_bin = (target == label).astype(np.uint8)
        if target_bin.sum() == 0:
            continue
        scores.append(iou_score(pred_bin, target_bin, smooth=smooth))
    return float(np.mean(scores)) if scores else 0.0


