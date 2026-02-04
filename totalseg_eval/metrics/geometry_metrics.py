"""
Geometry-aware metrics.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def inter_vertebra_relative_pose_error(
    gt_translations: np.ndarray,
    gt_rotations: np.ndarray | None,
    pred_translations: np.ndarray,
    pred_rotations: np.ndarray | None,
    *,
    order: Sequence[int] | None = None,
) -> dict:
    """
    Relative pose error between adjacent vertebrae.

    Args:
        gt_translations: (N, 3) GT translations.
        gt_rotations: (N, 3, 3) GT rotations or None.
        pred_translations: (N, 3) predicted translations.
        pred_rotations: (N, 3, 3) predicted rotations or None.
        order: Optional ordering of indices to evaluate adjacency.
    """
    if order is None:
        order = list(range(len(gt_translations)))
    if len(order) < 2:
        return {"translation_mean": 0.0, "rotation_mean_deg": 0.0}

    t_errors = []
    r_errors = []

    for i in range(len(order) - 1):
        a = order[i]
        b = order[i + 1]
        gt_dt = gt_translations[b] - gt_translations[a]
        pred_dt = pred_translations[b] - pred_translations[a]
        t_errors.append(np.linalg.norm(pred_dt - gt_dt))

        if gt_rotations is not None and pred_rotations is not None:
            gt_rel = gt_rotations[a].T @ gt_rotations[b]
            pred_rel = pred_rotations[a].T @ pred_rotations[b]
            r_errors.append(_geodesic_distance_deg(pred_rel, gt_rel))

    return {
        "translation_mean": float(np.mean(t_errors)) if t_errors else 0.0,
        "rotation_mean_deg": float(np.mean(r_errors)) if r_errors else 0.0,
    }


def spine_axis_smoothness_error(
    centroids: np.ndarray,
    *,
    order: Sequence[int] | None = None,
) -> float:
    """
    Variance of second derivative along ordered centroids.
    """
    if order is None:
        order = list(range(len(centroids)))
    if len(order) < 3:
        return 0.0
    ordered = centroids[np.array(order)]
    second_diff = ordered[2:] - 2 * ordered[1:-1] + ordered[:-2]
    curvature = np.linalg.norm(second_diff, axis=-1)
    return float(np.var(curvature))


def _geodesic_distance_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """
    Geodesic distance between rotation matrices in degrees.
    """
    cos_theta = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


