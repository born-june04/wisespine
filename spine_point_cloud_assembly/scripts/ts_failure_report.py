#!/usr/bin/env python3
"""
Generate TS (TotalSegmentator) structural failure reports from `totalseg_eval/` outputs.

Outputs:
  1) subject-level report CSV: ts_failure_report.csv
  2) per-instance report CSV:  failure_types.csv

Key failure signals (B/C alignment):
  - extra_vs_gt: predicted vertebra levels not present in GT label set (coverage mismatch; NOT necessarily spurious)
  - transition violations: validTransition breaks in predicted ordering
  - axis outliers: predicted instances whose centroids are far from the main spine axis
  - non-bone: predicted instances whose centroid neighborhood HU is inconsistent with bone

This script intentionally stays "rule-based" and lightweight (no training).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

# Ensure imports work when executed as a script: `python spine_point_cloud_assembly/scripts/...py`
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spine_point_cloud_assembly.utils.anatomy_transitions import (
    duplicates,
    nvc_transition,
    strict_nvc_transition,
    transition_violations,
)


def _parse_pred_label_id(path: Path) -> int | None:
    """
    predictions_total/<subject>/vertebrae_<NAME>.nii.gz -> label id (1..25)
    """
    name = path.name
    if not name.startswith("vertebrae_"):
        return None
    name = name[len("vertebrae_") :]
    if name.endswith(".nii.gz"):
        name = name[: -len(".nii.gz")]
    elif name.endswith(".nii"):
        name = name[: -len(".nii")]

    # NAME is like C1, T12, L5, S1
    if name.startswith(("C", "T", "L")) and len(name) >= 2:
        try:
            prefix = name[0]
            num = int(name[1:])
        except Exception:
            return None
        if prefix == "C" and 1 <= num <= 7:
            return num
        if prefix == "T" and 1 <= num <= 12:
            return 7 + num
        if prefix == "L" and 1 <= num <= 5:
            return 19 + num
    if name == "S1":
        return 25
    return None


def _centroid_and_volume_streaming(mask_path: Path) -> tuple[np.ndarray | None, int]:
    """
    Compute centroid (x,y,z) and volume (#vox) without loading the full 3D array into RAM.
    Uses nibabel proxy and iterates over z-slices.
    """
    import nibabel as nib

    img = nib.as_closest_canonical(nib.load(str(mask_path)))
    dataobj = img.dataobj
    sx, sy, sz = img.shape

    total = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_z = 0.0

    # Iterate over z (S) slices
    for z in range(sz):
        sl = np.asanyarray(dataobj[:, :, z])
        m = sl > 0
        if not np.any(m):
            continue
        ys, xs = np.nonzero(m)  # NOTE: returns (row=y, col=x) for 2D
        n = int(xs.size)
        total += n
        sum_x += float(xs.sum())
        sum_y += float(ys.sum())
        sum_z += float(z * n)

    if total == 0:
        return None, 0
    return np.array([sum_x / total, sum_y / total, sum_z / total], dtype=np.float32), total


def _default_ct_path(project_root: Path, subject_id: str) -> Path:
    raw_dir = project_root / "VerSe" / "dataset-03test" / "rawdata" / subject_id
    cts = sorted(raw_dir.glob(f"{subject_id}_*_ct.nii.gz"))
    if not cts:
        raise FileNotFoundError(f"Cannot find CT in {raw_dir} (expected {subject_id}_*_ct.nii.gz)")
    return cts[0]


def _sample_ct_patch_median_hu(
    *,
    ct_img_c,
    ct_affine_inv: np.ndarray,
    world_xyz: np.ndarray,
    radius: int = 2,
) -> float:
    """
    Sample a small cubic patch around world_xyz in CT voxel space and return median HU.
    Heuristic: vertebra masks should lie on high-HU bone regions.
    """
    v = ct_affine_inv @ np.array([float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]), 1.0], dtype=np.float64)
    x, y, z = int(round(float(v[0]))), int(round(float(v[1]))), int(round(float(v[2])))
    sx, sy, sz = ct_img_c.shape
    x0 = max(0, x - radius)
    x1 = min(sx, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(sy, y + radius + 1)
    z0 = max(0, z - radius)
    z1 = min(sz, z + radius + 1)
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return float("nan")
    patch = np.asanyarray(ct_img_c.dataobj[x0:x1, y0:y1, z0:z1], dtype=np.float32)
    if patch.size == 0:
        return float("nan")
    return float(np.median(patch))


def _sample_mask_hu_stats(
    *,
    mask_img_c,
    ct_img_c,
    ct_affine_inv: np.ndarray,
    max_samples: int = 4000,
    per_slice: int = 40,
    seed: int = 0,
) -> dict[str, float]:
    """
    Sample HU values inside a binary mask, robustly and memory-safely.

    We DO NOT use centroid HU because vertebra centroid often lies in trabecular/marrow (low HU).
    Instead we sample mask voxels and use high quantiles (p90/p95) as evidence of bone presence.
    """
    rng = np.random.default_rng(seed)
    obj = mask_img_c.dataobj
    sz = mask_img_c.shape[2]
    vals: list[float] = []

    for z in range(sz):
        sl = np.asanyarray(obj[:, :, z])
        ys, xs = np.nonzero(sl > 0)
        if xs.size == 0:
            continue
        take = int(min(per_slice, xs.size))
        idx = rng.choice(xs.size, size=take, replace=False)
        for k in idx:
            vx = float(xs[k])
            vy = float(ys[k])
            vz = float(z)
            world = mask_img_c.affine @ np.array([vx, vy, vz, 1.0], dtype=np.float64)
            cv = ct_affine_inv @ world
            cx = int(round(float(cv[0])))
            cy = int(round(float(cv[1])))
            cz = int(round(float(cv[2])))
            if 0 <= cx < ct_img_c.shape[0] and 0 <= cy < ct_img_c.shape[1] and 0 <= cz < ct_img_c.shape[2]:
                vals.append(float(ct_img_c.dataobj[cx, cy, cz]))
        if len(vals) >= max_samples:
            break

    if not vals:
        return {"hu_p50": float("nan"), "hu_p90": float("nan"), "hu_p95": float("nan"), "hu_n": 0.0}
    v = np.asarray(vals, dtype=np.float32)
    return {
        "hu_p50": float(np.percentile(v, 50)),
        "hu_p90": float(np.percentile(v, 90)),
        "hu_p95": float(np.percentile(v, 95)),
        "hu_n": float(v.size),
    }


def _axis_outliers(
    centroids: dict[int, np.ndarray],
    *,
    mad_k: float = 4.0,
    min_instances: int = 5,
) -> dict[int, float]:
    """
    Fit PCA axis to centroids and return outlier label->distance for those far from axis.
    """
    if len(centroids) < min_instances:
        return {}
    labels = sorted(centroids.keys())
    C = np.stack([centroids[l] for l in labels], axis=0).astype(np.float32)
    mean = C.mean(axis=0, keepdims=True)
    X = C - mean
    cov = (X.T @ X) / max(1, (X.shape[0] - 1))
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, np.argmax(eigvals)]  # principal axis
    v = v / (np.linalg.norm(v) + 1e-8)
    proj = (X @ v.reshape(3, 1)) * v.reshape(1, 3)
    dist = np.linalg.norm(X - proj, axis=1)
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med))) + 1e-8
    thr = med + mad_k * mad
    out: dict[int, float] = {}
    for lab, d in zip(labels, dist):
        if float(d) > thr:
            out[int(lab)] = float(d)
    return out


def _safe_get(d: dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr"),
        help="Repo root that contains VerSe/ and totalseg_eval/",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/totalseg_eval/metrics_results_total"),
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/totalseg_eval/predictions_total"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/spine_point_cloud_assembly/outputs/evaluation"),
    )
    parser.add_argument("--max-subjects", type=int, default=0, help="0 = all")
    parser.add_argument("--compute-centroids", action="store_true", help="enable axis-outlier detection (slower)")
    parser.add_argument(
        "--compute-hu",
        action="store_true",
        help="enable CT HU-based non-bone(spurious) detection using HU quantiles inside each predicted mask",
    )
    parser.add_argument("--hu-threshold", type=float, default=120.0, help="HU p90 below this => non-bone")
    args = parser.parse_args()

    metrics_files = sorted([p for p in args.metrics_dir.glob("*.json") if p.name != "summary.json"])
    if args.max_subjects and args.max_subjects > 0:
        metrics_files = metrics_files[: args.max_subjects]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_subject_csv = out_dir / "ts_failure_report.csv"
    out_instance_csv = out_dir / "failure_types.csv"

    subject_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []

    for mf in metrics_files:
        j = json.loads(mf.read_text())
        subject_id = mf.stem

        gt_labels = [int(x) for x in j.get("labels", [])]
        pred_labels = [int(x) for x in j.get("pred_labels", [])]
        gt_set = set(gt_labels)
        pred_set = set(pred_labels)

        # NOTE:
        # pred_set - gt_set is "extra vs GT labels" which often reflects GT/FOV coverage mismatch.
        # Do NOT treat this alone as "spurious non-spine".
        extra_vs_gt_labels = sorted(pred_set - gt_set)
        missing_vs_pred_labels = sorted(gt_set - pred_set)

        # Ordering sequences if available
        gt_order = [int(x) for x in _safe_get(j, "structural", "gt_order", default=[]) or []]
        pred_order = [int(x) for x in _safe_get(j, "structural", "pred_order", default=[]) or []]

        # Transition metrics (ours, based on validTransition)
        pred_nvc = nvc_transition(pred_order, consider_direction=False) if pred_order else 0.0
        pred_strict = strict_nvc_transition(pred_order, consider_direction=False) if pred_order else 0.0
        pred_viol = transition_violations(pred_order, consider_direction=False) if pred_order else []

        # Duplicates in order list (should be empty for predictions_total, but kept for completeness)
        dup = duplicates(pred_order) if pred_order else {}

        # TotalSeg's own reported metrics (if present)
        mean_dice = float(_safe_get(j, "overall", "mean_dice", default=float("nan")))
        mean_iou = float(_safe_get(j, "overall", "mean_iou", default=float("nan")))
        soed_ts = float(_safe_get(j, "structural", "soed", default=float("nan")))
        nvc_ts = float(_safe_get(j, "structural", "nvc", default=float("nan")))

        n_gt = len(gt_labels)
        n_pred = len(pred_labels)
        extra_rate = float(max(0, n_pred - n_gt) / max(1, n_gt))

        # Optional centroid-based spurious detection
        outlier_labels: dict[int, float] = {}
        centroid_map: dict[int, np.ndarray] = {}
        volume_map: dict[int, int] = {}
        hu_p50_map: dict[int, float] = {}
        hu_p90_map: dict[int, float] = {}
        hu_p95_map: dict[int, float] = {}
        hu_n_map: dict[int, float] = {}
        non_bone_labels: set[int] = set()

        compute_centroids = bool(args.compute_centroids)

        if compute_centroids:
            pred_dir = args.pred_root / subject_id
            for mp in sorted(pred_dir.glob("vertebrae_*.nii.gz")):
                lab = _parse_pred_label_id(mp)
                if lab is None:
                    continue
                c, vol = _centroid_and_volume_streaming(mp)
                if c is None:
                    continue
                centroid_map[int(lab)] = c
                volume_map[int(lab)] = int(vol)
            outlier_labels = _axis_outliers(centroid_map, mad_k=4.0)

        # HU-based non-bone check does NOT require centroids; it samples intensities inside each mask.
        if args.compute_hu:
            import nibabel as nib

            pred_dir = args.pred_root / subject_id
            ct_path = _default_ct_path(args.project_root, subject_id)
            ct_img_c = nib.as_closest_canonical(nib.load(str(ct_path)))
            ct_affine_inv = np.linalg.inv(ct_img_c.affine)

            for mp in sorted(pred_dir.glob("vertebrae_*.nii.gz")):
                lab = _parse_pred_label_id(mp)
                if lab is None:
                    continue
                m_img_c = nib.as_closest_canonical(nib.load(str(mp)))
                stats = _sample_mask_hu_stats(
                    mask_img_c=m_img_c,
                    ct_img_c=ct_img_c,
                    ct_affine_inv=ct_affine_inv,
                    max_samples=4000,
                    per_slice=40,
                    seed=0,
                )
                hu_p50_map[int(lab)] = float(stats["hu_p50"])
                hu_p90_map[int(lab)] = float(stats["hu_p90"])
                hu_p95_map[int(lab)] = float(stats["hu_p95"])
                hu_n_map[int(lab)] = float(stats["hu_n"])
                if np.isfinite(stats["hu_p90"]) and float(stats["hu_p90"]) < float(args.hu_threshold):
                    non_bone_labels.add(int(lab))

        # Per-instance rows
        for lab in sorted(pred_set):
            row = {
                "subject_id": subject_id,
                "label": int(lab),
                "in_gt": int(lab in gt_set),
                "is_extra_vs_gt": int(lab in extra_vs_gt_labels),
                "is_axis_outlier": int(lab in outlier_labels),
                "axis_outlier_dist": float(outlier_labels.get(int(lab), float("nan"))),
                "hu_p50": float(hu_p50_map.get(int(lab), float("nan"))),
                "hu_p90": float(hu_p90_map.get(int(lab), float("nan"))),
                "hu_p95": float(hu_p95_map.get(int(lab), float("nan"))),
                "hu_n": float(hu_n_map.get(int(lab), float("nan"))),
                "is_non_bone": int(int(lab) in non_bone_labels),
                # Spurious (non-spine) proxy should be evidence-based (HU / geometry), not GT coverage-based.
                "is_spurious_non_spine_proxy": int((int(lab) in outlier_labels) or (int(lab) in non_bone_labels)),
                "volume_vox": int(volume_map.get(int(lab), 0)),
                "centroid_x": float(centroid_map.get(int(lab), np.array([np.nan, np.nan, np.nan]))[0])
                if compute_centroids
                else float("nan"),
                "centroid_y": float(centroid_map.get(int(lab), np.array([np.nan, np.nan, np.nan]))[1])
                if compute_centroids
                else float("nan"),
                "centroid_z": float(centroid_map.get(int(lab), np.array([np.nan, np.nan, np.nan]))[2])
                if compute_centroids
                else float("nan"),
            }
            instance_rows.append(row)

        subject_rows.append(
            {
                "subject_id": subject_id,
                "mean_dice": mean_dice,
                "mean_iou": mean_iou,
                "n_gt": n_gt,
                "n_pred": n_pred,
                "extra_rate": extra_rate,
                "extra_vs_gt_count": len(extra_vs_gt_labels),
                "extra_vs_gt_labels": " ".join(map(str, extra_vs_gt_labels)),
                "missing_vs_pred_count": len(missing_vs_pred_labels),
                "missing_vs_pred_labels": " ".join(map(str, missing_vs_pred_labels)),
                "soed_totalseg": soed_ts,
                "nvc_totalseg": nvc_ts,
                "nvc_transition_undirected": float(pred_nvc),
                "strict_nvc_transition_undirected": float(pred_strict),
                "transition_violations_count": len(pred_viol),
                "transition_violations": json.dumps([asdict(v) for v in pred_viol]),
                "duplicate_labels_count": len(dup),
                "duplicate_labels": json.dumps(dup),
                "axis_outliers_count": len(outlier_labels),
                "axis_outlier_labels": " ".join(map(str, sorted(outlier_labels.keys()))),
                "non_bone_count": int(len(non_bone_labels)),
                "non_bone_labels": " ".join(map(str, sorted(non_bone_labels))),
                "spurious_non_spine_proxy_count": int(len(set(outlier_labels.keys()).union(non_bone_labels))),
                "gt_order": " ".join(map(str, gt_order)),
                "pred_order": " ".join(map(str, pred_order)),
            }
        )

    # Write CSVs
    if subject_rows:
        with out_subject_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(subject_rows[0].keys()))
            w.writeheader()
            w.writerows(subject_rows)

    if instance_rows:
        with out_instance_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(instance_rows[0].keys()))
            w.writeheader()
            w.writerows(instance_rows)

    print(f"✓ Wrote: {out_subject_csv}")
    print(f"✓ Wrote: {out_instance_csv}")


if __name__ == "__main__":
    main()


