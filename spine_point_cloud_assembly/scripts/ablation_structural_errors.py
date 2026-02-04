#!/usr/bin/env python3
"""
Ablation: inject neighbor swaps and measure structural error sensitivity.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.assembly_data_loader import AssemblyDataset


def cobb_angle(points_3d: np.ndarray) -> float:
    if points_3d.shape[0] < 6:
        return float('nan')
    mean = points_3d.mean(axis=0)
    centered = points_3d - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order[:2]]
    proj = centered @ basis
    order_y = np.argsort(proj[:, 1])
    n = len(order_y)
    lower = proj[order_y[:n // 3]]
    upper = proj[order_y[-n // 3:]]
    def fit_dir(pts):
        if len(pts) < 2:
            return None
        x = pts[:, 0]
        y = pts[:, 1]
        a, _ = np.polyfit(x, y, 1)
        v = np.array([1.0, a])
        v = v / np.linalg.norm(v)
        return v
    v1 = fit_dir(lower)
    v2 = fit_dir(upper)
    if v1 is None or v2 is None:
        return float('nan')
    angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(v1, v2)), -1, 1)))
    return float(angle)


def nvc_score(points_3d: np.ndarray) -> float:
    """Neighbor consistency based on cosine similarity of adjacent deltas."""
    if points_3d.shape[0] < 3:
        return float('nan')
    d = points_3d[1:] - points_3d[:-1]
    d1 = d[:-1]
    d2 = d[1:]
    denom = (np.linalg.norm(d1, axis=-1) * np.linalg.norm(d2, axis=-1) + 1e-8)
    cos = (d1 * d2).sum(axis=-1) / denom
    return float(np.mean(cos))


def curvature_score(points_3d: np.ndarray) -> float:
    """Mean second-difference norm."""
    if points_3d.shape[0] < 3:
        return float('nan')
    diff2 = points_3d[2:] - 2 * points_3d[1:-1] + points_3d[:-2]
    return float(np.mean(np.linalg.norm(diff2, axis=-1)))


def swap_labels(order_ids: list[int], a: int, b: int) -> list[int]:
    out = order_ids[:]
    if a in out and b in out:
        ia = out.index(a)
        ib = out.index(b)
        out[ia], out[ib] = out[ib], out[ia]
    return out


def main():
    parser = argparse.ArgumentParser(description="Ablation: neighbor swaps structural sensitivity")
    parser.add_argument('--embedding_dir', type=str, required=True)
    parser.add_argument('--point_cloud_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    dataset = AssemblyDataset(
        embedding_dir=Path(args.embedding_dir),
        point_cloud_dir=Path(args.point_cloud_dir),
        split=args.split,
        max_vertebrae=30,
        augment=False,
    )

    results = {
        "baseline": {"cobb": [], "curvature": [], "nvc": []},
        "swap_T12_L1": {"cobb": [], "curvature": [], "nvc": []},
        "swap_L1_L2": {"cobb": [], "curvature": [], "nvc": []},
    }

    # Label mapping (1-based)
    T12 = 19
    L1 = 20
    L2 = 21

    for i in range(len(dataset)):
        sample = dataset[i]
        valid = sample['mask'].numpy()
        if valid.sum() < 3:
            continue
        ids = sample['vertebra_ids'].numpy()[valid] + 1  # 1-based
        points = sample['points'].numpy()[valid]
        centroids = points.mean(axis=1)
        order = np.argsort(ids)
        ids_order = ids[order].tolist()
        centroids_order = centroids[order]

        base_cobb = cobb_angle(centroids_order)
        base_curv = curvature_score(centroids_order)
        base_nvc = nvc_score(centroids_order)
        if np.isfinite(base_cobb):
            results["baseline"]["cobb"].append(base_cobb)
        if np.isfinite(base_curv):
            results["baseline"]["curvature"].append(base_curv)
        if np.isfinite(base_nvc):
            results["baseline"]["nvc"].append(base_nvc)

        # Swap T12-L1
        ids_swapped = swap_labels(ids_order, T12, L1)
        idx_map = [ids_order.index(v) for v in ids_swapped]
        centroids_swapped = centroids_order[idx_map]
        c = cobb_angle(centroids_swapped)
        k = curvature_score(centroids_swapped)
        n = nvc_score(centroids_swapped)
        if np.isfinite(c):
            results["swap_T12_L1"]["cobb"].append(c)
        if np.isfinite(k):
            results["swap_T12_L1"]["curvature"].append(k)
        if np.isfinite(n):
            results["swap_T12_L1"]["nvc"].append(n)

        # Swap L1-L2
        ids_swapped = swap_labels(ids_order, L1, L2)
        idx_map = [ids_order.index(v) for v in ids_swapped]
        centroids_swapped = centroids_order[idx_map]
        c = cobb_angle(centroids_swapped)
        k = curvature_score(centroids_swapped)
        n = nvc_score(centroids_swapped)
        if np.isfinite(c):
            results["swap_L1_L2"]["cobb"].append(c)
        if np.isfinite(k):
            results["swap_L1_L2"]["curvature"].append(k)
        if np.isfinite(n):
            results["swap_L1_L2"]["nvc"].append(n)

    summary = {}
    for key, vals in results.items():
        summary[key] = {
            "cobb_mean": float(np.mean(vals["cobb"])) if vals["cobb"] else float('nan'),
            "curvature_mean": float(np.mean(vals["curvature"])) if vals["curvature"] else float('nan'),
            "nvc_mean": float(np.mean(vals["nvc"])) if vals["nvc"] else float('nan'),
            "count": len(vals["cobb"]),
        }

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        json.dump({"summary": summary, "raw": results}, f, indent=2)
    print(f"✓ Saved ablation results to {out_path}")


if __name__ == "__main__":
    main()

