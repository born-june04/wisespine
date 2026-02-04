#!/usr/bin/env python3
"""
Visualize global differences between raw (CT centroids) and assembled predictions.
Produces:
1) Ordering confusion heatmap
2) Centerline spline comparison for top-K largest differences
3) Cobb-like angle comparison (upper vs lower line fit)
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import SpineAssemblyTransformer, SpineAssemblySpinalField
from utils.assembly_data_loader import AssemblyDataset


def load_assembly_model(assembly_path: Path, device: torch.device):
    checkpoint = torch.load(assembly_path, map_location=device)
    config = checkpoint.get('config', {})
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    has_spinal_field = any('field_pool' in k or 's_head' in k for k in state_dict.keys())
    has_delta_pose = any('delta_pose_head' in k for k in state_dict.keys())
    if has_spinal_field:
        model = SpineAssemblySpinalField(
            embed_dim=config.get('embed_dim', 512),
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 6),
            num_heads=config.get('num_heads', 8),
            num_vertebra_types=26,
            use_mask_token=True,
            enable_delta_pose=has_delta_pose,
        ).to(device)
    else:
        model = SpineAssemblyTransformer(
            embed_dim=config.get('embed_dim', 512),
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 6),
            num_heads=config.get('num_heads', 8),
            num_vertebra_types=26,
            use_mask_token=True,
        ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_ct_centroids(ct_dir: Path, subject_id: str):
    candidate_dirs = []
    subject_dir = ct_dir / subject_id
    if subject_dir.exists():
        candidate_dirs.append(subject_dir)
    for dataset_dir in ct_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        cand = dataset_dir / subject_id
        if cand.exists():
            candidate_dirs.append(cand)
    if not candidate_dirs:
        for cand in ct_dir.rglob(subject_id):
            if cand.is_dir():
                candidate_dirs.append(cand)
                break
    for cand_dir in candidate_dirs:
        path = cand_dir / 'vertebra_centroids.npy'
        if path.exists():
            try:
                return np.load(path)
            except Exception:
                return None
    return None


def get_centroid_for_id(centroids: np.ndarray, v_id: int):
    if centroids is None:
        return None
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        return None
    if 1 <= v_id <= centroids.shape[0]:
        c = centroids[v_id - 1]
        if np.any(c != 0):
            return c
    if 0 <= v_id < centroids.shape[0]:
        c = centroids[v_id]
        if np.any(c != 0):
            return c
    return None


def compute_rigid_alignment(src: np.ndarray, dst: np.ndarray):
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = dst_mean - (R @ src_mean)
    return R, t


def project_to_plane(points: np.ndarray):
    """Project to 2D using PCA."""
    mean = points.mean(axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order[:2]]
    proj = centered @ basis
    return proj


def cobb_like_angle(points_2d: np.ndarray):
    """Angle between line fits to upper and lower halves in 2D."""
    if len(points_2d) < 6:
        return np.nan
    # Split by y-axis (vertical)
    order = np.argsort(points_2d[:, 1])
    n = len(order)
    lower = points_2d[order[:n // 3]]
    upper = points_2d[order[-n // 3:]]
    def fit_dir(pts):
        if len(pts) < 2:
            return None
        x = pts[:, 0]
        y = pts[:, 1]
        a, b = np.polyfit(x, y, 1)
        v = np.array([1.0, a])
        v = v / np.linalg.norm(v)
        return v
    v1 = fit_dir(lower)
    v2 = fit_dir(upper)
    if v1 is None or v2 is None:
        return np.nan
    angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(v1, v2)), -1, 1)))
    return angle


def main():
    parser = argparse.ArgumentParser(description="Visualize global differences")
    parser.add_argument('--assembly_path', type=str, required=True)
    parser.add_argument('--embedding_dir', type=str, required=True)
    parser.add_argument('--point_cloud_dir', type=str, required=True)
    parser.add_argument('--ct_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--output_dir', type=str, default='outputs/assembly/visualization')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = load_assembly_model(Path(args.assembly_path), device)

    dataset = AssemblyDataset(
        embedding_dir=Path(args.embedding_dir),
        point_cloud_dir=Path(args.point_cloud_dir),
        split=args.split,
        max_vertebrae=30,
        augment=False,
    )

    num_classes = 26
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    subject_stats = []

    for i in range(len(dataset)):
        sample = dataset[i]
        subject_id = sample['subject_id']
        ct_centroids = load_ct_centroids(Path(args.ct_dir), subject_id)
        if ct_centroids is None:
            continue

        embeddings = sample['embeddings'].unsqueeze(0).to(device)
        pad_mask = ~sample['mask'].unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(embeddings, pad_mask=pad_mask, mask_mask=None)
        ordering_pred = preds['ordering'][0].argmax(dim=-1).cpu().numpy()
        poses = {
            't': preds['pose']['t'][0].cpu().numpy(),
            'R': preds['pose']['R'][0].cpu().numpy(),
        }

        valid_mask = sample['mask'].cpu().numpy()
        valid_indices = np.where(valid_mask)[0]
        ordering_pred = ordering_pred[valid_indices]
        poses_t = poses['t'][valid_indices]
        vertebra_ids = sample['vertebra_ids'][valid_indices].cpu().numpy()  # 0-based

        # Confusion
        for gt, pr in zip(vertebra_ids, ordering_pred):
            if 0 <= gt < num_classes and 0 <= pr < num_classes:
                confusion[int(gt), int(pr)] += 1

        # Build CT centroid list for valid ids
        ct_list = []
        pred_list = []
        ids_list = []
        for j, v0 in enumerate(vertebra_ids):
            v_id = int(v0) + 1
            c = get_centroid_for_id(ct_centroids, v_id)
            if c is None:
                continue
            ct_list.append(c)
            pred_list.append(poses_t[j])
            ids_list.append(v_id)
        if len(ct_list) < 3:
            continue

        ct_arr = np.array(ct_list)
        pred_arr = np.array(pred_list)
        R, t = compute_rigid_alignment(pred_arr, ct_arr)
        pred_aligned = (pred_arr @ R.T) + t

        # Per-vertebra error
        errors = np.linalg.norm(pred_aligned - ct_arr, axis=1)
        mean_err = float(errors.mean())

        # Centerline projection
        order = np.argsort(ids_list)
        ct_sorted = ct_arr[order]
        pred_sorted = pred_aligned[order]
        ct_2d = project_to_plane(ct_sorted)
        pred_2d = project_to_plane(pred_sorted)
        angle_ct = cobb_like_angle(ct_2d)
        angle_pred = cobb_like_angle(pred_2d)

        subject_stats.append({
            'subject_id': subject_id,
            'ids': np.array(ids_list),
            'errors': errors,
            'mean_err': mean_err,
            'ct_2d': ct_2d,
            'pred_2d': pred_2d,
            'angle_ct': angle_ct,
            'angle_pred': angle_pred,
        })

    if not subject_stats:
        print("No subjects with CT centroids found.")
        return

    # Ordering confusion heatmap
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    im = ax.imshow(confusion, cmap='viridis')
    ax.set_title("Ordering Confusion (GT vs Pred)")
    ax.set_xlabel("Predicted ID")
    ax.set_ylabel("GT ID")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    conf_path = output_dir / "ordering_confusion.png"
    fig.savefig(conf_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"✓ Saved confusion heatmap: {conf_path}")

    # Global difference ranking
    subject_stats.sort(key=lambda x: x['mean_err'], reverse=True)
    top = subject_stats[:args.top_k]

    for stats in top:
        subject_id = stats['subject_id']
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=200)

        # Centerline comparison
        axes[0].plot(stats['ct_2d'][:, 0], stats['ct_2d'][:, 1], '-o', label='CT')
        axes[0].plot(stats['pred_2d'][:, 0], stats['pred_2d'][:, 1], '-o', label='Assembled')
        axes[0].set_title(f"{subject_id} Centerline")
        axes[0].axis('equal')
        axes[0].legend()

        # Error per vertebra
        axes[1].bar(np.arange(len(stats['errors'])), stats['errors'])
        axes[1].set_title("Per-vertebra error (mm)")
        axes[1].set_xlabel("Vertebra index")
        axes[1].set_ylabel("Distance")

        # Cobb-like angle comparison
        axes[2].axis('off')
        axes[2].text(
            0.1, 0.6,
            f"Mean error: {stats['mean_err']:.2f} mm\n"
            f"Cobb-like (CT): {stats['angle_ct']:.2f}°\n"
            f"Cobb-like (Pred): {stats['angle_pred']:.2f}°",
            fontsize=12,
        )

        out_path = output_dir / f"{subject_id}_global_diff.png"
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
        print(f"✓ Saved global diff figure: {out_path}")


if __name__ == "__main__":
    main()

