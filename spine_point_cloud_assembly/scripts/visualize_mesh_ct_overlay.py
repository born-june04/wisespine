#!/usr/bin/env python3
"""
Overlay per-vertebra mesh on CT axial slice.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import itertools

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import SpineAssemblyTransformer, SpineAssemblySpinalField
from utils.assembly_data_loader import AssemblyDataset


def get_label_name(label: int) -> str:
    if 1 <= label <= 7:
        return f"C{label}"
    if 8 <= label <= 19:
        return f"T{label - 7}"
    if 20 <= label <= 24:
        return f"L{label - 19}"
    return f"V{label}"


def get_label_color(label: int, num_labels: int = 26):
    try:
        cmap = plt.get_cmap('tab20', num_labels)
    except TypeError:
        cmap = plt.get_cmap('tab20')
    return cmap((label - 1) % num_labels)[:3]


def load_ct_volume(ct_dir: Path, subject_id: str):
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
        for fname in ['ct_volume_1mm.npy', 'processed_ct_1mm.npy', 'ct_volume.npy']:
            path = cand_dir / fname
            if path.exists():
                return np.load(path), path
    return None, None


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


def get_subject_sample(embedding_dir: Path, point_cloud_dir: Path, subject_id: str):
    dataset = AssemblyDataset(
        embedding_dir=embedding_dir,
        point_cloud_dir=point_cloud_dir,
        split='test',
        max_vertebrae=30,
        augment=False,
    )
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample['subject_id'] == subject_id:
            return sample
    raise ValueError(f"Subject {subject_id} not found in {embedding_dir}")


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
                return np.load(path), path
            except Exception:
                return None, path
    return None, None


def get_centroid_for_id(centroids: np.ndarray, v_id: int):
    if centroids is None:
        return None
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        return None
    # Try 1-based indexing
    if 1 <= v_id <= centroids.shape[0]:
        c = centroids[v_id - 1]
        if np.any(c != 0):
            return c
    # Try direct index
    if 0 <= v_id < centroids.shape[0]:
        c = centroids[v_id]
        if np.any(c != 0):
            return c
    return None


def load_ct_mask(ct_dir: Path, subject_id: str):
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
        path = cand_dir / 'mask_volume_1mm.npy'
        if path.exists():
            try:
                return np.load(path), path
            except Exception:
                return None, path
    return None, None


def get_mask_bbox(mask: np.ndarray, v_id: int):
    coords = np.where(mask == v_id)
    if len(coords[0]) == 0:
        return None
    p_min, p_max = coords[0].min(), coords[0].max()
    i_min, i_max = coords[1].min(), coords[1].max()
    l_min, l_max = coords[2].min(), coords[2].max()
    return (p_min, p_max, i_min, i_max, l_min, l_max)


def compute_rigid_alignment(src: np.ndarray, dst: np.ndarray):
    """Kabsch alignment: find R,t such that R@src + t ~= dst."""
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


def apply_axis_map(points: np.ndarray, perm, flips, ct_shape):
    pts = points[:, perm].copy()
    for axis, do_flip in enumerate(flips):
        if do_flip:
            pts[:, axis] = (ct_shape[axis] - 1) - pts[:, axis]
    return pts


def infer_best_axis_mapping(ct_mask: np.ndarray, mesh_files, max_points=2000):
    perms = list(itertools.permutations([0, 1, 2], 3))
    flips_list = list(itertools.product([False, True], repeat=3))
    ct_shape = ct_mask.shape
    best = None
    best_score = -1.0

    samples = []
    for mesh_file in mesh_files[:6]:
        v_id = int(mesh_file.stem.split('_')[1])
        data = np.load(mesh_file)
        vertices = data['vertices'].astype(np.float32)
        if len(vertices) > max_points:
            idx = np.random.choice(len(vertices), max_points, replace=False)
            vertices = vertices[idx]
        samples.append((v_id, vertices))

    for perm in perms:
        for flips in flips_list:
            total = 0
            hits = 0
            for v_id, vertices in samples:
                pts = apply_axis_map(vertices, perm, flips, ct_shape)
                pts_idx = np.round(pts).astype(int)
                valid = (
                    (pts_idx[:, 0] >= 0) & (pts_idx[:, 0] < ct_shape[0]) &
                    (pts_idx[:, 1] >= 0) & (pts_idx[:, 1] < ct_shape[1]) &
                    (pts_idx[:, 2] >= 0) & (pts_idx[:, 2] < ct_shape[2])
                )
                pts_idx = pts_idx[valid]
                total += len(pts_idx)
                if len(pts_idx):
                    hits += np.sum(ct_mask[pts_idx[:, 0], pts_idx[:, 1], pts_idx[:, 2]] == v_id)
            if total > 0:
                score = hits / total
                if score > best_score:
                    best_score = score
                    best = (perm, flips, score)

    return best


def main():
    parser = argparse.ArgumentParser(description="Overlay vertebra meshes on CT axial slices")
    parser.add_argument('--ct_dir', type=str, required=True,
                        help='CT directory (e.g., VerSe/processed)')
    parser.add_argument('--mesh_dir', type=str, required=True,
                        help='Mesh directory (outputs/meshes)')
    parser.add_argument('--subject_id', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/assembly/visualization',
                        help='Output directory')
    parser.add_argument('--axis_order', type=str, default='PIL',
                        help='Vertex axis order mapping to CT axes (default: PIL)')
    parser.add_argument('--slice_thickness', type=int, default=1,
                        help='Voxel thickness for axial slice overlay')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Overlay alpha for mesh points')
    parser.add_argument('--point_size', type=float, default=2.0,
                        help='Point size for overlay')
    parser.add_argument('--assembly_path', type=str, default=None,
                        help='Assembly model checkpoint for assembled overlay')
    parser.add_argument('--embedding_dir', type=str, default=None,
                        help='Assembly embedding directory')
    parser.add_argument('--point_cloud_dir', type=str, default=None,
                        help='Point cloud directory for AssemblyDataset')
    parser.add_argument('--use_pred_rotation', action='store_true',
                        help='Use predicted rotation for assembled overlay (default: translation only)')
    parser.add_argument('--use_global_align', action='store_true',
                        help='Use global rigid alignment from predicted t to CT centroids')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    ct_dir = Path(args.ct_dir)
    mesh_dir = Path(args.mesh_dir) / args.subject_id
    output_dir = Path(args.output_dir) / args.subject_id
    output_dir.mkdir(parents=True, exist_ok=True)

    ct, ct_path = load_ct_volume(ct_dir, args.subject_id)
    if ct is None:
        raise FileNotFoundError(f"CT not found for {args.subject_id} in {ct_dir}")
    ct_centroids, ct_centroids_path = load_ct_centroids(ct_dir, args.subject_id)
    if ct_centroids is not None:
        print(f"✓ Loaded CT centroids: {ct_centroids_path}")
    ct_mask, ct_mask_path = load_ct_mask(ct_dir, args.subject_id)
    if ct_mask is not None:
        print(f"✓ Loaded CT mask: {ct_mask_path}")

    # CT axes: (P, I, L)
    axis_map = {'P': 0, 'I': 1, 'L': 2}
    order = args.axis_order.upper()
    if len(order) != 3 or any(c not in axis_map for c in order):
        raise ValueError("axis_order must be a permutation of PIL")
    v_to_ct = [axis_map[c] for c in order]
    flips = (False, False, False)

    mesh_files = sorted(mesh_dir.glob('vertebra_*_mesh.npz'))
    if not mesh_files:
        raise FileNotFoundError(f"No mesh files in {mesh_dir}")

    # Auto-infer axis mapping if CT mask available
    if ct_mask is not None:
        best = infer_best_axis_mapping(ct_mask, mesh_files)
        if best is not None:
            v_to_ct, flips, score = best
            print(f"✓ Auto axis mapping: perm={v_to_ct}, flips={flips}, score={score:.3f}")

    # Optional: load assembly predictions
    use_assembly = args.assembly_path and args.embedding_dir and args.point_cloud_dir
    poses_valid = None
    ordering_pred_valid = None
    vertebra_ids_valid = None
    align_R = None
    align_t = None
    if use_assembly:
        device = torch.device(args.device)
        model = load_assembly_model(Path(args.assembly_path), device)
        sample = get_subject_sample(Path(args.embedding_dir), Path(args.point_cloud_dir), args.subject_id)
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
        ordering_pred_valid = ordering_pred[valid_indices]
        poses_valid = {
            't': poses['t'][valid_indices],
            'R': poses['R'][valid_indices],
        }
        vertebra_ids_valid = sample['vertebra_ids'][valid_indices].cpu().numpy()
        if ct_centroids is not None and args.use_global_align:
            src = []
            dst = []
            for i, v_id0 in enumerate(vertebra_ids_valid):
                v_id = int(v_id0) + 1
                c = get_centroid_for_id(ct_centroids, v_id)
                if c is None:
                    continue
                src.append(poses_valid['t'][i])
                dst.append(c)
            if len(src) >= 3:
                align_R, align_t = compute_rigid_alignment(np.array(src), np.array(dst))
                print("✓ Computed rigid alignment to CT centroids (global)")
            elif len(src) >= 1:
                align_t = np.array(dst[0]) - np.array(src[0])
                align_R = np.eye(3)
                print("✓ Using translation-only alignment to CT centroids (global)")

    for mesh_file in mesh_files:
        v_id = int(mesh_file.stem.split('_')[1])
        data = np.load(mesh_file)
        vertices = data['vertices'].astype(np.float32)
        # Map vertex columns to CT axes
        verts_ct = apply_axis_map(vertices, v_to_ct, flips, ct.shape)

        # Axial slice index in CT (P axis)
        c_ct = get_centroid_for_id(ct_centroids, v_id) if ct_centroids is not None else None
        if c_ct is not None:
            p_center = int(np.round(c_ct[0]))
        else:
            p_center = int(np.round(verts_ct[:, 0].mean()))
        p_center = np.clip(p_center, 0, ct.shape[0] - 1)
        half = max(0, args.slice_thickness // 2)
        p_min = max(0, p_center - half)
        p_max = min(ct.shape[0] - 1, p_center + half)

        # CT axial slice
        ct_slice = ct[p_center, :, :]

        # Select mesh points near the axial slice
        mask = (verts_ct[:, 0] >= p_min) & (verts_ct[:, 0] <= p_max)
        verts_slice = verts_ct[mask]
        if len(verts_slice) == 0:
            continue

        # Overlay raw (and assembled if available)
        if use_assembly and v_id in (vertebra_ids_valid + 1):
            idx = int(np.where((vertebra_ids_valid + 1) == v_id)[0][0])
            R = poses_valid['R'][idx]
            t = poses_valid['t'][idx]
            centroid_raw = vertices.mean(axis=0)
            verts_local = vertices - centroid_raw
            # Assemble mesh in CT space
            c_ct = get_centroid_for_id(ct_centroids, v_id) if ct_centroids is not None else None
            if c_ct is not None and not args.use_global_align:
                # Per-vertebra centroid anchoring (most stable for overlay)
                if args.use_pred_rotation:
                    verts_assembled = (verts_local @ R.T) + c_ct
                else:
                    verts_assembled = verts_local + c_ct
            else:
                # Global alignment using predicted translations
                if args.use_pred_rotation:
                    verts_assembled = (verts_local @ R.T) + t
                else:
                    verts_assembled = verts_local + t
                if align_R is not None:
                    verts_assembled = (verts_assembled @ align_R.T) + align_t
                else:
                    verts_assembled = verts_assembled + centroid_raw
            verts_assembled_ct = apply_axis_map(verts_assembled, v_to_ct, flips, ct.shape)
            mask_a = (verts_assembled_ct[:, 0] >= p_min) & (verts_assembled_ct[:, 0] <= p_max)
            verts_a_slice = verts_assembled_ct[mask_a]
            if len(verts_a_slice) == 0:
                # Fallback: use a thicker slab for assembled points
                widen = 5
                mask_a = (verts_assembled_ct[:, 0] >= p_center - widen) & (verts_assembled_ct[:, 0] <= p_center + widen)
                verts_a_slice = verts_assembled_ct[mask_a]
            if len(verts_a_slice) == 0:
                # Last resort: project all assembled points
                verts_a_slice = verts_assembled_ct

            # Compute zoom bbox from raw/assembled overlay
            def get_bbox(pts):
                y = pts[:, 1]
                x = pts[:, 2]
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                pad = 10
                x_min = max(0, int(x_min) - pad)
                x_max = min(ct_slice.shape[1] - 1, int(x_max) + pad)
                y_min = max(0, int(y_min) - pad)
                y_max = min(ct_slice.shape[0] - 1, int(y_max) + pad)
                return x_min, x_max, y_min, y_max

            if len(verts_a_slice) > 0:
                combined = np.vstack([verts_slice, verts_a_slice])
            else:
                combined = verts_slice
            x_min, x_max, y_min, y_max = get_bbox(combined)
            ct_zoom = ct_slice[y_min:y_max + 1, x_min:x_max + 1]

            # Optional: restrict overlays to CT mask bbox to avoid spurious points
            if ct_mask is not None:
                bbox = get_mask_bbox(ct_mask, v_id)
                if bbox is not None:
                    p0, p1, i0, i1, l0, l1 = bbox
                    def in_bbox(pts):
                        return (
                            (pts[:, 0] >= p0) & (pts[:, 0] <= p1) &
                            (pts[:, 1] >= i0) & (pts[:, 1] <= i1) &
                            (pts[:, 2] >= l0) & (pts[:, 2] <= l1)
                        )
                    verts_slice = verts_slice[in_bbox(verts_slice)]
                    if len(verts_a_slice):
                        verts_a_slice = verts_a_slice[in_bbox(verts_a_slice)]

                # Show only points that fall inside the CT mask for this vertebra
                def in_mask(pts):
                    if len(pts) == 0:
                        return pts
                    idx = np.round(pts).astype(int)
                    valid = (
                        (idx[:, 0] >= 0) & (idx[:, 0] < ct_mask.shape[0]) &
                        (idx[:, 1] >= 0) & (idx[:, 1] < ct_mask.shape[1]) &
                        (idx[:, 2] >= 0) & (idx[:, 2] < ct_mask.shape[2])
                    )
                    idx = idx[valid]
                    if len(idx) == 0:
                        return pts[:0]
                    hits = ct_mask[idx[:, 0], idx[:, 1], idx[:, 2]] == v_id
                    return pts[valid][hits]

                verts_slice = in_mask(verts_slice)
                if len(verts_a_slice):
                    verts_a_slice = in_mask(verts_a_slice)

                # Compute overlap ratio with CT mask (voxel hit rate)
                def overlap_ratio(pts):
                    if len(pts) == 0:
                        return 0.0
                    idx = np.round(pts).astype(int)
                    valid = (
                        (idx[:, 0] >= 0) & (idx[:, 0] < ct_mask.shape[0]) &
                        (idx[:, 1] >= 0) & (idx[:, 1] < ct_mask.shape[1]) &
                        (idx[:, 2] >= 0) & (idx[:, 2] < ct_mask.shape[2])
                    )
                    idx = idx[valid]
                    if len(idx) == 0:
                        return 0.0
                    hits = (ct_mask[idx[:, 0], idx[:, 1], idx[:, 2]] == v_id).sum()
                    return float(hits) / float(len(idx))

                raw_overlap = overlap_ratio(verts_slice)
                asm_overlap = overlap_ratio(verts_a_slice) if len(verts_a_slice) else 0.0
                delta = asm_overlap - raw_overlap
                print(f"DEBUG {v_id}: overlap raw={raw_overlap:.3f}, asm={asm_overlap:.3f}, Δ={delta:+.3f}")

            # Debug centroid distances (raw/assembled vs CT centroid)
            if c_ct is not None:
                raw_centroid = verts_ct.mean(axis=0)
                asm_centroid = verts_assembled_ct.mean(axis=0) if len(verts_assembled_ct) else np.array([np.nan]*3)
                raw_d = np.linalg.norm(raw_centroid - c_ct)
                asm_d = np.linalg.norm(asm_centroid - c_ct)
                print(f"DEBUG {v_id}: centroid raw->CT {raw_d:.2f}, asm->CT {asm_d:.2f}")

            fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=200)
            color_raw = (1.0, 1.0, 0.0)  # yellow
            color_asm = (1.0, 0.2, 0.6)  # pink

            # Left: full CT slice
            axes[0].imshow(ct_slice, cmap='gray', origin='lower')
            from matplotlib.patches import Rectangle
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             linewidth=2, edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)
            axes[0].set_title(f"CT axial (P={p_center})")
            axes[0].axis('off')

            # Middle: raw zoom
            axes[1].imshow(ct_zoom, cmap='gray', origin='lower')
            axes[1].scatter(verts_slice[:, 2] - x_min, verts_slice[:, 1] - y_min,
                            s=args.point_size, c=[color_raw], alpha=args.alpha, linewidths=0)
            axes[1].set_title(f"Raw mesh ({get_label_name(v_id)})")
            axes[1].axis('off')

            # Right: assembled zoom
            axes[2].imshow(ct_zoom, cmap='gray', origin='lower')
            if len(verts_a_slice) > 0:
                axes[2].scatter(verts_a_slice[:, 2] - x_min, verts_a_slice[:, 1] - y_min,
                                s=args.point_size, c=[color_asm], alpha=args.alpha, linewidths=0)
            axes[2].set_title("Assembled mesh")
            axes[2].axis('off')

            out_path = output_dir / f"ct_overlay_compare_vertebra_{v_id}.png"
            # Debug stats
            debug_path = output_dir / f"ct_overlay_compare_vertebra_{v_id}.npz"
            np.savez_compressed(
                debug_path,
                raw_min=verts_slice.min(axis=0),
                raw_max=verts_slice.max(axis=0),
                asm_min=verts_a_slice.min(axis=0) if len(verts_a_slice) else np.array([np.nan, np.nan, np.nan]),
                asm_max=verts_a_slice.max(axis=0) if len(verts_a_slice) else np.array([np.nan, np.nan, np.nan]),
                raw_count=len(verts_slice),
                asm_count=len(verts_a_slice),
                p_center=p_center,
                bbox=np.array([x_min, x_max, y_min, y_max]),
                ct_centroid=c_ct if c_ct is not None else np.array([np.nan, np.nan, np.nan]),
                raw_overlap=raw_overlap if ct_mask is not None else np.nan,
                asm_overlap=asm_overlap if ct_mask is not None else np.nan,
            )
            print(f"DEBUG {v_id}: raw_count={len(verts_slice)} asm_count={len(verts_a_slice)} bbox={x_min,x_max,y_min,y_max}")
        else:
            # Raw-only view (full CT + zoom)
            def get_bbox(pts):
                y = pts[:, 1]
                x = pts[:, 2]
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                pad = 10
                x_min = max(0, int(x_min) - pad)
                x_max = min(ct_slice.shape[1] - 1, int(x_max) + pad)
                y_min = max(0, int(y_min) - pad)
                y_max = min(ct_slice.shape[0] - 1, int(y_max) + pad)
                return x_min, x_max, y_min, y_max

            x_min, x_max, y_min, y_max = get_bbox(verts_slice)
            ct_zoom = ct_slice[y_min:y_max + 1, x_min:x_max + 1]
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
            color_raw = (1.0, 1.0, 0.0)
            axes[0].imshow(ct_slice, cmap='gray', origin='lower')
            from matplotlib.patches import Rectangle
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             linewidth=2, edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)
            axes[0].set_title(f"CT axial (P={p_center})")
            axes[0].axis('off')
            axes[1].imshow(ct_zoom, cmap='gray', origin='lower')
            axes[1].scatter(verts_slice[:, 2] - x_min, verts_slice[:, 1] - y_min,
                            s=args.point_size, c=[color_raw], alpha=args.alpha, linewidths=0)
            axes[1].set_title(f"Raw mesh ({get_label_name(v_id)})")
            axes[1].axis('off')
            out_path = output_dir / f"ct_overlay_vertebra_{v_id}.png"

        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
        print(f"✓ Saved overlay: {out_path}")


if __name__ == '__main__':
    main()

