#!/usr/bin/env python3
"""
Visualize Assembled Spine from Assembly Model Predictions

Reconstructs spine from predictions and visualizes using PyVista.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import SpineAssemblyTransformer, SpineAssemblySpinalField, SE3PointEncoder, features_to_irreps
from utils.assembly_data_loader import AssemblyDataset
from torch.utils.data import DataLoader

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("ERROR: PyVista not installed. Install with: pip install pyvista")
    sys.exit(1)


def load_models(encoder_path: Path, assembly_path: Path, device: torch.device):
    """Load encoder and assembly models"""
    print(f"Loading encoder from {encoder_path}")
    encoder_checkpoint = torch.load(encoder_path, map_location=device)
    encoder_config = encoder_checkpoint.get('config', {})
    
    encoder = SE3PointEncoder(
        out_dim=encoder_config.get('output_dim', 512),
        num_layers=encoder_config.get('num_layers', 4),
        num_radial=16,
        cutoff=5.0,
        max_num_neighbors=32,
        use_curvature=True,
    ).to(device)
    
    if 'model_state_dict' in encoder_checkpoint:
        encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    else:
        encoder.load_state_dict(encoder_checkpoint)
    encoder.eval()
    print("✓ Encoder loaded")
    
    print(f"Loading assembly model from {assembly_path}")
    assembly_checkpoint = torch.load(assembly_path, map_location=device)
    assembly_config = assembly_checkpoint.get('config', {})
    
    # Detect model type
    state_dict = assembly_checkpoint.get('model_state_dict', assembly_checkpoint)
    has_spinal_field = any('field_pool' in k or 's_head' in k for k in state_dict.keys())
    has_delta_pose = any('delta_pose_head' in k for k in state_dict.keys())
    
    if has_spinal_field:
        assembly = SpineAssemblySpinalField(
            embed_dim=assembly_config.get('embed_dim', 512),
            hidden_dim=assembly_config.get('hidden_dim', 256),
            num_layers=assembly_config.get('num_layers', 6),
            num_heads=assembly_config.get('num_heads', 8),
            num_vertebra_types=26,
            use_mask_token=True,
            enable_delta_pose=has_delta_pose,
        ).to(device)
        print("✓ Spinal Field model detected")
    else:
        assembly = SpineAssemblyTransformer(
            embed_dim=assembly_config.get('embed_dim', 512),
            hidden_dim=assembly_config.get('hidden_dim', 256),
            num_layers=assembly_config.get('num_layers', 6),
            num_heads=assembly_config.get('num_heads', 8),
            num_vertebra_types=26,
            use_mask_token=True,
        ).to(device)
        print("✓ Baseline model detected")
    
    if 'model_state_dict' in assembly_checkpoint:
        assembly.load_state_dict(assembly_checkpoint['model_state_dict'])
    else:
        assembly.load_state_dict(assembly_checkpoint)
    assembly.eval()
    print("✓ Assembly model loaded")
    
    return encoder, assembly


def reconstruct_spine(
    point_clouds: np.ndarray,  # (N, M, 3) - N vertebrae, M points each
    poses: dict,  # {'t': (N, 3), 'R': (N, 3, 3)}
    ordering: np.ndarray,  # (N,) - predicted types
    use_inverse: bool = False,  # Not used anymore, kept for compatibility
) -> np.ndarray:
    """
    Reconstruct assembled spine from point clouds and poses.
    
    Based on training target definition:
    - relative_translation = centroids - first_centroid
    - Point clouds are already centered per vertebra (mean ~ 0) from encoder
    - t[0] should be [0,0,0] (first vertebra at origin)
    - t[i] is position of vertebra i relative to first vertebra
    
    Reconstruction:
    - Apply rotation to centered local points: x_rotated = R @ x_local
    - Translate to relative position: x_global = x_rotated + t_i
    
    Args:
        point_clouds: (N, M, 3) - local point clouds (already centered per vertebra)
        poses: {'t': (N, 3), 'R': (N, 3, 3)} - predicted poses
        ordering: (N,) - predicted vertebra types
        use_inverse: Not used (kept for compatibility)
    
    Returns:
        assembled_points: (total_points, 3) - all vertebrae in global space
        vertebra_labels: (total_points,) - type label for each point
    """
    N = len(point_clouds)
    assembled_points = []
    vertebra_labels = []
    
    t = poses['t']  # (N, 3) - relative to first vertebra
    R = poses['R']  # (N, 3, 3)
    
    # Try different reconstruction strategies
    # Strategy 1: No rotation, just translation (if rotation is identity or causing issues)
    # Strategy 2: Full rotation + translation
    
    for i in range(N):
        points_local = point_clouds[i]  # (M, 3) - already centered (mean ~ 0)
        t_i = t[i]  # (3,) - position relative to first vertebra
        R_i = R[i]  # (3, 3)
        
        if use_inverse:
            # Inverse transform: x_global = (x_local - t) @ R^T
            # Assumes model learned: x_local = R @ x_global + t
            points_global = ((points_local - t_i) @ R_i.T)  # (M, 3)
        else:
            # Forward transform: x_global = R @ x_local + t
            # Assumes model learned: x_global = R @ x_local + t
            points_rotated = (R_i @ points_local.T).T  # (M, 3)
            points_global = points_rotated + t_i  # (M, 3)
        
        assembled_points.append(points_global)
        vertebra_labels.extend([int(ordering[i])] * len(points_global))
    
    assembled_points = np.concatenate(assembled_points, axis=0)  # (total, 3)
    vertebra_labels = np.array(vertebra_labels)  # (total,)
    
    return assembled_points, vertebra_labels


def save_assembled_spine(
    assembled_points: np.ndarray,  # (N, 3)
    vertebra_labels: np.ndarray,  # (N,)
    output_path: Path,
    format: str = 'ply',
):
    """
    Save assembled spine to file for visualization.
    
    Args:
        assembled_points: (N, 3) point cloud
        vertebra_labels: (N,) vertebra type labels
        output_path: Path to save file (without extension)
        format: File format ('ply', 'vtk', 'obj', or 'all')
    """
    # Create PyVista point cloud
    point_cloud = pv.PolyData(assembled_points)
    point_cloud['vertebra_type'] = vertebra_labels.astype(np.int32)
    
    # Add RGB colors based on vertebra type
    unique_labels = np.unique(vertebra_labels)
    num_labels = len(unique_labels)
    label_to_color = get_label_colors(vertebra_labels)
    
    # Assign colors to points
    colors = np.array([label_to_color[label] for label in vertebra_labels])  # (N, 3) RGB
    colors = (colors * 255).astype(np.uint8)  # Convert to 0-255 range
    point_cloud['RGB'] = colors
    
    saved_files = []
    
    if format in ['ply', 'all']:
        ply_path = output_path.with_suffix('.ply')
        point_cloud.save(str(ply_path))
        saved_files.append(ply_path)
        print(f"✓ Saved PLY file: {ply_path}")
        print(f"  - Points: {len(assembled_points)}")
        print(f"  - Vertebra types: {num_labels} (types {unique_labels.min()}-{unique_labels.max()})")
        print(f"  - Includes RGB colors and vertebra_type scalar")
    
    if format in ['vtk', 'all']:
        vtk_path = output_path.with_suffix('.vtk')
        point_cloud.save(str(vtk_path))
        saved_files.append(vtk_path)
        print(f"✓ Saved VTK file: {vtk_path}")
    
    if format in ['obj', 'all']:
        obj_path = output_path.with_suffix('.obj')
        point_cloud.save(str(obj_path))
        saved_files.append(obj_path)
        print(f"✓ Saved OBJ file: {obj_path}")
    
    # Also save metadata JSON with detailed vertebra information
    # Count points per vertebra type
    points_per_vertebra = {}
    for label in unique_labels:
        points_per_vertebra[int(label)] = int((vertebra_labels == label).sum())
    label_name_map = {int(label): get_label_name(int(label), vertebra_labels) for label in unique_labels}
    
    metadata = {
        'subject_id': str(output_path.stem).replace('assembled_spine_', ''),
        'num_points': int(len(assembled_points)),
        'num_vertebrae': int(num_labels),
        'vertebra_types': sorted(unique_labels.tolist()),
        'vertebra_names': [label_name_map[int(label)] for label in sorted(unique_labels.tolist())],
        'label_name_map': label_name_map,
        'points_per_vertebra': points_per_vertebra,
        'bounds': {
            'x': [float(assembled_points[:, 0].min()), float(assembled_points[:, 0].max())],
            'y': [float(assembled_points[:, 1].min()), float(assembled_points[:, 1].max())],
            'z': [float(assembled_points[:, 2].min()), float(assembled_points[:, 2].max())],
        },
        'center': assembled_points.mean(axis=0).tolist(),
        'description': 'Complete assembled spine from all vertebrae of one subject',
    }
    
    import json
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    saved_files.append(metadata_path)
    print(f"✓ Saved metadata: {metadata_path}")
    
    return saved_files, label_to_color


def get_label_colors(vertebra_labels: np.ndarray) -> dict:
    """Map vertebra labels to consistent RGB colors."""
    unique_labels = np.unique(vertebra_labels)
    num_labels = len(unique_labels)
    import matplotlib.pyplot as plt
    try:
        colormap = plt.get_cmap('tab20', num_labels)
    except TypeError:
        colormap = plt.get_cmap('tab20')
    return {label: colormap(i)[:3] for i, label in enumerate(unique_labels)}


def get_label_name(label: int, all_labels: np.ndarray) -> str:
    """Map vertebra numeric label to anatomical name (C/T/L)."""
    if all_labels.min() == 0 and all_labels.max() <= 25:
        label = label + 1  # zero-based -> one-based
    if 1 <= label <= 7:
        return f"C{label}"
    if 8 <= label <= 19:
        return f"T{label - 7}"
    if 20 <= label <= 24:
        return f"L{label - 19}"
    return f"V{label}"


def to_one_based_label(label: int, all_labels: np.ndarray) -> int:
    """Convert label to one-based if labels are zero-based."""
    if all_labels.min() == 0 and all_labels.max() <= 25:
        return int(label) + 1
    return int(label)


def load_vertebra_mesh(mesh_dir: Path, subject_id: str, label_one_based: int):
    mesh_path = mesh_dir / subject_id / f"vertebra_{label_one_based}_mesh.npz"
    if not mesh_path.exists():
        return None
    data = np.load(mesh_path)
    return data['vertices'].astype(np.float32), data['faces'].astype(np.int32)


def load_ct_volume(ct_dir: Path, subject_id: str):
    """Load CT volume if available. Expected .npy files under ct_dir/subject_id."""
    if ct_dir is None:
        return None, None
    subject_id = subject_id.strip()
    candidate_dirs = []
    subject_dir = ct_dir / subject_id
    if subject_dir.exists():
        candidate_dirs.append(subject_dir)
    # Search dataset subfolders (e.g., dataset-*/subject_id)
    for dataset_dir in ct_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        cand = dataset_dir / subject_id
        if cand.exists():
            candidate_dirs.append(cand)
    # Fallback: recursive search for subject folder
    if not candidate_dirs:
        for cand in ct_dir.rglob(subject_id):
            if cand.is_dir():
                candidate_dirs.append(cand)
                break
    for cand_dir in candidate_dirs:
        for fname in ['ct_volume_1mm.npy', 'processed_ct_1mm.npy', 'ct_volume.npy']:
            path = cand_dir / fname
            if path.exists():
                try:
                    return np.load(path), path
                except Exception:
                    return None, path
    return None, None


def apply_pose_to_vertices(vertices: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (vertices @ R.T) + t


def save_mesh_compare_png(
    mesh_dir: Path,
    subject_id: str,
    raw_labels: np.ndarray,
    assembled_labels: np.ndarray,
    poses_valid: dict,
    output_path: Path,
    ct_volume: np.ndarray | None = None,
):
    """
    Save side-by-side mesh comparison (raw vs assembled) using per-vertebra meshes.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    label_to_color = get_label_colors(np.concatenate([raw_labels, assembled_labels]))

    raw_meshes = []
    assembled_meshes = []
    for i, raw_label in enumerate(raw_labels):
        label_one_based = to_one_based_label(int(raw_label), raw_labels)
        mesh = load_vertebra_mesh(mesh_dir, subject_id, label_one_based)
        if mesh is None:
            continue
        vertices, faces = mesh
        raw_meshes.append((vertices, faces, int(raw_label)))

        R = poses_valid['R'][i]
        t = poses_valid['t'][i]
        vertices_assembled = apply_pose_to_vertices(vertices, R, t)
        assembled_meshes.append((vertices_assembled, faces, int(assembled_labels[i])))

    if not raw_meshes or not assembled_meshes:
        print("WARNING: Mesh files missing; skipping mesh comparison PNG.")
        return

    ncols = 3 if ct_volume is not None else 2
    fig = plt.figure(figsize=(6 * ncols, 6), dpi=200)
    axes = [
        fig.add_subplot(1, ncols, 1, projection='3d'),
        fig.add_subplot(1, ncols, 2, projection='3d'),
    ]
    titles = ["Raw segmentation meshes", "Assembled meshes (model)"]
    mesh_lists = [raw_meshes, assembled_meshes]

    for ax, title, meshes in zip(axes, titles, mesh_lists):
        all_vertices = []
        for vertices, faces, label in meshes:
            color = label_to_color[label]
            poly = Poly3DCollection(vertices[faces], alpha=0.8)
            poly.set_facecolor(color)
            poly.set_edgecolor('none')
            ax.add_collection3d(poly)
            all_vertices.append(vertices)
        all_vertices = np.concatenate(all_vertices, axis=0)
        ax.set_xlim(all_vertices[:, 0].min(), all_vertices[:, 0].max())
        ax.set_ylim(all_vertices[:, 1].min(), all_vertices[:, 1].max())
        ax.set_zlim(all_vertices[:, 2].min(), all_vertices[:, 2].max())
        ax.set_title(title)
        ax.set_axis_off()

    if ct_volume is not None:
        ax_ct = fig.add_subplot(1, ncols, 3)
        # Use sagittal slice (I axis) by default
        i_center = ct_volume.shape[1] // 2
        ct_slice = ct_volume[:, i_center, :]
        ax_ct.imshow(ct_slice, cmap='gray', origin='lower')
        ax_ct.set_title("CT sagittal slice (raw)")
        ax_ct.axis('off')

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=label_to_color[label], markersize=6,
               label=get_label_name(int(label), np.concatenate([raw_labels, assembled_labels])))
        for label in np.unique(np.concatenate([raw_labels, assembled_labels]))
    ]
    axes[1].legend(handles=handles, loc='upper right', frameon=True, fontsize=6)

    out_path = output_path.with_name(output_path.name + "_mesh_compare.png")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"✓ Saved mesh comparison PNG: {out_path}")


def save_combined_mesh_by_label(
    assembled_points: np.ndarray,
    vertebra_labels: np.ndarray,
    output_path: Path,
    label_to_color: dict,
):
    """
    Build one combined mesh with per-vertebra colors and save as a single file.
    """
    unique_labels = np.unique(vertebra_labels)
    meshes = []
    for label in unique_labels:
        points = assembled_points[vertebra_labels == label]
        if points.shape[0] < 50:
            continue
        cloud = pv.PolyData(points)
        try:
            mesh = cloud.delaunay_3d().extract_surface().clean().triangulate()
        except Exception as exc:
            print(f"WARNING: Mesh failed for label {label} ({exc}); using point cloud instead.")
            mesh = cloud
        if hasattr(mesh, "n_cells") and mesh.n_cells == 0:
            print(f"WARNING: Mesh empty for label {label}; using point cloud instead.")
            mesh = cloud
        color = (np.array(label_to_color[label]) * 255).astype(np.uint8)
        mesh['RGB'] = np.tile(color, (mesh.n_points, 1))
        meshes.append(mesh)

    if not meshes:
        print("WARNING: No meshes created; skipping combined mesh save.")
        return None

    combined = meshes[0]
    if len(meshes) > 1:
        combined = combined.merge(meshes[1:])

    mesh_path = output_path.with_name(output_path.name + "_mesh.ply")
    combined.save(str(mesh_path))
    print(f"✓ Saved combined vertebra mesh: {mesh_path}")
    return mesh_path


def _plot_sagittal_on_ax(
    ax,
    points: np.ndarray,
    labels: np.ndarray,
    axis: str,
    thickness: float,
    full_projection: bool,
    up_axis: str | None,
    flip_vertical: bool,
    label_to_color: dict,
    title: str,
):
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if axis not in axis_map:
        raise ValueError(f"Invalid sagittal axis: {axis}")
    axis_idx = axis_map[axis]

    if full_projection:
        mask = np.ones(len(points), dtype=bool)
    else:
        axis_vals = points[:, axis_idx]
        center = np.median(axis_vals)
        half = thickness / 2.0
        mask = np.abs(axis_vals - center) <= half
        if not np.any(mask):
            mask = np.ones(len(points), dtype=bool)

    slice_points = points[mask]
    slice_labels = labels[mask]

    axes = [0, 1, 2]
    axes.remove(axis_idx)
    if up_axis is not None:
        up_idx = axis_map[up_axis]
        if up_idx not in axes:
            raise ValueError(f"up_axis must be one of remaining axes: {axes}")
        ax_y = up_idx
        ax_x = axes[0] if axes[1] == ax_y else axes[1]
    else:
        ax_x, ax_y = axes

    colors = np.array([label_to_color[label] for label in slice_labels])

    pts2d = np.stack([slice_points[:, ax_x], slice_points[:, ax_y]], axis=1)
    pts2d_centered = pts2d - pts2d.mean(axis=0, keepdims=True)
    cov = np.cov(pts2d_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_vec = eigvecs[:, np.argmax(eigvals)]
    angle = np.arctan2(main_vec[0], main_vec[1])
    c, s = np.cos(-angle), np.sin(-angle)
    rot = np.array([[c, -s], [s, c]])
    pts2d_rot = (pts2d_centered @ rot.T)

    ax.scatter(
        pts2d_rot[:, 0],
        pts2d_rot[:, 1],
        s=2,
        c=colors,
        alpha=0.9,
        linewidths=0,
    )
    ax.set_aspect('equal', 'box')
    if flip_vertical:
        ax.invert_yaxis()
    ax.set_xlabel('Horizontal')
    ax.set_ylabel('Vertical')
    ax.set_title(title)
    ax.axis('off')


def save_sagittal_png(
    assembled_points: np.ndarray,
    vertebra_labels: np.ndarray,
    output_path: Path,
    axis: str = 'y',
    thickness: float = 5.0,
    dpi: int = 200,
    full_projection: bool = False,
    up_axis: str | None = None,
    flip_vertical: bool = False,
):
    """Save a sagittal 2D slice visualization as PNG."""
    import matplotlib.pyplot as plt

    label_to_color = get_label_colors(vertebra_labels)
    fig, ax = plt.subplots(figsize=(6, 8), dpi=dpi)
    if full_projection:
        title = f"Sagittal projection (axis={axis}, all points)"
    else:
        title = f"Sagittal slice (axis={axis}, thickness={thickness}mm)"
    _plot_sagittal_on_ax(
        ax,
        assembled_points,
        vertebra_labels,
        axis,
        thickness,
        full_projection,
        up_axis,
        flip_vertical,
        label_to_color,
        title,
    )

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=label_to_color[label], markersize=6,
               label=get_label_name(int(label), vertebra_labels))
        for label in np.unique(vertebra_labels)
    ]
    ax.legend(handles=handles, loc='upper right', frameon=True, fontsize=6)

    png_path = output_path.with_name(output_path.name + f"_sagittal_{axis}.png")
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"✓ Saved sagittal PNG: {png_path}")


def save_sagittal_compare_png(
    raw_points: np.ndarray,
    raw_labels: np.ndarray,
    assembled_points: np.ndarray,
    assembled_labels: np.ndarray,
    output_path: Path,
    axis: str = 'y',
    thickness: float = 5.0,
    full_projection: bool = False,
    up_axis: str | None = None,
    flip_vertical: bool = False,
):
    """Save side-by-side sagittal comparison (raw vs assembled)."""
    import matplotlib.pyplot as plt

    all_labels = np.concatenate([raw_labels, assembled_labels])
    label_to_color = get_label_colors(all_labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
    _plot_sagittal_on_ax(
        axes[0],
        raw_points,
        raw_labels,
        axis,
        thickness,
        full_projection,
        up_axis,
        flip_vertical,
        label_to_color,
        title="Raw segmentation (mask space)",
    )
    _plot_sagittal_on_ax(
        axes[1],
        assembled_points,
        assembled_labels,
        axis,
        thickness,
        full_projection,
        up_axis,
        flip_vertical,
        label_to_color,
        title="Assembled spine (model)",
    )

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=label_to_color[label], markersize=6,
               label=get_label_name(int(label), all_labels))
        for label in np.unique(all_labels)
    ]
    axes[1].legend(handles=handles, loc='upper right', frameon=True, fontsize=6)
    fig.tight_layout()
    png_path = output_path.with_name(output_path.name + f"_sagittal_compare_{axis}.png")
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"✓ Saved sagittal comparison PNG: {png_path}")


def save_sagittal_png_mesh(
    assembled_points: np.ndarray,
    output_path: Path,
    axis: str = 'y',
    thickness: float = 5.0,
    image_size: int = 1024,
    color_by_label: bool = False,
    vertebra_labels: np.ndarray | None = None,
    full_projection: bool = False,
    up_axis: str | None = None,
    flip_vertical: bool = False,
):
    """
    Save a sagittal slice PNG rendered from a mesh (PyVista).
    Falls back to point slice if mesh generation fails.
    """
    import matplotlib.pyplot as plt

    if not PYVISTA_AVAILABLE:
        raise RuntimeError("PyVista not available for mesh sagittal rendering")

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[axis]
    if full_projection:
        mask = np.ones(len(assembled_points), dtype=bool)
    else:
        axis_vals = assembled_points[:, axis_idx]
        center = np.median(axis_vals)
        half = thickness / 2.0
        mask = np.abs(axis_vals - center) <= half
        if not np.any(mask):
            print("WARNING: Mesh slice empty; falling back to point slice.")
            return False

    slice_points = assembled_points[mask]
    if slice_points.shape[0] < 200:
        print(f"WARNING: Too few slice points ({slice_points.shape[0]}); falling back to point slice.")
        return False

    # Project slice points to 2D plane and triangulate
    axes = [0, 1, 2]
    axes.remove(axis_idx)
    if up_axis is not None:
        up_idx = axis_map[up_axis]
        if up_idx not in axes:
            raise ValueError(f"up_axis must be one of remaining axes: {axes}")
        ax_v = up_idx
        ax_u = axes[0] if axes[1] == ax_v else axes[1]
    else:
        ax_u, ax_v = axes
    u = slice_points[:, ax_u]
    v = slice_points[:, ax_v]
    plane_points = np.stack([u, v, np.zeros_like(u)], axis=1)

    try:
        poly = pv.PolyData(plane_points)
        mesh2d = poly.delaunay_2d()
    except Exception as exc:
        print(f"WARNING: 2D mesh reconstruction failed ({exc}); falling back to point slice.")
        return False

    # Render off-screen
    png_path = output_path.with_name(output_path.name + f"_sagittal_{axis}_mesh.png")
    try:
        plotter = pv.Plotter(off_screen=True, window_size=(image_size, image_size))
        plotter.set_background("white")
        if color_by_label and vertebra_labels is not None:
            labels_slice = vertebra_labels[mask]
            unique_labels = np.unique(vertebra_labels)
            try:
                cmap = plt.get_cmap('tab20', len(unique_labels))
            except TypeError:
                cmap = plt.get_cmap('tab20')
            label_to_color = {label: cmap(i)[:3] for i, label in enumerate(unique_labels)}
            colors = np.array([label_to_color[label] for label in labels_slice])
            mesh2d['RGB'] = (colors * 255).astype(np.uint8)
            plotter.add_mesh(mesh2d, scalars='RGB', rgb=True, opacity=0.9)
        else:
            plotter.add_mesh(mesh2d, color="black")
        plotter.view_xy()
        if flip_vertical:
            plotter.camera.roll = 180
        plotter.show(screenshot=str(png_path), auto_close=True)
        print(f"✓ Saved sagittal mesh PNG: {png_path}")
        return True
    except Exception as exc:
        print(f"WARNING: Mesh render failed ({exc}); falling back to point slice.")
        return False


def visualize_assembled_spine(
    assembled_points: np.ndarray,  # (N, 3)
    vertebra_labels: np.ndarray,  # (N,)
    output_path: Path = None,
    show: bool = True,
    save_file: bool = True,
    file_format: str = 'ply',
):
    """
    Visualize assembled spine using PyVista.
    
    Args:
        assembled_points: (N, 3) point cloud
        vertebra_labels: (N,) vertebra type labels
        output_path: Path to save screenshot/file (optional)
        show: Whether to show interactive viewer
        save_file: Whether to save point cloud file (for remote viewing)
        file_format: File format to save ('ply', 'vtk', 'obj', or 'all')
    """
    # Save file if requested (for remote server)
    if save_file and output_path:
        save_assembled_spine(assembled_points, vertebra_labels, output_path, format=file_format)
    
    # Create PyVista point cloud
    point_cloud = pv.PolyData(assembled_points)
    point_cloud['vertebra_type'] = vertebra_labels
    
    # Create plotter
    plotter = pv.Plotter()
    
    # Color map for vertebra types
    # Use different colors for different types
    unique_labels = np.unique(vertebra_labels)
    colors = pv.colors.get_colormap('tab20', n_colors=len(unique_labels))
    
    # Add point cloud
    plotter.add_mesh(
        point_cloud,
        scalars='vertebra_type',
        point_size=3.0,
        render_points_as_spheres=True,
        cmap='tab20',
        show_scalar_bar=True,
        scalar_bar_args={'title': 'Vertebra Type'},
    )
    
    # Set background and camera
    plotter.set_background('white')
    plotter.add_axes(line_width=5, labels_off=False)
    
    # Add title
    plotter.add_text(
        'Assembled Spine Visualization',
        font_size=20,
        position='upper_left',
    )
    
    # Save screenshot if path provided
    if output_path and not save_file:
        screenshot_path = output_path.with_suffix('.png')
        plotter.screenshot(str(screenshot_path))
        print(f"✓ Screenshot saved to {screenshot_path}")
    
    # Show interactive viewer
    if show:
        print("Opening interactive 3D viewer...")
        print("Controls:")
        print("  - Left click + drag: Rotate")
        print("  - Right click + drag: Pan")
        print("  - Scroll: Zoom")
        print("  - Press 'q' to quit")
        plotter.show()
    
    return plotter


def main():
    parser = argparse.ArgumentParser(description='Visualize assembled spine')
    parser.add_argument('--encoder_path', type=str, required=True,
                        help='Path to encoder checkpoint')
    parser.add_argument('--assembly_path', type=str, required=True,
                        help='Path to assembly model checkpoint')
    parser.add_argument('--point_cloud_dir', type=str, required=True,
                        help='Directory containing point cloud data')
    parser.add_argument('--embedding_dir', type=str, default=None,
                        help='Directory with pre-extracted embeddings (optional, faster)')
    parser.add_argument('--subject_id', type=str, default=None,
                        help='Specific subject ID to visualize (if None, uses first available)')
    parser.add_argument('--split', type=str, default='test',
                        help='Data split (train/val/test)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save point cloud file (e.g., output.ply)')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not show interactive viewer (default for remote)')
    parser.add_argument('--save_format', type=str, default='ply',
                        choices=['ply', 'vtk', 'obj', 'all'],
                        help='File format to save (default: ply)')
    parser.add_argument('--save_sagittal', action='store_true',
                        help='Save sagittal 2D slice PNG')
    parser.add_argument('--sagittal_axis', type=str, default='y',
                        choices=['x', 'y', 'z'],
                        help='Axis to slice for sagittal view (default: y)')
    parser.add_argument('--sagittal_thickness', type=float, default=5.0,
                        help='Slice thickness in mm for sagittal view (default: 5.0)')
    parser.add_argument('--sagittal_mesh', action='store_true',
                        help='Render sagittal PNG from mesh slice (PyVista)')
    parser.add_argument('--sagittal_full', action='store_true', default=True,
                        help='Project all points in sagittal view (no slice)')
    parser.add_argument('--sagittal_color', action='store_true', default=True,
                        help='Color sagittal view by vertebra labels')
    parser.add_argument('--sagittal_up_axis', type=str, default='z',
                        choices=['x', 'y', 'z'],
                        help='Axis to use as vertical in sagittal view (default: z)')
    parser.add_argument('--sagittal_flip_vertical', action='store_true',
                        help='Flip sagittal vertical axis (invert up/down)')
    parser.add_argument('--mesh_dir', type=str, default=None,
                        help='Directory containing per-vertebra meshes (default: outputs/meshes)')
    parser.add_argument('--ct_dir', type=str, default=None,
                        help='Directory containing CT volumes (subject_id/ct_volume_1mm.npy)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    if not PYVISTA_AVAILABLE:
        print("ERROR: PyVista not available")
        return
    
    device = torch.device(args.device)
    mesh_dir = Path(args.mesh_dir) if args.mesh_dir else Path(args.embedding_dir).parent.parent / 'meshes'
    ct_dir = Path(args.ct_dir) if args.ct_dir else None
    
    # Load models
    encoder, assembly = load_models(
        Path(args.encoder_path),
        Path(args.assembly_path),
        device
    )
    
    # Load data
    print(f"\nLoading data...")
    
    if not args.embedding_dir:
        print("ERROR: --embedding_dir is required for visualization")
        return
    
    # Use pre-extracted embeddings (faster)
    print("Using pre-extracted embeddings...")
    dataset = AssemblyDataset(
        embedding_dir=Path(args.embedding_dir),
        point_cloud_dir=Path(args.point_cloud_dir),
        split=args.split,
        max_vertebrae=30,
        augment=False,
    )
    
    # Find subject
    if args.subject_id:
        # Find index of subject
        subject_idx = None
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample['subject_id'] == args.subject_id:
                subject_idx = i
                break
        if subject_idx is None:
            print(f"ERROR: Subject {args.subject_id} not found")
            return
        sample = dataset[subject_idx]
    else:
        # Use first sample
        sample = dataset[0]
        print(f"Using first available subject: {sample['subject_id']}")
    
    # Get embeddings and point clouds
    embeddings = sample['embeddings'].unsqueeze(0).to(device)  # (1, N, 512)
    pad_mask = ~sample['mask'].unsqueeze(0).to(device)  # (1, N)
    
    # Point clouds are already loaded in sample['points']
    points_tensor = sample['points']  # (N, M, 3)
    valid_mask = sample['mask'].cpu().numpy()  # (N,)
    
    # Convert to numpy and filter valid vertebrae
    point_clouds = []
    for i in range(len(valid_mask)):
        if valid_mask[i]:
            pc = points_tensor[i].cpu().numpy()  # (M, 3)
            # Remove padding (zero points)
            non_zero_mask = (pc != 0).any(axis=1)
            if non_zero_mask.sum() > 0:
                pc = pc[non_zero_mask]
            point_clouds.append(pc)
    
    print(f"Loaded {len(point_clouds)} vertebrae from subject {sample['subject_id']}")
    
    # Run assembly model
    print("\nRunning assembly model...")
    with torch.no_grad():
        predictions = assembly(embeddings, pad_mask=pad_mask, mask_mask=None)
    
    # Extract predictions
    ordering_logits = predictions['ordering'][0]  # (N, 27)
    ordering_pred = ordering_logits.argmax(dim=-1).cpu().numpy()  # (N,)
    poses = {
        't': predictions['pose']['t'][0].cpu().numpy(),  # (N, 3)
        'R': predictions['pose']['R'][0].cpu().numpy(),  # (N, 3, 3)
    }
    
    # Filter out padding
    valid_mask = sample['mask'].cpu().numpy()
    valid_indices = np.where(valid_mask)[0]
    
    # point_clouds is already filtered (only valid ones), so indices match
    ordering_pred_valid = ordering_pred[valid_indices]
    poses_valid = {
        't': poses['t'][valid_indices],
        'R': poses['R'][valid_indices],
    }
    
    print(f"Valid vertebrae: {len(valid_indices)}")
    print(f"Predicted types: {ordering_pred_valid}")
    
    # Reconstruct assembled spine with comprehensive checks
    print("\n" + "="*60)
    print("RECONSTRUCTION SANITY CHECKS")
    print("="*60)
    
    # Check C: Verify R is a proper rotation matrix (SO(3))
    print("\n[Check C] Verifying rotation matrices are valid SO(3)...")
    bad_rotations = []
    for i in range(len(poses_valid['R'])):
        R = poses_valid['R'][i]
        det = np.linalg.det(R)
        ortho = np.linalg.norm(R.T @ R - np.eye(3))
        if abs(det - 1) > 1e-2 or ortho > 1e-2:
            bad_rotations.append((i, det, ortho))
    
    if bad_rotations:
        print(f"  ⚠️ WARNING: Found {len(bad_rotations)} invalid rotation matrices:")
        for idx, det, ortho in bad_rotations[:5]:  # Show first 5
            print(f"    Vertebra {idx}: det={det:.6f}, ortho_error={ortho:.6f}")
    else:
        print("  ✓ All rotation matrices are valid SO(3)")
    
    # Debug: Print pose statistics
    print(f"\n[Debug] Pose statistics:")
    print(f"  Translation range: [{poses_valid['t'].min():.3f}, {poses_valid['t'].max():.3f}]")
    print(f"  Translation mean: {poses_valid['t'].mean(axis=0)}")
    print(f"  Translation std: {poses_valid['t'].std(axis=0)}")
    print(f"  First vertebra translation: {poses_valid['t'][0]}")
    print(f"  Point cloud local ranges:")
    for i, pc in enumerate(point_clouds):
        print(f"    Vertebra {i}: range [{pc.min(axis=0)}, {pc.max(axis=0)}], mean: {pc.mean(axis=0)}, std: {pc.std(axis=0)}")
    
    # Check B: Try both forward and inverse transforms
    print("\n[Check B] Trying both forward and inverse transforms...")
    
    # Method 1: Forward (x_global = R @ x_local + t)
    print("  Method 1 (Forward): x_global = R @ x_local + t")
    assembled_points_forward, _ = reconstruct_spine(
        point_clouds,
        poses_valid,
        ordering_pred_valid,
        use_inverse=False,
    )
    spread_forward = (assembled_points_forward.max(axis=0) - assembled_points_forward.min(axis=0)).max()
    print(f"    Spread: {spread_forward:.3f}")
    print(f"    Range: [{assembled_points_forward.min(axis=0)}, {assembled_points_forward.max(axis=0)}]")
    
    # Method 2: Inverse (x_global = (x_local - t) @ R^T)
    print("  Method 2 (Inverse): x_global = (x_local - t) @ R^T")
    assembled_points_inverse, _ = reconstruct_spine(
        point_clouds,
        poses_valid,
        ordering_pred_valid,
        use_inverse=True,
    )
    spread_inverse = (assembled_points_inverse.max(axis=0) - assembled_points_inverse.min(axis=0)).max()
    print(f"    Spread: {spread_inverse:.3f}")
    print(f"    Range: [{assembled_points_inverse.min(axis=0)}, {assembled_points_inverse.max(axis=0)}]")
    
    # Check A: Verify centroid alignment for both methods
    print("\n[Check A] Verifying centroid alignment...")
    
    # Compute expected centroids (assuming forward transform)
    centroids_local = np.array([pc.mean(axis=0) for pc in point_clouds])
    
    # Check if translation values are too small (problem indicator)
    translation_magnitudes = np.linalg.norm(poses_valid['t'], axis=1)
    point_cloud_sizes = np.array([np.linalg.norm([pc.max(axis=0) - pc.min(axis=0)]) for pc in point_clouds])
    print(f"  Translation magnitudes: {translation_magnitudes}")
    print(f"  Point cloud sizes: {point_cloud_sizes}")
    print(f"  Translation/Size ratio: {translation_magnitudes / (point_cloud_sizes + 1e-6)}")
    
    if np.all(translation_magnitudes < 0.1):
        print("  ⚠️ WARNING: Translation values are very small!")
        print("     This suggests vertebrae will overlap. Model may not have learned proper assembly.")
        print("     Expected: translation should be similar to point cloud size for proper separation.")
    
    centroids_global_expected_forward = np.array([
        (poses_valid['R'][i] @ centroids_local[i]) + poses_valid['t'][i]
        for i in range(len(point_clouds))
    ])
    
    # Compute actual centroids from reconstruction (forward)
    start = 0
    centroids_from_recon_forward = []
    for i, pc in enumerate(point_clouds):
        m = len(pc)
        centroids_from_recon_forward.append(assembled_points_forward[start:start+m].mean(axis=0))
        start += m
    centroids_from_recon_forward = np.array(centroids_from_recon_forward)
    
    centroid_diff_forward = np.linalg.norm(centroids_from_recon_forward - centroids_global_expected_forward, axis=1)
    print(f"  Forward method:")
    print(f"    Centroid diff mean: {centroid_diff_forward.mean():.6f}")
    print(f"    Centroid diff max: {centroid_diff_forward.max():.6f}")
    print(f"    Centroid diff std: {centroid_diff_forward.std():.6f}")
    
    # Check if centroids form a spine-like vertical alignment
    if len(centroids_from_recon_forward) > 1:
        # Check vertical spread (assuming spine is roughly vertical)
        vertical_spread_forward = centroids_from_recon_forward[:, 2].max() - centroids_from_recon_forward[:, 2].min()
        horizontal_spread_forward = np.sqrt(
            (centroids_from_recon_forward[:, 0].max() - centroids_from_recon_forward[:, 0].min())**2 +
            (centroids_from_recon_forward[:, 1].max() - centroids_from_recon_forward[:, 1].min())**2
        )
        print(f"    Vertical spread: {vertical_spread_forward:.3f}")
        print(f"    Horizontal spread: {horizontal_spread_forward:.3f}")
        print(f"    Vertical/Horizontal ratio: {vertical_spread_forward / (horizontal_spread_forward + 1e-6):.3f} (>1 = spine-like)")
    
    # Same for inverse
    centroids_global_expected_inverse = np.array([
        ((centroids_local[i] - poses_valid['t'][i]) @ poses_valid['R'][i].T)
        for i in range(len(point_clouds))
    ])
    
    start = 0
    centroids_from_recon_inverse = []
    for i, pc in enumerate(point_clouds):
        m = len(pc)
        centroids_from_recon_inverse.append(assembled_points_inverse[start:start+m].mean(axis=0))
        start += m
    centroids_from_recon_inverse = np.array(centroids_from_recon_inverse)
    
    centroid_diff_inverse = np.linalg.norm(centroids_from_recon_inverse - centroids_global_expected_inverse, axis=1)
    print(f"  Inverse method:")
    print(f"    Centroid diff mean: {centroid_diff_inverse.mean():.6f}")
    print(f"    Centroid diff max: {centroid_diff_inverse.max():.6f}")
    print(f"    Centroid diff std: {centroid_diff_inverse.std():.6f}")
    
    if len(centroids_from_recon_inverse) > 1:
        vertical_spread_inverse = centroids_from_recon_inverse[:, 2].max() - centroids_from_recon_inverse[:, 2].min()
        horizontal_spread_inverse = np.sqrt(
            (centroids_from_recon_inverse[:, 0].max() - centroids_from_recon_inverse[:, 0].min())**2 +
            (centroids_from_recon_inverse[:, 1].max() - centroids_from_recon_inverse[:, 1].min())**2
        )
        print(f"    Vertical spread: {vertical_spread_inverse:.3f}")
        print(f"    Horizontal spread: {horizontal_spread_inverse:.3f}")
        print(f"    Vertical/Horizontal ratio: {vertical_spread_inverse / (horizontal_spread_inverse + 1e-6):.3f} (>1 = spine-like)")
    
    # Choose best method based on:
    # 1. Centroid alignment (should be near 0)
    # 2. Spine-like shape (vertical/horizontal ratio > 1)
    # 3. Reasonable spread
    print("\n[Decision] Choosing best reconstruction method...")
    
    forward_score = 0
    inverse_score = 0
    
    # Score 1: Centroid alignment (lower is better)
    if centroid_diff_forward.mean() < centroid_diff_inverse.mean():
        forward_score += 1
        print("  Forward: Better centroid alignment")
    else:
        inverse_score += 1
        print("  Inverse: Better centroid alignment")
    
    # Score 2: Spine-like shape (vertical/horizontal ratio > 1)
    if len(centroids_from_recon_forward) > 1 and len(centroids_from_recon_inverse) > 1:
        vh_ratio_forward = vertical_spread_forward / (horizontal_spread_forward + 1e-6)
        vh_ratio_inverse = vertical_spread_inverse / (horizontal_spread_inverse + 1e-6)
        if vh_ratio_forward > vh_ratio_inverse:
            forward_score += 1
            print(f"  Forward: Better spine-like shape (V/H ratio: {vh_ratio_forward:.3f} > {vh_ratio_inverse:.3f})")
        else:
            inverse_score += 1
            print(f"  Inverse: Better spine-like shape (V/H ratio: {vh_ratio_inverse:.3f} > {vh_ratio_forward:.3f})")
    
    # Score 3: Reasonable spread (not too small, not too large)
    if 1.0 < spread_forward < 1000.0 and 1.0 < spread_inverse < 1000.0:
        # Both reasonable, prefer larger spread (more separated vertebrae)
        if spread_forward > spread_inverse:
            forward_score += 1
        else:
            inverse_score += 1
    elif 1.0 < spread_forward < 1000.0:
        forward_score += 1
    elif 1.0 < spread_inverse < 1000.0:
        inverse_score += 1
    
    # Final decision
    if forward_score >= inverse_score:
        print(f"\n  ✓ Using FORWARD transform (score: {forward_score} vs {inverse_score})")
        assembled_points = assembled_points_forward
        use_inverse_final = False
    else:
        print(f"\n  ✓ Using INVERSE transform (score: {inverse_score} vs {forward_score})")
        assembled_points = assembled_points_inverse
        use_inverse_final = True
    
    # Final reconstruction with chosen method
    assembled_points, vertebra_labels = reconstruct_spine(
        point_clouds,
        poses_valid,
        ordering_pred_valid,
        use_inverse=use_inverse_final,
    )
    
    print(f"\n[Final] Assembled spine statistics:")
    print(f"  Total points: {len(assembled_points)}")
    print(f"  Range: [{assembled_points.min(axis=0)}, {assembled_points.max(axis=0)}]")
    print(f"  Mean: {assembled_points.mean(axis=0)}")
    print(f"  Std: {assembled_points.std(axis=0)}")
    print(f"  Spread: {(assembled_points.max(axis=0) - assembled_points.min(axis=0)).max():.3f}")
    print("="*60)
    
    print(f"Total points: {len(assembled_points)}")
    print(f"Unique vertebra types: {np.unique(vertebra_labels)}")
    print(f"Number of vertebrae: {len(point_clouds)}")
    
    # Save point cloud file (for remote viewing)
    print(f"\nSaving assembled spine for subject {sample['subject_id']}...")
    print(f"  - {len(point_clouds)} vertebrae assembled into one point cloud")
    print(f"  - Total points: {len(assembled_points)}")
    
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        # Default output path with subject ID
        output_dir = Path(args.embedding_dir).parent / 'visualization'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"assembled_spine_{sample['subject_id']}"
    
    # Save file (for remote server - no display)
    saved_files, label_to_color = save_assembled_spine(
        assembled_points,
        vertebra_labels,
        output_path,
        format=args.save_format,
    )

    # Save combined mesh with per-vertebra colors
    save_combined_mesh_by_label(
        assembled_points,
        vertebra_labels,
        output_path,
        label_to_color,
    )

    if args.save_sagittal:
        if args.sagittal_mesh:
            ok = save_sagittal_png_mesh(
                assembled_points,
                output_path,
                axis=args.sagittal_axis,
                thickness=args.sagittal_thickness,
                color_by_label=args.sagittal_color,
                vertebra_labels=vertebra_labels,
                full_projection=args.sagittal_full,
                up_axis=args.sagittal_up_axis,
                flip_vertical=args.sagittal_flip_vertical,
            )
            if not ok:
                save_sagittal_png(
                    assembled_points,
                    vertebra_labels,
                    output_path,
                    axis=args.sagittal_axis,
                    thickness=args.sagittal_thickness,
                    full_projection=args.sagittal_full,
                    up_axis=args.sagittal_up_axis,
                    flip_vertical=args.sagittal_flip_vertical,
                )
        else:
            save_sagittal_png(
                assembled_points,
                vertebra_labels,
                output_path,
                axis=args.sagittal_axis,
                thickness=args.sagittal_thickness,
                full_projection=args.sagittal_full,
                up_axis=args.sagittal_up_axis,
                flip_vertical=args.sagittal_flip_vertical,
            )

        # Also save before/after comparison (raw segmentation vs assembled)
        vertebra_ids_valid = sample['vertebra_ids'][sample['mask']].cpu().numpy()
        raw_points = []
        raw_labels = []
        for i, pc in enumerate(point_clouds):
            raw_points.append(pc)
            raw_labels.append(np.full(len(pc), int(vertebra_ids_valid[i])))
        raw_points = np.concatenate(raw_points, axis=0)
        raw_labels = np.concatenate(raw_labels, axis=0)

        save_sagittal_compare_png(
            raw_points,
            raw_labels,
            assembled_points,
            vertebra_labels,
            output_path,
            axis=args.sagittal_axis,
            thickness=args.sagittal_thickness,
            full_projection=args.sagittal_full,
            up_axis=args.sagittal_up_axis,
            flip_vertical=args.sagittal_flip_vertical,
        )

        # Mesh-based before/after comparison (if mesh files exist)
        ct_volume = None
        ct_path = None
        if ct_dir:
            ct_volume, ct_path = load_ct_volume(ct_dir, sample['subject_id'])
            if ct_volume is None:
                print(f"WARNING: CT not found for {sample['subject_id']} in {ct_dir}")
            else:
                print(f"✓ Loaded CT for {sample['subject_id']}: {ct_path} shape={ct_volume.shape}")

        save_mesh_compare_png(
            mesh_dir,
            sample['subject_id'],
            vertebra_ids_valid,
            ordering_pred_valid,
            poses_valid,
            output_path,
            ct_volume=ct_volume,
        )
    
    # Visualize if requested (only works with display)
    if not args.no_show:
        print("\nCreating interactive visualization...")
        visualize_assembled_spine(
            assembled_points,
            vertebra_labels,
            output_path=output_path,
            show=True,
            save_file=False,  # Already saved above
        )
    
    print("\n✓ Visualization complete!")
    print(f"\n📁 Files saved to: {output_path.parent}")
    print(f"   - Point cloud: {output_path.with_suffix('.ply')}")
    print(f"   - Metadata: {output_path.with_suffix('.json')}")
    if args.save_sagittal:
        if args.sagittal_mesh:
            print(f"   - Sagittal PNG: {output_path.with_name(output_path.name + f'_sagittal_{args.sagittal_axis}_mesh.png')}")
        else:
            print(f"   - Sagittal PNG: {output_path.with_name(output_path.name + f'_sagittal_{args.sagittal_axis}.png')}")
    print(f"\n💡 To view locally, use PyVista:")
    print(f"   import pyvista as pv")
    print(f"   mesh = pv.read('{output_path.with_suffix('.ply')}')")
    print(f"   mesh.plot(scalars='vertebra_type', cmap='tab20')")


if __name__ == '__main__':
    main()

