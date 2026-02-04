"""
Phase 1 Visualization Sanity Check

Visualize:
1. Mesh extraction results
2. Point cloud sampling results
3. Feature computation (normals, curvature)
4. Multiple samples for verification
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.geometry import sample_point_cloud
from utils.features import compute_surface_normals, compute_curvature

# For loading original mask files
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

# VerSe utilities for preprocessing
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'verse' / 'utils'))
    from data_utilities import resample_nib, reorient_to
    VERSE_UTILS_AVAILABLE = True
except ImportError:
    VERSE_UTILS_AVAILABLE = False


def visualize_mesh(vertices, faces, title="Mesh", ax=None):
    """Visualize mesh in 3D"""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot mesh
    mesh = Poly3DCollection(vertices[faces], alpha=0.3, facecolor='cyan', edgecolor='blue', linewidths=0.5)
    ax.add_collection3d(mesh)
    
    # Set limits
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    return ax


def visualize_point_cloud(points, normals=None, curvature=None, title="Point Cloud", ax=None):
    """Visualize point cloud with optional normals and curvature"""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Color by curvature if available
    if curvature is not None:
        # Use k1 for coloring
        colors = curvature[:, 0] if curvature.shape[1] >= 1 else np.zeros(len(points))
        colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-6)
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                           c=colors, cmap='viridis', s=1, alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Curvature (k1)')
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.6)
    
    # Plot normals if available
    if normals is not None:
        # Sample normals for visualization (every 10th point)
        sample_indices = np.arange(0, len(points), max(1, len(points) // 100))
        sample_points = points[sample_indices]
        sample_normals = normals[sample_indices]
        
        # Normalize normals for visualization
        normal_length = np.linalg.norm(points, axis=1).max() * 0.05
        sample_normals = sample_normals / (np.linalg.norm(sample_normals, axis=1, keepdims=True) + 1e-6) * normal_length
        
        for i in range(len(sample_points)):
            ax.plot([sample_points[i, 0], sample_points[i, 0] + sample_normals[i, 0]],
                   [sample_points[i, 1], sample_points[i, 1] + sample_normals[i, 1]],
                   [sample_points[i, 2], sample_points[i, 2] + sample_normals[i, 2]],
                   'r-', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    return ax


def visualize_features_2d(points, normals, curvature, title="Features"):
    """Visualize features in 2D projections"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)
    
    # Projection 1: XY plane
    ax = axes[0, 0]
    if curvature is not None:
        colors = curvature[:, 0]
        colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-6)
        ax.scatter(points[:, 0], points[:, 1], c=colors, cmap='viridis', s=1, alpha=0.6)
    else:
        ax.scatter(points[:, 0], points[:, 1], s=1, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('XY Projection')
    ax.set_aspect('equal')
    
    # Projection 2: XZ plane
    ax = axes[0, 1]
    if curvature is not None:
        ax.scatter(points[:, 0], points[:, 2], c=colors, cmap='viridis', s=1, alpha=0.6)
    else:
        ax.scatter(points[:, 0], points[:, 2], s=1, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('XZ Projection')
    ax.set_aspect('equal')
    
    # Projection 3: YZ plane
    ax = axes[1, 0]
    if curvature is not None:
        ax.scatter(points[:, 1], points[:, 2], c=colors, cmap='viridis', s=1, alpha=0.6)
    else:
        ax.scatter(points[:, 1], points[:, 2], s=1, alpha=0.6)
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_title('YZ Projection')
    ax.set_aspect('equal')
    
    # Feature statistics
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"Point Cloud Statistics:\n"
    stats_text += f"  Points: {len(points)}\n"
    stats_text += f"  Bounding box: [{points.min(axis=0)}]\n"
    stats_text += f"                [{points.max(axis=0)}]\n"
    if normals is not None:
        normal_norms = np.linalg.norm(normals, axis=1)
        stats_text += f"\nNormals:\n"
        stats_text += f"  Mean norm: {normal_norms.mean():.4f}\n"
        stats_text += f"  Std norm: {normal_norms.std():.4f}\n"
    if curvature is not None:
        stats_text += f"\nCurvature:\n"
        stats_text += f"  k1: min={curvature[:, 0].min():.4f}, max={curvature[:, 0].max():.4f}, mean={curvature[:, 0].mean():.4f}\n"
        stats_text += f"  k2: min={curvature[:, 1].min():.4f}, max={curvature[:, 1].max():.4f}, mean={curvature[:, 1].mean():.4f}\n"
    ax.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    return fig


def load_paths_from_csv(csv_path, subject_csv_path=None):
    """Load paths from CSV file
    
    Args:
        csv_path: Primary CSV file (can be vertebra-level or subject-level)
        subject_csv_path: Optional subject-level CSV for CT paths (if csv_path is vertebra-level)
    """
    import pandas as pd
    
    if not Path(csv_path).exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        # Check if this is vertebra-level CSV (has vertebra_id column)
        is_vertebra_csv = 'vertebra_id' in df.columns
        
        # If vertebra-level CSV, try to load subject-level CSV for CT paths
        subject_paths_dict = {}
        if is_vertebra_csv and subject_csv_path is None:
            # Try to find subject-level CSV in the same directory
            csv_dir = Path(csv_path).parent
            subject_csv_candidates = [
                csv_dir / 'preprocessed_data_subject_with_paths.csv',
                csv_dir / 'preprocessed_data_subject.csv',
            ]
            for candidate in subject_csv_candidates:
                if candidate.exists():
                    try:
                        subject_df = pd.read_csv(candidate)
                        for _, row in subject_df.iterrows():
                            subj_id = row.get('subject_id', '')
                            if pd.isna(subj_id) or subj_id == '':
                                continue
                            ct_path_str = row.get('processed_ct_1mm_path', '')
                            if pd.isna(ct_path_str) or ct_path_str == '':
                                ct_path_str = ''
                            else:
                                ct_path_str = str(ct_path_str).strip()
                            if ct_path_str:
                                subject_paths_dict[subj_id] = ct_path_str
                        print(f"  ✓ Loaded CT paths from subject-level CSV: {candidate}")
                        break
                    except Exception as e:
                        print(f"  ⚠ Failed to load subject CSV {candidate}: {e}")
        
        # Also try explicit subject_csv_path if provided
        if subject_csv_path and Path(subject_csv_path).exists():
            try:
                subject_df = pd.read_csv(subject_csv_path)
                for _, row in subject_df.iterrows():
                    subj_id = row.get('subject_id', '')
                    if pd.isna(subj_id) or subj_id == '':
                        continue
                    ct_path_str = row.get('processed_ct_1mm_path', '')
                    if pd.isna(ct_path_str) or ct_path_str == '':
                        ct_path_str = ''
                    else:
                        ct_path_str = str(ct_path_str).strip()
                    if ct_path_str:
                        subject_paths_dict[subj_id] = ct_path_str
                print(f"  ✓ Loaded CT paths from explicit subject CSV: {subject_csv_path}")
            except Exception as e:
                print(f"  ⚠ Failed to load explicit subject CSV {subject_csv_path}: {e}")
        
        # Create a dictionary mapping subject_id to paths
        paths_dict = {}
        for _, row in df.iterrows():
            subject_id = row.get('subject_id', '')
            if pd.isna(subject_id) or subject_id == '':
                continue
            
            # Get paths, handle NaN values
            mask_path_str = row.get('processed_mask_1mm_path', '')
            ct_path_str = row.get('processed_ct_1mm_path', '')
            has_mask = row.get('has_mask', False)
            dataset = row.get('dataset', '')
            
            # If vertebra-level CSV, try to get CT path from subject-level CSV
            if is_vertebra_csv and subject_id in subject_paths_dict:
                ct_path_str = subject_paths_dict[subject_id]
            
            # Convert NaN/None to empty string
            if pd.isna(mask_path_str) or mask_path_str is None:
                mask_path_str = ''
            else:
                mask_path_str = str(mask_path_str).strip()
            
            if pd.isna(ct_path_str) or ct_path_str is None:
                ct_path_str = ''
            else:
                ct_path_str = str(ct_path_str).strip()
            
            # If has_mask is True but path is empty, try to construct path
            if has_mask and not mask_path_str:
                # Try to construct path from dataset and subject_id
                csv_dir = Path(csv_path).parent
                possible_mask_path = csv_dir / dataset / subject_id / 'mask_volume_1mm.npy'
                if possible_mask_path.exists():
                    mask_path_str = str(possible_mask_path)
            
            # If CT path is empty, try to construct path
            if not ct_path_str:
                csv_dir = Path(csv_path).parent
                # Try multiple possible CT file names
                possible_ct_paths = [
                    csv_dir / dataset / subject_id / 'ct_volume_1mm.npy',
                    csv_dir / dataset / subject_id / 'processed_ct_1mm.npy',
                ]
                for possible_ct_path in possible_ct_paths:
                    if possible_ct_path.exists():
                        ct_path_str = str(possible_ct_path)
                        break
            
            # Initialize or update paths_dict
            if subject_id not in paths_dict:
                paths_dict[subject_id] = {
                    'mask_path': mask_path_str,
                    'ct_path': ct_path_str,
                    'has_mask': bool(has_mask),
                    'dataset': str(dataset) if not pd.isna(dataset) else '',
                }
            else:
                # Update if we have better information
                if ct_path_str and not paths_dict[subject_id]['ct_path']:
                    paths_dict[subject_id]['ct_path'] = ct_path_str
                if mask_path_str and not paths_dict[subject_id]['mask_path']:
                    paths_dict[subject_id]['mask_path'] = mask_path_str
        
        return paths_dict
    except Exception as e:
        print(f"Warning: Failed to load CSV: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_original_mask(mask_dir, subject_id, paths_dict=None):
    """Load original mask volume"""
    # First try CSV paths if available
    if paths_dict and subject_id in paths_dict:
        mask_path_str = paths_dict[subject_id].get('mask_path', '')
        has_mask = paths_dict[subject_id].get('has_mask', False)
        
        if mask_path_str and mask_path_str != '' and mask_path_str != 'nan' and isinstance(mask_path_str, str):
            mask_file = Path(mask_path_str)
            if mask_file.exists():
                try:
                    mask = np.load(mask_file)  # (P, I, L)
                    return mask, mask_file
                except Exception as e:
                    print(f"  Warning: Failed to load mask from {mask_file}: {e}")
    
    # Fallback: Try to find mask_volume_1mm.npy in processed directory
    if mask_dir.exists():
        for dataset_dir in mask_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            mask_file = dataset_dir / subject_id / 'mask_volume_1mm.npy'
            if mask_file.exists():
                try:
                    mask = np.load(mask_file)  # (P, I, L)
                    return mask, mask_file
                except Exception as e:
                    pass
    
    return None, None


def load_original_ct(mask_dir, subject_id, paths_dict=None):
    """Load original CT volume"""
    # First try CSV paths if available
    if paths_dict and subject_id in paths_dict:
        # Try processed_ct_1mm_path first
        ct_path_str = paths_dict[subject_id].get('ct_path', '')
        if not ct_path_str or ct_path_str == '' or ct_path_str == 'nan':
            ct_path_str = paths_dict[subject_id].get('processed_ct_1mm_path', '')
        
        if ct_path_str and ct_path_str != '' and ct_path_str != 'nan':
            ct_file = Path(ct_path_str)
            if ct_file.exists():
                try:
                    ct = np.load(ct_file)  # (P, I, L)
                    return ct, ct_file
                except Exception as e:
                    print(f"  Warning: Failed to load CT from {ct_file}: {e}")
        else:
            # Try to construct path from dataset
            dataset = paths_dict[subject_id].get('dataset', '')
            if dataset:
                csv_dir = Path(mask_dir) if mask_dir else None
                if csv_dir and csv_dir.exists():
                    possible_ct_paths = [
                        csv_dir / dataset / subject_id / 'ct_volume_1mm.npy',
                        csv_dir / dataset / subject_id / 'processed_ct_1mm.npy',
                    ]
                    for ct_file in possible_ct_paths:
                        if ct_file.exists():
                            try:
                                ct = np.load(ct_file)  # (P, I, L)
                                return ct, ct_file
                            except Exception as e:
                                print(f"  Warning: Failed to load CT from {ct_file}: {e}")
    
    # Fallback: Try to find CT files in processed directory
    if mask_dir and Path(mask_dir).exists():
        for dataset_dir in Path(mask_dir).iterdir():
            if not dataset_dir.is_dir():
                continue
            subject_dir = dataset_dir / subject_id
            if not subject_dir.exists():
                continue
            
            # Try multiple possible CT file names
            ct_candidates = [
                'ct_volume_1mm.npy',
                'processed_ct_1mm.npy',
                'ct_volume.npy',
            ]
            for ct_filename in ct_candidates:
                ct_file = subject_dir / ct_filename
                if ct_file.exists():
                    try:
                        ct = np.load(ct_file)  # (P, I, L)
                        return ct, ct_file
                    except Exception as e:
                        print(f"  Warning: Failed to load CT from {ct_file}: {e}")
    
    return None, None


def visualize_ct_with_vertebra(ct, mask, vertebra_id, title="CT with Vertebra", ax=None):
    """Visualize CT image with vertebra mask overlay using bbox-based ROI cropping"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Extract vertebra mask
    vertebra_mask = (mask == vertebra_id).astype(np.float32)
    
    # Find bounding box
    coords = np.where(vertebra_mask > 0)
    if len(coords[0]) == 0:
        ax.text(0.5, 0.5, 'No mask found', ha='center', va='center', transform=ax.transAxes)
        return ax
    
    p_min, p_max = coords[0].min(), coords[0].max()
    i_min, i_max = coords[1].min(), coords[1].max()
    l_min, l_max = coords[2].min(), coords[2].max()
    
    # Calculate bbox dimensions
    p_size = p_max - p_min + 1
    i_size = i_max - i_min + 1
    l_size = l_max - l_min + 1
    
    # Calculate 1.5x expanded ROI size (symmetrical)
    p_roi_size = int(p_size * 1.5)
    i_roi_size = int(i_size * 1.5)
    l_roi_size = int(l_size * 1.5)
    
    # Calculate center of bbox
    p_center = (p_min + p_max) // 2
    i_center = (i_min + i_max) // 2
    l_center = (l_min + l_max) // 2
    
    # Calculate ROI bounds (symmetrical around center)
    p_start = max(0, p_center - p_roi_size // 2)
    p_end = min(ct.shape[0], p_center + p_roi_size // 2 + 1)
    i_start = max(0, i_center - i_roi_size // 2)
    i_end = min(ct.shape[1], i_center + i_roi_size // 2 + 1)
    l_start = max(0, l_center - l_roi_size // 2)
    l_end = min(ct.shape[2], l_center + l_roi_size // 2 + 1)
    
    # Crop CT and mask to ROI
    ct_roi = ct[p_start:p_end, i_start:i_end, l_start:l_end]
    mask_roi = vertebra_mask[p_start:p_end, i_start:i_end, l_start:l_end]
    
    # Get center slice in cropped ROI
    p_roi_center = (p_start + p_end) // 2 - p_start
    
    # Use axial view (P slice) - most common view
    ct_slice = ct_roi[p_roi_center, :, :]
    mask_slice = mask_roi[p_roi_center, :, :]
    
    # Display CT with mask overlay
    ax.imshow(ct_slice, cmap='gray', origin='lower', alpha=1.0)
    ax.imshow(mask_slice, cmap='Reds', alpha=0.3, origin='lower')
    ax.set_title(f'{title}\nAxial Slice (P={p_center}, ROI: {i_start}-{i_end}, {l_start}-{l_end})')
    ax.set_xlabel('L')
    ax.set_ylabel('I')
    
    return ax


def visualize_mask_slices(mask, vertebra_id, title="Original Mask", fig=None, axes=None):
    """Visualize mask slices in 3 views"""
    if fig is None:
        fig = plt.figure(figsize=(15, 5))
        axes = [fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)]
    
    # Extract vertebra mask
    vertebra_mask = (mask == vertebra_id).astype(np.float32)
    
    # Find bounding box
    coords = np.where(vertebra_mask > 0)
    if len(coords[0]) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, 'No mask found', ha='center', va='center', transform=ax.transAxes)
        return fig, axes
    
    p_min, p_max = coords[0].min(), coords[0].max()
    i_min, i_max = coords[1].min(), coords[1].max()
    l_min, l_max = coords[2].min(), coords[2].max()
    
    # Get center slices
    p_center = (p_min + p_max) // 2
    i_center = (i_min + i_max) // 2
    l_center = (l_min + l_max) // 2
    
    # Axial view (P slice)
    ax = axes[0]
    slice_axial = vertebra_mask[p_center, :, :]
    ax.imshow(slice_axial, cmap='gray', origin='lower')
    ax.set_title(f'Axial (P={p_center})\nVertebra {vertebra_id}')
    ax.set_xlabel('L')
    ax.set_ylabel('I')
    
    # Sagittal view (I slice)
    ax = axes[1]
    slice_sagittal = vertebra_mask[:, i_center, :]
    ax.imshow(slice_sagittal, cmap='gray', origin='lower')
    ax.set_title(f'Sagittal (I={i_center})\nVertebra {vertebra_id}')
    ax.set_xlabel('L')
    ax.set_ylabel('P')
    
    # Coronal view (L slice)
    ax = axes[2]
    slice_coronal = vertebra_mask[:, :, l_center]
    ax.imshow(slice_coronal, cmap='gray', origin='lower')
    ax.set_title(f'Coronal (L={l_center})\nVertebra {vertebra_id}')
    ax.set_xlabel('I')
    ax.set_ylabel('P')
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    return fig, axes


def check_sample(mesh_dir, pc_dir, mask_dir, subject_id, vertebra_id, output_dir, paths_dict=None):
    """Check a single sample"""
    print(f"\n{'='*60}")
    print(f"Checking: {subject_id} - Vertebra {vertebra_id}")
    print(f"{'='*60}")
    
    # Load mesh
    mesh_file = mesh_dir / subject_id / f'vertebra_{vertebra_id}_mesh.npz'
    if not mesh_file.exists():
        print(f"  ✗ Mesh file not found: {mesh_file}")
        return False
    
    mesh_data = np.load(mesh_file)
    vertices = mesh_data['vertices']
    faces = mesh_data['faces']
    print(f"  ✓ Mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Load point cloud
    pc_file = pc_dir / subject_id / f'vertebra_{vertebra_id}_points.npy'
    if not pc_file.exists():
        print(f"  ✗ Point cloud file not found: {pc_file}")
        return False
    
    points = np.load(pc_file)
    print(f"  ✓ Point cloud: {points.shape}")
    
    # Load features
    normals_file = pc_dir / subject_id / f'vertebra_{vertebra_id}_normals.npy'
    feature_file = pc_dir / subject_id / f'vertebra_{vertebra_id}_features.npz'
    curvature_file = pc_dir / subject_id / f'vertebra_{vertebra_id}_curvature.npz'
    
    normals = None
    curvature = None
    
    if normals_file.exists():
        normals = np.load(normals_file)
        print(f"  ✓ Normals: {normals.shape}")
    elif feature_file.exists():
        features_data = np.load(feature_file)
        if 'normals' in features_data:
            normals = features_data['normals']
            print(f"  ✓ Normals (from features.npz): {normals.shape}")
    
    if curvature_file.exists():
        curvature_data = np.load(curvature_file)
        k1 = curvature_data['k1']
        k2 = curvature_data['k2']
        curvature = np.stack([k1, k2], axis=-1)
        print(f"  ✓ Curvature: k1={k1.shape}, k2={k2.shape}")
    elif feature_file.exists():
        features_data = np.load(feature_file)
        if 'k1' in features_data and 'k2' in features_data:
            k1 = features_data['k1']
            k2 = features_data['k2']
            curvature = np.stack([k1, k2], axis=-1)
            print(f"  ✓ Curvature (from features.npz): k1={k1.shape}, k2={k2.shape}")
    
    # Load original mask and CT for comparison
    original_mask, mask_file = load_original_mask(mask_dir, subject_id, paths_dict)
    original_ct, ct_file = load_original_ct(mask_dir, subject_id, paths_dict)
    
    if original_mask is not None:
        print(f"  ✓ Original mask loaded: {original_mask.shape} from {mask_file.name}")
    else:
        print(f"  ⚠ Original mask not found")
        if paths_dict and subject_id in paths_dict:
            print(f"    CSV entry: mask_path={paths_dict[subject_id].get('mask_path', 'N/A')}")
    
    if original_ct is not None:
        print(f"  ✓ Original CT loaded: {original_ct.shape} from {ct_file.name}")
    else:
        print(f"  ⚠ Original CT not found")
        if paths_dict and subject_id in paths_dict:
            print(f"    CSV entry: ct_path={paths_dict[subject_id].get('ct_path', 'N/A')}")
            print(f"    CSV entry: processed_ct_1mm_path={paths_dict[subject_id].get('processed_ct_1mm_path', 'N/A')}")
            print(f"    CSV entry: dataset={paths_dict[subject_id].get('dataset', 'N/A')}")
        if mask_dir:
            print(f"    Tried fallback in: {mask_dir}")
    
    # Create visualizations
    output_subject_dir = output_dir / subject_id
    output_subject_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive visualization with original mask
    if original_mask is not None:
        # Large figure with original mask + mesh + point cloud
        fig = plt.figure(figsize=(18, 12))
        
        # Row 1: Original mask slices (3 views)
        ax_mask1 = fig.add_subplot(231)
        ax_mask2 = fig.add_subplot(232)
        ax_mask3 = fig.add_subplot(233)
        visualize_mask_slices(original_mask, vertebra_id, 
                              f"Original Mask: {subject_id} - Vertebra {vertebra_id}",
                              fig=fig, axes=[ax_mask1, ax_mask2, ax_mask3])
        
        # Row 2: Mesh and Point Cloud
        ax1 = fig.add_subplot(234, projection='3d')
        visualize_mesh(vertices, faces, f"Mesh: V{vertebra_id}", ax1)
        
        ax2 = fig.add_subplot(235, projection='3d')
        visualize_point_cloud(points, normals=normals, curvature=curvature, 
                             title=f"Point Cloud: V{vertebra_id}", ax=ax2)
        
        # Row 2, Col 3: Info panel
        ax_info = fig.add_subplot(236)
        ax_info.axis('off')
        info_text = f"Subject: {subject_id}\n"
        info_text += f"Vertebra ID: {vertebra_id}\n\n"
        info_text += f"Original Mask:\n"
        info_text += f"  Shape: {original_mask.shape} (P, I, L)\n"
        info_text += f"  Voxels: {(original_mask == vertebra_id).sum()}\n\n"
        info_text += f"Mesh:\n"
        info_text += f"  Vertices: {len(vertices)}\n"
        info_text += f"  Faces: {len(faces)}\n\n"
        info_text += f"Point Cloud:\n"
        info_text += f"  Points: {len(points)}\n"
        if normals is not None:
            info_text += f"  Normals: ✓\n"
        if curvature is not None:
            info_text += f"  Curvature: ✓\n"
        ax_info.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center', 
                    family='monospace', transform=ax_info.transAxes)
        
        plt.suptitle(f"{subject_id} - Vertebra {vertebra_id} Comparison", fontsize=16, y=0.995)
        plt.tight_layout()
        plt.savefig(output_subject_dir / f'vertebra_{vertebra_id}_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved comparison: {output_subject_dir / f'vertebra_{vertebra_id}_comparison.png'}")
    
    # Original visualization (mesh + point cloud + features)
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(221, projection='3d')
    visualize_mesh(vertices, faces, f"Mesh: {subject_id} V{vertebra_id}", ax1)
    
    # 2. Point cloud visualization
    ax2 = fig.add_subplot(222, projection='3d')
    visualize_point_cloud(points, normals=normals, curvature=curvature, 
                         title=f"Point Cloud: {subject_id} V{vertebra_id}", ax=ax2)
    
    # 3. Original CT with vertebra mask overlay (instead of normal distribution)
    ax3 = fig.add_subplot(223)
    if original_ct is not None and original_mask is not None:
        visualize_ct_with_vertebra(original_ct, original_mask, vertebra_id, 
                                  f"Original CT: V{vertebra_id}", ax=ax3)
    else:
        ax3.text(0.5, 0.5, 'CT/Mask not available', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Original CT with Vertebra')
    
    ax4 = fig.add_subplot(224)
    if curvature is not None:
        # Curvature distribution
        ax4.scatter(curvature[:, 0], curvature[:, 1], alpha=0.5, s=1)
        ax4.set_xlabel('k1 (Principal Curvature 1)')
        ax4.set_ylabel('k2 (Principal Curvature 2)')
        ax4.set_title('Curvature Distribution')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_subject_dir / f'vertebra_{vertebra_id}_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved visualization: {output_subject_dir / f'vertebra_{vertebra_id}_visualization.png'}")
    
    # 4. Features 2D projections
    if normals is not None and curvature is not None:
        fig = visualize_features_2d(points, normals, curvature, 
                                     f"Features: {subject_id} V{vertebra_id}")
        plt.savefig(output_subject_dir / f'vertebra_{vertebra_id}_features_2d.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved 2D features: {output_subject_dir / f'vertebra_{vertebra_id}_features_2d.png'}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Phase 1 Visualization Sanity Check')
    parser.add_argument('--mesh_dir', type=str, default='outputs/meshes',
                        help='Directory containing meshes')
    parser.add_argument('--point_cloud_dir', type=str, default='outputs/point_clouds',
                        help='Directory containing point clouds')
    parser.add_argument('--mask_dir', type=str, default='/gscratch/scrubbed/june0604/vindr/VerSe/processed',
                        help='Directory containing original mask and CT files (fallback if CSV not provided)')
    parser.add_argument('--csv_path', type=str, default='/gscratch/scrubbed/june0604/vindr/VerSe/processed/preprocessed_data_vertebra.csv',
                        help='Path to CSV file with processed paths (vertebra-level or subject-level)')
    parser.add_argument('--subject_csv_path', type=str, default=None,
                        help='Optional: Path to subject-level CSV for CT paths (if csv_path is vertebra-level)')
    parser.add_argument('--output_dir', type=str, default='outputs/visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--random', action='store_true',
                        help='Randomly select samples')
    
    args = parser.parse_args()
    
    mesh_dir = Path(args.mesh_dir)
    pc_dir = Path(args.point_cloud_dir)
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load paths from CSV if provided
    paths_dict = None
    if args.csv_path:
        csv_path = Path(args.csv_path)
        if csv_path.exists():
            subject_csv_path = Path(args.subject_csv_path) if args.subject_csv_path else None
            paths_dict = load_paths_from_csv(csv_path, subject_csv_path)
            if paths_dict:
                print(f"✓ Loaded {len(paths_dict)} subject paths from CSV: {csv_path}")
            else:
                print(f"⚠ Failed to load paths from CSV: {csv_path}")
        else:
            print(f"⚠ CSV file not found: {csv_path}")
    else:
        # Try default CSV locations (prefer subject-level for CT paths)
        project_root = Path(__file__).parent.parent.parent
        default_csvs = [
            project_root / 'VerSe' / 'processed' / 'preprocessed_data_subject_with_paths.csv',
            project_root / 'VerSe' / 'processed' / 'preprocessed_data_subject.csv',
            Path('../../VerSe/processed/preprocessed_data_subject_with_paths.csv'),
            Path('../../VerSe/processed/preprocessed_data_subject.csv'),
        ]
        for default_csv in default_csvs:
            if default_csv.exists():
                paths_dict = load_paths_from_csv(default_csv)
                if paths_dict:
                    print(f"✓ Loaded {len(paths_dict)} subject paths from default CSV: {default_csv}")
                    break
    
    print("="*60)
    print("Phase 1 Visualization Sanity Check")
    print("="*60)
    print(f"Mesh directory: {mesh_dir}")
    print(f"Point cloud directory: {pc_dir}")
    if mask_dir:
        print(f"Mask directory (fallback): {mask_dir}")
    if paths_dict:
        print(f"Using CSV paths: {len(paths_dict)} subjects")
    else:
        print(f"⚠ No CSV paths loaded, using mask_dir fallback")
    print(f"Output directory: {output_dir}")
    print()
    
    # Find all available samples
    all_samples = []
    for subject_dir in pc_dir.iterdir():
        if not subject_dir.is_dir():
            continue
        
        subject_id = subject_dir.name
        pc_files = list(subject_dir.glob('vertebra_*_points.npy'))
        
        for pc_file in pc_files:
            vertebra_id = int(pc_file.stem.split('_')[1])
            all_samples.append((subject_id, vertebra_id))
    
    print(f"Found {len(all_samples)} total samples")
    
    # Select samples to visualize
    if args.random:
        import random
        random.seed(42)
        samples = random.sample(all_samples, min(args.num_samples, len(all_samples)))
    else:
        samples = all_samples[:args.num_samples]
    
    print(f"Visualizing {len(samples)} samples...")
    print()
    
    # Visualize each sample
    success_count = 0
    for subject_id, vertebra_id in samples:
        try:
            if check_sample(mesh_dir, pc_dir, mask_dir, subject_id, vertebra_id, output_dir, paths_dict):
                success_count += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("="*60)
    print(f"Visualization Complete: {success_count}/{len(samples)} samples visualized")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

