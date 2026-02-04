"""
Geometry utilities for mesh extraction and point cloud sampling
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import logging

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    # Only warn if actually trying to use it
    # logging.warning("scikit-image not available. Mesh extraction will use alternative method.")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    # Only warn if actually trying to use trimesh functionality
    # logging.warning("trimesh not available. Some mesh operations may be limited.")

try:
    import point_cloud_utils as pcu
    PCU_AVAILABLE = True
except ImportError:
    PCU_AVAILABLE = False
    # Don't warn at import time - only warn when actually trying to use Poisson sampling


def extract_mesh_from_mask(
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    level: float = 0.5,
    method: str = 'marching_cubes',
) -> Optional[dict]:
    """
    Extract surface mesh from binary segmentation mask using marching cubes.
    
    Args:
        mask: Binary mask array (D, H, W) or (H, W, D)
        spacing: Voxel spacing (dz, dy, dx) or (dx, dy, dz)
        level: Iso-value for marching cubes (typically 0.5 for binary masks)
        method: 'marching_cubes' or 'marching_cubes_lewiner'
    
    Returns:
        Dictionary with keys:
            - 'vertices': (N, 3) array of vertex positions
            - 'faces': (M, 3) array of face indices
            - 'normals': (N, 3) array of vertex normals (if computed)
        Returns None if extraction fails.
    """
    if mask.ndim != 3:
        raise ValueError(f"Mask must be 3D, got shape {mask.shape}")
    
    # Ensure mask is binary
    mask_binary = (mask > 0.5).astype(np.float32)
    
    if mask_binary.sum() == 0:
        logging.warning("Empty mask, cannot extract mesh")
        return None
    
    if not SKIMAGE_AVAILABLE:
        raise ImportError(
            "scikit-image is required for mesh extraction.\n"
            "Please install it with: pip install scikit-image"
        )
    
    try:
        # Use marching cubes
        if method == 'marching_cubes':
            verts, faces, normals, values = measure.marching_cubes(
                mask_binary,
                level=level,
                spacing=spacing,
            )
        else:
            verts, faces, normals, values = measure.marching_cubes_lewiner(
                mask_binary,
                level=level,
                spacing=spacing,
            )
        
        # Check mesh validity
        if len(verts) == 0 or len(faces) == 0:
            logging.warning("Empty mesh extracted")
            return None
        
        # Ensure faces are valid (all indices < num_vertices)
        num_verts = len(verts)
        if faces.max() >= num_verts:
            logging.warning(f"Invalid face indices: max={faces.max()}, num_verts={num_verts}")
            # Filter invalid faces
            valid_faces = faces.max(axis=1) < num_verts
            faces = faces[valid_faces]
            if len(faces) == 0:
                return None
        
        result = {
            'vertices': verts.astype(np.float32),
            'faces': faces.astype(np.int32),
            'normals': normals.astype(np.float32) if normals is not None else None,
        }
        
        return result
        
    except Exception as e:
        logging.error(f"Mesh extraction failed: {e}")
        return None


def sample_point_cloud(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int = 2048,
    method: str = 'poisson',
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample points from mesh surface.
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices
        num_points: Target number of points
        method: 'poisson' (Poisson disk sampling) or 'uniform' (uniform surface sampling)
        seed: Random seed for reproducibility
    
    Returns:
        (num_points, 3) array of sampled point positions
    """
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Empty mesh, cannot sample points")
    
    if method == 'poisson':
        if not PCU_AVAILABLE:
            logging.warning("point-cloud-utils not available, falling back to uniform sampling")
            method = 'uniform'
        else:
            try:
                # Use Poisson disk sampling for better coverage
                # First, sample uniformly to get initial points
                uniform_points = _uniform_surface_sampling(vertices, faces, num_points * 2, seed)
                
                # Then apply Poisson disk sampling
                # Note: pcu.sample_mesh_poisson_disk requires mesh file or vertices+faces
                # For now, use uniform sampling as fallback
                # TODO: Implement proper Poisson disk sampling
                return uniform_points[:num_points]
            except Exception as e:
                logging.warning(f"Poisson sampling failed: {e}, using uniform")
                method = 'uniform'
    
    if method == 'uniform':
        return _uniform_surface_sampling(vertices, faces, num_points, seed)
    else:
        raise ValueError(f"Unknown sampling method: {method}")


def _uniform_surface_sampling(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Uniform surface sampling by sampling points on triangle faces.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Compute face areas
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Cross product for area calculation
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = np.cross(edge1, edge2)
    face_areas = 0.5 * np.linalg.norm(cross, axis=1)
    
    # Normalize to probabilities
    total_area = face_areas.sum()
    if total_area == 0:
        raise ValueError("Mesh has zero surface area")
    
    face_probs = face_areas / total_area
    
    # Sample faces according to area
    sampled_face_indices = np.random.choice(
        len(faces),
        size=num_points,
        p=face_probs,
    )
    
    # Sample points on each face using barycentric coordinates
    points = []
    for face_idx in sampled_face_indices:
        # Random barycentric coordinates
        u, v = np.random.rand(2)
        if u + v > 1:
            u, v = 1 - u, 1 - v
        w = 1 - u - v
        
        # Interpolate vertex positions
        face = faces[face_idx]
        point = (
            u * vertices[face[0]] +
            v * vertices[face[1]] +
            w * vertices[face[2]]
        )
        points.append(point)
    
    return np.array(points, dtype=np.float32)


def validate_mesh(vertices: np.ndarray, faces: np.ndarray) -> Tuple[bool, List[str]]:
    """
    Validate mesh topology and geometry.
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    # Check for empty mesh
    if len(vertices) == 0:
        issues.append("Empty vertices")
        return False, issues
    
    if len(faces) == 0:
        issues.append("Empty faces")
        return False, issues
    
    # Check face indices
    max_vertex_idx = len(vertices) - 1
    invalid_faces = (faces < 0) | (faces > max_vertex_idx)
    if invalid_faces.any():
        issues.append(f"Invalid face indices: {invalid_faces.sum()} faces")
    
    # Check for disconnected components (basic check)
    # TODO: Implement proper connectivity check
    
    # Check for degenerate faces (zero area)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = np.cross(edge1, edge2)
    face_areas = 0.5 * np.linalg.norm(cross, axis=1)
    degenerate = face_areas < 1e-6
    if degenerate.any():
        issues.append(f"Degenerate faces: {degenerate.sum()} faces")
    
    is_valid = len(issues) == 0
    return is_valid, issues

