"""
Directional feature computation for point clouds
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.neighbors import NearestNeighbors
import logging


def compute_surface_normals(
    points: np.ndarray,
    k: int = 20,
    method: str = 'pca',
) -> np.ndarray:
    """
    Compute surface normals for each point using local neighborhood PCA.
    
    Args:
        points: (N, 3) point cloud
        k: Number of nearest neighbors for local PCA
        method: 'pca' (principal component analysis) or 'cross_product'
    
    Returns:
        (N, 3) array of unit normals
    """
    if len(points) < k:
        logging.warning(f"Not enough points ({len(points)}) for k={k}, using k={len(points)-1}")
        k = max(1, len(points) - 1)
    
    if method == 'pca':
        return _compute_normals_pca(points, k)
    elif method == 'cross_product':
        return _compute_normals_cross_product(points, k)
    else:
        raise ValueError(f"Unknown method: {method}")


def _compute_normals_pca(points: np.ndarray, k: int) -> np.ndarray:
    """Compute normals using PCA on local neighborhood."""
    n_points = len(points)
    normals = np.zeros((n_points, 3), dtype=np.float32)
    
    # Build kNN
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    for i in range(n_points):
        # Get neighbors (excluding self)
        neighbor_indices = indices[i, 1:]  # Skip first (self)
        neighbors = points[neighbor_indices]
        
        # Center neighbors
        centroid = neighbors.mean(axis=0)
        centered = neighbors - centroid
        
        # PCA
        if len(centered) < 3:
            # Fallback: use direction to nearest neighbor
            if len(neighbor_indices) > 0:
                direction = points[neighbor_indices[0]] - points[i]
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    normals[i] = direction / norm
                else:
                    normals[i] = np.array([0, 0, 1])  # Default up
            else:
                normals[i] = np.array([0, 0, 1])
            continue
        
        # Compute covariance matrix
        cov = np.cov(centered.T)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Normal is eigenvector corresponding to smallest eigenvalue
        # (tangent plane normal)
        normal = eigenvectors[:, 0]
        
        # Ensure consistent orientation (pointing outward)
        # Simple heuristic: normal should point away from centroid
        to_point = points[i] - centroid
        if np.dot(normal, to_point) < 0:
            normal = -normal
        
        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normals[i] = normal / norm
        else:
            normals[i] = np.array([0, 0, 1])  # Default up
    
    return normals


def _compute_normals_cross_product(points: np.ndarray, k: int) -> np.ndarray:
    """Compute normals using cross product of local edges (simpler but less robust)."""
    n_points = len(points)
    normals = np.zeros((n_points, 3), dtype=np.float32)
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    for i in range(n_points):
        neighbor_indices = indices[i, 1:k+1]  # Skip self, take k neighbors
        
        if len(neighbor_indices) < 2:
            normals[i] = np.array([0, 0, 1])
            continue
        
        # Use first two neighbors to form a plane
        v1 = points[neighbor_indices[0]] - points[i]
        v2 = points[neighbor_indices[1]] - points[i]
        
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        
        if norm > 1e-6:
            normals[i] = normal / norm
        else:
            normals[i] = np.array([0, 0, 1])
    
    return normals


def compute_curvature(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    k: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute principal curvatures and directions using local PCA.
    
    Args:
        points: (N, 3) point cloud
        normals: (N, 3) surface normals (if None, will compute)
        k: Number of nearest neighbors
    
    Returns:
        k1: (N,) principal curvature (maximum)
        k2: (N,) principal curvature (minimum)
        d1: (N, 3) principal direction (corresponding to k1)
    """
    # Validate inputs
    if len(points) == 0:
        raise ValueError("Points array is empty")
    
    if points.shape[1] != 3:
        raise ValueError(f"Points must be (N, 3), got shape {points.shape}")
    
    # Ensure k is an integer
    k = int(k)
    if k < 1:
        k = 1
    if k >= len(points):
        k = max(1, len(points) - 1)
    
    if normals is None:
        normals = compute_surface_normals(points, k)
    
    if normals.shape != points.shape:
        raise ValueError(f"Normals shape {normals.shape} doesn't match points shape {points.shape}")
    
    n_points = len(points)
    k1 = np.zeros(n_points, dtype=np.float32)
    k2 = np.zeros(n_points, dtype=np.float32)
    d1 = np.zeros((n_points, 3), dtype=np.float32)
    
    # Store k as local variable to avoid shadowing
    k_neighbors = int(k)  # Use different name to avoid any shadowing
    
    try:
        nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='ball_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)
    except Exception as e:
        logging.warning(f"Failed to build kNN tree: {e}, using fallback")
        # Fallback: use all points
        for i in range(n_points):
            k1[i] = 0.0
            k2[i] = 0.0
            d1[i] = np.array([1, 0, 0], dtype=np.float32)
        return k1, k2, d1
    
    for i in range(n_points):
        try:
            neighbor_indices = indices[i, 1:]  # Skip self
            neighbors = points[neighbor_indices]
            
            if len(neighbors) < 3:
                k1[i] = 0.0
                k2[i] = 0.0
                d1[i] = np.array([1, 0, 0], dtype=np.float32)
                continue
            
            # Project neighbors onto tangent plane
            normal = normals[i]
            if np.linalg.norm(normal) < 1e-6:
                # Invalid normal, use default
                k1[i] = 0.0
                k2[i] = 0.0
                d1[i] = np.array([1, 0, 0], dtype=np.float32)
                continue
            
            centroid = neighbors.mean(axis=0)
            centered = neighbors - centroid
            
            # Project onto tangent plane (remove normal component)
            # Use explicit numpy functions to avoid any shadowing issues
            # IMPORTANT: Use fully qualified names to avoid any variable shadowing
            dot_product = np.dot(centered, normal)
            # Ensure we're using numpy's outer, not any local variable
            import numpy as _np
            outer_product = _np.outer(dot_product, normal)
            tangent_vectors = centered - outer_product
            
            # PCA on tangent plane
            if len(tangent_vectors) < 2:
                k1[i] = 0.0
                k2[i] = 0.0
                d1[i] = np.array([1, 0, 0], dtype=np.float32)
                continue
            
            # Ensure tangent_vectors is 2D
            if tangent_vectors.ndim == 1:
                tangent_vectors = tangent_vectors.reshape(-1, 1)
            
            cov = np.cov(tangent_vectors.T)
            if cov.size == 0 or cov.ndim < 2:
                k1[i] = 0.0
                k2[i] = 0.0
                d1[i] = np.array([1, 0, 0], dtype=np.float32)
                continue
            
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Principal directions (in tangent plane)
            # Largest eigenvalue -> principal direction
            if len(eigenvalues) == 0:
                k1[i] = 0.0
                k2[i] = 0.0
                d1[i] = np.array([1, 0, 0], dtype=np.float32)
                continue
            
            idx_max = np.argmax(np.abs(eigenvalues))
            d1_tangent = eigenvectors[:, idx_max]
            
            # Convert back to 3D (already in tangent plane)
            d1_norm = np.linalg.norm(d1_tangent)
            if d1_norm > 1e-6:
                d1[i] = (d1_tangent / d1_norm).astype(np.float32)
            else:
                d1[i] = np.array([1, 0, 0], dtype=np.float32)
            
            # Curvature approximation from eigenvalues
            # Larger eigenvalue -> higher curvature in that direction
            k1[i] = float(eigenvalues[idx_max])
            if len(eigenvalues) > 1:
                k2[i] = float(eigenvalues[np.argmin(np.abs(eigenvalues))])
            else:
                k2[i] = 0.0
                
        except Exception as e:
            # If any point fails, use default values
            logging.debug(f"Failed to compute curvature for point {i}: {e}")
            k1[i] = 0.0
            k2[i] = 0.0
            d1[i] = np.array([1, 0, 0], dtype=np.float32)
    
    return k1, k2, d1


def compute_local_frame(
    points: np.ndarray,
    normals: np.ndarray,
    vertebra_centroid: np.ndarray,
) -> np.ndarray:
    """
    Compute local anatomical coordinate frame for each point.
    
    Frame definition:
    - z: superior-inferior (along spine axis)
    - y: anterior-posterior
    - x: left-right
    
    Args:
        points: (N, 3) point cloud
        normals: (N, 3) surface normals
        vertebra_centroid: (3,) vertebra centroid
    
    Returns:
        (N, 3, 3) array of rotation matrices (each row is a frame)
    """
    n_points = len(points)
    frames = np.zeros((n_points, 3, 3), dtype=np.float32)
    
    # For now, use a simple frame based on normal and spine direction
    # TODO: Implement proper anatomical frame estimation
    
    # Spine direction (superior-inferior) - assume +z for now
    spine_dir = np.array([0, 0, 1])
    
    for i in range(n_points):
        normal = normals[i]
        
        # z-axis: spine direction (projected onto normal plane)
        z_axis = spine_dir - np.dot(spine_dir, normal) * normal
        z_norm = np.linalg.norm(z_axis)
        if z_norm > 1e-6:
            z_axis = z_axis / z_norm
        else:
            z_axis = np.array([0, 0, 1])
        
        # x-axis: cross product of normal and z
        x_axis = np.cross(normal, z_axis)
        x_norm = np.linalg.norm(x_axis)
        if x_norm > 1e-6:
            x_axis = x_axis / x_norm
        else:
            x_axis = np.array([1, 0, 0])
        
        # y-axis: cross product of z and x
        y_axis = np.cross(z_axis, x_axis)
        y_norm = np.linalg.norm(y_axis)
        if y_norm > 1e-6:
            y_axis = y_axis / y_norm
        else:
            y_axis = np.array([0, 1, 0])
        
        # Store frame
        frames[i, 0] = x_axis
        frames[i, 1] = y_axis
        frames[i, 2] = z_axis
    
    return frames

