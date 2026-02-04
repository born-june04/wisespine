#!/usr/bin/env python3
"""
Test script for geometry pipeline (Phase 1)

Tests mesh extraction, point cloud sampling, and feature computation
on a single sample to verify the pipeline works correctly.
"""

import sys
from pathlib import Path
import numpy as np
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spine_point_cloud_assembly.utils.geometry import (
    extract_mesh_from_mask,
    sample_point_cloud,
    validate_mesh,
)
from spine_point_cloud_assembly.utils.features import (
    compute_surface_normals,
    compute_curvature,
)


def create_test_mask():
    """Create a simple test mask (sphere)"""
    size = 64
    center = size // 2
    radius = 20
    
    x, y, z = np.ogrid[:size, :size, :size]
    mask = (x - center)**2 + (y - center)**2 + (z - center)**2 <= radius**2
    
    return mask.astype(np.float32)


def test_mesh_extraction():
    """Test mesh extraction from mask"""
    print("Testing mesh extraction...")
    
    mask = create_test_mask()
    spacing = (1.0, 1.0, 1.0)
    
    mesh_result = extract_mesh_from_mask(mask, spacing=spacing)
    
    assert mesh_result is not None, "Mesh extraction failed"
    assert 'vertices' in mesh_result, "Missing vertices"
    assert 'faces' in mesh_result, "Missing faces"
    
    vertices = mesh_result['vertices']
    faces = mesh_result['faces']
    
    assert len(vertices) > 0, "Empty vertices"
    assert len(faces) > 0, "Empty faces"
    assert vertices.shape[1] == 3, f"Invalid vertex shape: {vertices.shape}"
    assert faces.shape[1] == 3, f"Invalid face shape: {faces.shape}"
    
    # Validate mesh
    is_valid, issues = validate_mesh(vertices, faces)
    assert is_valid, f"Invalid mesh: {issues}"
    
    print(f"  ✓ Mesh extracted: {len(vertices)} vertices, {len(faces)} faces")
    return mesh_result


def test_point_cloud_sampling(mesh_result):
    """Test point cloud sampling from mesh"""
    print("Testing point cloud sampling...")
    
    vertices = mesh_result['vertices']
    faces = mesh_result['faces']
    num_points = 512
    
    points = sample_point_cloud(vertices, faces, num_points=num_points, method='uniform')
    
    assert points.shape == (num_points, 3), f"Invalid point shape: {points.shape}"
    assert not np.isnan(points).any(), "NaN values in points"
    assert not np.isinf(points).any(), "Inf values in points"
    
    print(f"  ✓ Point cloud sampled: {len(points)} points")
    return points


def test_feature_computation(points):
    """Test feature computation"""
    print("Testing feature computation...")
    
    # Test normals
    normals = compute_surface_normals(points, k=20)
    assert normals.shape == points.shape, f"Invalid normal shape: {normals.shape}"
    assert not np.isnan(normals).any(), "NaN values in normals"
    
    # Check unit normals
    norms = np.linalg.norm(normals, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Normals are not unit vectors"
    
    print(f"  ✓ Normals computed: {normals.shape}")
    
    # Test curvature
    k1, k2, d1 = compute_curvature(points, normals=normals, k=20)
    assert k1.shape == (len(points),), f"Invalid k1 shape: {k1.shape}"
    assert k2.shape == (len(points),), f"Invalid k2 shape: {k2.shape}"
    assert d1.shape == (len(points), 3), f"Invalid d1 shape: {d1.shape}"
    
    print(f"  ✓ Curvature computed: k1={k1.mean():.4f}, k2={k2.mean():.4f}")
    
    return normals, k1, k2, d1


def main():
    print("=" * 60)
    print("Testing Geometry Pipeline (Phase 1)")
    print("=" * 60)
    print()
    
    try:
        # Test 1: Mesh extraction
        mesh_result = test_mesh_extraction()
        print()
        
        # Test 2: Point cloud sampling
        points = test_point_cloud_sampling(mesh_result)
        print()
        
        # Test 3: Feature computation
        normals, k1, k2, d1 = test_feature_computation(points)
        print()
        
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

