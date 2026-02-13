#!/usr/bin/env python3
"""
Real Vertebra Fragmentation - Using actual L1 mesh.

Goal: Fragment L1 vertebra into 5-10 pieces for PyBullet fracture simulation.
"""

import numpy as np
import trimesh
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

print("="*70)
print("REAL VERTEBRA FRAGMENTATION")
print("="*70)

# Load real L1 vertebra
vertebra_path = "outputs/mujoco_per_vertebra/sub-verse563/meshes/L1.obj"
print(f"\nLoading: {vertebra_path}")

mesh = trimesh.load(vertebra_path, process=False)
print(f"✓ Loaded mesh:")
print(f"  Vertices: {len(mesh.vertices)}")
print(f"  Faces: {len(mesh.faces)}")
print(f"  Bounds: {mesh.bounds}")
print(f"  Extents: {mesh.extents}")

# Method: Plane-based slicing (more reliable than Voronoi for vertebrae)
def fragment_vertebra_planes(mesh, num_slices_z=3, num_slices_xy=2):
    """
    Fragment vertebra by slicing with planes.
    
    Strategy:
    1. Slice along Z axis (vertebra height) into 3 pieces
    2. Optionally slice each piece along X or Y for more fragments
    
    Args:
        mesh: trimesh object
        num_slices_z: number of slices along Z (height)
        num_slices_xy: additional slices in X/Y plane
    """
    print(f"\n--- Fragmenting with {num_slices_z} Z-slices, {num_slices_xy} XY-slices ---")
    
    bounds = mesh.bounds
    z_min, z_max = bounds[0][2], bounds[1][2]
    z_range = z_max - z_min
    z_step = z_range / num_slices_z
    
    fragments = []
    
    # Slice along Z axis
    for i in range(num_slices_z):
        z_start = z_min + i * z_step
        z_end = z_start + z_step
        
        print(f"\n  Slice {i+1}: Z=[{z_start:.4f}, {z_end:.4f}]")
        
        # Create a bounding box for this slice
        # We'll use a slightly expanded region to avoid missing vertices
        margin = 0.001  # 1mm margin
        
        # Filter vertices by Z coordinate
        vertices = mesh.vertices
        in_slice = (vertices[:, 2] >= z_start - margin) & (vertices[:, 2] < z_end + margin)
        
        if in_slice.sum() < 4:
            print(f"    ⚠️  Too few vertices ({in_slice.sum()}), skipping")
            continue
        
        # Get faces that have ALL vertices in this slice
        slice_faces = []
        vertex_indices = np.where(in_slice)[0]
        vertex_set = set(vertex_indices)
        
        for face in mesh.faces:
            if all(v_idx in vertex_set for v_idx in face):
                slice_faces.append(face)
        
        if len(slice_faces) < 1:
            print(f"    ⚠️  No complete faces, skipping")
            continue
        
        # Create vertex mapping (old index -> new index)
        v_map = {old_idx: new_idx for new_idx, old_idx in enumerate(vertex_indices)}
        
        # Remap faces
        new_faces = []
        for face in slice_faces:
            new_face = [v_map[v_idx] for v_idx in face]
            new_faces.append(new_face)
        
        # Create fragment mesh
        fragment = trimesh.Trimesh(
            vertices=vertices[in_slice],
            faces=np.array(new_faces),
            process=False
        )
        
        # Clean up (update mesh to remove degenerate faces)
        fragment.update_faces(fragment.nondegenerate_faces())
        fragment.update_faces(fragment.unique_faces())
        
        if len(fragment.vertices) > 3 and len(fragment.faces) > 0:
            fragments.append(fragment)
            print(f"    ✓ Fragment {len(fragments)}: {len(fragment.vertices)} vertices, {len(fragment.faces)} faces")
            print(f"      Centroid: {fragment.centroid}")
            print(f"      Volume: {fragment.volume:.6f}")
        else:
            print(f"    ⚠️  Invalid fragment, skipping")
    
    print(f"\n✓ Created {len(fragments)} fragments from Z-slicing")
    
    # Optional: Further subdivide each fragment along X or Y
    if num_slices_xy > 1 and len(fragments) > 0:
        print(f"\n--- Further subdividing with {num_slices_xy} XY-slices ---")
        all_fragments = []
        
        for frag_idx, frag in enumerate(fragments):
            # Slice along X axis
            x_min, x_max = frag.bounds[0][0], frag.bounds[1][0]
            x_range = x_max - x_min
            x_step = x_range / num_slices_xy
            
            for j in range(num_slices_xy):
                x_start = x_min + j * x_step
                x_end = x_start + x_step
                
                vertices = frag.vertices
                in_slice = (vertices[:, 0] >= x_start) & (vertices[:, 0] < x_end)
                
                if in_slice.sum() < 4:
                    continue
                
                # Same process as Z-slicing
                vertex_indices = np.where(in_slice)[0]
                vertex_set = set(vertex_indices)
                
                slice_faces = []
                for face in frag.faces:
                    if all(v_idx in vertex_set for v_idx in face):
                        slice_faces.append(face)
                
                if len(slice_faces) < 1:
                    continue
                
                v_map = {old_idx: new_idx for new_idx, old_idx in enumerate(vertex_indices)}
                new_faces = [[v_map[v_idx] for v_idx in face] for face in slice_faces]
                
                sub_fragment = trimesh.Trimesh(
                    vertices=vertices[in_slice],
                    faces=np.array(new_faces),
                    process=False
                )
                
                sub_fragment.update_faces(sub_fragment.nondegenerate_faces())
                sub_fragment.update_faces(sub_fragment.unique_faces())
                
                if len(sub_fragment.vertices) > 3 and len(sub_fragment.faces) > 0:
                    all_fragments.append(sub_fragment)
        
        if len(all_fragments) > len(fragments):
            fragments = all_fragments
            print(f"✓ Total fragments after XY subdivision: {len(fragments)}")
    
    return fragments


# Fragment the vertebra
fragments = fragment_vertebra_planes(mesh, num_slices_z=5, num_slices_xy=1)

if len(fragments) == 0:
    print("\n❌ Fragmentation failed! Trying alternative method...")
    # Fallback: use trimesh's built-in split
    fragments = mesh.split()
    print(f"✓ Split into {len(fragments)} connected components")

print(f"\n{'='*70}")
print(f"FRAGMENTATION RESULT: {len(fragments)} pieces")
print(f"{'='*70}")

# Save fragments
output_dir = "outputs/phase3_physics_fracture/pybullet_models/L1_fractured"
os.makedirs(output_dir, exist_ok=True)

for i, frag in enumerate(fragments):
    frag_path = os.path.join(output_dir, f"L1_frag_{i}.obj")
    frag.export(frag_path)
    print(f"  Saved: L1_frag_{i}.obj ({len(frag.vertices)} vertices)")

print(f"\n✓ All fragments saved to: {output_dir}")

# Visualize fragments
print(f"\n--- Creating visualization ---")

fig = plt.figure(figsize=(15, 10))

# Original mesh
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.set_title(f'Original L1 Vertebra\n({len(mesh.vertices)} vertices)', fontweight='bold')
vertices = mesh.vertices
faces = mesh.faces
poly = Poly3DCollection(vertices[faces], alpha=0.7, facecolor='tan', edgecolor='black', linewidths=0.1)
ax1.add_collection3d(poly)
ax1.set_xlim(mesh.bounds[:, 0])
ax1.set_ylim(mesh.bounds[:, 1])
ax1.set_zlim(mesh.bounds[:, 2])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Fragments (each with different color)
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.set_title(f'Fragmented ({len(fragments)} pieces)', fontweight='bold', color='red')
colors = plt.cm.tab10(np.linspace(0, 1, len(fragments)))
for i, frag in enumerate(fragments):
    vertices = frag.vertices
    faces = frag.faces
    poly = Poly3DCollection(vertices[faces], alpha=0.8, facecolor=colors[i], edgecolor='black', linewidths=0.2)
    ax2.add_collection3d(poly)
ax2.set_xlim(mesh.bounds[:, 0])
ax2.set_ylim(mesh.bounds[:, 1])
ax2.set_zlim(mesh.bounds[:, 2])
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

# Fragment sizes
ax3 = fig.add_subplot(2, 2, 3)
fragment_sizes = [len(f.vertices) for f in fragments]
ax3.bar(range(len(fragments)), fragment_sizes, color=colors)
ax3.set_xlabel('Fragment Index')
ax3.set_ylabel('Number of Vertices')
ax3.set_title('Fragment Sizes')
ax3.grid(True, alpha=0.3)

# Statistics
ax4 = fig.add_subplot(2, 2, 4)
ax4.axis('off')
stats_text = f"""
FRAGMENTATION STATISTICS

Original Mesh:
  Vertices: {len(mesh.vertices)}
  Faces: {len(mesh.faces)}
  Volume: {mesh.volume:.6f}

Fragments: {len(fragments)}
  
Fragment Details:
"""
for i, frag in enumerate(fragments):
    stats_text += f"\n  #{i}: {len(frag.vertices):4d} verts, {len(frag.faces):4d} faces"

ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
viz_path = os.path.join(output_dir, "fragmentation_visualization.png")
plt.savefig(viz_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved visualization: {viz_path}")

print(f"\n{'='*70}")
print(f"SUCCESS!")
print(f"{'='*70}")
print(f"\n📊 Results:")
print(f"  • Fragments created: {len(fragments)}")
print(f"  • Output directory: {output_dir}")
print(f"  • Visualization: {viz_path}")
print(f"\n🚀 Next step: Create PyBullet URDFs with breakable constraints!")
print(f"{'='*70}")

