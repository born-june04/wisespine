"""
CT Renderer: Convert MuJoCo mesh positions back to CT volume.

Strategy:
1. Load original CT (for reference resolution/spacing)
2. Get deformed vertebrae meshes from MuJoCo
3. Voxelize each mesh into 3D binary mask
4. Assign HU values (bone ~1000)
5. Combine all vertebrae into single CT
6. Save as NIfTI

Key challenges:
- Resolution matching (original CT is ~1mm, meshes are scaled)
- Efficient voxelization (trimesh or manual)
- Realistic HU values
- Handle overlapping vertebrae
"""

import numpy as np
import nibabel as nib
import mujoco
from pathlib import Path
import trimesh
from scipy import ndimage as ndi
from typing import Optional, Tuple


class CTRenderer:
    """Render MuJoCo deformed vertebrae back to CT volume."""
    
    def __init__(
        self,
        original_ct_path: str,
        mujoco_xml_path: str,
        mesh_dir: str,
        voxel_size: float = 1.0,  # mm
        bone_hu: float = 1000.0,
        background_hu: float = -1000.0,
    ):
        """
        Args:
            original_ct_path: Path to original CT (for reference)
            mujoco_xml_path: MuJoCo model with vertebrae
            mesh_dir: Directory with vertebra mesh files
            voxel_size: Voxel size in mm
            bone_hu: HU value for bone
            background_hu: HU value for background
        """
        self.original_ct_path = Path(original_ct_path)
        self.mesh_dir = Path(mesh_dir)
        self.voxel_size = voxel_size
        self.bone_hu = bone_hu
        self.background_hu = background_hu
        
        # Load original CT for reference
        print(f"Loading original CT: {original_ct_path}")
        self.original_nii = nib.load(str(original_ct_path))
        self.original_data = self.original_nii.get_fdata()
        self.affine = self.original_nii.affine
        self.header = self.original_nii.header
        
        print(f"  Shape: {self.original_data.shape}")
        print(f"  Spacing: {self.original_nii.header.get_zooms()}")
        print(f"  HU range: [{self.original_data.min():.0f}, {self.original_data.max():.0f}]")
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(mujoco_xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Get vertebra info
        self.vertebra_info = self._load_vertebra_meshes()
        print(f"  Loaded {len(self.vertebra_info)} vertebra meshes")
    
    def _load_vertebra_meshes(self):
        """Load vertebra meshes and get their body IDs."""
        vertebra_info = []
        
        for i in range(1, self.model.nbody):  # Skip world
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and body_name != 'world':
                # Load mesh
                mesh_file = self.mesh_dir / f"{body_name}.obj"
                if mesh_file.exists():
                    mesh = trimesh.load(str(mesh_file))
                    vertebra_info.append({
                        'name': body_name,
                        'body_id': i,
                        'mesh': mesh,
                    })
        
        return vertebra_info
    
    def render(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        output_shape: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Render current MuJoCo state to CT volume.
        
        Strategy:
        1. Get vertebra positions from MuJoCo (in meters)
        2. Convert to mm (physical coordinates)
        3. Use inverse affine to convert to voxel coordinates
        4. Voxelize and fill CT
        
        Args:
            model: MuJoCo model
            data: MuJoCo data (current state)
            output_shape: Output CT shape (default: match original)
        
        Returns:
            ct_volume: Rendered CT as numpy array
        """
        if output_shape is None:
            output_shape = self.original_data.shape
        
        # Initialize CT volume
        ct_volume = np.full(output_shape, self.background_hu, dtype=np.float32)
        
        print(f"\nRendering {len(self.vertebra_info)} vertebrae to CT...")
        
        # Get affine transformation matrices
        affine_inv = np.linalg.inv(self.affine)
        
        for info in self.vertebra_info:
            body_id = info['body_id']
            mesh_original = info['mesh']  # Original mesh (already in mm from OBJ export)
            name = info['name']
            
            # Get current body position and orientation from MuJoCo (in meters)
            pos_m = data.xpos[body_id]
            rot_mat = data.xmat[body_id].reshape(3, 3)
            
            # MuJoCo position is in meters, convert to mm
            pos_mm = pos_m * 1000.0
            
            # The mesh was exported with scale=0.001 (mm to m for MuJoCo)
            # So mesh.vertices are already in the scaled space (meters)
            # We need to get them back to mm
            vertices_mesh_mm = mesh_original.vertices * 1000.0  # meters to mm
            
            # Apply current rotation
            vertices_rotated = vertices_mesh_mm @ rot_mat.T
            
            # Apply current translation (in mm)
            vertices_physical = vertices_rotated + pos_mm
            
            # Transform from physical coordinates (mm) to voxel coordinates
            # affine_inv: [4x4] matrix that maps physical (mm) to voxel indices
            vertices_hom = np.hstack([vertices_physical, np.ones((len(vertices_physical), 1))])
            vertices_voxel = (affine_inv @ vertices_hom.T).T[:, :3]
            
            # Voxelize mesh
            voxel_mask = self._voxelize_mesh(vertices_voxel, mesh_original.faces, output_shape)
            
            # Assign bone HU values
            ct_volume[voxel_mask] = self.bone_hu
            
            # Print centroid for debugging
            if voxel_mask.sum() > 0:
                centroid = [np.mean(np.where(voxel_mask)[i]) for i in range(3)]
                print(f"  {name}: {voxel_mask.sum()} voxels at voxel ({centroid[0]:.0f}, {centroid[1]:.0f}, {centroid[2]:.0f})")
            else:
                print(f"  {name}: 0 voxels (out of bounds or error)")
        
        print(f"✓ Rendering complete")
        print(f"  Total bone voxels: {(ct_volume > 0).sum()}")
        
        return ct_volume
    
    def _voxelize_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Voxelize mesh using simple rasterization.
        
        Args:
            vertices: Nx3 vertex coordinates in voxel space
            faces: Mx3 face indices
            shape: Output volume shape
        
        Returns:
            mask: Binary mask of voxelized mesh
        """
        # Simple approach: use trimesh voxelization
        # Create mesh with voxel-space vertices
        mesh_voxel = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Use trimesh's voxelization (if available)
        try:
            # Pitch = voxel size (1 voxel)
            voxelized = mesh_voxel.voxelized(pitch=1.0)
            
            # Convert to dense array
            voxel_grid = voxelized.matrix
            
            # Pad/crop to match output shape
            mask = np.zeros(shape, dtype=bool)
            
            # Copy voxelized region into output
            # (This is simplified - assumes alignment)
            src_shape = voxel_grid.shape
            dst_shape = shape
            
            # Calculate copy region
            copy_shape = tuple(min(s, d) for s, d in zip(src_shape, dst_shape))
            
            slices_src = tuple(slice(0, s) for s in copy_shape)
            slices_dst = tuple(slice(0, s) for s in copy_shape)
            
            mask[slices_dst] = voxel_grid[slices_src]
            
            return mask
            
        except Exception as e:
            print(f"    Warning: Trimesh voxelization failed ({e}), using bounding box")
            # Fallback: just fill bounding box
            mask = np.zeros(shape, dtype=bool)
            
            # Get bounding box in voxel space
            vmin = np.floor(vertices.min(axis=0)).astype(int)
            vmax = np.ceil(vertices.max(axis=0)).astype(int)
            
            # Clip to volume bounds
            vmin = np.maximum(vmin, 0)
            vmax = np.minimum(vmax, shape)
            
            # Fill bounding box
            mask[vmin[0]:vmax[0], vmin[1]:vmax[1], vmin[2]:vmax[2]] = True
            
            return mask
    
    def save(self, ct_volume: np.ndarray, output_path: str):
        """Save rendered CT as NIfTI."""
        # Use original affine and header
        nii = nib.Nifti1Image(ct_volume, affine=self.affine, header=self.header)
        nib.save(nii, output_path)
        print(f"\n✓ Saved rendered CT: {output_path}")


def test_ct_rendering():
    """Test CT rendering with original MuJoCo state."""
    
    print("="*70)
    print("CT RENDERING TEST")
    print("="*70)
    
    # Paths
    original_ct = "VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz"
    mujoco_xml = "outputs/mujoco_per_vertebra/sub-verse563/spine_per_vertebra.xml"
    mesh_dir = "outputs/mujoco_per_vertebra/sub-verse563/meshes"
    
    # Initialize renderer
    renderer = CTRenderer(
        original_ct_path=original_ct,
        mujoco_xml_path=mujoco_xml,
        mesh_dir=mesh_dir,
    )
    
    # Render in initial state
    print("\n--- Test 1: Render initial state (no deformation) ---")
    mujoco.mj_resetData(renderer.model, renderer.data)
    mujoco.mj_forward(renderer.model, renderer.data)
    
    ct_initial = renderer.render(renderer.model, renderer.data)
    renderer.save(ct_initial, "outputs/rendered_ct_initial.nii.gz")
    
    # Render with deformation
    print("\n--- Test 2: Render with L1 displaced ---")
    mujoco.mj_resetData(renderer.model, renderer.data)
    
    # Apply force to L1
    l1_id = mujoco.mj_name2id(renderer.model, mujoco.mjtObj.mjOBJ_BODY, "L1")
    force = np.array([0, 0, 50.0])
    
    for _ in range(100):
        renderer.data.xfrc_applied[l1_id, :3] = force
        mujoco.mj_step(renderer.model, renderer.data)
    
    ct_deformed = renderer.render(renderer.model, renderer.data)
    renderer.save(ct_deformed, "outputs/rendered_ct_deformed.nii.gz")
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"✓ Initial CT:   outputs/rendered_ct_initial.nii.gz")
    print(f"✓ Deformed CT:  outputs/rendered_ct_deformed.nii.gz")
    print(f"✓ Original CT:  {original_ct}")
    print()
    print("👀 Please check these files in ITK-SNAP or 3D Slicer:")
    print("   1. Do vertebrae appear in correct positions?")
    print("   2. Are HU values realistic (bone ~1000)?")
    print("   3. Is the deformation visible in the second CT?")
    print("="*70)


if __name__ == "__main__":
    test_ct_rendering()

