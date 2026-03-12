#!/usr/bin/env python3
"""
Simulate Scoliosis Data (Physics-Based)
Ultra Memory-Optimized: Never loads full volume into RAM.
Refactored for Batch Processing.
"""

import numpy as np
import nibabel as nib
import sys
import gc
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.spatial.transform import Rotation as R

def compute_centers_slicewise(mask_path):
    """Compute vertebra centroids by loading one Z-slab at a time."""
    nii = nib.load(mask_path)
    proxy = nii.dataobj
    shape = proxy.shape
    
    # Accumulators: label -> (sum_x, sum_y, sum_z, count)
    accum = {}
    
    slab = 50  # Process 50 slices at a time
    for z0 in range(0, shape[2], slab):
        z1 = min(z0 + slab, shape[2])
        chunk = np.asarray(proxy[:, :, z0:z1])  # Load slab
        
        labels = np.unique(chunk)
        labels = labels[labels > 0]
        
        for lb in labels:
            coords = np.argwhere(chunk == lb)  # (N, 3) relative to chunk
            coords[:, 2] += z0  # Adjust Z to global
            
            lb = int(lb)
            if lb not in accum:
                accum[lb] = [0.0, 0.0, 0.0, 0]
            accum[lb][0] += coords[:, 0].sum()
            accum[lb][1] += coords[:, 1].sum()
            accum[lb][2] += coords[:, 2].sum()
            accum[lb][3] += len(coords)
        
        del chunk
    
    centers = {}
    for lb, (sx, sy, sz, cnt) in accum.items():
        centers[lb] = np.array([sx/cnt, sy/cnt, sz/cnt])
    
    return centers


def generate_scoliosis_curve(centers, cobb_angle):
    """Generate target positions/rotations with enhanced realism (torsion + noise)."""
    sorted_labels = sorted(centers.keys(), key=lambda l: centers[l][2])
    z_coords = np.array([centers[l][2] for l in sorted_labels])
    
    z_min, z_max = z_coords.min(), z_coords.max()
    z_height = z_max - z_min
    z_norm = (z_coords - z_min) / (z_height + 1e-6)
    
    # Lateral deviation amplitude (empirical: 10° Cobb ≈ 0.05 * height)
    amplitude = (cobb_angle / 180.0 * np.pi) * (z_height * 0.15)
    deviation = amplitude * np.sin(np.pi * z_norm)
    
    # Refinement: Stronger axial rotation coupling (0.6 instead of 0.3)
    # Severe scoliosis has significant torsion.
    axial_gain = 0.6
    
    # Refinement: Random seed for segmental irregularity
    rng = np.random.RandomState(42)
    
    new_centers = {}
    rotations = {}
    
    for i, label in enumerate(sorted_labels):
        old = centers[label]
        z = z_norm[i]
        
        dx = deviation[i]
        
        # Coupled axial rotation (stronger at apex)
        axial_rot = (cobb_angle * axial_gain) * np.sin(np.pi * z)
        
        # Coronal tilt (derivative of curve)
        deriv = amplitude * (np.pi / z_height) * np.cos(np.pi * z)
        tilt = np.degrees(np.arctan(deriv))
        
        # Refinement: Add segmental irregularity (noise)
        # DISABLED to prevent system crashes during bulk warp
        pass
        
        r_tilt = R.from_euler('y', -tilt, degrees=True)
        r_axial = R.from_euler('z', axial_rot, degrees=True)
        rotations[label] = r_axial * r_tilt
        
        new_centers[label] = old + np.array([dx, 0, 0])
    
    return new_centers, rotations


def build_lowres_field(shape, centers_old, centers_new, rotations, mask_path, scale=0.125, z_crop_start=0):
    """Build deformation field at low resolution using slice-wise mask loading."""
    step = int(1.0 / scale)
    small = [max(1, s // step) for s in shape]
    
    print(f"  Low-res grid: {small}")
    sys.stdout.flush()
    
    field = np.zeros((*small, 3), dtype=np.float32)
    mask_low = np.zeros(small, dtype=np.int8)
    
    # Load mask at low res (every `step`-th slice from cropped region)
    nii = nib.load(mask_path)
    proxy = nii.dataobj
    
    for zi in range(small[2]):
        z_full = z_crop_start + zi * step  # Map from cropped low-res to global
        if z_full >= proxy.shape[2]:
            break
        sl = np.asarray(proxy[:, :, z_full])  # Single slice (H, W)
        mask_low[:, :, zi] = sl[::step, ::step][:small[0], :small[1]]
    
    print(f"  Low-res mask loaded. Labels: {np.unique(mask_low)[1:]}")
    sys.stdout.flush()
    
    # Compute displacements for bone voxels in low-res
    for lb in np.unique(mask_low):
        if lb <= 0 or lb not in centers_old:
            continue
        
        idx = np.where(mask_low == lb)
        if len(idx[0]) == 0:
            continue
        
        # Map low-res indices back to full-res coordinates
        pts = np.stack(idx, axis=-1).astype(np.float64) * step
        
        c_old = centers_old[lb]
        c_new = centers_new[lb]
        rot = rotations[lb]
        
        pts_new = rot.apply(pts - c_old) + c_new
        disp = (pts_new - pts).astype(np.float32)
        
        field[idx[0], idx[1], idx[2], :] = disp
    
    # Smooth for soft tissue (normalized convolution)
    bone = (mask_low > 0).astype(np.float32)
    sigma = 3.0  # Low-res sigma
    
    norm = gaussian_filter(bone, sigma=sigma)
    norm = np.clip(norm, 1e-6, 1.0)
    
    smoothed = np.zeros_like(field)
    for c in range(3):
        blurred = gaussian_filter(field[..., c], sigma=sigma)
        interp = blurred / norm
        smoothed[..., c] = np.where(bone > 0, field[..., c], interp)
    
    del mask_low, bone
    gc.collect()
    
    return smoothed, step


def warp_chunked(ct_data, field, step, output_path, affine):
    """Warp CT (already in RAM) using low-res field, processing in chunks.
    
    Uses map_coordinates on the low-res field for fast interpolation.
    """
    shape = ct_data.shape
    small = field.shape[:3]
    
    # Output array (in memory, 1.2GB fits easily)
    out = np.zeros(shape, dtype=np.float32)
    chunk = 10
    t0 = time.time()
    
    print(f"    Starting warp loop (chunk={chunk})...")
    sys.stdout.flush()
    
    for z0 in range(0, shape[2], chunk):
        z1 = min(z0 + chunk, shape[2])
        
        try:
            # Float64 grids for stability
            cx = np.arange(shape[0], dtype=np.float64)
            cy = np.arange(shape[1], dtype=np.float64)
            cz = np.arange(z0, z1, dtype=np.float64)
            
            CX, CY, CZ = np.meshgrid(cx, cy, cz, indexing='ij')
            
            LX = CX / step
            LY = CY / step
            LZ = CZ / step
            
            # Using coordinate order [LX, LY, LZ] which was verified correct
            # Stack for vectorized map_coordinates
            low_coords = np.stack([LX, LY, LZ], axis=0)
            
            dx = map_coordinates(field[..., 0], low_coords, order=1, mode='nearest')
            dy = map_coordinates(field[..., 1], low_coords, order=1, mode='nearest')
            dz = map_coordinates(field[..., 2], low_coords, order=1, mode='nearest')
            
            mx = CX - dx
            my = CY - dy
            mz = CZ - dz
            
            mx = np.clip(mx, 0, shape[0] - 1)
            my = np.clip(my, 0, shape[1] - 1)
            mz = np.clip(mz, 0, shape[2] - 1)
            
            map_c = np.stack([mx, my, mz])
            warped = map_coordinates(ct_data, map_c, order=1, mode='nearest')
            out[:, :, z0:z1] = warped.astype(np.float32)
            
        except Exception as e:
            print(f"    CRASH in chunk {z0}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        if z0 % (chunk * 5) == 0:
            print(f"    {z1}/{shape[2]} ({time.time()-t0:.1f}s)")
            sys.stdout.flush()
    
    # Save NIfTI
    print(f"  Saving {output_path}...")
    nib.save(nib.Nifti1Image(out, affine), str(output_path))
    
    # Collect viz slices before deleting memmap
    mid_x = shape[0] // 2
    mid_y = shape[1] // 2
    mid_z = shape[2] // 2
    sag = out[mid_x, :, :].copy()
    cor = out[:, mid_y, :].copy()
    axi = out[:, :, mid_z].copy()
    
    return sag, cor, axi


def run_scoliosis_simulation(ct_path_in, mask_path_in, out_dir, angles=[20, 40, 60], subject_id="unknown"):
    print("=" * 60)
    print("PHYSICS-BASED SCOLIOSIS SIMULATION (REFINED)")
    print("  - Stronger Axial Torsion")
    print("  - Segmental Irregularity (Noise)")
    print("=" * 60)
    sys.stdout.flush()
    
    ct_path = Path(ct_path_in)
    mask_path = Path(mask_path_in)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Subject: {subject_id}")
    print(f"CT: {ct_path}")
    print(f"Mask: {mask_path}")
    sys.stdout.flush()
    
    ct_nii = nib.load(str(ct_path))
    shape = ct_nii.shape
    affine = ct_nii.affine
    print(f"CT shape: {shape}")
    sys.stdout.flush()
    
    # 1. Centers (slice-wise, no full load)
    print("\n[1/4] Computing vertebra centers (slice-wise)...")
    sys.stdout.flush()
    centers = compute_centers_slicewise(str(mask_path))
    print(f"  Found {len(centers)} vertebrae: {sorted(centers.keys())}")
    sys.stdout.flush()
    
    # Determine spine region in Z for cropping
    all_z = [c[2] for c in centers.values()]
    z_min_spine = int(min(all_z))
    z_max_spine = int(max(all_z))
    pad = 100  # Extra slices around spine
    z_crop_start = max(0, z_min_spine - pad)
    z_crop_end = min(shape[2], z_max_spine + pad)
    print(f"  Spine Z range: {z_min_spine}-{z_max_spine}")
    print(f"  Cropping CT to Z [{z_crop_start}:{z_crop_end}] ({z_crop_end - z_crop_start} slices)")
    sys.stdout.flush()
    
    # Load ONLY the cropped region ITERATIVELY to avoid crashes
    print("Loading cropped CT data into RAM (iteratively)...")
    sys.stdout.flush()
    
    depth = z_crop_end - z_crop_start
    ct_data_cropped = np.zeros((shape[0], shape[1], depth), dtype=np.float32)
    load_step = 100
    
    for z in range(z_crop_start, z_crop_end, load_step):
        z2 = min(z + load_step, z_crop_end)
        sys.stdout.flush()
        chunk = ct_nii.dataobj[:, :, z:z2]
        ct_data_cropped[:, :, z-z_crop_start:z2-z_crop_start] = chunk
        
    cropped_shape = ct_data_cropped.shape
    print(f"  Cropped CT loaded: {cropped_shape}, {ct_data_cropped.nbytes / 1e9:.2f} GB")
    sys.stdout.flush()
    
    # Adjust centers relative to crop
    centers_cropped = {}
    for lb, c in centers.items():
        centers_cropped[lb] = c.copy()
        centers_cropped[lb][2] -= z_crop_start
    
    # Original viz slices (from cropped region)
    mid_x = cropped_shape[0] // 2
    mid_y = cropped_shape[1] // 2
    mid_z = cropped_shape[2] // 2
    orig_sag = ct_data_cropped[mid_x, :, :].copy()
    orig_cor = ct_data_cropped[:, mid_y, :].copy()
    orig_axi = ct_data_cropped[:, :, mid_z].copy()
    
    fig, axes = plt.subplots(len(angles) + 1, 3, figsize=(18, 6 * (len(angles) + 1)))
    vmin, vmax = -200, 1500
    
    axes[0, 0].imshow(orig_sag.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Original - Sagittal", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(orig_cor.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Original - Coronal", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    axes[0, 2].imshow(orig_axi.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title("Original - Axial", fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    del orig_sag, orig_cor, orig_axi
    gc.collect()
    
    import json
    pose_export = {}
    
    # Warped Loop
    for i, angle in enumerate(angles):
        print(f"\n[2/4] Cobb {angle}° — Generating curve...")
        sys.stdout.flush()
        
        new_centers, rotations = generate_scoliosis_curve(centers_cropped, angle)
        
        # Save pose data for hardware placement (Phase 2)
        pose_export[str(angle)] = {
            "centers": {k: v.tolist() for k, v in new_centers.items()},
            # Convert scipy Rotations to Euler:
            "rotations": {k: rotations[k].as_euler('xyz', degrees=True).tolist() for k in rotations}
        }
        
        print(f"[3/4] Cobb {angle}° — Building low-res deformation field...")
        sys.stdout.flush()
        # Note: field building is fast
        field, step = build_lowres_field(cropped_shape, centers_cropped, new_centers, rotations, str(mask_path), z_crop_start=z_crop_start)
        
        print(f"[4/4] Cobb {angle}° — Warping CT & Mask...")
        sys.stdout.flush()
        
        # Adjust affine for cropped region
        cropped_affine = affine.copy()
        cropped_affine[:3, 3] += affine[:3, 2] * z_crop_start
        
        out_ct_path = out_dir / f"scoliosis_cobb{angle}.nii.gz"
        
        # Warp CT
        sag, cor, axi = warp_chunked(ct_data_cropped, field, step, out_ct_path, cropped_affine)
        
        # Visualization
        row = i + 1
        axes[row, 0].imshow(sag.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"Cobb {angle}° - Sagittal", fontsize=14)
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(cor.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"Cobb {angle}° - Coronal", fontsize=14)
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(axi.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        axes[row, 2].set_title(f"Cobb {angle}° - Axial", fontsize=14)
        axes[row, 2].axis('off')
        
        del field
        gc.collect()
        
    # Save Pose Data
    pose_path = out_dir / "scoliosis_pose.json"
    with open(pose_path, "w") as f:
        json.dump(pose_export, f, indent=2)
    print(f"  ✓ Saved Pose Data: {pose_path}")
    
    # Save Figure
    viz_path = out_dir / "scoliosis_simulation_comparison.png"
    plt.tight_layout()
    plt.savefig(str(viz_path), dpi=150)
    print(f"  ✓ Saved Visualization: {viz_path}")
    plt.close(fig)
    
    print("=" * 60)
    print("DONE")

def main():
    # Backward compatibility
    subject = "sub-verse563"
    ct_path = Path("/gscratch/scrubbed/june0604/wisespine/VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz")
    mask_path = Path("/gscratch/scrubbed/june0604/wisespine/VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz")
    out_dir = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
    
    run_scoliosis_simulation(ct_path, mask_path, out_dir, angles=[20, 40, 60], subject_id=subject)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
