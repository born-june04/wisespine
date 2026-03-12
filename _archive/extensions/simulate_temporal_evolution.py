#!/usr/bin/env python3
"""
E2: Temporal Dynamics — Multi-Timepoint Causal Response
========================================================

Adds a TIME AXIS to the causal response simulation.
Generates volumes/visualizations at: Day 0, 7, 30, 90, 365.

Temporal evolution:
1. Hematoma: Acute (50HU) → Subacute (30HU) → Chronic/Resorbed (10HU → 0HU)
2. Bone Graft: Chips (noisy 600HU) → Callus (smooth 400HU) → Cortical fusion (800HU)
3. Screw Halo: 1-vox → progressing width (1→2→3 voxels)
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, generate_binary_structure, gaussian_filter
from pathlib import Path
import gc

# Temporal parameters
TIMEPOINTS = {
    "Day 0": {"days": 0,   "hematoma_hu": 50, "graft_hu": 600, "graft_noise": 80, "halo_iter": 1},
    "Day 7": {"days": 7,   "hematoma_hu": 40, "graft_hu": 500, "graft_noise": 60, "halo_iter": 1},
    "Day 30": {"days": 30,  "hematoma_hu": 25, "graft_hu": 450, "graft_noise": 40, "halo_iter": 2},
    "Day 90": {"days": 90,  "hematoma_hu": 10, "graft_hu": 550, "graft_noise": 20, "halo_iter": 2},
    "Day 365": {"days": 365, "hematoma_hu": 0,  "graft_hu": 700, "graft_noise": 10, "halo_iter": 3},
}

def simulate_timepoint(post_data, pre_data, hw_data, params, shape):
    """Apply causal effects at a specific timepoint."""
    vol = post_data.copy()
    
    # 1. Hematoma evolution
    resected = (pre_data > 150) & (post_data < 100)
    if np.any(resected):
        hu = params["hematoma_hu"]
        if hu > 0:
            noise = np.random.normal(hu, 5, np.count_nonzero(resected)).astype(np.int16)
            vol[resected] = noise
        else:
            # Chronic: fully resorbed → scar tissue (soft tissue density)
            vol[resected] = np.random.normal(40, 3, np.count_nonzero(resected)).astype(np.int16)
    
    # 2. Graft remodeling  
    # Find graft zone (high variability region in post-op that wasn't bone pre-op)
    graft_mask = (post_data > 400) & (post_data < 900) & (pre_data < 100)
    if np.any(graft_mask):
        base_hu = params["graft_hu"]
        noise_level = params["graft_noise"]
        graft_vals = np.random.normal(base_hu, noise_level, np.count_nonzero(graft_mask)).astype(np.int16)
        vol[graft_mask] = graft_vals
        
        # Late timepoints: smooth the graft (callus maturation)
        if params["days"] >= 90:
            coords = np.argwhere(graft_mask)
            if len(coords) > 0:
                mn = coords.min(axis=0)
                mx = coords.max(axis=0) + 1
                s = tuple(slice(a, b) for a, b in zip(mn, mx))
                crop = vol[s].astype(np.float32)
                crop_mask = graft_mask[s]
                smoothed = gaussian_filter(crop, sigma=1.0)
                crop[crop_mask] = smoothed[crop_mask]
                vol[s] = crop.astype(np.int16)
    
    # 3. Screw halo progression
    hw_locs = np.argwhere(hw_data > 0)
    if len(hw_locs) > 0:
        struct = generate_binary_structure(3, 1)
        iters = params["halo_iter"]
        
        mn = np.maximum(0, hw_locs.min(axis=0) - iters - 2)
        mx = np.minimum(shape, hw_locs.max(axis=0) + iters + 2)
        s = tuple(slice(a, b) for a, b in zip(mn, mx))
        
        hw_crop = hw_data[s]
        vol_crop = vol[s]
        
        dilated = binary_dilation(hw_crop > 0, structure=struct, iterations=iters)
        halo = dilated & (hw_crop == 0) & (vol_crop > 150)
        
        if np.any(halo):
            # Progressive lucency (wider = more resorption)
            lucency_hu = max(30, 60 - (iters - 1) * 15)
            vol_crop[halo] = lucency_hu
            vol[s] = vol_crop
    
    # 4. Muscle edema resolution
    # Early: -20 HU. Late: resolves back to normal
    if params["days"] < 90:
        muscle_mask = (post_data > 30) & (post_data < 100)
        edema_band = np.zeros_like(vol, dtype=bool)
        if len(hw_locs) > 0:
            x_min, x_max = hw_locs[:, 0].min(), hw_locs[:, 0].max()
            z_min, z_max = hw_locs[:, 2].min(), hw_locs[:, 2].max()
            edema_band[max(0, x_min-40):x_min, :, z_min:z_max] = True
            edema_band[x_max:min(shape[0], x_max+40), :, z_min:z_max] = True
            edema_band = edema_band & muscle_mask
            
            if np.any(edema_band):
                # Decreasing edema over time
                reduction = int(20 * (1 - params["days"] / 90))
                if reduction > 0:
                    vol[edema_band] -= reduction
    
    return vol

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
    angle = 60
    
    post_path = root / f"scoliosis_cobb{angle}_postop.nii.gz"
    pre_path = root / f"scoliosis_cobb{angle}.nii.gz"
    hw_path = root / f"scoliosis_cobb{angle}_hardware.nii.gz"
    
    if not post_path.exists():
        print("Post-op volume not found.")
        return
    
    print("Loading volumes (chunked reading)...")
    nii_post = nib.load(post_path)
    nii_pre = nib.load(pre_path)
    nii_hw = nib.load(hw_path)
    
    shape = nii_post.shape
    affine = nii_post.affine
    
    # Find hardware region to limit processing
    print("Finding hardware ROI...")
    hw_full = np.asanyarray(nii_hw.dataobj).astype(np.uint8)
    hw_locs = np.argwhere(hw_full > 0)
    
    if len(hw_locs) == 0:
        print("No hardware found.")
        return
    
    z_min_hw = max(0, hw_locs[:, 2].min() - 20)
    z_max_hw = min(shape[2], hw_locs[:, 2].max() + 20)
    z_mid = (z_min_hw + z_max_hw) // 2
    x_mid = (hw_locs[:, 0].min() + hw_locs[:, 0].max()) // 2
    
    # Process each timepoint - only the ROI slab
    print(f"Processing ROI z=[{z_min_hw}:{z_max_hw}]...")
    s_z = slice(z_min_hw, z_max_hw)
    
    post_roi = np.asanyarray(nii_post.dataobj[:, :, s_z]).astype(np.int16)
    pre_roi = np.asanyarray(nii_pre.dataobj[:, :, s_z]).astype(np.int16)
    hw_roi = hw_full[:, :, s_z]
    roi_shape = post_roi.shape
    
    del hw_full
    gc.collect()
    
    # Generate timepoint snapshots
    snapshots = {}
    for tp_name, params in TIMEPOINTS.items():
        print(f"  Simulating {tp_name}...")
        vol_tp = simulate_timepoint(post_roi, pre_roi, hw_roi, params, roi_shape)
        snapshots[tp_name] = vol_tp
    
    # Visualization: 5 timepoints × 3 views (Sagittal-Hematoma, Axial-Edema, Zoom-Halo)
    print("Generating Timeline Visualization...")
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    
    tp_names = list(TIMEPOINTS.keys())
    vmin, vmax = -200, 1500
    
    # Find relative z_mid for ROI
    z_rel = z_mid - z_min_hw
    
    for col, tp_name in enumerate(tp_names):
        vol = snapshots[tp_name]
        params = TIMEPOINTS[tp_name]
        
        # Row 0: Sagittal (Hematoma)
        sag = vol[x_mid, :, max(0,z_rel-30):min(roi_shape[2],z_rel+30)].T
        axes[0, col].imshow(sag, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f"{tp_name}\nHematoma: {params['hematoma_hu']} HU", fontsize=10)
        if col == 0:
            axes[0, col].set_ylabel("Sagittal\n(Hematoma)", fontsize=11, fontweight='bold')
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        
        # Row 1: Axial (Edema)
        ax_slice = vol[:, :, z_rel].T
        axes[1, col].imshow(ax_slice, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        edema_status = "Active" if params['days'] < 90 else "Resolved"
        axes[1, col].set_title(f"Edema: {edema_status}", fontsize=10)
        if col == 0:
            axes[1, col].set_ylabel("Axial\n(Muscle Edema)", fontsize=11, fontweight='bold')
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])
        
        # Row 2: Zoom (Halo)
        hw_sl = hw_roi[:, :, z_rel]
        hw_pts = np.argwhere(hw_sl > 0)
        if len(hw_pts) > 0:
            cx, cy = hw_pts[hw_pts[:, 0].argmin()]
            s = 25
            zoom = vol[max(0,cx-s):min(roi_shape[0],cx+s), max(0,cy-s):min(roi_shape[1],cy+s), z_rel].T
        else:
            zoom = np.zeros((50, 50))
        axes[2, col].imshow(zoom, cmap='gray', origin='lower', vmin=vmin, vmax=2500)
        axes[2, col].set_title(f"Halo: {params['halo_iter']}vox width", fontsize=10)
        if col == 0:
            axes[2, col].set_ylabel("Zoom\n(Screw Halo)", fontsize=11, fontweight='bold')
        axes[2, col].set_xticks([]); axes[2, col].set_yticks([])
    
    plt.suptitle("Temporal Evolution of Causal Tissue Response\n(Day 0 → Day 365 Post-Surgery)",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    out_path = root / "temporal_evolution_timeline.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
