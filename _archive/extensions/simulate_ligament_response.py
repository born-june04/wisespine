#!/usr/bin/env python3
"""
E7: Ligament Response Simulation
==================================

Models ligament disruption from surgery and preservation.
- PLC (Posterior Ligament Complex) → disrupted during laminectomy
- ALL (Anterior Longitudinal Ligament) → preserved
- Ligamentum Flavum → partially removed

Physics: Surgical dissection disrupts posterior stabilizers.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from pathlib import Path

# Ligament anatomical zones (relative to vertebral body center)
LIGAMENTS = {
    "ALL": {
        "name": "Anterior Longitudinal Lig.",
        "position": "anterior",  # Y > center (anterior)
        "hu_intact": 80,
        "hu_disrupted": 80,  # Not disrupted
        "status": "PRESERVED",
        "color": "#2ecc71"
    },
    "PLL": {
        "name": "Posterior Longitudinal Lig.",
        "position": "posterior_central",
        "hu_intact": 70,
        "hu_disrupted": 70,  # Usually preserved
        "status": "PRESERVED",
        "color": "#3498db"
    },
    "LF": {
        "name": "Ligamentum Flavum",
        "position": "posterior_deep",
        "hu_intact": 60,
        "hu_disrupted": 20,  # Partially removed
        "status": "PARTIALLY REMOVED",
        "color": "#f39c12"
    },
    "ISL": {
        "name": "Interspinous Ligament",
        "position": "posterior_superficial",
        "hu_intact": 55,
        "hu_disrupted": -50,  # Disrupted (air/fluid)
        "status": "DISRUPTED",
        "color": "#e74c3c"
    },
    "SSL": {
        "name": "Supraspinous Ligament",
        "position": "posterior_most",
        "hu_intact": 50,
        "hu_disrupted": -50,  # Disrupted
        "status": "DISRUPTED",
        "color": "#c0392b"
    },
}

def apply_ligament_changes(volume, hw_data, shape):
    """Apply ligament status changes around surgical site."""
    
    hw_locs = np.argwhere(hw_data > 0)
    if len(hw_locs) == 0:
        return volume
    
    x_center = (hw_locs[:, 0].min() + hw_locs[:, 0].max()) // 2
    y_center = shape[1] // 2  # AP center
    z_min = hw_locs[:, 2].min()
    z_max = hw_locs[:, 2].max()
    
    # Find posterior extent of vertebral body
    # Posterior = lower Y values (convention dependent)
    # We'll use the hardware Y range to infer
    y_hw_min = hw_locs[:, 1].min()
    y_hw_max = hw_locs[:, 1].max()
    
    # Define ligament zones
    # ISL/SSL: Most posterior (beyond spinous process)
    # These are in the laminectomy zone and get disrupted
    
    # PLC zone: y < y_hw_min (behind the screws)
    posterior_zone = slice(max(0, y_hw_min - 30), y_hw_min)
    
    # Apply density changes for disrupted ligaments
    for lig_key, lig in LIGAMENTS.items():
        if lig["status"] == "PRESERVED":
            continue
        
        # Define spatial zone
        if lig["position"] in ["posterior_superficial", "posterior_most"]:
            # Far posterior
            y_s = max(0, y_hw_min - 25)
            y_e = max(0, y_hw_min - 5)
        elif lig["position"] == "posterior_deep":
            # Just behind vertebral body
            y_s = max(0, y_hw_min - 10)
            y_e = y_hw_min
        else:
            continue
        
        # Narrow band around midline
        x_s = max(0, x_center - 8)
        x_e = min(shape[0], x_center + 8)
        
        # Apply
        zone = volume[x_s:x_e, y_s:y_e, z_min:z_max]
        # Only modify soft tissue (< 100 HU, > -200 HU)
        soft_mask = (zone > -200) & (zone < 100)
        
        if np.any(soft_mask):
            target_hu = lig["hu_disrupted"]
            noise = np.random.normal(target_hu, 5, np.count_nonzero(soft_mask)).astype(np.int16)
            zone[soft_mask] = noise
            volume[x_s:x_e, y_s:y_e, z_min:z_max] = zone
    
    return volume

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
    angle = 60
    
    postop_path = root / f"scoliosis_cobb{angle}_postop.nii.gz"
    hw_path = root / f"scoliosis_cobb{angle}_hardware.nii.gz"
    clean_path = root / f"scoliosis_cobb{angle}.nii.gz"
    
    if not postop_path.exists():
        print("Post-op volume not found")
        return
    
    print("Loading volumes (ROI only)...")
    nii_post = nib.load(postop_path)
    nii_hw = nib.load(hw_path)
    nii_clean = nib.load(clean_path)
    
    shape = nii_post.shape
    
    # Load HW to find ROI
    hw_data = np.asanyarray(nii_hw.dataobj).astype(np.uint8)
    hw_locs = np.argwhere(hw_data > 0)
    
    if len(hw_locs) == 0:
        print("No hardware found")
        return
    
    z_min = max(0, hw_locs[:, 2].min() - 10)
    z_max = min(shape[2], hw_locs[:, 2].max() + 10)
    x_mid = (hw_locs[:, 0].min() + hw_locs[:, 0].max()) // 2
    z_mid = (z_min + z_max) // 2
    
    # Load ROI slabs
    s_z = slice(z_min, z_max)
    post_roi = np.asanyarray(nii_post.dataobj[:, :, s_z]).astype(np.int16)
    clean_roi = np.asanyarray(nii_clean.dataobj[:, :, s_z]).astype(np.int16)
    hw_roi = hw_data[:, :, s_z]
    
    del hw_data
    
    # Apply ligament changes
    print("Applying ligament response model...")
    ligament_vol = apply_ligament_changes(post_roi.copy(), hw_roi, post_roi.shape)
    
    # Save
    # Create full volume with ligament changes (copy post-op, replace ROI)
    full_vol = np.asanyarray(nii_post.dataobj).astype(np.int16)
    full_vol[:, :, s_z] = ligament_vol
    
    out_path = root / f"scoliosis_cobb{angle}_ligaments.nii.gz"
    nib.save(nib.Nifti1Image(full_vol, nii_post.affine), out_path)
    print(f"Saved: {out_path}")
    del full_vol
    
    # Visualization: Sagittal annotation
    print("Generating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    z_rel = z_mid - z_min
    
    # Sagittal views
    sag_clean = clean_roi[x_mid, :, :].T
    sag_post = post_roi[x_mid, :, :].T
    sag_lig = ligament_vol[x_mid, :, :].T
    
    axes[0].imshow(sag_clean, cmap='gray', origin='lower', vmin=-200, vmax=1200)
    axes[0].set_title("Pre-Op\n(Intact Ligaments)", fontsize=13, fontweight='bold')
    
    axes[1].imshow(sag_post, cmap='gray', origin='lower', vmin=-200, vmax=1200)
    axes[1].set_title("Post-Op\n(Before Ligament Model)", fontsize=13, fontweight='bold')
    
    axes[2].imshow(sag_lig, cmap='gray', origin='lower', vmin=-200, vmax=1200)
    axes[2].set_title("Post-Op\n(With Ligament Disruption)", fontsize=13, fontweight='bold')
    
    # Annotate ligaments on plot 2 (clean → reference)
    y_hw_min = np.argwhere(hw_roi[x_mid, :, :].any(axis=1)).min() if np.any(hw_roi[x_mid, :, :]) else 100
    
    for i, (key, lig) in enumerate(LIGAMENTS.items()):
        y_pos = max(5, y_hw_min - 5 - i * 5)
        status_icon = "✅" if lig["status"] == "PRESERVED" else ("⚠️" if "PARTIAL" in lig["status"] else "❌")
        
        axes[2].annotate(
            f"{status_icon} {lig['name']}\n({lig['status']})",
            xy=(y_pos, z_rel),
            xytext=(y_pos - 30, z_rel + 5 + i * 8),
            arrowprops=dict(arrowstyle='->', color=lig['color'], linewidth=1.5),
            fontsize=8, color=lig['color'],
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7)
        )
    
    plt.suptitle("E7: Ligament Response Model\nPosterior Ligament Complex Disruption During Laminectomy",
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    out_img = root / "ligament_response_visualization.png"
    plt.savefig(str(out_img), dpi=150, bbox_inches='tight')
    print(f"Saved: {out_img}")

if __name__ == "__main__":
    main()
