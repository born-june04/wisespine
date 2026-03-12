#!/usr/bin/env python3
"""
E5: Spinal Canal Compromise
=============================

Measures canal cross-section area pre/post pathology.
Computes stenosis percentage for fracture and tumor scenarios.

Physics: Burst fracture → retropulsed fragment into canal.
         Tumor → epidural mass effect.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, label
from pathlib import Path
import json

def measure_canal_area(volume_slice, bone_threshold=200):
    """
    Measure spinal canal area in an axial slice.
    Canal = enclosed space within the vertebral body (posterior) 
    bounded by bone (> threshold).
    """
    bone = volume_slice > bone_threshold
    
    # Fill holes in bone → the filled region minus bone = canal
    filled = binary_fill_holes(bone)
    canal = filled & ~bone
    
    # The canal is the largest connected component in the interior
    labeled, n = label(canal)
    if n == 0:
        return 0, np.zeros_like(canal, dtype=bool)
    
    # Find largest component (= canal, not noise)
    sizes = [np.sum(labeled == i) for i in range(1, n+1)]
    largest = np.argmax(sizes) + 1
    canal_mask = labeled == largest
    
    return int(np.sum(canal_mask)), canal_mask

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
    angle = 60
    
    # Compare pre and post pathology canal areas
    clean_path = root / f"scoliosis_cobb{angle}.nii.gz"
    postop_path = root / f"scoliosis_cobb{angle}_postop.nii.gz"
    causal_path = root / f"scoliosis_cobb{angle}_causal.nii.gz"
    hw_path = root / f"scoliosis_cobb{angle}_hardware.nii.gz"
    
    print("Loading volumes (sliced)...")
    nii_clean = nib.load(clean_path)
    nii_hw = nib.load(hw_path)
    
    hw_full = np.asanyarray(nii_hw.dataobj).astype(np.uint8)
    hw_locs = np.argwhere(hw_full > 0)
    
    if len(hw_locs) == 0:
        print("No hardware found")
        return
    
    z_min = hw_locs[:, 2].min()
    z_max = hw_locs[:, 2].max()
    del hw_full
    
    # Sample slices across hardware region
    n_samples = min(20, z_max - z_min)
    sample_z = np.linspace(z_min, z_max, n_samples, dtype=int)
    
    volumes_to_compare = {
        "Pre-Op (Clean)": clean_path,
        "Post-Op (Surgery)": postop_path,
        "Causal Response": causal_path,
    }
    
    results = {}
    canal_masks_per_vol = {}
    
    for vol_name, vol_path in volumes_to_compare.items():
        if not vol_path.exists():
            continue
            
        print(f"\nMeasuring canal: {vol_name}")
        nii = nib.load(vol_path)
        
        areas = []
        masks = {}
        for z in sample_z:
            sl = np.asanyarray(nii.dataobj[:, :, int(z)]).astype(np.int16)
            area, mask = measure_canal_area(sl)
            areas.append(area)
            masks[int(z)] = mask
        
        mean_area = np.mean(areas)
        results[vol_name] = {
            "areas": [int(a) for a in areas],
            "mean_area": round(float(mean_area), 1),
            "z_indices": [int(z) for z in sample_z]
        }
        canal_masks_per_vol[vol_name] = masks
        print(f"  Mean canal area: {mean_area:.1f} voxels²")
    
    # Compute stenosis
    if "Pre-Op (Clean)" in results:
        pre_area = results["Pre-Op (Clean)"]["mean_area"]
        for vol_name in results:
            post_area = results[vol_name]["mean_area"]
            stenosis = (1 - post_area / pre_area) * 100 if pre_area > 0 else 0
            results[vol_name]["stenosis_pct"] = round(float(stenosis), 1)
            print(f"  {vol_name}: Stenosis = {stenosis:.1f}%")
    
    # Visualization
    print("\nGenerating Canal Visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Pick representative slices
    z_rep = sample_z[len(sample_z)//2]
    
    vol_names_list = list(volumes_to_compare.keys())
    
    for col, vol_name in enumerate(vol_names_list):
        if vol_name not in canal_masks_per_vol:
            continue
        
        vol_path = volumes_to_compare[vol_name]
        nii = nib.load(vol_path)
        sl = np.asanyarray(nii.dataobj[:, :, int(z_rep)]).astype(np.int16)
        
        # Row 0: Axial slice with canal overlay
        axes[0, col].imshow(sl.T, cmap='gray', origin='lower', vmin=-200, vmax=1500)
        if int(z_rep) in canal_masks_per_vol[vol_name]:
            canal_m = canal_masks_per_vol[vol_name][int(z_rep)]
            overlay = np.ma.masked_where(~canal_m, canal_m.astype(float))
            axes[0, col].imshow(overlay.T, cmap='Reds', alpha=0.5, origin='lower')
        
        area = results[vol_name]["mean_area"]
        stenosis = results[vol_name].get("stenosis_pct", 0)
        axes[0, col].set_title(f"{vol_name}\nCanal Area: {area:.0f} vox²\nStenosis: {stenosis:.1f}%",
                              fontsize=11, fontweight='bold')
    
    # Row 1: Area profile along Z
    colors = ['#2ecc71', '#3498db', '#f39c12']
    for col, vol_name in enumerate(vol_names_list):
        if vol_name not in results:
            continue
        r = results[vol_name]
        axes[1, col].fill_between(r["z_indices"], r["areas"], alpha=0.3, color=colors[col])
        axes[1, col].plot(r["z_indices"], r["areas"], 'o-', color=colors[col], markersize=3)
        axes[1, col].set_xlabel("Z Slice")
        axes[1, col].set_ylabel("Canal Area (voxels²)")
        axes[1, col].set_title(f"Canal Area Profile", fontsize=11)
    
    plt.suptitle("E5: Spinal Canal Compromise Analysis\nRed overlay = detected canal cross-section",
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    out_img = root / "canal_compromise_analysis.png"
    plt.savefig(str(out_img), dpi=150, bbox_inches='tight')
    print(f"Saved: {out_img}")
    
    with open(root / "canal_compromise_metrics.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
