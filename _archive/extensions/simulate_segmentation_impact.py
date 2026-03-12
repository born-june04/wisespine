#!/usr/bin/env python3
"""
E1: Segmentation Impact Analysis (Layer 5 of Causal DAG)
=========================================================

Connects the simulation pipeline to its original purpose:
evaluating how each pathology/artifact degrades segmentation.

Since TotalSegmentator may not be available, we use PROXY METRICS:
1. Contrast-to-Noise Ratio (CNR) at vertebral boundaries
2. Edge Sharpness (Gradient Magnitude) at bone-soft tissue interface
3. Boundary Integrity Score per pathology

These metrics predict segmentation difficulty without running TS.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, gaussian_filter
from pathlib import Path
import json

def compute_cnr(volume, bone_mask, margin=3):
    """
    Contrast-to-Noise Ratio at bone boundaries.
    CNR = |μ_bone - μ_soft| / σ_soft
    
    Higher CNR → easier segmentation.
    """
    from scipy.ndimage import binary_dilation, binary_erosion
    
    # Interior bone (eroded)
    interior = binary_erosion(bone_mask, iterations=margin)
    # Boundary band
    boundary = binary_dilation(bone_mask, iterations=margin) & ~bone_mask
    
    if not np.any(interior) or not np.any(boundary):
        return 0.0
    
    mu_bone = np.mean(volume[interior])
    mu_soft = np.mean(volume[boundary])
    sigma_soft = np.std(volume[boundary])
    
    if sigma_soft == 0:
        return 0.0
    
    return abs(mu_bone - mu_soft) / sigma_soft

def compute_edge_sharpness(volume, bone_mask, margin=2):
    """
    Edge sharpness at bone boundary (mean gradient magnitude).
    Higher = sharper edges = easier segmentation.
    """
    from scipy.ndimage import binary_dilation
    
    boundary_band = binary_dilation(bone_mask, iterations=margin) & ~binary_dilation(bone_mask, iterations=-1)
    # Approximate: just use dilation minus erosion
    from scipy.ndimage import binary_erosion
    boundary_band = binary_dilation(bone_mask, iterations=margin) & ~binary_erosion(bone_mask, iterations=margin)
    
    if not np.any(boundary_band):
        return 0.0
    
    # Compute gradient magnitude on a representative slice
    # Full 3D gradient is expensive. Sample slices.
    z_indices = np.where(bone_mask.any(axis=(0, 1)))[0]
    if len(z_indices) == 0:
        return 0.0
    
    # Sample 10 slices
    sample_z = z_indices[np.linspace(0, len(z_indices)-1, min(10, len(z_indices)), dtype=int)]
    
    sharpness_vals = []
    for z in sample_z:
        sl = volume[:, :, z].astype(np.float32)
        gx = sobel(sl, axis=0)
        gy = sobel(sl, axis=1)
        grad_mag = np.sqrt(gx**2 + gy**2)
        
        boundary_sl = boundary_band[:, :, z]
        if np.any(boundary_sl):
            sharpness_vals.append(np.mean(grad_mag[boundary_sl]))
    
    return np.mean(sharpness_vals) if sharpness_vals else 0.0

def compute_boundary_integrity(volume, bone_mask):
    """
    Boundary integrity: what fraction of the bone boundary has
    a clear density jump (> 100 HU difference to neighbors).
    """
    from scipy.ndimage import binary_dilation, binary_erosion
    
    outer = binary_dilation(bone_mask, iterations=1) & ~bone_mask
    inner = bone_mask & ~binary_erosion(bone_mask, iterations=1)
    
    if not np.any(inner) or not np.any(outer):
        return 0.0
    
    # Sample slices for efficiency
    z_indices = np.where(bone_mask.any(axis=(0, 1)))[0]
    if len(z_indices) == 0:
        return 0.0
    
    sample_z = z_indices[np.linspace(0, len(z_indices)-1, min(10, len(z_indices)), dtype=int)]
    
    integrity_vals = []
    for z in sample_z:
        inner_val = np.mean(volume[:, :, z][inner[:, :, z]]) if np.any(inner[:, :, z]) else 0
        outer_val = np.mean(volume[:, :, z][outer[:, :, z]]) if np.any(outer[:, :, z]) else 0
        jump = abs(inner_val - outer_val)
        integrity_vals.append(1.0 if jump > 100 else jump / 100.0)
    
    return np.mean(integrity_vals)

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
    angle = 60
    
    # Define volumes to compare
    volumes = {
        "Clean (Scoliosis Only)": root / f"scoliosis_cobb{angle}.nii.gz",
        "With Hardware": root / f"scoliosis_cobb{angle}_artifacts.nii.gz",
        "With Tumors": root / f"scoliosis_cobb{angle}_tumors.nii.gz",
        "Post-Op (Surgery)": root / f"scoliosis_cobb{angle}_postop.nii.gz",
        "Causal Response": root / f"scoliosis_cobb{angle}_causal.nii.gz",
    }
    
    # We need a bone mask for boundary analysis
    # Use segmentation or threshold-based mask from clean volume
    print("Loading clean volume for bone mask...")
    nii_clean = nib.load(volumes["Clean (Scoliosis Only)"])
    clean_data = np.asanyarray(nii_clean.dataobj).astype(np.int16)
    
    # Simple bone mask: > 200 HU
    bone_mask = clean_data > 200
    
    # Hardware mask for additional analysis
    hw_path = root / f"scoliosis_cobb{angle}_hardware.nii.gz"
    
    results = {}
    
    for name, path in volumes.items():
        if not path.exists():
            print(f"  Skipping {name} (not found)")
            continue
            
        print(f"\nAnalyzing: {name}...")
        
        # Load volume chunk-wise to save memory
        nii = nib.load(path)
        data = np.asanyarray(nii.dataobj).astype(np.int16)
        
        cnr = compute_cnr(data, bone_mask)
        sharpness = compute_edge_sharpness(data, bone_mask)
        integrity = compute_boundary_integrity(data, bone_mask)
        
        results[name] = {
            "CNR": round(float(cnr), 2),
            "Edge_Sharpness": round(float(sharpness), 2),
            "Boundary_Integrity": round(float(integrity), 3),
            "Predicted_Dice_Impact": "Low" if cnr > 5 else ("Medium" if cnr > 2 else "High")
        }
        
        print(f"  CNR: {cnr:.2f}")
        print(f"  Edge Sharpness: {sharpness:.2f}")
        print(f"  Boundary Integrity: {integrity:.3f}")
        
        del data
    
    del clean_data, bone_mask
    
    # Visualization
    print("\nGenerating Visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    names = list(results.keys())
    colors = ['#2ecc71', '#e74c3c', '#9b59b6', '#3498db', '#f39c12'][:len(names)]
    
    # Plot 1: CNR
    cnr_vals = [results[n]["CNR"] for n in names]
    bars1 = axes[0].bar(range(len(names)), cnr_vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_title("Contrast-to-Noise Ratio (CNR)\nat Bone Boundaries", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("CNR (higher = easier segmentation)")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels([n.split('(')[0].strip() for n in names], rotation=30, ha='right', fontsize=9)
    axes[0].axhline(y=5, color='green', linestyle='--', alpha=0.5, label='Easy threshold')
    axes[0].axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Hard threshold')
    axes[0].legend(fontsize=8)
    
    # Plot 2: Edge Sharpness
    sharp_vals = [results[n]["Edge_Sharpness"] for n in names]
    bars2 = axes[1].bar(range(len(names)), sharp_vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_title("Edge Sharpness\n(Gradient Magnitude at Boundary)", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Mean Gradient (higher = sharper)")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels([n.split('(')[0].strip() for n in names], rotation=30, ha='right', fontsize=9)
    
    # Plot 3: Boundary Integrity
    integ_vals = [results[n]["Boundary_Integrity"] for n in names]
    bars3 = axes[2].bar(range(len(names)), integ_vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[2].set_title("Boundary Integrity Score\n(Fraction with clear HU jump)", fontsize=13, fontweight='bold')
    axes[2].set_ylabel("Score (1.0 = perfect)")
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels([n.split('(')[0].strip() for n in names], rotation=30, ha='right', fontsize=9)
    axes[2].set_ylim(0, 1.1)
    
    plt.suptitle("Layer 5: Segmentation Impact Analysis\nHow Each Pathology Degrades Boundary Detection",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_img = root / "segmentation_impact_analysis.png"
    plt.savefig(str(out_img), dpi=150, bbox_inches='tight')
    print(f"Saved: {out_img}")
    
    # Save metrics JSON
    out_json = root / "segmentation_impact_metrics.json"
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_json}")

if __name__ == "__main__":
    main()
