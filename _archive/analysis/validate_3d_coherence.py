#!/usr/bin/env python3
"""
V5: 3D Coherence & SSIM
=========================

Verifies spatial consistency of synthetic volumes:
1. SSIM between adjacent axial slices (should be ~0.95+)
2. Compare synthetic vs original SSIM profiles
3. MPR consistency (sagittal/coronal smoothness)
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import json

def compute_ssim_2d(img1, img2, data_range=None):
    """
    Simplified SSIM between two 2D images.
    Uses the standard SSIM formula with default constants.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    if data_range is None:
        data_range = max(img1.max() - img1.min(), img2.max() - img2.min())
        if data_range == 0:
            data_range = 1.0
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(ssim)

def compute_inter_slice_ssim(volume, sample_rate=5):
    """Compute SSIM between adjacent slices, sampled every N slices."""
    n_slices = volume.shape[2]
    ssim_values = []
    z_indices = []
    
    for z in range(0, n_slices - 1, sample_rate):
        sl1 = volume[:, :, z].astype(np.float64)
        sl2 = volume[:, :, z + 1].astype(np.float64)
        
        # Skip empty slices
        if np.std(sl1) < 1 or np.std(sl2) < 1:
            continue
        
        ssim = compute_ssim_2d(sl1, sl2, data_range=3000)
        ssim_values.append(ssim)
        z_indices.append(z)
    
    return z_indices, ssim_values

def compute_mpr_smoothness(volume, axis, position):
    """
    Compute smoothness of an MPR reconstruction.
    Smoothness = 1 - normalized gradient magnitude.
    """
    if axis == 'sagittal':
        plane = volume[position, :, :].astype(np.float64)
    elif axis == 'coronal':
        plane = volume[:, position, :].astype(np.float64)
    else:
        return 0.0
    
    # Gradient in the through-plane direction (z-axis in MPR)
    grad = np.diff(plane, axis=1)
    smoothness = 1.0 - (np.mean(np.abs(grad)) / max(1, np.std(plane)))
    
    return float(smoothness)

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
    angle = 60
    
    volumes = {
        "Original (Scoliosis)": root / f"scoliosis_cobb{angle}.nii.gz",
        "With Artifacts": root / f"scoliosis_cobb{angle}_artifacts.nii.gz",
        "Post-Op": root / f"scoliosis_cobb{angle}_postop.nii.gz",
        "Causal Response": root / f"scoliosis_cobb{angle}_causal.nii.gz",
    }
    
    results = {}
    all_ssim_profiles = {}
    
    for vol_name, vol_path in volumes.items():
        if not vol_path.exists():
            continue
        
        print(f"\nAnalyzing: {vol_name}")
        nii = nib.load(vol_path)
        data = np.asanyarray(nii.dataobj).astype(np.int16)
        shape = data.shape
        
        # 1. Inter-slice SSIM
        print("  Computing inter-slice SSIM...")
        z_idx, ssim_vals = compute_inter_slice_ssim(data, sample_rate=10)
        
        mean_ssim = np.mean(ssim_vals) if ssim_vals else 0
        min_ssim = np.min(ssim_vals) if ssim_vals else 0
        
        all_ssim_profiles[vol_name] = (z_idx, ssim_vals)
        
        print(f"    Mean SSIM: {mean_ssim:.4f}")
        print(f"    Min SSIM:  {min_ssim:.4f}")
        print(f"    Slices with SSIM < 0.90: {sum(1 for s in ssim_vals if s < 0.90)}/{len(ssim_vals)}")
        
        # 2. MPR Smoothness
        x_mid = shape[0] // 2
        y_mid = shape[1] // 2
        
        sag_smooth = compute_mpr_smoothness(data, 'sagittal', x_mid)
        cor_smooth = compute_mpr_smoothness(data, 'coronal', y_mid)
        
        print(f"    Sagittal Smoothness: {sag_smooth:.4f}")
        print(f"    Coronal Smoothness: {cor_smooth:.4f}")
        
        results[vol_name] = {
            "mean_ssim": round(mean_ssim, 4),
            "min_ssim": round(min_ssim, 4),
            "low_ssim_count": sum(1 for s in ssim_vals if s < 0.90),
            "total_slices": len(ssim_vals),
            "sagittal_smoothness": round(sag_smooth, 4),
            "coronal_smoothness": round(cor_smooth, 4),
            "coherent": mean_ssim > 0.90 and sag_smooth > 0.5
        }
        
        del data
    
    # Visualization
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: SSIM profiles
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    for i, (name, (z_idx, ssim_vals)) in enumerate(all_ssim_profiles.items()):
        axes[0, 0].plot(z_idx, ssim_vals, '-', color=colors[i % len(colors)], 
                       alpha=0.7, linewidth=1.5, label=name[:20])
    axes[0, 0].axhline(y=0.90, color='red', linestyle='--', alpha=0.5, label='Threshold (0.90)')
    axes[0, 0].set_title("Inter-Slice SSIM Profile", fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel("Z Slice")
    axes[0, 0].set_ylabel("SSIM")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylim(0.7, 1.01)
    
    # Plot 2: Mean SSIM comparison
    names = list(results.keys())
    mean_ssims = [results[n]["mean_ssim"] for n in names]
    bar_colors = ['#2ecc71' if results[n]["coherent"] else '#e74c3c' for n in names]
    axes[0, 1].bar(range(len(names)), mean_ssims, color=bar_colors, edgecolor='black')
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels([n[:15] for n in names], rotation=30, ha='right', fontsize=9)
    axes[0, 1].axhline(y=0.90, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_title("Mean Inter-Slice SSIM", fontsize=13, fontweight='bold')
    axes[0, 1].set_ylabel("SSIM")
    axes[0, 1].set_ylim(0.85, 1.0)
    
    # Plot 3: MPR Smoothness
    sag_vals = [results[n]["sagittal_smoothness"] for n in names]
    cor_vals = [results[n]["coronal_smoothness"] for n in names]
    x = np.arange(len(names))
    w = 0.35
    axes[1, 0].bar(x - w/2, sag_vals, w, label='Sagittal', color='#3498db', edgecolor='black')
    axes[1, 0].bar(x + w/2, cor_vals, w, label='Coronal', color='#e67e22', edgecolor='black')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([n[:15] for n in names], rotation=30, ha='right', fontsize=9)
    axes[1, 0].set_title("MPR Smoothness", fontsize=13, fontweight='bold')
    axes[1, 0].set_ylabel("Smoothness (1.0 = perfect)")
    axes[1, 0].legend()
    
    # Plot 4: Summary table
    axes[1, 1].axis('off')
    table_data = []
    for n in names:
        r = results[n]
        status = "✅ PASS" if r["coherent"] else "❌ FAIL"
        table_data.append([n[:20], f"{r['mean_ssim']:.4f}", f"{r['min_ssim']:.4f}",
                          f"{r['sagittal_smoothness']:.3f}", status])
    
    table = axes[1, 1].table(cellText=table_data,
                              colLabels=["Volume", "Mean SSIM", "Min SSIM", "Sag. Smooth", "Status"],
                              cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title("3D Coherence Summary", fontsize=13, fontweight='bold')
    
    plt.suptitle("V5: 3D Coherence & SSIM Validation\nVerifying Spatial Consistency of Synthetic Volumes",
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    out_img = root / "validation_3d_coherence.png"
    plt.savefig(str(out_img), dpi=150, bbox_inches='tight')
    print(f"Saved: {out_img}")
    
    with open(root / "validation_3d_coherence_results.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
