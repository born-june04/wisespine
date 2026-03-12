#!/usr/bin/env python3
"""
E6: Quantitative Validation
=============================

Compares synthetic CT density distributions against literature values.
Generates HU histograms, QQ plots, and KS-test results.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import json

# Literature reference HU ranges (from radiology textbooks)
LITERATURE_RANGES = {
    "Cortical Bone": {"range": (400, 1900), "peak": 1000, "std": 200},
    "Cancellous Bone": {"range": (100, 400), "peak": 250, "std": 80},
    "Soft Tissue (Muscle)": {"range": (35, 80), "peak": 50, "std": 15},
    "Fat": {"range": (-120, -50), "peak": -90, "std": 20},
    "Hematoma (Acute)": {"range": (30, 70), "peak": 50, "std": 10},
    "Metal (Titanium)": {"range": (2000, 3500), "peak": 3000, "std": 200},
    "Air": {"range": (-1000, -800), "peak": -950, "std": 30},
    "Disc (Normal)": {"range": (50, 90), "peak": 70, "std": 15},
}

def extract_tissue_distributions(volume, bone_threshold=200):
    """Extract HU distributions for different tissue types."""
    flat = volume.flatten().astype(float)
    
    distributions = {}
    
    # Cortical bone
    cortical = flat[(flat > 400) & (flat < 2000)]
    if len(cortical) > 100:
        distributions["Cortical Bone"] = cortical
    
    # Cancellous bone
    cancellous = flat[(flat > 100) & (flat <= 400)]
    if len(cancellous) > 100:
        distributions["Cancellous Bone"] = cancellous
    
    # Soft tissue
    soft = flat[(flat > 20) & (flat <= 100)]
    if len(soft) > 100:
        distributions["Soft Tissue (Muscle)"] = soft
    
    # Fat
    fat = flat[(flat > -150) & (flat < -40)]
    if len(fat) > 100:
        distributions["Fat"] = fat
    
    # Metal
    metal = flat[flat > 2000]
    if len(metal) > 100:
        distributions["Metal (Titanium)"] = metal
    
    return distributions

def ks_test_against_literature(observed, tissue_name):
    """KS-test: observed distribution vs. theoretical (from literature)."""
    if tissue_name not in LITERATURE_RANGES:
        return None
    
    ref = LITERATURE_RANGES[tissue_name]
    
    # Sample if too large
    if len(observed) > 10000:
        observed = np.random.choice(observed, 10000, replace=False)
    
    # Generate reference distribution
    reference = np.random.normal(ref["peak"], ref["std"], len(observed))
    reference = reference[(reference >= ref["range"][0]) & (reference <= ref["range"][1])]
    
    if len(reference) < 100:
        return None
    
    # KS test
    ks_stat, p_value = stats.ks_2samp(observed, reference)
    
    return {
        "ks_statistic": round(float(ks_stat), 4),
        "p_value": round(float(p_value), 6),
        "observed_mean": round(float(np.mean(observed)), 1),
        "observed_std": round(float(np.std(observed)), 1),
        "literature_mean": ref["peak"],
        "literature_std": ref["std"],
        "match": "GOOD" if ks_stat < 0.1 else ("FAIR" if ks_stat < 0.3 else "POOR")
    }

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
    angle = 60
    
    # Analyze the causal (final) volume
    vol_path = root / f"scoliosis_cobb{angle}_causal.nii.gz"
    if not vol_path.exists():
        vol_path = root / f"scoliosis_cobb{angle}_postop.nii.gz"
    
    print(f"Loading volume: {vol_path}")
    nii = nib.load(vol_path)
    
    # Sample ROI to save memory (hardware region)
    hw_path = root / f"scoliosis_cobb{angle}_hardware.nii.gz"
    nii_hw = nib.load(hw_path)
    hw_data = np.asanyarray(nii_hw.dataobj).astype(np.uint8)
    hw_locs = np.argwhere(hw_data > 0)
    
    if len(hw_locs) > 0:
        z_min = max(0, hw_locs[:, 2].min() - 50)
        z_max = min(nii.shape[2], hw_locs[:, 2].max() + 50)
    else:
        z_min = nii.shape[2] // 4
        z_max = 3 * nii.shape[2] // 4
    
    del hw_data
    
    print(f"Extracting ROI z=[{z_min}:{z_max}]...")
    vol_roi = np.asanyarray(nii.dataobj[:, :, z_min:z_max]).astype(np.int16)
    
    # Extract distributions
    print("Extracting tissue distributions...")
    distributions = extract_tissue_distributions(vol_roi)
    
    # KS Tests
    print("\nRunning KS-Tests vs. Literature...")
    ks_results = {}
    for tissue_name, obs_data in distributions.items():
        result = ks_test_against_literature(obs_data, tissue_name)
        if result:
            ks_results[tissue_name] = result
            match_symbol = {"GOOD": "✅", "FAIR": "⚠️", "POOR": "❌"}[result["match"]]
            print(f"  {tissue_name}: KS={result['ks_statistic']:.4f}, "
                  f"p={result['p_value']:.4f} → {match_symbol} {result['match']}")
    
    # Visualization
    print("\nGenerating Visualization...")
    n_tissues = len(distributions)
    fig, axes = plt.subplots(2, max(3, (n_tissues + 1) // 2), figsize=(20, 10))
    axes = axes.flatten()
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    for idx, (tissue_name, obs_data) in enumerate(distributions.items()):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # Observed histogram
        sample = np.random.choice(obs_data, min(50000, len(obs_data)), replace=False)
        ax.hist(sample, bins=50, density=True, alpha=0.6, color=colors[idx % len(colors)],
                edgecolor='black', linewidth=0.3, label='Synthetic')
        
        # Literature reference
        if tissue_name in LITERATURE_RANGES:
            ref = LITERATURE_RANGES[tissue_name]
            x = np.linspace(ref["range"][0], ref["range"][1], 200)
            pdf = stats.norm.pdf(x, ref["peak"], ref["std"])
            ax.plot(x, pdf, 'k--', linewidth=2, label='Literature')
        
        # Annotation
        if tissue_name in ks_results:
            r = ks_results[tissue_name]
            ax.text(0.98, 0.95, f"KS={r['ks_statistic']:.3f}\n{r['match']}", 
                    transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_title(tissue_name, fontsize=11, fontweight='bold')
        ax.set_xlabel("HU")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    
    # Hide unused axes
    for idx in range(len(distributions), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("E6: Quantitative Validation\nSynthetic HU Distributions vs. Literature",
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    out_img = root / "quantitative_validation.png"
    plt.savefig(str(out_img), dpi=150, bbox_inches='tight')
    print(f"Saved: {out_img}")
    
    # Save results
    with open(root / "quantitative_validation_results.json", 'w') as f:
        json.dump(ks_results, f, indent=2)

if __name__ == "__main__":
    main()
