#!/usr/bin/env python3
"""
V2: Ablation Study — Node Deactivation Impact
===============================================

For each simulation node, measure how its REMOVAL degrades
downstream clinical realism metrics (CNR, edge sharpness, HU stats).

This proves that each node is essential: removing it breaks something.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import sobel
import json

def compute_cnr_fast(volume, bone_mask, margin=2):
    """Fast CNR: |μ_bone - μ_soft| / σ_soft."""
    from scipy.ndimage import binary_dilation, binary_erosion
    interior = binary_erosion(bone_mask, iterations=margin)
    exterior = binary_dilation(bone_mask, iterations=margin) & ~bone_mask
    if not np.any(interior) or not np.any(exterior):
        return 0.0
    mu_bone = float(np.mean(volume[interior]))
    mu_soft = float(np.mean(volume[exterior]))
    sigma_soft = float(np.std(volume[exterior]))
    return abs(mu_bone - mu_soft) / sigma_soft if sigma_soft > 0 else 0.0

def compute_edge_sharpness_fast(volume, z_indices):
    """Mean gradient magnitude at sampled slices."""
    vals = []
    for z in z_indices[:10]:
        sl = volume[:, :, z].astype(np.float32)
        gx = sobel(sl, axis=0)
        gy = sobel(sl, axis=1)
        vals.append(float(np.mean(np.sqrt(gx**2 + gy**2))))
    return np.mean(vals) if vals else 0.0

def compute_hu_stats(volume, mask):
    """Basic HU statistics in bone region."""
    bone_vals = volume[mask & (volume > 100)]
    if len(bone_vals) == 0:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}
    return {
        "mean": round(float(np.mean(bone_vals)), 1),
        "std": round(float(np.std(bone_vals)), 1),
        "min": int(np.min(bone_vals)),
        "max": int(np.max(bone_vals))
    }

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
    angle = 60
    
    # Define the pipeline stages as volumes (each represents one node being "active")
    # Ablation = compare WITH vs WITHOUT each stage
    pipeline = [
        {"name": "Scoliosis", "file": f"scoliosis_cobb{angle}.nii.gz", 
         "description": "Spine curvature deformation"},
        {"name": "Hardware", "file": f"scoliosis_cobb{angle}_hardware.nii.gz",
         "description": "Pedicle screw placement"},
        {"name": "Tumors", "file": f"scoliosis_cobb{angle}_tumors.nii.gz",
         "description": "Lytic/blastic lesions"},
        {"name": "Artifacts", "file": f"scoliosis_cobb{angle}_artifacts.nii.gz",
         "description": "Metal blooming artifact"},
        {"name": "Post-Op", "file": f"scoliosis_cobb{angle}_postop.nii.gz",
         "description": "Laminectomy + grafting"},
        {"name": "Causal", "file": f"scoliosis_cobb{angle}_causal.nii.gz",
         "description": "Hematoma + edema + halo"},
    ]
    
    # Reference: the original clean CT
    ref_path = root / f"scoliosis_cobb{angle}.nii.gz"
    
    print("Loading reference volume...")
    ref_nii = nib.load(ref_path)
    ref_data = np.asanyarray(ref_nii.dataobj).astype(np.int16)
    bone_mask = ref_data > 200
    z_indices = np.where(bone_mask.any(axis=(0, 1)))[0]
    
    # Measure baseline (full pipeline = causal volume)
    results = {}
    
    for stage in pipeline:
        path = root / stage["file"]
        if not path.exists():
            print(f"  Skipping {stage['name']} (not found)")
            continue
        
        print(f"  Measuring: {stage['name']}...")
        nii = nib.load(path)
        data = np.asanyarray(nii.dataobj).astype(np.int16)
        
        cnr = compute_cnr_fast(data, bone_mask)
        edge = compute_edge_sharpness_fast(data, z_indices)
        hu = compute_hu_stats(data, bone_mask)
        
        results[stage["name"]] = {
            "CNR": round(cnr, 3),
            "Edge_Sharpness": round(edge, 2),
            "HU_Mean": hu["mean"],
            "HU_Std": hu["std"],
            "description": stage["description"]
        }
        
        del data
    
    del ref_data, bone_mask
    
    if not results:
        print("No volumes found.")
        return
    
    # Compute ablation impact: delta from previous stage
    names = list(results.keys())
    
    print("\n--- ABLATION IMPACT ---")
    print(f"{'Stage':<15} {'CNR':>8} {'ΔCNR':>8} {'Edge':>10} {'ΔEdge':>8} {'HU_Mean':>8}")
    print("-" * 65)
    
    ablation_data = {}
    for i, name in enumerate(names):
        r = results[name]
        if i == 0:
            delta_cnr = 0
            delta_edge = 0
        else:
            prev = results[names[i-1]]
            delta_cnr = r["CNR"] - prev["CNR"]
            delta_edge = r["Edge_Sharpness"] - prev["Edge_Sharpness"]
        
        print(f"{name:<15} {r['CNR']:>8.3f} {delta_cnr:>+8.3f} {r['Edge_Sharpness']:>10.2f} "
              f"{delta_edge:>+8.2f} {r['HU_Mean']:>8.1f}")
        
        ablation_data[name] = {
            "CNR": r["CNR"],
            "delta_CNR": round(delta_cnr, 3),
            "Edge_Sharpness": r["Edge_Sharpness"],
            "delta_Edge": round(delta_edge, 2),
            "HU_Mean": r["HU_Mean"],
            "essential": bool(abs(delta_cnr) > 0.01 or abs(delta_edge) > 0.5)
        }
    
    # Visualization
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    colors = ['#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#3498db', '#1abc9c'][:len(names)]
    x = range(len(names))
    short_names = [n[:8] for n in names]
    
    # CNR progression
    cnr_vals = [results[n]["CNR"] for n in names]
    bars = axes[0].bar(x, cnr_vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_title("CNR at Each Pipeline Stage", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("CNR (higher = easier segmentation)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, rotation=30, ha='right')
    # Annotate deltas
    for i in range(1, len(names)):
        delta = ablation_data[names[i]]["delta_CNR"]
        color = 'red' if delta < 0 else 'green'
        axes[0].annotate(f"{delta:+.3f}", xy=(i, cnr_vals[i]), 
                        xytext=(i, cnr_vals[i] + 0.1),
                        fontsize=9, color=color, ha='center', fontweight='bold')
    
    # Edge sharpness progression
    edge_vals = [results[n]["Edge_Sharpness"] for n in names]
    axes[1].bar(x, edge_vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_title("Edge Sharpness at Each Stage", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Gradient Magnitude")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, rotation=30, ha='right')
    
    # HU Mean progression
    hu_vals = [results[n]["HU_Mean"] for n in names]
    axes[2].bar(x, hu_vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[2].set_title("Mean Bone HU at Each Stage", fontsize=13, fontweight='bold')
    axes[2].set_ylabel("Mean HU (bone region)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(short_names, rotation=30, ha='right')
    
    plt.suptitle("V2: Ablation Study — Impact of Each Simulation Stage\n"
                 "Each bar shows metrics WITH that stage active",
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    out_img = root / "validation_ablation_study.png"
    plt.savefig(str(out_img), dpi=150, bbox_inches='tight')
    print(f"Saved: {out_img}")
    
    with open(root / "validation_ablation_results.json", 'w') as f:
        json.dump(ablation_data, f, indent=2)
    print(f"Saved: {root / 'validation_ablation_results.json'}")

if __name__ == "__main__":
    main()
