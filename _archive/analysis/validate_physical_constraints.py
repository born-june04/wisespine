#!/usr/bin/env python3
"""
V4: Physical Constraint Verification
======================================

Checks that fundamental physical laws are preserved across simulations:
1. Mass Conservation: Total mass (sum of HU) should be approximately conserved
2. Volume Conservation: Bone voxel count should be preserved during deformation
3. Anatomical Plausibility: No impossible density values
4. Boundary Integrity: No sharp discontinuities at simulation boundaries
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Physical constraints (from CT physics)
CONSTRAINTS = {
    "min_bone_hu": 100,      # Minimum valid bone density
    "max_bone_hu": 3500,     # Maximum (titanium)
    "min_air_hu": -1024,     # CT minimum
    "max_tissue_hu": 100,    # Maximum soft tissue
    "min_tissue_hu": -200,   # Minimum tissue (excluding air)
}

def check_mass_conservation(vol_pre, vol_post, name):
    """Check total HU mass before/after transformation."""
    mass_pre = float(np.sum(vol_pre.astype(np.float64)))
    mass_post = float(np.sum(vol_post.astype(np.float64)))
    
    if mass_pre == 0:
        return {"conserved": False, "ratio": 0, "delta_pct": 100}
    
    ratio = mass_post / mass_pre
    delta_pct = abs(1 - ratio) * 100
    
    return {
        "mass_pre": round(mass_pre / 1e9, 3),
        "mass_post": round(mass_post / 1e9, 3),
        "ratio": round(ratio, 4),
        "delta_pct": round(delta_pct, 2),
        "conserved": delta_pct < 10,  # <10% change = reasonable
        "stage": name
    }

def check_volume_conservation(vol_pre, vol_post, threshold=200):
    """Check bone voxel count preservation."""
    bone_pre = int(np.sum(vol_pre > threshold))
    bone_post = int(np.sum(vol_post > threshold))
    
    if bone_pre == 0:
        return {"conserved": False, "ratio": 0}
    
    ratio = bone_post / bone_pre
    delta_pct = abs(1 - ratio) * 100
    
    return {
        "bone_voxels_pre": bone_pre,
        "bone_voxels_post": bone_post,
        "ratio": round(ratio, 4),
        "delta_pct": round(delta_pct, 2),
        "conserved": delta_pct < 20  # Surgery can remove bone, so allow 20%
    }

def check_anatomical_plausibility(volume, name):
    """Check for impossible density values."""
    violations = {}
    
    # Check 1: No values below CT minimum
    below_min = int(np.sum(volume < -1024))
    violations["below_CT_min"] = below_min
    
    # Check 2: No values above reasonable maximum (unless metal)
    above_max = int(np.sum(volume > 4000))
    violations["above_reasonable_max"] = above_max
    
    # Check 3: Bone density in valid range (where bone exists)
    bone_mask = volume > 200
    if np.any(bone_mask):
        bone_vals = volume[bone_mask]
        violations["bone_mean_hu"] = round(float(np.mean(bone_vals)), 1)
        violations["bone_std_hu"] = round(float(np.std(bone_vals)), 1)
        violations["bone_in_valid_range"] = bool(
            np.mean(bone_vals) > 200 and np.mean(bone_vals) < 2000
        )
    
    # Check 4: No NaN or Inf
    violations["has_nan"] = bool(np.any(np.isnan(volume.astype(float))))
    violations["has_inf"] = bool(np.any(np.isinf(volume.astype(float))))
    
    # Overall plausibility
    violations["plausible"] = (
        below_min == 0 and 
        above_max == 0 and 
        not violations["has_nan"] and 
        not violations["has_inf"] and
        violations.get("bone_in_valid_range", True)
    )
    
    return violations

def check_boundary_smoothness(volume, z_indices):
    """Check for sharp discontinuities between adjacent slices."""
    diffs = []
    for i in range(len(z_indices) - 1):
        z1, z2 = z_indices[i], z_indices[i + 1]
        if z2 - z1 == 1:  # Adjacent
            sl1 = volume[:, :, z1].astype(np.float32)
            sl2 = volume[:, :, z2].astype(np.float32)
            diff = float(np.mean(np.abs(sl1 - sl2)))
            diffs.append(diff)
    
    if not diffs:
        return {"smooth": True, "mean_inter_slice_diff": 0, "max_inter_slice_diff": 0}
    
    return {
        "mean_inter_slice_diff": round(float(np.mean(diffs)), 2),
        "max_inter_slice_diff": round(float(np.max(diffs)), 2),
        "smooth": float(np.max(diffs)) < 200  # No massive jumps
    }

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
    angle = 60
    
    # Pipeline stages (sequential)
    stages = [
        ("Original→Scoliosis", f"scoliosis_cobb{angle}.nii.gz", None),
        ("Scoliosis→Hardware", f"scoliosis_cobb{angle}.nii.gz", f"scoliosis_cobb{angle}_artifacts.nii.gz"),
        ("Scoliosis→Tumors", f"scoliosis_cobb{angle}.nii.gz", f"scoliosis_cobb{angle}_tumors.nii.gz"),
        ("Scoliosis→PostOp", f"scoliosis_cobb{angle}.nii.gz", f"scoliosis_cobb{angle}_postop.nii.gz"),
        ("PostOp→Causal", f"scoliosis_cobb{angle}_postop.nii.gz", f"scoliosis_cobb{angle}_causal.nii.gz"),
    ]
    
    all_results = {
        "mass_conservation": [],
        "volume_conservation": [],
        "anatomical_plausibility": [],
        "boundary_smoothness": []
    }
    
    for stage_name, pre_file, post_file in stages:
        pre_path = root / pre_file
        
        if not pre_path.exists():
            continue
        
        print(f"\n--- {stage_name} ---")
        pre_data = np.asanyarray(nib.load(pre_path).dataobj).astype(np.int16)
        
        if post_file and (root / post_file).exists():
            post_data = np.asanyarray(nib.load(root / post_file).dataobj).astype(np.int16)
            
            # Mass conservation
            mc = check_mass_conservation(pre_data, post_data, stage_name)
            all_results["mass_conservation"].append(mc)
            status = "✅" if mc["conserved"] else "⚠️"
            print(f"  Mass: {status} ratio={mc['ratio']:.4f} (Δ={mc['delta_pct']:.2f}%)")
            
            # Volume conservation
            vc = check_volume_conservation(pre_data, post_data)
            vc["stage"] = stage_name
            all_results["volume_conservation"].append(vc)
            status = "✅" if vc["conserved"] else "⚠️"
            print(f"  Volume: {status} ratio={vc['ratio']:.4f} (Δ={vc['delta_pct']:.2f}%)")
            
            del post_data
        
        # Anatomical plausibility (for pre volume)
        ap = check_anatomical_plausibility(pre_data, stage_name)
        ap["stage"] = stage_name
        all_results["anatomical_plausibility"].append(ap)
        status = "✅" if ap["plausible"] else "❌"
        print(f"  Anatomy: {status} (NaN={ap['has_nan']}, Inf={ap['has_inf']}, "
              f"below_min={ap['below_CT_min']}, above_max={ap['above_reasonable_max']})")
        
        # Boundary smoothness
        z_indices = np.where((pre_data > 100).any(axis=(0, 1)))[0]
        sample_z = z_indices[::max(1, len(z_indices)//50)]  # Sample
        bs = check_boundary_smoothness(pre_data, sample_z)
        bs["stage"] = stage_name
        all_results["boundary_smoothness"].append(bs)
        status = "✅" if bs["smooth"] else "❌"
        print(f"  Smoothness: {status} mean_diff={bs['mean_inter_slice_diff']:.2f}")
        
        del pre_data
    
    # Visualization
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Mass conservation
    mc_data = all_results["mass_conservation"]
    if mc_data:
        names = [d["stage"][:15] for d in mc_data]
        ratios = [d["ratio"] for d in mc_data]
        colors = ['#2ecc71' if d["conserved"] else '#e74c3c' for d in mc_data]
        axes[0, 0].barh(range(len(names)), ratios, color=colors, edgecolor='black')
        axes[0, 0].axvline(x=1.0, color='blue', linestyle='--', label='Perfect (1.0)')
        axes[0, 0].set_yticks(range(len(names)))
        axes[0, 0].set_yticklabels(names, fontsize=9)
        axes[0, 0].set_xlabel("Mass Ratio (post/pre)")
        axes[0, 0].set_title("Mass Conservation", fontsize=13, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].set_xlim(0.8, 1.2)
    
    # Volume conservation
    vc_data = all_results["volume_conservation"]
    if vc_data:
        names = [d["stage"][:15] for d in vc_data]
        ratios = [d["ratio"] for d in vc_data]
        colors = ['#2ecc71' if d["conserved"] else '#e74c3c' for d in vc_data]
        axes[0, 1].barh(range(len(names)), ratios, color=colors, edgecolor='black')
        axes[0, 1].axvline(x=1.0, color='blue', linestyle='--', label='Perfect (1.0)')
        axes[0, 1].set_yticks(range(len(names)))
        axes[0, 1].set_yticklabels(names, fontsize=9)
        axes[0, 1].set_xlabel("Bone Volume Ratio")
        axes[0, 1].set_title("Volume Conservation", fontsize=13, fontweight='bold')
        axes[0, 1].legend()
    
    # Anatomical plausibility summary
    ap_data = all_results["anatomical_plausibility"]
    if ap_data:
        names = [d["stage"][:15] for d in ap_data]
        scores = [1.0 if d["plausible"] else 0.0 for d in ap_data]
        colors = ['#2ecc71' if s == 1 else '#e74c3c' for s in scores]
        axes[1, 0].barh(range(len(names)), scores, color=colors, edgecolor='black')
        axes[1, 0].set_yticks(range(len(names)))
        axes[1, 0].set_yticklabels(names, fontsize=9)
        axes[1, 0].set_xlabel("Plausibility (1=pass)")
        axes[1, 0].set_title("Anatomical Plausibility", fontsize=13, fontweight='bold')
        axes[1, 0].set_xlim(-0.1, 1.1)
    
    # Boundary smoothness
    bs_data = all_results["boundary_smoothness"]
    if bs_data:
        names = [d["stage"][:15] for d in bs_data]
        diffs = [d["mean_inter_slice_diff"] for d in bs_data]
        colors = ['#2ecc71' if d["smooth"] else '#e74c3c' for d in bs_data]
        axes[1, 1].barh(range(len(names)), diffs, color=colors, edgecolor='black')
        axes[1, 1].set_yticks(range(len(names)))
        axes[1, 1].set_yticklabels(names, fontsize=9)
        axes[1, 1].set_xlabel("Mean Inter-Slice Diff (HU)")
        axes[1, 1].set_title("Boundary Smoothness", fontsize=13, fontweight='bold')
    
    plt.suptitle("V4: Physical Constraint Verification\nGreen = Pass, Red = Violation",
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    out_img = root / "validation_physical_constraints.png"
    plt.savefig(str(out_img), dpi=150, bbox_inches='tight')
    print(f"Saved: {out_img}")
    
    with open(root / "validation_constraints_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

if __name__ == "__main__":
    main()
