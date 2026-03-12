#!/usr/bin/env python3
"""
E4: Disc Degeneration Simulation
==================================

Scoliosis → asymmetric disc loading → disc dehydration.
Models Pfirrmann grading (I-V) based on deviation angle.

Physics: Asymmetric axial load → nucleus pulposus dehydration → HU decrease.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass, gaussian_filter
from pathlib import Path
import json
import gc

# Pfirrmann Grading (MRI-based, adapted to CT HU proxy)
PFIRRMANN = {
    1: {"label": "Grade I (Normal)", "hu": 80, "height_ratio": 1.0},
    2: {"label": "Grade II (Mild)", "hu": 65, "height_ratio": 0.9},
    3: {"label": "Grade III (Moderate)", "hu": 50, "height_ratio": 0.75},
    4: {"label": "Grade IV (Severe)", "hu": 40, "height_ratio": 0.6},
    5: {"label": "Grade V (Collapsed)", "hu": 35, "height_ratio": 0.3},
}

def find_disc_spaces(volume, bone_threshold=200):
    """
    Find intervertebral disc spaces by detecting gaps between bone regions
    along the Z-axis (axial slices).
    """
    # Profile of bone content per slice
    bone_profile = np.sum(volume > bone_threshold, axis=(0, 1))
    
    # Smooth
    from scipy.ndimage import uniform_filter1d
    smooth_profile = uniform_filter1d(bone_profile.astype(float), size=5)
    
    # Find local minima (disc spaces)
    disc_spaces = []
    threshold = np.median(smooth_profile) * 0.5
    
    in_gap = False
    gap_start = 0
    
    for z in range(1, len(smooth_profile) - 1):
        if smooth_profile[z] < threshold and not in_gap:
            in_gap = True
            gap_start = z
        elif smooth_profile[z] >= threshold and in_gap:
            in_gap = False
            gap_center = (gap_start + z) // 2
            gap_width = z - gap_start
            if gap_width > 2 and gap_width < 30:  # Reasonable disc width
                disc_spaces.append({"z": gap_center, "width": gap_width, 
                                    "z_start": gap_start, "z_end": z})
    
    return disc_spaces

def apply_degeneration(volume, disc_space, grade, scoliosis_deviation=0.0):
    """Apply disc degeneration to a specific disc space."""
    p = PFIRRMANN[grade]
    z_s, z_e = disc_space["z_start"], disc_space["z_end"]
    z_center = disc_space["z"]
    
    # Define disc region (low density between vertebrae)
    disc_slab = volume[:, :, z_s:z_e]
    
    # Disc is soft tissue (< 100 HU, > -50 HU) in the axial center
    disc_mask = (disc_slab > -50) & (disc_slab < 150)
    
    if not np.any(disc_mask):
        return volume
    
    # Height collapse: reduce the active slice range
    new_width = max(2, int(disc_space["width"] * p["height_ratio"]))
    collapse = disc_space["width"] - new_width
    
    # Density change (dehydration)
    target_hu = p["hu"]
    noise = np.random.normal(target_hu, 5, np.count_nonzero(disc_mask)).astype(np.int16)
    disc_slab[disc_mask] = noise
    
    # Asymmetric degeneration based on scoliosis
    if abs(scoliosis_deviation) > 5:
        # Concave side gets more compression → more degeneration
        x_center = volume.shape[0] // 2
        for x in range(disc_slab.shape[0]):
            # More degeneration on concave side
            lateral_factor = 1.0 + 0.3 * (x - x_center) / x_center * np.sign(scoliosis_deviation)
            col_mask = disc_mask[x, :, :]
            if np.any(col_mask):
                vals = disc_slab[x, :, :][col_mask].astype(float)
                vals *= lateral_factor
                disc_slab[x, :, :][col_mask] = vals.astype(np.int16)
    
    volume[:, :, z_s:z_e] = disc_slab
    return volume

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
    angle = 60
    
    ct_path = root / f"scoliosis_cobb{angle}.nii.gz"
    if not ct_path.exists():
        print(f"Error: {ct_path} not found")
        return
    
    print("Loading scoliotic volume...")
    nii = nib.load(ct_path)
    vol = np.asanyarray(nii.dataobj).astype(np.int16)
    
    # Find disc spaces
    print("Detecting intervertebral disc spaces...")
    disc_spaces = find_disc_spaces(vol)
    print(f"  Found {len(disc_spaces)} disc spaces")
    
    if len(disc_spaces) == 0:
        print("No disc spaces found. Exiting.")
        return
    
    # Assign Pfirrmann grades based on position (apex = worst)
    # Apex of scoliosis typically has worst disc degeneration
    z_values = [d["z"] for d in disc_spaces]
    z_apex = (min(z_values) + max(z_values)) // 2  # Approximate apex
    
    results = []
    vol_original = vol.copy()
    
    for i, disc in enumerate(disc_spaces):
        # Grade based on distance from apex (closer = worse)
        dist_from_apex = abs(disc["z"] - z_apex)
        max_dist = max(abs(z_values[-1] - z_apex), abs(z_values[0] - z_apex))
        
        if max_dist > 0:
            norm_dist = dist_from_apex / max_dist
        else:
            norm_dist = 0.5
        
        # Near apex: Grade 4-5, Far from apex: Grade 1-2
        grade = max(1, min(5, int(5 - norm_dist * 4)))
        
        print(f"  Disc {i+1} at Z={disc['z']}: {PFIRRMANN[grade]['label']}")
        vol = apply_degeneration(vol, disc, grade, scoliosis_deviation=angle)
        
        results.append({
            "disc_index": i + 1,
            "z_level": int(disc["z"]),
            "grade": grade,
            "grade_label": PFIRRMANN[grade]["label"],
            "target_hu": PFIRRMANN[grade]["hu"]
        })
    
    # Save modified volume
    out_path = root / f"scoliosis_cobb{angle}_disc_degen.nii.gz"
    nib.save(nib.Nifti1Image(vol, nii.affine), out_path)
    print(f"Saved: {out_path}")
    
    # Visualization: Sagittal comparison
    print("Generating visualization...")
    x_mid = vol.shape[0] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    # Find Z range with discs
    z_range_min = max(0, min(z_values) - 30)
    z_range_max = min(vol.shape[2], max(z_values) + 30)
    
    # Original sagittal
    sag_orig = vol_original[x_mid, :, z_range_min:z_range_max].T
    axes[0].imshow(sag_orig, cmap='gray', origin='lower', vmin=-200, vmax=1200)
    axes[0].set_title("Original (Healthy Discs)", fontsize=13, fontweight='bold')
    
    # Degenerated sagittal
    sag_degen = vol[x_mid, :, z_range_min:z_range_max].T
    axes[1].imshow(sag_degen, cmap='gray', origin='lower', vmin=-200, vmax=1200)
    axes[1].set_title("Degenerated (Pfirrmann Grading)", fontsize=13, fontweight='bold')
    
    # Annotate disc levels
    for r in results:
        z_rel = r["z_level"] - z_range_min
        if 0 <= z_rel < sag_degen.shape[0]:
            color = ['green', 'yellowgreen', 'gold', 'orange', 'red'][r["grade"]-1]
            axes[1].axhline(y=z_rel, color=color, alpha=0.5, linewidth=1, linestyle='--')
            axes[1].text(sag_degen.shape[1]+2, z_rel, f"G{r['grade']}", 
                        fontsize=8, color=color, va='center')
    
    # Difference map
    diff = (sag_orig.astype(float) - sag_degen.astype(float))
    axes[2].imshow(diff, cmap='RdBu_r', origin='lower', vmin=-100, vmax=100)
    axes[2].set_title("Density Difference\n(Red = density loss)", fontsize=13, fontweight='bold')
    
    plt.suptitle(f"E4: Disc Degeneration (Cobb {angle}°)\nPfirrmann Grading: Apex = Worst Degeneration",
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    out_img = root / "disc_degeneration_visualization.png"
    plt.savefig(str(out_img), dpi=150, bbox_inches='tight')
    print(f"Saved: {out_img}")
    
    # Save results JSON
    with open(root / "disc_degeneration_grades.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
