#!/usr/bin/env python3
"""
Deep Physical Simulation Review (Memory-Optimized)
====================================================
Processes one volume at a time using memory-mapped IO.
Generates 6 review panels for exhaustive physics assessment.
"""

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json, gc

ROOT = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
OUT = ROOT / "deep_review"
OUT.mkdir(exist_ok=True)

CLINICAL_HU = {
    "Air":(-1024,-900), "Fat":(-120,-50), "Water":(-10,10),
    "Soft Tissue":(20,80), "Cancellous Bone":(100,400),
    "Cortical Bone":(400,1900), "Titanium":(2500,3500),
}

def load_slice_axial(fname, z):
    nii = nib.load(str(ROOT / fname))
    return np.asarray(nii.dataobj[:, :, z]).astype(np.int16), nii.shape

def load_slice_sagittal(fname, x):
    nii = nib.load(str(ROOT / fname))
    return np.asarray(nii.dataobj[x, :, :]).astype(np.int16), nii.shape

def load_slice_coronal(fname, y):
    nii = nib.load(str(ROOT / fname))
    return np.asarray(nii.dataobj[:, y, :]).astype(np.int16), nii.shape

def get_shape(fname):
    return nib.load(str(ROOT / fname)).shape

def load_subvolume(fname, z_range):
    """Load a z-slab to avoid full volume memory."""
    nii = nib.load(str(ROOT / fname))
    return np.asarray(nii.dataobj[:, :, z_range[0]:z_range[1]]).astype(np.int16)


# =========================================================================
# REVIEW 1: Multi-Plane All Stages
# =========================================================================
def review_multiplane():
    print("\n=== REVIEW 1: Multi-Plane Visual Comparison ===")
    stages = [
        ("Scoliosis (Base)", "scoliosis_cobb60.nii.gz"),
        ("+ Hardware", "scoliosis_cobb60_hardware.nii.gz"),
        ("+ Tumors", "scoliosis_cobb60_tumors.nii.gz"),
        ("+ Artifacts", "scoliosis_cobb60_artifacts.nii.gz"),
        ("+ Post-Op", "scoliosis_cobb60_postop.nii.gz"),
        ("+ Causal", "scoliosis_cobb60_causal.nii.gz"),
    ]
    
    fig, axes = plt.subplots(6, 3, figsize=(18, 36))
    fig.suptitle("Physical Simulation Review: Every Stage × 3 Planes\n"
                 "Bone Window (W=2000, C=400)", fontsize=16, fontweight='bold', y=0.995)
    
    vmin, vmax = -600, 1400  # Bone window
    
    for row, (label, fname) in enumerate(stages):
        path = ROOT / fname
        if not path.exists():
            for c in range(3): 
                axes[row, c].text(0.5, 0.5, "NOT FOUND", ha='center', va='center')
            continue
        
        shape = get_shape(fname)
        mid_x, mid_y, mid_z = shape[0]//2, shape[1]//2, shape[2]//2
        
        ax_sl, _ = load_slice_axial(fname, mid_z)
        axes[row, 0].imshow(ax_sl.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[row, 0].set_title(f"{label}\nAxial z={mid_z}", fontsize=10)
        axes[row, 0].axis('off')
        del ax_sl
        
        sag_sl, _ = load_slice_sagittal(fname, mid_x)
        axes[row, 1].imshow(sag_sl.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        axes[row, 1].set_title(f"Sagittal x={mid_x}", fontsize=10)
        axes[row, 1].axis('off')
        del sag_sl
        
        cor_sl, _ = load_slice_coronal(fname, mid_y)
        axes[row, 2].imshow(cor_sl.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        axes[row, 2].set_title(f"Coronal y={mid_y}", fontsize=10)
        axes[row, 2].axis('off')
        del cor_sl
        gc.collect()
    
    plt.tight_layout()
    plt.savefig(str(OUT / "review1_multiplane_all_stages.png"), dpi=150, bbox_inches='tight')
    plt.close(); gc.collect()
    print(f"  Saved: {OUT}/review1_multiplane_all_stages.png")


# =========================================================================
# REVIEW 2: HU Distributions
# =========================================================================
def review_hu_distributions():
    print("\n=== REVIEW 2: HU Distribution Analysis ===")
    stages = [
        ("Scoliosis", "scoliosis_cobb60.nii.gz"),
        ("Hardware", "scoliosis_cobb60_hardware.nii.gz"),
        ("Artifacts", "scoliosis_cobb60_artifacts.nii.gz"),
        ("Post-Op", "scoliosis_cobb60_postop.nii.gz"),
        ("Causal", "scoliosis_cobb60_causal.nii.gz"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("HU Distribution: Do Densities Match Clinical Reality?", fontsize=14, fontweight='bold')
    
    stats = {}
    for idx, (label, fname) in enumerate(stages):
        path = ROOT / fname
        if not path.exists(): continue
        ax = axes[idx // 3, idx % 3]
        
        shape = get_shape(fname)
        # Sample 20 evenly-spaced slices
        z_samples = np.linspace(0, shape[2]-1, 20, dtype=int)
        all_vals = []
        for z in z_samples:
            sl, _ = load_slice_axial(fname, int(z))
            all_vals.append(sl.flatten()[::5])  # subsample
            del sl
        sample = np.concatenate(all_vals)
        
        ax.hist(sample, bins=300, range=(-1024, 3500), density=True,
                alpha=0.7, color='steelblue', edgecolor='none')
        
        colors = ['#e8d44d', '#ff9800', '#2196F3', '#f44336', '#4CAF50', '#9C27B0', '#795548']
        for i, (tissue, (lo, hi)) in enumerate(CLINICAL_HU.items()):
            ax.axvspan(lo, hi, alpha=0.12, color=colors[i])
            ax.text((lo+hi)/2, ax.get_ylim()[1]*0.9 - i*ax.get_ylim()[1]*0.07,
                   tissue, fontsize=6, ha='center', color=colors[i], fontweight='bold')
        
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel("HU"); ax.set_xlim(-1100, 3600)
        
        # Stats
        bone = sample[sample > 200]
        soft = sample[(sample > -50) & (sample < 100)]
        metal = sample[sample > 2500]
        stats[label] = {
            "bone_mean": round(float(np.mean(bone)), 1) if len(bone)>0 else 0,
            "bone_std": round(float(np.std(bone)), 1) if len(bone)>0 else 0,
            "soft_mean": round(float(np.mean(soft)), 1) if len(soft)>0 else 0,
            "metal_vox": int(len(metal)),
            "metal_mean": round(float(np.mean(metal)), 1) if len(metal)>0 else 0,
            "range": [int(sample.min()), int(sample.max())],
        }
        del sample, all_vals; gc.collect()
    
    # Summary panel
    ax = axes[1, 2]; ax.axis('off')
    lines = ["HU Statistics\n"]
    for s, v in stats.items():
        lines.append(f"{s}: Bone μ={v['bone_mean']}±{v['bone_std']}, "
                    f"Soft μ={v['soft_mean']}, Metal={v['metal_vox']}vox")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=8,
           va='top', fontfamily='monospace', bbox=dict(facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(str(OUT / "review2_hu_distributions.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    with open(OUT / "review2_hu_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {OUT}/review2_hu_distributions.png")
    
    # Clinical check
    print("\n  --- Clinical HU Check ---")
    for s, v in stats.items():
        issues = []
        if v["bone_mean"] < 200 or v["bone_mean"] > 1500:
            issues.append(f"Bone μ={v['bone_mean']} outside [200,1500]")
        if v["metal_vox"] > 0 and (v["metal_mean"] < 2500 or v["metal_mean"] > 3500):
            issues.append(f"Metal μ={v['metal_mean']} outside [2500,3500]")
        print(f"    {s}: {'✅ PASS' if not issues else '⚠️ ' + '; '.join(issues)}")
    return stats


# =========================================================================
# REVIEW 3: Hardware Placement
# =========================================================================
def review_hardware():
    print("\n=== REVIEW 3: Hardware Placement ===")
    fname_base = "scoliosis_cobb60.nii.gz"
    fname_hw = "scoliosis_cobb60_hardware.nii.gz"
    
    shape = get_shape(fname_hw)
    
    # Find z-slices with hardware by scanning
    hw_z_profile = []
    for z in range(0, shape[2], 5):
        sl_hw, _ = load_slice_axial(fname_hw, z)
        sl_base, _ = load_slice_axial(fname_base, z)
        n_hw = int(np.sum((sl_hw > 2500) & (sl_base < 2500)))
        hw_z_profile.append((z, n_hw))
        del sl_hw, sl_base
    
    # Find best z
    hw_z_profile.sort(key=lambda x: x[1], reverse=True)
    best_z = hw_z_profile[0][0] if hw_z_profile[0][1] > 0 else shape[2]//2
    total_hw_vox = sum(x[1] for x in hw_z_profile)
    
    print(f"  Best hardware slice: z={best_z} ({hw_z_profile[0][1]} voxels)")
    print(f"  Total hardware voxels (sampled): ~{total_hw_vox * 5}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Hardware Placement: Are Screws in Vertebral Pedicles?", fontsize=14, fontweight='bold')
    
    # Load just the slices we need
    sl_base, _ = load_slice_axial(fname_base, best_z)
    sl_hw, _ = load_slice_axial(fname_hw, best_z)
    hw_mask = (sl_hw > 2500) & (sl_base < 2500)
    
    # Check if hardware is near bone
    from scipy.ndimage import binary_dilation
    bone_mask_2d = sl_base > 200
    bone_dilated = binary_dilation(bone_mask_2d, iterations=3)
    near_bone = hw_mask & bone_dilated
    pct = np.sum(near_bone) / max(1, np.sum(hw_mask)) * 100
    
    # Row 1: Axial
    vmin, vmax = -200, 2000
    axes[0, 0].imshow(sl_base.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 0].set_title(f"Original (z={best_z})", fontsize=11); axes[0, 0].axis('off')
    
    axes[0, 1].imshow(sl_hw.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 1].set_title("With Hardware", fontsize=11); axes[0, 1].axis('off')
    
    # Overlay
    base_norm = np.clip((sl_base.T + 200) / 2200, 0, 1)
    rgb = np.stack([base_norm]*3, axis=-1)
    rgb[hw_mask.T] = [1, 0, 0]
    axes[0, 2].imshow(rgb, origin='lower')
    axes[0, 2].set_title("Overlay (red=hardware)", fontsize=11); axes[0, 2].axis('off')
    del sl_base, sl_hw, hw_mask, bone_mask_2d, bone_dilated, near_bone
    
    # Row 2: Sagittal, coronal, stats
    mid_x, mid_y = shape[0]//2, shape[1]//2
    sag_base, _ = load_slice_sagittal(fname_base, mid_x)
    sag_hw, _ = load_slice_sagittal(fname_hw, mid_x)
    sag_mask = (sag_hw > 2500) & (sag_base < 2500)
    base_s = np.clip((sag_base.T + 200) / 2200, 0, 1)
    rgb_s = np.stack([base_s]*3, axis=-1)
    rgb_s[sag_mask.T] = [1, 0, 0]
    axes[1, 0].imshow(rgb_s, origin='lower', aspect='auto')
    axes[1, 0].set_title(f"Sagittal x={mid_x}", fontsize=11); axes[1, 0].axis('off')
    del sag_base, sag_hw, sag_mask
    
    cor_base, _ = load_slice_coronal(fname_base, mid_y)
    cor_hw, _ = load_slice_coronal(fname_hw, mid_y)
    cor_mask = (cor_hw > 2500) & (cor_base < 2500)
    base_c = np.clip((cor_base.T + 200) / 2200, 0, 1)
    rgb_c = np.stack([base_c]*3, axis=-1)
    rgb_c[cor_mask.T] = [1, 0, 0]
    axes[1, 1].imshow(rgb_c, origin='lower', aspect='auto')
    axes[1, 1].set_title(f"Coronal y={mid_y}", fontsize=11); axes[1, 1].axis('off')
    del cor_base, cor_hw, cor_mask
    
    axes[1, 2].axis('off')
    axes[1, 2].text(0.1, 0.9,
        f"Hardware Report\n{'='*30}\n\n"
        f"HW voxels at best z: {hw_z_profile[0][1]}\n"
        f"Estimated total: ~{total_hw_vox*5}\n"
        f"% near bone: {pct:.1f}%\n"
        f"Valid placement: {'✅' if pct > 80 else '❌'}\n",
        transform=axes[1, 2].transAxes, fontsize=11, va='top', fontfamily='monospace',
        bbox=dict(facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(str(OUT / "review3_hardware_placement.png"), dpi=150, bbox_inches='tight')
    plt.close(); gc.collect()
    print(f"  Saved: {OUT}/review3_hardware_placement.png")
    return {"pct_near_bone": round(pct, 1), "valid": pct > 80}


# =========================================================================
# REVIEW 4+5: Post-Op + Artifacts (combined for memory)
# =========================================================================
def review_postop_and_artifacts():
    print("\n=== REVIEW 4: Post-Op & Causal Changes ===")
    
    shape = get_shape("scoliosis_cobb60.nii.gz")
    mid_z = shape[2] // 2
    mid_x = shape[0] // 2
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle("Post-Op + Causal Response: Laminectomy → Hematoma → Edema → Halo",
                fontsize=14, fontweight='bold')
    
    ww, wl = 1500, 300
    vmin, vmax = wl - ww/2, wl + ww/2
    
    # Row 1: Axial comparison at mid_z
    for col, (label, fname) in enumerate([
        ("Pre-Op", "scoliosis_cobb60.nii.gz"),
        ("Post-Op", "scoliosis_cobb60_postop.nii.gz"),
        ("Causal", "scoliosis_cobb60_causal.nii.gz"),
    ]):
        sl, _ = load_slice_axial(fname, mid_z)
        axes[0, col].imshow(sl.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, col].set_title(f"{label} (z={mid_z})", fontsize=11)
        axes[0, col].axis('off')
        if col == 0:
            pre_sl = sl.copy()
        elif col == 2:
            causal_sl = sl.copy()
        del sl
    
    # Difference map
    diff_sl = causal_sl.astype(np.float32) - pre_sl.astype(np.float32)
    im = axes[0, 3].imshow(diff_sl.T, cmap='RdBu_r', vmin=-100, vmax=100, origin='lower')
    axes[0, 3].set_title("ΔHU (Causal - PreOp)", fontsize=11)
    axes[0, 3].axis('off')
    plt.colorbar(im, ax=axes[0, 3], shrink=0.8, label='ΔHU')
    del pre_sl, causal_sl, diff_sl
    
    # Row 2: Sagittal + artifact comparison
    for col, (label, fname) in enumerate([
        ("Pre-Op Sag", "scoliosis_cobb60.nii.gz"),
        ("Hardware Sag", "scoliosis_cobb60_hardware.nii.gz"),
        ("Artifact Sag", "scoliosis_cobb60_artifacts.nii.gz"),
    ]):
        sl, _ = load_slice_sagittal(fname, mid_x)
        axes[1, col].imshow(sl.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        axes[1, col].set_title(label, fontsize=11)
        axes[1, col].axis('off')
        del sl
    
    # Artifact difference
    sl_hw, _ = load_slice_sagittal("scoliosis_cobb60_hardware.nii.gz", mid_x)
    sl_art, _ = load_slice_sagittal("scoliosis_cobb60_artifacts.nii.gz", mid_x)
    art_diff = sl_art.astype(np.float32) - sl_hw.astype(np.float32)
    im2 = axes[1, 3].imshow(art_diff.T, cmap='RdBu_r', vmin=-200, vmax=200, origin='lower', aspect='auto')
    axes[1, 3].set_title("Artifact ΔHU (streaks)", fontsize=11)
    axes[1, 3].axis('off')
    plt.colorbar(im2, ax=axes[1, 3], shrink=0.8, label='ΔHU')
    
    # Count streak polarities
    bright = int(np.sum(art_diff > 50))
    dark = int(np.sum(art_diff < -50))
    print(f"  Artifact streaks (single slice): bright={bright}, dark={dark}")
    print(f"  Both polarities: {'✅' if bright > 0 and dark > 0 else '❌'}")
    
    del sl_hw, sl_art, art_diff; gc.collect()
    plt.tight_layout()
    plt.savefig(str(OUT / "review4_postop_artifacts.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUT}/review4_postop_artifacts.png")
    return {"bright_streaks": bright, "dark_streaks": dark, "both": bright > 0 and dark > 0}


# =========================================================================
# REVIEW 5: Scoliosis Progression
# =========================================================================
def review_scoliosis():
    print("\n=== REVIEW 5: Scoliosis Deformation Quality ===")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle("Scoliosis: Cobb 20°→40°→60°\nAre vertebral bodies preserved? Is curvature progressive?",
                fontsize=14, fontweight='bold')
    
    vmin, vmax = -600, 1400
    result = {}
    
    for row, angle in enumerate([20, 40, 60]):
        fname = f"scoliosis_cobb{angle}.nii.gz"
        path = ROOT / fname
        if not path.exists():
            for c in range(3):
                axes[row, c].text(0.5, 0.5, "NOT FOUND", ha='center', va='center')
            continue
        
        shape = get_shape(fname)
        mid_x, mid_y, mid_z = shape[0]//2, shape[1]//2, shape[2]//2
        
        ax_sl, _ = load_slice_axial(fname, mid_z)
        axes[row, 0].imshow(ax_sl.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[row, 0].set_title(f"Cobb {angle}° — Axial", fontsize=11)
        axes[row, 0].axis('off')
        del ax_sl
        
        cor, _ = load_slice_coronal(fname, mid_y)
        axes[row, 1].imshow(cor.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        axes[row, 1].set_title(f"Cobb {angle}° — Coronal (curvature)", fontsize=11)
        axes[row, 1].axis('off')
        
        # Measure lateral deviation from coronal
        bone_profile = np.sum(cor > 200, axis=0)  # bone per z
        com_per_z = []
        for z in range(cor.shape[1]):
            col = cor[:, z]
            bone_px = np.argwhere(col > 200)
            if len(bone_px) > 10:
                com_per_z.append(float(np.mean(bone_px)))
        
        if com_per_z:
            lat_dev = max(com_per_z) - min(com_per_z)
            result[f"cobb{angle}"] = round(lat_dev, 1)
            print(f"  Cobb {angle}°: Lateral deviation = {lat_dev:.1f} px")
        del cor
        
        sag, _ = load_slice_sagittal(fname, mid_x)
        axes[row, 2].imshow(sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        axes[row, 2].set_title(f"Cobb {angle}° — Sagittal", fontsize=11)
        axes[row, 2].axis('off')
        del sag
        gc.collect()
    
    plt.tight_layout()
    plt.savefig(str(OUT / "review5_scoliosis_quality.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUT}/review5_scoliosis_quality.png")
    return result


# =========================================================================
# REVIEW 6: Tumor Realism
# =========================================================================
def review_tumors():
    print("\n=== REVIEW 6: Tumor Simulation ===")
    fname_base = "scoliosis_cobb60.nii.gz"
    fname_tumor = "scoliosis_cobb60_tumors.nii.gz"
    
    if not (ROOT / fname_tumor).exists():
        print("  Tumor volume not found.")
        return {}
    
    shape = get_shape(fname_tumor)
    
    # Find z with max tumor effect
    best_z = shape[2] // 2
    max_diff = 0
    for z in range(0, shape[2], 10):
        sl_b, _ = load_slice_axial(fname_base, z)
        sl_t, _ = load_slice_axial(fname_tumor, z)
        d = np.sum(np.abs(sl_t.astype(np.float32) - sl_b.astype(np.float32)) > 20)
        if d > max_diff:
            max_diff = d
            best_z = z
        del sl_b, sl_t
    
    print(f"  Best tumor slice: z={best_z} ({max_diff} changed voxels)")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Tumor Simulation: Osteolytic/Blastic Lesions", fontsize=14, fontweight='bold')
    
    vmin, vmax = -200, 1400
    sl_b, _ = load_slice_axial(fname_base, best_z)
    sl_t, _ = load_slice_axial(fname_tumor, best_z)
    
    axes[0].imshow(sl_b.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title(f"Before Tumor (z={best_z})", fontsize=11); axes[0].axis('off')
    
    axes[1].imshow(sl_t.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title("With Tumor", fontsize=11); axes[1].axis('off')
    
    diff = sl_t.astype(np.float32) - sl_b.astype(np.float32)
    im = axes[2].imshow(diff.T, cmap='RdBu_r', vmin=-200, vmax=200, origin='lower')
    axes[2].set_title("Tumor ΔHU", fontsize=11); axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], shrink=0.8, label='ΔHU')
    
    # Lytic = bone destruction (negative) / Blastic = bone formation (positive)
    lytic = int(np.sum(diff < -50))
    blastic = int(np.sum(diff > 50))
    print(f"  Lytic voxels (bone destruction): {lytic}")
    print(f"  Blastic voxels (bone formation): {blastic}")
    
    del sl_b, sl_t, diff
    plt.tight_layout()
    plt.savefig(str(OUT / "review6_tumor_realism.png"), dpi=150, bbox_inches='tight')
    plt.close(); gc.collect()
    print(f"  Saved: {OUT}/review6_tumor_realism.png")
    return {"lytic": lytic, "blastic": blastic, "best_z": best_z}


if __name__ == "__main__":
    print("=" * 60)
    print("DEEP PHYSICAL SIMULATION REVIEW (Memory-Optimized)")
    print("=" * 60)
    
    results = {}
    review_multiplane()
    results["hu"] = review_hu_distributions()
    results["hardware"] = review_hardware()
    results["postop_artifacts"] = review_postop_and_artifacts()
    results["scoliosis"] = review_scoliosis()
    results["tumors"] = review_tumors()
    
    with open(OUT / "deep_review_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("ALL REVIEWS COMPLETE")
    print(f"Figures saved to: {OUT}")
    print("=" * 60)
