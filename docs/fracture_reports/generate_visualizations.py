#!/usr/bin/env python3
"""
Enhanced Fracture Type Visualization Generator v2

Key improvements over v1:
  - Fracture location annotations (arrows, circles, labels)
  - Both axial AND sagittal views per type
  - Zoomed-in fracture region
  - Fracture mask overlay in red
  - Before/After with annotation
"""

import sys
import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.gridspec import GridSpec
from scipy.ndimage import (
    gaussian_filter, map_coordinates, binary_erosion, 
    binary_dilation, distance_transform_edt, label as ndlabel
)

sys.path.insert(0, '/gscratch/scrubbed/june0604/wisespine/pipeline')
from modules.ct_physics import (
    generate_hierarchical_fracture_surface,
    simulate_burst_retropulsion,
)

OUTPUT_DIR = '/gscratch/scrubbed/june0604/wisespine/docs/fracture_reports/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# DATA LOADING
# ============================================================

def load_verse_data():
    """Load VerSe CT and segmentation mask."""
    ct_path = '/gscratch/scrubbed/june0604/wisespine/VerSe/dataset-01training/rawdata/sub-verse503/sub-verse503_dir-ax_ct.nii.gz'
    mask_path = '/gscratch/scrubbed/june0604/wisespine/VerSe/dataset-01training/derivatives/sub-verse503/sub-verse503_dir-ax_seg-vert_msk.nii.gz'
    
    print(f"Loading CT: {ct_path}")
    ct_nii = nib.load(ct_path)
    ct_data = ct_nii.get_fdata().astype(np.float32)
    
    print(f"Loading Mask: {mask_path}")
    mask_nii = nib.load(mask_path)
    mask_data = mask_nii.get_fdata().astype(np.int32)
    
    print(f"  CT shape: {ct_data.shape}, range: [{ct_data.min():.0f}, {ct_data.max():.0f}]")
    print(f"  Mask shape: {mask_data.shape}, labels: {np.unique(mask_data)}")
    
    return ct_data, mask_data, ct_nii.affine


def extract_vertebra_slices(ct_data, mask_data, label=None):
    """Extract axial + sagittal slices for a vertebra, cropped around it."""
    labels = [l for l in np.unique(mask_data) if l > 0]
    if label is None:
        label = labels[len(labels)//2]
    
    print(f"  Using vertebra label: {label}")
    vert_mask = (mask_data == label)
    
    coords = np.where(vert_mask)
    if len(coords[0]) == 0:
        raise ValueError(f"No voxels for label {label}")
    
    cx, cy, cz = int(np.mean(coords[0])), int(np.mean(coords[1])), int(np.mean(coords[2]))
    
    pad = 35
    
    # Axial slice (X-Y plane at Z=cz)
    ax_ct = ct_data[:, :, cz].copy()
    ax_mask = vert_mask[:, :, cz].copy()
    ax_coords = np.where(ax_mask)
    if len(ax_coords[0]) > 0:
        r0 = max(0, ax_coords[0].min() - pad)
        r1 = min(ax_ct.shape[0], ax_coords[0].max() + pad)
        c0 = max(0, ax_coords[1].min() - pad)
        c1 = min(ax_ct.shape[1], ax_coords[1].max() + pad)
        ax_ct = ax_ct[r0:r1, c0:c1]
        ax_mask = ax_mask[r0:r1, c0:c1]
    
    # Sagittal slice (X-Z plane at Y=cy)
    sag_ct = ct_data[:, cy, :].copy()
    sag_mask = vert_mask[:, cy, :].copy()
    sag_coords = np.where(sag_mask)
    if len(sag_coords[0]) > 0:
        r0s = max(0, sag_coords[0].min() - pad)
        r1s = min(sag_ct.shape[0], sag_coords[0].max() + pad)
        c0s = max(0, sag_coords[1].min() - pad)
        c1s = min(sag_ct.shape[1], sag_coords[1].max() + pad)
        sag_ct = sag_ct[r0s:r1s, c0s:c1s]
        sag_mask = sag_mask[r0s:r1s, c0s:c1s]
    
    return {
        'axial_ct': ax_ct, 'axial_mask': ax_mask,
        'sagittal_ct': sag_ct, 'sagittal_mask': sag_mask,
        'label': label,
    }

# ============================================================
# FRACTURE SIMULATIONS — returns (fractured_ct, fracture_change_mask)
# ============================================================

def simulate_a1_wedge(ct, mask, severity=0.5, seed=42):
    """AO A1: Gradient compression on anterior column."""
    result = ct.copy()
    np.random.seed(seed)
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return result, np.zeros_like(mask, dtype=bool)
    
    x_min, x_max = coords[0].min(), coords[0].max()
    z_min, z_max = coords[1].min(), coords[1].max()
    H, W = ct.shape
    
    deformation = np.zeros((H, W, 2), dtype=np.float32)
    for x in range(x_min, x_max + 1):
        for z in range(z_min, z_max + 1):
            if not mask[x, z]:
                continue
            rel_z = (z - z_min) / (z_max - z_min + 1e-8)
            compression = severity * 10 * rel_z  # anterior gets more
            deformation[x, z, 1] = -compression
    
    deformation[..., 0] = gaussian_filter(deformation[..., 0], sigma=3)
    deformation[..., 1] = gaussian_filter(deformation[..., 1], sigma=3)
    
    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))
    coords_new = np.array([(y_grid + deformation[..., 0]).ravel(), 
                            (x_grid + deformation[..., 1]).ravel()])
    result = map_coordinates(ct, coords_new, order=1, mode='nearest').reshape(H, W)
    
    # Add endplate fracture line
    fracture_z = z_min + int((z_max - z_min) * 0.25)
    frac_mask, frac_field = generate_hierarchical_fracture_surface(
        ct.shape, fracture_z, roughness=severity * 0.7, seed=seed
    )
    combined = frac_mask & mask
    result[combined] *= (1 + frac_field[combined])
    
    # Mark changed regions
    change_mask = np.abs(result - ct) > 15
    change_mask = change_mask & mask
    
    return np.clip(result, -1000, 3000), change_mask


def simulate_a2_split(ct, mask, severity=0.5, seed=42):
    """AO A2: Coronal plane split through both endplates."""
    result = ct.copy()
    np.random.seed(seed)
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return result, np.zeros_like(mask, dtype=bool)
    
    x_min, x_max = coords[0].min(), coords[0].max()
    z_min, z_max = coords[1].min(), coords[1].max()
    x_mid = (x_min + x_max) // 2
    H, W = ct.shape
    
    deformation = np.zeros((H, W, 2), dtype=np.float32)
    split_gap = severity * 5
    for x in range(x_min, x_max + 1):
        for z in range(z_min, z_max + 1):
            if not mask[x, z]:
                continue
            if x < x_mid:
                deformation[x, z, 0] = -split_gap
            else:
                deformation[x, z, 0] = split_gap
    
    deformation[..., 0] = gaussian_filter(deformation[..., 0], sigma=2)
    deformation[..., 1] = gaussian_filter(deformation[..., 1], sigma=2)
    
    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))
    coords_new = np.array([(y_grid + deformation[..., 0]).ravel(),
                            (x_grid + deformation[..., 1]).ravel()])
    result = map_coordinates(ct, coords_new, order=1, mode='nearest').reshape(H, W)
    
    # Draw fracture line (horizontal)
    fracture_line_mask = np.zeros((H, W), dtype=bool)
    frac_width = 2 + int(severity * 3)
    for z in range(z_min, z_max + 1):
        noise = int(np.random.randn() * severity * 2)
        fx = x_mid + noise
        x_s = max(0, fx - frac_width // 2)
        x_e = min(H, fx + frac_width // 2 + 1)
        fracture_line_mask[x_s:x_e, z] = True
        result[x_s:x_e, z] *= 0.3  # dark fracture line
    
    change_mask = (np.abs(result - ct) > 15) & mask
    change_mask = change_mask | (fracture_line_mask & mask)
    
    return np.clip(result, -1000, 3000), change_mask


def simulate_a3_incomplete_burst(ct, mask, severity=0.5, seed=42):
    """AO A3: Uniform compression + limited posterior wall fracture."""
    result = ct.copy()
    np.random.seed(seed)
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return result, np.zeros_like(mask, dtype=bool)
    
    x_min, x_max = coords[0].min(), coords[0].max()
    z_min, z_max = coords[1].min(), coords[1].max()
    x_mid, z_mid = (x_min + x_max) // 2, (z_min + z_max) // 2
    H, W = ct.shape
    
    deformation = np.zeros((H, W, 2), dtype=np.float32)
    for x in range(x_min, x_max + 1):
        for z in range(z_min, z_max + 1):
            if not mask[x, z]:
                continue
            rel_z = (z - z_min) / (z_max - z_min + 1e-8)
            compression = severity * 6 * (1 - abs(2 * rel_z - 1))
            deformation[x, z, 1] = -compression
            rel_x = (x - x_mid) / (x_max - x_min + 1e-8)
            deformation[x, z, 0] = severity * 3 * rel_x
    
    deformation[..., 0] = gaussian_filter(deformation[..., 0], sigma=2.5)
    deformation[..., 1] = gaussian_filter(deformation[..., 1], sigma=2.5)
    
    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))
    coords_new = np.array([(y_grid + deformation[..., 0]).ravel(),
                            (x_grid + deformation[..., 1]).ravel()])
    result = map_coordinates(ct, coords_new, order=1, mode='nearest').reshape(H, W)
    
    # Posterior wall fracture (limited)
    post_thick = max(2, int((x_max - x_min) * 0.12))
    posterior = mask.copy()
    posterior[x_min + post_thick:, :] = False
    
    frac_mask, frac_field = generate_hierarchical_fracture_surface(
        ct.shape, z_mid, roughness=severity * 0.5, seed=seed + 100
    )
    posteror_frac = frac_mask & posterior
    result[posteror_frac] *= (1 + frac_field[posteror_frac])
    
    # Edge marking
    edge = binary_dilation(posterior, iterations=1) & ~posterior & mask
    result[edge] *= (0.6 + np.random.rand(edge.sum()) * 0.2)
    
    change_mask = (np.abs(result - ct) > 15) & mask
    change_mask = change_mask | posteror_frac
    
    return np.clip(result, -1000, 3000), change_mask


def simulate_a4_complete_burst(ct, mask, severity=0.5, seed=42):
    """AO A4: Explosive burst + retropulsion."""
    result, frag_mask = simulate_burst_retropulsion(
        ct, mask, severity=severity, canal_direction='posterior', seed=seed
    )
    change_mask = (np.abs(result - ct) > 15) | frag_mask
    change_mask = change_mask & (mask | frag_mask)
    return np.clip(result, -1000, 3000), change_mask

# ============================================================
# ENHANCED VISUALIZATION
# ============================================================

def annotate_fracture(ax, change_mask, mask, fracture_type, view='axial'):
    """Add red overlay + arrows + labels showing fracture location."""
    H, W = change_mask.shape
    
    # 1. Red overlay on fracture region
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    overlay[change_mask, 0] = 1.0   # Red
    overlay[change_mask, 3] = 0.35  # Semi-transparent
    ax.imshow(overlay, origin='lower', aspect='auto')
    
    # 2. Green contour of original mask
    ax.contour(mask.T if view == 'native' else mask, 
               colors='#00ff00', linewidths=0.6, levels=[0.5], 
               origin='lower', alpha=0.5)
    
    # 3. Find centroid of change region for arrow
    change_coords = np.where(change_mask)
    if len(change_coords[0]) == 0:
        return
    
    cx = int(np.mean(change_coords[1]))  # column (display x)
    cy = int(np.mean(change_coords[0]))  # row (display y)
    
    # Arrow from outside pointing to fracture
    arrow_labels = {
        'a1': ('Anterior\nWedge\nCompression', (cx + W * 0.25, cy - H * 0.2)),
        'a2': ('Coronal\nSplit Line', (cx + W * 0.25, cy)),
        'a3': ('Posterior Wall\nFracture\n(Limited)', (cx - W * 0.3, cy)),
        'a4': ('Retropulsion\nFragment →\nCanal', (cx - W * 0.3, cy - H * 0.15)),
    }
    
    if fracture_type in arrow_labels:
        label, start = arrow_labels[fracture_type]
        # Clamp arrow start to image bounds
        start = (max(5, min(W-5, start[0])), max(5, min(H-5, start[1])))
        
        ax.annotate(
            label,
            xy=(cx, cy), xytext=start,
            fontsize=8, color='yellow', fontweight='bold',
            ha='center', va='center',
            arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='yellow'),
        )


def create_fracture_report_figure(
    ax_orig, ax_mask, ax_frac, ax_change,
    sag_orig, sag_mask, sag_frac, sag_change,
    title, subtitle, fracture_type, save_path
):
    """Create comprehensive per-type figure with 3 rows."""
    fig = plt.figure(figsize=(20, 16), facecolor='#0a0a0a')
    fig.suptitle(title, fontsize=24, color='white', fontweight='bold', y=0.97)
    fig.text(0.5, 0.945, subtitle, fontsize=13, color='#bbbbbb', ha='center')
    
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.2,
                  left=0.04, right=0.96, top=0.92, bottom=0.03)
    
    vmin, vmax = -200, 1200
    
    # === ROW 1: AXIAL VIEWS ===
    row_label_y = 0.90
    fig.text(0.02, 0.82, 'AXIAL', fontsize=14, color='#00BCD4', fontweight='bold',
             rotation=90, va='center', ha='center')
    
    # 1a. Original axial
    a1 = fig.add_subplot(gs[0, 0])
    a1.imshow(ax_orig, cmap='bone', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
    a1.set_title('Original', color='#4CAF50', fontsize=12, fontweight='bold')
    a1.axis('off')
    
    # 1b. Fractured axial
    a2 = fig.add_subplot(gs[0, 1])
    a2.imshow(ax_frac, cmap='bone', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
    a2.set_title('Fractured', color='#F44336', fontsize=12, fontweight='bold')
    a2.axis('off')
    
    # 1c. Annotated fracture (fractured + red overlay + arrows)
    a3 = fig.add_subplot(gs[0, 2])
    a3.imshow(ax_frac, cmap='bone', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
    annotate_fracture(a3, ax_change, ax_mask, fracture_type, view='axial')
    a3.set_title('Fracture Location', color='#FF9800', fontsize=12, fontweight='bold')
    a3.axis('off')
    
    # 1d. Zoomed axial fracture region
    a4 = fig.add_subplot(gs[0, 3])
    # Find and zoom into fracture area
    ch_coords = np.where(ax_change)
    if len(ch_coords[0]) > 0:
        zpad = 10
        yr0 = max(0, ch_coords[0].min() - zpad)
        yr1 = min(ax_frac.shape[0], ch_coords[0].max() + zpad)
        xr0 = max(0, ch_coords[1].min() - zpad)
        xr1 = min(ax_frac.shape[1], ch_coords[1].max() + zpad)
        zoom_ct = ax_frac[yr0:yr1, xr0:xr1]
        zoom_change = ax_change[yr0:yr1, xr0:xr1]
        a4.imshow(zoom_ct, cmap='bone', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
        overlay = np.zeros((*zoom_ct.shape, 4), dtype=np.float32)
        overlay[zoom_change, 0] = 1.0
        overlay[zoom_change, 3] = 0.4
        a4.imshow(overlay, origin='lower', aspect='equal')
    else:
        a4.imshow(ax_frac, cmap='bone', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
    a4.set_title('Zoomed Fracture', color='#E91E63', fontsize=12, fontweight='bold')
    a4.axis('off')
    
    # === ROW 2: SAGITTAL VIEWS ===
    fig.text(0.02, 0.53, 'SAGITTAL', fontsize=14, color='#00BCD4', fontweight='bold',
             rotation=90, va='center', ha='center')
    
    # 2a. Original sagittal
    s1 = fig.add_subplot(gs[1, 0])
    s1.imshow(sag_orig, cmap='bone', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
    s1.set_title('Original', color='#4CAF50', fontsize=12, fontweight='bold')
    s1.axis('off')
    
    # 2b. Fractured sagittal
    s2 = fig.add_subplot(gs[1, 1])
    s2.imshow(sag_frac, cmap='bone', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
    s2.set_title('Fractured', color='#F44336', fontsize=12, fontweight='bold')
    s2.axis('off')
    
    # 2c. Annotated sagittal
    s3 = fig.add_subplot(gs[1, 2])
    s3.imshow(sag_frac, cmap='bone', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
    annotate_fracture(s3, sag_change, sag_mask, fracture_type, view='sagittal')
    s3.set_title('Fracture Location', color='#FF9800', fontsize=12, fontweight='bold')
    s3.axis('off')
    
    # 2d. Zoomed sagittal fracture
    s4 = fig.add_subplot(gs[1, 3])
    sch_coords = np.where(sag_change)
    if len(sch_coords[0]) > 0:
        zpad = 8
        yr0s = max(0, sch_coords[0].min() - zpad)
        yr1s = min(sag_frac.shape[0], sch_coords[0].max() + zpad)
        xr0s = max(0, sch_coords[1].min() - zpad)
        xr1s = min(sag_frac.shape[1], sch_coords[1].max() + zpad)
        zoom_sag = sag_frac[yr0s:yr1s, xr0s:xr1s]
        zoom_sag_ch = sag_change[yr0s:yr1s, xr0s:xr1s]
        s4.imshow(zoom_sag, cmap='bone', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
        overlay_s = np.zeros((*zoom_sag.shape, 4), dtype=np.float32)
        overlay_s[zoom_sag_ch, 0] = 1.0
        overlay_s[zoom_sag_ch, 3] = 0.4
        s4.imshow(overlay_s, origin='lower', aspect='equal')
    else:
        s4.imshow(sag_frac, cmap='bone', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
    s4.set_title('Zoomed Fracture', color='#E91E63', fontsize=12, fontweight='bold')
    s4.axis('off')
    
    # === ROW 3: ANALYSIS — Diff map + HU histogram + deformation magnitude ===
    fig.text(0.02, 0.22, 'ANALYSIS', fontsize=14, color='#00BCD4', fontweight='bold',
             rotation=90, va='center', ha='center')
    
    # 3a. Axial difference map
    d1 = fig.add_subplot(gs[2, 0])
    axial_diff = ax_frac.astype(float) - ax_orig.astype(float)
    im_d1 = d1.imshow(axial_diff, cmap='RdBu_r', vmin=-500, vmax=500, origin='lower', aspect='equal')
    d1.set_title('Axial ΔHU', color='#FF9800', fontsize=12, fontweight='bold')
    d1.axis('off')
    plt.colorbar(im_d1, ax=d1, fraction=0.046, pad=0.04, 
                 label='ΔHU', orientation='horizontal')
    
    # 3b. Sagittal difference map
    d2 = fig.add_subplot(gs[2, 1])
    sag_diff = sag_frac.astype(float) - sag_orig.astype(float)
    im_d2 = d2.imshow(sag_diff, cmap='RdBu_r', vmin=-500, vmax=500, origin='lower', aspect='equal')
    d2.set_title('Sagittal ΔHU', color='#FF9800', fontsize=12, fontweight='bold')
    d2.axis('off')
    plt.colorbar(im_d2, ax=d2, fraction=0.046, pad=0.04,
                 label='ΔHU', orientation='horizontal')
    
    # 3c. HU histogram comparison (bone region)
    d3 = fig.add_subplot(gs[2, 2])
    orig_hu = ax_orig[ax_mask].flatten()
    frac_hu = ax_frac[ax_mask].flatten()
    d3.hist(orig_hu, bins=60, alpha=0.6, color='#4CAF50', label='Original', density=True)
    d3.hist(frac_hu, bins=60, alpha=0.6, color='#F44336', label='Fractured', density=True)
    d3.set_xlabel('HU Value', color='white', fontsize=10)
    d3.set_ylabel('Density', color='white', fontsize=10)
    d3.set_title('Bone HU Distribution', color='white', fontsize=12, fontweight='bold')
    d3.legend(fontsize=9)
    d3.set_facecolor('#1a1a1a')
    d3.tick_params(colors='white')
    for sp in ['bottom', 'left']:
        d3.spines[sp].set_color('white')
    for sp in ['top', 'right']:
        d3.spines[sp].set_visible(False)
    
    # 3d. Stats summary box
    d4 = fig.add_subplot(gs[2, 3])
    d4.set_facecolor('#1a1a1a')
    d4.axis('off')
    
    # Compute stats
    bone_orig_mean = orig_hu.mean() if len(orig_hu) > 0 else 0
    bone_frac_mean = frac_hu.mean() if len(frac_hu) > 0 else 0
    frac_area = ax_change.sum()
    total_bone = ax_mask.sum()
    frac_pct = frac_area / total_bone * 100 if total_bone > 0 else 0
    hu_change = bone_frac_mean - bone_orig_mean
    
    stats_text = (
        f"📊 Quantitative Summary\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Bone HU (orig):  {bone_orig_mean:.0f} HU\n"
        f"Bone HU (frac):  {bone_frac_mean:.0f} HU\n"
        f"Mean ΔHU:        {hu_change:+.0f} HU\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Affected area:   {frac_area} px\n"
        f"Total bone:      {total_bone} px\n"
        f"Affected ratio:  {frac_pct:.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Max |ΔHU|:       {np.abs(axial_diff[ax_mask]).max():.0f}\n"
    )
    d4.text(0.1, 0.9, stats_text, transform=d4.transAxes, fontsize=10,
            color='#e0e0e0', fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#222222', edgecolor='#444444'))
    d4.set_title('Statistics', color='white', fontsize=12, fontweight='bold')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ Saved: {save_path}")


def create_severity_gallery(ax_orig, ax_mask, sag_orig, sag_mask,
                            fracture_func, type_name, save_path, seed=42):
    """Severity gallery: mild → moderate → severe, both axial + sagittal."""
    severities = [0.3, 0.6, 0.9]
    labels = ['Mild (0.3)', 'Moderate (0.6)', 'Severe (0.9)']
    colors = ['#FFC107', '#FF9800', '#F44336']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor='#0a0a0a')
    fig.suptitle(f'{type_name} — Severity Progression', fontsize=18, 
                 color='white', fontweight='bold', y=0.97)
    
    vmin, vmax = -200, 1200
    
    # Row 0: Axial
    axes[0, 0].imshow(ax_orig, cmap='bone', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
    axes[0, 0].set_title('Original (Axial)', color='#4CAF50', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Row 1: Sagittal
    axes[1, 0].imshow(sag_orig, cmap='bone', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
    axes[1, 0].set_title('Original (Sagittal)', color='#4CAF50', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    for i, (sev, lbl, col) in enumerate(zip(severities, labels, colors)):
        ax_frac, ax_ch = fracture_func(ax_orig, ax_mask, severity=sev, seed=seed)
        sag_frac, sag_ch = fracture_func(sag_orig, sag_mask, severity=sev, seed=seed)
        
        # Axial
        axes[0, i+1].imshow(ax_frac, cmap='bone', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
        # Red overlay
        ov = np.zeros((*ax_frac.shape, 4), dtype=np.float32)
        ov[ax_ch, 0] = 1.0; ov[ax_ch, 3] = 0.25
        axes[0, i+1].imshow(ov, origin='lower', aspect='equal')
        axes[0, i+1].set_title(f'{lbl} (Axial)', color=col, fontsize=11, fontweight='bold')
        axes[0, i+1].axis('off')
        
        # Sagittal
        axes[1, i+1].imshow(sag_frac, cmap='bone', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
        ov_s = np.zeros((*sag_frac.shape, 4), dtype=np.float32)
        ov_s[sag_ch, 0] = 1.0; ov_s[sag_ch, 3] = 0.25
        axes[1, i+1].imshow(ov_s, origin='lower', aspect='equal')
        axes[1, i+1].set_title(f'{lbl} (Sagittal)', color=col, fontsize=11, fontweight='bold')
        axes[1, i+1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ Saved: {save_path}")


def create_summary_comparison(ax_orig, ax_mask, sag_orig, sag_mask, save_path, seed=42):
    """All 4 types side-by-side comparison."""
    funcs = [
        (simulate_a1_wedge, 'A1: Wedge'),
        (simulate_a2_split, 'A2: Split'),
        (simulate_a3_incomplete_burst, 'A3: Inc. Burst'),
        (simulate_a4_complete_burst, 'A4: Comp. Burst'),
    ]
    
    fig, axes = plt.subplots(2, 5, figsize=(25, 10), facecolor='#0a0a0a')
    fig.suptitle('AO Spine Fracture Classification — Physics-Based Simulation Comparison',
                 fontsize=20, color='white', fontweight='bold', y=0.98)
    
    vmin, vmax = -200, 1200
    
    # Original
    axes[0, 0].imshow(ax_orig, cmap='bone', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
    axes[0, 0].set_title('Original\n(Axial)', color='#4CAF50', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    axes[1, 0].imshow(sag_orig, cmap='bone', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
    axes[1, 0].set_title('Original\n(Sagittal)', color='#4CAF50', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    for i, (func, name) in enumerate(funcs):
        ax_frac, ax_ch = func(ax_orig, ax_mask, severity=0.6, seed=seed)
        sag_frac, sag_ch = func(sag_orig, sag_mask, severity=0.6, seed=seed)
        
        # Axial with annotation
        axes[0, i+1].imshow(ax_frac, cmap='bone', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
        ov = np.zeros((*ax_frac.shape, 4), dtype=np.float32)
        ov[ax_ch, 0] = 1.0; ov[ax_ch, 3] = 0.3
        axes[0, i+1].imshow(ov, origin='lower', aspect='equal')
        axes[0, i+1].set_title(f'{name}\n(Axial)', color='#F44336', fontsize=11, fontweight='bold')
        axes[0, i+1].axis('off')
        
        # Sagittal with annotation
        axes[1, i+1].imshow(sag_frac, cmap='bone', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
        ov_s = np.zeros((*sag_frac.shape, 4), dtype=np.float32)
        ov_s[sag_ch, 0] = 1.0; ov_s[sag_ch, 3] = 0.3
        axes[1, i+1].imshow(ov_s, origin='lower', aspect='equal')
        axes[1, i+1].set_title(f'{name}\n(Sagittal)', color='#F44336', fontsize=11, fontweight='bold')
        axes[1, i+1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ Saved: {save_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("  Enhanced Fracture Visualization Generator v2")
    print("=" * 70)
    
    ct_data, mask_data, affine = load_verse_data()
    data = extract_vertebra_slices(ct_data, mask_data)
    
    ax_ct = data['axial_ct']
    ax_mask = data['axial_mask']
    sag_ct = data['sagittal_ct']
    sag_mask = data['sagittal_mask']
    
    print(f"  Axial: {ax_ct.shape}, bone px: {ax_mask.sum()}")
    print(f"  Sagittal: {sag_ct.shape}, bone px: {sag_mask.sum()}")
    
    configs = [
        {'func': simulate_a1_wedge, 'type': 'a1',
         'name': 'AO Type A1: Wedge Compression',
         'sub': 'Anterior height loss · Preserved posterior wall · Compression + Flexion mechanism',
         'fname': 'AO_A1_Wedge_Compression'},
        {'func': simulate_a2_split, 'type': 'a2',
         'name': 'AO Type A2: Split Fracture',
         'sub': 'Coronal plane split · Both endplates involved · Axial loading mechanism',
         'fname': 'AO_A2_Split_Fracture'},
        {'func': simulate_a3_incomplete_burst, 'type': 'a3',
         'name': 'AO Type A3: Incomplete Burst',
         'sub': 'Posterior wall fracture · Canal compromise <25% · High axial load',
         'fname': 'AO_A3_Incomplete_Burst'},
        {'func': simulate_a4_complete_burst, 'type': 'a4',
         'name': 'AO Type A4: Complete Burst',
         'sub': 'Retropulsion into spinal canal · Canal compromise >25% · Explosive axial force',
         'fname': 'AO_A4_Complete_Burst'},
    ]
    
    seed = 42
    for cfg in configs:
        print(f"\n{'='*55}")
        print(f"  {cfg['name']}")
        print(f"{'='*55}")
        
        ax_frac, ax_ch = cfg['func'](ax_ct, ax_mask, severity=0.6, seed=seed)
        sag_frac, sag_ch = cfg['func'](sag_ct, sag_mask, severity=0.6, seed=seed)
        
        # Detailed per-type figure
        create_fracture_report_figure(
            ax_ct, ax_mask, ax_frac, ax_ch,
            sag_ct, sag_mask, sag_frac, sag_ch,
            cfg['name'], cfg['sub'], cfg['type'],
            os.path.join(OUTPUT_DIR, f"{cfg['fname']}.png")
        )
        
        # Severity gallery
        create_severity_gallery(
            ax_ct, ax_mask, sag_ct, sag_mask,
            cfg['func'], cfg['name'],
            os.path.join(OUTPUT_DIR, f"{cfg['fname']}_severity.png"),
            seed=seed
        )
    
    # Summary comparison
    print(f"\n{'='*55}")
    print("  All Types Summary")
    print(f"{'='*55}")
    create_summary_comparison(
        ax_ct, ax_mask, sag_ct, sag_mask,
        os.path.join(OUTPUT_DIR, 'AO_all_types_summary.png'),
        seed=seed
    )
    
    print(f"\n{'='*70}")
    print(f"  ✅ All done! Output: {OUTPUT_DIR}")
    print(f"{'='*70}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        sz = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"    {f}: {sz:.1f} KB")


if __name__ == '__main__':
    main()
