#!/usr/bin/env python3
"""Render fracture simulation on a REAL vertebra from VerSe CT data.

Loads actual CT volume + vertebra segmentation mask, extracts a single vertebra,
applies fracture damage to the CT voxels, and renders realistic views showing:
  - Intact vertebra (original CT HU values)
  - Fractured vertebra (damage applied: darkened voxels + crack gaps)
  - Animated progression
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pathlib import Path
import nibabel as nib
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import imageio

from fracture_simulator_v2 import (
    BoneFractureSimulator, AO_LOAD_CONFIGS, DAMAGE_CMAP,
)

OUTPUT_DIR = Path(__file__).parent.parent.parent / 'docs' / 'fracture_reports' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# VerSe data paths
VERSE_DIR = Path(__file__).parent.parent.parent / 'VerSe' / 'dataset-01training'
CT_PATH = VERSE_DIR / 'rawdata' / 'sub-verse503' / 'sub-verse503_dir-ax_ct.nii.gz'
MASK_PATH = VERSE_DIR / 'derivatives' / 'sub-verse503' / 'sub-verse503_dir-ax_seg-vert_msk.nii.gz'

# Bone window
BONE_WL, BONE_WW = 400, 1800  # center, width
BONE_MIN = BONE_WL - BONE_WW // 2  # -500
BONE_MAX = BONE_WL + BONE_WW // 2   # 1300

# Force multiplier for real vertebra (more occupancy → force diluted)
FORCE_SCALE = 50.0


def configure_for_real_vertebra(sim):
    """Override simulator parameters for real (non-synthetic) vertebra geometry.
    
    The region classifier uses hardcoded bounds tuned for synthetic geometry,
    which misclassifies real vertebra particles (mostly as cortical_shell).
    Reset thresholds to uniform and boost damage rate for visible fracture.
    """
    # Reset threshold scale — real particles shouldn't use synthetic regions
    sim._threshold_scale = np.ones(sim.N, dtype=np.float32)
    
    # Lower the damage threshold for faster initiation
    sim.material['damage_threshold'] = 0.002  # was 0.01
    
    # Boost damage rate (real has more particles competing)
    sim.material['damage_rate'] = 0.15  # was 0.05
    
    # Lower COD threshold so cracks show earlier
    sim.material['cod_threshold'] = 0.5  # was 0.8


def load_vertebra(ct_path, mask_path, target_label=None, pad=5):
    """Load a single vertebra from CT + mask.

    Returns:
        crop_ct: 3D array of HU values (cropped to vertebra)
        crop_mask: binary mask of the vertebra
        label_used: which label was selected
    """
    print(f"  Loading CT: {ct_path}")
    ct_nii = nib.load(str(ct_path))
    ct = ct_nii.get_fdata().astype(np.float32)

    print(f"  Loading mask: {mask_path}")
    mask_nii = nib.load(str(mask_path))
    mask = mask_nii.get_fdata().astype(np.int32)

    # Find available labels
    labels = np.unique(mask)
    labels = labels[labels > 0]
    print(f"  Available vertebra labels: {labels}")

    if target_label is None:
        # Pick a lumbar vertebra (labels 20-25 in VerSe convention)
        # or the largest vertebra if no lumbar available
        lumbar = [l for l in labels if 20 <= l <= 25]
        if lumbar:
            target_label = lumbar[len(lumbar) // 2]
        else:
            # Pick label with most voxels
            sizes = [(l, (mask == l).sum()) for l in labels]
            target_label = max(sizes, key=lambda x: x[1])[0]

    print(f"  Using label {target_label}")

    # Extract single vertebra
    vert_mask = (mask == target_label)
    if vert_mask.sum() == 0:
        raise ValueError(f"Label {target_label} not found in mask")

    # Crop to bounding box + padding
    coords = np.argwhere(vert_mask)
    lo = coords.min(axis=0) - pad
    hi = coords.max(axis=0) + pad
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.array(ct.shape))

    crop_ct = ct[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].copy()
    crop_mask = vert_mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].copy()

    print(f"  Vertebra crop shape: {crop_ct.shape}, "
          f"voxels in mask: {crop_mask.sum()}")

    return crop_ct, crop_mask, target_label


def sample_particles_from_mask(mask, n_particles=50000, seed=42):
    """Sample particle positions from a 3D binary mask.

    Returns positions normalized to [0, 1] range.
    """
    rng = np.random.default_rng(seed)
    coords = np.argwhere(mask)
    n_voxels = len(coords)

    if n_particles <= n_voxels:
        idx = rng.choice(n_voxels, size=n_particles, replace=False)
    else:
        idx = rng.choice(n_voxels, size=n_particles, replace=True)

    positions = coords[idx].astype(np.float64)

    # Add sub-voxel jitter
    positions += rng.uniform(-0.3, 0.3, positions.shape)

    # Normalize to [0, 1]
    pmin = positions.min(axis=0)
    pmax = positions.max(axis=0)
    prange = np.maximum(pmax - pmin, 1.0)
    positions = (positions - pmin) / prange

    return positions


def apply_damage_to_ct(ct, mask, damage_3d, crack_threshold=0.5):
    """Apply damage field to CT voxels.

    damage_3d: same shape as ct, values 0-1
    - damage < 0.5: reduce HU (micro-fractures)
    - damage >= 0.5: create dark lines (macro-cracks)
    - damage >= 0.8: air (complete fracture gap)
    """
    result = ct.copy()

    # Where there's bone mask and damage
    dmg = damage_3d * mask

    # Reduce HU proportionally to damage
    # Intact bone: ~300-1000 HU. Fractured: approaches -100 (soft tissue)
    reduction = dmg * 800  # Max reduction 800 HU
    result -= reduction

    # Macro-cracks: make very dark (air/hemorrhage)
    cracks = dmg > crack_threshold
    result[cracks] = np.minimum(result[cracks], -200 + (1 - dmg[cracks]) * 100)

    # Complete fracture gaps: air
    gaps = dmg > 0.8
    result[gaps] = -900  # Air

    return result


def particles_to_volume(positions, damage, shape, mask):
    """Map particle damage back to 3D volume (vectorized)."""
    damage_vol = np.zeros(shape, dtype=np.float32)

    # Vectorized: map [0,1] positions to voxel coords
    vi = np.clip((positions[:, 0] * (shape[0] - 1)).astype(int), 0, shape[0] - 1)
    vj = np.clip((positions[:, 1] * (shape[1] - 1)).astype(int), 0, shape[1] - 1)
    vk = np.clip((positions[:, 2] * (shape[2] - 1)).astype(int), 0, shape[2] - 1)

    # Use maximum damage at each voxel
    np.maximum.at(damage_vol, (vi, vj, vk), damage.astype(np.float32))

    # Smooth to fill gaps (wider kernel for real geometry)
    damage_vol = ndi.gaussian_filter(damage_vol, sigma=2.0)
    damage_vol = np.clip(damage_vol * 3.0, 0, 1)  # Boost after smoothing
    damage_vol *= mask  # Only within bone

    return damage_vol


def render_ct_slice(ax, volume, mask=None, title='', cmap='bone',
                    axis=0, slice_idx=None, overlay_damage=None):
    """Render a CT slice with optional damage overlay."""
    if slice_idx is None:
        if mask is not None:
            # Find slice with most bone
            bone_count = mask.sum(axis=tuple(i for i in range(3) if i != axis))
            slice_idx = bone_count.argmax()
        else:
            slice_idx = volume.shape[axis] // 2

    slices = [slice(None)] * 3
    slices[axis] = slice_idx
    img = volume[tuple(slices)]

    # Window/level for bone
    img_norm = np.clip((img - BONE_MIN) / (BONE_MAX - BONE_MIN), 0, 1)

    ax.imshow(img_norm.T, origin='lower', cmap=cmap, vmin=0, vmax=1)

    if overlay_damage is not None:
        dmg_slice = overlay_damage[tuple(slices)]
        # Show damage as red overlay
        dmg_rgb = np.zeros((*dmg_slice.T.shape, 4))
        dmg_rgb[:, :, 0] = 1.0  # red
        dmg_rgb[:, :, 3] = dmg_slice.T * 0.6  # alpha from damage
        ax.imshow(dmg_rgb, origin='lower')

    ax.set_title(title, color='white', fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')


def generate_fracture_on_real_vertebra(ao_type, ct, mask, output_path):
    """Full visualization: intact vs fractured real vertebra in 3 views."""
    config = AO_LOAD_CONFIGS[ao_type]

    # Sample particles from actual mask geometry
    positions = sample_particles_from_mask(mask, n_particles=50000, seed=42)

    # Run fracture simulation with scaled force for real geometry
    sim = BoneFractureSimulator(positions.copy(), seed=42)
    sim.setup_ao_loading(ao_type)
    sim.set_stress_mode('grid')
    sim.enable_anisotropy(ratio=2.0)
    configure_for_real_vertebra(sim)
    scaled_force = config['max_force'] * FORCE_SCALE
    history = sim.run(n_steps=200, max_force=scaled_force,
                      record_every=10, verbose=False)

    damage = sim.damage.copy()
    final = history[-1]

    # Map damage back to 3D volume
    damage_vol = particles_to_volume(positions, damage, ct.shape, mask)

    # Create fractured CT
    fractured_ct = apply_damage_to_ct(ct, mask, damage_vol)

    # ---- Render ----
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor('#0a0a12')

    gs = GridSpec(3, 4, figure=fig, hspace=0.25, wspace=0.2,
                  height_ratios=[1, 1, 0.6])

    fig.suptitle(f'{config["name"]} — Real Vertebra CT',
                 fontsize=20, fontweight='bold', color='white', y=0.97)
    fig.text(0.5, 0.935,
             f'{config["mechanism"]}  |  Grid P2G/G2P  |  '
             f'Damaged={final["damaged_frac"]*100:.1f}%  |  '
             f'Fractured={final["fractured_frac"]*100:.1f}%',
             fontsize=11, color='#aaa', ha='center')

    axes = {'axial': 2, 'sagittal': 0, 'coronal': 1}
    view_names = ['Axial', 'Sagittal', 'Coronal']

    # Row 0: Intact vertebra CT
    for col, (vname, axis) in enumerate(zip(view_names, [2, 0, 1])):
        ax = fig.add_subplot(gs[0, col])
        render_ct_slice(ax, ct, mask, title=f'Intact — {vname}',
                       cmap='bone', axis=axis)

    # Row 0, col 3: damage statistics
    ax_info = fig.add_subplot(gs[0, 3])
    ax_info.set_facecolor('#0a0a12')
    ax_info.axis('off')
    dmg_by_region = sim.get_damage_by_region()
    info_text = f"AO Type: {ao_type}\n"
    info_text += f"Mechanism: {config['mechanism'][:40]}\n"
    info_text += f"\nDamaged: {final['damaged_frac']*100:.1f}%\n"
    info_text += f"Fractured: {final['fractured_frac']*100:.1f}%\n"
    info_text += f"Max Damage: {final['max_damage']:.3f}\n"
    info_text += f"\nRegion Damage:\n"
    for r, v in sorted(dmg_by_region.items(), key=lambda x: -x[1])[:5]:
        bar = '█' * int(v * 20)
        info_text += f"  {r[:15]:<15} {v:.2f} {bar}\n"
    ax_info.text(0.05, 0.95, info_text, fontsize=10, color='white',
                 va='top', fontfamily='monospace', transform=ax_info.transAxes,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e',
                          edgecolor='#444'))

    # Row 1: Fractured vertebra CT (bone window shows cracks as dark lines)
    for col, (vname, axis) in enumerate(zip(view_names, [2, 0, 1])):
        ax = fig.add_subplot(gs[1, col])
        render_ct_slice(ax, fractured_ct, mask,
                       title=f'Fractured — {vname}',
                       cmap='bone', axis=axis,
                       overlay_damage=damage_vol)

    # Row 1, col 3: colorbar
    ax_cb = fig.add_subplot(gs[1, 3])
    sm = plt.cm.ScalarMappable(cmap='hot_r', norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax_cb, fraction=0.8, pad=0.02)
    cbar.set_label('Damage Overlay (D)', color='white', fontsize=10)
    cbar.ax.tick_params(colors='white')
    ax_cb.axis('off')

    # Row 2: Damage progression timeline
    ax_time = fig.add_subplot(gs[2, :2])
    ax_time.set_facecolor('#0d0d1a')
    steps = [h['step'] for h in history]
    damaged = [h['damaged_frac'] * 100 for h in history]
    fractured = [h['fractured_frac'] * 100 for h in history]
    ax_time.fill_between(steps, damaged, alpha=0.3, color='#ff6b6b')
    ax_time.plot(steps, damaged, color='#ff6b6b', linewidth=2, label='Damaged %')
    ax_time.fill_between(steps, fractured, alpha=0.3, color='#feca57')
    ax_time.plot(steps, fractured, color='#feca57', linewidth=2, label='Fractured %')
    ax_time.set_xlabel('Step', color='white', fontsize=11)
    ax_time.set_ylabel('%', color='white', fontsize=11)
    ax_time.set_title('Damage Progression', color='white', fontsize=12, fontweight='bold')
    ax_time.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    ax_time.tick_params(colors='#888')
    for sp in ax_time.spines.values(): sp.set_edgecolor('#333')

    # Row 2: HU histogram
    ax_hist = fig.add_subplot(gs[2, 2:])
    ax_hist.set_facecolor('#0d0d1a')
    bone = ct[mask > 0].ravel()
    frac = fractured_ct[mask > 0].ravel()
    ax_hist.hist(bone, bins=100, alpha=0.5, color='#54a0ff', label='Intact', density=True)
    ax_hist.hist(frac, bins=100, alpha=0.5, color='#ff6b6b', label='Fractured', density=True)
    ax_hist.set_xlabel('HU', color='white', fontsize=11)
    ax_hist.set_title('HU Distribution (Bone Only)', color='white', fontsize=12, fontweight='bold')
    ax_hist.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    ax_hist.tick_params(colors='#888')
    for sp in ax_hist.spines.values(): sp.set_edgecolor('#333')

    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"  ✅ {ao_type} real vertebra fracture: {output_path}")

    return sim, history, damage_vol, fractured_ct


def generate_fracture_gif(ao_type, ct, mask, output_path, fps=5):
    """Animated GIF showing fracture progression on real CT vertebra."""
    config = AO_LOAD_CONFIGS[ao_type]
    positions = sample_particles_from_mask(mask, n_particles=50000, seed=42)

    sim = BoneFractureSimulator(positions.copy(), seed=42)
    sim.setup_ao_loading(ao_type)
    sim.set_stress_mode('grid')
    sim.enable_anisotropy(ratio=2.0)
    configure_for_real_vertebra(sim)

    n_steps = 200
    frames = []
    scaled_force = config['max_force'] * FORCE_SCALE

    for step in range(n_steps + 1):
        force = scaled_force * step / n_steps
        sim.compute_stress(force)
        sim.evolve_damage()

        if step % 8 == 0:
            damage = sim.damage.copy()
            damage_vol = particles_to_volume(positions, damage, ct.shape, mask)
            fractured_ct = apply_damage_to_ct(ct, mask, damage_vol)

            damaged_pct = (damage > 0.01).mean() * 100
            fractured_pct = (damage > 0.8).mean() * 100

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.patch.set_facecolor('#0a0a12')

            for vi, (vname, axis) in enumerate(zip(
                ['Axial', 'Sagittal', 'Coronal'], [2, 0, 1]
            )):
                render_ct_slice(axes[vi], fractured_ct, mask,
                               title=vname, cmap='bone', axis=axis,
                               overlay_damage=damage_vol)

            fig.suptitle(
                f'{config["name"]} — Step {step}/{n_steps}  |  '
                f'Force={force:.2f}  |  '
                f'Damaged={damaged_pct:.1f}%  |  Fractured={fractured_pct:.1f}%',
                color='white', fontsize=13, fontweight='bold', y=0.98
            )
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            img = np.asarray(buf)[:, :, :3].copy()
            frames.append(img)
            plt.close(fig)

    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"  ✅ {ao_type} CT fracture GIF: {output_path} ({len(frames)} frames)")


def generate_ao_comparison(ct, mask, output_path):
    """4-column comparison: A1-A4 on the SAME real vertebra."""
    positions = sample_particles_from_mask(mask, n_particles=50000, seed=42)

    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor('#0a0a12')
    gs = GridSpec(3, 4, figure=fig, hspace=0.25, wspace=0.15,
                  height_ratios=[1, 1, 0.5])

    ao_types = ['A1', 'A2', 'A3', 'A4']

    for col, ao_type in enumerate(ao_types):
        config = AO_LOAD_CONFIGS[ao_type]
        sim = BoneFractureSimulator(positions.copy(), seed=42)
        sim.setup_ao_loading(ao_type)
        sim.set_stress_mode('grid')
        sim.enable_anisotropy(ratio=2.0)
        configure_for_real_vertebra(sim)
        scaled_force = config['max_force'] * FORCE_SCALE
        history = sim.run(n_steps=200, max_force=scaled_force,
                          record_every=50, verbose=False)
        damage = sim.damage.copy()
        damage_vol = particles_to_volume(positions, damage, ct.shape, mask)
        fractured_ct = apply_damage_to_ct(ct, mask, damage_vol)
        final = history[-1]

        # Row 0: Axial
        ax = fig.add_subplot(gs[0, col])
        render_ct_slice(ax, fractured_ct, mask,
                       title=f'{config["name"]}',
                       cmap='bone', axis=2, overlay_damage=damage_vol)

        # Row 1: Sagittal
        ax = fig.add_subplot(gs[1, col])
        render_ct_slice(ax, fractured_ct, mask,
                       title=f'{final["damaged_frac"]*100:.0f}%D, {final["fractured_frac"]*100:.0f}%F',
                       cmap='bone', axis=0, overlay_damage=damage_vol)

        # Row 2: Timeline
        ax = fig.add_subplot(gs[2, col])
        ax.set_facecolor('#0d0d1a')
        steps = [h['step'] for h in history]
        damaged = [h['damaged_frac'] * 100 for h in history]
        ax.fill_between(steps, damaged, alpha=0.3, color='#ff6b6b')
        ax.plot(steps, damaged, color='#ff6b6b', linewidth=1.5)
        ax.set_xlabel('Step', color='#aaa', fontsize=8)
        ax.tick_params(colors='#666', labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('#333')

    for row, label in enumerate(['Axial CT', 'Sagittal CT', 'Timeline']):
        fig.text(0.01, 0.82 - row * 0.32, label,
                 fontsize=11, color='#aaa', rotation=90, va='center')

    fig.suptitle('AO Fracture Comparison — Real Vertebra CT (VerSe)',
                 fontsize=16, fontweight='bold', color='white', y=0.98)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"  ✅ AO comparison on real CT: {output_path}")


if __name__ == '__main__':
    print("=" * 70)
    print("Fracture Simulation on Real VerSe Vertebra CT")
    print("=" * 70)

    # Load real vertebra
    print("\n1. Loading vertebra from VerSe...")
    ct, mask, label = load_vertebra(CT_PATH, MASK_PATH)

    # Per-AO fracture cards on real CT
    print("\n2. Generating per-AO fracture visualizations...")
    for ao in ['A1', 'A4']:
        generate_fracture_on_real_vertebra(
            ao, ct, mask, OUTPUT_DIR / f'v2_{ao}_real_ct.png')

    # A1-A4 comparison on real CT
    print("\n3. Generating AO comparison on real CT...")
    generate_ao_comparison(ct, mask, OUTPUT_DIR / 'v2_ao_real_comparison.png')

    # Animated GIFs
    print("\n4. Generating fracture progression GIFs...")
    for ao in ['A1', 'A4']:
        generate_fracture_gif(ao, ct, mask,
                             OUTPUT_DIR / f'v2_{ao}_real_fracture.gif', fps=5)

    print(f"\n✅ Done! Output: {OUTPUT_DIR}")
