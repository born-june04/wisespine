#!/usr/bin/env python3
"""Render 3D mesh + CT combined views of REAL fractured vertebra.

Key improvements:
 - Oblique 3/4 camera angles that show vertebra volume clearly
 - Visible red stress diffusion spreading across mesh surface
 - Combined CT slice + 3D mesh layout in same figure
 - Vectorized face coloring for speed

Outputs:
  - v2_{AO}_combined.png:  Per-AO combined CT + 3D mesh (intact→fractured)
  - v2_combined_comparison.png: A1-A4 all in one figure
  - v2_{AO}_mesh_diffusion.gif: Animated stress diffusion on 3D mesh
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pathlib import Path
import nibabel as nib
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio

from fracture_simulator_v2 import BoneFractureSimulator, AO_LOAD_CONFIGS

OUTPUT_DIR = Path(__file__).parent.parent.parent / 'docs' / 'fracture_reports' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VERSE_DIR = Path(__file__).parent.parent.parent / 'VerSe' / 'dataset-01training'
CT_PATH = VERSE_DIR / 'rawdata' / 'sub-verse503' / 'sub-verse503_dir-ax_ct.nii.gz'
MASK_PATH = VERSE_DIR / 'derivatives' / 'sub-verse503' / 'sub-verse503_dir-ax_seg-vert_msk.nii.gz'

FORCE_SCALE = 50.0
BONE_WL, BONE_WW = 400, 1800
BONE_MIN = BONE_WL - BONE_WW // 2
BONE_MAX = BONE_WL + BONE_WW // 2


def configure_for_real_vertebra(sim):
    """Override simulator params for real geometry — high damage for visible cracks."""
    sim._threshold_scale = np.ones(sim.N, dtype=np.float32)
    sim.material['damage_threshold'] = 0.001     # Lower threshold → earlier onset
    sim.material['damage_rate'] = 0.8             # 5× faster → reach D~0.6-0.8
    sim.material['cod_threshold'] = 0.3           # Cracks open sooner


# ---------- data loading ----------
def load_vertebra(ct_path, mask_path, target_label=None, pad=5):
    ct = nib.load(str(ct_path)).get_fdata().astype(np.float32)
    mask = nib.load(str(mask_path)).get_fdata().astype(np.int32)
    labels = np.unique(mask); labels = labels[labels > 0]
    print(f"  Labels: {labels}")
    if target_label is None:
        lumbar = [l for l in labels if 20 <= l <= 25]
        target_label = lumbar[len(lumbar)//2] if lumbar else labels[-1]
    print(f"  Using label {target_label}")
    vm = (mask == target_label)
    c = np.argwhere(vm)
    lo = np.maximum(c.min(0) - pad, 0)
    hi = np.minimum(c.max(0) + pad, np.array(ct.shape))
    crop_ct = ct[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].copy()
    crop_mask = vm[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].copy()
    print(f"  Crop: {crop_ct.shape}, voxels: {crop_mask.sum()}")
    return crop_ct, crop_mask, target_label


def extract_mesh(mask, step_size=2):
    smooth = ndi.gaussian_filter(mask.astype(np.float32), sigma=1.0)
    verts, faces, _, _ = marching_cubes(smooth, level=0.5, step_size=step_size)
    return verts, faces


def sample_particles(mask, n=50000, seed=42):
    rng = np.random.default_rng(seed)
    coords = np.argwhere(mask)
    idx = rng.choice(len(coords), min(n, len(coords)), replace=False)
    pos = coords[idx].astype(np.float64) + rng.uniform(-0.3, 0.3, (len(idx), 3))
    pmin, pmax = pos.min(0), pos.max(0)
    return (pos - pmin) / np.maximum(pmax - pmin, 1.0)


def particles_to_volume(positions, damage, shape, mask, for_mesh=False):
    vol = np.zeros(shape, dtype=np.float32)
    vi = np.clip((positions[:,0]*(shape[0]-1)).astype(int), 0, shape[0]-1)
    vj = np.clip((positions[:,1]*(shape[1]-1)).astype(int), 0, shape[1]-1)
    vk = np.clip((positions[:,2]*(shape[2]-1)).astype(int), 0, shape[2]-1)
    np.maximum.at(vol, (vi, vj, vk), damage.astype(np.float32))
    if for_mesh:
        # Wide blur to propagate internal damage to surface
        vol = ndi.gaussian_filter(vol, sigma=4.0)
        vol = np.clip(vol * 15.0, 0, 1)  # Aggressive boost
    else:
        vol = ndi.gaussian_filter(vol, sigma=2.0)
        vol = np.clip(vol * 5.0, 0, 1)   # Boost for CT cracks
    vol *= mask
    return vol


def stress_to_volume(positions, stress, shape, mask):
    """Map particle STRESS (not damage) to 3D volume for diffusion viz."""
    vol = np.zeros(shape, dtype=np.float32)
    vi = np.clip((positions[:,0]*(shape[0]-1)).astype(int), 0, shape[0]-1)
    vj = np.clip((positions[:,1]*(shape[1]-1)).astype(int), 0, shape[1]-1)
    vk = np.clip((positions[:,2]*(shape[2]-1)).astype(int), 0, shape[2]-1)
    np.maximum.at(vol, (vi, vj, vk), stress.astype(np.float32))
    vol = ndi.gaussian_filter(vol, sigma=6.0)  # Wide for surface visibility
    if vol.max() > 0:
        vol /= vol.max()
    vol = np.clip(vol * 2.0, 0, 1)  # Boost
    vol *= mask
    return vol


# ---------- face coloring (vectorized) ----------
def color_faces_by_field(verts, faces, field_vol, shape, cmap_fn):
    """Vectorized: sample field at face centroids, map to colors."""
    centroids = verts[faces].mean(axis=1)  # (F, 3)
    ci = np.clip(centroids[:,0].astype(int), 0, shape[0]-1)
    cj = np.clip(centroids[:,1].astype(int), 0, shape[1]-1)
    ck = np.clip(centroids[:,2].astype(int), 0, shape[2]-1)
    vals = field_vol[ci, cj, ck]  # (F,)
    return cmap_fn(vals)


def damage_cmap(vals):
    """Bone white → amber → DARK CRACK for damage [0,1].
    Key: high damage (>0.10) starts showing dark crack lines."""
    colors = np.zeros((len(vals), 4))
    colors[:, 3] = 1.0
    v = np.clip(vals * 3.0, 0, 1)  # D=0.33 → full
    
    # Sharp crack onset at D=0.10, fully dark at D=0.25
    crack = np.clip((vals - 0.10) / 0.15, 0, 1)
    
    # Base: bone white → amber
    colors[:, 0] = 0.95 - 0.30 * v
    colors[:, 1] = 0.92 - 0.82 * v
    colors[:, 2] = 0.85 - 0.85 * v
    
    # Red glow zone (early damage — orange/red)
    mid = np.clip(v * 2, 0, 1) * np.clip((1.0 - v) * 2, 0, 1)
    colors[:, 0] = np.minimum(colors[:, 0] + mid * 0.4, 1.0)
    
    # CRACK: Override to very dark where D > 0.10
    colors[:, 0] = colors[:, 0] * (1.0 - crack * 0.88)
    colors[:, 1] = colors[:, 1] * (1.0 - crack * 0.96)
    colors[:, 2] = colors[:, 2] * (1.0 - crack * 0.96)
    # Slight dark red tint for deep cracks
    colors[:, 0] = np.maximum(colors[:, 0], crack * 0.10)
    return colors


def stress_cmap(vals):
    """Bone white → yellow → bright red for stress [0,1]."""
    colors = np.zeros((len(vals), 4))
    colors[:, 3] = 1.0
    colors[:, 0] = 0.95
    colors[:, 1] = 0.92 - 0.82 * vals
    colors[:, 2] = 0.85 - 0.85 * vals
    high = np.clip(vals * 2, 0, 1)
    colors[:, 0] = np.minimum(0.95 + high * 0.05, 1.0)
    return colors


def combined_cmap(damage_vals, stress_vals):
    """Combine damage (dark cracks) + stress (red glow).
    Low damage + high stress → red/orange glow (stress diffusion)
    High damage → dark crack lines (fracture)"""
    colors = damage_cmap(damage_vals)
    
    # Stress glow only in non-cracked zones
    s = np.clip(stress_vals * 2.0, 0, 1)
    crack = np.clip((damage_vals - 0.10) / 0.15, 0, 1)
    no_crack = 1.0 - crack
    glow = s * no_crack
    colors[:, 0] = np.minimum(colors[:, 0] + glow * 0.6, 1.0)
    colors[:, 1] = np.maximum(colors[:, 1] - glow * 0.5, 0.0)
    colors[:, 2] = np.maximum(colors[:, 2] - glow * 0.5, 0.0)
    return colors


# ---------- 3D mesh rendering ----------
def render_mesh(ax, verts, faces, face_colors, elev=25, azim=-50,
                title='', zoom=0.85, damage_vals=None):
    """Render mesh with visible crack lines on damaged faces.
    damage_vals: per-face damage values. Faces with D>0.10 get dark edge wireframe.
    """
    mesh = Poly3DCollection(verts[faces], linewidths=0)
    mesh.set_facecolor(face_colors)
    mesh.set_edgecolor('none')
    ax.add_collection3d(mesh)

    # Overlay dark crack lines on high-damage faces
    if damage_vals is not None:
        crack_mask = damage_vals > 0.08
        if crack_mask.any():
            crack_faces = faces[crack_mask]
            crack_d = damage_vals[crack_mask]
            intensity = np.clip((crack_d - 0.08) / 0.15, 0.3, 1.0)
            # Draw crack faces with THICK dark edges
            crack_mesh = Poly3DCollection(
                verts[crack_faces], linewidths=2.5
            )
            # Semi-transparent dark overlay on cracked faces
            n_crack = len(crack_faces)
            crack_fc = np.zeros((n_crack, 4))
            crack_fc[:, 0] = 0.08  # Very dark
            crack_fc[:, 1] = 0.02
            crack_fc[:, 2] = 0.02
            crack_fc[:, 3] = intensity * 0.5  # Visible overlay
            crack_mesh.set_facecolor(crack_fc)
            # Near-black edges = crack lines — fully opaque
            crack_ec = np.zeros((n_crack, 4))
            crack_ec[:, 0] = 0.05
            crack_ec[:, 1] = 0.0
            crack_ec[:, 2] = 0.0
            crack_ec[:, 3] = intensity * 0.95  # Nearly fully opaque
            crack_mesh.set_edgecolor(crack_ec)
            ax.add_collection3d(crack_mesh)

    # Equal aspect ratio with zoom
    xlim = [verts[:,0].min(), verts[:,0].max()]
    ylim = [verts[:,1].min(), verts[:,1].max()]
    zlim = [verts[:,2].min(), verts[:,2].max()]
    max_range = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]) / zoom
    mx = (xlim[0]+xlim[1])/2; my = (ylim[0]+ylim[1])/2; mz = (zlim[0]+zlim[1])/2
    ax.set_xlim(mx-max_range/2, mx+max_range/2)
    ax.set_ylim(my-max_range/2, my+max_range/2)
    ax.set_zlim(mz-max_range/2, mz+max_range/2)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=11, fontweight='bold', color='white', pad=8)
    ax.set_axis_off()
    ax.set_facecolor('#0a0a12')


def render_ct_slice(ax, volume, mask, damage_vol=None, axis=0, title=''):
    """Render bone-windowed CT slice showing dark fracture cracks.
    Low damage → subtle red tint (stress diffusion)
    High damage → dark crack lines visible as HU collapse"""
    bone_count = mask.sum(axis=tuple(i for i in range(3) if i != axis))
    si = bone_count.argmax()
    sl = [slice(None)]*3; sl[axis] = si
    img = volume[tuple(sl)]  # Already has HU modifications from apply_damage_to_ct
    norm = np.clip((img - BONE_MIN) / (BONE_MAX - BONE_MIN), 0, 1)

    rgb = np.stack([norm]*3, axis=-1)
    if damage_vol is not None:
        d = damage_vol[tuple(sl)]
        # Split into stress zone (low D) and crack zone (high D)
        stress_zone = np.clip(d * 3.0, 0, 1) * np.clip(1.0 - (d - 0.2) / 0.1, 0, 1)
        crack_zone = np.clip((d - 0.15) / 0.15, 0, 1)  # D > 0.15 → crack line
        
        # Stress diffusion: subtle warm red tint (non-crack areas)
        rgb[:,:,0] = np.clip(rgb[:,:,0] + stress_zone * 0.4, 0, 1)
        rgb[:,:,1] = rgb[:,:,1] * (1 - stress_zone * 0.3)
        rgb[:,:,2] = rgb[:,:,2] * (1 - stress_zone * 0.3)
        
        # CRACK lines: make dark — the CT HU is already low, darken further
        rgb[:,:,0] = rgb[:,:,0] * (1 - crack_zone * 0.7)
        rgb[:,:,1] = rgb[:,:,1] * (1 - crack_zone * 0.85)
        rgb[:,:,2] = rgb[:,:,2] * (1 - crack_zone * 0.85)

    ax.imshow(rgb, cmap='bone', aspect='auto')
    ax.set_title(title, fontsize=10, fontweight='bold', color='white')
    ax.axis('off')


def apply_damage_to_ct(ct, mask, damage_vol):
    """Modify CT HU values based on damage."""
    out = ct.copy()
    bone = mask > 0
    d = damage_vol[bone]
    micro = (d > 0.1) & (d <= 0.5)
    macro = (d > 0.5) & (d <= 0.8)
    full = d > 0.8
    vals = out[bone]
    vals[micro] *= (1 - 0.3 * d[micro])
    rng = np.random.default_rng(42)
    vals[macro] = rng.uniform(-200, -100, macro.sum())
    vals[full] = -900
    out[bone] = vals
    return out


# ---------- camera angles ----------
# Oblique views that show the vertebra volume clearly
VIEWS = [
    ('Oblique Anterior',    25, -50),   # 3/4 front view — body + pedicles visible
    ('Superior (Top-Down)', 85, -90),   # classical axial — shows body, canal, processes
    ('Oblique Posterior',   25, 130),   # see posterior elements from behind
]

VIEWS_COMPACT = [
    ('Oblique',  25, -50),
    ('Superior', 85, -90),
]


# ---------- main generators ----------
def generate_combined_card(ao_type, ct, mask, verts, faces, output_path):
    """Per-AO combined visualization: CT slices + 3D mesh, intact → fractured."""
    config = AO_LOAD_CONFIGS[ao_type]
    positions = sample_particles(mask)

    sim = BoneFractureSimulator(positions.copy(), seed=42)
    sim.setup_ao_loading(ao_type)
    sim.set_stress_mode('grid')
    sim.enable_anisotropy(ratio=2.0)
    configure_for_real_vertebra(sim)
    scaled_force = config['max_force'] * FORCE_SCALE
    history = sim.run(n_steps=200, max_force=scaled_force,
                      record_every=10, verbose=False)
    damage = sim.damage.copy()
    stress = sim.stress.copy()
    final = history[-1]

    damage_vol_ct = particles_to_volume(positions, damage, ct.shape, mask, for_mesh=False)
    damage_vol_mesh = particles_to_volume(positions, damage, ct.shape, mask, for_mesh=True)
    stress_vol = stress_to_volume(positions, stress, ct.shape, mask)
    fractured_ct = apply_damage_to_ct(ct, mask, damage_vol_ct)

    # Compute face colors: combined damage + stress diffusion glow
    centroids = verts[faces].mean(axis=1)
    ci = np.clip(centroids[:,0].astype(int), 0, ct.shape[0]-1)
    cj = np.clip(centroids[:,1].astype(int), 0, ct.shape[1]-1)
    ck = np.clip(centroids[:,2].astype(int), 0, ct.shape[2]-1)
    face_damage = damage_vol_mesh[ci, cj, ck]
    face_stress = stress_vol[ci, cj, ck]
    face_colors_frac = combined_cmap(face_damage, face_stress)
    bone_color = np.full((len(faces), 4), [0.95,0.92,0.85,0.95])

    # Layout: 2 rows × 4 cols
    # Row 0: Intact CT axial | Intact CT sagittal | 3D mesh oblique | 3D mesh superior
    # Row 1: Frac CT axial   | Frac CT sagittal   | 3D mesh oblique | 3D mesh superior
    fig = plt.figure(figsize=(24, 12))
    fig.patch.set_facecolor('#0a0a12')

    gs = GridSpec(2, 4, figure=fig, hspace=0.15, wspace=0.08)

    fig.suptitle(f'{config["name"]} — Real Vertebra (VerSe L4)',
                 fontsize=18, fontweight='bold', color='white', y=0.98)
    fig.text(0.5, 0.94,
             f'{config["mechanism"]}  |  '
             f'Damaged={final["damaged_frac"]*100:.1f}%  |  '
             f'Fractured={final["fractured_frac"]*100:.1f}%',
             fontsize=12, color='#aaa', ha='center')

    # Top row — intact
    ax = fig.add_subplot(gs[0, 0])
    render_ct_slice(ax, ct, mask, axis=0, title='Intact — Axial CT')

    ax = fig.add_subplot(gs[0, 1])
    render_ct_slice(ax, ct, mask, axis=1, title='Intact — Sagittal CT')

    for ci_v, (vname, elev, azim) in enumerate(VIEWS_COMPACT):
        ax = fig.add_subplot(gs[0, 2+ci_v], projection='3d')
        render_mesh(ax, verts, faces, bone_color, elev=elev, azim=azim,
                   title=f'Intact — {vname}')

    # Bottom row — fractured
    ax = fig.add_subplot(gs[1, 0])
    render_ct_slice(ax, fractured_ct, mask, damage_vol=damage_vol_ct,
                   axis=0, title='Fractured — Axial CT')

    ax = fig.add_subplot(gs[1, 1])
    render_ct_slice(ax, fractured_ct, mask, damage_vol=damage_vol_ct,
                   axis=1, title='Fractured — Sagittal CT')

    for ci_v, (vname, elev, azim) in enumerate(VIEWS_COMPACT):
        ax = fig.add_subplot(gs[1, 2+ci_v], projection='3d')
        render_mesh(ax, verts, faces, face_colors_frac, elev=elev, azim=azim,
                   title=f'Fractured — {vname}', damage_vals=face_damage)

    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"  ✅ {ao_type} combined: {output_path}")
    return damage_vol_ct


def generate_combined_comparison(ct, mask, verts, faces, output_path):
    """A1-A4 comparison: 4 columns × 2 rows (3D mesh oblique + superior)."""
    positions = sample_particles(mask)

    fig = plt.figure(figsize=(24, 14))
    fig.patch.set_facecolor('#0a0a12')
    gs = GridSpec(3, 4, figure=fig, hspace=0.15, wspace=0.05,
                  height_ratios=[1, 1, 0.4])

    ao_types = ['A1', 'A2', 'A3', 'A4']
    stats = []

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
        stress = sim.stress.copy()
        final = history[-1]
        stats.append(final)

        damage_vol_mesh = particles_to_volume(positions, damage, ct.shape, mask, for_mesh=True)
        stress_vol = stress_to_volume(positions, stress, ct.shape, mask)

        centroids = verts[faces].mean(axis=1)
        ci = np.clip(centroids[:,0].astype(int), 0, ct.shape[0]-1)
        cj = np.clip(centroids[:,1].astype(int), 0, ct.shape[1]-1)
        ck = np.clip(centroids[:,2].astype(int), 0, ct.shape[2]-1)
        fd = damage_vol_mesh[ci, cj, ck]
        fs = stress_vol[ci, cj, ck]
        face_colors = combined_cmap(fd, fs)

        # Row 0: Oblique 3/4 view
        ax = fig.add_subplot(gs[0, col], projection='3d')
        render_mesh(ax, verts, faces, face_colors, elev=25, azim=-50,
                   title=config['name'], damage_vals=fd)

        # Row 1: Superior view
        ax = fig.add_subplot(gs[1, col], projection='3d')
        render_mesh(ax, verts, faces, face_colors, elev=85, azim=-90,
                   title=f'{final["damaged_frac"]*100:.0f}%D / {final["fractured_frac"]*100:.0f}%F',
                   damage_vals=fd)

        # Row 2: Timeline
        ax = fig.add_subplot(gs[2, col])
        ax.set_facecolor('#0a0a12')
        steps = [h['step'] for h in history]
        dpct = [h['damaged_frac']*100 for h in history]
        fpct = [h['fractured_frac']*100 for h in history]
        ax.fill_between(steps, dpct, alpha=0.4, color='#ff4444')
        ax.fill_between(steps, fpct, alpha=0.4, color='#ffaa00')
        ax.plot(steps, dpct, color='#ff4444', lw=1.5)
        ax.plot(steps, fpct, color='#ffaa00', lw=1.5)
        ax.set_xlabel('Step', color='#888', fontsize=8)
        ax.tick_params(colors='#666', labelsize=7)
        for sp in ax.spines.values(): sp.set_color('#333')

    fig.suptitle('AO Fracture Comparison — Real Vertebra 3D Mesh + Stress Diffusion',
                 fontsize=16, fontweight='bold', color='white', y=0.99)

    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"  ✅ Combined comparison: {output_path}")


def generate_diffusion_gif(ao_type, ct, mask, verts, faces, output_path, fps=4):
    """Animated GIF: stress diffusion spreading as red glow across mesh surface."""
    config = AO_LOAD_CONFIGS[ao_type]
    positions = sample_particles(mask)

    sim = BoneFractureSimulator(positions.copy(), seed=42)
    sim.setup_ao_loading(ao_type)
    sim.set_stress_mode('grid')
    sim.enable_anisotropy(ratio=2.0)
    configure_for_real_vertebra(sim)

    n_steps = 200
    frames = []
    scaled_force = config['max_force'] * FORCE_SCALE

    centroids = verts[faces].mean(axis=1)
    ci = np.clip(centroids[:,0].astype(int), 0, ct.shape[0]-1)
    cj = np.clip(centroids[:,1].astype(int), 0, ct.shape[1]-1)
    ck = np.clip(centroids[:,2].astype(int), 0, ct.shape[2]-1)

    for step in range(n_steps + 1):
        force = scaled_force * step / n_steps
        sim.compute_stress(force)
        sim.evolve_damage()

        if step % 8 == 0:
            damage = sim.damage.copy()
            stress = sim.stress.copy()

            damage_vol_ct = particles_to_volume(positions, damage, ct.shape, mask, for_mesh=False)
            damage_vol_mesh = particles_to_volume(positions, damage, ct.shape, mask, for_mesh=True)
            stress_vol = stress_to_volume(positions, stress, ct.shape, mask)

            fd = damage_vol_mesh[ci, cj, ck]
            fs = stress_vol[ci, cj, ck]
            face_colors = combined_cmap(fd, fs)

            damaged_pct = (damage > 0.01).mean() * 100
            fractured_pct = (damage > 0.8).mean() * 100

            fig = plt.figure(figsize=(18, 8))
            fig.patch.set_facecolor('#0a0a12')

            # 3 panels: oblique mesh | superior mesh | CT axial
            gs = GridSpec(1, 3, figure=fig, wspace=0.05, width_ratios=[1, 1, 0.8])

            ax = fig.add_subplot(gs[0, 0], projection='3d')
            render_mesh(ax, verts, faces, face_colors, elev=25, azim=-50,
                       title='Oblique', damage_vals=fd)

            ax = fig.add_subplot(gs[0, 1], projection='3d')
            render_mesh(ax, verts, faces, face_colors, elev=85, azim=-90,
                       title='Superior', damage_vals=fd)

            # CT axial slice with damage overlay
            ax = fig.add_subplot(gs[0, 2])
            fractured_ct = apply_damage_to_ct(ct, mask, damage_vol_ct)
            render_ct_slice(ax, fractured_ct, mask, damage_vol=damage_vol_ct,
                          axis=0, title='CT Axial')

            fig.suptitle(
                f'{config["name"]}  |  Step {step}/{n_steps}  |  '
                f'Force={force:.0f}  |  D={damaged_pct:.1f}%  |  F={fractured_pct:.1f}%',
                color='white', fontsize=13, fontweight='bold', y=0.97)

            fig.tight_layout(rect=[0, 0, 1, 0.93])
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            img = np.asarray(buf)[:, :, :3].copy()
            frames.append(img)
            plt.close(fig)

    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"  ✅ {ao_type} diffusion GIF: {output_path} ({len(frames)} frames)")


def generate_zoom_fracture(ao_type, ct, mask, verts, faces, output_path):
    """Zoom into the fracture zone: 3D mesh close-up + CT slices centered on damage peak."""
    config = AO_LOAD_CONFIGS[ao_type]
    positions = sample_particles(mask)

    sim = BoneFractureSimulator(positions.copy(), seed=42)
    sim.setup_ao_loading(ao_type)
    sim.set_stress_mode('grid')
    sim.enable_anisotropy(ratio=2.0)
    configure_for_real_vertebra(sim)
    scaled_force = config['max_force'] * FORCE_SCALE
    history = sim.run(n_steps=200, max_force=scaled_force,
                      record_every=10, verbose=False)
    damage = sim.damage.copy()
    stress = sim.stress.copy()
    final = history[-1]

    damage_vol_ct = particles_to_volume(positions, damage, ct.shape, mask, for_mesh=False)
    damage_vol_mesh = particles_to_volume(positions, damage, ct.shape, mask, for_mesh=True)
    stress_vol = stress_to_volume(positions, stress, ct.shape, mask)
    fractured_ct = apply_damage_to_ct(ct, mask, damage_vol_ct)

    # Find damage peak in voxel space
    peak = np.unravel_index(damage_vol_ct.argmax(), damage_vol_ct.shape)
    print(f"  Damage peak at voxel: {peak}, max={damage_vol_ct.max():.3f}")

    # --- Zoomed CT slices centered on damage peak ---
    zoom_half = 35  # half-width of zoom window in voxels
    def get_zoom_slice(vol, damage, axis, center_idx):
        """Extract zoomed region around damage center."""
        sl = [slice(None)] * 3
        sl[axis] = center_idx
        img = vol[tuple(sl)]
        d = damage[tuple(sl)]
        # Crop to zoom region
        other_axes = [i for i in range(3) if i != axis]
        c0, c1 = peak[other_axes[0]], peak[other_axes[1]]
        r0 = max(0, c0 - zoom_half); r1 = min(img.shape[0], c0 + zoom_half)
        c0_ = max(0, c1 - zoom_half); c1_ = min(img.shape[1], c1 + zoom_half)
        return img[r0:r1, c0_:c1_], d[r0:r1, c0_:c1_]

    # --- Zoomed mesh: restrict axis limits around damage center ---
    # Convert peak voxel to mesh vertex space
    peak_v = np.array(peak, dtype=float)

    # Face colors
    centroids = verts[faces].mean(axis=1)
    ci = np.clip(centroids[:,0].astype(int), 0, ct.shape[0]-1)
    cj = np.clip(centroids[:,1].astype(int), 0, ct.shape[1]-1)
    ck = np.clip(centroids[:,2].astype(int), 0, ct.shape[2]-1)
    fd = damage_vol_mesh[ci, cj, ck]
    fs = stress_vol[ci, cj, ck]
    face_colors = combined_cmap(fd, fs)

    # --- Figure layout: 2×3 ---
    # Row 0: Full mesh oblique | Full CT axial | Full CT sagittal
    # Row 1: Zoomed mesh       | Zoomed CT axial | Zoomed CT sagittal
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor('#0a0a12')
    gs = GridSpec(2, 3, figure=fig, hspace=0.18, wspace=0.12)

    fig.suptitle(f'{config["name"]} — Fracture Zone Close-Up',
                 fontsize=18, fontweight='bold', color='white', y=0.98)
    fig.text(0.5, 0.94,
             f'{config["mechanism"]}  |  '
             f'Damaged={final["damaged_frac"]*100:.1f}%  |  '
             f'Fractured={final["fractured_frac"]*100:.1f}%',
             fontsize=12, color='#aaa', ha='center')

    # Row 0: Full views
    ax = fig.add_subplot(gs[0, 0], projection='3d')
    render_mesh(ax, verts, faces, face_colors, elev=25, azim=-50,
               title='Full — Oblique', zoom=0.85, damage_vals=fd)
    # Add crosshair indicator — draw box around zoom region
    ax.plot([peak_v[0]]*2, [peak_v[1]-zoom_half, peak_v[1]+zoom_half],
            [peak_v[2], peak_v[2]], color='cyan', lw=2, alpha=0.8)

    ax = fig.add_subplot(gs[0, 1])
    render_ct_slice(ax, fractured_ct, mask, damage_vol=damage_vol_ct,
                   axis=0, title=f'Full — Axial (slice {peak[0]})')

    ax = fig.add_subplot(gs[0, 2])
    render_ct_slice(ax, fractured_ct, mask, damage_vol=damage_vol_ct,
                   axis=1, title=f'Full — Sagittal (slice {peak[1]})')

    # Row 1: Zoomed views
    ax = fig.add_subplot(gs[1, 0], projection='3d')
    render_mesh(ax, verts, faces, face_colors, elev=25, azim=-50,
               title='ZOOM — Fracture Zone', zoom=2.5, damage_vals=fd)
    # Restrict view to fracture zone
    zr = zoom_half * 1.2
    ax.set_xlim(peak_v[0]-zr, peak_v[0]+zr)
    ax.set_ylim(peak_v[1]-zr, peak_v[1]+zr)
    ax.set_zlim(peak_v[2]-zr, peak_v[2]+zr)

    # Zoomed CT axial
    ax = fig.add_subplot(gs[1, 1])
    img_z, d_z = get_zoom_slice(fractured_ct, damage_vol_ct, 0, peak[0])
    norm_z = np.clip((img_z - BONE_MIN) / (BONE_MAX - BONE_MIN), 0, 1)
    rgb_z = np.stack([norm_z]*3, axis=-1)
    # Stress zone (warm tint) vs crack zone (dark lines)
    sz = np.clip(d_z * 3.0, 0, 1) * np.clip(1.0 - (d_z - 0.2) / 0.1, 0, 1)
    cz = np.clip((d_z - 0.15) / 0.15, 0, 1)
    rgb_z[:,:,0] = np.clip(rgb_z[:,:,0] + sz * 0.4, 0, 1)
    rgb_z[:,:,1] = rgb_z[:,:,1] * (1 - sz * 0.3)
    rgb_z[:,:,2] = rgb_z[:,:,2] * (1 - sz * 0.3)
    rgb_z[:,:,0] = rgb_z[:,:,0] * (1 - cz * 0.7)
    rgb_z[:,:,1] = rgb_z[:,:,1] * (1 - cz * 0.85)
    rgb_z[:,:,2] = rgb_z[:,:,2] * (1 - cz * 0.85)
    ax.imshow(rgb_z, aspect='auto', interpolation='bilinear')
    ax.set_title('ZOOM — Axial CT', fontsize=11, fontweight='bold', color='cyan')
    ax.axis('off')

    # Zoomed CT sagittal
    ax = fig.add_subplot(gs[1, 2])
    img_z, d_z = get_zoom_slice(fractured_ct, damage_vol_ct, 1, peak[1])
    norm_z = np.clip((img_z - BONE_MIN) / (BONE_MAX - BONE_MIN), 0, 1)
    rgb_z = np.stack([norm_z]*3, axis=-1)
    sz = np.clip(d_z * 3.0, 0, 1) * np.clip(1.0 - (d_z - 0.2) / 0.1, 0, 1)
    cz = np.clip((d_z - 0.15) / 0.15, 0, 1)
    rgb_z[:,:,0] = np.clip(rgb_z[:,:,0] + sz * 0.4, 0, 1)
    rgb_z[:,:,1] = rgb_z[:,:,1] * (1 - sz * 0.3)
    rgb_z[:,:,2] = rgb_z[:,:,2] * (1 - sz * 0.3)
    rgb_z[:,:,0] = rgb_z[:,:,0] * (1 - cz * 0.7)
    rgb_z[:,:,1] = rgb_z[:,:,1] * (1 - cz * 0.85)
    rgb_z[:,:,2] = rgb_z[:,:,2] * (1 - cz * 0.85)
    ax.imshow(rgb_z, aspect='auto', interpolation='bilinear')
    ax.set_title('ZOOM — Sagittal CT', fontsize=11, fontweight='bold', color='cyan')
    ax.axis('off')

    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print(f"  ✅ {ao_type} zoom fracture: {output_path}")


def generate_cascade_analysis(ct, mask, output_path):
    """Cascade analysis: damage heatmaps + damage curves per AO type.
    
    Produces a rich analysis figure inspired by the old cascade plots:
    - Row 0: Per-AO damage cascade heatmaps (anatomical zones × time)
    - Row 1: Per-AO damage curves (peak, mean, %damaged, %fractured over time)
    - Row 2: Final-state bar chart comparing AO types
    """
    positions = sample_particles(mask)
    ao_types = ['A1', 'A2', 'A3', 'A4']
    
    # Define anatomical zones by spatial partitioning
    # Split vertebra into anterior/central/posterior (by Y), endplate/mid (by Z)
    p_min = positions.min(axis=0)
    p_max = positions.max(axis=0)
    p_range = p_max - p_min
    
    zone_names = ['Anterior\nBody', 'Central\nBody', 'Posterior\nBody',
                  'Cortical\nShell', 'Endplate', 'Pedicle', 'Lamina']
    n_zones = len(zone_names)
    
    def classify_zones(pos):
        """Assign each particle to an anatomical zone."""
        norm = (pos - p_min) / (p_range + 1e-8)  # normalized [0,1]
        zones = np.zeros(len(pos), dtype=int)
        # Y dimension: anterior (0-0.33), central (0.33-0.66), posterior (0.66-1.0)
        zones[norm[:, 1] < 0.33] = 0  # Anterior body
        zones[(norm[:, 1] >= 0.33) & (norm[:, 1] < 0.66)] = 1  # Central body
        zones[norm[:, 1] >= 0.66] = 2  # Posterior body
        # Overrides based on other dims
        # Cortical shell: near surface (extreme X or Y or Z positions)
        dist = np.minimum(norm, 1 - norm).min(axis=1)
        zones[dist < 0.08] = 3  # Cortical shell
        # Endplate: top/bottom Z
        zones[(norm[:, 2] < 0.12) | (norm[:, 2] > 0.88)] = 4  # Endplate
        # Pedicle: posterior + mid-height + lateral
        pedicle_mask = ((norm[:, 1] > 0.55) & (norm[:, 2] > 0.3) & 
                       (norm[:, 2] < 0.7) & ((norm[:, 0] < 0.25) | (norm[:, 0] > 0.75)))
        zones[pedicle_mask] = 5  # Pedicle
        # Lamina: far posterior + mid-height
        lamina_mask = (norm[:, 1] > 0.82) & (norm[:, 2] > 0.25) & (norm[:, 2] < 0.75)
        zones[lamina_mask] = 6  # Lamina
        return zones
    
    zones = classify_zones(positions)
    
    # Run simulations and collect history per AO type
    all_histories = {}
    n_record_steps = 40  # Number of timepoints to record
    
    for ao_type in ao_types:
        config = AO_LOAD_CONFIGS[ao_type]
        sim = BoneFractureSimulator(positions.copy(), seed=42)
        sim.setup_ao_loading(ao_type)
        sim.set_stress_mode('grid')
        sim.enable_anisotropy(ratio=2.0)
        configure_for_real_vertebra(sim)
        scaled_force = config['max_force'] * FORCE_SCALE
        
        zone_damage_history = np.zeros((n_record_steps, n_zones))
        peak_damage_hist = []
        mean_damage_hist = []
        damaged_frac_hist = []
        fractured_frac_hist = []
        step_indices = []
        
        n_steps = 200
        record_every = max(1, n_steps // n_record_steps)
        idx = 0
        
        for step in range(n_steps + 1):
            force = scaled_force * step / n_steps
            sim.compute_stress(force)
            sim.evolve_damage()
            
            if step % record_every == 0 and idx < n_record_steps:
                d = sim.damage.copy()
                step_indices.append(step)
                peak_damage_hist.append(d.max())
                mean_damage_hist.append(d[d > 0.001].mean() if (d > 0.001).any() else 0)
                damaged_frac_hist.append((d > 0.01).mean())
                fractured_frac_hist.append((d > 0.8).mean())
                
                for z in range(n_zones):
                    zmask = zones == z
                    if zmask.any():
                        zone_damage_history[idx, z] = d[zmask].mean()
                idx += 1
        
        all_histories[ao_type] = {
            'zone_damage': zone_damage_history[:idx],
            'peak': peak_damage_hist,
            'mean': mean_damage_hist,
            'damaged_frac': damaged_frac_hist,
            'fractured_frac': fractured_frac_hist,
            'steps': step_indices,
            'config': config,
        }
    
    # --- Create figure ---
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor('#0a0a12')
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.25,
                  height_ratios=[1.0, 0.8, 0.6])
    
    fig.suptitle('AO Fracture Cascade Analysis — Damage Propagation by Anatomical Zone',
                 fontsize=16, fontweight='bold', color='white', y=0.97)
    
    # Colors matching the cascade style
    cmap_heat = plt.cm.inferno  # Hot heatmap for damage
    
    # --- Row 0: Damage cascade heatmaps ---
    for col, ao_type in enumerate(ao_types):
        hist = all_histories[ao_type]
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor('#0a0a12')
        
        # Heatmap: zones × time
        data = hist['zone_damage'].T  # shape: (n_zones, n_steps)
        im = ax.imshow(data, aspect='auto', cmap=cmap_heat, 
                       vmin=0, vmax=max(0.01, data.max()),
                       interpolation='bilinear',
                       extent=[0, len(hist['steps'])-1, n_zones-0.5, -0.5])
        
        ax.set_yticks(range(n_zones))
        ax.set_yticklabels(zone_names, fontsize=7, color='#ccc')
        ax.set_xlabel('Simulation Step', fontsize=8, color='#aaa')
        
        # Convert x ticks to actual step numbers
        n_pts = len(hist['steps'])
        tick_pos = np.linspace(0, n_pts-1, 5).astype(int)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([str(hist['steps'][i]) for i in tick_pos], 
                          fontsize=7, color='#aaa')
        
        ax.set_title(f"{hist['config']['name']}", fontsize=10, 
                    fontweight='bold', color='#ff6644', pad=8)
        ax.tick_params(colors='#666', labelsize=7)
        for sp in ax.spines.values():
            sp.set_color('#333')
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.ax.tick_params(labelsize=6, colors='#aaa')
        cbar.set_label('Mean Damage', fontsize=7, color='#aaa')
    
    # --- Row 1: Damage curves per AO type ---
    ao_colors = {'A1': '#ff4444', 'A2': '#ff8844', 'A3': '#ffaa00', 'A4': '#ff2288'}
    
    for col, ao_type in enumerate(ao_types):
        hist = all_histories[ao_type]
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor('#0a0a12')
        
        steps = hist['steps']
        
        # Peak damage
        ax.plot(steps, hist['peak'], color='#ff2222', lw=2.0, label='Peak D')
        ax.fill_between(steps, hist['peak'], alpha=0.15, color='#ff2222')
        
        # Mean damage (non-zero particles)
        ax.plot(steps, hist['mean'], color='#ffaa00', lw=1.5, label='Mean D', ls='--')
        
        # %Damaged and %Fractured as filled areas
        damaged_pct = [d * 100 for d in hist['damaged_frac']]
        fractured_pct = [f * 100 for f in hist['fractured_frac']]
        ax2 = ax.twinx()
        ax2.fill_between(steps, damaged_pct, alpha=0.2, color='#44aaff')
        ax2.plot(steps, damaged_pct, color='#44aaff', lw=1.0, label='%Damaged', ls=':')
        ax2.fill_between(steps, fractured_pct, alpha=0.2, color='#ff44ff')
        ax2.plot(steps, fractured_pct, color='#ff44ff', lw=1.0, label='%Fractured', ls=':')
        ax2.set_ylim(0, max(max(damaged_pct) * 1.5, 1))
        ax2.set_ylabel('%', fontsize=7, color='#888')
        ax2.tick_params(colors='#666', labelsize=6)
        
        # Threshold lines
        ax.axhline(y=0.08, color='#ff4444', lw=0.5, ls=':', alpha=0.5)
        ax.axhline(y=0.25, color='#ff8844', lw=0.5, ls=':', alpha=0.5)
        ax.text(steps[-1]*0.02, 0.085, 'crack onset', fontsize=5, color='#ff4444', alpha=0.7)
        ax.text(steps[-1]*0.02, 0.255, 'full crack', fontsize=5, color='#ff8844', alpha=0.7)
        
        ax.set_xlabel('Step', fontsize=7, color='#aaa')
        ax.set_ylabel('Damage Index', fontsize=7, color='#aaa')
        ax.set_ylim(0, max(max(hist['peak']) * 1.2, 0.05))
        ax.tick_params(colors='#666', labelsize=6)
        ax.legend(fontsize=6, loc='upper left', framealpha=0.3, 
                 labelcolor='#ccc', facecolor='#1a1a2e')
        for sp in ax.spines.values():
            sp.set_color('#333')
        for sp in ax2.spines.values():
            sp.set_color('#333')
    
    # --- Row 2: Final-state comparison bar chart ---
    ax = fig.add_subplot(gs[2, :2])
    ax.set_facecolor('#0a0a12')
    
    x = np.arange(len(ao_types))
    bar_w = 0.35
    
    final_damaged = [all_histories[ao]['damaged_frac'][-1] * 100 for ao in ao_types]
    final_fractured = [all_histories[ao]['fractured_frac'][-1] * 100 for ao in ao_types]
    
    bars1 = ax.bar(x - bar_w/2, final_damaged, bar_w, color='#ff4444', alpha=0.8, 
                   label='% Damaged (D>0.01)')
    bars2 = ax.bar(x + bar_w/2, final_fractured, bar_w, color='#ffaa00', alpha=0.8,
                   label='% Fractured (D>0.8)')
    
    ax.set_xticks(x)
    ao_labels = [all_histories[ao]['config']['name'] for ao in ao_types]
    ax.set_xticklabels(ao_labels, fontsize=8, color='#ccc')
    ax.set_ylabel('% Particles', fontsize=9, color='#aaa')
    ax.set_title('Final Damage Distribution by AO Type', fontsize=11, 
                fontweight='bold', color='white', pad=8)
    ax.legend(fontsize=8, framealpha=0.3, labelcolor='#ccc', facecolor='#1a1a2e')
    ax.tick_params(colors='#666', labelsize=7)
    for sp in ax.spines.values():
        sp.set_color('#333')
    
    # Bar value labels
    for bar in bars1:
        if bar.get_height() > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{bar.get_height():.1f}%', ha='center', fontsize=7, color='#ff6666')
    for bar in bars2:
        if bar.get_height() > 0.1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{bar.get_height():.1f}%', ha='center', fontsize=7, color='#ffcc00')
    
    # Peak damage comparison
    ax3 = fig.add_subplot(gs[2, 2:])
    ax3.set_facecolor('#0a0a12')
    
    peak_damages = [all_histories[ao]['peak'][-1] for ao in ao_types]
    mean_damages = [all_histories[ao]['mean'][-1] for ao in ao_types]
    
    bars3 = ax3.bar(x - bar_w/2, peak_damages, bar_w, color='#ff2222', alpha=0.8,
                    label='Peak Damage')
    bars4 = ax3.bar(x + bar_w/2, mean_damages, bar_w, color='#44aaff', alpha=0.8,
                    label='Mean Damage (non-zero)')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(ao_labels, fontsize=8, color='#ccc')
    ax3.set_ylabel('Damage Index', fontsize=9, color='#aaa')
    ax3.set_title('Peak vs Mean Damage by AO Type', fontsize=11,
                 fontweight='bold', color='white', pad=8)
    ax3.legend(fontsize=8, framealpha=0.3, labelcolor='#ccc', facecolor='#1a1a2e')
    ax3.tick_params(colors='#666', labelsize=7)
    for sp in ax3.spines.values():
        sp.set_color('#333')
    
    # Threshold line
    ax3.axhline(y=0.08, color='#ff4444', lw=0.8, ls='--', alpha=0.5)
    ax3.text(0, 0.085, 'crack onset', fontsize=6, color='#ff4444', alpha=0.7)
    
    # Bar value labels
    for bar in bars3:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', fontsize=7, color='#ff6666')
    for bar in bars4:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', fontsize=7, color='#66ccff')
    
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print(f"  ✅ Cascade analysis: {output_path}")


if __name__ == '__main__':
    print("=" * 70)
    print("Combined CT + 3D Mesh Fracture — Real VerSe Vertebra")
    print("=" * 70)

    print("\n1. Loading vertebra...")
    ct, mask, label = load_vertebra(CT_PATH, MASK_PATH)

    print("\n2. Extracting mesh (marching cubes)...")
    verts, faces = extract_mesh(mask, step_size=2)
    print(f"  {len(verts)} vertices, {len(faces)} faces")

    print("\n3. Per-AO combined CT + 3D mesh cards...")
    for ao in ['A1', 'A2', 'A3', 'A4']:
        generate_combined_card(ao, ct, mask, verts, faces,
                               OUTPUT_DIR / f'v2_{ao}_combined.png')

    print("\n4. A1-A4 comparison...")
    generate_combined_comparison(ct, mask, verts, faces,
                                 OUTPUT_DIR / 'v2_combined_comparison.png')

    print("\n5. Stress diffusion GIFs...")
    for ao in ['A1', 'A4']:
        generate_diffusion_gif(ao, ct, mask, verts, faces,
                               OUTPUT_DIR / f'v2_{ao}_mesh_diffusion.gif', fps=4)

    print("\n6. Zoom-in fracture views...")
    for ao in ['A1', 'A2', 'A3', 'A4']:
        generate_zoom_fracture(ao, ct, mask, verts, faces,
                               OUTPUT_DIR / f'v2_{ao}_zoom.png')

    print("\n7. Cascade analysis plots...")
    generate_cascade_analysis(ct, mask, OUTPUT_DIR / 'v2_cascade_analysis.png')

    print(f"\n✅ Done! {OUTPUT_DIR}")
