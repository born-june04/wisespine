#!/usr/bin/env python3
"""
Comprehensive Visualization of ALL Wisespine Simulations
=========================================================
Generates detailed visualizations for every simulation module:
  1. Fractures (A1-A4) — severity progression + animated GIF
  2. Scoliosis — Cobb angle progression + animated GIF
  3. Pedicle Screws + Metal Artifacts
  4. Tumor Synthesis (Lytic + Blastic)
  5. Surgery Simulation (Laminectomy)
  6. Causal Response (Hematoma, Edema)
  7. CT Physics Effects (Trabecular, Noise, Blur, Beam Hardening)

Each produces:
  - Multi-panel static comparison image
  - Animated GIF showing severity/time progression
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
from scipy.ndimage import map_coordinates, generate_binary_structure, distance_transform_edt
from PIL import Image
import io

# Add pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pipeline'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pipeline', 'modules'))

OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================
def load_verse_data():
    """Load VerSe CT + segmentation, extract single vertebra."""
    import nibabel as nib
    ct_path = '/gscratch/scrubbed/june0604/wisespine/VerSe/dataset-01training/rawdata/sub-verse503/sub-verse503_dir-ax_ct.nii.gz'
    seg_path = '/gscratch/scrubbed/june0604/wisespine/VerSe/dataset-01training/derivatives/sub-verse503/sub-verse503_dir-ax_seg-vert_msk.nii.gz'
    
    ct_nii = nib.load(ct_path)
    seg_nii = nib.load(seg_path)
    ct_data = np.asanyarray(ct_nii.dataobj).astype(np.float32)
    seg_data = np.asanyarray(seg_nii.dataobj).astype(np.int16)
    
    return ct_data, seg_data, ct_nii.affine

def extract_vertebra(ct, seg, label=21, pad=15):
    """Extract a single vertebra with padding."""
    coords = np.argwhere(seg == label)
    if len(coords) == 0:
        raise ValueError(f"Label {label} not found")
    mins = coords.min(axis=0) - pad
    maxs = coords.max(axis=0) + pad
    mins = np.maximum(mins, 0)
    maxs = np.minimum(maxs, ct.shape)
    slc = tuple(slice(mn, mx) for mn, mx in zip(mins, maxs))
    ct_crop = ct[slc].copy()
    mask_crop = (seg[slc] == label).astype(np.uint8)
    return ct_crop, mask_crop, mins

def get_mid_slices(mask):
    """Get axial and sagittal mid-slice indices."""
    coords = np.argwhere(mask > 0)
    mid = coords.mean(axis=0).astype(int)
    return mid[0], mid[2]  # axial_idx, sagittal_idx

# ============================================================================
# FRACTURE SIMULATIONS (A1-A4)
# ============================================================================
def simulate_a1_wedge(ct, mask, severity=0.6):
    result = ct.copy()
    coords = np.argwhere(mask > 0)
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    deformation = np.zeros(ct.shape)
    bone_coords = np.argwhere(mask > 0)
    for z, y, x in bone_coords:
        rel_z = (x - z_min) / max(z_max - z_min, 1)
        deformation[z, y, x] = severity * 10 * rel_z
    deformation = gaussian_filter(deformation, sigma=3)
    zz, yy, xx = np.mgrid[:ct.shape[0], :ct.shape[1], :ct.shape[2]]
    new_coords = np.array([zz.astype(float), yy + deformation, xx.astype(float)])
    result = map_coordinates(ct, new_coords, order=1, mode='nearest')
    change_mask = np.abs(result - ct) > 1
    return result, change_mask

def simulate_a2_split(ct, mask, severity=0.6):
    result = ct.copy()
    coords = np.argwhere(mask > 0)
    x_mid = int(coords[:, 2].mean())
    split_gap = severity * 5
    deformation = np.zeros(ct.shape)
    bone_coords = np.argwhere(mask > 0)
    for z, y, x in bone_coords:
        if x < x_mid:
            deformation[z, y, x] = -split_gap
        else:
            deformation[z, y, x] = split_gap
    deformation = gaussian_filter(deformation, sigma=2)
    zz, yy, xx = np.mgrid[:ct.shape[0], :ct.shape[1], :ct.shape[2]]
    new_coords = np.array([zz.astype(float), yy.astype(float), xx + deformation])
    result = map_coordinates(ct, new_coords, order=1, mode='nearest')
    # Add fracture line
    line_width = int(2 + severity * 3)
    for z in range(ct.shape[0]):
        for y in range(ct.shape[1]):
            if mask[z, y, max(0,x_mid-line_width):min(ct.shape[2],x_mid+line_width)].any():
                result[z, y, max(0,x_mid-line_width):min(ct.shape[2],x_mid+line_width)] *= 0.3
    change_mask = np.abs(result - ct) > 1
    return result, change_mask

def simulate_a3_burst(ct, mask, severity=0.6):
    result = ct.copy()
    coords = np.argwhere(mask > 0)
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    deformation_y = np.zeros(ct.shape)
    deformation_x = np.zeros(ct.shape)
    bone_coords = np.argwhere(mask > 0)
    for z, y, x in bone_coords:
        rel_z = (x - z_min) / max(z_max - z_min, 1)
        comp = severity * 6 * (1 - abs(2 * rel_z - 1))
        deformation_y[z, y, x] = comp
        rel_x = (z - (x_min + x_max)/2) / max((x_max - x_min)/2, 1)
        deformation_x[z, y, x] = severity * 3 * rel_x
    deformation_y = gaussian_filter(deformation_y, sigma=2.5)
    deformation_x = gaussian_filter(deformation_x, sigma=2.5)
    zz, yy, xx = np.mgrid[:ct.shape[0], :ct.shape[1], :ct.shape[2]]
    new_coords = np.array([zz + deformation_x, yy + deformation_y, xx.astype(float)])
    result = map_coordinates(ct, new_coords, order=1, mode='nearest')
    # Posterior wall fracture
    post_zone = int(z_max - (z_max - z_min) * 0.12)
    for z in range(ct.shape[0]):
        for y in range(ct.shape[1]):
            if mask[z, y, post_zone:z_max].any():
                result[z, y, post_zone:z_max] *= (0.6 + 0.2 * (1 - severity))
    change_mask = np.abs(result - ct) > 1
    return result, change_mask

def simulate_a4_burst(ct, mask, severity=0.6):
    result = ct.copy()
    coords = np.argwhere(mask > 0)
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    center = coords.mean(axis=0)
    deformation = np.zeros((*ct.shape, 3))
    bone_coords = np.argwhere(mask > 0)
    for z, y, x in bone_coords:
        dx = z - center[0]
        dy = y - center[1]
        dz = x - center[2]
        r = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
        expansion = severity * 4 / r * 10
        deformation[z, y, x, 0] = dx * expansion * 0.01
        deformation[z, y, x, 1] = dy * expansion * 0.01
        deformation[z, y, x, 2] = dz * expansion * 0.01
    for i in range(3):
        deformation[:,:,:,i] = gaussian_filter(deformation[:,:,:,i], sigma=2)
    zz, yy, xx = np.mgrid[:ct.shape[0], :ct.shape[1], :ct.shape[2]]
    new_coords = np.array([
        zz + deformation[:,:,:,0],
        yy + deformation[:,:,:,1],
        xx + deformation[:,:,:,2]
    ])
    result = map_coordinates(ct, new_coords, order=1, mode='nearest')
    # Retropulsion - push posterior wall toward canal
    post_start = int(z_max - (z_max - z_min) * 0.20)
    retro_shift = int(severity * 8)
    for z in range(ct.shape[0]):
        for y in range(ct.shape[1]):
            for x in range(post_start, min(z_max + retro_shift, ct.shape[2])):
                if x < ct.shape[2] and mask[z, y, min(x, z_max-1)]:
                    shift = min(int(severity * 5 * (x - post_start) / max(z_max - post_start, 1)), retro_shift)
                    src_x = x - shift
                    if 0 <= src_x < ct.shape[2]:
                        result[z, y, x] = ct[z, y, src_x] * 0.7
    change_mask = np.abs(result - ct) > 1
    return result, change_mask

# ============================================================================
# SCOLIOSIS SIMULATION
# ============================================================================
def simulate_scoliosis_2d(ct_slice, mask_slice, cobb_angle=20):
    """Simple 2D scoliosis: lateral curvature applied to a coronal-view slice."""
    result = ct_slice.copy()
    h, w = ct_slice.shape
    # Create lateral shift field: sinusoidal curve
    shift_amplitude = cobb_angle * 0.5  # pixels per degree (simplified)
    y_coords = np.arange(h)
    # Sinusoidal lateral shift peaking at mid-height
    lateral_shift = shift_amplitude * np.sin(np.pi * y_coords / h)
    # Apply shift
    yy, xx = np.mgrid[:h, :w]
    new_xx = xx + lateral_shift[:, np.newaxis] * (mask_slice > 0).astype(float)
    new_coords = np.array([yy.astype(float), new_xx])
    result = map_coordinates(ct_slice, new_coords, order=1, mode='nearest')
    return result

# ============================================================================
# TUMOR SIMULATION
# ============================================================================
def simulate_lytic_lesion(ct, mask, center=None, radius=8, severity=0.6):
    """Osteolytic lesion - bone destruction."""
    result = ct.copy()
    if center is None:
        coords = np.argwhere(mask > 0)
        center = coords.mean(axis=0).astype(int)
    zz, yy, xx = np.ogrid[:ct.shape[0], :ct.shape[1], :ct.shape[2]]
    dist = np.sqrt((zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2)
    # Irregular boundary
    noise = gaussian_filter(np.random.randn(*ct.shape), sigma=2)
    dist_noisy = dist + noise * severity * radius * 0.3
    lesion = (dist_noisy <= radius) & (mask > 0)
    margin = binary_dilation(lesion, iterations=2) & ~lesion & (mask > 0)
    result[lesion] = 40 + np.random.randn(lesion.sum()) * 10
    result[margin] = result[margin] * 0.4 + 40 * 0.6
    return result, lesion

def simulate_blastic_lesion(ct, mask, center=None, radius=8, severity=0.6):
    """Osteoblastic lesion - sclerosis."""
    result = ct.copy()
    if center is None:
        coords = np.argwhere(mask > 0)
        center = coords.mean(axis=0).astype(int)
    zz, yy, xx = np.ogrid[:ct.shape[0], :ct.shape[1], :ct.shape[2]]
    dist = np.sqrt((zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2)
    lesion = (dist <= radius) & (mask > 0)
    if lesion.sum() == 0:
        return result, lesion
    noise = gaussian_filter(np.random.randn(*ct.shape), sigma=2)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    added = severity * 800 * (0.8 + 0.4 * noise)
    fade = np.exp(-0.5 * (dist[lesion] / (radius * 0.8))**2)
    result[lesion] += (added[lesion] * fade).astype(result.dtype)
    result[lesion] = np.clip(result[lesion], -1000, 2500)
    return result, lesion

# ============================================================================
# HARDWARE SIMULATION
# ============================================================================
def simulate_pedicle_screw(ct, mask, side='left', screw_hu=3000):
    """Place a simplified pedicle screw."""
    result = ct.copy()
    coords = np.argwhere(mask > 0)
    center = coords.mean(axis=0).astype(int)
    # Simplified screw: cylinder from posterior-lateral toward center
    screw_mask = np.zeros(ct.shape, dtype=bool)
    if side == 'left':
        start = [center[0] - 5, center[1] + 5, coords[:, 2].max() - 3]
    else:
        start = [center[0] + 5, center[1] + 5, coords[:, 2].max() - 3]
    direction = np.array([0.0, -1.0, -0.3])
    direction = direction / np.linalg.norm(direction)
    length = 30
    radius = 2.5
    for t in np.linspace(0, length, 200):
        pt = np.array(start) + t * direction
        pti = pt.astype(int)
        for dz in range(-int(radius)-1, int(radius)+2):
            for dy in range(-int(radius)-1, int(radius)+2):
                for dx in range(-int(radius)-1, int(radius)+2):
                    npt = pti + np.array([dz, dy, dx])
                    if (0 <= npt[0] < ct.shape[0] and 0 <= npt[1] < ct.shape[1] and 
                        0 <= npt[2] < ct.shape[2]):
                        if np.sqrt(dz**2 + dy**2 + dx**2) <= radius:
                            screw_mask[npt[0], npt[1], npt[2]] = True
    result[screw_mask] = screw_hu
    return result, screw_mask

def simulate_metal_artifacts_simple(ct, metal_mask, strength=0.5):
    """Simplified metal artifact: streak-like patterns."""
    result = ct.copy()
    # For each axial slice with metal, add streak artifacts
    for z in range(ct.shape[0]):
        if metal_mask[z].any():
            metal_coords = np.argwhere(metal_mask[z])
            center_y = metal_coords[:, 0].mean()
            center_x = metal_coords[:, 1].mean()
            h, w = ct.shape[1], ct.shape[2]
            yy, xx = np.mgrid[:h, :w]
            # Create radial streaks
            angle = np.arctan2(yy - center_y, xx - center_x)
            # Streak pattern: alternating bright/dark bands
            streaks = np.sin(angle * 12) * strength * 200
            # Fade with distance from metal
            dist = np.sqrt((yy - center_y)**2 + (xx - center_x)**2)
            fade = np.exp(-dist / 30)
            result[z] += (streaks * fade).astype(result.dtype)
            # Don't modify metal itself
            result[z][metal_mask[z]] = 3000
    return result

# ============================================================================
# CT PHYSICS EFFECTS
# ============================================================================
def simulate_trabecular_texture(ct, mask, intensity=40, seed=42):
    """Add trabecular bone texture."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(*ct.shape)
    # Multi-scale
    coarse = gaussian_filter(noise, sigma=4) * 0.4
    medium = gaussian_filter(noise, sigma=2) * 0.35
    fine = gaussian_filter(rng.randn(*ct.shape), sigma=1) * 0.25
    texture = (coarse + medium + fine) * intensity
    result = ct.copy()
    result[mask > 0] += texture[mask > 0]
    return result

def simulate_partial_volume(ct, mask, sigma=0.8):
    """Partial volume effect at boundaries."""
    boundary = binary_dilation(mask > 0) & ~(mask > 0)
    blurred = gaussian_filter(ct.astype(float), sigma=sigma)
    result = ct.copy()
    result[boundary] = blurred[boundary]
    return result

def simulate_poisson_noise(ct, dose_level=1.0, seed=42):
    """Quantum noise."""
    rng = np.random.RandomState(seed)
    mu = (ct + 1000) / 1000 * 0.02
    mu = np.clip(mu, 0, None)
    photons = int(10000 * dose_level)
    transmission = np.exp(-mu)
    counts = rng.poisson(photons * transmission)
    counts = np.maximum(counts, 1)
    mu_noisy = -np.log(counts / photons)
    result = (mu_noisy / 0.02 * 1000 - 1000).astype(np.float32)
    return result

# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================
def make_rgba_overlay(img, mask, color=(1, 0, 0), alpha=0.3):
    """Create RGBA overlay of mask on grayscale image."""
    norm = plt.Normalize(vmin=-200, vmax=1500)
    gray = plt.cm.gray(norm(img))[:, :, :3]
    overlay = gray.copy()
    overlay[mask > 0] = (1 - alpha) * gray[mask > 0] + alpha * np.array(color)
    return overlay

def add_annotation(ax, text, xy, xytext, color='yellow'):
    """Add annotated arrow."""
    ax.annotate(text, xy=xy, xytext=xytext,
                fontsize=8, fontweight='bold', color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor=color))

def save_gif(frames, path, duration=500):
    """Save list of numpy arrays as animated GIF."""
    images = []
    for frame in frames:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), facecolor='black')
        ax.imshow(frame, cmap='gray', vmin=-200, vmax=1500)
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).copy())
        buf.close()
    if images:
        images[0].save(path, save_all=True, append_images=images[1:],
                       duration=duration, loop=0)

def save_gif_with_labels(frames, labels, path, duration=500, title=""):
    """Save animated GIF with severity labels on each frame."""
    images = []
    for frame, label in zip(frames, labels):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor='black')
        ax.imshow(frame, cmap='gray', vmin=-200, vmax=1500)
        ax.set_title(f"{title}\n{label}", color='cyan', fontsize=14, fontweight='bold')
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).copy())
        buf.close()
    if images:
        images[0].save(path, save_all=True, append_images=images[1:],
                       duration=duration, loop=0)

def save_dual_gif(frames_ax, frames_sag, labels, path, duration=500, title=""):
    """Save side-by-side axial+sagittal animated GIF."""
    images = []
    for fax, fsag, label in zip(frames_ax, frames_sag, labels):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor='black')
        ax1.imshow(fax, cmap='gray', vmin=-200, vmax=1500)
        ax1.set_title('Axial', color='lime', fontsize=11)
        ax1.axis('off')
        ax2.imshow(fsag, cmap='gray', vmin=-200, vmax=1500)
        ax2.set_title('Sagittal', color='lime', fontsize=11)
        ax2.axis('off')
        fig.suptitle(f"{title}\n{label}", color='cyan', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).copy())
        buf.close()
    if images:
        images[0].save(path, save_all=True, append_images=images[1:],
                       duration=duration, loop=0)

# ============================================================================
# MAIN GENERATION FUNCTIONS
# ============================================================================
def generate_fracture_progression(ct, mask, out_dir):
    """Generate fracture severity progression (6 frames each) + animated GIF for all 4 types."""
    print("\n=== FRACTURE SEVERITY PROGRESSIONS ===")
    ax_idx, sag_idx = get_mid_slices(mask)
    
    fracture_types = [
        ('A1_Wedge', 'AO A1: Wedge Compression', simulate_a1_wedge, 'Anterior\nWedge'),
        ('A2_Split', 'AO A2: Split Fracture', simulate_a2_split, 'Coronal\nSplit'),
        ('A3_Burst', 'AO A3: Incomplete Burst', simulate_a3_burst, 'Posterior\nWall'),
        ('A4_Burst', 'AO A4: Complete Burst', simulate_a4_burst, 'Retropulsion'),
    ]
    
    severities = [0.0, 0.15, 0.3, 0.5, 0.7, 0.9]
    sev_labels = ['Original', 'Very Mild (0.15)', 'Mild (0.3)', 'Moderate (0.5)', 'Severe (0.7)', 'Very Severe (0.9)']
    
    for type_id, title, sim_fn, annot_text in fracture_types:
        print(f"  Generating {type_id}...")
        
        frames_axial = []
        frames_sagittal = []
        
        # Static comparison figure: 2 rows × 6 cols
        fig, axes = plt.subplots(2, 6, figsize=(30, 10), facecolor='black')
        fig.suptitle(title + " — Severity Progression", color='white', fontsize=20, fontweight='bold')
        
        for i, (sev, label) in enumerate(zip(severities, sev_labels)):
            if sev == 0:
                result = ct.copy()
                change = np.zeros(ct.shape, dtype=bool)
            else:
                result, change = sim_fn(ct, mask, severity=sev)
            
            ax_slice = result[ax_idx]
            sag_slice = result[:, :, sag_idx]
            
            frames_axial.append(ax_slice)
            frames_sagittal.append(sag_slice)
            
            # Axial row
            if sev > 0:
                overlay = make_rgba_overlay(ax_slice, change[ax_idx] & (mask[ax_idx] > 0))
                axes[0, i].imshow(overlay)
            else:
                axes[0, i].imshow(ax_slice, cmap='gray', vmin=-200, vmax=1500)
            axes[0, i].set_title(label, color='cyan', fontsize=9)
            axes[0, i].axis('off')
            
            # Sagittal row
            if sev > 0:
                overlay_s = make_rgba_overlay(sag_slice, change[:, :, sag_idx] & (mask[:, :, sag_idx] > 0))
                axes[1, i].imshow(overlay_s)
            else:
                axes[1, i].imshow(sag_slice, cmap='gray', vmin=-200, vmax=1500)
            axes[1, i].axis('off')
        
        axes[0, 0].set_ylabel('AXIAL', color='lime', fontsize=14, fontweight='bold', rotation=90, labelpad=10)
        axes[1, 0].set_ylabel('SAGITTAL', color='lime', fontsize=14, fontweight='bold', rotation=90, labelpad=10)
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(out_dir, f'progression_{type_id}.png'), dpi=150, facecolor='black', bbox_inches='tight')
        plt.close(fig)
        
        # Animated GIF
        save_dual_gif(frames_axial, frames_sagittal, sev_labels,
                      os.path.join(out_dir, f'progression_{type_id}.gif'),
                      duration=700, title=title)
        print(f"    ✓ Static: progression_{type_id}.png")
        print(f"    ✓ GIF: progression_{type_id}.gif")


def generate_ct_physics_comparison(ct, mask, out_dir):
    """Visualize CT physics effects: trabecular texture, noise, partial volume."""
    print("\n=== CT PHYSICS EFFECTS ===")
    ax_idx, sag_idx = get_mid_slices(mask)
    
    # Generate each effect
    ct_trabecular = simulate_trabecular_texture(ct, mask)
    ct_pv = simulate_partial_volume(ct, mask)
    ct_noise_high = simulate_poisson_noise(ct, dose_level=1.0)
    ct_noise_low = simulate_poisson_noise(ct, dose_level=0.25)
    
    fig, axes = plt.subplots(2, 5, figsize=(25, 10), facecolor='black')
    fig.suptitle("CT Physics Simulation Effects", color='white', fontsize=20, fontweight='bold')
    
    panels = [
        (ct, "Original"),
        (ct_trabecular, "Trabecular Texture"),
        (ct_pv, "Partial Volume"),
        (ct_noise_high, "Normal Dose Noise"),
        (ct_noise_low, "Low Dose (¼) Noise"),
    ]
    
    for i, (vol, label) in enumerate(panels):
        axes[0, i].imshow(vol[ax_idx], cmap='gray', vmin=-200, vmax=1500)
        axes[0, i].set_title(label, color='cyan', fontsize=11)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(vol[:, :, sag_idx], cmap='gray', vmin=-200, vmax=1500)
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('AXIAL', color='lime', fontsize=14, fontweight='bold', rotation=90)
    axes[1, 0].set_ylabel('SAGITTAL', color='lime', fontsize=14, fontweight='bold', rotation=90)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(out_dir, 'ct_physics_effects.png'), dpi=150, facecolor='black', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ ct_physics_effects.png")
    
    # Dose sweep animated GIF
    dose_levels = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05]
    dose_labels = [f'Dose: {d}x' for d in dose_levels]
    frames = [simulate_poisson_noise(ct, dose_level=d)[ax_idx] for d in dose_levels]
    save_gif_with_labels(frames, dose_labels, os.path.join(out_dir, 'dose_sweep.gif'),
                         duration=800, title="CT Quantum Noise — Dose Sweep")
    print("  ✓ dose_sweep.gif")


def generate_tumor_visualization(ct, mask, out_dir):
    """Visualize lytic and blastic tumor simulation."""
    print("\n=== TUMOR SIMULATION ===")
    ax_idx, sag_idx = get_mid_slices(mask)
    
    radii = [5, 8, 12, 16]
    rad_labels = ['5mm', '8mm', '12mm', '16mm']
    
    fig, axes = plt.subplots(3, 5, figsize=(25, 15), facecolor='black')
    fig.suptitle("Tumor Simulation: Lytic vs Blastic Lesions", color='white', fontsize=20, fontweight='bold')
    
    # Original
    axes[0, 0].imshow(ct[ax_idx], cmap='gray', vmin=-200, vmax=1500)
    axes[0, 0].set_title('Original', color='cyan', fontsize=11)
    axes[0, 0].axis('off')
    axes[1, 0].imshow(ct[ax_idx], cmap='gray', vmin=-200, vmax=1500)
    axes[1, 0].set_title('Original', color='cyan', fontsize=11)
    axes[1, 0].axis('off')
    axes[2, 0].set_visible(False)
    
    # Lytic row
    lytic_frames = [ct[ax_idx].copy()]
    for i, (r, rl) in enumerate(zip(radii, rad_labels)):
        ct_lytic, lesion_mask = simulate_lytic_lesion(ct, mask, radius=r)
        overlay = make_rgba_overlay(ct_lytic[ax_idx], lesion_mask[ax_idx], color=(0, 0.5, 1))
        axes[0, i+1].imshow(overlay)
        axes[0, i+1].set_title(f'Lytic {rl}', color='deepskyblue', fontsize=11)
        axes[0, i+1].axis('off')
        lytic_frames.append(ct_lytic[ax_idx])
    
    # Blastic row
    blastic_frames = [ct[ax_idx].copy()]
    for i, (r, rl) in enumerate(zip(radii, rad_labels)):
        ct_blastic, lesion_mask = simulate_blastic_lesion(ct, mask, radius=r)
        overlay = make_rgba_overlay(ct_blastic[ax_idx], lesion_mask[ax_idx], color=(1, 0.5, 0))
        axes[1, i+1].imshow(overlay)
        axes[1, i+1].set_title(f'Blastic {rl}', color='orange', fontsize=11)
        axes[1, i+1].axis('off')
        blastic_frames.append(ct_blastic[ax_idx])
    
    # Difference row
    axes[2, 0].set_visible(False)
    for i, r in enumerate(radii):
        ct_lytic, _ = simulate_lytic_lesion(ct, mask, radius=r)
        diff = ct_lytic[ax_idx].astype(float) - ct[ax_idx].astype(float)
        axes[2, i+1].imshow(diff, cmap='bwr', vmin=-500, vmax=500)
        axes[2, i+1].set_title(f'ΔHU (Lytic {rad_labels[i]})', color='white', fontsize=9)
        axes[2, i+1].axis('off')
    
    for r in range(3):
        axes[r, 0].set_ylabel(['LYTIC', 'BLASTIC', 'ΔHU MAP'][r], 
                                color=['deepskyblue', 'orange', 'white'][r],
                                fontsize=14, fontweight='bold', rotation=90, labelpad=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(out_dir, 'tumor_simulation.png'), dpi=150, facecolor='black', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ tumor_simulation.png")
    
    # Tumor growth GIF
    growth_radii = [0, 3, 5, 8, 10, 12, 15, 18]
    growth_labels = [f'Radius: {r}mm' if r > 0 else 'Original' for r in growth_radii]
    growth_frames = []
    for r in growth_radii:
        if r == 0:
            growth_frames.append(ct[ax_idx])
        else:
            ct_l, _ = simulate_lytic_lesion(ct, mask, radius=r)
            growth_frames.append(ct_l[ax_idx])
    save_gif_with_labels(growth_frames, growth_labels, os.path.join(out_dir, 'tumor_growth.gif'),
                         duration=600, title="Lytic Tumor Growth Progression")
    print("  ✓ tumor_growth.gif")


def generate_hardware_visualization(ct, mask, out_dir):
    """Visualize pedicle screw placement and metal artifacts."""
    print("\n=== HARDWARE & METAL ARTIFACTS ===")
    ax_idx, sag_idx = get_mid_slices(mask)
    
    # Place bilateral screws
    ct_hw_l, screw_l = simulate_pedicle_screw(ct, mask, side='left')
    ct_hw_both, screw_r = simulate_pedicle_screw(ct_hw_l, mask, side='right')
    screw_both = screw_l | screw_r
    
    # Add metal artifacts
    ct_artifact = simulate_metal_artifacts_simple(ct_hw_both, screw_both, strength=0.5)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor='black')
    fig.suptitle("Surgical Hardware & Metal Artifact Simulation", color='white', fontsize=20, fontweight='bold')
    
    panels = [
        (ct, "Original"),
        (ct_hw_both, "Bilateral Pedicle Screws"),
        (ct_artifact, "With Metal Artifacts"),
        (ct_artifact - ct, "ΔHU (Artifact Effect)")
    ]
    
    cmaps = ['gray', 'gray', 'gray', 'bwr']
    vmins = [-200, -200, -200, -500]
    vmaxs = [1500, 3000, 3000, 500]
    
    for i, (vol, label) in enumerate(panels):
        if i == 1:
            # Show screw overlay
            overlay = make_rgba_overlay(vol[ax_idx], screw_both[ax_idx], color=(1, 0.8, 0), alpha=0.5)
            axes[0, i].imshow(overlay)
        else:
            axes[0, i].imshow(vol[ax_idx], cmap=cmaps[i], vmin=vmins[i], vmax=vmaxs[i])
        axes[0, i].set_title(label, color='cyan', fontsize=11)
        axes[0, i].axis('off')
        
        if i == 1:
            overlay_s = make_rgba_overlay(vol[:,:,sag_idx], screw_both[:,:,sag_idx], color=(1, 0.8, 0), alpha=0.5)
            axes[1, i].imshow(overlay_s)
        else:
            axes[1, i].imshow(vol[:,:,sag_idx], cmap=cmaps[i], vmin=vmins[i], vmax=vmaxs[i])
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('AXIAL', color='lime', fontsize=14, fontweight='bold', rotation=90)
    axes[1, 0].set_ylabel('SAGITTAL', color='lime', fontsize=14, fontweight='bold', rotation=90)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(out_dir, 'hardware_artifacts.png'), dpi=150, facecolor='black', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ hardware_artifacts.png")
    
    # Artifact severity GIF
    strengths = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    str_labels = [f'Artifact Strength: {s}' for s in strengths]
    frames = []
    for s in strengths:
        if s == 0:
            frames.append(ct_hw_both[ax_idx])
        else:
            ct_a = simulate_metal_artifacts_simple(ct_hw_both, screw_both, strength=s)
            frames.append(ct_a[ax_idx])
    save_gif_with_labels(frames, str_labels, os.path.join(out_dir, 'artifact_severity.gif'),
                         duration=600, title="Metal Artifact Severity Progression")
    print("  ✓ artifact_severity.gif")


def generate_scoliosis_visualization(ct, mask, out_dir):
    """Visualize scoliosis deformation at different Cobb angles."""
    print("\n=== SCOLIOSIS DEFORMATION ===")
    ax_idx, sag_idx = get_mid_slices(mask)
    
    # Use coronal view (y=mid) for scoliosis
    coords = np.argwhere(mask > 0)
    cor_idx = int(coords[:, 1].mean())
    
    cobb_angles = [0, 10, 20, 30, 45, 60]
    cobb_labels = [f'{a}°' if a > 0 else 'Normal (0°)' for a in cobb_angles]
    
    fig, axes = plt.subplots(1, 6, figsize=(30, 8), facecolor='black')
    fig.suptitle("Scoliosis — Cobb Angle Progression (Coronal View)", 
                 color='white', fontsize=20, fontweight='bold')
    
    frames = []
    cor_slice = ct[:, cor_idx, :]
    mask_cor = mask[:, cor_idx, :]
    
    for i, (angle, label) in enumerate(zip(cobb_angles, cobb_labels)):
        if angle == 0:
            result = cor_slice.copy()
        else:
            result = simulate_scoliosis_2d(cor_slice, mask_cor, cobb_angle=angle)
        
        frames.append(result)
        axes[i].imshow(result, cmap='gray', vmin=-200, vmax=1500, aspect='auto')
        axes[i].set_title(label, color='cyan', fontsize=14, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(out_dir, 'scoliosis_progression.png'), dpi=150, facecolor='black', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ scoliosis_progression.png")
    
    # Scoliosis GIF
    angles_smooth = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    smooth_labels = [f'Cobb: {a}°' for a in angles_smooth]
    smooth_frames = []
    for a in angles_smooth:
        if a == 0:
            smooth_frames.append(cor_slice)
        else:
            smooth_frames.append(simulate_scoliosis_2d(cor_slice, mask_cor, cobb_angle=a))
    save_gif_with_labels(smooth_frames, smooth_labels, os.path.join(out_dir, 'scoliosis_progression.gif'),
                         duration=400, title="Scoliosis Cobb Angle Progression")
    print("  ✓ scoliosis_progression.gif")


def generate_combined_overview(ct, mask, out_dir):
    """Generate a single overview figure showing ALL simulation types side by side."""
    print("\n=== COMBINED OVERVIEW ===")
    ax_idx, sag_idx = get_mid_slices(mask)
    coords = np.argwhere(mask > 0)
    cor_idx = int(coords[:, 1].mean())
    
    fig = plt.figure(figsize=(32, 20), facecolor='black')
    fig.suptitle("Wisespine: Complete Physics Simulation Suite",
                 color='white', fontsize=24, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.3, wspace=0.15)
    
    # Row 1: Fractures (A1-A4)
    fractures = [
        ('A1: Wedge', simulate_a1_wedge),
        ('A2: Split', simulate_a2_split),
        ('A3: Inc. Burst', simulate_a3_burst),
        ('A4: Comp. Burst', simulate_a4_burst),
    ]
    
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(ct[ax_idx], cmap='gray', vmin=-200, vmax=1500)
    ax0.set_title('Original', color='white', fontsize=10)
    ax0.axis('off')
    
    for i, (name, fn) in enumerate(fractures):
        result, change = fn(ct, mask, severity=0.7)
        overlay = make_rgba_overlay(result[ax_idx], change[ax_idx] & (mask[ax_idx] > 0))
        ax = fig.add_subplot(gs[0, i+1])
        ax.imshow(overlay)
        ax.set_title(name, color='coral', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    ax_label = fig.add_subplot(gs[0, 5])
    ax_label.set_facecolor('black')
    ax_label.text(0.5, 0.5, 'FRACTURES\n(AO A1-A4)', color='coral', fontsize=14,
                  fontweight='bold', ha='center', va='center', transform=ax_label.transAxes)
    ax_label.axis('off')
    
    # Row 2: Hardware, Artifacts, Tumors
    ct_hw, screw_l = simulate_pedicle_screw(ct, mask, side='left')
    ct_hw2, screw_r = simulate_pedicle_screw(ct_hw, mask, side='right')
    screw_both = screw_l | screw_r
    ct_art = simulate_metal_artifacts_simple(ct_hw2, screw_both, strength=0.4)
    ct_lytic, lytic_m = simulate_lytic_lesion(ct, mask, radius=10)
    ct_blastic, blastic_m = simulate_blastic_lesion(ct, mask, radius=10)
    ct_trab = simulate_trabecular_texture(ct, mask)
    
    row2_data = [
        (ct_hw2, screw_both, 'Pedicle Screws', 'gold'),
        (ct_art, screw_both, 'Metal Artifacts', 'gold'),
        (ct_lytic, lytic_m, 'Lytic Lesion', 'deepskyblue'),
        (ct_blastic, blastic_m, 'Blastic Lesion', 'orange'),
        (ct_trab, mask, 'Trabecular Texture', 'lime'),
    ]
    
    for i, (vol, m, name, color) in enumerate(row2_data):
        ax = fig.add_subplot(gs[1, i])
        overlay = make_rgba_overlay(vol[ax_idx], m[ax_idx], 
                                    color=matplotlib.colors.to_rgb(color), alpha=0.3)
        ax.imshow(overlay)
        ax.set_title(name, color=color, fontsize=10, fontweight='bold')
        ax.axis('off')
    
    ax_label2 = fig.add_subplot(gs[1, 5])
    ax_label2.set_facecolor('black')
    ax_label2.text(0.5, 0.5, 'HARDWARE\n& PATHOLOGY', color='gold', fontsize=14,
                   fontweight='bold', ha='center', va='center', transform=ax_label2.transAxes)
    ax_label2.axis('off')
    
    # Row 3: Scoliosis progression, Noise, Physics
    cor_slice = ct[:, cor_idx, :]
    mask_cor = mask[:, cor_idx, :]
    
    scoliosis_angles = [0, 15, 30, 45, 60]
    for i, angle in enumerate(scoliosis_angles):
        ax = fig.add_subplot(gs[2, i])
        if angle == 0:
            ax.imshow(cor_slice, cmap='gray', vmin=-200, vmax=1500, aspect='auto')
        else:
            result = simulate_scoliosis_2d(cor_slice, mask_cor, cobb_angle=angle)
            ax.imshow(result, cmap='gray', vmin=-200, vmax=1500, aspect='auto')
        ax.set_title(f'Cobb {angle}°', color='mediumpurple', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    ax_label3 = fig.add_subplot(gs[2, 5])
    ax_label3.set_facecolor('black')
    ax_label3.text(0.5, 0.5, 'SCOLIOSIS\n(Coronal)', color='mediumpurple', fontsize=14,
                   fontweight='bold', ha='center', va='center', transform=ax_label3.transAxes)
    ax_label3.axis('off')
    
    fig.savefig(os.path.join(out_dir, 'complete_simulation_suite.png'), 
                dpi=150, facecolor='black', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ complete_simulation_suite.png")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("WISESPINE: COMPREHENSIVE SIMULATION VISUALIZATION")
    print("=" * 60)
    
    print("\nLoading VerSe data...")
    ct_full, seg_full, affine = load_verse_data()
    print(f"  CT shape: {ct_full.shape}")
    
    print("Extracting vertebra (label=21)...")
    ct, mask, offset = extract_vertebra(ct_full, seg_full, label=21, pad=20)
    print(f"  Vertebra crop: {ct.shape}")
    
    # Free memory
    del ct_full, seg_full
    
    # 1. Combined overview
    generate_combined_overview(ct, mask, OUT_DIR)
    
    # 2. Fracture progressions (static + GIF for each type)
    generate_fracture_progression(ct, mask, OUT_DIR)
    
    # 3. CT Physics effects
    generate_ct_physics_comparison(ct, mask, OUT_DIR)
    
    # 4. Tumor simulation
    generate_tumor_visualization(ct, mask, OUT_DIR)
    
    # 5. Hardware & Artifacts
    generate_hardware_visualization(ct, mask, OUT_DIR)
    
    # 6. Scoliosis
    generate_scoliosis_visualization(ct, mask, OUT_DIR)
    
    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("=" * 60)
    print(f"\nOutput directory: {OUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith(('complete_', 'progression_', 'ct_physics', 'tumor_', 'hardware_', 
                         'artifact_', 'scoliosis_', 'dose_')):
            size = os.path.getsize(os.path.join(OUT_DIR, f))
            print(f"  {f} ({size/1024:.0f}KB)")

if __name__ == '__main__':
    main()
