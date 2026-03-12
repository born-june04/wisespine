#!/usr/bin/env python3
"""WiseSpine v5 — True Fracture Visualization with Crack Surfaces

Element deletion + fragment separation + 3D isosurface rendering.
Shows actual cracks forming in the vertebra, not just color overlays.

Key approach:
  1. Run FEM simulation → damage field per element
  2. Element deletion: remove voxels where damage > threshold
  3. Connected components → identify bone fragments
  4. Apply displacement warping → fragments physically separate
  5. Render each fragment in different color with visible crack gaps

Usage:
  python pipeline/modules/fracture_visualization.py --output-dir ./fracture_3d --cuda
"""

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from fracture_engine_v5 import (
    VoxelFEMEngine, CausalParameters, _AO_COLORS, HAS_MPL
)
from _gen_real_fracture_visuals import load_vertebra

if HAS_MPL:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import Normalize
    from matplotlib import cm

from scipy.ndimage import label as ndimage_label
from scipy.ndimage import zoom, binary_erosion

try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# ============================================================================
#  FRAGMENT EXTRACTION
# ============================================================================

def extract_fragments(bone_mask, damage_vol, crack_threshold=0.5):
    """Remove damaged voxels and find connected fragments.
    
    Args:
        bone_mask: 3D binary mask of bone
        damage_vol: 3D damage field [0, 1]
        crack_threshold: damage above this = element deleted (crack)
    
    Returns:
        fragment_labels: 3D array with fragment IDs (0=crack/air)
        n_fragments: number of connected fragments
        crack_mask: 3D binary mask of deleted (cracked) elements
    """
    # Delete elements above threshold → creates gaps = cracks
    crack_mask = damage_vol > crack_threshold
    surviving_mask = (bone_mask > 0) & (~crack_mask)
    
    # Erode slightly to widen cracks for visibility
    surviving_mask = binary_erosion(surviving_mask, iterations=1) | surviving_mask
    
    # Connected component labeling → fragments
    fragment_labels, n_fragments = ndimage_label(surviving_mask.astype(np.int32))
    
    return fragment_labels, n_fragments, crack_mask


def separate_fragments(fragment_labels, ux_vol, uy_vol, uz_vol,
                       separation_factor=3.0):
    """Apply displacement + extra separation between fragments.
    
    Each fragment centroid gets an additional displacement push
    to make the separation visually clear.
    """
    n_frags = fragment_labels.max()
    if n_frags <= 1:
        return ux_vol.copy(), uy_vol.copy(), uz_vol.copy()
    
    # Find centroid of entire bone
    all_bone = fragment_labels > 0
    global_centroid = np.array([
        np.mean(np.where(all_bone)[0]),
        np.mean(np.where(all_bone)[1]),
        np.mean(np.where(all_bone)[2]),
    ])
    
    ux_sep = ux_vol.copy()
    uy_sep = uy_vol.copy()
    uz_sep = uz_vol.copy()
    
    for frag_id in range(1, n_frags + 1):
        frag_mask = fragment_labels == frag_id
        if frag_mask.sum() < 10:  # skip tiny fragments
            continue
        
        # Fragment centroid
        coords = np.where(frag_mask)
        centroid = np.array([coords[0].mean(), coords[1].mean(), coords[2].mean()])
        
        # Direction from global center → this fragment
        direction = centroid - global_centroid
        dist = np.linalg.norm(direction)
        if dist > 0.1:
            direction /= dist
        else:
            direction = np.array([0, 0, 1])
        
        # Add extra separation displacement
        ux_sep[frag_mask] += direction[0] * separation_factor
        uy_sep[frag_mask] += direction[1] * separation_factor
        uz_sep[frag_mask] += direction[2] * separation_factor
    
    return ux_sep, uy_sep, uz_sep


# ============================================================================
#  FRAGMENT COLORS
# ============================================================================

FRAGMENT_COLORS = [
    (0.90, 0.87, 0.80, 0.85),  # bone
    (0.95, 0.70, 0.65, 0.85),  # warm fragment
    (0.75, 0.82, 0.90, 0.85),  # cool fragment
    (0.85, 0.78, 0.70, 0.85),  # tan fragment
    (0.80, 0.85, 0.75, 0.85),  # sage fragment
    (0.88, 0.75, 0.82, 0.85),  # mauve fragment
    (0.78, 0.80, 0.88, 0.85),  # slate fragment
]


def get_fragment_face_colors(verts, faces, fragment_labels):
    """Assign color per face based on which fragment it belongs to."""
    colors = np.zeros((len(faces), 4))
    for fi in range(len(faces)):
        # Sample fragment ID at face center
        center = verts[faces[fi]].mean(axis=0).astype(int)
        x = np.clip(center[0], 0, fragment_labels.shape[0]-1)
        y = np.clip(center[1], 0, fragment_labels.shape[1]-1)
        z = np.clip(center[2], 0, fragment_labels.shape[2]-1)
        frag_id = fragment_labels[x, y, z]
        colors[fi] = FRAGMENT_COLORS[frag_id % len(FRAGMENT_COLORS)]
    return colors


# ============================================================================
#  3D RENDERING
# ============================================================================

def render_vertebra_3d(ax, verts, faces, face_colors, title='', elev=15, azim=-60):
    """Render 3D mesh."""
    if len(faces) == 0:
        ax.text(0.5, 0.5, 0.5, 'No surface', ha='center', color='white')
        return
    mesh = Poly3DCollection(verts[faces], alpha=0.85, linewidths=0.05)
    mesh.set_facecolor(face_colors)
    mesh.set_edgecolor((0.3, 0.3, 0.3, 0.1))
    ax.add_collection3d(mesh)
    
    # Axis bounds
    max_range = max(verts.ptp(axis=0)) / 2 * 1.2
    mid = verts.mean(axis=0)
    ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
    ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
    ax.set_zlim(mid[2]-max_range, mid[2]+max_range)
    ax.set_title(title, fontsize=11, color='white', pad=3)
    ax.set_facecolor('#0d1117')
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#222')
    ax.tick_params(colors='#444', labelsize=5)
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    ax.view_init(elev=elev, azim=azim)


def warp_vertices(verts, ux, uy, uz, mag=10.0):
    """Warp mesh vertices by displacement field."""
    warped = verts.copy()
    sh = ux.shape
    for i in range(len(verts)):
        x = int(np.clip(verts[i,0], 0, sh[0]-1))
        y = int(np.clip(verts[i,1], 0, sh[1]-1))
        z = int(np.clip(verts[i,2], 0, sh[2]-1))
        warped[i,0] += ux[x,y,z] * mag
        warped[i,1] += uy[x,y,z] * mag
        warped[i,2] += uz[x,y,z] * mag
    return warped


# ============================================================================
#  MAIN: A0-A4 FRACTURE PANORAMA
# ============================================================================

def run_fracture_visualization(output_dir, verse_root, use_cuda=False):
    """Generate A0-A4 fracture visualizations with actual cracks."""
    print("=" * 70)
    print("WiseSpine v5 — True Fracture Visualization")
    print("  Element deletion + fragment separation + 3D rendering")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading sub-verse503...")
    ct_path = os.path.join(verse_root, 'rawdata', 'sub-verse503',
                           'sub-verse503_dir-ax_ct.nii.gz')
    mask_path = os.path.join(verse_root, 'derivatives', 'sub-verse503',
                             'sub-verse503_dir-ax_seg-vert_msk.nii.gz')
    ct, mask, label, spacing = load_vertebra(ct_path, mask_path, return_spacing=True)
    mask = (mask > 0).astype(np.int32)
    voxel_size = float(spacing.mean())
    
    ds = max(1, int(np.cbrt(mask.sum() / 200000))) if use_cuda else \
         max(1, int(np.cbrt(mask.sum() / 30000)))
    
    engine = VoxelFEMEngine(mask, ct, voxel_size_mm=voxel_size,
                            downsample=ds, seed=42, use_cuda=use_cuda)
    
    # Full-res mask for visualization
    vis_mask = engine._orig_mask if engine.ds > 1 else engine.mask
    
    # Scenarios A0 → A4
    scenarios = [
        ('A0\nIntact', CausalParameters(1.0, 10.0, 0.0, 1.2)),
        ('A1\nWedge', CausalParameters(3.0, 20.0, 0.0, 0.8)),
        ('A2\nSplit', CausalParameters(5.0, 10.0, 0.0, 0.7)),
        ('A3\nBurst', CausalParameters(6.0, 10.0, 0.0, 0.55)),
        ('A4\nComplete Burst', CausalParameters(8.0, 5.0, 0.0, 0.4)),
    ]
    
    # ===== Extract original surface =====
    print("\nExtracting original surface...")
    vis_float = (vis_mask > 0).astype(np.float32)
    mc_result = marching_cubes(vis_float, level=0.5, step_size=2)
    verts_orig, faces_orig = mc_result[0], mc_result[1]
    print(f"  Original surface: {len(verts_orig)} verts, {len(faces_orig)} faces")
    
    # ===== Run all scenarios =====
    sim_results = []
    for name, params in scenarios:
        print(f"\n{'─'*50}")
        print(f"  {name.split(chr(10))[0]}: F={params.force_magnitude}kN, BMD={params.bmd_factor}")
        print(f"{'─'*50}")
        engine.set_causal_params(params)
        result = engine.simulate(max_damage_iters=5)
        
        # Extract displacement + damage volumes (full res)
        u = engine._displacement
        damage = engine._damage
        ux_e = np.zeros(engine.n_elements)
        uy_e = np.zeros(engine.n_elements)
        uz_e = np.zeros(engine.n_elements)
        for n in range(8):
            ux_e += u[engine._elem_dofs[:, n*3+0]]
            uy_e += u[engine._elem_dofs[:, n*3+1]]
            uz_e += u[engine._elem_dofs[:, n*3+2]]
        ux_e /= 8; uy_e /= 8; uz_e /= 8
        
        sim_results.append({
            'name': name, 'params': params, 'result': result,
            'ux_vol': engine._to_3d(ux_e),
            'uy_vol': engine._to_3d(uy_e),
            'uz_vol': engine._to_3d(uz_e),
            'dmg_vol': engine._to_3d(damage.astype(np.float32)),
        })
        print(f"  → {result.ao_type}, yield={result.yielded_fraction*100:.1f}%")
    
    # ===== RENDER: A0-A4 PANORAMA =====
    print("\n" + "="*70)
    print("Rendering fracture panorama with element deletion...")
    
    n = len(sim_results)
    fig = plt.figure(figsize=(6*n, 18), facecolor='#0d1117')
    
    for col, res in enumerate(sim_results):
        ao = res['result'].ao_type
        ao_c = _AO_COLORS.get(ao, '#888')
        dmg = res['dmg_vol']
        
        # --- Row 1: Original (intact) ---
        ax1 = fig.add_subplot(3, n, col+1, projection='3d')
        bone_colors = np.full((len(faces_orig), 4), [0.92, 0.89, 0.82, 0.8])
        render_vertebra_3d(ax1, verts_orig, faces_orig, bone_colors,
                           title=f'{res["name"]}', elev=15, azim=-60)
        
        # --- Row 2: After fracture — element deletion + cracks ---
        ax2 = fig.add_subplot(3, n, n+col+1, projection='3d')
        
        # Element deletion: remove damaged voxels
        threshold = 0.3 if ao in ('A0', 'A1') else 0.25
        frag_labels, n_frags, crack_mask = extract_fragments(
            vis_mask, dmg, crack_threshold=threshold)
        
        # Get remaining bone surface
        remaining = (frag_labels > 0).astype(np.float32)
        if remaining.sum() > 100:
            try:
                mc = marching_cubes(remaining, level=0.5, step_size=2)
                v_frac, f_frac = mc[0], mc[1]
                fcolors = get_fragment_face_colors(v_frac, f_frac, frag_labels)
                
                # Apply displacement warping
                mag = 8.0 if ao in ('A0', 'A1') else 12.0
                v_warped = warp_vertices(v_frac, res['ux_vol'], res['uy_vol'],
                                         res['uz_vol'], mag=mag)
                
                render_vertebra_3d(ax2, v_warped, f_frac, fcolors,
                    title=f'{ao} — {n_frags} fragments\nyield={res["result"].yielded_fraction*100:.0f}%',
                    elev=15, azim=-60)
            except Exception as e:
                ax2.text(0.5, 0.5, 0.5, f'Error: {e}', color='red',
                         transform=ax2.transAxes)
        
        # --- Row 3: Exploded view — fragments separated ---
        ax3 = fig.add_subplot(3, n, 2*n+col+1, projection='3d')
        
        if remaining.sum() > 100 and n_frags > 1:
            try:
                # Extra separation between fragments
                sep = 5.0 if ao in ('A3', 'A4') else 2.0
                ux_s, uy_s, uz_s = separate_fragments(
                    frag_labels, res['ux_vol'], res['uy_vol'], res['uz_vol'],
                    separation_factor=sep)
                
                v_exploded = warp_vertices(v_frac, ux_s, uy_s, uz_s, mag=mag)
                render_vertebra_3d(ax3, v_exploded, f_frac, fcolors,
                    title=f'Exploded ({n_frags} fragments)',
                    elev=25, azim=-45)
            except Exception as e:
                render_vertebra_3d(ax3, v_warped, f_frac, fcolors,
                    title='(no separation)', elev=25, azim=-45)
        else:
            # Single fragment or too little damage
            if remaining.sum() > 100:
                render_vertebra_3d(ax3, v_warped, f_frac, fcolors,
                    title='Intact (1 fragment)', elev=25, azim=-45)
    
    # Row labels
    fig.text(0.01, 0.83, 'Original', fontsize=14, color='#4fc3f7',
             rotation=90, va='center', fontweight='bold')
    fig.text(0.01, 0.50, 'Fractured', fontsize=14, color='#ff5252',
             rotation=90, va='center', fontweight='bold')
    fig.text(0.01, 0.17, 'Exploded', fontsize=14, color='#ffa726',
             rotation=90, va='center', fontweight='bold')
    
    fig.suptitle('Vertebral Fracture: Element Deletion + Fragment Separation',
                 fontsize=18, color='white', fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.03, 0, 1, 0.95])
    out = os.path.join(output_dir, 'fracture_panorama_3d.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()
    
    # ===== INDIVIDUAL: 4-view per AO type =====
    for res in sim_results:
        _render_detail_views(res, vis_mask, output_dir)
    
    # ===== 2D CROSS-SECTION: sagittal crack view =====
    _render_2d_crack_sections(sim_results, vis_mask, ct, output_dir)
    
    # ===== GIF: crack progression =====
    _render_crack_gif(engine, vis_mask, ct, output_dir, use_cuda)
    
    print(f"\n✅ All visualizations saved to {output_dir}/")


def _render_detail_views(res, vis_mask, output_dir):
    """4-view detail for one AO type."""
    ao = res['result'].ao_type
    dmg = res['dmg_vol']
    threshold = 0.3 if ao in ('A0', 'A1') else 0.25
    frag_labels, n_frags, _ = extract_fragments(vis_mask, dmg, threshold)
    remaining = (frag_labels > 0).astype(np.float32)
    if remaining.sum() < 100:
        return
    
    try:
        mc = marching_cubes(remaining, level=0.5, step_size=2)
        v, f = mc[0], mc[1]
    except:
        return
    
    fcolors = get_fragment_face_colors(v, f, frag_labels)
    mag = 8.0 if ao in ('A0', 'A1') else 12.0
    v_w = warp_vertices(v, res['ux_vol'], res['uy_vol'], res['uz_vol'], mag=mag)
    
    views = [('Anterior', 0, -90), ('Lateral', 0, 0),
             ('Superior', 90, 0), ('Oblique', 20, -45)]
    
    fig = plt.figure(figsize=(20, 5), facecolor='#0d1117')
    for i, (name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        render_vertebra_3d(ax, v_w, f, fcolors, title=name, elev=elev, azim=azim)
    
    ao_c = _AO_COLORS.get(ao, '#888')
    fig.suptitle(f'{ao} — {n_frags} fragments | F={res["params"].force_magnitude}kN, '
                 f'BMD={res["params"].bmd_factor}',
                 fontsize=14, color=ao_c, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(output_dir, f'detail_{ao}_4views.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


def _render_2d_crack_sections(sim_results, vis_mask, ct_orig, output_dir):
    """2D sagittal/coronal/axial showing cracks as black gaps."""
    n = len(sim_results)
    fig, axes = plt.subplots(3, n, figsize=(5*n, 14), facecolor='#0d1117')
    
    mid_x = vis_mask.shape[0] // 2
    mid_y = vis_mask.shape[1] // 2
    mid_z = vis_mask.shape[2] // 2
    
    # Get original CT at full res
    ct_full = ct_orig
    if ct_full.shape != vis_mask.shape:
        ct_full = zoom(ct_orig.astype(np.float32),
                       np.array(vis_mask.shape) / np.array(ct_orig.shape), order=1)
    
    for col, res in enumerate(sim_results):
        ao = res['result'].ao_type
        ao_c = _AO_COLORS.get(ao, '#888')
        dmg = res['dmg_vol']
        
        threshold = 0.3 if ao in ('A0', 'A1') else 0.25
        crack = dmg > threshold
        
        # Create fractured CT: cracks become black (air HU)
        ct_frac = ct_full.copy()
        ct_frac[crack & (vis_mask > 0)] = -1000  # air
        
        kw = dict(cmap='bone', origin='lower', vmin=-200, vmax=800, aspect='auto')
        
        # Sagittal
        axes[0, col].imshow(ct_frac[mid_x].T, **kw)
        axes[0, col].set_title(f'{res["name"].split(chr(10))[0]}\n{ao}',
                               fontsize=11, color=ao_c, fontweight='bold')
        axes[0, col].axis('off')
        
        # Coronal
        axes[1, col].imshow(ct_frac[:, mid_y, :].T, **kw)
        axes[1, col].axis('off')
        
        # Axial
        axes[2, col].imshow(ct_frac[:, :, mid_z].T, **kw)
        axes[2, col].axis('off')
    
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor('#0d1117')
    
    fig.text(0.01, 0.78, 'Sagittal', fontsize=12, color='#888',
             rotation=90, va='center')
    fig.text(0.01, 0.50, 'Coronal', fontsize=12, color='#888',
             rotation=90, va='center')
    fig.text(0.01, 0.22, 'Axial', fontsize=12, color='#888',
             rotation=90, va='center')
    
    fig.suptitle('Fracture Cross-Sections (black = crack gaps)',
                 fontsize=16, color='white', fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.03, 0, 1, 0.95])
    out = os.path.join(output_dir, 'crack_cross_sections.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


def _render_crack_gif(engine, vis_mask, ct_orig, output_dir, use_cuda=False):
    """Animated GIF showing crack growing over load steps."""
    from PIL import Image
    import io
    
    print("\n  Generating crack progression GIF...")
    
    # Re-run burst scenario with frame capture
    engine.set_causal_params(CausalParameters(8.0, 5.0, 0.0, 0.5))
    engine._capture_frames = True
    engine._frames = []
    result = engine.simulate(max_damage_iters=5)
    engine._capture_frames = False
    
    if not engine._frames:
        print("  ⚠ No frames captured")
        return
    
    frames_pil = []
    for fi, frame in enumerate(engine._frames):
        dmg = frame['damage']
        dmg_vol = engine._to_3d(dmg.astype(np.float32))
        
        # Element deletion at progressive threshold
        threshold = 0.3
        crack = dmg_vol > threshold
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='#0d1117')
        
        mid_x = vis_mask.shape[0] // 2
        ct_full = ct_orig
        if ct_full.shape != vis_mask.shape:
            ct_full = zoom(ct_orig.astype(np.float32),
                           np.array(vis_mask.shape)/np.array(ct_orig.shape), order=1)
        
        # Left: CT with cracks
        ct_frac = ct_full.copy()
        ct_frac[crack & (vis_mask > 0)] = -1000
        ax1.imshow(ct_frac[mid_x].T, cmap='bone', origin='lower',
                   vmin=-200, vmax=800, aspect='auto')
        ax1.set_title(f'Step {frame["step"]+1}, Iter {frame["iteration"]+1}',
                      color='white', fontsize=12)
        ax1.axis('off')
        ax1.set_facecolor('#0d1117')
        
        # Right: damage map
        dmg_slice = dmg_vol[mid_x].T
        bone_slice = (vis_mask[mid_x] > 0).T
        dmg_masked = np.ma.masked_where(~bone_slice, dmg_slice)
        ax2.imshow(ct_full[mid_x].T, cmap='bone', origin='lower',
                   vmin=-200, vmax=800, aspect='auto', alpha=0.3)
        ax2.imshow(dmg_masked, cmap='hot', origin='lower',
                   vmin=0, vmax=1, aspect='auto', alpha=0.8)
        ax2.contour(dmg_slice, levels=[threshold], colors=['cyan'],
                    linewidths=1.5, origin='lower')
        ax2.set_title(f'Damage (cyan = crack front)', color='white', fontsize=12)
        ax2.axis('off')
        ax2.set_facecolor('#0d1117')
        
        n_cracked = crack.sum()
        n_bone = (vis_mask > 0).sum()
        fig.suptitle(f'Crack Progression — A4 Burst (cracked: '
                     f'{n_cracked/n_bone*100:.1f}% of bone)',
                     fontsize=14, color='white', fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, facecolor='#0d1117')
        buf.seek(0)
        frames_pil.append(Image.open(buf).copy())
        plt.close(fig)
    
    if frames_pil:
        out = os.path.join(output_dir, 'crack_progression.gif')
        frames_pil[0].save(out, save_all=True, append_images=frames_pil[1:],
                           duration=500, loop=0)
        print(f"  Saved: {out} ({len(frames_pil)} frames)")


# ============================================================================
#  MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='WiseSpine v5 — True Fracture Visualization')
    parser.add_argument('--output-dir', type=str, default='./fracture_3d')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    
    verse_root = os.path.join(os.path.dirname(__file__), '..', '..',
                              'VerSe', 'dataset-01training')
    run_fracture_visualization(args.output_dir, verse_root, args.cuda)
