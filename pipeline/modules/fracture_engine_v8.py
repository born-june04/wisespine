#!/usr/bin/env python3
"""
Hybrid Fracture Engine (v8)
============================

Two-stage approach that combines the best of v6 and v7:

Stage 1 — Implicit Phase Field (v6):
    Accurate crack field φ via staggered Ku=F + K_φ·φ=F_φ.
    Crack paths emerge from energy minimization.

Stage 2 — Fragment Dynamics (explicit):
    Erode elements where φ > threshold.
    Identify fragments (connected components).
    Animate fragment separation via lightweight explicit dynamics.

Usage:
    python fracture_engine_v8.py --cuda
"""

import os, sys, time, io
import numpy as np
from scipy.ndimage import zoom, label as ndimage_label
import scipy.sparse as sp

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

sys.path.insert(0, os.path.dirname(__file__))
from fracture_engine_v5 import (
    CausalParameters, FEMResult, _AO_COLORS,
)
from fracture_engine_v6 import PhaseFieldEngine

EROSION_THRESHOLD = 0.7  # φ_max of element — higher = less erosion, bigger fragments


class HybridFractureEngine:
    """Two-stage hybrid: implicit crack + explicit fragment dynamics."""
    
    def __init__(self, mask, ct, voxel_size_mm=1.0, downsample=4,
                 seed=42, use_cuda=False):
        self.seed = seed
        self.ds = max(int(downsample), 1)
        self.use_cuda = use_cuda
        self._orig_mask = mask
        self._orig_ct = ct
        self.voxel_size = voxel_size_mm
        self._frames = []
        
        # Create v6 engine for Stage 1
        self._v6 = PhaseFieldEngine(
            mask, ct, voxel_size_mm=voxel_size_mm,
            downsample=self.ds, seed=seed, use_cuda=use_cuda
        )
        
        print(f"  v8 Hybrid ready: {self._v6.n_elements} elements")
    
    def set_causal_params(self, params):
        params.validate()
        self.params = params
        self._v6.set_causal_params(params)
    
    # ================================================================
    #  STAGE 1: Implicit Phase Field (reuse v6)
    # ================================================================
    
    def _stage1_crack_field(self, n_load_steps=4, max_stagger=5, verbose=True):
        """Run v6 implicit solver for accurate crack field."""
        if verbose:
            print("\n" + "="*60)
            print("  STAGE 1: Implicit Phase Field Crack")
            print("="*60)
        
        self._v6._capture_frames = True
        self._v6._frames = []
        result = self._v6.simulate(
            n_load_steps=n_load_steps,
            max_stagger_iters=max_stagger,
            verbose=verbose
        )
        self._v6._capture_frames = False
        
        phi = self._v6._phi
        u = self._v6._displacement
        
        if verbose:
            phi_elem_max = phi[self._v6._elem_nodes].max(axis=1)
            n_cracked = (phi_elem_max > EROSION_THRESHOLD).sum()
            print(f"\n  Stage 1 complete: φ_max={phi.max():.3f}, "
                  f"cracked (>{EROSION_THRESHOLD:.1f}): "
                  f"{n_cracked}/{self._v6.n_elements} "
                  f"({n_cracked/self._v6.n_elements*100:.1f}%)")
        
        return phi, u, result
    
    # ================================================================
    #  STAGE 2: Fragment Dynamics (lightweight explicit)
    # ================================================================
    
    def _stage2_fragment_dynamics(self, phi, u_init, n_steps=200,
                                   verbose=True):
        """Animate fragment separation after erosion.
        
        This is NOT for crack propagation — that's already done in Stage 1.
        This only handles the post-fracture fragment motion (falling, separating).
        """
        if verbose:
            print("\n" + "="*60)
            print("  STAGE 2: Fragment Dynamics")
            print("="*60)
        
        v6 = self._v6
        h = v6.h
        n_elem = v6.n_elements
        elem_ijk = v6._elem_ijk
        elem_nodes = v6._elem_nodes
        
        # Erode elements
        phi_elem_max = phi[elem_nodes].max(axis=1)
        active = phi_elem_max < EROSION_THRESHOLD
        n_eroded = (~active).sum()
        
        if verbose:
            print(f"  Eroded: {n_eroded}/{n_elem} ({n_eroded/n_elem*100:.1f}%)")
        
        # Build active volume and find fragments
        active_vol = np.zeros(v6.shape, dtype=np.int32)
        active_ijk = elem_ijk[active]
        if len(active_ijk) > 0:
            active_vol[active_ijk[:,0], active_ijk[:,1], active_ijk[:,2]] = 1
        
        labeled, n_frags = ndimage_label(active_vol)
        
        # Filter out tiny fragments (< 50 voxels) — merge into largest
        if n_frags > 1:
            frag_sizes = {}
            for fid in range(1, n_frags + 1):
                frag_sizes[fid] = (labeled == fid).sum()
            largest_fid = max(frag_sizes, key=frag_sizes.get)
            
            MIN_FRAG_SIZE = 50
            small_frags = [fid for fid, sz in frag_sizes.items() 
                          if sz < MIN_FRAG_SIZE and fid != largest_fid]
            if small_frags:
                for fid in small_frags:
                    labeled[labeled == fid] = largest_fid
                # Relabel sequentially
                labeled, n_frags = ndimage_label(labeled > 0)
        
        if verbose:
            print(f"  Fragments: {n_frags}")
        
        if n_frags <= 1:
            if verbose:
                print("  Only 1 fragment — no separation dynamics needed")
            # Still generate frames for animation
            for i in range(min(n_steps, 20)):
                self._frames.append({
                    'phi': phi.copy(),
                    'active': active.copy(),
                    'fragment_labels': labeled.copy(),
                    'n_eroded': n_eroded,
                    'n_frags': n_frags,
                    'frag_offsets': {},
                    'step': i,
                })
            return active, labeled, n_frags
        
        # Fragment properties
        frag_info = {}
        for fid in range(1, n_frags + 1):
            frag_mask = labeled == fid
            frag_vox = np.argwhere(frag_mask)
            centroid = frag_vox.mean(axis=0)
            volume = len(frag_vox)
            
            # Height info for gravity
            si = (centroid[2] - elem_ijk[:,2].min()) / max(elem_ijk[:,2].ptp(), 1)
            
            frag_info[fid] = {
                'centroid': centroid,
                'volume': volume,
                'si': si,
                'velocity': np.zeros(3),
                'offset': np.zeros(3),
            }
        
        # Simple rigid body dynamics for fragments
        # Find biggest fragment (stays fixed)
        biggest = max(frag_info, key=lambda k: frag_info[k]['volume'])
        
        # Subtle separation — just enough to show crack gap
        dt_frag = 0.3
        max_offset = 2.0  # max 2 voxels separation
        
        for step in range(n_steps):
            for fid, info in frag_info.items():
                if fid == biggest:
                    continue  # anchor biggest fragment
                
                # Very gentle push away from center of biggest fragment
                dir_from_center = info['centroid'] - frag_info[biggest]['centroid']
                dir_norm = np.linalg.norm(dir_from_center)
                if dir_norm > 0:
                    push = dir_from_center / dir_norm * 0.02
                    f = push
                else:
                    f = np.zeros(3)
                
                # Euler with heavy damping
                info['velocity'] = info['velocity'] * 0.8 + f * dt_frag
                info['offset'] += info['velocity'] * dt_frag
                # Cap to subtle separation
                info['offset'] = np.clip(info['offset'], -max_offset, max_offset)
            
            # Capture frame
            if step % max(n_steps // 20, 1) == 0:
                offsets = {fid: info['offset'].copy() 
                          for fid, info in frag_info.items()}
                self._frames.append({
                    'phi': phi.copy(),
                    'active': active.copy(),
                    'fragment_labels': labeled.copy(),
                    'n_eroded': n_eroded,
                    'n_frags': n_frags,
                    'frag_offsets': offsets,
                    'step': step,
                })
        
        if verbose:
            for fid, info in frag_info.items():
                off = np.linalg.norm(info['offset'])
                print(f"    Fragment {fid}: {info['volume']} voxels, "
                      f"offset={off:.1f}mm")
        
        return active, labeled, n_frags
    
    # ================================================================
    #  FULL SIMULATION
    # ================================================================
    
    def simulate(self, n_load_steps=4, max_stagger=5, n_frag_steps=200,
                 verbose=True):
        """Run both stages."""
        self._frames = []
        
        # Stage 1: crack field
        phi, u, result = self._stage1_crack_field(
            n_load_steps=n_load_steps,
            max_stagger=max_stagger,
            verbose=verbose
        )
        
        # Stage 2: fragment dynamics
        active, labeled, n_frags = self._stage2_fragment_dynamics(
            phi, u, n_steps=n_frag_steps, verbose=verbose
        )
        
        # Update result with fragment info
        result.n_fragments = n_frags
        self._phi = phi
        self._active = active
        self._labeled = labeled
        self._result = result
        
        return result
    
    # ================================================================
    #  3D VISUALIZATION (marching cubes + matplotlib)
    # ================================================================
    
    def _extract_surface(self, volume, level=0.5):
        """Extract surface mesh from binary volume using marching cubes."""
        from skimage.measure import marching_cubes
        # Pad to ensure closed surface
        padded = np.pad(volume.astype(float), 1, mode='constant', constant_values=0)
        try:
            verts, faces, normals, _ = marching_cubes(padded, level=level)
            verts -= 1  # undo padding offset
            return verts, faces, normals
        except:
            return None, None, None
    
    def _render_3d_bone(self, ax, verts, faces, face_colors=None,
                         alpha=0.9, edge_alpha=0.1):
        """Render a 3D bone mesh on a matplotlib axes."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Build triangle collection
        triangles = verts[faces]
        
        mesh = Poly3DCollection(triangles, alpha=alpha)
        
        if face_colors is not None:
            mesh.set_facecolors(face_colors)
        else:
            mesh.set_facecolor([1.0, 0.96, 0.88, alpha])  # bright bone
        
        mesh.set_edgecolor([0.7, 0.65, 0.6, edge_alpha * 0.5])
        mesh.set_linewidth(0.3)
        ax.add_collection3d(mesh)
    
    def _set_3d_axes(self, ax, verts, elev=25, azim=-60):
        """Configure 3D axes with proper limits and clean look."""
        margin = 2
        ax.set_xlim(verts[:,0].min()-margin, verts[:,0].max()+margin)
        ax.set_ylim(verts[:,1].min()-margin, verts[:,1].max()+margin)
        ax.set_zlim(verts[:,2].min()-margin, verts[:,2].max()+margin)
        ax.set_facecolor('#1a1e2e')
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        # Equal aspect
        ranges = np.array([verts[:,i].ptp() for i in range(3)])
        max_range = ranges.max() / 2
        mid = verts.mean(axis=0)
        ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
        ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
        ax.set_zlim(mid[2]-max_range, mid[2]+max_range)
    
    def _phi_to_face_colors(self, phi_vol, verts, faces):
        """Map φ values to face colors: intact=bone, cracked=red."""
        shape = np.array(phi_vol.shape)
        n_faces = len(faces)
        
        # Sample φ at face centroids
        centroids = verts[faces].mean(axis=1)  # (n_faces, 3)
        # Clamp to volume bounds
        ijk = np.clip(centroids, 0, shape - 1).astype(int)
        phi_face = phi_vol[ijk[:,0], ijk[:,1], ijk[:,2]]
        
        # Color map: bright bone (φ=0) → yellow (φ=0.3) → red (φ=0.8) → dark red (φ=1)
        colors = np.zeros((n_faces, 4))
        for i in range(n_faces):
            p = phi_face[i]
            if p < 0.2:
                # Intact bone: bright ivory
                colors[i] = [1.0, 0.96, 0.88, 0.95]
            elif p < 0.5:
                # Early damage: warm yellow
                t = (p - 0.2) / 0.3
                colors[i] = [1.0, 0.96 - 0.3*t, 0.88 - 0.5*t, 0.95]
            elif p < 0.8:
                # Moderate crack: bright orange-red
                t = (p - 0.5) / 0.3
                colors[i] = [1.0, 0.66 - 0.4*t, 0.38 - 0.2*t, 0.9]
            else:
                # Full crack: vivid red
                t = (p - 0.8) / 0.2
                colors[i] = [0.95 - 0.2*t, 0.2, 0.1, 0.85 - 0.2*t]
        
        return colors
    
    def save_fracture_animation(self, out_path, fps=4):
        """Save animated GIF with clear flow:
        1. Intro: quick 360° rotation of intact bone
        2. Crack growth: static view, cracks appearing
        3. Final: static view with zoom-in panels on crack zones
        """
        if not HAS_MPL or not HAS_PIL:
            print("  ⚠ Missing matplotlib/PIL")
            return
        
        v6 = self._v6
        s1_frames = self._v6._frames
        
        # Build bone surface
        bone_vol = np.zeros(v6.shape, dtype=float)
        bone_vol[v6._elem_ijk[:,0], v6._elem_ijk[:,1], v6._elem_ijk[:,2]] = 1.0
        verts, faces, normals = self._extract_surface(bone_vol, level=0.5)
        if verts is None:
            print("  ⚠ Marching cubes failed")
            return
        
        print(f"  Surface mesh: {len(verts)} verts, {len(faces)} tris")
        pil_frames = []
        base_azim, base_elev = -60, 25
        bg = '#1a1e2e'
        CANVAS_W, CANVAS_H = 1920, 960
        bg_rgb = (26, 30, 46)
        
        # ============================================================
        # PART 1: Rotate while loading — 360° with cracks growing
        # ============================================================
        print("  Part 1: Rotating + loading...")
        # One frame per load step, spread over 360°
        step_frames = {}
        for frame in s1_frames:
            step_frames[frame.get('step', 0)] = frame
        key_frames = [step_frames[s] for s in sorted(step_frames.keys())]
        n_kf = len(key_frames)
        
        ct_ds = v6.ct
        
        for fi, frame in enumerate(key_frames):
            phi = frame['phi']
            phi_elem = phi[v6._elem_nodes].mean(axis=1)
            phi_vol = np.zeros(v6.shape, dtype=np.float32)
            phi_vol[v6._elem_ijk[:,0], v6._elem_ijk[:,1], v6._elem_ijk[:,2]] = phi_elem
            
            # Rotate through 360° over the load steps
            azim = base_azim + (360 * fi / n_kf)
            
            fig = plt.figure(figsize=(10, 8), facecolor=bg)
            ax3d = fig.add_subplot(111, projection='3d', facecolor=bg)
            face_colors = self._phi_to_face_colors(phi_vol, verts, faces)
            self._render_3d_bone(ax3d, verts, faces, face_colors)
            self._set_3d_axes(ax3d, verts, elev=base_elev, azim=azim)
            
            load_pct = (frame.get('step', 0) + 1) * 100 // max(n_kf, 1)
            n_cracked = (phi_elem > 0.5).sum()
            ax3d.set_title(f'Load: {load_pct}% | φ_max={phi.max():.2f} | '
                          f'cracked: {n_cracked}',
                          color='white', fontsize=13, fontweight='bold', pad=20)
            img = self._fig_to_pil(fig)
            # Place on fixed canvas (centered)
            canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), bg_rgb)
            x_off = (CANVAS_W - img.width) // 2
            y_off = (CANVAS_H - img.height) // 2
            canvas.paste(img, (x_off, y_off))
            pil_frames.append(canvas)
        
        # ============================================================
        # PART 2: Static final — 3D view + zoom-in panels on cracks
        # ============================================================
        print("  Part 2: Final + zoom panels...")
        if self._active is not None and self._labeled is not None:
            labeled = self._labeled
            active = self._active
            n_frags = labeled.max()
            phi = self._phi
            
            phi_elem = phi[v6._elem_nodes].mean(axis=1)
            phi_vol = np.zeros(v6.shape, dtype=np.float32)
            phi_vol[v6._elem_ijk[:,0], v6._elem_ijk[:,1], v6._elem_ijk[:,2]] = phi_elem
            
            frag_colors = [
                [0.3, 0.7, 0.9], [0.95, 0.92, 0.8], [0.4, 0.9, 0.5],
                [0.9, 0.6, 0.3], [0.7, 0.4, 0.9], [0.9, 0.4, 0.5],
            ]
            
            # Find crack regions for zoom-ins
            eroded_ijk = v6._elem_ijk[~active]
            if len(eroded_ijk) > 0:
                crack_center = eroded_ijk.mean(axis=0)
            else:
                crack_center = np.array(v6.shape) / 2
            
            # Generate 5 frames of the final state (held longer)
            for hold in range(5):
                # --- Render 3D fragment view separately ---
                fig_3d = plt.figure(figsize=(8, 8), facecolor=bg)
                ax_main = fig_3d.add_subplot(111, projection='3d', facecolor=bg)
                
                all_v = []
                for fid in range(1, n_frags + 1):
                    frag_vol = (labeled == fid).astype(float)
                    fv, ff, fn = self._extract_surface(frag_vol, level=0.5)
                    if fv is None or len(fv) == 0:
                        continue
                    
                    centroid = fv.mean(axis=0)
                    big_c = verts.mean(axis=0)
                    d = centroid - big_c
                    dn = np.linalg.norm(d)
                    if dn > 0:
                        fv = fv + d / dn * 2
                    
                    color = frag_colors[(fid - 1) % len(frag_colors)]
                    fc = np.tile(color + [0.9], (len(ff), 1))
                    self._render_3d_bone(ax_main, fv, ff, fc, alpha=0.9)
                    all_v.append(fv)
                
                if len(eroded_ijk) > 0:
                    eroded_vol = np.zeros(v6.shape, dtype=float)
                    eroded_vol[eroded_ijk[:,0], eroded_ijk[:,1], eroded_ijk[:,2]] = 1.0
                    ev, ef, en_ = self._extract_surface(eroded_vol, level=0.5)
                    if ev is not None and len(ev) > 0:
                        crack_fc = np.tile([1.0, 0.2, 0.15, 0.2], (len(ef), 1))
                        self._render_3d_bone(ax_main, ev, ef, crack_fc, alpha=0.2)
                        all_v.append(ev)
                
                if all_v:
                    self._set_3d_axes(ax_main, np.vstack(all_v),
                                     elev=base_elev, azim=base_azim)
                
                n_e = (~active).sum()
                ax_main.set_title(f'{n_frags} Fragments | eroded={n_e}',
                                color='white', fontsize=14, fontweight='bold', pad=15)
                img_3d = self._fig_to_pil(fig_3d)
                
                # --- Render zoom panels separately ---
                ci = np.clip(crack_center.astype(int), 0,
                            np.array(v6.shape) - 1)
                ct_ds = v6.ct
                
                fig_zoom = plt.figure(figsize=(5, 8), facecolor=bg)
                for pi, (title, sl_idx, axis) in enumerate([
                    ('Sagittal @ crack', ci[0], 0),
                    ('Coronal @ crack',  ci[1], 1),
                    ('Axial @ crack',    ci[2], 2),
                ]):
                    ax_z = fig_zoom.add_subplot(3, 1, pi + 1, facecolor=bg)
                    ct_sl = np.take(ct_ds, sl_idx, axis=axis).T
                    ax_z.imshow(ct_sl, cmap='bone', vmin=-200, vmax=800,
                               origin='lower', aspect='auto')
                    
                    phi_sl = np.take(phi_vol, sl_idx, axis=axis).T
                    phi_rgba = plt.cm.hot(phi_sl)
                    phi_rgba[..., 3] = np.clip(phi_sl * 2.5, 0, 0.85)
                    ax_z.imshow(phi_rgba, origin='lower', aspect='auto')
                    
                    ax_z.set_title(title, color='white', fontsize=12,
                                  fontweight='bold')
                    ax_z.axis('off')
                
                fig_zoom.suptitle('Crack Zoom-In', color='white',
                                fontsize=13, fontweight='bold')
                plt.tight_layout()
                img_zoom = self._fig_to_pil(fig_zoom)
                
                # --- Compose onto fixed canvas ---
                canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), bg_rgb)
                left_w = CANVAS_W // 2
                right_w = CANVAS_W - left_w
                # 3D: scale to fill left half
                img_3d_s = img_3d.resize((left_w, CANVAS_H), Image.LANCZOS)
                canvas.paste(img_3d_s, (0, 0))
                # Zoom: scale to fit height, keep aspect ratio, center on right
                zoom_scale = CANVAS_H / img_zoom.height
                zoom_w = int(img_zoom.width * zoom_scale)
                img_zoom_s = img_zoom.resize((zoom_w, CANVAS_H), Image.LANCZOS)
                zoom_x = left_w + (right_w - zoom_w) // 2
                canvas.paste(img_zoom_s, (zoom_x, 0))
                pil_frames.append(canvas)
        
        # Save GIF — all frames are already CANVAS_W × CANVAS_H
        if pil_frames:
            durations = [300] * n_kf + [1500] * (len(pil_frames) - n_kf)
            pil_frames[0].save(out_path, save_all=True,
                              append_images=pil_frames[1:],
                              duration=durations, loop=0)
            print(f"  Saved: {out_path} ({len(pil_frames)} frames)")
    
    def _fig_to_pil(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, facecolor=fig.get_facecolor(),
                   pad_inches=0.1)
        buf.seek(0)
        img = Image.open(buf).copy()
        plt.close(fig)
        return img
    
    def save_final_state(self, out_path):
        """Save final fracture state as 3D PNG with two views."""
        if not HAS_MPL:
            return
        
        v6 = self._v6
        phi = self._phi
        active = self._active
        labeled = self._labeled
        r = self._result
        
        fig = plt.figure(figsize=(16, 8), facecolor='#0d1117')
        
        # Build volumes
        bone_vol = np.zeros(v6.shape, dtype=float)
        bone_vol[v6._elem_ijk[:,0], v6._elem_ijk[:,1], v6._elem_ijk[:,2]] = 1.0
        
        phi_elem = phi[v6._elem_nodes].mean(axis=1)
        phi_vol = np.zeros(v6.shape, dtype=np.float32)
        phi_vol[v6._elem_ijk[:,0], v6._elem_ijk[:,1], v6._elem_ijk[:,2]] = phi_elem
        
        verts, faces, normals = self._extract_surface(bone_vol, level=0.5)
        if verts is None:
            print("  ⚠ Marching cubes failed")
            return
        
        # Left: crack field on intact bone (two angles)
        for i, (azim, elev) in enumerate([(-60, 25), (120, 15)]):
            ax = fig.add_subplot(1, 2, i+1, projection='3d', facecolor='#0d1117')
            
            if i == 0:
                # Crack damage view
                face_colors = self._phi_to_face_colors(phi_vol, verts, faces)
                self._render_3d_bone(ax, verts, faces, face_colors)
                self._set_3d_axes(ax, verts, elev=elev, azim=azim)
                ax.set_title('Crack Damage (φ field)', color='white', 
                           fontsize=12, fontweight='bold', pad=20)
            else:
                # Fragment view
                n_frags = labeled.max() if labeled is not None else 0
                frag_colors = [
                    [0.3, 0.7, 0.9], [0.9, 0.85, 0.7], [0.4, 0.9, 0.5],
                    [0.9, 0.6, 0.3], [0.7, 0.4, 0.9], [0.9, 0.4, 0.5],
                ]
                all_verts = []
                for fid in range(1, n_frags + 1):
                    frag_vol = (labeled == fid).astype(float)
                    fv, ff, fn = self._extract_surface(frag_vol, level=0.5)
                    if fv is None or len(fv) == 0:
                        continue
                    # Slight separation for visibility
                    centroid = fv.mean(axis=0)
                    big_centroid = verts.mean(axis=0)
                    direction = centroid - big_centroid
                    d_norm = np.linalg.norm(direction)
                    if d_norm > 0:
                        fv = fv + direction / d_norm * 2  # 2 voxel separation
                    color = frag_colors[(fid - 1) % len(frag_colors)]
                    fc = np.tile(color + [0.85], (len(ff), 1))
                    self._render_3d_bone(ax, fv, ff, fc, alpha=0.85)
                    all_verts.append(fv)
                
                # Eroded as ghost
                eroded_ijk = v6._elem_ijk[~active]
                if len(eroded_ijk) > 0:
                    eroded_vol = np.zeros(v6.shape, dtype=float)
                    eroded_vol[eroded_ijk[:,0], eroded_ijk[:,1], eroded_ijk[:,2]] = 1.0
                    ev, ef, en_ = self._extract_surface(eroded_vol, level=0.5)
                    if ev is not None and len(ev) > 0:
                        crack_fc = np.tile([0.9, 0.15, 0.1, 0.25], (len(ef), 1))
                        self._render_3d_bone(ax, ev, ef, crack_fc, alpha=0.25)
                        all_verts.append(ev)
                
                if all_verts:
                    self._set_3d_axes(ax, np.vstack(all_verts), elev=elev, azim=azim)
                
                n_e = (~active).sum()
                ax.set_title(f'{n_frags} Fragments | eroded={n_e}', color='white',
                           fontsize=12, fontweight='bold', pad=20)
        
        fig.suptitle(
            f'v8 Hybrid Fracture — {r.ao_type if r else "?"} | '
            f'{r.n_fragments if r else "?"} fragments',
            fontsize=15, color='white', fontweight='bold', y=0.98)
        fig.savefig(out_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ============================================================================
#  DEMO
# ============================================================================

def demo(use_cuda=False):
    from _gen_real_fracture_visuals import load_vertebra
    
    print("=" * 70)
    print("WiseSpine v8 — Hybrid Fracture Engine")
    print("  Stage 1: Implicit Phase Field (v6) → accurate cracks")
    print("  Stage 2: Explicit Fragment Dynamics → separation animation")
    print("=" * 70)
    
    verse_root = os.path.join(os.path.dirname(__file__), '..', '..',
                              'VerSe', 'dataset-01training')
    ct_path = os.path.join(verse_root, 'rawdata', 'sub-verse503',
                           'sub-verse503_dir-ax_ct.nii.gz')
    mask_path = os.path.join(verse_root, 'derivatives', 'sub-verse503',
                             'sub-verse503_dir-ax_seg-vert_msk.nii.gz')
    
    ct, mask, label, spacing = load_vertebra(ct_path, mask_path, return_spacing=True)
    mask = (mask > 0).astype(np.int32)
    voxel_size = float(spacing.mean())
    
    # Phase field needs implicit solve — target ~15k elements
    # 68k elem at ds=3 took 800s/solve. At ds=5 → ~10k → ~50s/solve
    ds = max(5, int(np.cbrt(mask.sum() / 15000)))
    print(f"  Downsample: ds={ds} (target ~15k elements for implicit)")
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'fracture_v8_demo')
    os.makedirs(out_dir, exist_ok=True)
    
    # Log file
    log_path = os.path.join(out_dir, 'v8_log.txt')
    import sys as _sys
    class Tee:
        def __init__(self, *fps):
            self.fps = fps
        def write(self, s):
            for f in self.fps: f.write(s); f.flush()
        def flush(self):
            for f in self.fps: f.flush()
    log_fp = open(log_path, 'w')
    _sys.stdout = Tee(_sys.stdout, log_fp)
    
    scenarios = [
        ('burst', CausalParameters(4.0, 5.0, 0.0, 0.5)),
    ]
    
    for name, params in scenarios:
        print(f"\n{'='*60}")
        print(f"  Scenario: {name}")
        print(f"{'='*60}")
        
        engine = HybridFractureEngine(
            mask, ct, voxel_size_mm=voxel_size,
            downsample=ds, seed=42, use_cuda=use_cuda
        )
        engine.set_causal_params(params)
        
        result = engine.simulate(
            n_load_steps=20,      # 5% increments — gradual + displacement cap
            max_stagger=5,        # enough for convergence
            n_frag_steps=100,
        )
        
        # Save outputs
        gif_path = os.path.join(out_dir, f'v8_{name}_fracture.gif')
        engine.save_fracture_animation(gif_path)
        
        png_path = os.path.join(out_dir, f'v8_{name}_final.png')
        engine.save_final_state(png_path)
    
    print(f"\nOutputs saved to {out_dir}/")
    print(f"Log: {log_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    demo(use_cuda=args.cuda)
