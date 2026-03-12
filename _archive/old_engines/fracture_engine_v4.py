#!/usr/bin/env python3
"""
Voxel-Based Fracture Engine v4
==============================

Direct voxel-grid fracture simulation — no particle sampling, no coordinate
transforms, no K calibration.  Operates on the CT voxel grid natively.

Key difference from particle-based v3:
  v3: mask → particles → [P2G] → grid → [G2P] → particles → [inverse map] → voxels
  v4: mask → voxels → [diffuse on same grid] → voxels → done

Physics components:
  1. Per-voxel material strength from HU (Morgan & Keaveny 2003)
  2. Endplate stress initialization with directional bias
  3. 3D stress diffusion with cortical barrier
  4. Per-voxel damage evolution
  5. Emergent AO classifier from damage patterns

Author: Wisespine Team
Date: 2026-03-06
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt, zoom
from scipy.ndimage import map_coordinates
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import sys
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================================
#  DATA CLASSES (shared with v3)
# ============================================================================

@dataclass
class CausalParameters:
    """Physical parameters that causally determine fracture outcome."""
    force_magnitude: float = 5.0      # 1-10, ~kN scale
    flexion_angle: float = 15.0       # degrees, 0=axial, +30=flexion
    lateral_angle: float = 0.0        # degrees
    bmd_factor: float = 1.0           # 0.3-1.5, 1.0=normal
    cortical_thickness: float = 1.0   # 0.5-1.5

    def validate(self):
        self.force_magnitude = np.clip(self.force_magnitude, 0.5, 15.0)
        self.flexion_angle = np.clip(self.flexion_angle, -10.0, 45.0)
        self.lateral_angle = np.clip(self.lateral_angle, -20.0, 20.0)
        self.bmd_factor = np.clip(self.bmd_factor, 0.3, 1.5)
        self.cortical_thickness = np.clip(self.cortical_thickness, 0.3, 2.0)

    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in
                ['force_magnitude', 'flexion_angle', 'lateral_angle',
                 'bmd_factor', 'cortical_thickness']}


@dataclass
class AOResult:
    """Result of emergent AO classification."""
    ao_type: str
    confidence: float
    posterior_wall_damage: float
    anterior_height_loss: float
    posterior_height_loss: float
    canal_compromise: float
    n_fragments: int
    max_damage: float
    damaged_fraction: float
    fractured_fraction: float

    def summary(self) -> str:
        return (
            f"AO Type: {self.ao_type} (conf={self.confidence:.2f})\n"
            f"  Ant. height loss: {self.anterior_height_loss*100:.1f}%\n"
            f"  Post. height loss: {self.posterior_height_loss*100:.1f}%\n"
            f"  Post. wall damage: {self.posterior_wall_damage*100:.1f}%\n"
            f"  Canal compromise: {self.canal_compromise*100:.1f}%\n"
            f"  Fragments: {self.n_fragments}\n"
            f"  Damaged: {self.damaged_fraction*100:.1f}%\n"
            f"  Fractured: {self.fractured_fraction*100:.1f}%"
        )


# ============================================================================
#  VOXEL FRACTURE ENGINE
# ============================================================================

class VoxelFractureEngine:
    """Voxel-based vertebral fracture simulator.

    Operates directly on the CT voxel grid:
      1. HU → per-voxel material strength
      2. Endplate loading → stress field
      3. 3D diffusion on bone mask → stress distribution
      4. stress > strength → damage accumulation
      5. Damage pattern → emergent AO type
    """

    def __init__(self, mask: np.ndarray, ct: np.ndarray = None,
                 downsample: int = 1, seed: int = 42):
        """
        Args:
            mask: 3D binary mask of vertebra (int, bone > 0)
            ct: 3D CT volume (float, HU values). Same shape as mask.
            downsample: Downsample factor for simulation (1=full, 2=half, etc.)
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.ds = max(int(downsample), 1)

        # Store originals for visualization
        self._orig_mask = mask
        self._orig_ct = ct

        # Downsample for simulation
        if self.ds > 1:
            self.mask = zoom(mask.astype(np.float32),
                             1.0 / self.ds, order=0).astype(np.int32)
            self.mask = (self.mask > 0).astype(np.int32)
            if ct is not None:
                self.ct = zoom(ct.astype(np.float32),
                               1.0 / self.ds, order=1)
            else:
                self.ct = None
        else:
            self.mask = mask.copy()
            self.ct = ct.copy() if ct is not None else None

        self.shape = self.mask.shape
        self.bone_mask = self.mask > 0
        self.n_bone_voxels = int(self.bone_mask.sum())

        # Precompute geometry
        self._compute_geometry()

        # State
        self.params = None
        self._damage = None
        self._stress = None
        self._result = None

    # ================================================================
    #  GEOMETRY
    # ================================================================

    def _compute_geometry(self):
        """Precompute geometric features of the vertebra."""
        coords = np.argwhere(self.bone_mask)
        self._bone_min = coords.min(axis=0).astype(np.float32)
        self._bone_max = coords.max(axis=0).astype(np.float32)
        self._bone_extent = self._bone_max - self._bone_min
        self._bone_range = np.maximum(self._bone_extent, 1.0)

        # Distance transform for cortical depth
        self._edt = distance_transform_edt(self.bone_mask).astype(np.float32)
        self._max_edt = max(self._edt.max(), 1.0)

        # Relative positions within bone bbox [0, 1]
        # axis 0 = left-right (or axial slices)
        # axis 1 = anterior-posterior
        # axis 2 = superior-inferior
        ii, jj, kk = np.mgrid[:self.shape[0], :self.shape[1], :self.shape[2]]
        self._ap_ratio = np.zeros(self.shape, dtype=np.float32)
        self._si_ratio = np.zeros(self.shape, dtype=np.float32)
        self._ap_ratio[self.bone_mask] = (
            (jj[self.bone_mask] - self._bone_min[1]) / self._bone_range[1]
        )
        self._si_ratio[self.bone_mask] = (
            (kk[self.bone_mask] - self._bone_min[2]) / self._bone_range[2]
        )

        # Cortical depth ratio: 0=surface, 1=deepest
        self._depth_ratio = np.zeros(self.shape, dtype=np.float32)
        self._depth_ratio[self.bone_mask] = (
            self._edt[self.bone_mask] / self._max_edt
        )

    # ================================================================
    #  COMPONENT 1: Material Properties
    # ================================================================

    def _compute_strength(self, params: CausalParameters) -> np.ndarray:
        """Per-voxel fracture strength from CT HU values.

        Uses Morgan & Keaveny (2003) power-law relationship.
        Returns strength array same shape as mask, in [0, ~1] range.
        """
        strength = np.zeros(self.shape, dtype=np.float32)

        if self.ct is not None:
            # Normalize HU to [0, 1] relative to bone range
            bone_hu = self.ct[self.bone_mask]
            hu_min, hu_max = bone_hu.min(), max(bone_hu.max(), 1)
            # Power-law: strength ∝ (HU_normalized)^1.83
            hu_norm = np.clip((self.ct - hu_min) / (hu_max - hu_min), 0.01, 1.0)
            strength[self.bone_mask] = hu_norm[self.bone_mask] ** 1.83
        else:
            # No CT: uniform strength
            strength[self.bone_mask] = 0.5

        # BMD factor
        strength *= params.bmd_factor

        # Cortical shell: stronger
        cortical = self._depth_ratio < (0.20 * params.cortical_thickness)
        strength[cortical & self.bone_mask] *= 1.5 * params.cortical_thickness

        # Endplate: weaker (thinner cortex)
        endplate = (self._si_ratio > 0.85) | (self._si_ratio < 0.15)
        strength[endplate & self.bone_mask] *= 0.8

        return strength

    # ================================================================
    #  COMPONENT 2: Endplate Stress Initialization
    # ================================================================

    def _initialize_stress(self, params: CausalParameters) -> np.ndarray:
        """Create initial stress field from bilateral endplate loading.

        Both superior and inferior endplates receive loading (real axial
        compression). Flexion angle biases stress anteriorly.
        """
        stress = np.zeros(self.shape, dtype=np.float32)
        flex_rad = np.radians(params.flexion_angle)
        lat_rad = np.radians(params.lateral_angle)

        # Pre-compute left-right ratios
        lr_full = np.zeros(self.shape, dtype=np.float32)
        lr_full[self.bone_mask] = (
            (np.mgrid[:self.shape[0], :self.shape[1], :self.shape[2]][0]
             [self.bone_mask].astype(np.float32) - self._bone_min[0])
            / self._bone_range[0]
        )

        def _apply_endplate_stress(endplate_mask, weight_factor=1.0):
            if not endplate_mask.any():
                return
            ap = self._ap_ratio[endplate_mask]
            lr = lr_full[endplate_mask]
            direction_weight = 1.0 + np.sin(flex_rad) * (ap - 0.5) * 2.0
            direction_weight += np.sin(lat_rad) * (lr - 0.5) * 1.0
            direction_weight = np.clip(direction_weight, 0.1, 3.0)
            n_ep = int(endplate_mask.sum())
            stress_per_voxel = params.force_magnitude / max(n_ep, 1)
            stress[endplate_mask] += (
                stress_per_voxel * direction_weight * n_ep * weight_factor
            )

        # Superior endplate (top 15%)
        _apply_endplate_stress(
            (self._si_ratio > 0.85) & self.bone_mask, weight_factor=1.0
        )
        # Inferior endplate (bottom 15%) — axial compression is bilateral
        _apply_endplate_stress(
            (self._si_ratio < 0.15) & self.bone_mask, weight_factor=0.7
        )

        # Central body also gets some axial stress at high force
        # (simulates through-body compression)
        if params.force_magnitude > 4.0:
            mid_body = (
                (self._si_ratio > 0.35) & (self._si_ratio < 0.65) &
                (self._depth_ratio < 0.25) &  # near surface / trabecular
                self.bone_mask
            )
            if mid_body.any():
                body_stress = params.force_magnitude * 0.15
                stress[mid_body] += body_stress

        return stress

    # ================================================================
    #  COMPONENT 3: Stress Diffusion
    # ================================================================

    def _diffuse_stress(self, stress: np.ndarray,
                        cortical_strength: np.ndarray,
                        n_iterations: int = 80) -> np.ndarray:
        """Diffuse stress through bone voxels with cortical barrier.

        Uses iterative 3D convolution (Jacobi-like relaxation) on the
        bone mask. Cortical voxels attenuate stress propagation.
        """
        # 3x3x3 averaging kernel (face + edge + corner neighbors)
        kernel = np.ones((3, 3, 3), dtype=np.float32)
        kernel[1, 1, 1] = 0
        kernel /= kernel.sum()

        source = stress.copy()  # preserve for re-injection
        alpha = 0.5  # diffusion rate

        # Cortical barrier strength (0=no barrier, 1=full block)
        barrier = np.zeros(self.shape, dtype=np.float32)
        cortical = self._depth_ratio < 0.15
        barrier[cortical & self.bone_mask] = 0.15 * cortical_strength[
            cortical & self.bone_mask
        ]

        for it in range(n_iterations):
            neighbor_avg = ndi.convolve(stress, kernel,
                                        mode='constant', cval=0.0)

            # Update only bone voxels
            update = (
                (1 - alpha) * stress[self.bone_mask] +
                alpha * neighbor_avg[self.bone_mask] +
                source[self.bone_mask] * 0.08  # source re-injection
            )

            stress[self.bone_mask] = update

            # Cortical barrier: attenuate every 5th iteration
            if it % 5 == 0:
                stress[self.bone_mask] *= (1.0 - barrier[self.bone_mask])

            # Zero outside bone
            stress[~self.bone_mask] = 0.0

        return stress

    # ================================================================
    #  COMPONENT 4: Damage Evolution
    # ================================================================

    def _evolve_damage(self, stress: np.ndarray, strength: np.ndarray,
                       damage: np.ndarray, rate: float) -> np.ndarray:
        """Accumulate damage where stress exceeds strength.

        Per-voxel: damage += rate × (stress - strength) / strength_max
        where stress > strength.
        """
        overstressed = (stress > strength) & self.bone_mask & (strength > 0)

        if overstressed.any():
            overstress = (stress[overstressed] - strength[overstressed])
            strength_max = strength[self.bone_mask].max()
            increment = rate * overstress / max(strength_max, 0.001)
            damage[overstressed] = np.minimum(
                damage[overstressed] + increment, 1.0
            )

        return damage

    # ================================================================
    #  COMPONENT 5: AO Classification
    # ================================================================

    def _classify_ao(self, damage: np.ndarray,
                     deformation: np.ndarray) -> AOResult:
        """Classify AO type from emergent damage patterns."""
        if not self.bone_mask.any():
            return AOResult('A0', 1.0, 0, 0, 0, 0, 0, 0, 0, 0)

        bone_damage = damage[self.bone_mask]
        bone_deform = deformation[self.bone_mask]
        bone_ap = self._ap_ratio[self.bone_mask]
        bone_si = self._si_ratio[self.bone_mask]

        # Posterior wall damage
        posterior = bone_ap < 0.30
        post_wall_dmg = bone_damage[posterior].mean() if posterior.any() else 0.0

        # Height loss: anterior vs posterior
        anterior_zone = bone_ap > 0.65
        ant_h = np.abs(bone_deform[anterior_zone]).mean() if anterior_zone.any() else 0.0
        ant_h /= max(self._bone_extent[2], 1)

        post_body = (bone_ap > 0.20) & (bone_ap < 0.45)
        post_h = np.abs(bone_deform[post_body]).mean() if post_body.any() else 0.0
        post_h /= max(self._bone_extent[2], 1)

        # Canal compromise
        canal = post_wall_dmg * 1.2
        if post_wall_dmg > 0.4:
            canal += 0.15
        canal = min(canal, 1.0)

        # Fragment count
        frag_mask = damage > 0.7
        if frag_mask.any():
            labeled, n_labels = ndi.label(frag_mask & self.bone_mask)
            # Filter small fragments (< 1% of bone)
            min_size = max(self.n_bone_voxels * 0.01, 5)
            n_fragments = 0
            for lbl in range(1, n_labels + 1):
                if (labeled == lbl).sum() >= min_size:
                    n_fragments += 1
        else:
            n_fragments = 0

        # Overall metrics
        max_damage = bone_damage.max()
        damaged_frac = (bone_damage > 0.5).mean()
        fractured_frac = (bone_damage > 0.8).mean()

        # --- Classification ---
        if max_damage < 0.3 or damaged_frac < 0.05:
            ao_type, confidence = 'A0', 0.9
        elif post_wall_dmg < 0.20:
            if ant_h > 0.15 and ant_h > post_h * 1.3:
                ao_type, confidence = 'A1', min(ant_h / 0.15, 2.0) / 2.0
            elif ant_h > 0.10 or post_h > 0.08:
                ao_type, confidence = 'A2', 0.7
            else:
                ao_type, confidence = 'A1', 0.5
        elif post_wall_dmg < 0.30:
            ao_type, confidence = 'A2', 0.65
        elif canal < 0.25 or post_wall_dmg < 0.40:
            ao_type, confidence = 'A3', 0.8
        else:
            ao_type, confidence = 'A4', min(canal / 0.5, 1.0)

        return AOResult(
            ao_type=ao_type, confidence=confidence,
            posterior_wall_damage=post_wall_dmg,
            anterior_height_loss=ant_h, posterior_height_loss=post_h,
            canal_compromise=canal, n_fragments=n_fragments,
            max_damage=max_damage, damaged_fraction=damaged_frac,
            fractured_fraction=fractured_frac,
        )

    # ================================================================
    #  COMPRESSION DEFORMATION (Height Loss)
    # ================================================================

    def _compute_height_loss(self, damage: np.ndarray,
                             params: CausalParameters) -> np.ndarray:
        """Compute height loss field from damage pattern."""
        deformation = np.zeros(self.shape, dtype=np.float32)
        if damage.max() < 0.01:
            return deformation

        flex_rad = np.radians(params.flexion_angle)

        # Per-voxel height loss: proportional to damage × direction
        # More damage near superior → more compression
        si = self._si_ratio
        ap = self._ap_ratio

        # Flexion bias: anterior compresses more
        direction_bias = 1.0 + np.sin(flex_rad) * (ap - 0.5) * 2.0
        direction_bias = np.clip(direction_bias, 0.3, 2.0)

        # Height loss proportional to damage and position
        max_loss = self._bone_extent[2] * 0.3  # max 30% height loss
        deformation[self.bone_mask] = (
            -damage[self.bone_mask] *
            si[self.bone_mask] *
            direction_bias[self.bone_mask] *
            max_loss *
            params.force_magnitude / 10.0
        )

        # Smooth
        if deformation[self.bone_mask].any():
            deformation = ndi.gaussian_filter(deformation, sigma=1.0)
            deformation *= self.bone_mask

        return deformation

    # ================================================================
    #  MAIN SIMULATION
    # ================================================================

    def set_causal_params(self, params: CausalParameters):
        """Set causal parameters for next simulation."""
        params.validate()
        self.params = params

    def simulate(self, n_steps: int = 150, verbose: bool = True) -> AOResult:
        """Run the voxel-based fracture simulation.

        Args:
            n_steps: Number of simulation steps (force ramp).
            verbose: Print progress.

        Returns:
            AOResult with emergent AO classification.
        """
        params = self.params
        if params is None:
            raise ValueError("Call set_causal_params() first.")

        # Component 1: Material strength
        strength = self._compute_strength(params)

        # Component 2: Initial stress field
        stress_init = self._initialize_stress(params)

        # Component 3: Cortical strength for barrier
        cortical_s = np.ones(self.shape, dtype=np.float32)
        cortical_s[self.bone_mask] = self._depth_ratio[self.bone_mask] < 0.15

        # Initialize damage
        damage = np.zeros(self.shape, dtype=np.float32)

        # Damage rate scales with force
        base_rate = 0.08
        damage_rate = base_rate * (params.force_magnitude / 5.0)

        # Force ramp
        force_ramp = np.linspace(0, 1.0, n_steps)

        for step in range(n_steps):
            # Scale stress by ramp
            stress = stress_init * force_ramp[step]

            # Diffuse stress through bone
            stress = self._diffuse_stress(
                stress, cortical_s,
                n_iterations=60
            )

            # Stiffness degradation: damaged regions carry less stress
            stress *= (1.0 - damage) ** 2

            # Evolve damage
            damage = self._evolve_damage(stress, strength, damage, damage_rate)

            # Capture frames for animation
            if hasattr(self, '_capture_frames') and self._capture_frames:
                if step % self._frame_interval == 0 or step == n_steps - 1:
                    damaged_frac = (damage[self.bone_mask] > 0.5).mean()
                    self._frames.append({
                        'step': step,
                        'ramp': force_ramp[step],
                        'damage': damage.copy(),
                        'max_damage': damage.max(),
                        'damaged_frac': damaged_frac,
                    })

            if verbose and step % 50 == 0:
                damaged_frac = (damage[self.bone_mask] > 0.5).mean()
                bone_stress = stress[self.bone_mask]
                bone_strength = strength[self.bone_mask]
                n_over = ((bone_stress > bone_strength) & (bone_strength > 0)).sum()
                print(
                    f"  Step {step:4d}/{n_steps} | "
                    f"Ramp={force_ramp[step]:.2f} | "
                    f"MaxD={damage.max():.3f} | "
                    f"Damaged={damaged_frac*100:.1f}% | "
                    f"Overstressed={n_over}/{self.n_bone_voxels}"
                )

        # Component: Height loss
        deformation = self._compute_height_loss(damage, params)

        # Component 5: Classify
        result = self._classify_ao(damage, deformation)

        # Store state
        self._damage = damage
        self._deformation = deformation
        self._stress_final = stress
        self._strength = strength
        self._result = result

        if verbose:
            print(f"\n  ★ Emergent AO Classification: {result.ao_type}")
            print(result.summary())

        return result

    # ================================================================
    #  SHAPE MATCHING (zoom off-by-one fix)
    # ================================================================

    @staticmethod
    def _match_shape(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Crop/pad array to exactly match target shape."""
        result = arr
        # Crop if too large
        slices = tuple(slice(0, min(s, t)) for s, t in
                       zip(result.shape, target_shape))
        result = result[slices]
        # Pad if too small
        pad_width = [(0, max(0, t - s)) for s, t in
                     zip(result.shape, target_shape)]
        if any(p[1] > 0 for p in pad_width):
            result = np.pad(result, pad_width, mode='constant',
                            constant_values=0)
        return result

    # ================================================================
    #  FRACTURED CT
    # ================================================================

    def get_fractured_ct(self) -> np.ndarray:
        """Return CT volume with damage applied."""
        ct = self._orig_ct if self._orig_ct is not None else self.ct
        if ct is None:
            raise ValueError("No CT volume.")
        if self._damage is None:
            raise ValueError("Run simulate() first.")

        # Upsample damage if downsampled
        if self.ds > 1:
            damage_full = zoom(self._damage, self.ds, order=1)
            deform_full = zoom(self._deformation, self.ds, order=1)
            # Match exact original shape (zoom can be off by ±1)
            orig_shape = self._orig_mask.shape
            damage_full = self._match_shape(damage_full, orig_shape)
            deform_full = self._match_shape(deform_full, orig_shape)
        else:
            damage_full = self._damage
            deform_full = self._deformation

        fractured = ct.copy()
        mask = self._orig_mask > 0

        # Apply damage: reduce HU in damaged regions
        fractured[mask] -= damage_full[mask] * 200  # HU reduction
        # Add noise to fracture lines
        noise = self._rng.normal(0, 30, size=fractured.shape).astype(np.float32)
        fractured[mask] += damage_full[mask] * noise[mask]

        # Apply deformation
        if deform_full.max() > 0.01 or deform_full.min() < -0.01:
            coords = np.mgrid[:ct.shape[0], :ct.shape[1], :ct.shape[2]].astype(np.float32)
            coords[2] += deform_full
            coords[2] = np.clip(coords[2], 0, ct.shape[2] - 1)
            fractured = map_coordinates(fractured, coords, order=1, mode='nearest')
            fractured = fractured.astype(np.float32)

        return fractured

    # ================================================================
    #  VISUALIZATION
    # ================================================================

    def plot_fracture_panels(self, result: AOResult,
                             output_path: str = None):
        """4-panel sagittal dashboard."""
        if not HAS_MPL:
            print("  [skip] matplotlib unavailable")
            return

        ct = self._orig_ct if self._orig_ct is not None else self.ct
        if ct is None or self._damage is None:
            print("  [skip] No CT or no simulation results")
            return

        params = self.params

        # Use original resolution for display
        if self.ds > 1:
            damage_disp = zoom(self._damage, self.ds, order=1)
            orig_shape = self._orig_mask.shape
            damage_disp = self._match_shape(damage_disp, orig_shape)
        else:
            damage_disp = self._damage

        mid_x = ct.shape[0] // 2
        ct_slice = ct[mid_x]
        dmg_slice = damage_disp[mid_x]

        fractured_ct = self.get_fractured_ct()
        frac_slice = fractured_ct[mid_x]

        fig = plt.figure(figsize=(18, 5.5), facecolor='#1a1a2e')
        gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.7], wspace=0.15)

        hu_kw = dict(cmap='gray', origin='lower', vmin=-100, vmax=800,
                     aspect='auto')

        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(ct_slice.T, **hu_kw)
        ax1.set_title('Original CT', fontsize=12, color='white', pad=8)
        ax1.axis('off')
        ax1.set_facecolor('#1a1a2e')

        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(ct_slice.T, **hu_kw, alpha=0.5)
        dmg_im = ax2.imshow(dmg_slice.T, cmap='hot', origin='lower',
                             alpha=0.65, vmin=0, vmax=1, aspect='auto')
        ax2.set_title('Damage Field', fontsize=12, color='white', pad=8)
        ax2.axis('off')
        ax2.set_facecolor('#1a1a2e')
        cbar = plt.colorbar(dmg_im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Damage', fontsize=9, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

        ax3 = fig.add_subplot(gs[2])
        ax3.imshow(frac_slice.T, **hu_kw)
        ax3.set_title('Fractured CT', fontsize=12, color='white', pad=8)
        ax3.axis('off')
        ax3.set_facecolor('#1a1a2e')

        ax4 = fig.add_subplot(gs[3])
        ax4.axis('off')
        ax4.set_facecolor('#1a1a2e')

        ao_colors = {'A0': '#2196F3', 'A1': '#4CAF50', 'A2': '#FFC107',
                     'A3': '#FF5722', 'A4': '#9C27B0'}
        ao_color = ao_colors.get(result.ao_type, '#888')

        metrics = (
            f"Emergent AO: {result.ao_type}\n"
            f"Confidence: {result.confidence:.0%}\n"
            f"{'─' * 22}\n"
            f"Force:    {params.force_magnitude:.1f} kN\n"
            f"BMD:      {params.bmd_factor:.1f}\n"
            f"Flexion:  {params.flexion_angle:.0f}°\n"
            f"{'─' * 22}\n"
            f"Ant. ΔH:  {result.anterior_height_loss*100:.1f}%\n"
            f"Post. ΔH: {result.posterior_height_loss*100:.1f}%\n"
            f"Post.Wall: {result.posterior_wall_damage*100:.1f}%\n"
            f"Canal:    {result.canal_compromise*100:.1f}%\n"
            f"{'─' * 22}\n"
            f"Damaged:  {result.damaged_fraction*100:.1f}%\n"
            f"Fractured: {result.fractured_fraction*100:.1f}%\n"
            f"Fragments: {result.n_fragments}"
        )
        ax4.text(0.05, 0.95, metrics, transform=ax4.transAxes,
                 fontsize=10, color='white', verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=ao_color,
                           alpha=0.3, edgecolor=ao_color))

        fig.suptitle(
            f"AO {result.ao_type}  •  Force {params.force_magnitude:.1f} kN  •  "
            f"BMD {params.bmd_factor:.1f}  •  Flexion {params.flexion_angle:.0f}°",
            fontsize=14, color='white', y=0.98, fontweight='bold')

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor(), edgecolor='none')
            print(f"  Saved: {output_path}")
        else:
            plt.show()
        plt.close(fig)

    # ================================================================
    #  ANIMATED GIF
    # ================================================================

    def simulate_animated(self, n_steps: int = 150, n_frames: int = 30,
                          verbose: bool = True) -> AOResult:
        """Run simulation and capture frames for GIF animation.

        Args:
            n_steps: Total simulation steps.
            n_frames: Number of frames to capture.
            verbose: Print progress.
        """
        self._capture_frames = True
        self._frame_interval = max(1, n_steps // n_frames)
        self._frames = []
        result = self.simulate(n_steps=n_steps, verbose=verbose)
        self._capture_frames = False
        return result

    def save_animation_gif(self, output_path: str, fps: int = 8):
        """Save captured simulation frames as animated GIF.

        Must call simulate_animated() first.
        """
        if not self._frames:
            print("  [skip] No frames captured. Run simulate_animated() first.")
            return
        if not HAS_MPL:
            print("  [skip] matplotlib unavailable")
            return

        try:
            import imageio.v2 as imageio
        except ImportError:
            try:
                import imageio
            except ImportError:
                print("  [skip] imageio not installed")
                return

        ct = self._orig_ct
        if ct is None:
            print("  [skip] No CT volume")
            return

        params = self.params
        result = self._result
        ao_color = _AO_COLORS.get(result.ao_type, '#888')

        mid_x = ct.shape[0] // 2
        ct_slice = ct[mid_x]

        # Render each frame
        rendered_frames = []
        print(f"  Rendering {len(self._frames)} frames...")

        for fi, frame in enumerate(self._frames):
            # Upsample damage to original resolution
            if self.ds > 1:
                dmg = self._match_shape(
                    zoom(frame['damage'], self.ds, order=1),
                    self._orig_mask.shape
                )
            else:
                dmg = frame['damage']

            dmg_slice = dmg[mid_x]

            fig, ax = plt.subplots(figsize=(7, 6), facecolor='#1a1a2e')
            ax.set_facecolor('#1a1a2e')

            # CT background
            ax.imshow(ct_slice.T, cmap='gray', origin='lower',
                      vmin=-100, vmax=800, aspect='auto')
            # Damage overlay
            ax.imshow(dmg_slice.T, cmap='hot', origin='lower',
                      alpha=0.6, vmin=0, vmax=1, aspect='auto')
            ax.axis('off')

            # Title with metrics
            step_pct = frame['ramp'] * 100
            ax.set_title(
                f"Force Ramp: {step_pct:.0f}%  •  "
                f"Damage: {frame['damaged_frac']*100:.1f}%  •  "
                f"Max: {frame['max_damage']:.2f}",
                fontsize=12, color='white', pad=10
            )

            # Force ramp progress bar at bottom
            bar_y = 0.02
            bar_h = 0.025
            ax_bar = fig.add_axes([0.1, bar_y, 0.8, bar_h])
            ax_bar.set_xlim(0, 1)
            ax_bar.set_ylim(0, 1)
            ax_bar.barh(0.5, frame['ramp'], height=0.8,
                        color='#e94560', alpha=0.9)
            ax_bar.barh(0.5, 1.0, height=0.8, color='#333',
                        alpha=0.3, zorder=0)
            ax_bar.set_xlabel(
                f'F={params.force_magnitude:.0f}kN  BMD={params.bmd_factor:.1f}',
                fontsize=9, color='white', labelpad=2)
            ax_bar.tick_params(left=False, labelleft=False, bottom=False,
                               labelbottom=False)
            ax_bar.set_facecolor('#1a1a2e')
            for spine in ax_bar.spines.values():
                spine.set_color('#444')

            # Render to numpy array
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            rendered_frames.append(img)
            plt.close(fig)

        # Assemble GIF
        # Hold last frame longer
        rendered_frames.extend([rendered_frames[-1]] * 5)

        imageio.mimsave(output_path, rendered_frames, fps=fps, loop=0)
        print(f"  Saved GIF: {output_path} ({len(self._frames)} frames, {fps} fps)")

    # ================================================================
    #  COUNTERFACTUAL / SWEEP
    # ================================================================

    def counterfactual(self, **param_changes) -> AOResult:
        """Run simulation with modified parameters."""
        if self.params is None:
            raise ValueError("Run simulate() first.")
        cf = CausalParameters(**self.params.to_dict())
        for k, v in param_changes.items():
            if hasattr(cf, k):
                setattr(cf, k, v)
            else:
                raise ValueError(f"Unknown parameter: {k}")
        self.set_causal_params(cf)
        return self.simulate(verbose=False)

    def parameter_sweep(self, param: str, values: list,
                        verbose: bool = True) -> List[Tuple[float, AOResult]]:
        """Sweep a single parameter."""
        base = self.params.to_dict() if self.params else {}
        results = []
        for val in values:
            p = CausalParameters(**base)
            setattr(p, param, val)
            self.set_causal_params(p)
            r = self.simulate(verbose=False)
            results.append((val, r))
            if verbose:
                print(f"  {param}={val:6.2f} → {r.ao_type} "
                      f"(conf={r.confidence:.2f}, dmg={r.damaged_fraction*100:.1f}%)")
        return results


# ============================================================================
#  VISUALIZATION HELPERS
# ============================================================================

_AO_COLORS = {'A0': '#2196F3', 'A1': '#4CAF50', 'A2': '#FFC107',
               'A3': '#FF5722', 'A4': '#9C27B0'}


def _plot_force_sweep(sweep_results, output_dir, filename='v4_force_sweep.png',
                      title_suffix=''):
    if not HAS_MPL:
        return
    forces = [r[0] for r in sweep_results]
    damages = [r[1].damaged_fraction * 100 for r in sweep_results]
    ao_types = [r[1].ao_type for r in sweep_results]
    colors = [_AO_COLORS.get(t, '#888') for t in ao_types]

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')

    bars = ax.bar(forces, damages, color=colors, edgecolor='white',
                  linewidth=0.6, width=0.65, alpha=0.85, zorder=3)
    ax.plot(forces, damages, color='#e94560', marker='o', markersize=5,
            linewidth=1.5, zorder=4, label='Damage %')

    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=_AO_COLORS[k], label=k)
               for k in ['A0', 'A1', 'A2', 'A3', 'A4']]
    leg = ax.legend(handles=patches, title='AO Type', loc='upper left',
                    fontsize=9, facecolor='#16213e', edgecolor='#444',
                    labelcolor='white')
    leg.get_title().set_color('white')

    for bar, ao, dmg in zip(bars, ao_types, damages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f'{ao}\n{dmg:.1f}%', ha='center', va='bottom', fontsize=8,
                color='white', fontweight='bold')

    ax.set_xlabel('Force Magnitude (kN)', fontsize=12, color='white')
    ax.set_ylabel('Damaged Fraction (%)', fontsize=12, color='white')
    ax.set_title(f'Force Sweep — Emergent AO Classification{title_suffix}',
                 fontsize=13, color='white', pad=12)
    ax.set_xticks(forces)
    ax.set_ylim(0, max(damages) * 1.35 + 5)
    ax.tick_params(colors='white')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#444')
    ax.grid(axis='y', color='#333', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out = os.path.join(output_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"  Saved: {out}")
    plt.close(fig)


def _plot_bmd_sweep(sweep_results, output_dir, filename='v4_bmd_sweep.png',
                    title_suffix=''):
    """BMD counterfactual bar chart."""
    if not HAS_MPL:
        return
    bmds = [r[0] for r in sweep_results]
    damages = [r[1].damaged_fraction * 100 for r in sweep_results]
    ao_types = [r[1].ao_type for r in sweep_results]
    colors = [_AO_COLORS.get(t, '#888') for t in ao_types]

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')

    bars = ax.bar([f'{b:.1f}' for b in bmds], damages, color=colors,
                  edgecolor='white', linewidth=0.6, width=0.55, alpha=0.85,
                  zorder=3)
    ax.plot(range(len(bmds)), damages, color='#e94560', marker='o',
            markersize=5, linewidth=1.5, zorder=4)

    for bar, ao, dmg in zip(bars, ao_types, damages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f'{ao}\n{dmg:.1f}%', ha='center', va='bottom', fontsize=8,
                color='white', fontweight='bold')

    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=_AO_COLORS[k], label=k)
               for k in ['A0', 'A1', 'A2', 'A3', 'A4']]
    leg = ax.legend(handles=patches, title='AO Type', loc='upper right',
                    fontsize=9, facecolor='#16213e', edgecolor='#444',
                    labelcolor='white')
    leg.get_title().set_color('white')

    ax.set_xlabel('BMD Factor', fontsize=12, color='white')
    ax.set_ylabel('Damaged Fraction (%)', fontsize=12, color='white')
    ax.set_title(f'BMD Counterfactual — Same Force, Different Bone Quality{title_suffix}',
                 fontsize=13, color='white', pad=12)
    ax.set_ylim(0, max(damages) * 1.35 + 5)
    ax.tick_params(colors='white')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#444')
    ax.grid(axis='y', color='#333', linestyle='--', alpha=0.5)

    # Add annotation arrow showing clinical insight
    ax.annotate('Osteoporosis\n(BMD ↓)', xy=(len(bmds)-1, damages[-1]),
                xytext=(len(bmds)-1.5, damages[-1]*0.6),
                fontsize=9, color='#e94560', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#e94560', lw=1.5),
                ha='center')

    plt.tight_layout()
    out = os.path.join(output_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"  Saved: {out}")
    plt.close(fig)


def _plot_multi_slice(engine, output_dir, filename='v4_multi_slice.png'):
    """3-plane view: sagittal, axial, coronal with damage overlay."""
    if not HAS_MPL:
        return
    ct = engine._orig_ct
    if ct is None or engine._damage is None:
        return

    if engine.ds > 1:
        damage = engine._match_shape(
            zoom(engine._damage, engine.ds, order=1),
            engine._orig_mask.shape
        )
    else:
        damage = engine._damage

    result = engine._result
    params = engine.params

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor='#1a1a2e')
    hu_kw = dict(cmap='gray', origin='lower', vmin=-100, vmax=800, aspect='auto')
    dmg_kw = dict(cmap='hot', origin='lower', alpha=0.6, vmin=0, vmax=1, aspect='auto')

    titles = ['Sagittal (mid)', 'Axial (mid)', 'Coronal (mid)']
    mid = [ct.shape[i] // 2 for i in range(3)]

    # Row 1: Original CT
    slices_ct = [ct[mid[0]], ct[:, mid[1]], ct[:, :, mid[2]]]
    for i, (ax, sl, title) in enumerate(zip(axes[0], slices_ct, titles)):
        ax.imshow(sl.T if i < 2 else sl.T, **hu_kw)
        ax.set_title(f'{title} — Original', fontsize=11, color='white', pad=6)
        ax.axis('off')
        ax.set_facecolor('#1a1a2e')

    # Row 2: Damage overlay
    slices_dmg = [damage[mid[0]], damage[:, mid[1]], damage[:, :, mid[2]]]
    for i, (ax, sl_ct, sl_dmg, title) in enumerate(
            zip(axes[1], slices_ct, slices_dmg, titles)):
        ax.imshow(sl_ct.T, **hu_kw, alpha=0.4)
        im = ax.imshow(sl_dmg.T, **dmg_kw)
        ax.set_title(f'{title} — Damage', fontsize=11, color='white', pad=6)
        ax.axis('off')
        ax.set_facecolor('#1a1a2e')

    # Colorbar
    cbar = fig.colorbar(im, ax=axes[1, -1], fraction=0.046, pad=0.04)
    cbar.set_label('Damage', fontsize=10, color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    ao_color = _AO_COLORS.get(result.ao_type, '#888')
    fig.suptitle(
        f"AO {result.ao_type}  •  Force {params.force_magnitude:.1f} kN  •  "
        f"BMD {params.bmd_factor:.1f}  •  "
        f"Damaged {result.damaged_fraction*100:.1f}%",
        fontsize=14, color='white', y=0.98, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(output_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"  Saved: {out}")
    plt.close(fig)


def _plot_progression(engine, output_dir, params_list,
                      filename='v4_progression.png'):
    """Damage progression across scenarios (row = scenario, col = metric).

    params_list: list of (label, CausalParameters)
    """
    if not HAS_MPL:
        return

    n = len(params_list)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n), facecolor='#1a1a2e')
    if n == 1:
        axes = axes[np.newaxis, :]

    ct = engine._orig_ct
    hu_kw = dict(cmap='gray', origin='lower', vmin=-100, vmax=800, aspect='auto')
    dmg_kw = dict(cmap='hot', origin='lower', alpha=0.65, vmin=0, vmax=1, aspect='auto')

    for row, (label, params) in enumerate(params_list):
        engine.set_causal_params(params)
        result = engine.simulate(verbose=False)

        if engine.ds > 1:
            damage = engine._match_shape(
                zoom(engine._damage, engine.ds, order=1),
                engine._orig_mask.shape
            )
        else:
            damage = engine._damage

        mid_x = ct.shape[0] // 2
        ct_slice = ct[mid_x]
        dmg_slice = damage[mid_x]

        fractured_ct = engine.get_fractured_ct()
        frac_slice = fractured_ct[mid_x]

        ao_color = _AO_COLORS.get(result.ao_type, '#888')

        # Col 0: Original + label
        axes[row, 0].imshow(ct_slice.T, **hu_kw)
        axes[row, 0].set_title(
            f'{label}', fontsize=11, color=ao_color, pad=6, fontweight='bold')
        axes[row, 0].axis('off')
        axes[row, 0].set_facecolor('#1a1a2e')

        # Col 1: Damage overlay
        axes[row, 1].imshow(ct_slice.T, **hu_kw, alpha=0.4)
        axes[row, 1].imshow(dmg_slice.T, **dmg_kw)
        axes[row, 1].set_title(
            f'AO {result.ao_type} — Dmg {result.damaged_fraction*100:.1f}%',
            fontsize=11, color=ao_color, pad=6, fontweight='bold')
        axes[row, 1].axis('off')
        axes[row, 1].set_facecolor('#1a1a2e')

        # Col 2: Fractured CT
        axes[row, 2].imshow(frac_slice.T, **hu_kw)
        info = (f"F={params.force_magnitude:.0f}kN  BMD={params.bmd_factor:.1f}\n"
                f"Flex={params.flexion_angle:.0f}°  Post.Wall={result.posterior_wall_damage*100:.0f}%")
        axes[row, 2].text(0.02, 0.98, info, transform=axes[row, 2].transAxes,
                          fontsize=9, color='white', va='top',
                          fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor=ao_color, alpha=0.4))
        axes[row, 2].set_title('Fractured CT', fontsize=11, color='white', pad=6)
        axes[row, 2].axis('off')
        axes[row, 2].set_facecolor('#1a1a2e')

    fig.suptitle('Fracture Progression — Force × BMD Scenarios',
                 fontsize=15, color='white', y=0.99, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(output_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"  Saved: {out}")
    plt.close(fig)


# ============================================================================
#  DEMO
# ============================================================================

def demo(output_dir=None):
    """Run voxel-based fracture demo with real VerSe CT data."""
    if output_dir is None:
        output_dir = './fracture_v4_demo'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Voxel-Based Fracture Engine v4 — Real Vertebra Demo")
    print("=" * 70)

    # ---- Load real VerSe data ----
    print("\nLoading VerSe sub-verse503...")
    verse_root = os.path.join(os.path.dirname(__file__), '..', '..',
                              'VerSe', 'dataset-01training')
    ct_path = os.path.join(verse_root, 'rawdata', 'sub-verse503',
                           'sub-verse503_dir-ax_ct.nii.gz')
    mask_path = os.path.join(verse_root, 'derivatives', 'sub-verse503',
                             'sub-verse503_dir-ax_seg-vert_msk.nii.gz')

    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from _gen_real_fracture_visuals import load_vertebra
        ct, mask, _label = load_vertebra(ct_path, mask_path)
        mask = (mask > 0).astype(np.int32)
        print(f"  CT shape: {ct.shape}, bone voxels: {mask.sum():,}")
    except Exception as e:
        print(f"\n  ❌ Failed to load VerSe data: {e}")
        print(f"  Expected paths:")
        print(f"    CT:   {os.path.abspath(ct_path)}")
        print(f"    Mask: {os.path.abspath(mask_path)}")
        return

    # Determine downsample factor
    ds = max(1, int(np.cbrt(mask.sum() / 50000)))
    print(f"  Downsample: {ds}x" if ds > 1 else "  No downsampling needed")

    engine = VoxelFractureEngine(mask, ct, downsample=ds, seed=42)
    print(f"  Simulation grid: {engine.shape}, bone voxels: {engine.n_bone_voxels:,}")

    # ---- Scenario 1: Flexion → A1 ----
    print("\n1. Scenario: Force=4, flexion=30° → expect A1 (wedge)")
    engine.set_causal_params(CausalParameters(
        force_magnitude=4.0, flexion_angle=30.0, bmd_factor=0.8))
    r1 = engine.simulate(n_steps=150)

    # ---- Scenario 2: Burst → A3/A4 ----
    print("\n2. Scenario: Force=8, axial → expect A3/A4 (burst)")
    engine.set_causal_params(CausalParameters(
        force_magnitude=8.0, flexion_angle=5.0, bmd_factor=0.6))
    r2 = engine.simulate(n_steps=150)

    # ---- Scenario 3: Minimal → A0 ----
    print("\n3. Scenario: Force=2, normal BMD → expect A0")
    engine.set_causal_params(CausalParameters(
        force_magnitude=2.0, flexion_angle=10.0, bmd_factor=1.2))
    r3 = engine.simulate(n_steps=150)

    # ---- Force sweep ----
    print("\n4. Force sweep (BMD=0.7, flexion=15°):")
    engine.set_causal_params(CausalParameters(
        flexion_angle=15.0, bmd_factor=0.7))
    sweep = engine.parameter_sweep('force_magnitude',
                                    [2, 3, 4, 5, 6, 7, 8, 9, 10])

    # ---- BMD counterfactual with sweep ----
    print("\n5. BMD counterfactual (force=5, flexion=15°):")
    engine.set_causal_params(CausalParameters(
        force_magnitude=5.0, flexion_angle=15.0))
    bmd_sweep = engine.parameter_sweep('bmd_factor',
                                        [1.2, 1.0, 0.8, 0.7, 0.5, 0.3],
                                        verbose=True)

    # ---- All Plots ----
    print(f"\n6. Saving plots to {output_dir} ...")

    # Plot 1: Force sweep bar chart
    _plot_force_sweep(sweep, output_dir,
                      title_suffix=' (BMD=0.7, flexion=15°)')

    # Plot 2: BMD sweep bar chart
    _plot_bmd_sweep(bmd_sweep, output_dir,
                    title_suffix=' (Force=5 kN, flexion=15°)')

    # Plot 3: Animated GIF — Burst fracture progression
    print("\n  Generating animated GIF (burst fracture)...")
    engine.set_causal_params(CausalParameters(
        force_magnitude=8.0, flexion_angle=5.0, bmd_factor=0.6))
    engine.simulate_animated(n_steps=150, n_frames=30, verbose=False)
    engine.save_animation_gif(
        os.path.join(output_dir, 'v4_burst_animation.gif'), fps=8)

    # Plot 4: 4-panel dashboard (uses same burst simulation state)
    engine.plot_fracture_panels(
        engine._result,
        output_path=os.path.join(output_dir, 'v4_panels_burst.png'))

    # Plot 5: Animated GIF — Wedge fracture progression
    print("  Generating animated GIF (wedge fracture)...")
    engine.set_causal_params(CausalParameters(
        force_magnitude=4.0, flexion_angle=30.0, bmd_factor=0.8))
    engine.simulate_animated(n_steps=150, n_frames=30, verbose=False)
    engine.save_animation_gif(
        os.path.join(output_dir, 'v4_wedge_animation.gif'), fps=8)

    # Plot 4: Multi-slice 3-plane view
    _plot_multi_slice(engine, output_dir)

    # Plot 5: Progression comparison (5 scenarios)
    progression_scenarios = [
        ('A0: Force=2, BMD=1.2 (Normal)',
         CausalParameters(force_magnitude=2.0, flexion_angle=10.0, bmd_factor=1.2)),
        ('A1: Force=4, BMD=0.8 (Wedge)',
         CausalParameters(force_magnitude=4.0, flexion_angle=30.0, bmd_factor=0.8)),
        ('A2: Force=5, BMD=1.0 (Split)',
         CausalParameters(force_magnitude=5.0, flexion_angle=15.0, bmd_factor=1.0)),
        ('A3: Force=6, BMD=0.7 (Incomplete Burst)',
         CausalParameters(force_magnitude=6.0, flexion_angle=10.0, bmd_factor=0.7)),
        ('A4: Force=8, BMD=0.5 (Complete Burst)',
         CausalParameters(force_magnitude=8.0, flexion_angle=5.0, bmd_factor=0.5)),
    ]
    _plot_progression(engine, output_dir, progression_scenarios)

    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print(f"   Scenario 1 (wedge):   {r1.ao_type}")
    print(f"   Scenario 2 (burst):   {r2.ao_type}")
    print(f"   Scenario 3 (minimal): {r3.ao_type}")
    print(f"   Force sweep: {[r.ao_type for _, r in sweep]}")
    print("=" * 70)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Voxel-Based Fracture Engine v4')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()
    demo(args.output_dir)
