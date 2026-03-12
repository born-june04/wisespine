#!/usr/bin/env python3
"""
Emergent AO Fracture Engine v3
===============================

Physics-based vertebral fracture simulator where AO fracture type
EMERGES from causal parameters instead of being specified as input.

Key paradigm shift from v2:
  v2: setup_ao_loading("A4") → fixed loading → simulation → A4 result
  v3: set_causal_params(force=8, flexion=10°, bmd=0.7) → simulation → "result: A4"

This enables:
  - Forward prediction: "At BMD=0.7, will 5kN cause A1 or A3?"
  - Counterfactual reasoning: "If BMD were 0.5, would this become A4?"
  - Backward inference: "Given A4 result, what force was applied?"

Physics components:
  1. HU-based per-voxel material properties (Morgan & Keaveny 2003)
  2. Directional force field with endplate initiation
  3. Cortical shell as stress barrier in grid diffusion
  4. Compression deformation field for height loss
  5. Emergent AO classifier from damage patterns

Author: Wisespine Team
Date: 2026-03-06
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from fracture_simulator_v2 import (
    BoneFractureSimulator,
    classify_regions_from_mask,
    MATERIAL,
    DAMAGE_CMAP,
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================================
#  CAUSAL PARAMETERS
# ============================================================================

@dataclass
class CausalParameters:
    """Physical parameters that causally determine fracture outcome.

    These are the ONLY inputs to the simulation. The AO fracture type
    is determined by the physics, not by the user.

    Attributes:
        force_magnitude: Axial load in normalized units (1-10, ~kN scale).
                        1-3: normal activity, 4-6: fall, 7-10: high-energy trauma
        flexion_angle:   Forward bending angle in degrees (-10 to +45).
                        0 = pure axial, +30 = flexion-compression
        lateral_angle:   Lateral tilt in degrees (-20 to +20).
                        0 = symmetric, +15 = lateral bending
        bmd_factor:      Bone mineral density multiplier (0.3-1.5).
                        1.0 = normal, 0.5 = severe osteoporosis
        cortical_thickness: Cortical shell thickness multiplier (0.5-1.5).
                           Controls shell barrier strength.
    """
    force_magnitude: float = 5.0
    flexion_angle: float = 15.0     # degrees
    lateral_angle: float = 0.0      # degrees
    bmd_factor: float = 1.0
    cortical_thickness: float = 1.0

    def validate(self):
        """Clamp parameters to physically valid ranges."""
        self.force_magnitude = np.clip(self.force_magnitude, 0.5, 15.0)
        self.flexion_angle = np.clip(self.flexion_angle, -10.0, 45.0)
        self.lateral_angle = np.clip(self.lateral_angle, -20.0, 20.0)
        self.bmd_factor = np.clip(self.bmd_factor, 0.3, 1.5)
        self.cortical_thickness = np.clip(self.cortical_thickness, 0.3, 2.0)

    def to_dict(self) -> Dict:
        return {
            'force_magnitude': self.force_magnitude,
            'flexion_angle': self.flexion_angle,
            'lateral_angle': self.lateral_angle,
            'bmd_factor': self.bmd_factor,
            'cortical_thickness': self.cortical_thickness,
        }


# ============================================================================
#  AO CLASSIFICATION RESULT
# ============================================================================

@dataclass
class AOResult:
    """Result of emergent AO classification from simulation."""
    ao_type: str                     # 'A0', 'A1', 'A2', 'A3', 'A4'
    confidence: float                # 0-1 classification confidence
    posterior_wall_damage: float     # 0-1 posterior wall integrity
    anterior_height_loss: float     # ratio (0 = no loss, 0.5 = 50%)
    posterior_height_loss: float    # ratio
    canal_compromise: float         # 0-1 (0 = none, 0.5 = 50%)
    n_fragments: int                # detected fragment count
    max_damage: float
    damaged_fraction: float         # fraction with D > 0.5
    fractured_fraction: float       # fraction with D > 0.8

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
#  EMERGENT FRACTURE ENGINE
# ============================================================================

class EmergentFractureEngine:
    """Physics-based fracture engine with emergent AO classification.

    Instead of specifying AO type, provide causal parameters.
    The fracture type emerges from the simulation physics.

    Usage:
        engine = EmergentFractureEngine(mask_3d, ct_3d)
        engine.set_causal_params(CausalParameters(force=7, flexion=5, bmd=0.6))
        result = engine.simulate()
        print(result.ao_type)  # e.g., "A3"
        fractured_ct = engine.get_fractured_ct()
    """

    def __init__(
        self,
        mask: np.ndarray,
        ct: Optional[np.ndarray] = None,
        n_particles: int = 50000,
        grid_res: int = 64,
        seed: int = 42,
    ):
        """Initialize engine from bone mask and optional CT.

        Args:
            mask: 3D binary bone mask
            ct: 3D CT volume (HU values). If provided, used for
                per-voxel material properties.
            n_particles: Number of simulation particles
            grid_res: Grid resolution for stress transfer
            seed: Random seed
        """
        self.mask = mask
        self.ct = ct
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.grid_res = grid_res

        # Sample particles from mask
        self.positions = self._sample_particles(mask, n_particles)
        self.n_particles = len(self.positions)

        # Compute mask geometry
        self._compute_geometry(mask)

        # Per-voxel material properties from CT
        if ct is not None:
            self._voxel_threshold = self._compute_hu_threshold(ct, mask)
        else:
            self._voxel_threshold = None

        # State
        self.params: Optional[CausalParameters] = None
        self._sim: Optional[BoneFractureSimulator] = None
        self._damage_3d: Optional[np.ndarray] = None
        self._deformation_3d: Optional[np.ndarray] = None
        self._result: Optional[AOResult] = None

    # ================================================================
    #  GEOMETRY ANALYSIS
    # ================================================================

    def _sample_particles(self, mask, n_particles):
        """Sample particle positions from mask, normalized to [0,1] of bone bbox."""
        coords = np.argwhere(mask > 0)
        n_voxels = len(coords)

        if n_particles <= n_voxels:
            idx = self.rng.choice(n_voxels, size=n_particles, replace=False)
        else:
            idx = self.rng.choice(n_voxels, size=n_particles, replace=True)

        positions = coords[idx].astype(np.float64)
        positions += self.rng.uniform(-0.3, 0.3, positions.shape)

        pmin = positions.min(axis=0)
        pmax = positions.max(axis=0)
        prange = np.maximum(pmax - pmin, 1.0)
        positions = (positions - pmin) / prange

        # Save bbox so we can correctly invert normalisation elsewhere
        self._pmin = pmin.astype(np.float32)
        self._prange = prange.astype(np.float32)

        return positions.astype(np.float32)

    def _compute_geometry(self, mask):
        """Analyze mask geometry for force application and classification."""
        bone_coords = np.argwhere(mask > 0)
        self._bone_min = bone_coords.min(axis=0).astype(float)
        self._bone_max = bone_coords.max(axis=0).astype(float)
        self._bone_center = (self._bone_min + self._bone_max) / 2.0
        self._bone_extent = self._bone_max - self._bone_min

        # EDT for cortical/cancellous separation
        self._edt = distance_transform_edt(mask > 0).astype(np.float32)
        self._max_edt = max(self._edt.max(), 1.0)

        # Normalised positions: map from voxel to [0,1] for bone bbox
        self._bone_range = np.maximum(self._bone_extent, 1.0)

        # Correct inverse mapping: positions were normalised by (pmin, prange)
        # stored in self._pmin / self._prange (set in _sample_particles).
        # voxel = positions * prange + pmin
        voxel_pos = (
            self.positions * self._prange.astype(np.float64) +
            self._pmin.astype(np.float64)
        )

        # Relative position within bone bbox: 0=posterior/inferior, 1=anterior/superior
        rel_pos = (voxel_pos - self._bone_min) / self._bone_range
        rel_pos = rel_pos.astype(np.float32)

        self._rel_pos = rel_pos
        self._ap_ratio = np.clip(rel_pos[:, 1], 0, 1)  # 0=posterior, 1=anterior
        self._si_ratio = np.clip(rel_pos[:, 2], 0, 1)  # 0=inferior, 1=superior

        # Cortical depth per particle (0=surface, 1=deepest interior)
        vi = np.clip(voxel_pos[:, 0].astype(int), 0, mask.shape[0]-1)
        vj = np.clip(voxel_pos[:, 1].astype(int), 0, mask.shape[1]-1)
        vk = np.clip(voxel_pos[:, 2].astype(int), 0, mask.shape[2]-1)
        self._particle_depth = self._edt[vi, vj, vk] / self._max_edt
        self._particle_voxel_idx = (vi, vj, vk)

    # ================================================================
    #  HU → MATERIAL PROPERTIES (Component 1)
    # ================================================================

    def _compute_hu_threshold(self, ct, mask):
        """Map CT HU values to per-voxel fracture thresholds.

        Based on Morgan & Keaveny (2003):
            E = 6.850 × ρ^1.49
            ρ ≈ HU / 1000 (simplified apparent density)

        Returns threshold multiplier per particle.
        """
        # Use correct inverse mapping (positions are bbox-normalised)
        voxel_pos = (
            self.positions * self._prange.astype(np.float64) +
            self._pmin.astype(np.float64)
        )
        vi = np.clip(voxel_pos[:, 0].astype(int), 0, mask.shape[0]-1)
        vj = np.clip(voxel_pos[:, 1].astype(int), 0, mask.shape[1]-1)
        vk = np.clip(voxel_pos[:, 2].astype(int), 0, mask.shape[2]-1)

        # Get HU at each particle position
        particle_hu = ct[vi, vj, vk].astype(np.float32)

        # Reference HU for normal cancellous bone
        reference_hu = 300.0

        # Clamp HU to valid bone range
        particle_hu = np.clip(particle_hu, 50.0, 2000.0)

        # Power-law relationship: threshold ∝ (HU/ref)^1.5
        # Higher HU (denser bone) → higher fracture threshold
        threshold_scale = (particle_hu / reference_hu) ** 1.5

        # Normalize: median = 1.0
        median_scale = np.median(threshold_scale)
        if median_scale > 0:
            threshold_scale /= median_scale

        return threshold_scale.astype(np.float32)

    # ================================================================
    #  DIRECTIONAL FORCE FIELD (Component 2)
    # ================================================================

    def _generate_force_field(self, params: CausalParameters):
        """Generate directional force field across superior endplate.

        Instead of point-loading, distributes force across the entire
        superior endplate with direction and eccentricity from
        flexion/lateral angles.

        Returns:
            loading_points: list of [3] positions
            force_vectors: list of [3] direction vectors
            force_weights: relative intensity at each point
        """
        flex_rad = np.radians(params.flexion_angle)
        lat_rad = np.radians(params.lateral_angle)

        # Force direction: axial + flexion component + lateral component
        # z = axial (down), y = anterior-posterior, x = lateral
        fx = np.sin(lat_rad) * 0.3   # lateral shear
        fy = np.sin(flex_rad) * 0.3   # anterior shear from flexion
        fz = -1.0                     # axial compression (primary)
        force_dir = np.array([fx, fy, fz], dtype=np.float32)
        force_dir /= np.linalg.norm(force_dir)

        # Distribute loading points across superior endplate
        # Superior endplate = particles in top 15% of SI range
        superior_mask = self._si_ratio > 0.85
        if superior_mask.sum() < 10:
            superior_mask = self._si_ratio > 0.75

        superior_positions = self.positions[superior_mask]
        n_sup = len(superior_positions)

        # Sub-sample endplate for loading points (not all 50K particles)
        n_load_points = min(n_sup, 50)
        if n_sup > n_load_points:
            idx = self.rng.choice(n_sup, size=n_load_points, replace=False)
            load_positions = superior_positions[idx]
        else:
            load_positions = superior_positions

        # Compute force intensity at each loading point based on flexion
        # Flexion > 0 → anterior gets more force (A1-type loading)
        # Flexion ≈ 0 → uniform (A3/A4-type loading)
        load_ap_ratio = np.zeros(len(load_positions))
        for i, pos in enumerate(load_positions):
            # Map position to AP ratio
            vp = pos * (np.array(self.mask.shape, dtype=np.float32) - 1)
            load_ap_ratio[i] = (vp[1] - self._bone_min[1]) / self._bone_range[1]

        # Flexion bias: anterior particles get more force
        flexion_bias = np.sin(flex_rad) if flex_rad > 0 else 0
        # AP weighting: 0=posterior, 1=anterior
        # With flexion, anterior (ap_ratio high) gets amplified
        weight = 1.0 + flexion_bias * 2.0 * (load_ap_ratio - 0.5)

        # Lateral bias
        if abs(params.lateral_angle) > 1.0:
            load_lr_ratio = np.zeros(len(load_positions))
            for i, pos in enumerate(load_positions):
                vp = pos * (np.array(self.mask.shape, dtype=np.float32) - 1)
                load_lr_ratio[i] = (vp[0] - self._bone_min[0]) / self._bone_range[0]
            lat_bias = np.sin(lat_rad)
            weight += lat_bias * (load_lr_ratio - 0.5)

        weight = np.maximum(weight, 0.1)
        weight /= weight.sum()

        # Build loading points and vectors
        loading_points = [pos.copy() for pos in load_positions]
        force_vectors = [force_dir.copy() for _ in load_positions]
        force_weights = weight.tolist()

        return loading_points, force_vectors, force_weights

    # ================================================================
    #  CORTICAL SHELL BARRIER (Component 3)
    # ================================================================

    def _apply_cortical_barrier(self, sim: BoneFractureSimulator,
                                 params: CausalParameters):
        """Build cortical shell mask on the stress grid (vectorized).

        The cortical shell blocks stress propagation until it fails,
        naturally creating wedge (A1) vs burst (A3/A4) patterns.
        """
        res = sim.grid_res
        dx = sim.grid_dx
        cortical_depth_threshold = 0.20 * params.cortical_thickness

        # Only process occupied cells — vectorized (avoids 64³ Python loop)
        occ_cells = np.argwhere(sim.grid_occupancy)  # (M, 3)
        if len(occ_cells) == 0:
            self._cortical_grid = np.zeros((res, res, res), dtype=np.float32)
            return

        # Map grid cell centres to voxel indices.
        # Grid coords are in [0,1] of particle normalised space (same bbox as positions).
        pos = occ_cells.astype(np.float32) * dx  # (M, 3) in [0,1] particle space
        vi = np.clip(
            (pos[:, 0] * self._prange[0] + self._pmin[0]).astype(int),
            0, self.mask.shape[0] - 1,
        )
        vj = np.clip(
            (pos[:, 1] * self._prange[1] + self._pmin[1]).astype(int),
            0, self.mask.shape[1] - 1,
        )
        vk = np.clip(
            (pos[:, 2] * self._prange[2] + self._pmin[2]).astype(int),
            0, self.mask.shape[2] - 1,
        )

        depth = self._edt[vi, vj, vk] / self._max_edt
        is_cortical = depth < cortical_depth_threshold

        cortical_grid = np.zeros((res, res, res), dtype=np.float32)
        cortical_grid[
            occ_cells[is_cortical, 0],
            occ_cells[is_cortical, 1],
            occ_cells[is_cortical, 2],
        ] = 1.0
        self._cortical_grid = cortical_grid

    def _cortical_modulated_diffusion(self, sim, applied_force,
                                       force_weights=None):
        """Grid stress diffusion with cortical shell barrier.

        Replaces the standard _compute_stress_grid with a version
        where cortical cells attenuate stress propagation based on
        their integrity (1 - local_damage).
        """
        res = sim.grid_res
        dx = sim.grid_dx

        sim.grid_stress[:] = 0.0

        # ---- P2G: Distribute force from endplate loading points (vectorized) ----
        spread = 3
        n_points = len(sim.damage_points)

        # Pre-build offset grid for the Gaussian neighbourhood once
        offsets = np.mgrid[
            -spread:spread+1, -spread:spread+1, -spread:spread+1
        ].reshape(3, -1).T.astype(np.float32)        # (7³=343, 3)
        r2 = (offsets ** 2).sum(axis=1)
        w_kernel = np.exp(-r2 / (2.0 * spread))       # (343,)

        for idx, dp in enumerate(sim.damage_points):
            gi = int(np.clip(dp[0] / dx, 0, res - 1))
            gj = int(np.clip(dp[1] / dx, 0, res - 1))
            gk = int(np.clip(dp[2] / dx, 0, res - 1))
            w_point = force_weights[idx] if force_weights else 1.0 / max(n_points, 1)

            # Neighbour indices
            ni = offsets[:, 0].astype(int) + gi
            nj = offsets[:, 1].astype(int) + gj
            nk = offsets[:, 2].astype(int) + gk

            # Mask: within bounds + occupied
            valid = (
                (ni >= 0) & (ni < res) &
                (nj >= 0) & (nj < res) &
                (nk >= 0) & (nk < res)
            )
            ni, nj, nk = ni[valid], nj[valid], nk[valid]
            wv = w_kernel[valid]
            occ = sim.grid_occupancy[ni, nj, nk]
            ni, nj, nk, wv = ni[occ], nj[occ], nk[occ], wv[occ]

            np.add.at(
                sim.grid_stress,
                (ni, nj, nk),
                applied_force * wv * w_point * n_points,
            )

        # ---- Grid diffusion with cortical barrier ----
        source = sim.grid_stress.copy()
        n_iterations = 100
        cortical = self._cortical_grid

        # Damage per grid cell — vectorized (avoids N-particle Python loop)
        pidx = sim._particle_grid_idx          # (N, 3)
        damage_grid = np.zeros((res, res, res), dtype=np.float32)
        np.maximum.at(
            damage_grid,
            (pidx[:, 0], pidx[:, 1], pidx[:, 2]),
            sim.damage,
        )

        # Shell integrity: 1.0 when intact, 0.0 when fully damaged
        shell_integrity = (1.0 - damage_grid) * cortical

        for iteration in range(n_iterations):
            neighbor_avg = ndi.convolve(
                sim.grid_stress,
                sim._connectivity_kernel,
                mode='constant', cval=0.0
            )

            alpha = 0.6
            update_mask = sim.grid_occupancy

            sim.grid_stress[update_mask] = (
                (1 - alpha) * sim.grid_stress[update_mask] +
                alpha * neighbor_avg[update_mask] +
                source[update_mask] * 0.10
            )

            # ★ CORTICAL BARRIER: attenuate stress at cortical cells
            # Applied every 5th iteration (not every step) to avoid over-blocking
            # When shell is intact, stress is reduced by 40%
            # When shell is damaged, stress passes through freely
            if iteration % 5 == 0:
                barrier_mask = cortical > 0.5
                if barrier_mask.any():
                    attenuation = 1.0 - 0.4 * shell_integrity[barrier_mask]
                    sim.grid_stress[barrier_mask] *= attenuation

            sim.grid_stress[~sim.grid_occupancy] = 0.0

        # Scale grid stress by auto-calibrated K (NOT dividing by grid_max).
        #
        # Grid stress from P2G→diffusion is proportional to applied_force.
        # Dividing by grid_max would cancel that proportionality!
        # Instead we multiply by a fixed K computed once during calibration
        # (see simulate() step 1), converting raw grid units to stress units
        # comparable to damage_threshold (~0.5).
        if applied_force > 0 and hasattr(self, '_K_calibrated'):
            sim.grid_stress *= self._K_calibrated

        # ---- G2P (vectorized) ----
        sim.stress = sim.grid_stress[
            pidx[:, 0], pidx[:, 1], pidx[:, 2]
        ].copy()
        sim.stress *= (1.0 - sim.damage) ** 2

    # ================================================================
    #  PARTICLE → VOLUME MAPPING (internal, correct bbox coords)
    # ================================================================

    def _particles_to_damage_volume(self, damage: np.ndarray) -> np.ndarray:
        """Map particle damage to 3D voxel volume using the stored bbox mapping.

        Particles are normalised to [0,1] of the bone bounding box (pmin, prange).
        The external `particles_to_volume` helper in _gen_real_fracture_visuals
        assumes positions span the full mask shape, which is incorrect here.
        This method uses the correct inverse transform.
        """
        shape = self.mask.shape
        damage_vol = np.zeros(shape, dtype=np.float32)

        # Correct inverse: voxel = positions * prange + pmin
        vi = np.clip(
            (self.positions[:, 0] * self._prange[0] + self._pmin[0]).astype(int),
            0, shape[0] - 1,
        )
        vj = np.clip(
            (self.positions[:, 1] * self._prange[1] + self._pmin[1]).astype(int),
            0, shape[1] - 1,
        )
        vk = np.clip(
            (self.positions[:, 2] * self._prange[2] + self._pmin[2]).astype(int),
            0, shape[2] - 1,
        )

        # Aggregate: max damage per voxel
        np.maximum.at(damage_vol, (vi, vj, vk), damage.astype(np.float32))

        # Light smoothing to fill single-voxel gaps (sigma=0.5 preserves peaks)
        if damage_vol.max() > 0:
            damage_vol = ndi.gaussian_filter(damage_vol, sigma=0.5)

        damage_vol *= (self.mask > 0)
        return damage_vol

    # ================================================================
    #  COMPRESSION DEFORMATION FIELD (Component 4)
    # ================================================================

    def _compute_height_loss(self, damage_3d, params):
        """Compute compression deformation field from damage.

        Damaged regions compress along the loading direction, causing
        vertebral body height loss. Flexion angle determines whether
        anterior or posterior compresses more.

        Returns:
            deformation_3d: [shape] displacement field along SI axis (negative = compression)
        """
        shape = self.mask.shape
        deformation = np.zeros(shape, dtype=np.float32)
        bone_mask = self.mask > 0

        if not bone_mask.any():
            return deformation

        # Maximum height loss: 50% of vertebral body height
        max_compression_ratio = 0.50

        # SI axis = axis 2
        si_range = self._bone_extent[2]

        # For each voxel with damage, compute compression
        dmg = damage_3d.copy()
        dmg[~bone_mask] = 0.0

        # SI position ratio (0=bottom, 1=top)
        coords = np.argwhere(bone_mask)
        si_positions = (coords[:, 2] - self._bone_min[2]) / max(self._bone_range[2], 1)

        # AP position ratio (0=posterior, 1=anterior)
        ap_positions = (coords[:, 1] - self._bone_min[1]) / max(self._bone_range[1], 1)

        # Flexion effect: anterior compresses more than posterior
        flex_rad = np.radians(params.flexion_angle)
        # flexion_weight: how much each AP position contributes to compression
        # At flexion=30°, anterior gets 2x compression vs posterior
        flexion_weight = 1.0 + np.sin(flex_rad) * 2.0 * (ap_positions - 0.5)
        flexion_weight = np.maximum(flexion_weight, 0.1)

        # Compression magnitude: proportional to damage × position
        # Top of vertebra moves down most, bottom stays fixed
        local_damage = dmg[coords[:, 0], coords[:, 1], coords[:, 2]]
        compression = (
            local_damage *
            si_positions *          # top compresses more
            flexion_weight *        # anterior compresses more with flexion
            max_compression_ratio * si_range
        )

        # Apply as negative displacement along SI axis
        deformation[coords[:, 0], coords[:, 1], coords[:, 2]] = -compression

        # Smooth the deformation field
        deformation = ndi.gaussian_filter(deformation, sigma=2.0)
        deformation[~bone_mask] = 0.0

        return deformation

    def _apply_deformation_to_ct(self, ct, mask, deformation):
        """Apply compression deformation to CT volume.

        Uses scipy map_coordinates for sub-voxel interpolation.
        """
        from scipy.ndimage import map_coordinates

        shape = ct.shape
        result = ct.copy()

        # Build coordinate grid
        coords = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float32)

        # Apply deformation along SI axis (axis 2)
        coords[2] += deformation

        # Clamp to valid range
        coords[2] = np.clip(coords[2], 0, shape[2] - 1)

        # Interpolate
        result = map_coordinates(ct, coords, order=1, mode='nearest')
        result = result.astype(np.float32)

        return result

    # ================================================================
    #  EMERGENT AO CLASSIFIER (Component 5)
    # ================================================================

    def _classify_ao(self, damage_3d, deformation_3d,
                     n_fragments,
                     override_damaged_frac=None,
                     override_fractured_frac=None) -> AOResult:
        """Classify AO type from emergent damage patterns.

        Uses morphological analysis of the damage field:
        - Posterior wall integrity → wedge vs burst
        - Height loss pattern → A1 vs A2
        - Canal compromise → A3 vs A4
        - Overall severity → A0 if minimal

        Args:
            override_damaged_frac: If provided, use this instead of
                computing from damage_3d (particle-level is more accurate).
            override_fractured_frac: Same for fractured fraction.
        """
        bone_mask = self.mask > 0
        if not bone_mask.any():
            return AOResult('A0', 1.0, 0, 0, 0, 0, 0, 0, 0, 0)

        bone_coords = np.argwhere(bone_mask)
        bone_range = self._bone_range

        # AP ratio per voxel (0=posterior, 1=anterior)
        ap_ratio = (bone_coords[:, 1] - self._bone_min[1]) / max(bone_range[1], 1)
        # SI ratio per voxel (0=inferior, 1=superior)
        si_ratio = (bone_coords[:, 2] - self._bone_min[2]) / max(bone_range[2], 1)

        voxel_damage = damage_3d[bone_coords[:, 0], bone_coords[:, 1], bone_coords[:, 2]]
        voxel_deform = deformation_3d[bone_coords[:, 0], bone_coords[:, 1], bone_coords[:, 2]]

        # --- Posterior wall damage (key A1/A2 vs A3/A4 discriminator) ---
        posterior_zone = ap_ratio < 0.30
        if posterior_zone.any():
            posterior_wall_damage = voxel_damage[posterior_zone].mean()
        else:
            posterior_wall_damage = 0.0

        # --- Height loss: anterior vs posterior ---
        anterior_zone = ap_ratio > 0.65
        if anterior_zone.any():
            ant_height_loss = np.abs(voxel_deform[anterior_zone]).mean()
            ant_height_loss /= max(self._bone_extent[2], 1)
        else:
            ant_height_loss = 0.0

        posterior_body = (ap_ratio > 0.20) & (ap_ratio < 0.45)
        if posterior_body.any():
            post_height_loss = np.abs(voxel_deform[posterior_body]).mean()
            post_height_loss /= max(self._bone_extent[2], 1)
        else:
            post_height_loss = 0.0

        # --- Canal compromise (based on posterior wall damage only) ---
        canal_compromise = posterior_wall_damage * 1.2
        if posterior_wall_damage > 0.4:
            canal_compromise += 0.15
        canal_compromise = min(canal_compromise, 1.0)

        # --- Overall metrics ---
        max_damage = voxel_damage.max() if len(voxel_damage) > 0 else 0
        # Use particle-level stats if provided (more accurate)
        if override_damaged_frac is not None:
            damaged_frac = override_damaged_frac
        else:
            damaged_frac = (voxel_damage > 0.5).mean() if len(voxel_damage) > 0 else 0
        if override_fractured_frac is not None:
            fractured_frac = override_fractured_frac
        else:
            fractured_frac = (voxel_damage > 0.8).mean() if len(voxel_damage) > 0 else 0

        # --- Classification logic ---
        # Based on AO Spine criteria (Magerl 1994, Vaccaro 2013)
        # Thresholds calibrated per guideline:
        #
        #  AO | Post.Wall     | Ant.HeightLoss    | Canal
        #  A0 | intact (<5%)  | <5%               | 0%
        #  A1 | intact (<20%) | >15% ant>post      | 0%
        #  A2 | intact (<20%) | >10% bilateral     | 0%
        #  A3 | damaged (>30%)| >10%               | <25%
        #  A4 | damaged (>40%)| >15%               | >25%
        if max_damage < 0.3 or damaged_frac < 0.05:
            # A0: minimal or no fracture
            ao_type = 'A0'
            confidence = 0.9
        elif posterior_wall_damage < 0.20:
            # Posterior wall intact (<20%) → pure compression (A1 or A2)
            if ant_height_loss > 0.15 and ant_height_loss > post_height_loss * 1.3:
                # A1: significant anterior-dominant wedge
                ao_type = 'A1'
                confidence = min(ant_height_loss / 0.15, 2.0) / 2.0
            elif ant_height_loss > 0.10 or post_height_loss > 0.08:
                # A2: bilateral endplate compression (split/pincer)
                ao_type = 'A2'
                confidence = 0.7
            else:
                # Mild wedge (early A1)
                ao_type = 'A1'
                confidence = 0.5
        elif posterior_wall_damage < 0.30:
            # Transition zone (20–30%): moderate posterior involvement
            ao_type = 'A2'
            confidence = 0.65
        elif canal_compromise < 0.25 or posterior_wall_damage < 0.40:
            # A3: posterior wall clearly damaged (>30%) but not complete burst
            ao_type = 'A3'
            confidence = 0.8
        else:
            # A4: complete burst — posterior wall >40% + canal compromise >25%
            ao_type = 'A4'
            confidence = min(canal_compromise / 0.5, 1.0)

        return AOResult(
            ao_type=ao_type,
            confidence=confidence,
            posterior_wall_damage=posterior_wall_damage,
            anterior_height_loss=ant_height_loss,
            posterior_height_loss=post_height_loss,
            canal_compromise=canal_compromise,
            n_fragments=n_fragments,
            max_damage=max_damage,
            damaged_fraction=damaged_frac,
            fractured_fraction=fractured_frac,
        )

    # ================================================================
    #  MAIN SIMULATION
    # ================================================================

    def set_causal_params(self, params: CausalParameters):
        """Set causal parameters for the next simulation."""
        params.validate()
        self.params = params

    def simulate(
        self,
        n_steps: int = 200,
        verbose: bool = True,
    ) -> AOResult:
        """Run the full emergent fracture simulation.

        Returns AOResult with classified fracture type and metrics.
        """
        if self.params is None:
            self.params = CausalParameters()

        params = self.params

        # ---- Initialize simulator ----
        sim = BoneFractureSimulator(
            self.positions.copy(),
            material=MATERIAL.copy(),
            seed=self.seed,
        )
        sim.set_stress_mode('grid')
        sim.enable_anisotropy(ratio=2.0)

        # ---- Component 1: HU-based material properties ----
        # threshold_scale is per-particle multiplier; actual threshold
        # is set later based on force_magnitude and bmd_factor

        if self._voxel_threshold is not None:
            sim._threshold_scale = self._voxel_threshold * params.bmd_factor
        else:
            sim._threshold_scale = np.full(
                sim.N, params.bmd_factor, dtype=np.float32
            )

        # Region-based adjustments on top of HU threshold
        regions = classify_regions_from_mask(self.positions, self.mask)
        sim.regions = regions
        cortical_particles = regions == 3
        endplate_particles = regions == 4
        sim._threshold_scale[cortical_particles] *= 1.5 * params.cortical_thickness
        sim._threshold_scale[endplate_particles] *= 0.8  # endplate weaker

        # ---- Component 2: Directional force field ----
        loading_points, force_vectors, force_weights = \
            self._generate_force_field(params)
        sim.set_loading(loading_points, force_vectors)

        # ---- Component 3: Cortical barrier setup ----
        self._apply_cortical_barrier(sim, params)

        # ---- Run simulation loop ----
        sim.material['damage_rate'] = 0.005 * params.force_magnitude
        sim.material['damage_threshold'] = 0.5 * params.bmd_factor
        sim.material['cod_threshold'] = 0.5
        max_force = params.force_magnitude * 50.0

        force_ramp = np.linspace(0, max_force, n_steps)
        history = []

        # ★ AUTO-CALIBRATE K using percentile-based approach.
        #
        # Problem: Stress distribution is spike-shaped (max >> mean).
        # Peak-based calibration only affects a few particles near loading.
        #
        # Solution: Calibrate K so that the P-th percentile of non-zero
        # particle stress equals threshold at max ramp. This ensures
        # a controlled fraction (~20%) of particles exceed threshold.
        #
        # Higher force → same K but higher stress → more particles exceed
        # threshold → more damage. Force sensitivity is preserved because
        # stress scales linearly with applied_force (no grid_max division).
        TARGET_PERCENTILE = 80  # top 20% of stressed particles will damage at max ramp

        # CRITICAL: delete stale K so calibration measures RAW stress
        if hasattr(self, '_K_calibrated'):
            del self._K_calibrated
        sim.grid_stress[:] = 0.0
        sim.stress[:] = 0.0
        sim.damage[:] = 0.0

        ref_force = max_force  # calibrate at full ramp
        self._cortical_modulated_diffusion(sim, ref_force, force_weights)
        # Now sim.stress has RAW values (no K applied yet)
        raw_stress = sim.stress.copy()
        nonzero_stress = raw_stress[raw_stress > 0]

        threshold = sim.material['sigma_ult'] * sim.material['damage_threshold']
        if len(nonzero_stress) > 10:
            p_value = np.percentile(nonzero_stress, TARGET_PERCENTILE)
            if p_value > 0:
                # Scale so p_value × K = threshold
                # → at max force, 20% of particles exceed threshold
                self._K_calibrated = threshold / p_value
            else:
                self._K_calibrated = 0.01
        else:
            self._K_calibrated = 0.01  # fallback

        if verbose:
            print(f"  Auto-calibrated K={self._K_calibrated:.6f} "
                  f"(P{TARGET_PERCENTILE} raw={np.percentile(nonzero_stress, TARGET_PERCENTILE) if len(nonzero_stress) > 10 else 0:.4f}, "
                  f"threshold={threshold:.3f}, "
                  f"active={len(nonzero_stress)}/{sim.N})")

        # Reset ALL state from calibration run
        sim.grid_stress[:] = 0.0
        sim.stress[:] = 0.0
        sim.damage[:] = 0.0

        for i in range(n_steps):
            self._cortical_modulated_diffusion(
                sim, force_ramp[i], force_weights
            )
            sim.evolve_damage()

            if i % 20 == 0 or i == n_steps - 1:
                snapshot = {
                    'step': i,
                    'force': force_ramp[i],
                    'max_damage': sim.damage.max(),
                    'damaged_frac': (sim.damage > 0.5).sum() / sim.N,
                    'fractured_frac': (sim.damage > 0.8).sum() / sim.N,
                }
                history.append(snapshot)

                if verbose and i % 50 == 0:
                    threshold = sim.material['sigma_ult'] * sim.material['damage_threshold']
                    overstressed_n = (sim.stress > threshold).sum()
                    positive_stress = sim.stress[sim.stress > 0]
                    mean_stress = positive_stress.mean() if len(positive_stress) > 0 else 0.0
                    print(
                        f"  Step {i:4d}/{n_steps} | "
                        f"Force={force_ramp[i]:.1f} | "
                        f"MaxD={snapshot['max_damage']:.3f} | "
                        f"Damaged={snapshot['damaged_frac']*100:.1f}% | "
                        f"Stress: max={sim.stress.max():.4f} "
                        f"mean={mean_stress:.4f} "
                        f">{threshold:.3f}:{overstressed_n}/{sim.N}"
                    )

        # ---- Post-simulation: compute deformation ----
        sim.compute_deformation()
        # Fragment detection with size filter
        frag_result = sim.detect_fragments(damage_threshold=0.7)
        # Filter: only count fragments with >1% of total particles
        min_frag_size = max(sim.N * 0.01, 5)
        if frag_result['fragment_sizes']:
            n_fragments = sum(
                1 for sz in frag_result['fragment_sizes'].values()
                if sz >= min_frag_size
            )
        else:
            n_fragments = 0

        # Map damage back to 3D volume using correct bbox mapping
        damage_3d = self._particles_to_damage_volume(sim.damage)

        # ---- Component 4: Height loss deformation ----
        deformation_3d = self._compute_height_loss(damage_3d, params)

        # ---- Component 5: Classify AO type ----
        # Pass particle-level damage stats (more accurate than voxel-based
        # since voxel mapping may dilute values via smoothing/sparse sampling)
        particle_damaged_frac = (sim.damage > 0.5).sum() / sim.N
        particle_fractured_frac = (sim.damage > 0.8).sum() / sim.N
        result = self._classify_ao(
            damage_3d, deformation_3d, n_fragments,
            override_damaged_frac=particle_damaged_frac,
            override_fractured_frac=particle_fractured_frac,
        )

        # Store state
        self._sim = sim
        self._damage_3d = damage_3d
        self._deformation_3d = deformation_3d
        self._result = result
        self._history = history

        if verbose:
            print(f"\n  ★ Emergent AO Classification: {result.ao_type}")
            print(result.summary())

        return result

    def get_fractured_ct(self) -> np.ndarray:
        """Get the fractured CT volume after simulation.

        Applies both HU damage mapping and compression deformation.
        """
        if self.ct is None:
            raise ValueError("No CT volume provided. Initialize with ct=...")
        if self._damage_3d is None:
            raise ValueError("Run simulate() first.")

        from _gen_real_fracture_visuals import apply_damage_to_ct

        # Step 1: Apply HU damage mapping
        fractured = apply_damage_to_ct(
            self.ct, self.mask, self._damage_3d, seed=self.seed
        )

        # Step 2: Apply compression deformation (height loss)
        if self._deformation_3d is not None:
            fractured = self._apply_deformation_to_ct(
                fractured, self.mask, self._deformation_3d
            )

        return fractured

    # ================================================================
    #  VISUALIZATION (Phase 3)
    # ================================================================

    def plot_fracture_panels(
        self,
        result: AOResult,
        output_path: str = None,
    ) -> None:
        """Generate 4-panel sagittal visualization dashboard.

        Panel 1 — Original CT (sagittal mid-slice)
        Panel 2 — Damage field overlay on CT
        Panel 3 — Deformed CT with height-loss applied
        Panel 4 — Metrics summary table

        Title: "AO Type: {ao_type} | Force: {force}kN | BMD: {bmd}"
        """
        if not HAS_MPL:
            print("  [skip] matplotlib not available")
            return
        if self.ct is None:
            print("  [skip] No CT volume — initialize with ct=...")
            return
        if self._damage_3d is None:
            raise ValueError("Run simulate() first.")

        params = self.params
        mid_x = self.mask.shape[0] // 2

        # Sagittal slice: AP (axis 1) × SI (axis 2)
        ct_slice = self.ct[mid_x]
        damage_slice = self._damage_3d[mid_x]

        fractured_ct = self.get_fractured_ct()
        frac_slice = fractured_ct[mid_x]

        # ---- Figure setup: dark background ----
        fig = plt.figure(figsize=(18, 5.5), facecolor='#1a1a2e')
        gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.7], wspace=0.15)

        hu_kw = dict(cmap='gray', origin='lower', vmin=-100, vmax=800,
                     aspect='auto')

        # Panel 1: Original CT
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(ct_slice.T, **hu_kw)
        ax1.set_title('Original CT', fontsize=12, color='white', pad=8)
        ax1.axis('off')
        ax1.set_facecolor('#1a1a2e')

        # Panel 2: Damage overlay
        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(ct_slice.T, **hu_kw, alpha=0.5)
        dmg_im = ax2.imshow(damage_slice.T, cmap='hot', origin='lower',
                             alpha=0.65, vmin=0, vmax=1, aspect='auto')
        ax2.set_title('Damage Field', fontsize=12, color='white', pad=8)
        ax2.axis('off')
        ax2.set_facecolor('#1a1a2e')
        cbar = plt.colorbar(dmg_im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Damage (0–1)', fontsize=9, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

        # Panel 3: Fractured CT
        ax3 = fig.add_subplot(gs[2])
        ax3.imshow(frac_slice.T, **hu_kw)
        ax3.set_title('Fractured CT', fontsize=12, color='white', pad=8)
        ax3.axis('off')
        ax3.set_facecolor('#1a1a2e')

        # Panel 4: Metrics summary
        ax4 = fig.add_subplot(gs[3])
        ax4.axis('off')
        ax4.set_facecolor('#1a1a2e')

        ao_color = _AO_COLORS.get(result.ao_type, '#888')
        metrics_text = (
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
        ax4.text(
            0.05, 0.95, metrics_text,
            transform=ax4.transAxes, fontsize=10, color='white',
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=ao_color,
                      alpha=0.3, edgecolor=ao_color),
        )

        # Title
        fig.suptitle(
            f"AO {result.ao_type}  •  Force {params.force_magnitude:.1f} kN  •  "
            f"BMD {params.bmd_factor:.1f}  •  Flexion {params.flexion_angle:.0f}°",
            fontsize=14, color='white', y=0.98, fontweight='bold',
        )

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor(), edgecolor='none')
            print(f"  Saved: {output_path}")
        else:
            plt.show()
        plt.close(fig)

    # ================================================================
    #  COUNTERFACTUAL INTERFACE
    # ================================================================

    def counterfactual(self, **param_changes) -> AOResult:
        """Run a counterfactual simulation with modified parameters.

        Example:
            engine.simulate()  # original
            result_if_osteoporotic = engine.counterfactual(bmd_factor=0.5)
        """
        if self.params is None:
            raise ValueError("Run simulate() first, then counterfactual().")

        cf_params = CausalParameters(**self.params.to_dict())
        for key, value in param_changes.items():
            if hasattr(cf_params, key):
                setattr(cf_params, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")

        self.set_causal_params(cf_params)
        return self.simulate(verbose=False)

    def parameter_sweep(
        self,
        sweep_param: str,
        values: List[float],
        verbose: bool = True,
    ) -> List[Tuple[float, AOResult]]:
        """Sweep a single parameter and observe AO type transitions.

        Example:
            results = engine.parameter_sweep('force_magnitude', [2, 4, 6, 8, 10])
        """
        if self.params is None:
            self.params = CausalParameters()

        base_params = self.params.to_dict()
        results = []

        for val in values:
            params = CausalParameters(**base_params)
            setattr(params, sweep_param, val)
            self.set_causal_params(params)
            result = self.simulate(n_steps=150, verbose=False)
            results.append((val, result))

            if verbose:
                print(
                    f"  {sweep_param}={val:6.2f} → "
                    f"{result.ao_type} (conf={result.confidence:.2f}, "
                    f"dmg={result.damaged_fraction*100:.1f}%)"
                )

        return results


# ============================================================================
#  VISUALIZATION HELPERS
# ============================================================================

_AO_COLORS = {'A0': '#2196F3', 'A1': '#4CAF50', 'A2': '#FFC107',
               'A3': '#FF5722', 'A4': '#9C27B0'}


def _plot_force_sweep(sweep_results, output_dir, filename='v3_force_sweep.png',
                      title_suffix=''):
    """Bar + line plot of force sweep: x=force, y=damage%, color=AO type."""
    if not HAS_MPL:
        return
    forces = [r[0] for r in sweep_results]
    damages = [r[1].damaged_fraction * 100 for r in sweep_results]
    ao_types = [r[1].ao_type for r in sweep_results]
    colors = [_AO_COLORS.get(t, '#888') for t in ao_types]
    post_wall = [r[1].posterior_wall_damage * 100 for r in sweep_results]

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')

    # Bars: damage fraction
    bars = ax.bar(forces, damages, color=colors, edgecolor='white',
                  linewidth=0.6, width=0.65, alpha=0.85, zorder=3)

    # Line: trend
    ax.plot(forces, damages, color='#e94560', marker='o', markersize=5,
            linewidth=1.5, zorder=4, label='Damage %')

    # Posterior wall line
    ax.plot(forces, post_wall, color='#0f3460', marker='s', markersize=4,
            linewidth=1, linestyle='--', zorder=4, label='Post. wall %',
            alpha=0.7)

    # Legend patches
    import matplotlib.patches as mpatches
    ao_patches = [
        mpatches.Patch(color=_AO_COLORS[k], label=k)
        for k in ['A0', 'A1', 'A2', 'A3', 'A4']
    ]
    legend1 = ax.legend(handles=ao_patches, title='AO Type',
                        loc='upper left', fontsize=9,
                        facecolor='#16213e', edgecolor='#444',
                        title_fontsize=10, labelcolor='white')
    legend1.get_title().set_color('white')
    ax.add_artist(legend1)
    ax.legend(loc='upper right', fontsize=9, facecolor='#16213e',
              edgecolor='#444', labelcolor='white')

    # Annotate each bar
    for bar, ao, dmg in zip(bars, ao_types, damages):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f'{ao}\n{dmg:.1f}%',
                ha='center', va='bottom', fontsize=8, color='white',
                fontweight='bold')

    ax.set_xlabel('Force Magnitude (kN)', fontsize=12, color='white')
    ax.set_ylabel('Fraction (%)', fontsize=12, color='white')
    ax.set_title(f'Force Sweep — Emergent AO Classification{title_suffix}',
                 fontsize=13, color='white', pad=12)
    ax.set_xticks(forces)
    ax.set_ylim(0, max(max(damages), max(post_wall)) * 1.35 + 5)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#333', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out = os.path.join(output_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"  Saved: {out}")
    plt.close(fig)


def _plot_bmd_sweep(bmd_results, output_dir, filename='v3_bmd_sweep.png'):
    """Bar plot of BMD counterfactual sweep with dark theme."""
    if not HAS_MPL:
        return
    bmds = [r[0] for r in bmd_results]
    damages = [r[1].damaged_fraction * 100 for r in bmd_results]
    ao_types = [r[1].ao_type for r in bmd_results]
    colors = [_AO_COLORS.get(t, '#888') for t in ao_types]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')

    bars = ax.bar(bmds, damages, color=colors, edgecolor='white',
                  linewidth=0.6, width=0.08, alpha=0.85, zorder=3)

    ax.plot(bmds, damages, color='#e94560', marker='o', markersize=5,
            linewidth=1.5, zorder=4)

    import matplotlib.patches as mpatches
    ao_patches = [
        mpatches.Patch(color=_AO_COLORS[k], label=k)
        for k in ['A0', 'A1', 'A2', 'A3', 'A4']
    ]
    legend = ax.legend(handles=ao_patches, title='AO Type',
                       loc='upper left', fontsize=9,
                       facecolor='#16213e', edgecolor='#444',
                       title_fontsize=10, labelcolor='white')
    legend.get_title().set_color('white')

    for bar, ao, dmg in zip(bars, ao_types, damages):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{ao}\n{dmg:.1f}%',
                ha='center', va='bottom', fontsize=9, color='white',
                fontweight='bold')

    ax.set_xlabel('BMD Factor', fontsize=12, color='white')
    ax.set_ylabel('Damaged Fraction (%)', fontsize=12, color='white')
    ax.set_title('BMD Counterfactual — Same Force, Different Bone Density',
                 fontsize=13, color='white', pad=12)
    ax.invert_xaxis()
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#333', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out = os.path.join(output_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"  Saved: {out}")
    plt.close(fig)


# ============================================================================
#  DEMO / SELF-TEST
# ============================================================================

def demo_emergent_fracture(output_dir=None):
    """Demonstrate emergent AO classification with parameter sweep."""
    if output_dir is None:
        output_dir = '/tmp/fracture_v3_demo'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Emergent AO Fracture Engine v3 — Demo")
    print("=" * 70)

    # Create synthetic vertebra mask for demo
    print("\n1. Creating synthetic vertebra...")
    shape = (40, 40, 30)
    mask = np.zeros(shape, dtype=np.int32)

    # Ellipsoidal vertebral body
    cz, cy, cx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    center = np.array([20, 20, 15])
    radii = np.array([15, 12, 12])
    dist = ((cz - center[0])/radii[0])**2 + ((cy - center[1])/radii[1])**2 + ((cx - center[2])/radii[2])**2
    mask[dist < 1.0] = 1
    print(f"   Mask shape: {shape}, voxels: {mask.sum()}")

    # Synthetic CT (bone HU)
    ct = np.random.uniform(200, 600, shape).astype(np.float32)
    ct[mask == 0] = -100  # air outside bone

    engine = EmergentFractureEngine(mask, ct, n_particles=10000, seed=42)

    # ---- Scenario 1: Low force + flexion → A1 (wedge) ----
    print("\n2. Scenario: Low force + flexion → expect A1 (wedge)")
    engine.set_causal_params(CausalParameters(
        force_magnitude=4.0,
        flexion_angle=30.0,
        bmd_factor=0.8,
    ))
    result1 = engine.simulate(n_steps=150)

    # ---- Scenario 2: High force + axial → A3/A4 (burst) ----
    print("\n3. Scenario: High force + axial → expect A3/A4 (burst)")
    engine.set_causal_params(CausalParameters(
        force_magnitude=8.0,
        flexion_angle=5.0,
        bmd_factor=0.6,
    ))
    result2 = engine.simulate(n_steps=150)

    # ---- Scenario 3: Low force + normal BMD → A0 (minimal) ----
    print("\n4. Scenario: Low force + normal BMD → expect A0")
    engine.set_causal_params(CausalParameters(
        force_magnitude=2.0,
        flexion_angle=10.0,
        bmd_factor=1.2,
    ))
    result3 = engine.simulate(n_steps=150)

    # ---- Parameter sweep: force magnitude ----
    print("\n5. Force magnitude sweep (BMD=0.7, flexion=15°):")
    engine.set_causal_params(CausalParameters(
        flexion_angle=15.0,
        bmd_factor=0.7,
    ))
    sweep_results = engine.parameter_sweep(
        'force_magnitude', [2, 3, 4, 5, 6, 7, 8, 9, 10]
    )

    # ---- Counterfactual: BMD effect ----
    print("\n6. Counterfactual: Same force (5kN, flexion=15°), different BMD")
    engine.set_causal_params(CausalParameters(
        force_magnitude=5.0,
        flexion_angle=15.0,
        bmd_factor=1.0,
    ))
    result_normal = engine.simulate(n_steps=150, verbose=False)

    bmd_sweep = []
    for bmd in [1.2, 1.0, 0.7, 0.5, 0.3]:
        r = engine.counterfactual(bmd_factor=bmd)
        bmd_sweep.append((bmd, r))
        print(f"   BMD={bmd:.1f} → {r.ao_type} (dmg={r.damaged_fraction*100:.1f}%)")

    # ---- Save plots ----
    print(f"\n7. Saving plots to {output_dir} ...")
    _plot_force_sweep(
        sweep_results, output_dir,
        filename='v3_force_sweep.png',
        title_suffix=' (BMD=0.7, flexion=15°)',
    )
    _plot_bmd_sweep(bmd_sweep, output_dir)

    # 3-panel visualization for the burst scenario (most visually interesting)
    engine.set_causal_params(CausalParameters(
        force_magnitude=8.0, flexion_angle=5.0, bmd_factor=0.6,
    ))
    engine.simulate(n_steps=150, verbose=False)
    engine.plot_fracture_panels(
        engine._result,
        output_path=os.path.join(output_dir, 'v3_panels_burst.png'),
    )

    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print(f"   Scenario 1 (flexion): {result1.ao_type}")
    print(f"   Scenario 2 (burst):   {result2.ao_type}")
    print(f"   Scenario 3 (minimal): {result3.ao_type}")
    print(f"   Force sweep: {[r.ao_type for _, r in sweep_results]}")
    print("=" * 70)

    return {
        'scenarios': [result1, result2, result3],
        'sweep': sweep_results,
        'bmd_sweep': bmd_sweep,
    }


def demo_real_vertebra(output_dir=None):
    """Demo with real VerSe CT data (sub-verse503, label 22).

    Expected output:
        Loading VerSe sub-verse503, label 22...
        Running emergent simulation (force=6, bmd=0.8, flexion=20°)...
        ★ Emergent: A1 (wedge)
          Anterior height loss: 25%
          Saved: v3_emergent_real.png
          Saved: v3_emergent_sweep.png
    """
    if output_dir is None:
        output_dir = '/tmp/fracture_v3_demo'
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load real VerSe data ----
    verse_root = os.path.join(os.path.dirname(__file__), '..', '..', 'VerSe',
                              'dataset-01training')
    ct_path = os.path.join(verse_root, 'rawdata', 'sub-verse503',
                           'sub-verse503_dir-ax_ct.nii.gz')
    mask_path = os.path.join(verse_root, 'derivatives', 'sub-verse503',
                             'sub-verse503_dir-ax_seg-vert_msk.nii.gz')

    ct = mask = None
    print("Loading VerSe sub-verse503, label 22...")
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from _gen_real_fracture_visuals import load_vertebra
        ct, mask, _label = load_vertebra(ct_path, mask_path)
        print(f"  CT shape: {ct.shape}, mask voxels: {mask.sum()}")
    except Exception as e:
        print(f"  [warn] Could not load VerSe data ({e}). Using synthetic vertebra.")

    if ct is None or mask is None:
        shape = (50, 50, 40)
        mask = np.zeros(shape, dtype=np.int32)
        cz, cy, cx = np.mgrid[:shape[0], :shape[1], :shape[2]]
        center = [25, 25, 20]
        radii = [18, 14, 14]
        dist = (
            ((cz - center[0]) / radii[0]) ** 2 +
            ((cy - center[1]) / radii[1]) ** 2 +
            ((cx - center[2]) / radii[2]) ** 2
        )
        mask[dist < 1.0] = 1
        rng = np.random.default_rng(42)
        ct = rng.uniform(200, 600, shape).astype(np.float32)
        ct[mask == 0] = -100

    n_vox = int(mask.sum())
    n_particles = min(max(n_vox // 4, 5000), 30000)

    engine = EmergentFractureEngine(mask, ct, n_particles=n_particles, seed=42)

    # ---- Main scenario: moderate force + flexion → A1 ----
    print("\nRunning emergent simulation (force=6, bmd=0.8, flexion=20°)...")
    engine.set_causal_params(CausalParameters(
        force_magnitude=6.0, flexion_angle=20.0, bmd_factor=0.8,
    ))
    result = engine.simulate(n_steps=200)
    print(f"★ Emergent: {result.ao_type}")
    print(f"  Anterior height loss: {result.anterior_height_loss*100:.1f}%")

    # 3-panel visualization
    out_panels = os.path.join(output_dir, 'v3_emergent_real.png')
    engine.plot_fracture_panels(result, out_panels)

    # ---- Force sweep on the same vertebra ----
    print("\nForce sweep (BMD=0.8, flexion=20°):")
    engine.set_causal_params(CausalParameters(flexion_angle=20.0, bmd_factor=0.8))
    sweep = engine.parameter_sweep('force_magnitude', [2, 3, 4, 5, 6, 7, 8, 9, 10])

    _plot_force_sweep(
        sweep, output_dir,
        filename='v3_emergent_sweep.png',
        title_suffix=' — Real Vertebra (BMD=0.8, flexion=20°)',
    )

    return result, sweep


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Emergent AO Fracture Engine v3')
    parser.add_argument('--demo', action='store_true',
                        help='Run synthetic demo with force/BMD sweep')
    parser.add_argument('--real-vertebra', action='store_true',
                        help='Run demo on real VerSe CT (sub-verse503)')
    parser.add_argument('--sweep', action='store_true',
                        help='Alias for --demo (parameter sweep)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory for output figures (default: /tmp/fracture_v3_demo)')
    args = parser.parse_args()

    if args.real_vertebra:
        demo_real_vertebra(args.output_dir)
    elif args.demo or args.sweep or len(sys.argv) == 1:
        demo_emergent_fracture(args.output_dir)
