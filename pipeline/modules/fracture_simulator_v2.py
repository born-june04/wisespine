#!/usr/bin/env python3
"""
Bone Fracture Simulator v2
===========================

NumPy/SciPy-based bone fracture simulation with iterative physics improvements.
Reproduces the Taichi simulator baseline and extends with:
  Phase 0: Baseline (1/r² stress, isotropic, single-point)
  Phase 1: Grid-based stress transfer
  Phase 2: AO-type-specific loading
  Phase 3: Bone anisotropy
  Phase 4: Fragment generation
  Phase 5: Cascade visualization

Usage:
  python fracture_simulator_v2.py --test-baseline
  python fracture_simulator_v2.py --test-all
  python fracture_simulator_v2.py --gif

Author: Wisespine Team
Date: 2026-02-12
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import argparse
import sys

# Optional imports for visualization
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib import animation
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    try:
        import imageio
        HAS_IMAGEIO = True
    except ImportError:
        HAS_IMAGEIO = False

# ============================================================================
#  CONSTANTS — Bone Material Properties (Normalized)
# ============================================================================

MATERIAL = {
    'E_bone': 1.0,       # Young's modulus (normalized)
    'nu_bone': 0.3,      # Poisson's ratio
    'sigma_ult': 1.0,    # Ultimate stress (normalized)
    'damage_threshold': 0.5,  # Fraction of σ_ult for damage initiation
    'damage_rate': 0.01,      # Damage increment per step
    'cod_threshold': 0.8,     # Damage level for crack opening
    'cod_magnitude': 0.02,    # Crack opening displacement scale
}

# Damage state thresholds for visualization
DAMAGE_STATES = {
    'intact':       (0.00, 0.05),
    'stressed':     (0.05, 0.10),
    'microfracture': (0.10, 0.50),
    'fracture':     (0.50, 0.80),
    'collapsed':    (0.80, 0.95),
    'comminuted':   (0.95, 1.01),
}

# Colors: ivory → yellow → orange → red → dark red
DAMAGE_CMAP = LinearSegmentedColormap.from_list('bone_damage', [
    (0.0, (0.95, 0.92, 0.85)),   # ivory - intact
    (0.1, (0.95, 0.92, 0.75)),   # slight yellow - stressed
    (0.3, (0.95, 0.75, 0.30)),   # orange - microfracture
    (0.6, (0.90, 0.30, 0.10)),   # red - fracture
    (0.8, (0.70, 0.15, 0.10)),   # dark red - collapsed
    (1.0, (0.40, 0.08, 0.05)),   # very dark - comminuted
]) if HAS_MPL else None


# ============================================================================
#  AO FRACTURE CLASSIFICATION — Loading Configurations (Phase 2)
# ============================================================================

# Each config defines:
#   - name: Clinical name
#   - mechanism: Clinical mechanism
#   - loading_points: Relative offsets from vertebra center [anterior(+y), lateral(±x), superior(+z)]
#   - force_vectors: Unit direction vectors for force application at each point
#   - max_force: Peak force (normalized)
#   - description: Brief biomechanical description

AO_LOAD_CONFIGS = {
    'A1': {
        'name': 'A1 Wedge Compression',
        'mechanism': 'Flexion-compression (anterior column failure)',
        'loading_points': [
            [0.0, 0.07, 0.05],    # Superior anterior endplate
            [0.0, 0.05, -0.04],   # Inferior anterior body
        ],
        'force_vectors': [
            [0.0, 0.2, -1.0],    # Axial + slight anterior shear
            [0.0, -0.1, 1.0],    # Counter-force from below
        ],
        'max_force': 3.0,
        'cod_scale': 1.5,  # COD multiplier for this type
    },
    'A2': {
        'name': 'A2 Split Fracture',
        'mechanism': 'Coronal plane separation (axial + lateral)',
        'loading_points': [
            [0.06, 0.0, 0.0],    # Right lateral body
            [-0.06, 0.0, 0.0],   # Left lateral body
        ],
        'force_vectors': [
            [1.0, 0.0, -0.5],    # Right-lateral + axial
            [-1.0, 0.0, -0.5],   # Left-lateral + axial (splitting force)
        ],
        'max_force': 3.5,
        'cod_scale': 2.0,
    },
    'A3': {
        'name': 'A3 Incomplete Burst',
        'mechanism': 'Centered axial compression (partial posterior wall)',
        'loading_points': [
            [0.0, 0.02, 0.06],   # Central superior endplate
            [0.0, -0.02, 0.0],   # Central body (posterior-side)
        ],
        'force_vectors': [
            [0.0, 0.0, -1.0],    # Pure axial compression
            [0.0, -0.3, 0.0],    # Slight posterior push (nucleus pressure)
        ],
        'max_force': 4.0,
        'cod_scale': 2.5,
    },
    'A4': {
        'name': 'A4 Complete Burst',
        'mechanism': 'Explosive radial fracture + retropulsion',
        'loading_points': [
            [0.0, 0.0, 0.06],    # Central superior (main axial load)
            [0.0, 0.0, -0.05],   # Central inferior (axial reaction)
            [0.0, -0.06, 0.0],   # Posterior wall (retropulsion source)
            [0.05, 0.0, 0.0],    # Right lateral (burst expansion)
            [-0.05, 0.0, 0.0],   # Left lateral (burst expansion)
        ],
        'force_vectors': [
            [0.0, 0.0, -1.0],    # Main axial compression
            [0.0, 0.0, 1.0],     # Axial reaction
            [0.0, -1.0, 0.0],    # Retropulsion into canal
            [1.0, 0.0, 0.0],     # Right burst
            [-1.0, 0.0, 0.0],    # Left burst
        ],
        'max_force': 5.0,
        'cod_scale': 3.0,
    },
}


# ============================================================================
#  PARTICLE GENERATION
# ============================================================================

def generate_vertebra_particles(
    n_particles: int = 50000,
    shape: str = 'ellipsoid',
    seed: int = 42
) -> np.ndarray:
    """Generate synthetic vertebra particle cloud.

    Creates a simplified vertebral body + posterior elements geometry
    in normalized [0, 1] space centered at (0.5, 0.5, 0.5).

    Args:
        n_particles: Target number of particles
        shape: 'ellipsoid' or 'box'
        seed: Random seed

    Returns:
        positions: [N, 3] particle positions
    """
    rng = np.random.RandomState(seed)

    # ---- Vertebral body: oblate ellipsoid ----
    # Semi-axes: a=0.15 (L-R), b=0.10 (A-P), c=0.08 (S-I)
    n_body = int(n_particles * 0.75)
    body_pts = rng.randn(n_body * 3, 3)  # oversample, then filter
    body_pts[:, 0] *= 0.15  # left-right
    body_pts[:, 1] *= 0.10  # anterior-posterior
    body_pts[:, 2] *= 0.08  # superior-inferior
    # Keep only points inside ellipsoid
    r2 = (body_pts[:, 0] / 0.15)**2 + (body_pts[:, 1] / 0.10)**2 + (body_pts[:, 2] / 0.08)**2
    body_pts = body_pts[r2 <= 1.0][:n_body]

    # ---- Posterior elements: two pedicles + lamina ----
    n_post = n_particles - len(body_pts)
    # Left pedicle
    n_ped = n_post // 3
    left_ped = rng.randn(n_ped, 3) * np.array([0.02, 0.05, 0.03])
    left_ped[:, 0] -= 0.08
    left_ped[:, 1] -= 0.12

    # Right pedicle
    right_ped = rng.randn(n_ped, 3) * np.array([0.02, 0.05, 0.03])
    right_ped[:, 0] += 0.08
    right_ped[:, 1] -= 0.12

    # Lamina (connecting arch)
    n_lam = n_post - 2 * n_ped
    lamina = rng.randn(n_lam, 3) * np.array([0.10, 0.02, 0.03])
    lamina[:, 1] -= 0.18

    # Combine and center at (0.5, 0.5, 0.5)
    all_pts = np.vstack([body_pts, left_ped, right_ped, lamina])
    all_pts += 0.5

    return all_pts.astype(np.float32)


def classify_regions(positions: np.ndarray) -> np.ndarray:
    """Classify particles into anatomical regions.

    Returns:
        region_ids: [N] integer region IDs
            0 = anterior body, 1 = central body, 2 = posterior body
            3 = cortical shell, 4 = endplate (sup/inf)
            5 = pedicle, 6 = lamina, 7 = canal zone
    """
    center = 0.5
    regions = np.zeros(len(positions), dtype=np.int32)
    rel = positions - center

    # Distance from center
    r_xy = np.sqrt(rel[:, 0]**2 + rel[:, 1]**2)

    # Anterior (y > 0 from center) vs posterior (y < 0)
    is_anterior = rel[:, 1] > 0.02
    is_posterior_body = (rel[:, 1] < -0.02) & (rel[:, 1] > -0.10)
    is_pedicle = (rel[:, 1] < -0.10) & (np.abs(rel[:, 0]) > 0.05)
    is_lamina = (rel[:, 1] < -0.15)
    is_endplate = np.abs(rel[:, 2]) > 0.06

    # Cortical shell: outer 20% of body
    body_radius = 0.12
    is_cortical = (r_xy > body_radius * 0.8) & ~is_pedicle & ~is_lamina

    # Assign regions
    regions[is_anterior] = 0             # anterior body
    regions[~is_anterior & ~is_posterior_body & ~is_pedicle & ~is_lamina] = 1  # central
    regions[is_posterior_body] = 2        # posterior body
    regions[is_cortical] = 3             # cortical shell
    regions[is_endplate] = 4             # endplate
    regions[is_pedicle] = 5              # pedicle
    regions[is_lamina] = 6               # lamina

    return regions


# ============================================================================
#  FRACTURE SIMULATOR CLASS
# ============================================================================

class BoneFractureSimulator:
    """Physics-based bone fracture simulation.

    Implements Continuum Damage Mechanics (CDM) with iteratively
    upgradeable physics models.
    """

    def __init__(
        self,
        positions: np.ndarray,
        material: Dict = None,
        seed: int = 42,
    ):
        """
        Args:
            positions: [N, 3] particle positions in [0, 1] normalized space
            material: Material property dictionary
            seed: Random seed
        """
        self.N = len(positions)
        self.original_positions = positions.copy()
        self.positions = positions.copy()
        self.material = material or MATERIAL.copy()
        self.rng = np.random.RandomState(seed)

        # State fields
        self.damage = np.zeros(self.N, dtype=np.float32)
        self.stress = np.zeros(self.N, dtype=np.float32)
        self.regions = classify_regions(positions)

        # History for visualization
        self.history: List[Dict] = []
        self.step_count = 0

        # Force configuration
        self.damage_points: List[np.ndarray] = []
        self.force_vectors: List[np.ndarray] = []

    def set_loading(
        self,
        damage_points: List[np.ndarray],
        force_vectors: Optional[List[np.ndarray]] = None,
    ):
        """Set force loading configuration.

        Args:
            damage_points: List of [3] loading point positions
            force_vectors: Optional list of [3] force direction vectors.
                          If None, uses isotropic scalar force.
        """
        self.damage_points = [np.array(p, dtype=np.float32) for p in damage_points]
        if force_vectors is not None:
            self.force_vectors = [np.array(v, dtype=np.float32) for v in force_vectors]
        else:
            self.force_vectors = [None] * len(damage_points)

    def set_stress_mode(self, mode: str):
        """Set stress computation mode.

        Args:
            mode: 'baseline' (1/r² decay) or 'grid' (P2G/G2P transfer)
        """
        assert mode in ('baseline', 'grid'), f"Unknown stress mode: {mode}"
        self.stress_mode = mode

        if mode == 'grid' and not hasattr(self, '_grid_initialized'):
            self._init_grid()

    def setup_ao_loading(self, ao_type: str):
        """Configure loading from AO classification.

        Args:
            ao_type: 'A1', 'A2', 'A3', or 'A4'
        """
        assert ao_type in AO_LOAD_CONFIGS, f"Unknown AO type: {ao_type}"
        config = AO_LOAD_CONFIGS[ao_type]

        # Convert relative offsets to absolute positions
        center = self.original_positions.mean(axis=0)
        damage_points = [
            center + np.array(offset, dtype=np.float32)
            for offset in config['loading_points']
        ]

        # Normalize force vectors
        force_vectors = []
        for fv in config['force_vectors']:
            v = np.array(fv, dtype=np.float32)
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            force_vectors.append(v)

        self.set_loading(damage_points, force_vectors)

        # Adjust COD scale for this fracture type
        self.material['cod_magnitude'] = 0.02 * config.get('cod_scale', 1.0)

        self._ao_type = ao_type
        self._ao_config = config

    # ================================================================
    #  GRID INFRASTRUCTURE (Phase 1)
    # ================================================================

    def _init_grid(self, resolution: int = 64):
        """Initialize the stress transfer grid."""
        self.grid_res = resolution
        self.grid_dx = 1.0 / resolution

        # Build occupancy grid: which cells contain material
        self.grid_occupancy = np.zeros(
            (resolution, resolution, resolution), dtype=bool
        )
        self.grid_mass = np.zeros(
            (resolution, resolution, resolution), dtype=np.float32
        )
        self.grid_stress = np.zeros(
            (resolution, resolution, resolution), dtype=np.float32
        )

        # Precompute particle-to-grid mapping
        self._particle_grid_idx = np.floor(
            self.original_positions / self.grid_dx
        ).astype(int)
        self._particle_grid_idx = np.clip(
            self._particle_grid_idx, 0, resolution - 1
        )

        # Fill occupancy from particles
        for i in range(self.N):
            gi, gj, gk = self._particle_grid_idx[i]
            self.grid_occupancy[gi, gj, gk] = True
            self.grid_mass[gi, gj, gk] += 1.0

        # Normalize mass
        self.grid_mass /= max(self.grid_mass.max(), 1.0)

        # Precompute connectivity kernel: 6-connected (face-sharing)
        self._connectivity_kernel = np.zeros((3, 3, 3), dtype=np.float32)
        self._connectivity_kernel[1, 1, 0] = 1.0  # -z
        self._connectivity_kernel[1, 1, 2] = 1.0  # +z
        self._connectivity_kernel[1, 0, 1] = 1.0  # -y
        self._connectivity_kernel[1, 2, 1] = 1.0  # +y
        self._connectivity_kernel[0, 1, 1] = 1.0  # -x
        self._connectivity_kernel[2, 1, 1] = 1.0  # +x
        self._connectivity_kernel /= 6.0

        self._grid_initialized = True
        print(f"  Grid initialized: {resolution}³, "
              f"occupied={self.grid_occupancy.sum()}/{resolution**3} cells")

    # ================================================================
    #  STRESS COMPUTATION — Phase 0: 1/r² baseline
    # ================================================================

    def _compute_stress_baseline(self, applied_force: float):
        """Compute stress field using 1/r² point-load decay.

        This is the baseline model matching the original Taichi simulator.
        """
        self.stress[:] = 0.0

        for dp in self.damage_points:
            dist = np.linalg.norm(self.original_positions - dp, axis=1)

            # 1/r² decay with regularization
            local_stress = np.where(
                dist < 0.001,
                applied_force * 10.0,
                applied_force / (dist**2 * 100.0 + 0.1)
            )

            # Stiffness degradation: E_eff = E₀(1-D)²
            effective_stiffness = (1.0 - self.damage) ** 2
            local_stress *= effective_stiffness

            # Accumulate from multiple loading points
            self.stress += local_stress

    # ================================================================
    #  STRESS COMPUTATION — Phase 1: Grid-based P2G/G2P
    # ================================================================

    def _compute_stress_grid(self, applied_force: float):
        """Compute stress using grid-based transfer.

        Three-step process:
        1. P2G: Deposit force at loading points onto grid
        2. Grid: Diffuse stress through connected material (Jacobi iteration)
        3. G2P: Gather stress from grid back to particles

        Key improvement: stress ONLY propagates through material-occupied
        cells, naturally blocking transmission through air/fracture gaps.
        """
        res = self.grid_res
        dx = self.grid_dx

        # Reset grid stress
        self.grid_stress[:] = 0.0

        # ---- Step 1: P2G — Deposit forces at loading points ----
        spread = 3  # half-width of deposit kernel (7x7x7)
        for dp in self.damage_points:
            # Map loading point to grid cell
            gi = int(np.clip(dp[0] / dx, 0, res - 1))
            gj = int(np.clip(dp[1] / dx, 0, res - 1))
            gk = int(np.clip(dp[2] / dx, 0, res - 1))

            # Deposit force with wide spread
            for di in range(-spread, spread + 1):
                for dj in range(-spread, spread + 1):
                    for dk in range(-spread, spread + 1):
                        ni, nj, nk = gi + di, gj + dj, gk + dk
                        if 0 <= ni < res and 0 <= nj < res and 0 <= nk < res:
                            if self.grid_occupancy[ni, nj, nk]:
                                r2 = di*di + dj*dj + dk*dk
                                w = np.exp(-r2 / (2.0 * spread))  # Gaussian
                                self.grid_stress[ni, nj, nk] += applied_force * w

        # ---- Step 2: Grid — Iterative stress diffusion ----
        # Poisson-like relaxation: stress spreads through material connectivity.
        # Higher iteration count = wider spread through bone geometry.
        n_iterations = 50
        source = self.grid_stress.copy()  # Save initial source
        for iteration in range(n_iterations):
            # Compute neighbor average (only through material)
            neighbor_avg = ndi.convolve(
                self.grid_stress,
                self._connectivity_kernel,
                mode='constant',
                cval=0.0
            )

            # Diffusion update: blend with neighbors + re-inject source
            alpha = 0.6  # strong diffusion for wide spread
            update_mask = self.grid_occupancy
            self.grid_stress[update_mask] = (
                (1 - alpha) * self.grid_stress[update_mask] +
                alpha * neighbor_avg[update_mask] +
                source[update_mask] * 0.10  # persistent source
            )

            # Zero out non-material cells
            self.grid_stress[~self.grid_occupancy] = 0.0

        # Normalize peak to match baseline magnitude
        grid_max = self.grid_stress.max()
        if grid_max > 0:
            target_peak = applied_force * 10.0
            self.grid_stress *= target_peak / grid_max

        # ---- Step 3: G2P — Gather stress from grid to particles ----
        self.stress[:] = 0.0
        for i in range(self.N):
            gi, gj, gk = self._particle_grid_idx[i]
            self.stress[i] = self.grid_stress[gi, gj, gk]

        # Apply stiffness degradation
        effective_stiffness = (1.0 - self.damage) ** 2
        self.stress *= effective_stiffness

    def compute_stress(self, applied_force: float):
        """Compute stress field using configured mode."""
        mode = getattr(self, 'stress_mode', 'baseline')
        if mode == 'grid':
            self._compute_stress_grid(applied_force)
        else:
            self._compute_stress_baseline(applied_force)

    # ================================================================
    #  DAMAGE EVOLUTION — Phase 3: Anisotropic threshold
    # ================================================================

    def enable_anisotropy(self, ratio: float = 2.0):
        """Enable anisotropic damage threshold.

        Trabecular bone is stronger in the vertical (z/SI) direction.
        Damage threshold is higher for vertical stress, lower for horizontal.

        Args:
            ratio: Vertical/horizontal strength ratio (default 2.0)
        """
        self._anisotropy_enabled = True
        self._anisotropy_ratio = ratio

        # Precompute per-particle threshold scale based on dominant
        # stress direction relative to vertical
        # Cortical shell is stronger in all directions
        cortical_mask = self.regions == 3
        self._threshold_scale = np.ones(self.N, dtype=np.float32)

        # Cortical: 50% stronger
        self._threshold_scale[cortical_mask] = 1.5

        # Endplate: 20% weaker (cancellous junction)
        endplate_mask = self.regions == 4
        self._threshold_scale[endplate_mask] = 0.8

    def evolve_damage(self):
        """Accumulate damage where stress exceeds threshold.

        Phase 3: Uses anisotropic threshold if enabled.
        """
        mat = self.material
        base_threshold = mat['sigma_ult'] * mat['damage_threshold']

        # Phase 3: Scale threshold per particle if anisotropy enabled
        if getattr(self, '_anisotropy_enabled', False):
            # Direction-dependent threshold: for each particle, compute
            # the angle between the load-to-particle vector and vertical
            threshold = np.full(self.N, base_threshold)
            for dp in self.damage_points:
                to_p = self.original_positions - dp
                dist = np.linalg.norm(to_p, axis=1)
                dist_safe = np.maximum(dist, 0.001)
                # Cosine of angle with vertical (z-axis)
                cos_angle = np.abs(to_p[:, 2]) / dist_safe
                # Vertical stress (cos≈1): stronger → higher threshold
                # Horizontal stress (cos≈0): weaker → lower threshold
                direction_scale = 1.0 + (self._anisotropy_ratio - 1.0) * cos_angle
                threshold = np.minimum(threshold, base_threshold * direction_scale)

            # Apply cortical/endplate scaling
            threshold *= self._threshold_scale
        else:
            threshold = base_threshold

        overstressed = self.stress > threshold
        if not np.any(overstressed):
            return

        if isinstance(threshold, np.ndarray):
            overstress = (self.stress[overstressed] - threshold[overstressed]) / mat['sigma_ult']
        else:
            overstress = (self.stress[overstressed] - threshold) / mat['sigma_ult']

        increment = overstress * mat['damage_rate']
        self.damage[overstressed] = np.minimum(
            self.damage[overstressed] + increment, 1.0
        )

    # ================================================================
    #  FRAGMENT DETECTION — Phase 4
    # ================================================================

    def detect_fragments(self, damage_threshold: float = 0.9) -> Dict:
        """Identify separate fragments using connected components.

        Projects particle damage to a 3D grid, binarizes (intact vs broken),
        then uses scipy.ndimage.label to find connected regions.

        Args:
            damage_threshold: Damage level to consider as fully broken

        Returns:
            Dict with fragment_ids per particle, n_fragments, fragment_sizes
        """
        res = 48  # grid resolution for fragment detection
        dx = 1.0 / res

        # Build intact-material grid
        intact_grid = np.zeros((res, res, res), dtype=bool)
        p_idx = np.floor(self.original_positions / dx).astype(int)
        p_idx = np.clip(p_idx, 0, res - 1)

        for i in range(self.N):
            gi, gj, gk = p_idx[i]
            if self.damage[i] < damage_threshold:
                intact_grid[gi, gj, gk] = True

        # Find connected components in intact material
        structure = ndi.generate_binary_structure(3, 1)  # 6-connected
        labeled, n_fragments = ndi.label(intact_grid, structure=structure)

        # Map fragments back to particles
        fragment_ids = np.zeros(self.N, dtype=np.int32)
        for i in range(self.N):
            gi, gj, gk = p_idx[i]
            fragment_ids[i] = labeled[gi, gj, gk]

        # Compute fragment sizes
        fragment_sizes = {}
        for fid in range(1, n_fragments + 1):
            fragment_sizes[fid] = int((fragment_ids == fid).sum())

        self._fragment_ids = fragment_ids
        self._n_fragments = n_fragments

        return {
            'fragment_ids': fragment_ids,
            'n_fragments': n_fragments,
            'fragment_sizes': fragment_sizes,
        }

    # ================================================================
    #  DEFORMATION — Phase 4: Fragment-aware
    # ================================================================

    def compute_deformation(self):
        """Compute particle displacements based on damage state.

        Phase 4: Includes fragment-based rigid body displacement
        for separated bone fragments.
        """
        mat = self.material

        for dp in self.damage_points:
            to_particle = self.original_positions - dp
            dist = np.linalg.norm(to_particle, axis=1, keepdims=True)
            dist_safe = np.maximum(dist, 0.001)
            direction = to_particle / dist_safe

            # High damage: crack opening displacement
            high_dmg = self.damage > mat['cod_threshold']
            cod = (self.damage[high_dmg] - mat['cod_threshold']) * mat['cod_magnitude']
            self.positions[high_dmg] = (
                self.original_positions[high_dmg] + direction[high_dmg] * cod[:, None]
            )

            # Low damage: elastic compression toward load
            low_dmg = ~high_dmg
            elastic_strain = self.stress[low_dmg] * 0.001 * (1.0 - self.damage[low_dmg])
            self.positions[low_dmg] = (
                self.original_positions[low_dmg]
                - direction[low_dmg] * (elastic_strain[:, None] * 0.1)
            )

        # Phase 4: Apply fragment separation if fragments detected
        if hasattr(self, '_fragment_ids') and self._n_fragments > 1:
            self._apply_fragment_displacement()

    def _apply_fragment_displacement(self):
        """Apply rigid body displacements to separated fragments.

        Small fragments get pushed away from the main body.
        Posterior fragments get retropulsion toward canal.
        """
        if self._n_fragments <= 1:
            return

        # Find largest fragment (main body)
        frag_sizes = {}
        for fid in range(1, self._n_fragments + 1):
            frag_sizes[fid] = (self._fragment_ids == fid).sum()

        main_frag = max(frag_sizes, key=frag_sizes.get)
        main_center = self.positions[self._fragment_ids == main_frag].mean(axis=0)

        for fid in range(1, self._n_fragments + 1):
            if fid == main_frag:
                continue

            mask = self._fragment_ids == fid
            if not mask.any():
                continue

            frag_center = self.positions[mask].mean(axis=0)
            frag_size_ratio = frag_sizes[fid] / frag_sizes[main_frag]

            # Direction from main body center to fragment center
            separation_dir = frag_center - main_center
            sep_norm = np.linalg.norm(separation_dir)
            if sep_norm > 0.001:
                separation_dir /= sep_norm

            # Displacement magnitude: smaller fragments move more
            displacement = 0.01 * (1.0 - frag_size_ratio) * self.material.get('cod_magnitude', 0.02) * 10

            # Is this a posterior fragment? Apply retropulsion
            if frag_center[1] < main_center[1] - 0.02:  # posterior
                retropulsion = np.array([0.0, -0.01, 0.0], dtype=np.float32)
                self.positions[mask] += retropulsion

            self.positions[mask] += separation_dir * displacement

    # ================================================================
    #  SIMULATION LOOP
    # ================================================================

    def step(self, applied_force: float):
        """Execute one simulation step."""
        self.compute_stress(applied_force)
        self.evolve_damage()
        self.compute_deformation()
        self.step_count += 1

    def run(
        self,
        n_steps: int = 200,
        max_force: float = 3.0,
        record_every: int = 5,
        verbose: bool = True,
    ) -> List[Dict]:
        """Run full simulation with gradual force ramp.

        Args:
            n_steps: Number of simulation steps
            max_force: Maximum applied force
            record_every: Record state every N steps
            verbose: Print progress

        Returns:
            history: List of state snapshots
        """
        self.history = []
        force_ramp = np.linspace(0, max_force, n_steps)

        for i in range(n_steps):
            self.step(force_ramp[i])

            if i % record_every == 0 or i == n_steps - 1:
                snapshot = {
                    'step': i,
                    'force': force_ramp[i],
                    'damage': self.damage.copy(),
                    'stress': self.stress.copy(),
                    'positions': self.positions.copy(),
                    'max_damage': self.damage.max(),
                    'mean_damage': self.damage.mean(),
                    'damaged_frac': (self.damage > 0.5).sum() / self.N,
                    'fractured_frac': (self.damage > 0.8).sum() / self.N,
                }
                self.history.append(snapshot)

                if verbose and i % (record_every * 5) == 0:
                    print(
                        f"  Step {i:4d}/{n_steps} | "
                        f"Force={force_ramp[i]:.2f} | "
                        f"MaxD={snapshot['max_damage']:.3f} | "
                        f"Damaged={snapshot['damaged_frac']*100:.1f}% | "
                        f"Fractured={snapshot['fractured_frac']*100:.1f}%"
                    )

        return self.history

    # ================================================================
    #  STATISTICS
    # ================================================================

    def get_damage_by_region(self) -> Dict[str, float]:
        """Get mean damage per anatomical region."""
        region_names = [
            'anterior_body', 'central_body', 'posterior_body',
            'cortical_shell', 'endplate', 'pedicle', 'lamina',
        ]
        result = {}
        for rid, name in enumerate(region_names):
            mask = self.regions == rid
            if mask.any():
                result[name] = float(self.damage[mask].mean())
            else:
                result[name] = 0.0
        return result


# ============================================================================
#  VISUALIZATION
# ============================================================================

def create_2d_projection(
    positions: np.ndarray,
    damage: np.ndarray,
    axis: int = 2,
    resolution: int = 128,
) -> np.ndarray:
    """Project 3D particle data to 2D grid for visualization.

    Args:
        positions: [N, 3] particle positions
        damage: [N] damage values
        axis: Projection axis (0=sagittal, 1=coronal, 2=axial)
        resolution: Grid resolution

    Returns:
        grid: [resolution, resolution] damage heatmap
    """
    axes = [i for i in range(3) if i != axis]
    ax0, ax1 = axes

    grid_damage = np.zeros((resolution, resolution), dtype=np.float32)
    grid_count = np.zeros((resolution, resolution), dtype=np.float32)

    # Map positions to grid indices
    coords_0 = ((positions[:, ax0] - 0.2) / 0.6 * resolution).astype(int)
    coords_1 = ((positions[:, ax1] - 0.2) / 0.6 * resolution).astype(int)

    # Clip to valid range
    coords_0 = np.clip(coords_0, 0, resolution - 1)
    coords_1 = np.clip(coords_1, 0, resolution - 1)

    # Accumulate
    np.add.at(grid_damage, (coords_0, coords_1), damage)
    np.add.at(grid_count, (coords_0, coords_1), 1.0)

    # Average
    valid = grid_count > 0
    grid_damage[valid] /= grid_count[valid]

    return grid_damage


def generate_baseline_gif(
    history: List[Dict],
    output_path: str,
    positions_orig: np.ndarray,
    regions: np.ndarray,
    fps: int = 10,
):
    """Generate GIF showing damage propagation over time.

    Creates 3-panel view: axial, sagittal, and damage-by-region curve.
    """
    if not HAS_MPL or not HAS_IMAGEIO:
        print("Warning: matplotlib or imageio not available, skipping GIF")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#0a0a12')
    frames = []

    region_names = [
        'Anterior', 'Central', 'Posterior',
        'Cortical', 'Endplate', 'Pedicle', 'Lamina',
    ]
    region_colors = ['#ff6b6b', '#feca57', '#48dbfb', '#ff9ff3',
                     '#54a0ff', '#5f27cd', '#01a3a4']

    # Precompute damage-by-region time series
    n_frames = len(history)
    n_regions = 7
    region_ts = np.zeros((n_frames, n_regions))
    for fi, snap in enumerate(history):
        for rid in range(n_regions):
            mask = regions == rid
            if mask.any():
                region_ts[fi, rid] = snap['damage'][mask].mean()

    steps = [s['step'] for s in history]

    for fi, snap in enumerate(history):
        for ax in axes:
            ax.clear()

        # --- Panel 1: Axial view (top-down, axis=2) ---
        grid_ax = create_2d_projection(positions_orig, snap['damage'], axis=2)
        axes[0].imshow(
            grid_ax.T, origin='lower', cmap=DAMAGE_CMAP,
            vmin=0, vmax=1, aspect='equal'
        )
        axes[0].set_title('Axial View (Top-Down)', color='white', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Left ← → Right', color='#888', fontsize=9)
        axes[0].set_ylabel('Posterior ← → Anterior', color='#888', fontsize=9)
        axes[0].tick_params(colors='#555')

        # --- Panel 2: Sagittal view (side, axis=0) ---
        grid_sag = create_2d_projection(positions_orig, snap['damage'], axis=0)
        axes[1].imshow(
            grid_sag.T, origin='lower', cmap=DAMAGE_CMAP,
            vmin=0, vmax=1, aspect='equal'
        )
        axes[1].set_title('Sagittal View (Side)', color='white', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Posterior ← → Anterior', color='#888', fontsize=9)
        axes[1].set_ylabel('Inferior ← → Superior', color='#888', fontsize=9)
        axes[1].tick_params(colors='#555')

        # --- Panel 3: Damage curves by region ---
        axes[2].set_facecolor('#0d0d1a')
        for rid in range(n_regions):
            axes[2].plot(
                steps[:fi + 1], region_ts[:fi + 1, rid],
                color=region_colors[rid], linewidth=2, label=region_names[rid]
            )
        axes[2].set_xlim(0, steps[-1])
        axes[2].set_ylim(0, 1.05)
        axes[2].set_xlabel('Simulation Step', color='#aaa', fontsize=10)
        axes[2].set_ylabel('Mean Damage', color='#aaa', fontsize=10)
        axes[2].set_title('Damage by Region', color='white', fontsize=12, fontweight='bold')
        axes[2].legend(loc='upper left', fontsize=7, facecolor='#1a1a2e',
                       edgecolor='#333', labelcolor='white')
        axes[2].tick_params(colors='#555')
        for spine in axes[2].spines.values():
            spine.set_color('#333')

        # Suptitle
        fig.suptitle(
            f'Phase 0: Baseline — Step {snap["step"]} | '
            f'Force={snap["force"]:.2f} | '
            f'Max Damage={snap["max_damage"]:.3f} | '
            f'Fractured={snap["fractured_frac"]*100:.1f}%',
            color='white', fontsize=13, fontweight='bold', y=0.98
        )

        fig.tight_layout(rect=[0, 0, 1, 0.94])

        # Render to image
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3].copy()  # RGBA → RGB
        frames.append(img)

    plt.close(fig)

    # Write GIF
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"  ✅ GIF saved: {output_path} ({len(frames)} frames)")


def generate_cascade_plot(
    history: List[Dict],
    regions: np.ndarray,
    output_path: str,
):
    """Generate cascade-style damage plot (user's favorite)."""
    if not HAS_MPL:
        return

    region_names = [
        'Anterior\nBody', 'Central\nBody', 'Posterior\nBody',
        'Cortical\nShell', 'Endplate', 'Pedicle', 'Lamina',
    ]
    n_regions = len(region_names)
    n_frames = len(history)

    # Build damage matrix [time × region]
    damage_matrix = np.zeros((n_frames, n_regions))
    for fi, snap in enumerate(history):
        for rid in range(n_regions):
            mask = regions == rid
            if mask.any():
                damage_matrix[fi, rid] = snap['damage'][mask].mean()

    steps = [s['step'] for s in history]

    fig, (ax_heat, ax_lines) = plt.subplots(
        1, 2, figsize=(16, 7),
        gridspec_kw={'width_ratios': [2, 1], 'wspace': 0.05}
    )
    fig.patch.set_facecolor('#0a0a12')

    # --- Left: Heatmap ---
    ax_heat.set_facecolor('#0d0d1a')
    im = ax_heat.imshow(
        damage_matrix, aspect='auto', cmap=DAMAGE_CMAP,
        vmin=0, vmax=1, origin='lower',
        extent=[0, n_regions, steps[0], steps[-1]]
    )
    ax_heat.set_xticks(np.arange(n_regions) + 0.5)
    ax_heat.set_xticklabels(region_names, fontsize=9, color='white')
    ax_heat.set_ylabel('Simulation Step', color='white', fontsize=11)
    ax_heat.set_title('Damage Cascade — Baseline (1/r²)',
                      color='white', fontsize=13, fontweight='bold')
    ax_heat.tick_params(colors='#888')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_heat, pad=0.02, shrink=0.9)
    cbar.set_label('Damage Level', color='white', fontsize=10)
    cbar.ax.tick_params(colors='#888')

    # --- Right: Line plots ---
    ax_lines.set_facecolor('#0d0d1a')
    colors = ['#ff6b6b', '#feca57', '#48dbfb', '#ff9ff3',
              '#54a0ff', '#5f27cd', '#01a3a4']
    for rid in range(n_regions):
        ax_lines.plot(
            damage_matrix[:, rid], steps,
            color=colors[rid], linewidth=2,
            label=region_names[rid].replace('\n', ' ')
        )
    ax_lines.set_xlim(0, 1.05)
    ax_lines.set_ylim(steps[0], steps[-1])
    ax_lines.set_xlabel('Mean Damage', color='white', fontsize=11)
    ax_lines.set_yticklabels([])
    ax_lines.legend(loc='lower right', fontsize=7, facecolor='#1a1a2e',
                    edgecolor='#333', labelcolor='white')
    ax_lines.tick_params(colors='#888')
    for spine in ax_lines.spines.values():
        spine.set_color('#333')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Cascade plot saved: {output_path}")


def generate_stress_snapshot(
    positions: np.ndarray,
    stress: np.ndarray,
    damage: np.ndarray,
    damage_points: List[np.ndarray],
    output_path: str,
):
    """Generate stress + damage field snapshot (2D projections)."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0a0a12')

    # Stress field
    grid_stress = create_2d_projection(positions, stress, axis=2)
    axes[0].imshow(
        grid_stress.T, origin='lower', cmap='inferno', aspect='equal'
    )
    axes[0].set_title('Stress Field (Axial)', color='white', fontsize=12, fontweight='bold')
    # Mark damage points
    for dp in damage_points:
        px = int((dp[0] - 0.2) / 0.6 * 128)
        py = int((dp[1] - 0.2) / 0.6 * 128)
        axes[0].plot(px, py, 'w*', markersize=15, markeredgecolor='red', markeredgewidth=1)
    axes[0].tick_params(colors='#555')

    # Damage field
    grid_damage = create_2d_projection(positions, damage, axis=2)
    axes[1].imshow(
        grid_damage.T, origin='lower', cmap=DAMAGE_CMAP,
        vmin=0, vmax=1, aspect='equal'
    )
    axes[1].set_title('Damage Field (Axial)', color='white', fontsize=12, fontweight='bold')
    axes[1].tick_params(colors='#555')

    fig.suptitle('Phase 0 Baseline — Final State',
                 color='white', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Snapshot saved: {output_path}")


# ============================================================================
#  TEST / DEMO FUNCTIONS
# ============================================================================

def test_baseline(output_dir: str = None):
    """Run baseline test: single-point 1/r² loading."""
    print("=" * 70)
    print("Phase 0: Baseline Fracture Simulation (1/r² Point-Load)")
    print("=" * 70)

    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent.parent / 'docs' / 'fracture_reports' / 'figures')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate particles
    print("\n1. Generating vertebra particles...")
    positions = generate_vertebra_particles(n_particles=50000, seed=42)
    regions = classify_regions(positions)
    print(f"   Particles: {len(positions)}")
    for rid, name in enumerate(['anterior', 'central', 'posterior',
                                 'cortical', 'endplate', 'pedicle', 'lamina']):
        count = (regions == rid).sum()
        print(f"   {name:>12}: {count:6d} ({count/len(positions)*100:.1f}%)")

    # Setup simulator
    print("\n2. Setting up simulator...")
    sim = BoneFractureSimulator(positions, seed=42)

    # Damage point: anterior-superior (A1-like loading)
    center = positions.mean(axis=0)
    bounds_min = positions.min(axis=0)
    bounds_max = positions.max(axis=0)

    dp = np.array([
        center[0] + (bounds_max[0] - center[0]) * 0.3,
        center[1] + (bounds_max[1] - center[1]) * 0.5,  # anterior
        center[2] + (bounds_max[2] - center[2]) * 0.3,   # superior
    ])
    sim.set_loading([dp])
    print(f"   Damage point: {dp}")

    # Run simulation
    print("\n3. Running simulation (200 steps, force 0→3.0)...")
    history = sim.run(n_steps=200, max_force=3.0, record_every=4)

    # Final stats
    print(f"\n4. Final statistics:")
    region_damage = sim.get_damage_by_region()
    for name, dmg in region_damage.items():
        bar = '█' * int(dmg * 30)
        print(f"   {name:>16}: {dmg:.3f} [{bar}]")

    # Generate visualizations
    print("\n5. Generating visualizations...")

    gif_path = str(Path(output_dir) / 'phase0_baseline.gif')
    generate_baseline_gif(history, gif_path, positions, regions, fps=8)

    cascade_path = str(Path(output_dir) / 'phase0_cascade.png')
    generate_cascade_plot(history, regions, cascade_path)

    snapshot_path = str(Path(output_dir) / 'phase0_snapshot.png')
    generate_stress_snapshot(positions, sim.stress, sim.damage, [dp], snapshot_path)

    print("\n" + "=" * 70)
    print("✅ Phase 0 baseline complete!")
    print(f"   Output: {output_dir}")
    print("=" * 70)

    return sim, history


def test_grid_stress(output_dir: str = None):
    """Phase 1: Compare 1/r² vs grid-based stress transfer."""
    print("=" * 70)
    print("Phase 1: Grid-Based Stress Transfer (P2G/G2P)")
    print("=" * 70)

    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent.parent / 'docs' / 'fracture_reports' / 'figures')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate particles
    print("\n1. Generating vertebra particles...")
    positions = generate_vertebra_particles(n_particles=50000, seed=42)
    regions = classify_regions(positions)
    print(f"   Particles: {len(positions)}")

    center = positions.mean(axis=0)
    bounds_max = positions.max(axis=0)
    dp = np.array([
        center[0] + (bounds_max[0] - center[0]) * 0.3,
        center[1] + (bounds_max[1] - center[1]) * 0.5,
        center[2] + (bounds_max[2] - center[2]) * 0.3,
    ])

    # --- Run BASELINE ---
    print("\n2a. Running BASELINE (1/r²)...")
    sim_base = BoneFractureSimulator(positions.copy(), seed=42)
    sim_base.set_loading([dp])
    sim_base.set_stress_mode('baseline')
    history_base = sim_base.run(n_steps=200, max_force=3.0, record_every=4)

    # --- Run GRID ---
    print("\n2b. Running GRID-BASED (P2G/G2P)...")
    sim_grid = BoneFractureSimulator(positions.copy(), seed=42)
    sim_grid.set_loading([dp])
    sim_grid.set_stress_mode('grid')
    history_grid = sim_grid.run(n_steps=200, max_force=3.0, record_every=4)

    # --- Compare stats ---
    print("\n3. Comparison:")
    print(f"   {'Region':<18} {'Baseline':>10} {'Grid':>10} {'Δ':>10}")
    print("   " + "-" * 50)
    base_dmg = sim_base.get_damage_by_region()
    grid_dmg = sim_grid.get_damage_by_region()
    for name in base_dmg:
        b = base_dmg[name]
        g = grid_dmg[name]
        delta = g - b
        print(f"   {name:<18} {b:10.3f} {g:10.3f} {delta:+10.3f}")

    # --- Generate comparison GIF ---
    print("\n4. Generating comparison visualizations...")
    gif_path = str(Path(output_dir) / 'phase1_comparison.gif')
    _generate_comparison_gif(
        history_base, history_grid,
        positions, regions, gif_path, fps=8
    )

    # --- Generate comparison cascade ---
    cascade_path = str(Path(output_dir) / 'phase1_cascade_comparison.png')
    _generate_comparison_cascade(
        history_base, history_grid,
        regions, cascade_path
    )

    print("\n" + "=" * 70)
    print("✅ Phase 1 complete!")
    print(f"   Output: {output_dir}")
    print("=" * 70)

    return sim_base, sim_grid


def _generate_comparison_gif(
    history_base, history_grid,
    positions, regions,
    output_path, fps=8,
):
    """Side-by-side GIF: baseline (left) vs grid (right)."""
    if not HAS_MPL or not HAS_IMAGEIO:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor('#0a0a12')
    frames = []

    n_frames = min(len(history_base), len(history_grid))

    for fi in range(n_frames):
        snap_b = history_base[fi]
        snap_g = history_grid[fi]

        for row in axes:
            for ax in row:
                ax.clear()

        # Top-left: Baseline axial
        grid_b = create_2d_projection(positions, snap_b['damage'], axis=2)
        axes[0, 0].imshow(grid_b.T, origin='lower', cmap=DAMAGE_CMAP, vmin=0, vmax=1)
        axes[0, 0].set_title('Baseline (1/r²) — Axial', color='white', fontsize=11, fontweight='bold')
        axes[0, 0].tick_params(colors='#555')

        # Top-right: Grid axial
        grid_g = create_2d_projection(positions, snap_g['damage'], axis=2)
        axes[0, 1].imshow(grid_g.T, origin='lower', cmap=DAMAGE_CMAP, vmin=0, vmax=1)
        axes[0, 1].set_title('Grid P2G/G2P — Axial', color='white', fontsize=11, fontweight='bold')
        axes[0, 1].tick_params(colors='#555')

        # Bottom-left: Baseline sagittal
        grid_bs = create_2d_projection(positions, snap_b['damage'], axis=0)
        axes[1, 0].imshow(grid_bs.T, origin='lower', cmap=DAMAGE_CMAP, vmin=0, vmax=1)
        axes[1, 0].set_title('Baseline — Sagittal', color='white', fontsize=11, fontweight='bold')
        axes[1, 0].tick_params(colors='#555')

        # Bottom-right: Grid sagittal
        grid_gs = create_2d_projection(positions, snap_g['damage'], axis=0)
        axes[1, 1].imshow(grid_gs.T, origin='lower', cmap=DAMAGE_CMAP, vmin=0, vmax=1)
        axes[1, 1].set_title('Grid — Sagittal', color='white', fontsize=11, fontweight='bold')
        axes[1, 1].tick_params(colors='#555')

        fig.suptitle(
            f'Phase 1: 1/r² vs Grid Stress — Step {snap_b["step"]} | '
            f'Force={snap_b["force"]:.2f}\n'
            f'Baseline: {snap_b["damaged_frac"]*100:.1f}% damaged | '
            f'Grid: {snap_g["damaged_frac"]*100:.1f}% damaged',
            color='white', fontsize=12, fontweight='bold', y=0.98
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3].copy()
        frames.append(img)

    plt.close(fig)
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"  ✅ Comparison GIF saved: {output_path} ({len(frames)} frames)")


def _generate_comparison_cascade(
    history_base, history_grid,
    regions, output_path,
):
    """Side-by-side cascade plots: baseline vs grid."""
    if not HAS_MPL:
        return

    region_names = [
        'Anterior\nBody', 'Central\nBody', 'Posterior\nBody',
        'Cortical\nShell', 'Endplate', 'Pedicle', 'Lamina',
    ]
    n_regions = len(region_names)
    colors = ['#ff6b6b', '#feca57', '#48dbfb', '#ff9ff3',
              '#54a0ff', '#5f27cd', '#01a3a4']

    fig, (ax_b, ax_g) = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor('#0a0a12')

    for label, history, ax in [
        ('Baseline (1/r²)', history_base, ax_b),
        ('Grid (P2G/G2P)', history_grid, ax_g),
    ]:
        n_frames = len(history)
        damage_matrix = np.zeros((n_frames, n_regions))
        for fi, snap in enumerate(history):
            for rid in range(n_regions):
                mask = regions == rid
                if mask.any():
                    damage_matrix[fi, rid] = snap['damage'][mask].mean()

        steps = [s['step'] for s in history]

        ax.set_facecolor('#0d0d1a')
        im = ax.imshow(
            damage_matrix, aspect='auto', cmap=DAMAGE_CMAP,
            vmin=0, vmax=1, origin='lower',
            extent=[0, n_regions, steps[0], steps[-1]]
        )
        ax.set_xticks(np.arange(n_regions) + 0.5)
        ax.set_xticklabels(region_names, fontsize=8, color='white')
        ax.set_ylabel('Simulation Step', color='white', fontsize=10)
        ax.set_title(f'Damage Cascade — {label}',
                     color='white', fontsize=12, fontweight='bold')
        ax.tick_params(colors='#888')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Cascade comparison saved: {output_path}")

def test_ao_types(output_dir: str = None):
    """Phase 2: Compare all 4 AO fracture types."""
    print("=" * 70)
    print("Phase 2: AO-Type-Specific Loading (A1-A4)")
    print("=" * 70)

    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent.parent / 'docs' / 'fracture_reports' / 'figures')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    positions = generate_vertebra_particles(n_particles=50000, seed=42)
    regions = classify_regions(positions)

    histories = {}
    sims = {}

    for ao_type in ['A1', 'A2', 'A3', 'A4']:
        config = AO_LOAD_CONFIGS[ao_type]
        print(f"\n--- {config['name']} ---")
        print(f"    Mechanism: {config['mechanism']}")

        sim = BoneFractureSimulator(positions.copy(), seed=42)
        sim.setup_ao_loading(ao_type)
        sim.set_stress_mode('grid')
        history = sim.run(
            n_steps=200,
            max_force=config['max_force'],
            record_every=4,
            verbose=True,
        )
        histories[ao_type] = history
        sims[ao_type] = sim

        # Print region damage
        dmg = sim.get_damage_by_region()
        print(f"    Final damage: " + " | ".join(f"{k}={v:.3f}" for k, v in dmg.items()))

    # --- Generate 4-panel comparison GIF ---
    print("\nGenerating 4-panel AO comparison GIF...")
    gif_path = str(Path(output_dir) / 'phase2_ao_comparison.gif')
    _generate_ao_comparison_gif(histories, positions, regions, gif_path, fps=8)

    # --- Generate 4-panel cascade ---
    print("Generating 4-panel AO cascade comparison...")
    cascade_path = str(Path(output_dir) / 'phase2_ao_cascade.png')
    _generate_ao_cascade(histories, regions, cascade_path)

    print("\n" + "=" * 70)
    print("✅ Phase 2 complete!")
    print(f"   Output: {output_dir}")
    print("=" * 70)

    return sims, histories


def _generate_ao_comparison_gif(
    histories, positions, regions, output_path, fps=8,
):
    """4-panel GIF: A1-A4 side by side."""
    if not HAS_MPL or not HAS_IMAGEIO:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor('#0a0a12')
    frames = []

    ao_types = ['A1', 'A2', 'A3', 'A4']
    n_frames = min(len(h) for h in histories.values())

    for fi in range(n_frames):
        for i, ao_type in enumerate(ao_types):
            row, col = divmod(i, 2)
            ax = axes[row, col]
            ax.clear()

            snap = histories[ao_type][fi]
            grid = create_2d_projection(positions, snap['damage'], axis=2)
            ax.imshow(grid.T, origin='lower', cmap=DAMAGE_CMAP, vmin=0, vmax=1)

            config = AO_LOAD_CONFIGS[ao_type]
            ax.set_title(
                f'{config["name"]}\n'
                f'Damaged: {snap["damaged_frac"]*100:.1f}% | '
                f'Max: {snap["max_damage"]:.2f}',
                color='white', fontsize=10, fontweight='bold'
            )
            ax.tick_params(colors='#555')

        step = histories['A1'][fi]['step']
        force = histories['A1'][fi]['force']
        fig.suptitle(
            f'Phase 2: AO Fracture Type Comparison — Step {step} | Force={force:.2f}',
            color='white', fontsize=13, fontweight='bold', y=0.98
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3].copy()
        frames.append(img)

    plt.close(fig)
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"  ✅ AO comparison GIF saved: {output_path} ({len(frames)} frames)")


def _generate_ao_cascade(histories, regions, output_path):
    """4-panel cascade: one per AO type."""
    if not HAS_MPL:
        return

    region_names = [
        'Anterior\nBody', 'Central\nBody', 'Posterior\nBody',
        'Cortical\nShell', 'Endplate', 'Pedicle', 'Lamina',
    ]
    n_regions = len(region_names)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.patch.set_facecolor('#0a0a12')

    ao_types = ['A1', 'A2', 'A3', 'A4']
    for i, ao_type in enumerate(ao_types):
        ax = axes[i]
        history = histories[ao_type]
        n_frames = len(history)

        damage_matrix = np.zeros((n_frames, n_regions))
        for fi, snap in enumerate(history):
            for rid in range(n_regions):
                mask = regions == rid
                if mask.any():
                    damage_matrix[fi, rid] = snap['damage'][mask].mean()

        steps = [s['step'] for s in history]

        ax.set_facecolor('#0d0d1a')
        ax.imshow(
            damage_matrix, aspect='auto', cmap=DAMAGE_CMAP,
            vmin=0, vmax=1, origin='lower',
            extent=[0, n_regions, steps[0], steps[-1]]
        )
        ax.set_xticks(np.arange(n_regions) + 0.5)
        ax.set_xticklabels(region_names, fontsize=7, color='white', rotation=45, ha='right')
        if i == 0:
            ax.set_ylabel('Simulation Step', color='white', fontsize=10)
        config = AO_LOAD_CONFIGS[ao_type]
        ax.set_title(config['name'], color='white', fontsize=11, fontweight='bold')
        ax.tick_params(colors='#888')

    fig.suptitle('Damage Cascades by AO Fracture Type',
                 color='white', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ AO cascade saved: {output_path}")

def test_full_physics(output_dir: str = None):
    """Phase 5: Full physics — AO types with anisotropy + fragment detection."""
    print("=" * 70)
    print("Phase 5: Full Physics (Anisotropy + Fragments)")
    print("=" * 70)

    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent.parent / 'docs' / 'fracture_reports' / 'figures')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    positions = generate_vertebra_particles(n_particles=50000, seed=42)
    regions = classify_regions(positions)

    histories = {}
    fragment_info = {}

    for ao_type in ['A1', 'A2', 'A3', 'A4']:
        config = AO_LOAD_CONFIGS[ao_type]
        print(f"\n--- {config['name']} (Full Physics) ---")

        sim = BoneFractureSimulator(positions.copy(), seed=42)
        sim.setup_ao_loading(ao_type)
        sim.set_stress_mode('grid')
        sim.enable_anisotropy(ratio=2.0)

        history = sim.run(
            n_steps=200,
            max_force=config['max_force'],
            record_every=4,
            verbose=True,
        )

        # Detect fragments
        frag_result = sim.detect_fragments(damage_threshold=0.9)
        print(f"    Fragments: {frag_result['n_fragments']}")
        if frag_result['fragment_sizes']:
            top3 = sorted(frag_result['fragment_sizes'].items(),
                          key=lambda x: x[1], reverse=True)[:3]
            for fid, size in top3:
                print(f"      Fragment {fid}: {size} particles")

        histories[ao_type] = history
        fragment_info[ao_type] = frag_result

    # 4-panel GIF
    print("\nGenerating full-physics AO comparison GIF...")
    gif_path = str(Path(output_dir) / 'phase5_full_physics.gif')
    _generate_ao_comparison_gif(histories, positions, regions, gif_path, fps=8)

    # 4-panel cascade
    print("Generating full-physics cascade...")
    cascade_path = str(Path(output_dir) / 'phase5_cascade.png')
    _generate_ao_cascade(histories, regions, cascade_path)

    # Summary stats
    print("\n" + "=" * 70)
    print("Phase 5 Summary: AO Types with Full Physics")
    print("-" * 70)
    print(f"  {'Type':<25} {'Damaged%':>10} {'Fractured%':>12} {'Fragments':>10}")
    print("  " + "-" * 60)
    for ao_type in ['A1', 'A2', 'A3', 'A4']:
        h = histories[ao_type][-1]
        n_frag = fragment_info[ao_type]['n_fragments']
        print(f"  {AO_LOAD_CONFIGS[ao_type]['name']:<25} "
              f"{h['damaged_frac']*100:>9.1f}% "
              f"{h['fractured_frac']*100:>11.1f}% "
              f"{n_frag:>10d}")

    print("\n" + "=" * 70)
    print("✅ Phase 5 complete!")
    print(f"   Output: {output_dir}")
    print("=" * 70)

    return histories, fragment_info


# ============================================================================
#  MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bone Fracture Simulator v2')
    parser.add_argument('--test-baseline', action='store_true',
                        help='Run Phase 0 baseline test')
    parser.add_argument('--test-grid-stress', action='store_true',
                        help='Run Phase 1 grid stress comparison')
    parser.add_argument('--test-ao-types', action='store_true',
                        help='Run Phase 2 AO type comparison')
    parser.add_argument('--test-full-physics', action='store_true',
                        help='Run Phase 5 full physics')
    parser.add_argument('--test-all', action='store_true',
                        help='Run all implemented phases')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for figures')
    args = parser.parse_args()

    if args.test_baseline:
        test_baseline(args.output_dir)
    elif args.test_grid_stress:
        test_grid_stress(args.output_dir)
    elif args.test_ao_types:
        test_ao_types(args.output_dir)
    elif args.test_full_physics:
        test_full_physics(args.output_dir)
    elif args.test_all:
        test_baseline(args.output_dir)
        test_grid_stress(args.output_dir)
        test_ao_types(args.output_dir)
        test_full_physics(args.output_dir)
    elif len(sys.argv) == 1:
        test_full_physics()
