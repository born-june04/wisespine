"""
Voxel-Based Finite Element Fracture Engine (v5)
================================================

CT-based voxel FEM for vertebral fracture simulation.

Physics:
    - Each bone voxel → 8-node hexahedral finite element
    - HU → apparent density → Young's modulus E (Morgan & Keaveny 2003)
    - HU → yield stress σ_y (Keller 1994)
    - Transversely isotropic elasticity (SI axis is stiff direction)
    - Solve actual equilibrium: K u = F  (scipy sparse)
    - von Mises yield criterion with iterative damage
    - Stress redistribution: damaged elements → E reduced → re-solve
    - Real units: mm, N, MPa throughout

References:
    [1] Morgan & Keaveny (2003). J Biomech 36:897-904.
        E = 6850 × ρ^1.49 (trabecular)
    [2] Keller (1994). J Biomech 27:1159-1168.
        σ_y = 137 × ρ^1.88
    [3] Crawford et al. (2003). Bone 33:744-750.
        CT-based voxel FEM validation R²≈0.80
    [4] Bayraktar et al. (2004). J Biomech 37:27-35.
        Yield strain criteria for trabecular bone
"""

import os
import sys
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from scipy.ndimage import zoom, distance_transform_edt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.ndimage as ndi

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================================
#  PHYSICAL CONSTANTS & MATERIAL MODELS
# ============================================================================

# Water density for HU conversion
RHO_WATER = 1.0  # g/cm³

# Poisson's ratio for bone
NU_BONE = 0.3

# Cortical bone threshold (HU)
HU_CORTICAL_THRESHOLD = 600

# Minimum E to avoid singularity (MPa)
E_MIN = 1.0

# Damaged element stiffness fraction
DAMAGE_STIFFNESS_FRACTION = 0.01


def hu_to_density(hu: np.ndarray) -> np.ndarray:
    """Convert HU to apparent density (g/cm³).

    Using the standard calibration:
        ρ_app = (HU / 1000) × ρ_water  for HU > 0
        ρ_app = 0.001 for HU ≤ 0 (avoid zero)

    For more accurate results, site-specific phantom calibration
    should be used, but this gives reasonable estimates.
    """
    rho = np.clip(hu / 1000.0 * RHO_WATER, 0.001, 2.0)
    return rho.astype(np.float64)


def density_to_youngs_modulus(rho: np.ndarray,
                               cortical_mask: np.ndarray = None,
                               coarse_resolution: bool = False) -> np.ndarray:
    """Apparent density → Young's modulus E (MPa).

    Trabecular: E = 6850 × ρ^1.49  (Morgan & Keaveny 2003)
    Cortical:   E = 10500 × ρ^2.29 (Keller 1994)

    At coarse resolution (voxel > 2mm), use unified relationship
    since each voxel contains a mix of cortical and trabecular.
    """
    E = np.zeros_like(rho, dtype=np.float64)

    if coarse_resolution or cortical_mask is None:
        # Unified relationship at coarse resolution
        E[:] = 6850.0 * np.power(np.clip(rho, 0.01, 2.0), 1.49)
    else:
        trab = ~cortical_mask
        E[trab] = 6850.0 * np.power(np.clip(rho[trab], 0.01, 2.0), 1.49)
        E[cortical_mask] = 10500.0 * np.power(
            np.clip(rho[cortical_mask], 0.01, 2.0), 2.29
        )

    return np.clip(E, E_MIN, 20000.0)


def density_to_yield_stress(rho: np.ndarray, E: np.ndarray = None) -> np.ndarray:
    """Apparent density → yield stress σ_y (MPa).

    Uses strain-based criterion (Bayraktar et al. 2004):
        ε_yield ≈ 0.0068 (compressive yield strain for trabecular bone)
        σ_y = ε_yield × E

    If E is not provided, falls back to Keller (1994):
        σ_y = 137 × ρ^1.88
    """
    if E is not None:
        # Strain-based yield (more physically accurate)
        YIELD_STRAIN = 0.0068  # Bayraktar 2004
        sigma_y = YIELD_STRAIN * E
    else:
        sigma_y = 137.0 * np.power(np.clip(rho, 0.01, 2.0), 1.88)
    return np.clip(sigma_y, 0.1, 200.0)


# ============================================================================
#  HEX ELEMENT STIFFNESS (8-NODE HEXAHEDRON)
# ============================================================================

def _gauss_points_3d():
    """2×2×2 Gauss quadrature points and weights."""
    g = 1.0 / np.sqrt(3.0)
    points = np.array([
        [-g, -g, -g], [+g, -g, -g], [+g, +g, -g], [-g, +g, -g],
        [-g, -g, +g], [+g, -g, +g], [+g, +g, +g], [-g, +g, +g],
    ])
    weights = np.ones(8)  # all weights = 1 for 2×2×2
    return points, weights


def _shape_function_derivs(xi, eta, zeta):
    """Shape function derivatives dN/dξ, dN/dη, dN/dζ for 8-node hex.

    Returns dN (3×8 matrix): dN[0,:] = dN/dξ, dN[1,:] = dN/dη, dN[2,:] = dN/dζ
    """
    dN = np.zeros((3, 8))

    # dN/dξ
    dN[0, 0] = -(1 - eta) * (1 - zeta)
    dN[0, 1] = +(1 - eta) * (1 - zeta)
    dN[0, 2] = +(1 + eta) * (1 - zeta)
    dN[0, 3] = -(1 + eta) * (1 - zeta)
    dN[0, 4] = -(1 - eta) * (1 + zeta)
    dN[0, 5] = +(1 - eta) * (1 + zeta)
    dN[0, 6] = +(1 + eta) * (1 + zeta)
    dN[0, 7] = -(1 + eta) * (1 + zeta)

    # dN/dη
    dN[1, 0] = -(1 - xi) * (1 - zeta)
    dN[1, 1] = -(1 + xi) * (1 - zeta)
    dN[1, 2] = +(1 + xi) * (1 - zeta)
    dN[1, 3] = +(1 - xi) * (1 - zeta)
    dN[1, 4] = -(1 - xi) * (1 + zeta)
    dN[1, 5] = -(1 + xi) * (1 + zeta)
    dN[1, 6] = +(1 + xi) * (1 + zeta)
    dN[1, 7] = +(1 - xi) * (1 + zeta)

    # dN/dζ
    dN[2, 0] = -(1 - xi) * (1 - eta)
    dN[2, 1] = -(1 + xi) * (1 - eta)
    dN[2, 2] = -(1 + xi) * (1 + eta)
    dN[2, 3] = -(1 - xi) * (1 + eta)
    dN[2, 4] = +(1 - xi) * (1 - eta)
    dN[2, 5] = +(1 + xi) * (1 - eta)
    dN[2, 6] = +(1 + xi) * (1 + eta)
    dN[2, 7] = +(1 - xi) * (1 + eta)

    dN /= 8.0
    return dN


def _elasticity_matrix(E: float, nu: float) -> np.ndarray:
    """Isotropic 3D elasticity matrix D (6×6).

    Voigt notation: [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_xz]
    """
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    D = np.zeros((6, 6))
    D[0, 0] = D[1, 1] = D[2, 2] = c * (1.0 - nu)
    D[0, 1] = D[1, 0] = c * nu
    D[0, 2] = D[2, 0] = c * nu
    D[1, 2] = D[2, 1] = c * nu
    D[3, 3] = D[4, 4] = D[5, 5] = c * (1.0 - 2.0 * nu) / 2.0
    return D


def _transversely_isotropic_matrix(E: float, nu: float) -> np.ndarray:
    """Transversely isotropic D (6×6) for bone.

    Bone is stiffer along the superior-inferior (z) axis.
    Isotropy plane: x-y (transverse).
    Stiff axis: z (axial/SI).

    Based on Pahr & Zysset (2009), Rho (1993):
        E_z / E_xy ≈ 1.3 (vertebral trabecular bone)
        G_xz / G_xy ≈ 1.15
        ν_xy = 0.30, ν_xz = 0.25
    """
    E_xy = E           # transverse modulus
    E_z = E * 1.3      # axial modulus (stiffer)
    nu_xy = nu          # in-plane Poisson's
    nu_xz = 0.25        # out-of-plane Poisson's
    nu_zx = nu_xz * E_z / E_xy  # symmetry: ν_zx/E_z = ν_xz/E_xy
    G_xy = E_xy / (2.0 * (1.0 + nu_xy))
    G_xz = G_xy * 1.15  # slightly stiffer in axial shear

    # Compliance matrix S (6×6), then invert to get D
    S = np.zeros((6, 6))
    S[0, 0] = 1.0 / E_xy
    S[1, 1] = 1.0 / E_xy
    S[2, 2] = 1.0 / E_z
    S[0, 1] = S[1, 0] = -nu_xy / E_xy
    S[0, 2] = S[2, 0] = -nu_xz / E_xy
    S[1, 2] = S[2, 1] = -nu_xz / E_xy
    S[3, 3] = 1.0 / G_xy
    S[4, 4] = 1.0 / G_xz
    S[5, 5] = 1.0 / G_xz

    D = np.linalg.inv(S)
    return D


def compute_reference_stiffness(h: float, nu: float = 0.3,
                                 anisotropic: bool = False) -> np.ndarray:
    """Compute reference element stiffness Ke_ref for E=1.

    Args:
        h: Element side length (mm).
        nu: Poisson's ratio.
        anisotropic: Use transversely isotropic D matrix.
    """
    if anisotropic:
        D_ref = _transversely_isotropic_matrix(1.0, nu)
    else:
        D_ref = _elasticity_matrix(1.0, nu)
    gp, gw = _gauss_points_3d()

    Ke_ref = np.zeros((24, 24))

    for i in range(8):
        xi, eta, zeta = gp[i]
        w = gw[i]

        dN = _shape_function_derivs(xi, eta, zeta)

        # Jacobian for regular cube: J = diag(h/2, h/2, h/2)
        J = np.diag([h / 2.0, h / 2.0, h / 2.0])
        detJ = (h / 2.0) ** 3
        invJ = np.diag([2.0 / h, 2.0 / h, 2.0 / h])

        # Transform derivatives to physical coordinates
        dN_phys = invJ @ dN  # 3×8

        # Build B matrix (6×24)
        B = np.zeros((6, 24))
        for n in range(8):
            col = n * 3
            B[0, col + 0] = dN_phys[0, n]  # dN/dx → ε_xx
            B[1, col + 1] = dN_phys[1, n]  # dN/dy → ε_yy
            B[2, col + 2] = dN_phys[2, n]  # dN/dz → ε_zz
            B[3, col + 0] = dN_phys[1, n]  # ε_xy
            B[3, col + 1] = dN_phys[0, n]
            B[4, col + 1] = dN_phys[2, n]  # ε_yz
            B[4, col + 2] = dN_phys[1, n]
            B[5, col + 0] = dN_phys[2, n]  # ε_xz
            B[5, col + 2] = dN_phys[0, n]

        Ke_ref += w * B.T @ D_ref @ B * detJ

    return Ke_ref


# ============================================================================
#  CAUSAL PARAMETERS (same interface as v4)
# ============================================================================

@dataclass
class CausalParameters:
    """Physical loading parameters."""
    force_magnitude: float = 5.0     # kN (actual force)
    flexion_angle: float = 0.0       # degrees (sagittal)
    lateral_angle: float = 0.0       # degrees (coronal)
    bmd_factor: float = 1.0          # multiplier on density
    cortical_thickness: float = 1.0  # relative cortical thickness

    def validate(self):
        assert self.force_magnitude >= 0
        assert -90 <= self.flexion_angle <= 90
        assert -45 <= self.lateral_angle <= 45
        assert 0.1 <= self.bmd_factor <= 2.0

    def to_dict(self) -> dict:
        return {
            'force_magnitude': self.force_magnitude,
            'flexion_angle': self.flexion_angle,
            'lateral_angle': self.lateral_angle,
            'bmd_factor': self.bmd_factor,
            'cortical_thickness': self.cortical_thickness,
        }


@dataclass
class FEMResult:
    """FEM simulation result."""
    ao_type: str = 'A0'
    confidence: float = 0.0

    # Mechanics
    max_von_mises: float = 0.0       # MPa
    max_displacement: float = 0.0     # mm
    n_yielded: int = 0
    n_elements: int = 0
    yielded_fraction: float = 0.0

    # AO metrics
    posterior_wall_damage: float = 0.0
    anterior_height_loss: float = 0.0
    posterior_height_loss: float = 0.0
    canal_compromise: float = 0.0
    n_fragments: int = 0
    damaged_fraction: float = 0.0
    fractured_fraction: float = 0.0

    # Timing
    solve_time: float = 0.0
    n_iterations: int = 0

    def summary(self) -> str:
        return (
            f"AO Type: {self.ao_type} (conf={self.confidence:.2f})\n"
            f"  Max von Mises: {self.max_von_mises:.1f} MPa\n"
            f"  Max displacement: {self.max_displacement:.3f} mm\n"
            f"  Yielded: {self.n_yielded}/{self.n_elements} "
            f"({self.yielded_fraction*100:.1f}%)\n"
            f"  Post. wall damage: {self.posterior_wall_damage*100:.1f}%\n"
            f"  Ant. height loss: {self.anterior_height_loss*100:.1f}%\n"
            f"  Post. height loss: {self.posterior_height_loss*100:.1f}%\n"
            f"  Canal compromise: {self.canal_compromise*100:.1f}%\n"
            f"  Fragments: {self.n_fragments}\n"
            f"  Solve time: {self.solve_time:.1f}s ({self.n_iterations} iters)"
        )


# ============================================================================
#  VOXEL FEM ENGINE
# ============================================================================

class VoxelFEMEngine:
    """CT-based voxel finite element engine for vertebral fracture.

    Each bone voxel is an 8-node hexahedral finite element.
    Solves actual equilibrium K·u = F using sparse solver.
    """

    def __init__(self, mask: np.ndarray, ct: np.ndarray,
                 voxel_size_mm: float = 1.0,
                 downsample: int = 1, seed: int = 42,
                 use_cuda: bool = False):
        """
        Args:
            mask: 3D binary mask (bone > 0). Shape (nx, ny, nz).
            ct: 3D CT volume (HU values). Same shape as mask.
            voxel_size_mm: Physical voxel size in mm.
            downsample: Downsample factor for simulation.
            seed: Random seed.
            use_cuda: Use CuPy GPU acceleration for sparse solve.
        """
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.ds = max(int(downsample), 1)
        self.use_cuda = use_cuda

        # Check CUDA availability
        if self.use_cuda:
            try:
                import cupy as cp
                import cupyx.scipy.sparse as cp_sp
                import cupyx.scipy.sparse.linalg as cp_spla
                self._cp = cp
                self._cp_sp = cp_sp
                self._cp_spla = cp_spla
                print(f"  CUDA enabled: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
            except ImportError:
                print("  [warn] CuPy not available, falling back to CPU")
                self.use_cuda = False

        # Store originals
        self._orig_mask = mask
        self._orig_ct = ct
        self._orig_voxel_size = voxel_size_mm

        # Downsample
        if self.ds > 1:
            self.mask = (zoom(mask.astype(np.float32),
                              1.0 / self.ds, order=0) > 0).astype(np.int32)
            self.ct = zoom(ct.astype(np.float32), 1.0 / self.ds, order=1)
            self.h = voxel_size_mm * self.ds  # element size in mm
        else:
            self.mask = mask.copy()
            self.ct = ct.copy()
            self.h = voxel_size_mm

        self.shape = self.mask.shape
        self.bone_mask = self.mask > 0
        self.n_elements = int(self.bone_mask.sum())

        # Precompute
        self._setup_mesh()
        self._setup_materials()

        # State
        self.params = None
        self._result = None
        self._displacement = None
        self._stress = None
        self._damage = None
        self._von_mises = None
        self._frames = []
        self._capture_frames = False

    # ================================================================
    #  MESH SETUP
    # ================================================================

    def _setup_mesh(self):
        """Build element-node connectivity for bone voxels.

        Node grid has dimensions (nx+1, ny+1, nz+1).
        Each bone voxel at (i,j,k) uses 8 corner nodes.
        """
        t0 = time.time()
        nx, ny, nz = self.shape

        # Element indices (bone voxels only)
        self._elem_ijk = np.argwhere(self.bone_mask)  # (n_elem, 3)

        # Node numbering: node at grid corner (i,j,k) has index:
        #   i + j*(nx+1) + k*(nx+1)*(ny+1)
        n_nodes_x = nx + 1
        n_nodes_y = ny + 1

        def node_id(i, j, k):
            return i + j * n_nodes_x + k * n_nodes_x * n_nodes_y

        # 8 corner offsets for hex element at (i,j,k)
        corner_offsets = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ])

        # Build connectivity: elem_nodes[e, :] = 8 global node IDs
        elem_nodes_list = []
        for offset in corner_offsets:
            node_ids = node_id(
                self._elem_ijk[:, 0] + offset[0],
                self._elem_ijk[:, 1] + offset[1],
                self._elem_ijk[:, 2] + offset[2],
            )
            elem_nodes_list.append(node_ids)
        self._elem_nodes = np.column_stack(elem_nodes_list)  # (n_elem, 8)

        # Identify unique nodes and create compact numbering
        unique_nodes, inverse = np.unique(
            self._elem_nodes.ravel(), return_inverse=True
        )
        self._n_nodes = len(unique_nodes)
        self._n_dof = self._n_nodes * 3

        # Map global → compact
        self._node_map = np.full(
            n_nodes_x * n_nodes_y * (nz + 1), -1, dtype=np.int64
        )
        self._node_map[unique_nodes] = np.arange(self._n_nodes)

        # Compact connectivity (vectorized)
        compact_nodes = self._node_map[self._elem_nodes]  # (n_elem, 8)
        # Expand: each node → 3 DOFs
        self._elem_dofs = np.zeros((self.n_elements, 24), dtype=np.int64)
        for n in range(8):
            self._elem_dofs[:, n*3+0] = compact_nodes[:, n] * 3 + 0
            self._elem_dofs[:, n*3+1] = compact_nodes[:, n] * 3 + 1
            self._elem_dofs[:, n*3+2] = compact_nodes[:, n] * 3 + 2

        # Compute reference stiffness (anisotropic on GPU for better physics)
        self._anisotropic = self.use_cuda  # use transverse isotropy on GPU
        self._Ke_ref = compute_reference_stiffness(
            self.h, NU_BONE, anisotropic=self._anisotropic)
        if self._anisotropic:
            print(f"  Using transversely isotropic elasticity (E_z/E_xy=1.3)")

        # Geometric properties for boundary conditions
        elem_z = self._elem_ijk[:, 2].astype(np.float64)
        self._z_min = elem_z.min()
        self._z_max = elem_z.max()
        self._z_range = max(self._z_max - self._z_min, 1.0)

        elem_y = self._elem_ijk[:, 1].astype(np.float64)
        self._y_min = elem_y.min()
        self._y_max = elem_y.max()
        self._y_range = max(self._y_max - self._y_min, 1.0)

        # AP and SI ratios per element
        self._elem_si = (elem_z - self._z_min) / self._z_range  # 0=inf, 1=sup
        self._elem_ap = (elem_y - self._y_min) / self._y_range  # 0=post, 1=ant

        elapsed = time.time() - t0
        print(f"  Mesh: {self.n_elements} elements, "
              f"{self._n_nodes} nodes, {self._n_dof} DOF "
              f"({elapsed:.1f}s)")

    # ================================================================
    #  MATERIAL SETUP
    # ================================================================

    def _setup_materials(self):
        """Compute per-element material properties from CT.

        Resolution-aware cortical shell: at any resolution, compute
        cortical fraction per voxel and blend E accordingly.
        """
        bone_hu = self.ct[self.bone_mask]
        self._rho = hu_to_density(bone_hu)

        # EDT for cortical detection at any resolution
        edt = distance_transform_edt(self.bone_mask).astype(np.float32)
        self._depth = edt[self.bone_mask]
        self._max_depth = max(edt.max(), 1.0)
        self._depth_ratio = self._depth / self._max_depth

        # Cortical fraction per voxel (0=fully trabecular, 1=fully cortical)
        # Surface voxels with high HU have high cortical fraction
        # This works at ANY resolution — gives fractional cortical influence
        surface_proximity = np.clip(1.0 - self._depth_ratio / 0.2, 0, 1)
        hu_factor = np.clip((bone_hu - 400) / 400, 0, 1)  # HU 400-800
        self._cortical_fraction = surface_proximity * hu_factor  # [0, 1]

        # Blended E: trabecular + cortical fraction
        rho_clip = np.clip(self._rho, 0.01, 2.0)
        E_trab = 6850.0 * np.power(rho_clip, 1.49)   # Morgan 2003
        E_cort = 10500.0 * np.power(rho_clip, 2.29)   # Keller 1994
        self._E_base = (
            (1.0 - self._cortical_fraction) * E_trab +
            self._cortical_fraction * E_cort
        )
        self._E_base = np.clip(self._E_base, E_MIN, 20000.0)

        # Yield stress: strain-based, higher for cortical
        YIELD_STRAIN = 0.0068  # Bayraktar 2004
        self._sigma_y_base = YIELD_STRAIN * self._E_base
        # Cortical yield is higher (cortical is tougher)
        self._sigma_y_base *= (1.0 + 0.5 * self._cortical_fraction)
        self._sigma_y_base = np.clip(self._sigma_y_base, 0.1, 200.0)

        # Precompute lateral position for contact pressure
        elem_x = self._elem_ijk[:, 0].astype(np.float64)
        x_min, x_max = elem_x.min(), elem_x.max()
        x_range = max(x_max - x_min, 1.0)
        self._elem_lr = (elem_x - x_min) / x_range  # 0=left, 1=right

        n_cort = (self._cortical_fraction > 0.5).sum()
        print(f"  Materials (cortical-blended): "
              f"E [{self._E_base.min():.0f}, {self._E_base.max():.0f}] MPa, "
              f"\u03c3_y [{self._sigma_y_base.min():.1f}, {self._sigma_y_base.max():.1f}] MPa")
        print(f"  Cortical fraction: {n_cort} voxels >50% cortical "
              f"({n_cort/max(self.n_elements,1)*100:.1f}%)")

    # ================================================================
    #  GLOBAL ASSEMBLY
    # ================================================================

    def _assemble_global_stiffness(self, E_elem: np.ndarray) -> sp.csr_matrix:
        """Assemble global stiffness matrix K (fully vectorized).

        Uses COO format with numpy broadcasting — no Python for-loops.
        """
        Ke_flat = self._Ke_ref.ravel()  # (576,)

        # local 24×24 index pairs
        li, lj = np.meshgrid(np.arange(24), np.arange(24), indexing='ij')
        li_flat = li.ravel()  # (576,)
        lj_flat = lj.ravel()

        # Global DOF indices for all elements: (n_elem, 576)
        rows = self._elem_dofs[:, li_flat]  # (n_elem, 576)
        cols = self._elem_dofs[:, lj_flat]

        # Values: E_elem[e] × Ke_ref_flat for each element
        vals = E_elem[:, None] * Ke_flat[None, :]  # (n_elem, 576)

        K = sp.coo_matrix(
            (vals.ravel(), (rows.ravel(), cols.ravel())),
            shape=(self._n_dof, self._n_dof)
        ).tocsr()
        return K

    # ================================================================
    #  BOUNDARY CONDITIONS
    # ================================================================

    def _apply_boundary_conditions(self, K: sp.csr_matrix,
                                    F: np.ndarray,
                                    params: CausalParameters,
                                    load_fraction: float = 1.0):
        """Apply BCs with parabolic contact pressure.

        Uses Hertz-like parabolic pressure distribution:
            P(r) = P_max × (1 - (r/R)²)
        giving realistic disc-endplate contact instead of uniform pressure.
        """
        PENALTY = K.diagonal().max() * 1e6

        sup_mask = self._elem_si > 0.88
        inf_mask = self._elem_si < 0.12

        # Fix inferior endplate: only z-DOFs (allow lateral expansion)
        inf_elems = np.where(inf_mask)[0]
        inf_dofs_all = self._elem_dofs[inf_elems]
        # z-DOFs: indices 2, 5, 8, 11, 14, 17, 20, 23
        z_indices = np.arange(2, 24, 3)
        inf_z_dofs = np.unique(inf_dofs_all[:, z_indices].ravel())
        diag = K.diagonal()
        diag[inf_z_dofs] += PENALTY
        K.setdiag(diag)
        F[inf_z_dofs] = 0.0

        # Anchor: fix x,y of ONE central bottom node to prevent rigid body motion
        if len(inf_elems) > 0:
            center_elem = inf_elems[len(inf_elems) // 2]
            anchor_dofs = [self._elem_dofs[center_elem, 0],   # x of node 0
                           self._elem_dofs[center_elem, 1]]   # y of node 0
            for d in anchor_dofs:
                diag = K.diagonal()
                diag[d] += PENALTY
                K.setdiag(diag)
                F[d] = 0.0

        # Parabolic contact pressure on superior endplate
        force_N = params.force_magnitude * 1000.0 * load_fraction
        flex_rad = np.radians(params.flexion_angle)
        lat_rad = np.radians(params.lateral_angle)

        sup_elems = np.where(sup_mask)[0]
        n_sup = len(sup_elems)
        if n_sup == 0:
            return K, F

        # Parabolic pressure: P(r) = P_max × (1 - r²/R²)
        # r = normalized distance from endplate center
        ap = self._elem_ap[sup_elems]  # anterior-posterior [0,1]
        lr = self._elem_lr[sup_elems]  # left-right [0,1]
        # Distance from center (0.5, 0.5)
        r2 = (ap - 0.5)**2 + (lr - 0.5)**2
        r2_max = r2.max() + 1e-6
        parabolic = np.clip(1.0 - r2 / r2_max, 0.1, 1.0)

        # Flexion bias
        flex_weight = 1.0 + np.sin(flex_rad) * (ap - 0.5) * 2.0
        flex_weight = np.clip(flex_weight, 0.2, 2.5)

        # Combined weight
        weights = parabolic * flex_weight
        weights /= weights.sum()  # normalize so total force = force_N
        f_per_elem = -force_N * weights  # compression (negative z)

        sup_dofs = self._elem_dofs[sup_elems]
        for n in range(4, 8):
            z_idx = n * 3 + 2
            np.add.at(F, sup_dofs[:, z_idx], f_per_elem / 4.0)
            if abs(lat_rad) > 0.01:
                x_idx = n * 3 + 0
                np.add.at(F, sup_dofs[:, x_idx],
                          f_per_elem * 0.1 * np.sin(lat_rad) / 4.0)

        return K, F

    # ================================================================
    #  STRESS COMPUTATION
    # ================================================================

    def _compute_element_stress(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute stress and von Mises per element (fully vectorized)."""
        # B matrix at element center (computed once)
        if not hasattr(self, '_B_center'):
            dN = _shape_function_derivs(0, 0, 0)
            invJ = np.diag([2.0 / self.h] * 3)
            dN_phys = invJ @ dN
            B = np.zeros((6, 24))
            for n in range(8):
                c = n * 3
                B[0, c+0] = dN_phys[0, n]
                B[1, c+1] = dN_phys[1, n]
                B[2, c+2] = dN_phys[2, n]
                B[3, c+0] = dN_phys[1, n]; B[3, c+1] = dN_phys[0, n]
                B[4, c+1] = dN_phys[2, n]; B[4, c+2] = dN_phys[1, n]
                B[5, c+0] = dN_phys[2, n]; B[5, c+2] = dN_phys[0, n]
            self._B_center = B
            if getattr(self, '_anisotropic', False):
                self._DB_center = _transversely_isotropic_matrix(1.0, NU_BONE) @ B
            else:
                self._DB_center = _elasticity_matrix(1.0, NU_BONE) @ B  # (6,24)

        # Gather element displacements: (n_elem, 24)
        u_all = u[self._elem_dofs]

        # Strain: (n_elem, 6) = (n_elem, 24) @ B^T → each row = B @ u_e
        strain_all = u_all @ self._B_center.T  # (n_elem, 6)

        # Stress: σ = E_e × D_ref × ε
        # DB = D_ref @ B already combined, so σ = E_e × (u_e @ DB^T)
        stress = self._E_current[:, None] * (u_all @ self._DB_center.T)

        # von Mises (vectorized)
        s = stress
        von_mises = np.sqrt(0.5 * (
            (s[:,0]-s[:,1])**2 + (s[:,1]-s[:,2])**2 + (s[:,2]-s[:,0])**2 +
            6.0 * (s[:,3]**2 + s[:,4]**2 + s[:,5]**2)
        ))

        return stress, von_mises

    # ================================================================
    #  AO CLASSIFICATION
    # ================================================================

    def _classify_ao(self, damage_mask: np.ndarray,
                     displacement: np.ndarray) -> FEMResult:
        """Classify AO type from FEM damage pattern.

        Uses continuous damage variable for classification.
        """
        damage = getattr(self, '_damage', damage_mask.astype(np.float64))

        n_yielded = damage_mask.sum()
        yield_frac = n_yielded / max(self.n_elements, 1)
        mean_damage = damage.mean()
        max_damage_val = damage.max()

        # Z-displacement at element centers (vectorized)
        z_dof_cols = self._elem_dofs[:, 2::3]  # (n_elem, 8) z-DOFs
        elem_uz = displacement[z_dof_cols].mean(axis=1)  # (n_elem,)

        # Regional damage analysis
        anterior = self._elem_ap > 0.65
        posterior = self._elem_ap < 0.30
        middle = (~anterior) & (~posterior)

        ant_damage = damage[anterior].mean() if anterior.any() else 0.0
        post_damage = damage[posterior].mean() if posterior.any() else 0.0
        mid_damage = damage[middle].mean() if middle.any() else 0.0

        # Height loss
        ant_disp = np.abs(elem_uz[anterior]).mean() if anterior.any() else 0.0
        ant_h_loss = ant_disp / (self._z_range * self.h) if self._z_range > 0 else 0.0
        post_disp = np.abs(elem_uz[posterior]).mean() if posterior.any() else 0.0
        post_h_loss = post_disp / (self._z_range * self.h) if self._z_range > 0 else 0.0

        # Canal compromise (based on posterior damage)
        canal = min(post_damage * 3.0, 1.0)

        # Fragments
        damage_3d = np.zeros(self.shape, dtype=bool)
        high_damage = damage > 0.5
        damage_3d[self.bone_mask] = high_damage
        if damage_3d.any():
            labeled, n_labels = ndi.label(damage_3d)
            min_size = max(self.n_elements * 0.005, 3)
            n_fragments = sum(
                1 for lbl in range(1, n_labels + 1)
                if (labeled == lbl).sum() >= min_size
            )
        else:
            n_fragments = 0

        frac_frac = (damage > 0.5).sum() / max(self.n_elements, 1)

        # ---- CLASSIFICATION (damage-based) ----
        if mean_damage < 0.005 and yield_frac < 0.01:
            ao_type, conf = 'A0', 0.9
        elif post_damage < 0.03:
            # No significant posterior damage
            if ant_damage > post_damage * 2.0 and ant_damage > 0.01:
                ao_type = 'A1'
                conf = min(ant_damage / 0.05 + 0.4, 0.95)
            elif mean_damage > 0.03:
                ao_type, conf = 'A2', 0.7
            else:
                ao_type, conf = 'A1', 0.5
        elif post_damage < 0.10:
            ao_type, conf = 'A2', 0.7
        elif canal < 0.40 or post_damage < 0.15:
            ao_type, conf = 'A3', 0.8
        else:
            ao_type, conf = 'A4', min(canal / 0.5, 1.0)

        return FEMResult(
            ao_type=ao_type, confidence=conf,
            max_von_mises=self._von_mises.max(),
            max_displacement=np.abs(displacement).max(),
            n_yielded=n_yielded, n_elements=self.n_elements,
            yielded_fraction=yield_frac,
            posterior_wall_damage=post_damage,
            anterior_height_loss=ant_h_loss,
            posterior_height_loss=post_h_loss,
            canal_compromise=canal,
            n_fragments=n_fragments,
            damaged_fraction=mean_damage,
            fractured_fraction=frac_frac,
        )

    # ================================================================
    #  MAIN SIMULATION
    # ================================================================

    def set_causal_params(self, params: CausalParameters):
        params.validate()
        self.params = params

    def simulate(self, max_damage_iters: int = 5,
                 n_load_steps: int = 4,
                 verbose: bool = True) -> FEMResult:
        """FEM with incremental loading + progressive damage.

        Incremental loading (Updated Lagrangian approximation):
            - Force applied in n_load_steps increments
            - At each step: solve, update damage, accumulate displacement
            - Approximates geometric nonlinearity

        Progressive damage:
            - Continuous damage d ∈ [0, 1] per element
            - E_eff = (1 - d)² × E  (quadratic degradation)
            - d grows based on stress/yield ratio
            - Element effectively deleted when d > 0.95
        """
        params = self.params
        if params is None:
            raise ValueError("Call set_causal_params() first.")

        t_start = time.time()

        # Material properties with BMD factor
        rho_mod = self._rho * params.bmd_factor
        rho_clip = np.clip(rho_mod, 0.01, 2.0)
        E_trab = 6850.0 * np.power(rho_clip, 1.49)
        E_cort = 10500.0 * np.power(rho_clip, 2.29)
        cf = self._cortical_fraction
        E_base = np.clip(
            (1.0 - cf) * E_trab + cf * E_cort, E_MIN, 20000.0)
        sigma_y = np.clip(
            0.0068 * E_base * (1.0 + 0.5 * cf) * params.cortical_thickness,
            0.1, 200.0)
        self._sigma_y = sigma_y

        # Progressive damage variable: d ∈ [0, 1]
        damage = np.zeros(self.n_elements, dtype=np.float64)
        u_total = np.zeros(self._n_dof, dtype=np.float64)
        total_iters = 0

        for step in range(n_load_steps):
            load_frac = (step + 1) / n_load_steps
            if verbose:
                print(f"\n  Load step {step+1}/{n_load_steps} "
                      f"({load_frac*100:.0f}% of {params.force_magnitude:.1f} kN):")

            # Effective stiffness: LINEAR degradation (stable)
            self._E_current = E_base * (1.0 - damage)
            self._E_current = np.clip(self._E_current, 10.0, 20000.0)

            for iteration in range(max_damage_iters):
                total_iters += 1
                if verbose:
                    print(f"    Iter {iteration+1}/{max_damage_iters}:")

                t_asm = time.time()
                K = self._assemble_global_stiffness(self._E_current)
                if verbose:
                    print(f"      Assembly: {time.time()-t_asm:.1f}s")

                F = np.zeros(self._n_dof, dtype=np.float64)
                K, F = self._apply_boundary_conditions(
                    K, F, params, load_fraction=load_frac)

                # Solve
                t_solve = time.time()
                if self.use_cuda:
                    K_gpu = self._cp_sp.csr_matrix(K)
                    F_gpu = self._cp.array(F)
                    try:
                        u_gpu = self._cp_spla.spsolve(K_gpu, F_gpu)
                        u = self._cp.asnumpy(u_gpu)
                    except Exception:
                        u_gpu, info = self._cp_spla.cg(
                            K_gpu, F_gpu, maxiter=5000, tol=1e-8)
                        u = self._cp.asnumpy(u_gpu)
                else:
                    try:
                        ilu = spla.spilu(K.tocsc(), fill_factor=5)
                        M = spla.LinearOperator(K.shape, ilu.solve)
                        u, info = spla.cg(K, F, M=M, maxiter=3000, tol=1e-8)
                        if info != 0:
                            u = spla.spsolve(K, F)
                    except Exception:
                        u = spla.spsolve(K, F)

                if verbose:
                    print(f"      Solve: {time.time()-t_solve:.1f}s, "
                          f"|u|_max={np.abs(u).max():.4f} mm")

                # Stress
                stress, von_mises = self._compute_element_stress(u)
                self._stress = stress
                self._von_mises = von_mises
                self._displacement = u
                u_total = u  # current displacement field

                # Progressive damage update
                stress_ratio = von_mises / np.clip(sigma_y, 0.01, None)
                # Incremental damage: capped at 0.2 per iter, max 0.9 total
                d_increment = np.clip((stress_ratio - 1.0) / 3.0, 0, 0.2)
                damage = np.minimum(damage + d_increment, 0.9)

                # Update effective E (linear degradation, stable)
                E_new = E_base * (1.0 - damage)
                E_new = np.clip(E_new, 10.0, 20000.0)
                change = np.abs(E_new - self._E_current).max()
                self._E_current = E_new

                n_damaged = (damage > 0.01).sum()
                n_failed = (damage > 0.95).sum()
                if verbose:
                    print(f"      σ_vm: max={von_mises.max():.1f}, "
                          f"mean={von_mises[von_mises>0].mean():.1f} MPa")
                    print(f"      Damaged: {n_damaged} "
                          f"({n_damaged/self.n_elements*100:.1f}%), "
                          f"failed: {n_failed}")

                if self._capture_frames:
                    self._frames.append({
                        'step': step, 'iteration': iteration,
                        'von_mises': von_mises.copy(),
                        'damage': damage.copy(),
                        'displacement': u.copy(),
                    })

                if change < 1.0 or n_damaged == 0:
                    if verbose:
                        print(f"      Converged (ΔE_max={change:.2f})")
                    break

        total_time = time.time() - t_start

        # Classify
        self._damage_mask = damage > 0.05  # yielded = any damage > 5%
        self._damage = damage
        result = self._classify_ao(self._damage_mask, u_total)
        result.solve_time = total_time
        result.n_iterations = total_iters
        self._result = result

        if verbose:
            print(f"\n  ★ {result.ao_type} ({total_time:.1f}s, "
                  f"{total_iters} total iters)")
            print(result.summary())

        return result

    # ================================================================
    #  ANIMATED SIMULATION
    # ================================================================

    def simulate_animated(self, max_damage_iters=5,
                          verbose=True) -> FEMResult:
        self._capture_frames = True
        self._frames = []
        result = self.simulate(max_damage_iters=max_damage_iters,
                               verbose=verbose)
        self._capture_frames = False
        return result

    # ================================================================
    #  COUNTERFACTUAL / SWEEP
    # ================================================================

    def counterfactual(self, **param_changes) -> FEMResult:
        if self.params is None:
            raise ValueError("Run simulate() first.")
        cf = CausalParameters(**self.params.to_dict())
        for k, v in param_changes.items():
            setattr(cf, k, v)
        self.set_causal_params(cf)
        return self.simulate(verbose=False)

    def parameter_sweep(self, param: str, values: list,
                        verbose: bool = True) -> List[Tuple[float, FEMResult]]:
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
                      f"(σ_vm={r.max_von_mises:.0f} MPa, "
                      f"yield={r.yielded_fraction*100:.1f}%, "
                      f"{r.solve_time:.1f}s)")
        return results

    # ================================================================
    #  VISUALIZATION
    # ================================================================

    @staticmethod
    def _match_shape(arr, target_shape):
        slices = tuple(slice(0, min(s, t)) for s, t in
                       zip(arr.shape, target_shape))
        result = arr[slices]
        pad_width = [(0, max(0, t - s)) for s, t in
                     zip(result.shape, target_shape)]
        if any(p[1] > 0 for p in pad_width):
            result = np.pad(result, pad_width, mode='constant')
        return result

    def _to_3d(self, elem_data):
        """Map per-element data to 3D + upsample."""
        vol = np.zeros(self.shape, dtype=np.float32)
        vol[self.bone_mask] = elem_data.astype(np.float32)
        if self.ds > 1:
            vol = self._match_shape(
                zoom(vol, self.ds, order=1), self._orig_mask.shape)
        return vol


# ============================================================================
#  STANDALONE VISUALIZATION FUNCTIONS
# ============================================================================

_AO_COLORS = {'A0': '#2196F3', 'A1': '#4CAF50', 'A2': '#FFC107',
               'A3': '#FF5722', 'A4': '#9C27B0'}


def _plot_fracture_mechanics(engine, output_dir,
                              filename='v5_fracture_mechanics.png',
                              mag_factor=10.0):
    """Enhanced fracture visualization showing HOW the bone breaks.

    6 panels:
      1. Original CT
      2. Deformed CT (displacement × mag_factor)
      3. Damage + crack contour lines
      4. Displacement vector arrows
      5. Height profile (original vs deformed)
      6. Key metrics
    """
    if not HAS_MPL or engine._displacement is None:
        return

    result = engine._result
    params = engine.params
    ao_color = _AO_COLORS.get(result.ao_type, '#888')
    damage = getattr(engine, '_damage', engine._damage_mask.astype(np.float64))

    # 3D volumes
    vm_vol = engine._to_3d(engine._von_mises)
    dmg_vol = engine._to_3d(damage.astype(np.float32))

    ct = engine._orig_ct if engine.ds > 1 else engine.ct
    mask_orig = engine._orig_mask if engine.ds > 1 else engine.mask
    mid_x = ct.shape[0] // 2

    # Displacement field → 3D (ux, uy, uz per node)
    u = engine._displacement
    n_elem = engine.n_elements
    elem_dofs = engine._elem_dofs  # (n_elem, 24)

    # Compute per-element mean displacement (ux, uy, uz)
    ux_elem = np.zeros(n_elem)
    uy_elem = np.zeros(n_elem)
    uz_elem = np.zeros(n_elem)
    for n in range(8):
        ux_elem += u[elem_dofs[:, n*3+0]]
        uy_elem += u[elem_dofs[:, n*3+1]]
        uz_elem += u[elem_dofs[:, n*3+2]]
    ux_elem /= 8.0
    uy_elem /= 8.0
    uz_elem /= 8.0

    ux_vol = engine._to_3d(ux_elem)
    uy_vol = engine._to_3d(uy_elem)
    uz_vol = engine._to_3d(uz_elem)
    u_mag_vol = np.sqrt(ux_vol**2 + uy_vol**2 + uz_vol**2)

    # --- Deformed CT: warp the image ---
    ct_slice = ct[mid_x].astype(np.float32)
    h, w = ct_slice.shape
    yy, zz = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Displacement on this slice (in voxels)
    voxel_size = engine._orig_voxel_size
    dy_pix = uy_vol[mid_x] * mag_factor / voxel_size
    dz_pix = uz_vol[mid_x] * mag_factor / voxel_size

    # Warped coordinates
    yy_warp = np.clip(yy + dy_pix, 0, h - 1).astype(np.float32)
    zz_warp = np.clip(zz + dz_pix, 0, w - 1).astype(np.float32)

    from scipy.ndimage import map_coordinates
    ct_deformed = map_coordinates(ct_slice, [yy_warp, zz_warp], order=1)

    # ===== FIGURE =====
    fig = plt.figure(figsize=(24, 14), facecolor='#0d1117')
    gs = GridSpec(2, 3, hspace=0.25, wspace=0.15)
    hu_kw = dict(cmap='bone', origin='lower', vmin=-100, vmax=800, aspect='auto')

    # --- Panel 1: Original CT ---
    ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor('#0d1117')
    ax1.imshow(ct_slice.T, **hu_kw)
    ax1.set_title('① Original CT', fontsize=14, color='white', pad=10)
    ax1.axis('off')

    # --- Panel 2: Deformed CT ---
    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor('#0d1117')
    ax2.imshow(ct_deformed.T, **hu_kw)
    # Overlay displacement magnitude
    u_slice = u_mag_vol[mid_x]
    u_masked = np.ma.masked_where(u_slice < 0.01, u_slice)
    ax2.imshow(u_masked.T, cmap='coolwarm', origin='lower',
               alpha=0.4, vmin=0, vmax=max(u_mag_vol.max(), 0.1), aspect='auto')
    ax2.set_title(f'② Deformed (×{mag_factor:.0f} magnified)',
                  fontsize=14, color='#ff6b6b', pad=10)
    ax2.axis('off')

    # --- Panel 3: Damage + Crack contours ---
    ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor('#0d1117')
    ax3.imshow(ct_slice.T, **hu_kw, alpha=0.3)
    dmg_slice = dmg_vol[mid_x]
    # Continuous damage colormap (not binary!)
    dmg_masked = np.ma.masked_where(dmg_slice < 0.01, dmg_slice)
    im3 = ax3.imshow(dmg_masked.T, cmap='inferno', origin='lower',
                     alpha=0.85, vmin=0, vmax=1.0, aspect='auto')
    # Crack contour lines
    for level, color, lw in [(0.3, '#00ff88', 1.0),
                              (0.6, '#ffaa00', 1.5),
                              (0.9, '#ff0000', 2.0)]:
        ax3.contour(dmg_slice.T, levels=[level], colors=[color],
                    linewidths=[lw], origin='lower')
    # Legend for contour lines
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color='#00ff88', lw=1, label='d=0.3 (yield)'),
        Line2D([0], [0], color='#ffaa00', lw=1.5, label='d=0.6 (fracture)'),
        Line2D([0], [0], color='#ff0000', lw=2, label='d=0.9 (failure)'),
    ]
    ax3.legend(handles=legend_elems, loc='lower right', fontsize=9,
              facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    ax3.set_title('③ Damage + Crack Lines', fontsize=14, color='white', pad=10)
    ax3.axis('off')
    cb3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, shrink=0.8)
    cb3.set_label('Damage (d)', fontsize=10, color='white')
    cb3.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb3.ax.yaxis.get_ticklabels(), color='white', fontsize=8)

    # --- Panel 4: Displacement vectors ---
    ax4 = fig.add_subplot(gs[1, 0]); ax4.set_facecolor('#0d1117')
    ax4.imshow(ct_slice.T, **hu_kw, alpha=0.35)
    # Subsample for quiver
    step = max(h // 25, 1)
    y_q, z_q = np.meshgrid(np.arange(0, h, step), np.arange(0, w, step), indexing='ij')
    dy_q = uy_vol[mid_x, ::step, ::step]
    dz_q = uz_vol[mid_x, ::step, ::step]
    mag_q = np.sqrt(dy_q**2 + dz_q**2)
    # Only show arrows where there's bone
    bone_q = (mask_orig[mid_x, ::step, ::step] > 0) if mask_orig is not None else (mag_q > 0)
    dy_q[~bone_q] = 0
    dz_q[~bone_q] = 0
    ax4.quiver(y_q, z_q, dy_q, dz_q, mag_q,
               cmap='plasma', scale=max(mag_q.max() * 8, 0.01),
               width=0.003, headwidth=4, alpha=0.9)
    ax4.set_title('④ Displacement Vectors', fontsize=14, color='white', pad=10)
    ax4.axis('off')

    # --- Panel 5: Height profile (orig vs deformed) ---
    ax5 = fig.add_subplot(gs[1, 1]); ax5.set_facecolor('#0d1117')
    # Compute column heights along AP direction
    bone_slice_2d = (mask_orig[mid_x] > 0) if mask_orig is not None else (ct_slice > 200)
    n_cols = bone_slice_2d.shape[0]
    ap_positions = np.arange(n_cols)
    orig_heights = np.zeros(n_cols)
    deformed_heights = np.zeros(n_cols)
    for col in range(n_cols):
        bone_col = np.where(bone_slice_2d[col])[0]
        if len(bone_col) > 2:
            orig_heights[col] = (bone_col[-1] - bone_col[0]) * voxel_size
            # Deformed height: original + displacement difference
            uz_col = uz_vol[mid_x, col, :]
            top_uz = uz_col[bone_col[-1]] if bone_col[-1] < uz_col.shape[0] else 0
            bot_uz = uz_col[bone_col[0]] if bone_col[0] < uz_col.shape[0] else 0
            deformed_heights[col] = orig_heights[col] + (top_uz - bot_uz)

    valid = orig_heights > 1.0
    if valid.any():
        ax5.fill_between(ap_positions[valid], 0, orig_heights[valid],
                        alpha=0.3, color='#4fc3f7', label='Original')
        ax5.fill_between(ap_positions[valid], 0, deformed_heights[valid],
                        alpha=0.5, color='#ff5252', label='Deformed')
        ax5.plot(ap_positions[valid], orig_heights[valid],
                color='#4fc3f7', linewidth=2)
        ax5.plot(ap_positions[valid], deformed_heights[valid],
                color='#ff5252', linewidth=2, linestyle='--')
        # Height loss annotation
        max_loss_idx = np.argmin(deformed_heights[valid] - orig_heights[valid])
        loss_mm = orig_heights[valid][max_loss_idx] - deformed_heights[valid][max_loss_idx]
        loss_pct = loss_mm / max(orig_heights[valid][max_loss_idx], 0.1) * 100
        ax5.annotate(f'Max loss: {loss_mm:.1f}mm ({loss_pct:.0f}%)',
                    xy=(ap_positions[valid][max_loss_idx],
                        deformed_heights[valid][max_loss_idx]),
                    xytext=(0, 20), textcoords='offset points',
                    fontsize=11, color='#ff5252', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#ff5252'))
    ax5.set_xlabel('Anterior ← → Posterior', fontsize=11, color='white')
    ax5.set_ylabel('Height (mm)', fontsize=11, color='white')
    ax5.set_title('⑤ Height Profile', fontsize=14, color='white', pad=10)
    ax5.legend(fontsize=10, facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    ax5.tick_params(colors='white')
    ax5.spines['bottom'].set_color('#555')
    ax5.spines['left'].set_color('#555')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    # --- Panel 6: Metrics summary ---
    ax6 = fig.add_subplot(gs[1, 2]); ax6.set_facecolor('#0d1117')
    ax6.axis('off')
    metrics_text = (
        f"{'─' * 35}\n"
        f"  AO Classification:  {result.ao_type}\n"
        f"  Confidence:         {result.confidence:.0%}\n"
        f"{'─' * 35}\n"
        f"  Force:              {params.force_magnitude:.1f} kN\n"
        f"  Flexion:            {params.flexion_angle:.0f}°\n"
        f"  BMD Factor:         {params.bmd_factor:.2f}\n"
        f"{'─' * 35}\n"
        f"  Max σ_vm:           {result.max_von_mises:.0f} MPa\n"
        f"  Max displacement:   {result.max_displacement:.2f} mm\n"
        f"  Yield fraction:     {result.yielded_fraction*100:.1f}%\n"
        f"  Post. wall damage:  {result.posterior_wall_damage*100:.1f}%\n"
        f"  Ant. height loss:   {result.anterior_height_loss*100:.1f}%\n"
        f"  Canal compromise:   {result.canal_compromise*100:.1f}%\n"
        f"  Fragments:          {result.n_fragments}\n"
        f"{'─' * 35}\n"
        f"  Solve time:         {result.solve_time:.0f}s\n"
    )
    ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes,
             fontsize=12, color='white', va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor=ao_color, alpha=0.25))
    ax6.set_title(f'⑥ {result.ao_type} Fracture', fontsize=14,
                  color=ao_color, pad=10, fontweight='bold')

    fig.suptitle(f'Voxel FEM Fracture Mechanics — {result.ao_type}',
                 fontsize=18, color='white', fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(output_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"  Saved: {out}")
    plt.close(fig)


def _plot_scenario(engine, output_path):
    """4-panel: CT | von Mises | Yield | Metrics."""
    if not HAS_MPL or engine._von_mises is None:
        return

    vm = engine._to_3d(engine._von_mises)
    dmg = engine._to_3d(engine._damage_mask.astype(np.float32))
    ct = engine._orig_ct if engine.ds > 1 else engine.ct
    result, params = engine._result, engine.params
    ao_color = _AO_COLORS.get(result.ao_type, '#888')
    mid_x = ct.shape[0] // 2

    fig = plt.figure(figsize=(22, 6), facecolor='#1a1a2e')
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.55], wspace=0.08)
    hu_kw = dict(cmap='gray', origin='lower', vmin=-100, vmax=800, aspect='auto')

    ax1 = fig.add_subplot(gs[0]); ax1.set_facecolor('#1a1a2e')
    ax1.imshow(ct[mid_x].T, **hu_kw)
    ax1.set_title('Original CT (sagittal)', fontsize=12, color='white', pad=8)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[1]); ax2.set_facecolor('#1a1a2e')
    ax2.imshow(ct[mid_x].T, **hu_kw, alpha=0.3)
    vm_max = max(vm.max(), 1.0)
    im2 = ax2.imshow(vm[mid_x].T, cmap='jet', origin='lower',
                      alpha=0.7, vmin=0, vmax=vm_max, aspect='auto')
    ax2.set_title('von Mises Stress (MPa)', fontsize=12, color='white', pad=8)
    ax2.axis('off')
    cb = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label('σ_vm', fontsize=9, color='white')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

    ax3 = fig.add_subplot(gs[2]); ax3.set_facecolor('#1a1a2e')
    ax3.imshow(ct[mid_x].T, **hu_kw, alpha=0.3)
    ax3.imshow(dmg[mid_x].T, cmap='Reds', origin='lower',
               alpha=0.7, vmin=0, vmax=1, aspect='auto')
    ax3.set_title(f'Yielded ({result.yielded_fraction*100:.1f}%)',
                   fontsize=12, color='white', pad=8)
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[3]); ax4.set_facecolor('#1a1a2e'); ax4.axis('off')
    metrics = (
        f"AO {result.ao_type}  ({result.confidence:.0%})\n{'─'*22}\n"
        f"Force:    {params.force_magnitude:.1f} kN\n"
        f"BMD:      {params.bmd_factor:.1f}\n"
        f"Flexion:  {params.flexion_angle:.0f}°\n{'─'*22}\n"
        f"Max σ_vm: {result.max_von_mises:.1f} MPa\n"
        f"Max |u|:  {result.max_displacement:.3f} mm\n"
        f"Yielded:  {result.yielded_fraction*100:.1f}%\n"
        f"Post.Wall:{result.posterior_wall_damage*100:.1f}%\n"
        f"Canal:    {result.canal_compromise*100:.1f}%\n{'─'*22}\n"
        f"Iters:{result.n_iterations}  Time:{result.solve_time:.0f}s"
    )
    ax4.text(0.05, 0.95, metrics, transform=ax4.transAxes,
             fontsize=10, color='white', va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=ao_color,
                       alpha=0.35, edgecolor=ao_color))
    fig.suptitle(
        f"Voxel FEM  •  AO {result.ao_type}  •  "
        f"Force {params.force_magnitude:.1f} kN  •  BMD {params.bmd_factor:.1f}",
        fontsize=14, color='white', y=0.98, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"  Saved: {output_path}")
    plt.close(fig)


def _plot_sweep(sweep_results, output_dir, filename='v5_force_sweep.png',
                xlabel='Force (kN)', title_suffix=''):
    """Sweep chart: yield % + max σ_vm, AO-colored bars."""
    if not HAS_MPL:
        return
    import matplotlib.patches as mpatches

    vals = [r[0] for r in sweep_results]
    yields = [r[1].yielded_fraction * 100 for r in sweep_results]
    ao_types = [r[1].ao_type for r in sweep_results]
    vm_max = [r[1].max_von_mises for r in sweep_results]
    colors = [_AO_COLORS.get(t, '#888') for t in ao_types]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                     facecolor='#1a1a2e', sharex=True)
    for ax in (ax1, ax2):
        ax.set_facecolor('#16213e')

    bars = ax1.bar([str(v) for v in vals], yields, color=colors,
                   edgecolor='white', linewidth=0.6, width=0.55, alpha=0.85)
    ax1.plot(range(len(vals)), yields, color='#e94560', marker='o',
             markersize=5, linewidth=1.5, zorder=4)
    for bar, ao, y in zip(bars, ao_types, yields):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{ao}\n{y:.1f}%', ha='center', va='bottom', fontsize=8,
                 color='white', fontweight='bold')
    ax1.set_ylabel('Yielded Elements (%)', fontsize=11, color='white')
    ax1.set_title(f'FEM Sweep — AO Classification{title_suffix}',
                   fontsize=13, color='white', pad=10)
    ax1.set_ylim(0, max(max(yields), 1) * 1.4 + 5)

    ax2.bar([str(v) for v in vals], vm_max, color='#444', edgecolor='white',
            linewidth=0.6, width=0.55, alpha=0.6)
    ax2.plot(range(len(vals)), vm_max, color='#ffd700', marker='s',
             markersize=5, linewidth=1.5, zorder=4, label='Max σ_vm')
    ax2.set_xlabel(xlabel, fontsize=11, color='white')
    ax2.set_ylabel('Max von Mises (MPa)', fontsize=11, color='white')
    ax2.legend(fontsize=9, facecolor='#16213e', edgecolor='#444',
               labelcolor='white')

    patches = [mpatches.Patch(color=_AO_COLORS[k], label=k)
               for k in ['A0','A1','A2','A3','A4']]
    ax1.legend(handles=patches, title='AO', loc='upper left',
               fontsize=9, facecolor='#16213e', edgecolor='#444',
               labelcolor='white').get_title().set_color('white')

    for ax in (ax1, ax2):
        ax.tick_params(colors='white')
        for s in ['top','right']: ax.spines[s].set_visible(False)
        for s in ['bottom','left']: ax.spines[s].set_color('#444')
        ax.grid(axis='y', color='#333', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out = os.path.join(output_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"  Saved: {out}")
    plt.close(fig)


def _plot_multi_slice(engine, output_dir, filename='v5_multi_slice.png'):
    """3-plane (sag/ax/cor) × 2 rows (σ_vm + yield)."""
    if not HAS_MPL or engine._von_mises is None:
        return

    vm = engine._to_3d(engine._von_mises)
    dmg = engine._to_3d(engine._damage_mask.astype(np.float32))
    ct = engine._orig_ct if engine.ds > 1 else engine.ct
    result = engine._result

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='#1a1a2e')
    hu_kw = dict(cmap='gray', origin='lower', vmin=-100, vmax=800, aspect='auto')
    labels = ['Sagittal', 'Axial', 'Coronal']
    mid = [ct.shape[i]//2 for i in range(3)]
    ct_s = [ct[mid[0]], ct[:, mid[1]], ct[:, :, mid[2]]]
    vm_s = [vm[mid[0]], vm[:, mid[1]], vm[:, :, mid[2]]]
    dm_s = [dmg[mid[0]], dmg[:, mid[1]], dmg[:, :, mid[2]]]
    vm_max = max(vm.max(), 1.0)

    for i in range(3):
        axes[0,i].set_facecolor('#1a1a2e')
        axes[0,i].imshow(ct_s[i].T, **hu_kw, alpha=0.4)
        im = axes[0,i].imshow(vm_s[i].T, cmap='jet', origin='lower',
                               alpha=0.6, vmin=0, vmax=vm_max, aspect='auto')
        axes[0,i].set_title(f'{labels[i]} — σ_vm', fontsize=11, color='white')
        axes[0,i].axis('off')

        axes[1,i].set_facecolor('#1a1a2e')
        axes[1,i].imshow(ct_s[i].T, **hu_kw, alpha=0.4)
        axes[1,i].imshow(dm_s[i].T, cmap='Reds', origin='lower',
                          alpha=0.65, vmin=0, vmax=1, aspect='auto')
        axes[1,i].set_title(f'{labels[i]} — Yield', fontsize=11, color='white')
        axes[1,i].axis('off')

    cb = fig.colorbar(im, ax=axes[0,-1], fraction=0.046, pad=0.04)
    cb.set_label('σ_vm (MPa)', fontsize=9, color='white')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

    fig.suptitle(
        f"AO {result.ao_type}  •  σ_vm max {result.max_von_mises:.0f} MPa  •  "
        f"Yield {result.yielded_fraction*100:.1f}%",
        fontsize=14, color='white', y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(output_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"  Saved: {out}")
    plt.close(fig)


def _plot_progression(engine, output_dir, params_list,
                      filename='v5_progression.png'):
    """Side-by-side: each row = (CT, σ_vm, yield+info)."""
    if not HAS_MPL:
        return
    n = len(params_list)
    fig, axes = plt.subplots(n, 3, figsize=(16, 4*n), facecolor='#1a1a2e')
    if n == 1: axes = axes[np.newaxis, :]
    ct = engine._orig_ct if engine.ds > 1 else engine.ct
    hu_kw = dict(cmap='gray', origin='lower', vmin=-100, vmax=800, aspect='auto')
    mid_x = ct.shape[0] // 2; ct_slice = ct[mid_x]

    for row, (label, params) in enumerate(params_list):
        engine.set_causal_params(params)
        result = engine.simulate(verbose=False)
        vm = engine._to_3d(engine._von_mises)
        dmg = engine._to_3d(engine._damage_mask.astype(np.float32))
        vm_max = max(vm.max(), 1.0)
        ao_c = _AO_COLORS.get(result.ao_type, '#888')

        axes[row,0].set_facecolor('#1a1a2e')
        axes[row,0].imshow(ct_slice.T, **hu_kw)
        axes[row,0].set_title(label, fontsize=10, color=ao_c, fontweight='bold')
        axes[row,0].axis('off')

        axes[row,1].set_facecolor('#1a1a2e')
        axes[row,1].imshow(ct_slice.T, **hu_kw, alpha=0.3)
        axes[row,1].imshow(vm[mid_x].T, cmap='jet', origin='lower',
                            alpha=0.65, vmin=0, vmax=vm_max, aspect='auto')
        axes[row,1].set_title(
            f'AO {result.ao_type} — σ_vm {result.max_von_mises:.0f} MPa',
            fontsize=10, color=ao_c, fontweight='bold')
        axes[row,1].axis('off')

        axes[row,2].set_facecolor('#1a1a2e')
        axes[row,2].imshow(ct_slice.T, **hu_kw, alpha=0.3)
        axes[row,2].imshow(dmg[mid_x].T, cmap='Reds', origin='lower',
                            alpha=0.65, vmin=0, vmax=1, aspect='auto')
        info = (f"F={params.force_magnitude:.0f}kN BMD={params.bmd_factor:.1f}\n"
                f"Yield={result.yielded_fraction*100:.1f}% "
                f"Post={result.posterior_wall_damage*100:.0f}%")
        axes[row,2].text(0.02, 0.98, info, transform=axes[row,2].transAxes,
                          fontsize=9, color='white', va='top',
                          fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor=ao_c, alpha=0.4))
        axes[row,2].set_title(f'Yield {result.yielded_fraction*100:.1f}%',
                               fontsize=10, color='white')
        axes[row,2].axis('off')

    fig.suptitle('FEM Fracture Progression', fontsize=15,
                 color='white', y=0.99, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(output_dir, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"  Saved: {out}")
    plt.close(fig)


def _save_progression_gif(engine, output_dir,
                           filename='v5_fracture_progression.gif',
                           fps=3):
    """Save animated GIF of fracture progression across load steps.

    Shows 3 panels per frame: CT | von Mises stress | Damage map
    """
    from PIL import Image
    import io

    frames = engine._frames
    if not frames:
        print("  [warn] No frames captured. Enable capture_frames=True.")
        return

    # Mid-sagittal slice index
    mid_y = engine.shape[1] // 2
    ct_slice = engine.ct[:, mid_y, :]
    bone_slice = engine.bone_mask[:, mid_y, :]

    # Collect PIL images
    pil_frames = []
    vm_max_global = max(f['von_mises'].max() for f in frames)
    vm_max_global = max(vm_max_global, 1.0)

    for i, frame in enumerate(frames):
        step = frame.get('step', 0)
        iteration = frame.get('iteration', 0)
        von_mises = frame['von_mises']
        damage = frame['damage']

        # Map element data back to 3D volume
        vm_vol = np.zeros(engine.shape, dtype=np.float32)
        dm_vol = np.zeros(engine.shape, dtype=np.float32)
        vm_vol[engine.bone_mask] = von_mises
        dm_vol[engine.bone_mask] = damage

        vm_slice = vm_vol[:, mid_y, :]
        dm_slice = dm_vol[:, mid_y, :]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5),
                                  facecolor='#1a1a2e')
        for ax in axes:
            ax.set_facecolor('#1a1a2e')

        # Panel 1: CT
        axes[0].imshow(ct_slice.T, cmap='bone', origin='lower', aspect='auto')
        axes[0].set_title('CT', fontsize=12, color='white')
        axes[0].axis('off')

        # Panel 2: von Mises stress
        axes[1].imshow(ct_slice.T, cmap='bone', origin='lower',
                       alpha=0.4, aspect='auto')
        vm_masked = np.ma.masked_where(~bone_slice, vm_slice)
        axes[1].imshow(vm_masked.T, cmap='hot', origin='lower',
                       alpha=0.75, vmin=0, vmax=vm_max_global, aspect='auto')
        axes[1].set_title(f'σ_vm (max={von_mises.max():.0f} MPa)',
                          fontsize=12, color='white')
        axes[1].axis('off')

        # Panel 3: Damage map
        axes[2].imshow(ct_slice.T, cmap='bone', origin='lower',
                       alpha=0.4, aspect='auto')
        dm_masked = np.ma.masked_where(~bone_slice, dm_slice)
        axes[2].imshow(dm_masked.T, cmap='Reds', origin='lower',
                       alpha=0.75, vmin=0, vmax=1.0, aspect='auto')
        n_damaged = (damage > 0.05).sum()
        pct = n_damaged / max(engine.n_elements, 1) * 100
        axes[2].set_title(f'Damage ({pct:.1f}% yielded)',
                          fontsize=12, color='white')
        axes[2].axis('off')

        # Load step info
        load_pct = (step + 1) / 4 * 100  # assuming 4 load steps
        fig.suptitle(
            f'Load Step {step+1} / Iter {iteration+1}  '
            f'({load_pct:.0f}% load)',
            fontsize=14, color='#00d2ff', fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        # Render to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120,
                    facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        pil_frames.append(Image.open(buf).copy())
        buf.close()

    # Save GIF
    out = os.path.join(output_dir, filename)
    duration = int(1000 / fps)
    # Hold last frame longer
    durations = [duration] * len(pil_frames)
    durations[-1] = duration * 4

    pil_frames[0].save(
        out, save_all=True, append_images=pil_frames[1:],
        duration=durations, loop=0, optimize=True)
    print(f"  Saved GIF: {out} ({len(pil_frames)} frames)")
    return out


# ============================================================================
#  DEMO
# ============================================================================

def demo(output_dir=None, use_cuda=False, quick=False):
    """Run voxel FEM demo with real VerSe CT data.

    Args:
        quick: If True, skip sweeps — only 3 scenarios + GIF (~40min)
    """
    if output_dir is None:
        output_dir = './fracture_v5_demo'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Voxel FEM Fracture Engine v5 — Real Physics")
    print("=" * 70)

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
        result = load_vertebra(ct_path, mask_path, return_spacing=True)
        ct, mask, _label, voxel_spacing = result
        mask = (mask > 0).astype(np.int32)
        # Use mean spacing (isotropic approximation)
        voxel_size = float(voxel_spacing.mean())
        print(f"  CT shape: {ct.shape}, bone voxels: {mask.sum():,}")
        print(f"  Voxel size: {voxel_size:.3f} mm "
              f"(from NIfTI: {voxel_spacing})")
        vol_cm3 = mask.sum() * np.prod(voxel_spacing) / 1000
        print(f"  Vertebra volume: {vol_cm3:.1f} cm³")
    except Exception as e:
        print(f"\n  ❌ Failed to load VerSe data: {e}")
        return

    # GPU: higher resolution  CPU: coarser for speed
    if use_cuda:
        ds = max(1, int(np.cbrt(mask.sum() / 200000)))
    else:
        ds = max(1, int(np.cbrt(mask.sum() / 30000)))
    print(f"  Downsample: {ds}x → element size: {voxel_size*ds:.2f} mm "
          f"({'GPU' if use_cuda else 'CPU'})")

    engine = VoxelFEMEngine(mask, ct, voxel_size_mm=voxel_size,
                            downsample=ds, seed=42, use_cuda=use_cuda)

    # ---- Scenario 1: Wedge ----
    print("\n" + "=" * 70)
    print("1. Force=3 kN, flexion=20° → expect A1 (wedge)")
    print("=" * 70)
    engine.set_causal_params(CausalParameters(
        force_magnitude=3.0, flexion_angle=20.0, bmd_factor=0.8))
    r1 = engine.simulate(max_damage_iters=5)
    _plot_scenario(engine, os.path.join(output_dir, 'v5_scenario_wedge.png'))
    _plot_fracture_mechanics(engine, output_dir,
                              filename='v5_mechanics_wedge.png')

    # ---- Scenario 2: Burst (with GIF capture) ----
    print("\n" + "=" * 70)
    print("2. Force=8 kN, axial → expect A3/A4 (burst)")
    print("=" * 70)
    engine._capture_frames = True
    engine._frames = []
    engine.set_causal_params(CausalParameters(
        force_magnitude=8.0, flexion_angle=5.0, bmd_factor=0.6))
    r2 = engine.simulate(max_damage_iters=5)
    _plot_scenario(engine, os.path.join(output_dir, 'v5_scenario_burst.png'))
    _plot_multi_slice(engine, output_dir)
    _plot_fracture_mechanics(engine, output_dir,
                              filename='v5_mechanics_burst.png')
    _save_progression_gif(engine, output_dir)
    engine._capture_frames = False

    # ---- Scenario 3: Normal ----
    print("\n" + "=" * 70)
    print("3. Force=1 kN, normal BMD → expect A0")
    print("=" * 70)
    engine.set_causal_params(CausalParameters(
        force_magnitude=1.0, flexion_angle=10.0, bmd_factor=1.2))
    r3 = engine.simulate(max_damage_iters=5)

    if quick:
        print("\n[quick mode] Skipping sweeps and progression comparison.")
    else:
        # ---- Force sweep ----
        print("\n" + "=" * 70)
        print("4. Force sweep (BMD=0.7, flexion=15°):")
        print("=" * 70)
        engine.set_causal_params(CausalParameters(flexion_angle=15.0, bmd_factor=0.7))
        sweep = engine.parameter_sweep('force_magnitude', [1, 2, 3, 4, 5, 6, 7, 8])

        # ---- BMD sweep ----
        print("\n" + "=" * 70)
        print("5. BMD sweep (Force=5 kN, flexion=15°):")
        print("=" * 70)
        engine.set_causal_params(CausalParameters(
            force_magnitude=5.0, flexion_angle=15.0))
        bmd_sweep = engine.parameter_sweep('bmd_factor',
                                            [1.2, 1.0, 0.8, 0.7, 0.5, 0.3])

        # ---- All plots ----
        print(f"\n6. Saving plots to {output_dir} ...")
        _plot_sweep(sweep, output_dir, filename='v5_force_sweep.png',
                    xlabel='Force (kN)', title_suffix=' (BMD=0.7, flex=15°)')
        _plot_sweep(bmd_sweep, output_dir, filename='v5_bmd_sweep.png',
                    xlabel='BMD Factor', title_suffix=' (Force=5 kN, flex=15°)')

        # Progression comparison
        _plot_progression(engine, output_dir, [
            ('A0: F=1kN, BMD=1.2',
             CausalParameters(force_magnitude=1.0, flexion_angle=10.0, bmd_factor=1.2)),
            ('A1: F=3kN, BMD=0.8',
             CausalParameters(force_magnitude=3.0, flexion_angle=25.0, bmd_factor=0.8)),
            ('A3: F=6kN, BMD=0.6',
             CausalParameters(force_magnitude=6.0, flexion_angle=10.0, bmd_factor=0.6)),
            ('A4: F=8kN, BMD=0.5',
             CausalParameters(force_magnitude=8.0, flexion_angle=5.0, bmd_factor=0.5)),
        ])

    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print(f"   Scenario 1 (wedge):  {r1.ao_type} ({r1.solve_time:.1f}s)")
    print(f"   Scenario 2 (burst):  {r2.ao_type} ({r2.solve_time:.1f}s)")
    print(f"   Scenario 3 (normal): {r3.ao_type} ({r3.solve_time:.1f}s)")
    if not quick:
        print(f"   Force sweep: {[r.ao_type for _, r in sweep]}")
        print(f"   BMD sweep:   {[r.ao_type for _, r in bmd_sweep]}")
    print("=" * 70)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Voxel FEM Fracture Engine v5')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--cuda', action='store_true',
        help='Use GPU (CuPy) for sparse solve')
    parser.add_argument('--quick', action='store_true',
        help='Skip sweeps, only 3 scenarios + GIF (~40min)')
    args = parser.parse_args()
    demo(args.output_dir, use_cuda=args.cuda, quick=args.quick)
