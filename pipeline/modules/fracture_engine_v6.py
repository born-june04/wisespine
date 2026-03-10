#!/usr/bin/env python3
"""
Voxel-Based Phase Field Fracture Engine (v6)
=============================================

Extends v5 with variational Phase Field Fracture (Bourdin-Francfort-Marigo).
Crack surfaces emerge naturally from energy minimization — no ad-hoc damage.

Physics:
    - Displacement field u: mechanical equilibrium ∇·[g(φ)·σ] = 0
    - Phase field φ ∈ [0,1]: crack field, 0=intact, 1=cracked
    - Staggered solver: alternate u-step (fix φ) and φ-step (fix u)
    - Tension-compression split: only tension drives cracks (Miehe 2010)
    - Spatially varying Gc: cortical bone (2-5 kJ/m²) vs trabecular (0.1-0.5 kJ/m²)

References:
    - Bourdin, Francfort, Marigo (2000) — variational fracture
    - Miehe, Welschinger, Hofacker (2010) — phase field + spectral split
    - Nalla et al. (2003) — bone fracture toughness Gc
    - Hao et al. (2019) — phase field for cortical bone

Usage:
    python fracture_engine_v6.py --cuda
"""

import os, sys, time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
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

# Import v5 base components
from fracture_engine_v5 import (
    hu_to_density, density_to_youngs_modulus, density_to_yield_stress,
    _gauss_points_3d, _shape_function_derivs, _elasticity_matrix,
    _transversely_isotropic_matrix, compute_reference_stiffness,
    CausalParameters, FEMResult, _AO_COLORS, HAS_MPL,
    E_MIN, NU_BONE, HU_CORTICAL_THRESHOLD,
)

# Phase field constants
# Unit conversion: 1 J/m² = 1 N/m = 10⁻³ N/mm
# FEM units: mm, N, MPa(=N/mm²), so Gc in N/mm
GC_CORTICAL = 3.0       # N/mm (= 3000 J/m²; Nalla 2003)
GC_TRABECULAR = 0.3     # N/mm (= 300 J/m²; estimate for cancellous)
RESIDUAL_STIFFNESS = 1e-6   # k in g(φ) = (1-φ)² + k


# ============================================================================
#  PHASE FIELD ENGINE
# ============================================================================

class PhaseFieldEngine:
    """Voxel FEM with Phase Field Fracture.
    
    Two coupled fields:
        u (displacement): 3 DOFs per node — mechanical equilibrium
        φ (phase field):  1 DOF per node — crack field [0,1]
    
    Solves via staggered scheme (Miehe 2010):
        1. Fix φ, solve for u: K_u · u = F
        2. Fix u, solve for φ: K_φ · φ = F_φ
        3. Repeat until convergence
    """
    
    def __init__(self, mask: np.ndarray, ct: np.ndarray,
                 voxel_size_mm: float = 1.0, downsample: int = 1,
                 seed: int = 42, use_cuda: bool = False):
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.ds = max(int(downsample), 1)
        self.use_cuda = use_cuda
        
        if self.use_cuda:
            try:
                import cupy as cp
                import cupyx.scipy.sparse as cp_sp
                import cupyx.scipy.sparse.linalg as cp_spla
                self._cp = cp; self._cp_sp = cp_sp; self._cp_spla = cp_spla
                print(f"  CUDA: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
            except ImportError:
                print("  [warn] CuPy not available, CPU fallback")
                self.use_cuda = False
        
        # Store originals
        self._orig_mask = mask
        self._orig_ct = ct
        self._orig_voxel_size = voxel_size_mm
        
        # Downsample
        if self.ds > 1:
            self.mask = (zoom(mask.astype(np.float32), 1/self.ds, order=0) > 0).astype(np.int32)
            self.ct = zoom(ct.astype(np.float32), 1/self.ds, order=1)
            self.h = voxel_size_mm * self.ds
        else:
            self.mask = mask.copy(); self.ct = ct.copy()
            self.h = voxel_size_mm
        
        self.shape = self.mask.shape
        self.bone_mask = self.mask > 0
        self.n_elements = int(self.bone_mask.sum())
        
        self._setup_mesh()
        self._setup_materials()
        self._setup_phase_field()
        
        self.params = None
        self._result = None
        self._frames = []
        self._capture_frames = False
        
        # Pre-cache non-bone nodes for φ BCs (avoid recomputing each iteration)
        bone_nodes = np.unique(self._elem_nodes.ravel())
        self._non_bone_nodes = np.setdiff1d(np.arange(self._n_dof_phi), bone_nodes)
        print(f"  Non-bone nodes (φ=0 constrained): {len(self._non_bone_nodes)}")
    
    # ================================================================
    #  MESH (same as v5)
    # ================================================================
    
    def _setup_mesh(self):
        t0 = time.time()
        nx, ny, nz = self.shape
        self._elem_ijk = np.argwhere(self.bone_mask)
        
        n_nodes_x = nx + 1
        n_nodes_y = ny + 1
        
        def node_id(i, j, k):
            return i + j * n_nodes_x + k * n_nodes_x * n_nodes_y
        
        corner_offsets = np.array([
            [0,0,0],[1,0,0],[1,1,0],[0,1,0],
            [0,0,1],[1,0,1],[1,1,1],[0,1,1],
        ])
        
        elem_nodes_list = []
        for off in corner_offsets:
            elem_nodes_list.append(node_id(
                self._elem_ijk[:,0]+off[0],
                self._elem_ijk[:,1]+off[1],
                self._elem_ijk[:,2]+off[2]))
        self._elem_nodes = np.column_stack(elem_nodes_list)
        
        # Mechanical DOFs (3 per node)
        self._elem_dofs = np.zeros((self.n_elements, 24), dtype=np.int64)
        for n in range(8):
            self._elem_dofs[:, n*3:n*3+3] = self._elem_nodes[:, n:n+1]*3 + np.arange(3)
        
        self._n_nodes = (nx+1)*(ny+1)*(nz+1)
        self._n_dof_u = self._n_nodes * 3
        self._n_dof_phi = self._n_nodes  # 1 DOF per node for φ
        
        # Reference stiffness
        self._Ke_ref = compute_reference_stiffness(self.h, NU_BONE, anisotropic=True)
        
        # Relative positions for BCs
        ijk = self._elem_ijk.astype(np.float32)
        self._elem_si = (ijk[:,2] - ijk[:,2].min()) / max(ijk[:,2].ptp(), 1)
        self._elem_ap = (ijk[:,1] - ijk[:,1].min()) / max(ijk[:,1].ptp(), 1)
        self._elem_lr = (ijk[:,0] - ijk[:,0].min()) / max(ijk[:,0].ptp(), 1)
        
        print(f"  Mesh: {self.n_elements} elements, {self._n_nodes} nodes, "
              f"DOF_u={self._n_dof_u}, DOF_φ={self._n_dof_phi} ({time.time()-t0:.1f}s)")
    
    # ================================================================
    #  MATERIALS (same as v5)
    # ================================================================
    
    def _setup_materials(self):
        ct_vals = self.ct[self.bone_mask]
        self._rho = np.clip(ct_vals / 1000.0, 0.001, 2.0)
        
        E_trab = 6850.0 * np.power(self._rho, 1.49)
        E_cort = 10500.0 * np.power(self._rho, 2.29)
        
        # Cortical fraction: high at SURFACE (low distance), low in INTERIOR
        # sigmoid(−x) gives cf≈1 at surface, cf≈0 deep inside
        dist = distance_transform_edt(self.bone_mask) * self.h
        cf = 1.0 / (1.0 + np.exp((dist[self.bone_mask] - 0.5) / 0.3))
        self._cortical_fraction = cf
        
        self._E_base = np.clip((1-cf)*E_trab + cf*E_cort, E_MIN, 20000.0)
        
        # Spatially varying Gc (already in N/mm from constants)
        self._Gc = (1-cf) * GC_TRABECULAR + cf * GC_CORTICAL
        
        print(f"  Materials: E=[{self._E_base.min():.0f}, {self._E_base.max():.0f}] MPa, "
              f"Gc=[{self._Gc.min():.3f}, {self._Gc.max():.3f}] N/mm")
    
    # ================================================================
    #  PHASE FIELD SETUP
    # ================================================================
    
    def _setup_phase_field(self):
        """Setup phase field scalar DOFs and Laplacian stiffness."""
        t0 = time.time()
        
        # Phase field regularization length
        self._l0 = 2.5 * self.h  # ~2.5× element size
        
        # Precompute reference element matrices for scalar field
        # K_phi_e = ∫ (Gc·l0·∇N^T·∇N + Gc/l0·N^T·N) dV
        # Uses FULL 2×2×2 Gauss quadrature for both terms (consistency)
        
        gp, gw = _gauss_points_3d()
        invJ = np.diag([2.0/self.h]*3)
        detJ = (self.h / 2.0) ** 3
        
        # K_diff_ref = ∫ (∇N)^T · (∇N) dV  (diffusion, coefficient=1)
        K_diff = np.zeros((8, 8))
        for (xi, eta, zeta), w in zip(gp, gw):
            dN = _shape_function_derivs(xi, eta, zeta)
            dN_phys = invJ @ dN  # (3, 8)
            K_diff += w * (dN_phys.T @ dN_phys) * detJ
        self._K_diff_ref = K_diff  # (8, 8)
        
        # K_mass_ref = ∫ N^T · N dV  (reaction part, for Gc/l0=1)
        # Reuse gp, gw, detJ from K_diff computation above
        M = np.zeros((8, 8))
        for (xi, eta, zeta), w in zip(gp, gw):
            # Shape functions at this point
            N = np.array([
                (1-xi)*(1-eta)*(1-zeta)/8, (1+xi)*(1-eta)*(1-zeta)/8,
                (1+xi)*(1+eta)*(1-zeta)/8, (1-xi)*(1+eta)*(1-zeta)/8,
                (1-xi)*(1-eta)*(1+zeta)/8, (1+xi)*(1-eta)*(1+zeta)/8,
                (1+xi)*(1+eta)*(1+zeta)/8, (1-xi)*(1+eta)*(1+zeta)/8,
            ])
            M += w * np.outer(N, N) * detJ
        self._K_mass_ref = M  # (8, 8)
        
        print(f"  Phase field: l₀={self._l0:.2f}mm ({time.time()-t0:.1f}s)")
    
    # ================================================================
    #  MECHANICAL ASSEMBLY (degraded by φ)
    # ================================================================
    
    def _assemble_mech_stiffness(self, E_elem, phi_elem):
        """Assemble K_u with degradation g(φ) = (1-φ)² + k."""
        g_phi = (1.0 - phi_elem)**2 + RESIDUAL_STIFFNESS
        E_degraded = E_elem * g_phi
        
        Ke_flat = self._Ke_ref.ravel()
        li, lj = np.meshgrid(np.arange(24), np.arange(24), indexing='ij')
        rows = self._elem_dofs[:, li.ravel()]
        cols = self._elem_dofs[:, lj.ravel()]
        vals = E_degraded[:, None] * Ke_flat[None, :]
        
        return sp.coo_matrix(
            (vals.ravel(), (rows.ravel(), cols.ravel())),
            shape=(self._n_dof_u, self._n_dof_u)).tocsr()
    
    # ================================================================
    #  STRAIN ENERGY: TENSION-COMPRESSION SPLIT
    # ================================================================
    
    def _compute_strain_energy(self, u, E_field=None):
        """Compute tensile strain energy ψ⁺ per element (Amor vol-dev split).
        
        Amor et al. (2009) volumetric-deviatoric decomposition:
            ψ⁺ = ½ K ⟨tr(ε)⟩²₊ + μ (ε_dev : ε_dev)
            ψ⁻ = ½ K ⟨tr(ε)⟩²₋
        Only ψ⁺ drives crack evolution. Computationally cheaper than
        full Miehe spectral decomposition while physically appropriate
        for quasi-brittle materials like bone.
        """
        # B matrix at center
        if not hasattr(self, '_B_center'):
            dN = _shape_function_derivs(0, 0, 0)
            invJ = np.diag([2.0/self.h]*3)
            dN_phys = invJ @ dN
            B = np.zeros((6, 24))
            for n in range(8):
                B[0, n*3+0] = dN_phys[0, n]
                B[1, n*3+1] = dN_phys[1, n]
                B[2, n*3+2] = dN_phys[2, n]
                B[3, n*3+0] = dN_phys[1, n]; B[3, n*3+1] = dN_phys[0, n]
                B[4, n*3+1] = dN_phys[2, n]; B[4, n*3+2] = dN_phys[1, n]
                B[5, n*3+0] = dN_phys[2, n]; B[5, n*3+2] = dN_phys[0, n]
            self._B_center = B
        
        # Element displacements → strain (vectorized)
        u_elem = u[self._elem_dofs]  # (n_elem, 24)
        strain = u_elem @ self._B_center.T  # (n_elem, 6)
        
        # Voigt strain → tensor for each element
        # [εxx, εyy, εzz, γxy, γyz, γxz]
        exx = strain[:, 0]; eyy = strain[:, 1]; ezz = strain[:, 2]
        exy = strain[:, 3]/2; eyz = strain[:, 4]/2; exz = strain[:, 5]/2
        
        # Strain eigenvalues (principal strains) — vectorized 3×3 eigendecomp
        # For efficiency, use trace-based decomposition
        tr_eps = exx + eyy + ezz
        tr_eps_pos = np.maximum(tr_eps, 0)  # ⟨tr(ε)⟩₊
        tr_eps_neg = np.minimum(tr_eps, 0)  # ⟨tr(ε)⟩₋
        
        # Deviatoric strain squared: tr(ε²) = εxx² + εyy² + εzz² + 2(εxy² + εyz² + εxz²)
        tr_eps2 = exx**2 + eyy**2 + ezz**2 + 2*(exy**2 + eyz**2 + exz**2)
        
        # Material constants per element
        # Use E_field if provided (BMD-adjusted), otherwise base
        E = E_field if E_field is not None else self._E_base
        nu = NU_BONE
        lam = E * nu / ((1+nu) * (1-2*nu))  # Lamé first parameter
        mu = E / (2*(1+nu))                   # Shear modulus
        
        # Amor volumetric-deviatoric split:
        # ψ⁺ = ½ K ⟨tr(ε)⟩²₊ + μ (ε_dev : ε_dev)
        # ψ⁻ = ½ K ⟨tr(ε)⟩²₋
        K_bulk = lam + 2*mu/3  # bulk modulus
        
        # Deviatoric part: ε_dev = ε - ⅓tr(ε)·I
        # tr(ε_dev²) = tr(ε²) - ⅓tr(ε)²
        tr_edev2 = tr_eps2 - tr_eps**2 / 3.0
        tr_edev2 = np.maximum(tr_edev2, 0)
        
        # ψ⁺ (volumetric tension + all deviatoric)
        psi_plus = 0.5 * K_bulk * tr_eps_pos**2 + mu * tr_edev2
        
        # Compute von Mises stress (vectorized — no Python loop)
        D = _elasticity_matrix(1.0, nu)  # reference D for E=1
        stress_all = (strain @ D.T) * E[:, None]  # scale by per-element E
        s = stress_all
        von_mises = np.sqrt(0.5*((s[:,0]-s[:,1])**2 + (s[:,1]-s[:,2])**2 + 
                                  (s[:,2]-s[:,0])**2 + 6*(s[:,3]**2+s[:,4]**2+s[:,5]**2)))
        
        return psi_plus, von_mises, strain
    
    # ================================================================
    #  PHASE FIELD ASSEMBLY
    # ================================================================
    
    def _assemble_phase_system(self, psi_plus, history):
        """Assemble phase field system: K_φ · φ = F_φ.
        
        Equation per element:
            (Gc/l0 + 2·H) · φ - Gc·l0·∇²φ = 2·H
        where H = max(ψ⁺) over history (irreversibility).
        
        In FEM form:
            K_φ = Gc·l0·K_diff + (Gc/l0 + 2·H)·K_mass
            F_φ = 2·H · f_mass
        """
        Gc = self._Gc   # per element
        l0 = self._l0
        
        # History variable: enforce irreversibility
        H = np.maximum(psi_plus, history)  # (n_elem,)
        
        # Per-element coefficients
        diff_coeff = Gc * l0                    # diffusion coefficient
        react_coeff = Gc / l0 + 2.0 * H        # reaction coefficient
        rhs_coeff = 2.0 * H                    # right-hand side
        
        # Assembly (8×8 per element, vectorized)
        Kd_flat = self._K_diff_ref.ravel()   # (64,)
        Km_flat = self._K_mass_ref.ravel()   # (64,)
        
        li, lj = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')
        li_f, lj_f = li.ravel(), lj.ravel()
        
        # Global node indices for phase field (1 DOF per node)
        phi_dofs = self._elem_nodes  # (n_elem, 8) — node IDs = φ DOF IDs
        
        rows = phi_dofs[:, li_f]  # (n_elem, 64)
        cols = phi_dofs[:, lj_f]
        
        # Values: diff_coeff[e] * Kd + react_coeff[e] * Km
        vals = (diff_coeff[:, None] * Kd_flat[None, :] + 
                react_coeff[:, None] * Km_flat[None, :])
        
        K_phi = sp.coo_matrix(
            (vals.ravel(), (rows.ravel(), cols.ravel())),
            shape=(self._n_dof_phi, self._n_dof_phi)).tocsr()
        
        # RHS: F_φ = ∫ 2H · N dV = rhs_coeff[e] × M_ref × [1,1,...,1] per element
        N_int = self._K_mass_ref.sum(axis=1)  # ∫ N_i dV for each node (8,)
        F_phi = np.zeros(self._n_dof_phi)
        rhs_vals = rhs_coeff[:, None] * N_int[None, :]  # (n_elem, 8)
        np.add.at(F_phi, phi_dofs, rhs_vals)
        
        return K_phi, F_phi, H
    
    # ================================================================
    #  BOUNDARY CONDITIONS (same approach as v5)
    # ================================================================
    
    def _apply_boundary_conditions(self, K, F, params, load_fraction=1.0):
        """Same parabolic contact BCs as v5."""
        PENALTY = K.diagonal().max() * 1e6
        
        sup_mask = self._elem_si > 0.88
        inf_mask = self._elem_si < 0.12
        
        # Fix inferior z-DOFs
        inf_elems = np.where(inf_mask)[0]
        inf_dofs_all = self._elem_dofs[inf_elems]
        z_idx = np.arange(2, 24, 3)
        inf_z_dofs = np.unique(inf_dofs_all[:, z_idx].ravel())
        diag = K.diagonal(); diag[inf_z_dofs] += PENALTY; K.setdiag(diag)
        F[inf_z_dofs] = 0.0
        
        # Anchor central bottom node
        if len(inf_elems) > 0:
            ce = inf_elems[len(inf_elems)//2]
            for d in [self._elem_dofs[ce,0], self._elem_dofs[ce,1]]:
                diag = K.diagonal(); diag[d] += PENALTY; K.setdiag(diag); F[d] = 0.0
        
        # Parabolic pressure on superior
        force_N = params.force_magnitude * 1000.0 * load_fraction
        flex_rad = np.radians(params.flexion_angle)
        sup_elems = np.where(sup_mask)[0]
        if len(sup_elems) == 0: return K, F
        
        ap = self._elem_ap[sup_elems]; lr = self._elem_lr[sup_elems]
        r2 = (ap-0.5)**2 + (lr-0.5)**2
        parabolic = np.clip(1 - r2/(r2.max()+1e-6), 0.1, 1.0)
        flex_w = np.clip(1 + np.sin(flex_rad)*(ap-0.5)*2, 0.2, 2.5)
        w = parabolic * flex_w; w /= w.sum()
        f_per = -force_N * w
        
        sup_dofs = self._elem_dofs[sup_elems]
        for n in range(4, 8):
            np.add.at(F, sup_dofs[:, n*3+2], f_per/4.0)
        
        return K, F
    
    def _apply_phi_boundary_conditions(self, K_phi, F_phi):
        """Constrain φ=0 at nodes NOT connected to any bone element.
        
        Prevents crack field from leaking into empty space.
        Uses pre-cached non-bone node list for performance.
        """
        non_bone = self._non_bone_nodes
        if len(non_bone) > 0:
            PENALTY = K_phi.diagonal().max() * 1e6
            diag = K_phi.diagonal()
            diag[non_bone] += PENALTY
            K_phi.setdiag(diag)
            F_phi[non_bone] = 0.0
        
        return K_phi, F_phi
    
    # ================================================================
    #  SPARSE SOLVE
    # ================================================================
    
    def _solve_sparse(self, K, F, label=""):
        """Solve Kx=F using GPU or CPU."""
        t0 = time.time()
        n = K.shape[0]
        print(f"        [{label}] Solving {n} DOFs...", end='', flush=True)
        if self.use_cuda:
            Kg = self._cp_sp.csr_matrix(K)
            Fg = self._cp.array(F)
            try:
                # Use CG for φ system (SPD, faster) and spsolve for u
                if label == "φ":
                    xg, info = self._cp_spla.cg(Kg, Fg, maxiter=3000, tol=1e-8)
                else:
                    xg = self._cp_spla.spsolve(Kg, Fg)
                x = self._cp.asnumpy(xg)
            except:
                xg, _ = self._cp_spla.cg(Kg, Fg, maxiter=5000, tol=1e-8)
                x = self._cp.asnumpy(xg)
        else:
            try:
                if label == "φ":
                    # φ system is SPD → CG is optimal
                    x, info = spla.cg(K, F, maxiter=3000, tol=1e-8)
                    if info != 0: x = spla.spsolve(K, F)
                else:
                    ilu = spla.spilu(K.tocsc(), fill_factor=5)
                    M = spla.LinearOperator(K.shape, ilu.solve)
                    x, info = spla.cg(K, F, M=M, maxiter=3000, tol=1e-8)
                    if info != 0: x = spla.spsolve(K, F)
            except:
                x = spla.spsolve(K, F)
        dt = time.time() - t0
        print(f" {dt:.1f}s")
        return x, dt
    
    # ================================================================
    #  ELEMENT-TO-NODE φ MAPPING
    # ================================================================
    
    def _elem_phi(self, phi_nodal):
        """Average nodal φ to element centers."""
        return phi_nodal[self._elem_nodes].mean(axis=1)
    
    def _to_3d(self, elem_data):
        """Map per-element data to 3D volume (upsampled to original res)."""
        vol = np.zeros(self.shape, dtype=np.float32)
        vol[self._elem_ijk[:,0], self._elem_ijk[:,1], self._elem_ijk[:,2]] = elem_data
        if self.ds > 1:
            vol = zoom(vol, self.ds, order=1)
            target = self._orig_mask.shape
            vol = vol[:target[0], :target[1], :target[2]]
        return vol
    
    def _phi_to_3d(self, phi_nodal):
        """Map nodal φ to 3D volume."""
        phi_elem = self._elem_phi(phi_nodal)
        return self._to_3d(phi_elem)
    
    # ================================================================
    #  AO CLASSIFICATION (reuse from v5)
    # ================================================================
    
    def _classify_ao(self, phi_nodal, u):
        """Classify fracture type using phase field crack pattern."""
        phi_e = self._elem_phi(phi_nodal)
        damage_mask = phi_e > 0.5  # cracked = φ > 0.5
        
        n_yield = damage_mask.sum()
        frac = n_yield / self.n_elements
        
        # Regional analysis
        ant = self._elem_ap > 0.5
        post = self._elem_ap < 0.5
        mid_third = (self._elem_si > 0.33) & (self._elem_si < 0.67)
        
        ant_dam = phi_e[ant].mean() if ant.sum() > 0 else 0
        post_dam = phi_e[post].mean() if post.sum() > 0 else 0
        
        # Displacement-based height loss
        uz_elem = np.zeros(self.n_elements)
        for n in range(8):
            uz_elem += u[self._elem_dofs[:, n*3+2]]
        uz_elem /= 8.0
        
        sup = self._elem_si > 0.8
        uz_top = uz_elem[sup].mean() if sup.sum() > 0 else 0
        inf_m = self._elem_si < 0.2
        uz_bot = uz_elem[inf_m].mean() if inf_m.sum() > 0 else 0
        height_loss = abs(uz_top - uz_bot) / max(self._elem_ijk[:,2].ptp() * self.h, 1) 
        
        # Canal compromise
        post_mid = post & mid_third
        canal = phi_e[post_mid].mean() if post_mid.sum() > 0 else 0
        
        # Classification
        if frac < 0.02:
            ao, conf = 'A0', 1-frac/0.02
        elif ant_dam > post_dam * 1.5 and canal < 0.15:
            ao, conf = 'A1', min(ant_dam/max(post_dam,0.01), 3)/3
        elif frac < 0.35 and canal < 0.3:
            ao, conf = 'A2', frac/0.35
        elif canal >= 0.3 and canal < 0.6:
            ao, conf = 'A3', canal/0.6
        else:
            ao, conf = 'A4', min(frac + canal, 1.0)
        
        return FEMResult(
            ao_type=ao, confidence=float(np.clip(conf,0,1)),
            max_von_mises=0, max_displacement=float(np.abs(u).max()),
            n_yielded=int(n_yield), n_elements=self.n_elements,
            yielded_fraction=float(frac),
            anterior_height_loss=float(height_loss),
            posterior_height_loss=0, canal_compromise=float(canal),
            n_fragments=int((phi_e > 0.9).sum()),
        )
    
    # ================================================================
    #  MAIN SIMULATION
    # ================================================================
    
    def set_causal_params(self, params: CausalParameters):
        params.validate()
        self.params = params
    
    def simulate(self, n_load_steps: int = 4, max_stagger_iters: int = 5,
                 verbose: bool = True) -> FEMResult:
        """Phase Field Fracture simulation with staggered solver.
        
        For each load step:
            Repeat until convergence:
                1. u-step: fix φ, solve mechanical equilibrium
                2. φ-step: fix u, solve phase field evolution
        """
        params = self.params
        if params is None:
            raise ValueError("Call set_causal_params() first.")
        
        t_start = time.time()
        
        # Material with BMD factor
        rho_mod = self._rho * params.bmd_factor
        rho_clip = np.clip(rho_mod, 0.01, 2.0)
        E_trab = 6850.0 * np.power(rho_clip, 1.49)
        E_cort = 10500.0 * np.power(rho_clip, 2.29)
        cf = self._cortical_fraction
        E_base = np.clip((1-cf)*E_trab + cf*E_cort, E_MIN, 20000.0)
        
        # Initialize fields
        phi = np.zeros(self._n_dof_phi)           # phase field
        u = np.zeros(self._n_dof_u)               # displacement
        history = np.zeros(self.n_elements)         # max ψ⁺ history
        total_iters = 0
        
        for step in range(n_load_steps):
            load_frac = (step + 1) / n_load_steps
            if verbose:
                print(f"\n  Load step {step+1}/{n_load_steps} "
                      f"({load_frac*100:.0f}% of {params.force_magnitude:.1f} kN):")
            
            for si in range(max_stagger_iters):
                total_iters += 1
                
                # ---- U-STEP: Mechanical equilibrium ----
                t_asm = time.time()
                phi_elem = self._elem_phi(phi)
                K_u = self._assemble_mech_stiffness(E_base, phi_elem)
                F_u = np.zeros(self._n_dof_u)
                K_u, F_u = self._apply_boundary_conditions(K_u, F_u, params, load_frac)
                if verbose:
                    print(f"      K_u assembled ({time.time()-t_asm:.1f}s)")
                u_new, t_u = self._solve_sparse(K_u, F_u, "u")
                
                # ---- STRAIN ENERGY (use BMD-adjusted E) ----
                t_psi = time.time()
                psi_plus, von_mises, _ = self._compute_strain_energy(u_new, E_field=E_base)
                if verbose:
                    print(f"      ψ⁺ computed ({time.time()-t_psi:.1f}s), max={psi_plus.max():.2f}")
                
                # ---- φ-STEP: Phase field evolution ----
                t_asm2 = time.time()
                K_phi, F_phi, history = self._assemble_phase_system(psi_plus, history)
                K_phi, F_phi = self._apply_phi_boundary_conditions(K_phi, F_phi)
                if verbose:
                    print(f"      K_φ assembled ({time.time()-t_asm2:.1f}s)")
                phi_new, t_phi = self._solve_sparse(K_phi, F_phi, "φ")
                
                # Enforce bounds and irreversibility
                phi_new = np.clip(phi_new, 0, 1)
                phi_new = np.maximum(phi_new, phi)  # irreversibility
                
                # Convergence check
                du = np.abs(u_new - u).max()
                dphi = np.abs(phi_new - phi).max()
                
                u = u_new
                phi = phi_new
                
                phi_elem = self._elem_phi(phi)
                n_cracked = (phi_elem > 0.5).sum()
                max_phi = phi.max()
                
                if verbose:
                    print(f"    Stagger {si+1}: u({t_u:.1f}s) φ({t_phi:.1f}s) | "
                          f"Δu={du:.4f} Δφ={dphi:.4f} | "
                          f"φ_max={max_phi:.3f} cracked={n_cracked}")
                
                # Capture frame
                if self._capture_frames:
                    self._frames.append({
                        'step': step, 'iteration': si,
                        'phi': phi.copy(), 'u': u.copy(),
                        'von_mises': von_mises.copy(),
                        'psi_plus': psi_plus.copy(),
                    })
                
                if du < 1e-4 and dphi < 1e-4:
                    if verbose: print(f"    Converged.")
                    break
        
        total_time = time.time() - t_start
        
        # Store results
        self._phi = phi
        self._displacement = u
        self._von_mises = von_mises
        self._history = history
        
        result = self._classify_ao(phi, u)
        result.solve_time = total_time
        result.n_iterations = total_iters
        result.max_von_mises = float(von_mises.max())
        self._result = result
        
        if verbose:
            print(f"\n  ★ {result.ao_type} ({total_time:.1f}s, {total_iters} iters)")
            print(f"    φ_max={phi.max():.3f}, cracked elements: "
                  f"{(self._elem_phi(phi)>0.5).sum()}/{self.n_elements}")
            print(result.summary())
        
        return result


# ============================================================================
#  DEMO
# ============================================================================

def demo(use_cuda=False, quick=False):
    """Run Phase Field Fracture demo."""
    from _gen_real_fracture_visuals import load_vertebra
    
    print("=" * 70)
    print("WiseSpine v6 — Phase Field Fracture Engine")
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
    
    # Phase field is 2× more expensive (u + φ per iter), so target fewer elements
    ds = max(2, int(np.cbrt(mask.sum() / 100000))) if use_cuda else \
         max(2, int(np.cbrt(mask.sum() / 20000)))
    print(f"  Downsample: ds={ds} (target ≤100k elements for phase field)")
    
    engine = PhaseFieldEngine(mask, ct, voxel_size_mm=voxel_size,
                              downsample=ds, seed=42, use_cuda=use_cuda)
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'fracture_v6_demo')
    os.makedirs(out_dir, exist_ok=True)
    
    # Test scenarios
    scenarios = [
        ('wedge', CausalParameters(3.0, 20.0, 0.0, 0.8)),
        ('burst', CausalParameters(8.0, 5.0, 0.0, 0.5)),
    ]
    
    for name, params in scenarios:
        print(f"\n{'='*60}")
        print(f"  Scenario: {name}")
        print(f"{'='*60}")
        
        engine.set_causal_params(params)
        engine._capture_frames = True
        engine._frames = []
        result = engine.simulate()
        engine._capture_frames = False
        
        if HAS_MPL:
            _plot_phase_field_result(engine, name, out_dir)
    
    print(f"\nOutputs saved to {out_dir}/")


def _plot_phase_field_result(engine, scenario_name, out_dir):
    """Visualize phase field crack surfaces."""
    phi = engine._phi
    u = engine._displacement
    phi_vol = engine._phi_to_3d(phi)
    
    ct_orig = engine._orig_ct
    mask_orig = engine._orig_mask
    if ct_orig.shape != phi_vol.shape:
        from scipy.ndimage import zoom
        ct_show = zoom(ct_orig.astype(np.float32),
                       np.array(phi_vol.shape)/np.array(ct_orig.shape), order=1)
    else:
        ct_show = ct_orig
    
    mid_x = phi_vol.shape[0] // 2
    mid_y = phi_vol.shape[1] // 2
    mid_z = phi_vol.shape[2] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='#0d1117')
    
    kw = dict(origin='lower', aspect='auto')
    
    # Row 1: CT + crack overlay
    for ax, (sl, plane) in zip(axes[0], [
        (ct_show[mid_x].T, 'Sagittal'),
        (ct_show[:, mid_y, :].T, 'Coronal'),
        (ct_show[:, :, mid_z].T, 'Axial'),
    ]):
        ax.set_facecolor('#0d1117')
        ax.imshow(sl, cmap='bone', vmin=-200, vmax=800, **kw)
        ax.set_title(f'{plane}', color='white', fontsize=12)
        ax.axis('off')
    
    # Overlay crack (φ > 0.3) as colored band
    phi_slices = [phi_vol[mid_x].T, phi_vol[:, mid_y, :].T, phi_vol[:, :, mid_z].T]
    bone_slices = [(mask_orig[mid_x]>0).T if mask_orig.shape==phi_vol.shape else
                   ((zoom(mask_orig.astype(float), 
                          np.array(phi_vol.shape)/np.array(mask_orig.shape), order=0)
                    )[mid_x]>0).T,
                   None, None]
    
    for ax, phi_sl in zip(axes[0], phi_slices):
        phi_masked = np.ma.masked_where(phi_sl < 0.1, phi_sl)
        ax.imshow(phi_masked, cmap='hot', vmin=0, vmax=1, alpha=0.7, **kw)
        # Crack front contour
        if phi_sl.max() > 0.3:
            ax.contour(phi_sl, levels=[0.5], colors=['cyan'], linewidths=2, **{k:v for k,v in kw.items() if k=='origin'})
    
    # Row 2: Phase field only (crack surface map)
    for ax, (phi_sl, plane) in zip(axes[1], [
        (phi_vol[mid_x].T, 'φ Sagittal'),
        (phi_vol[:, mid_y, :].T, 'φ Coronal'),
        (phi_vol[:, :, mid_z].T, 'φ Axial'),
    ]):
        ax.set_facecolor('#0d1117')
        ax.imshow(phi_sl, cmap='inferno', vmin=0, vmax=1, **kw)
        if phi_sl.max() > 0.3:
            ax.contour(phi_sl, levels=[0.3, 0.5, 0.8], colors=['cyan', 'white', 'red'],
                       linewidths=[1, 2, 1], **{k:v for k,v in kw.items() if k=='origin'})
        ax.set_title(f'{plane} — crack surface', color='white', fontsize=12)
        ax.axis('off')
    
    r = engine._result
    fig.suptitle(f'Phase Field Fracture — {scenario_name.upper()} | '
                 f'{r.ao_type} (yield={r.yielded_fraction*100:.0f}%, '
                 f'φ_max={phi.max():.2f})',
                 fontsize=16, color='white', fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(out_dir, f'v6_phasefield_{scenario_name}.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='WiseSpine v6 Phase Field Fracture')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    demo(use_cuda=args.cuda, quick=args.quick)
