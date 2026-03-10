#!/usr/bin/env python3
"""
Full FEM Fracture Engine (v7)
==============================

Unified Phase Field + Explicit Dynamics + Element Erosion.
Bone actually cracks and fragments separate physically.

Pipeline per timestep:
    1. Phase field φ evolution (implicit, cheap scalar solve)
    2. Element erosion: φ > 0.95 → remove element → actual gap
    3. Internal force: F_int = B^T · g(φ) · σ  (vectorized, no matrix assembly)
    4. Contact detection: penalty forces prevent fragment overlap
    5. Explicit dynamics: a = (F_ext - F_int + F_contact) / m

References:
    Borden et al. (2012) — Phase field + explicit dynamics
    Hofacker & Miehe (2013) — Dynamic phase field fracture

Usage:
    python fracture_engine_v7.py --cuda
"""

import os, sys, time, io
import numpy as np
from scipy.ndimage import zoom, distance_transform_edt, label as ndimage_label
import scipy.sparse as sp
import scipy.sparse.linalg as spla

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
    _gauss_points_3d, _shape_function_derivs, _elasticity_matrix,
    compute_reference_stiffness, CausalParameters, FEMResult, _AO_COLORS,
    E_MIN, NU_BONE,
)

# Phase field constants (N/mm units)
GC_CORTICAL = 3.0       # N/mm (= 3000 J/m²)
GC_TRABECULAR = 0.3     # N/mm (= 300 J/m²)
RESIDUAL_K = 1e-6        # residual stiffness in g(φ)
EROSION_THRESHOLD = 0.95  # φ above this → element removed


class FullFractureEngine:
    """Unified Phase Field + Explicit Dynamics fracture engine.
    
    Bone cracks (Phase Field), elements erode (actual gaps),
    and fragments separate physically (explicit dynamics).
    """
    
    def __init__(self, mask, ct, voxel_size_mm=1.0, downsample=4,
                 seed=42, use_cuda=False):
        self.seed = seed
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
        self._setup_explicit()
        
        self.params = None
        self._frames = []
        
        print(f"  Ready: {self.n_elements} elements, h={self.h:.2f}mm")
    
    # ================================================================
    #  MESH
    # ================================================================
    
    def _setup_mesh(self):
        t0 = time.time()
        nx, ny, nz = self.shape
        self._elem_ijk = np.argwhere(self.bone_mask)
        
        n_nodes_x, n_nodes_y = nx + 1, ny + 1
        
        def node_id(i, j, k):
            return i + j * n_nodes_x + k * n_nodes_x * n_nodes_y
        
        offsets = np.array([
            [0,0,0],[1,0,0],[1,1,0],[0,1,0],
            [0,0,1],[1,0,1],[1,1,1],[0,1,1]])
        
        nodes = []
        for off in offsets:
            nodes.append(node_id(
                self._elem_ijk[:,0]+off[0],
                self._elem_ijk[:,1]+off[1],
                self._elem_ijk[:,2]+off[2]))
        self._elem_nodes = np.column_stack(nodes)
        
        self._elem_dofs = np.zeros((self.n_elements, 24), dtype=np.int64)
        for n in range(8):
            self._elem_dofs[:, n*3:n*3+3] = self._elem_nodes[:, n:n+1]*3 + np.arange(3)
        
        self._n_nodes = (nx+1)*(ny+1)*(nz+1)
        self._n_dof_u = self._n_nodes * 3
        self._n_dof_phi = self._n_nodes
        
        # Position info for BCs
        ijk = self._elem_ijk.astype(np.float32)
        self._elem_si = (ijk[:,2] - ijk[:,2].min()) / max(ijk[:,2].ptp(), 1)
        self._elem_ap = (ijk[:,1] - ijk[:,1].min()) / max(ijk[:,1].ptp(), 1)
        self._elem_lr = (ijk[:,0] - ijk[:,0].min()) / max(ijk[:,0].ptp(), 1)
        
        # B matrix (strain-displacement) at element center
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
        self._B = B  # (6, 24)
        
        # Element volume
        self._V_elem = self.h ** 3
        
        print(f"  Mesh: {self.n_elements} elem, {self._n_nodes} nodes, "
              f"DOF_u={self._n_dof_u} ({time.time()-t0:.1f}s)")
    
    # ================================================================
    #  MATERIALS
    # ================================================================
    
    def _setup_materials(self):
        ct_vals = self.ct[self.bone_mask]
        self._rho = np.clip(ct_vals / 1000.0, 0.001, 2.0)
        
        E_trab = 6850.0 * np.power(self._rho, 1.49)
        E_cort = 10500.0 * np.power(self._rho, 2.29)
        
        dist = distance_transform_edt(self.bone_mask) * self.h
        cf = 1.0 / (1.0 + np.exp((dist[self.bone_mask] - 0.5) / 0.3))
        self._cortical_fraction = cf
        
        self._E_base = np.clip((1-cf)*E_trab + cf*E_cort, E_MIN, 20000.0)
        self._Gc = (1-cf) * GC_TRABECULAR + cf * GC_CORTICAL
        
        # Elasticity matrix (reference, E=1)
        self._D_ref = _elasticity_matrix(1.0, NU_BONE)
        
        print(f"  Materials: E=[{self._E_base.min():.0f}, {self._E_base.max():.0f}] MPa, "
              f"Gc=[{self._Gc.min():.3f}, {self._Gc.max():.3f}] N/mm")
    
    # ================================================================
    #  PHASE FIELD SETUP
    # ================================================================
    
    def _setup_phase_field(self):
        self._l0 = 2.5 * self.h
        
        gp, gw = _gauss_points_3d()
        invJ = np.diag([2.0/self.h]*3)
        detJ = (self.h / 2.0) ** 3
        
        K_diff = np.zeros((8, 8))
        K_mass = np.zeros((8, 8))
        for (xi, eta, zeta), w in zip(gp, gw):
            dN = _shape_function_derivs(xi, eta, zeta)
            dN_phys = invJ @ dN
            K_diff += w * (dN_phys.T @ dN_phys) * detJ
            
            N = np.array([
                (1-xi)*(1-eta)*(1-zeta)/8, (1+xi)*(1-eta)*(1-zeta)/8,
                (1+xi)*(1+eta)*(1-zeta)/8, (1-xi)*(1+eta)*(1-zeta)/8,
                (1-xi)*(1-eta)*(1+zeta)/8, (1+xi)*(1-eta)*(1+zeta)/8,
                (1+xi)*(1+eta)*(1+zeta)/8, (1-xi)*(1+eta)*(1+zeta)/8,
            ])
            K_mass += w * np.outer(N, N) * detJ
        
        self._K_diff_ref = K_diff
        self._K_mass_ref = K_mass
        
        # Pre-cache non-bone nodes
        bone_nodes = np.unique(self._elem_nodes.ravel())
        self._non_bone_nodes = np.setdiff1d(np.arange(self._n_dof_phi), bone_nodes)
        
        print(f"  Phase field: l₀={self._l0:.2f}mm")
    
    # ================================================================
    #  EXPLICIT DYNAMICS SETUP
    # ================================================================
    
    def _setup_explicit(self):
        """Compute lumped mass matrix and CFL time step."""
        # Lumped mass: distribute element mass equally to 8 nodes
        # M_node = Σ(ρ × V / 8) for connected elements  
        # Density in g/cm³ → kg/mm³: ρ (g/cm³) × 10⁻⁶ = kg/mm³
        # But we work in N-mm-MPa: mass in mg (milligrams) so F=ma → N=mg×mm/s²
        # ρ (g/cm³) = ρ (mg/mm³) × 10⁻³ ... 
        # Simpler: use consistent units. MPa=N/mm², mm, N.
        # ρ in tonnes/mm³ = g/cm³ × 10⁻⁹  (so that F=ma gives N=t×mm/s²×10⁶)
        # Actually: MPa system: length=mm, force=N, mass=tonne (10³ kg)
        # ρ (g/cm³) → tonnes/mm³: multiply by 10⁻⁹
        
        rho_t = self._rho * 1e-9  # tonnes/mm³ (consistent with N, mm, MPa)
        elem_mass = rho_t * self._V_elem  # tonnes per element
        
        # Scatter to nodes (3 DOFs each get same mass)
        M = np.zeros(self._n_dof_u)
        node_mass_contrib = elem_mass / 8.0  # (n_elem,) per node
        for n in range(8):
            for d in range(3):
                np.add.at(M, self._elem_dofs[:, n*3+d], node_mass_contrib)
        
        # Avoid zero mass
        M = np.maximum(M, M[M > 0].min() * 1e-3)
        self._M_lumped = M
        
        # CFL time step: Δt < h / c, c = √(E/ρ)
        c_max = np.sqrt(self._E_base.max() / rho_t.min())  # mm/s
        self._dt_cfl = 0.8 * self.h / c_max  # seconds
        
        print(f"  Explicit: M_total={M.sum()*1e9:.2f}g, "
              f"c_max={c_max:.0f}mm/s, Δt_CFL={self._dt_cfl*1e6:.1f}μs")
    
    # ================================================================
    #  INTERNAL FORCE (vectorized, no matrix assembly)
    # ================================================================
    
    def _compute_internal_force(self, u, E_field, phi_elem, active):
        """Compute F_int = Σ B^T · g(φ) · D · B · u_e × V for active elements.
        
        This is the key advantage of explicit: NO global matrix assembly.
        Element-level computation, embarrassingly parallel.
        """
        # Degradation
        g_phi = (1.0 - phi_elem) ** 2 + RESIDUAL_K  # (n_elem,)
        
        # Element displacements → strain → stress → force (vectorized)
        u_elem = u[self._elem_dofs[active]]  # (n_active, 24)
        strain = u_elem @ self._B.T  # (n_active, 6)
        
        # Stress: σ = g(φ) × E × D_ref × ε
        E_active = E_field[active]
        g_active = g_phi[active]
        stress = (strain @ self._D_ref.T) * (E_active * g_active)[:, None]  # (n_active, 6)
        
        # Internal force: f_e = B^T · σ × V
        f_elem = (stress @ self._B) * self._V_elem  # (n_active, 24)
        
        # Scatter to global
        F_int = np.zeros(self._n_dof_u)
        active_dofs = self._elem_dofs[active]
        for d in range(24):
            np.add.at(F_int, active_dofs[:, d], f_elem[:, d])
        
        # Also compute von Mises and strain energy for phase field
        s = stress
        von_mises = np.sqrt(0.5*((s[:,0]-s[:,1])**2 + (s[:,1]-s[:,2])**2 +
                                  (s[:,2]-s[:,0])**2 + 6*(s[:,3]**2+s[:,4]**2+s[:,5]**2)))
        
        # ψ⁺ (Amor split: volumetric tension + deviatoric)
        exx, eyy, ezz = strain[:,0], strain[:,1], strain[:,2]
        exy, eyz, exz = strain[:,3]/2, strain[:,4]/2, strain[:,5]/2
        tr_eps = exx + eyy + ezz
        tr_eps2 = exx**2 + eyy**2 + ezz**2 + 2*(exy**2 + eyz**2 + exz**2)
        
        nu = NU_BONE
        lam = E_active * nu / ((1+nu)*(1-2*nu))
        mu = E_active / (2*(1+nu))
        K_bulk = lam + 2*mu/3
        tr_edev2 = np.maximum(tr_eps2 - tr_eps**2/3, 0)
        psi_plus_active = 0.5 * K_bulk * np.maximum(tr_eps, 0)**2 + mu * tr_edev2
        
        # Map to all elements
        vm_all = np.zeros(self.n_elements)
        vm_all[active] = von_mises
        psi_all = np.zeros(self.n_elements)
        psi_all[active] = psi_plus_active
        
        return F_int, vm_all, psi_all
    
    # ================================================================
    #  PHASE FIELD SOLVE (implicit, cheap)
    # ================================================================
    
    def _solve_phase_field(self, psi_plus, history, phi_prev, active):
        """Solve φ implicitly. Only called every N mechanical steps."""
        Gc = self._Gc
        l0 = self._l0
        H = np.maximum(psi_plus, history)
        
        diff_coeff = Gc * l0
        react_coeff = Gc / l0 + 2.0 * H
        rhs_coeff = 2.0 * H
        
        Kd_flat = self._K_diff_ref.ravel()
        Km_flat = self._K_mass_ref.ravel()
        li, lj = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')
        li_f, lj_f = li.ravel(), lj.ravel()
        
        # Only assemble for active elements
        phi_dofs = self._elem_nodes[active]
        d_c = diff_coeff[active]; r_c = react_coeff[active]; rh_c = rhs_coeff[active]
        
        rows = phi_dofs[:, li_f]
        cols = phi_dofs[:, lj_f]
        vals = d_c[:, None] * Kd_flat[None, :] + r_c[:, None] * Km_flat[None, :]
        
        K_phi = sp.coo_matrix(
            (vals.ravel(), (rows.ravel(), cols.ravel())),
            shape=(self._n_dof_phi, self._n_dof_phi)).tocsr()
        
        N_int = self._K_mass_ref.sum(axis=1)
        F_phi = np.zeros(self._n_dof_phi)
        rhs_vals = rh_c[:, None] * N_int[None, :]
        np.add.at(F_phi, phi_dofs, rhs_vals)
        
        # BCs: non-bone nodes
        if len(self._non_bone_nodes) > 0:
            d_max = K_phi.diagonal().max()
            if d_max > 0:
                PENALTY = d_max * 1e6
                diag = K_phi.diagonal()
                diag[self._non_bone_nodes] += PENALTY
                K_phi.setdiag(diag)
                F_phi[self._non_bone_nodes] = 0.0
        
        # Solve (CG, SPD)
        if self.use_cuda:
            Kg = self._cp_sp.csr_matrix(K_phi)
            Fg = self._cp.array(F_phi)
            xg, _ = self._cp_spla.cg(Kg, Fg, maxiter=2000, tol=1e-7)
            phi = self._cp.asnumpy(xg)
        else:
            phi, info = spla.cg(K_phi, F_phi, maxiter=2000, tol=1e-7)
            if info != 0:
                phi = spla.spsolve(K_phi, F_phi)
        
        phi = np.clip(phi, 0, 1)
        phi = np.maximum(phi, phi_prev)  # irreversibility
        
        return phi, H
    
    # ================================================================
    #  CONTACT (penalty-based)
    # ================================================================
    
    def _compute_contact_forces(self, u, active, phi_elem):
        """Simple penalty contact between fragments.
        
        For eroded elements, their surface nodes may be on different fragments.
        Apply repulsive force when fragments approach each other.
        """
        F_contact = np.zeros(self._n_dof_u)
        
        # Find eroded elements (boundaries between fragments)
        eroded = ~active
        if eroded.sum() == 0:
            return F_contact
        
        # Get surface nodes of eroded elements
        eroded_nodes = np.unique(self._elem_nodes[eroded].ravel())
        
        # For each eroded node, check if it's shared by active elements
        # If so, apply repulsive force based on displacement
        # (simplified: push nodes apart along their displacement direction)
        for node in eroded_nodes[:min(len(eroded_nodes), 500)]:  # limit for speed
            dofs = [node*3, node*3+1, node*3+2]
            # Displacement at this node
            u_node = u[dofs]
            u_mag = np.linalg.norm(u_node)
            if u_mag > 0.01:  # threshold: contact only if significant displacement
                # Penalty: push back proportional to displacement
                k_contact = self._E_base.mean() * 0.1
                F_contact[dofs] -= k_contact * u_node * 0.01
        
        return F_contact
    
    # ================================================================
    #  EXTERNAL FORCE
    # ================================================================
    
    def _compute_external_force(self, params, load_fraction=1.0):
        """Parabolic pressure on superior endplate."""
        F_ext = np.zeros(self._n_dof_u)
        
        force_N = params.force_magnitude * 1000.0 * load_fraction
        flex_rad = np.radians(params.flexion_angle)
        
        sup_mask = self._elem_si > 0.88
        inf_mask = self._elem_si < 0.12
        sup_elems = np.where(sup_mask)[0]
        
        if len(sup_elems) == 0:
            return F_ext
        
        ap = self._elem_ap[sup_elems]; lr = self._elem_lr[sup_elems]
        r2 = (ap-0.5)**2 + (lr-0.5)**2
        parabolic = np.clip(1 - r2/(r2.max()+1e-6), 0.1, 1.0)
        flex_w = np.clip(1 + np.sin(flex_rad)*(ap-0.5)*2, 0.2, 2.5)
        w = parabolic * flex_w; w /= w.sum()
        f_per = -force_N * w
        
        sup_dofs = self._elem_dofs[sup_elems]
        for n in range(4, 8):
            np.add.at(F_ext, sup_dofs[:, n*3+2], f_per / 4.0)
        
        return F_ext
    
    # ================================================================
    #  BOUNDARY CONDITIONS (applied as velocity constraints)
    # ================================================================
    
    def _apply_bc_constraints(self, v, u):
        """Fix bottom z-velocity and anchor."""
        inf_mask = self._elem_si < 0.12
        inf_elems = np.where(inf_mask)[0]
        if len(inf_elems) == 0:
            return v
        
        # Fix z at bottom
        inf_dofs = self._elem_dofs[inf_elems]
        z_dofs = np.unique(inf_dofs[:, np.arange(2, 24, 3)].ravel())
        v[z_dofs] = 0.0
        
        # Anchor xy at center bottom
        ce = inf_elems[len(inf_elems)//2]
        v[self._elem_dofs[ce, 0]] = 0.0
        v[self._elem_dofs[ce, 1]] = 0.0
        
        return v
    
    # ================================================================
    #  HELPER: element φ
    # ================================================================
    
    def _elem_phi(self, phi):
        return phi[self._elem_nodes].mean(axis=1)
    
    def _to_3d(self, elem_data):
        vol = np.zeros(self.shape, dtype=np.float32)
        vol[self._elem_ijk[:,0], self._elem_ijk[:,1], self._elem_ijk[:,2]] = elem_data
        if self.ds > 1:
            vol = zoom(vol, self.ds, order=1)
            t = self._orig_mask.shape
            vol = vol[:t[0], :t[1], :t[2]]
        return vol
    
    # ================================================================
    #  MAIN SIMULATION
    # ================================================================
    
    def set_causal_params(self, params):
        params.validate()
        self.params = params
    
    def simulate(self, total_time_ms=0.5, phi_update_every=10,
                 verbose=True) -> FEMResult:
        """Full fracture simulation.
        
        Args:
            total_time_ms: simulation duration in milliseconds
            phi_update_every: update phase field every N mechanical steps
        """
        params = self.params
        if params is None:
            raise ValueError("Call set_causal_params() first.")
        
        t_start = time.time()
        
        # BMD-adjusted material
        rho_mod = self._rho * params.bmd_factor
        rho_clip = np.clip(rho_mod, 0.01, 2.0)
        cf = self._cortical_fraction
        E_trab = 6850.0 * np.power(rho_clip, 1.49)
        E_cort = 10500.0 * np.power(rho_clip, 2.29)
        E_field = np.clip((1-cf)*E_trab + cf*E_cort, E_MIN, 20000.0)
        
        # State
        u = np.zeros(self._n_dof_u)
        v = np.zeros(self._n_dof_u)
        phi = np.zeros(self._n_dof_phi)
        history = np.zeros(self.n_elements)
        active = np.ones(self.n_elements, dtype=bool)
        
        dt = self._dt_cfl
        total_time_s = total_time_ms * 1e-3
        n_steps = max(int(total_time_s / dt), 100)
        
        # External force (ramp up over first 20% of time)
        F_ext_full = self._compute_external_force(params)
        
        if verbose:
            print(f"\n  Explicit dynamics: {n_steps} steps, "
                  f"Δt={dt*1e6:.1f}μs, T={total_time_ms:.2f}ms")
            print(f"  φ update every {phi_update_every} steps")
        
        # Damping (mass-proportional, prevent oscillation)
        alpha_damp = 0.05  # critical damping fraction
        
        n_eroded_prev = 0
        
        for step in range(n_steps):
            # Load ramp
            load_frac = min(1.0, step / max(n_steps * 0.2, 1))
            F_ext = F_ext_full * load_frac
            
            # ---- Phase field update (every N steps) ----
            if step % phi_update_every == 0 and step > 0:
                phi, history = self._solve_phase_field(
                    history, history, phi, active)
                
                # Element erosion
                phi_elem = self._elem_phi(phi)
                newly_eroded = active & (phi_elem > EROSION_THRESHOLD)
                active[newly_eroded] = False
                n_eroded = (~active).sum()
                
                if n_eroded > n_eroded_prev and verbose:
                    print(f"    Step {step}/{n_steps}: "
                          f"eroded={n_eroded} ({n_eroded/self.n_elements*100:.1f}%), "
                          f"φ_max={phi.max():.3f}")
                n_eroded_prev = n_eroded
            
            # ---- Internal force (active elements only) ----
            phi_elem = self._elem_phi(phi)
            F_int, vm, psi = self._compute_internal_force(
                u, E_field, phi_elem, active)
            
            # Update history for phase field
            history = np.maximum(history, psi)
            
            # ---- Contact ----
            F_contact = self._compute_contact_forces(u, active, phi_elem)
            
            # ---- Acceleration ----
            F_total = F_ext - F_int + F_contact
            a = F_total / self._M_lumped
            
            # ---- Damping ----
            a -= alpha_damp * v / dt
            
            # ---- Central difference ----
            v += a * dt
            v = self._apply_bc_constraints(v, u)
            u += v * dt
            
            # ---- Capture frame for animation ----
            if step % max(n_steps // 50, 1) == 0:
                self._frames.append({
                    'step': step, 'time_us': step * dt * 1e6,
                    'u': u.copy(), 'phi': phi.copy(),
                    'active': active.copy(), 'von_mises': vm.copy(),
                    'n_eroded': int((~active).sum()),
                })
            
            # Progress
            if verbose and step % max(n_steps // 10, 1) == 0:
                ke = 0.5 * np.sum(self._M_lumped * v**2)
                print(f"    [{step:5d}/{n_steps}] t={step*dt*1e6:.0f}μs "
                      f"|u|={np.abs(u).max():.4f}mm "
                      f"KE={ke:.2e} "
                      f"eroded={(~active).sum()}")
        
        wall = time.time() - t_start
        
        # Final phase field update
        phi, history = self._solve_phase_field(psi, history, phi, active)
        phi_elem = self._elem_phi(phi)
        active[phi_elem > EROSION_THRESHOLD] = False
        
        # Store
        self._phi = phi
        self._displacement = u
        self._velocity = v
        self._active = active
        self._von_mises = vm
        self._E_field = E_field
        
        # Classify
        result = self._classify(phi, u, active)
        result.solve_time = wall
        result.n_iterations = n_steps
        self._result = result
        
        if verbose:
            print(f"\n  ★ {result.ao_type} ({wall:.1f}s, {n_steps} steps)")
            print(f"    Eroded: {(~active).sum()}/{self.n_elements} "
                  f"({(~active).sum()/self.n_elements*100:.1f}%)")
            print(f"    φ_max={phi.max():.3f}, |u|_max={np.abs(u).max():.4f}mm")
        
        return result
    
    def _classify(self, phi, u, active):
        """AO classification from crack pattern."""
        phi_e = self._elem_phi(phi)
        cracked = phi_e > 0.5
        frac = cracked.sum() / self.n_elements
        
        ant = self._elem_ap > 0.5; post = self._elem_ap < 0.5
        mid = (self._elem_si > 0.33) & (self._elem_si < 0.67)
        
        ant_dam = phi_e[ant].mean() if ant.sum() > 0 else 0
        post_dam = phi_e[post].mean() if post.sum() > 0 else 0
        canal = phi_e[post & mid].mean() if (post & mid).sum() > 0 else 0
        
        # Fragment count from connected components of active elements
        active_vol = np.zeros(self.shape, dtype=np.int32)
        active_ijk = self._elem_ijk[active]
        if len(active_ijk) > 0:
            active_vol[active_ijk[:,0], active_ijk[:,1], active_ijk[:,2]] = 1
        _, n_frags = ndimage_label(active_vol)
        
        if frac < 0.02: ao = 'A0'
        elif ant_dam > post_dam * 1.5 and canal < 0.15: ao = 'A1'
        elif frac < 0.35 and canal < 0.3: ao = 'A2'
        elif canal >= 0.3 and canal < 0.6: ao = 'A3'
        else: ao = 'A4'
        
        return FEMResult(
            ao_type=ao, confidence=0.8,
            max_von_mises=float(self._von_mises.max()) if hasattr(self, '_von_mises') else 0,
            max_displacement=float(np.abs(u).max()),
            n_yielded=int(cracked.sum()), n_elements=self.n_elements,
            yielded_fraction=float(frac),
            anterior_height_loss=0, posterior_height_loss=0,
            canal_compromise=float(canal), n_fragments=n_frags,
        )
    
    # ================================================================
    #  VISUALIZATION
    # ================================================================
    
    def save_fracture_animation(self, out_path, fps=10):
        """Save animated GIF showing crack progression + fragment separation."""
        if not self._frames or not HAS_MPL or not HAS_PIL:
            print("  ⚠ Cannot save animation (no frames or missing PIL/matplotlib)")
            return
        
        print(f"  Rendering {len(self._frames)} frames...")
        pil_frames = []
        
        mid_x = self.shape[0] // 2
        ct_ds = self.ct
        
        for fi, frame in enumerate(self._frames):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#0d1117')
            
            phi_vol = np.zeros(self.shape, dtype=np.float32)
            phi_e = self._elem_phi(frame['phi'])
            phi_vol[self._elem_ijk[:,0], self._elem_ijk[:,1],
                    self._elem_ijk[:,2]] = phi_e
            
            active_vol = np.zeros(self.shape, dtype=np.int32)
            act_ijk = self._elem_ijk[frame['active']]
            if len(act_ijk) > 0:
                active_vol[act_ijk[:,0], act_ijk[:,1], act_ijk[:,2]] = 1
            
            kw = dict(origin='lower', aspect='auto')
            
            # Panel 1: CT with eroded regions shown as black
            ct_frac = ct_ds.copy()
            eroded_mask = (self.bone_mask) & (active_vol == 0)
            ct_frac[eroded_mask] = ct_ds.min()
            axes[0].imshow(ct_frac[mid_x].T, cmap='bone', vmin=-200, vmax=800, **kw)
            axes[0].set_title(f't={frame["time_us"]:.0f}μs | eroded={frame["n_eroded"]}',
                             color='white', fontsize=10)
            axes[0].axis('off'); axes[0].set_facecolor('#0d1117')
            
            # Panel 2: Phase field
            axes[1].imshow(phi_vol[mid_x].T, cmap='inferno', vmin=0, vmax=1, **kw)
            if phi_vol[mid_x].max() > 0.3:
                axes[1].contour(phi_vol[mid_x].T, levels=[0.5, 0.95],
                               colors=['cyan', 'red'], linewidths=[1.5, 1],
                               origin='lower')
            axes[1].set_title(f'φ (cyan=crack, red=erosion)', color='white', fontsize=10)
            axes[1].axis('off'); axes[1].set_facecolor('#0d1117')
            
            # Panel 3: Active bone (white) with gaps (black)
            axes[2].imshow(active_vol[mid_x].T, cmap='gray', vmin=0, vmax=1, **kw)
            axes[2].set_title(f'Bone (eroded={frame["n_eroded"]}/{self.n_elements})',
                             color='white', fontsize=10)
            axes[2].axis('off'); axes[2].set_facecolor('#0d1117')
            
            r = self._result
            fig.suptitle(f'v7 Full Fracture — {r.ao_type if r else "?"}',
                        fontsize=14, color='white', fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, facecolor='#0d1117')
            buf.seek(0)
            pil_frames.append(Image.open(buf).copy())
            plt.close(fig)
        
        if pil_frames:
            pil_frames[0].save(out_path, save_all=True,
                              append_images=pil_frames[1:],
                              duration=1000//fps, loop=0)
            print(f"  Saved: {out_path} ({len(pil_frames)} frames)")


# ============================================================================
#  DEMO
# ============================================================================

def demo(use_cuda=False):
    from _gen_real_fracture_visuals import load_vertebra
    
    print("=" * 70)
    print("WiseSpine v7 — Full FEM Fracture Engine")
    print("  Phase Field + Explicit Dynamics + Element Erosion")
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
    
    ds = max(3, int(np.cbrt(mask.sum() / 50000)))
    print(f"  Downsample: ds={ds}")
    
    engine = FullFractureEngine(mask, ct, voxel_size_mm=voxel_size,
                                downsample=ds, seed=42, use_cuda=use_cuda)
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'fracture_v7_demo')
    os.makedirs(out_dir, exist_ok=True)
    
    scenarios = [
        ('burst', CausalParameters(8.0, 5.0, 0.0, 0.5)),
    ]
    
    for name, params in scenarios:
        print(f"\n{'='*60}")
        print(f"  Scenario: {name}")
        print(f"{'='*60}")
        
        engine.set_causal_params(params)
        engine._frames = []
        result = engine.simulate(total_time_ms=0.5)
        
        # Save animation
        gif_path = os.path.join(out_dir, f'v7_{name}_fracture.gif')
        engine.save_fracture_animation(gif_path)
        
        # Save final state plot
        if HAS_MPL:
            _plot_final_state(engine, name, out_dir)
    
    print(f"\nOutputs saved to {out_dir}/")


def _plot_final_state(engine, scenario, out_dir):
    """Final fracture state: 3-plane view with eroded gaps."""
    phi = engine._phi
    u = engine._displacement
    active = engine._active
    
    phi_vol = np.zeros(engine.shape, dtype=np.float32)
    phi_e = engine._elem_phi(phi)
    phi_vol[engine._elem_ijk[:,0], engine._elem_ijk[:,1],
            engine._elem_ijk[:,2]] = phi_e
    
    active_vol = np.zeros(engine.shape, dtype=np.int32)
    act_ijk = engine._elem_ijk[active]
    if len(act_ijk) > 0:
        active_vol[act_ijk[:,0], act_ijk[:,1], act_ijk[:,2]] = 1
    
    # Fragment labeling
    frag_labels, n_frags = ndimage_label(active_vol)
    
    mid = [s//2 for s in engine.shape]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='#0d1117')
    kw = dict(origin='lower', aspect='auto')
    
    ct = engine.ct
    slices = [
        (ct[mid[0]].T, phi_vol[mid[0]].T, frag_labels[mid[0]].T, 'Sagittal'),
        (ct[:, mid[1], :].T, phi_vol[:, mid[1], :].T, frag_labels[:, mid[1], :].T, 'Coronal'),
        (ct[:, :, mid[2]].T, phi_vol[:, :, mid[2]].T, frag_labels[:, :, mid[2]].T, 'Axial'),
    ]
    
    for col, (ct_sl, phi_sl, frag_sl, name) in enumerate(slices):
        # Row 1: CT with crack gaps
        ct_frac = ct_sl.copy()
        bone_sl = engine.bone_mask
        ax = axes[0, col]
        ax.imshow(ct_sl, cmap='bone', vmin=-200, vmax=800, **kw)
        # Overlay φ
        phi_masked = np.ma.masked_where(phi_sl < 0.1, phi_sl)
        ax.imshow(phi_masked, cmap='hot', vmin=0, vmax=1, alpha=0.6, **kw)
        if phi_sl.max() > 0.3:
            ax.contour(phi_sl, levels=[0.5, 0.95], colors=['cyan', 'lime'],
                      linewidths=[2, 1], origin='lower')
        ax.set_title(f'{name} — CT + crack', color='white', fontsize=12)
        ax.axis('off'); ax.set_facecolor('#0d1117')
        
        # Row 2: Fragments (each color = separate fragment)
        ax2 = axes[1, col]
        if n_frags > 1:
            ax2.imshow(frag_sl, cmap='tab10', vmin=0, vmax=max(n_frags, 1), **kw)
        else:
            ax2.imshow(frag_sl > 0, cmap='gray', **kw)
        ax2.set_title(f'{name} — {n_frags} fragments', color='white', fontsize=12)
        ax2.axis('off'); ax2.set_facecolor('#0d1117')
    
    r = engine._result
    fig.suptitle(f'v7 Full Fracture — {scenario.upper()} | {r.ao_type} | '
                f'{n_frags} fragments | eroded={(~active).sum()}',
                fontsize=16, color='white', fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(out_dir, f'v7_{scenario}_final.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='v7 Full FEM Fracture')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    demo(use_cuda=args.cuda)
