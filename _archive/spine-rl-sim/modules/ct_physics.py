"""
CT Physics Simulation Module

Simulates realistic CT imaging physics artifacts:
- Partial volume effect
- Poisson quantum noise
- Detector blur (PSF)
- Beam hardening (simplified)
- Anisotropic trabecular texture

Date: 2026-02-06
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional


# =============================================================================
# TRABECULAR TEXTURE GENERATION
# =============================================================================

def generate_perlin_noise_2d(shape: Tuple[int, int], scale: float = 10.0, 
                              octaves: int = 4, seed: int = 42) -> np.ndarray:
    """Generate 2D Perlin-like noise using octave summation.
    
    Args:
        shape: Output shape (H, W)
        scale: Base frequency scale
        octaves: Number of octave layers
        seed: Random seed
        
    Returns:
        Noise array normalized to [-1, 1]
    """
    np.random.seed(seed)
    noise = np.zeros(shape, dtype=np.float32)
    
    for octave in range(octaves):
        freq = 2 ** octave
        amplitude = 0.5 ** octave
        
        # Generate random gradients at grid points
        grid_h = max(2, int(shape[0] / (scale / freq)))
        grid_w = max(2, int(shape[1] / (scale / freq)))
        
        # Simple interpolated noise
        random_grid = np.random.randn(grid_h, grid_w).astype(np.float32)
        
        # Upsample to full resolution
        from scipy.ndimage import zoom
        zoom_factors = (shape[0] / grid_h, shape[1] / grid_w)
        octave_noise = zoom(random_grid, zoom_factors, order=1)
        
        # Ensure correct shape
        octave_noise = octave_noise[:shape[0], :shape[1]]
        
        noise += amplitude * octave_noise
    
    # Normalize to [-1, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8) * 2 - 1
    return noise


def generate_multiscale_trabecular_texture(
    shape: Tuple[int, ...],
    mask: np.ndarray = None,
    scale_coarse: float = 20.0,
    scale_medium: float = 8.0,
    scale_fine: float = 3.0,
    intensity: float = 40.0,
    local_asymmetry: bool = True,
    seed: int = 42
) -> np.ndarray:
    """Generate multi-scale anisotropic trabecular bone texture.
    
    Improvement over single-scale version:
    - Three hierarchical scales (coarse struts, medium trabeculae, fine micro)
    - Local orientation variation for asymmetry
    - More realistic HU distribution
    
    Args:
        shape: Output shape (H, W) or (H, W, D)
        mask: Optional bone mask for local orientation
        scale_coarse: Coarse structural struts scale
        scale_medium: Medium trabecular scale
        scale_fine: Fine micro-texture scale
        intensity: Total HU intensity of texture
        local_asymmetry: Add local orientation variation
        seed: Random seed
        
    Returns:
        Multi-scale trabecular texture array
    """
    is_3d = len(shape) == 3
    H, W = shape[:2] if is_3d else shape
    
    if is_3d:
        texture = np.zeros(shape, dtype=np.float32)
        for z in range(shape[2]):
            texture[:, :, z] = generate_multiscale_trabecular_texture(
                shape[:2], mask[:, :, z] if mask is not None else None,
                scale_coarse, scale_medium, scale_fine, intensity, local_asymmetry, seed + z
            )
        return texture
    
    np.random.seed(seed)
    
    # === Layer 1: Coarse structural struts (load-bearing columns) ===
    coarse = generate_perlin_noise_2d((H, W), scale=scale_coarse, seed=seed)
    # Strong vertical anisotropy
    coarse = gaussian_filter(coarse, sigma=(0.3, 4.0))
    
    # === Layer 2: Medium trabeculae (primary network) ===
    medium = generate_perlin_noise_2d((H, W), scale=scale_medium, seed=seed+100)
    medium = gaussian_filter(medium, sigma=(0.5, 2.5))
    # Add cross-links
    medium_cross = generate_perlin_noise_2d((H, W), scale=scale_medium*1.5, seed=seed+200)
    medium_cross = gaussian_filter(medium_cross, sigma=(2.5, 0.5))
    medium = 0.65 * medium + 0.35 * medium_cross
    
    # === Layer 3: Fine micro-texture (high frequency detail) ===
    fine = generate_perlin_noise_2d((H, W), scale=scale_fine, octaves=2, seed=seed+300)
    fine = gaussian_filter(fine, sigma=(0.3, 1.2))
    
    # === Local asymmetry: Spatially varying orientation ===
    if local_asymmetry:
        # Create smooth orientation field
        orientation_field = generate_perlin_noise_2d((H, W), scale=40.0, seed=seed+500)
        orientation_field = gaussian_filter(orientation_field, sigma=10)
        
        # Perturb medium layer based on orientation
        # Shift the texture slightly based on local orientation
        y_coords, x_coords = np.meshgrid(np.arange(W), np.arange(H))
        shift_x = (orientation_field * 3).astype(int)
        shift_y = (orientation_field * 2).astype(int)
        
        # Apply small local shifts for asymmetry
        shifted_medium = np.zeros_like(medium)
        for i in range(H):
            for j in range(W):
                si, sj = min(H-1, max(0, i + shift_x[i, j] // 10)), min(W-1, max(0, j + shift_y[i, j] // 10))
                shifted_medium[i, j] = medium[si, sj]
        medium = 0.7 * medium + 0.3 * shifted_medium
    
    # === Combine layers with weights ===
    # Coarse: structural framework
    # Medium: primary trabecular network  
    # Fine: surface detail
    combined = 0.35 * coarse + 0.45 * medium + 0.20 * fine
    
    # === Add non-Gaussian HU distribution (more continuous) ===
    # Transform to have more intermediate values
    combined = np.tanh(combined * 1.5) * 0.8 + combined * 0.2
    
    # Normalize and scale
    combined = (combined - combined.mean()) / (combined.std() + 1e-8)
    texture = combined * intensity * 0.7  # Slightly reduce intensity for more gray tones
    
    return texture


def generate_anisotropic_trabecular_texture(
    shape: Tuple[int, ...],
    direction: str = 'vertical',
    scale: float = 8.0,
    anisotropy: float = 3.0,
    intensity: float = 30.0,
    seed: int = 42
) -> np.ndarray:
    """Generate anisotropic trabecular bone-like texture (legacy wrapper).
    
    For backward compatibility. New code should use generate_multiscale_trabecular_texture.
    """
    # Delegate to multi-scale version
    return generate_multiscale_trabecular_texture(
        shape=shape,
        mask=None,
        scale_medium=scale,
        intensity=intensity,
        local_asymmetry=True,
        seed=seed
    )


# =============================================================================
# FRACTURE SURFACE PHYSICS (P2)
# =============================================================================

def generate_hierarchical_fracture_surface(
    shape: Tuple[int, int],
    fracture_location: int,
    roughness: float = 0.4,
    scales: list = [1, 2, 4, 8],
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate rough, jagged fracture surface using multi-scale noise.
    
    Real fractures have:
    - Micro-fragmentation at multiple scales
    - Sharp, jagged edges
    - Asymmetric patterns
    
    Args:
        shape: (H, W) shape of the image
        fracture_location: Z-coordinate of fracture center
        roughness: Overall roughness intensity (0-1)
        scales: List of noise scales for hierarchical detail
        seed: Random seed
        
    Returns:
        fracture_mask: Boolean mask of fracture region
        fracture_field: HU modification field
    """
    H, W = shape
    np.random.seed(seed)
    
    # Generate hierarchical noise for fracture edge
    edge_perturbation = np.zeros(H, dtype=np.float32)
    
    for i, scale in enumerate(scales):
        amplitude = roughness * (0.6 ** i)  # Decreasing amplitude at finer scales
        freq = scale * np.pi / 20
        
        # Add wave component
        phase = np.random.rand() * 2 * np.pi
        wave = np.sin(np.arange(H) * freq + phase) * amplitude * 3
        
        # Add random jitter
        jitter = np.random.randn(H) * amplitude * 2
        
        edge_perturbation += wave + jitter
    
    # Smooth slightly to avoid pixel-level noise
    edge_perturbation = gaussian_filter(edge_perturbation, sigma=1.5)
    
    # Create fracture mask with rough edges
    fracture_mask = np.zeros(shape, dtype=bool)
    fracture_field = np.zeros(shape, dtype=np.float32)
    
    fracture_width = 3 + int(roughness * 3)  # Variable thickness
    
    for x in range(H):
        z_center = fracture_location + int(edge_perturbation[x])
        z_start = max(0, z_center - fracture_width // 2)
        z_end = min(W, z_center + fracture_width // 2 + 1)
        
        # Variable width along fracture
        local_width = fracture_width + int(np.random.randn() * 0.5)
        z_start = max(0, z_center - local_width // 2)
        z_end = min(W, z_center + local_width // 2 + 1)
        
        fracture_mask[x, z_start:z_end] = True
        
        # HU gradient across fracture (darker at center)
        for z in range(z_start, z_end):
            dist_from_center = abs(z - z_center) / (local_width / 2 + 0.1)
            fracture_field[x, z] = -0.65 + 0.3 * dist_from_center  # -65% to -35% HU reduction
    
    # Add micro-fragments near fracture
    fragment_region = ndi.binary_dilation(fracture_mask, iterations=3) & ~fracture_mask
    fragment_noise = np.random.randn(*shape) * roughness * 50
    fracture_field[fragment_region] = fragment_noise[fragment_region] * 0.01  # Subtle density variation
    
    return fracture_mask, fracture_field


def simulate_burst_retropulsion(
    ct: np.ndarray,
    mask: np.ndarray,
    severity: float = 0.5,
    canal_direction: str = 'posterior',
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate burst fracture with posterior fragment retropulsion.
    
    Burst fracture characteristics:
    - Posterior wall fragment displacement toward canal
    - Angular (tilted) fragment displacement
    - Canal narrowing effect
    - Asymmetric collapse
    
    Args:
        ct: CT image
        mask: Vertebra mask
        severity: Fracture severity (0-1)
        canal_direction: Direction of spinal canal ('posterior' = lower x)
        seed: Random seed
        
    Returns:
        deformed_ct: CT with retropulsion
        fragment_mask: Mask of displaced fragments
    """
    np.random.seed(seed)
    H, W = ct.shape
    
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return ct, np.zeros_like(mask, dtype=bool)
    
    x_min, x_max = coords[0].min(), coords[0].max()
    z_min, z_max = coords[1].min(), coords[1].max()
    x_mid = (x_min + x_max) // 2
    z_mid = (z_min + z_max) // 2
    
    # Identify posterior wall (lower x values typically)
    posterior_thickness = max(3, int((x_max - x_min) * 0.15))
    posterior_region = mask.copy()
    posterior_region[x_min + posterior_thickness:, :] = False
    
    # Create fragment from posterior wall
    fragment_mask = posterior_region.copy()
    
    # Create deformation field for retropulsion
    deformation = np.zeros((*ct.shape, 2), dtype=np.float32)
    
    # Fragment displacement: angular (tilted) toward canal
    fragment_coords = np.where(fragment_mask)
    if len(fragment_coords[0]) > 0:
        for i in range(len(fragment_coords[0])):
            x, z = fragment_coords[0][i], fragment_coords[1][i]
            
            # Distance from fragment center
            rel_z = (z - z_mid) / (z_max - z_min + 1)
            
            # Angular displacement (tilted fragment)
            # More displacement at one end than the other
            tilt_factor = 0.5 + rel_z * 0.5  # 0.5 to 1.0
            
            # Displacement toward canal (negative x = posterior)
            canal_displacement = -severity * (5 + np.random.randn() * 0.5) * tilt_factor
            
            # Slight lateral displacement
            lateral = np.random.randn() * severity * 1.5
            
            deformation[x, z, 0] = canal_displacement
            deformation[x, z, 1] = lateral
    
    # Apply compression to main body
    main_body = mask & ~fragment_mask
    for z in range(z_min, z_max):
        t = (z - z_min) / (z_max - z_min + 1)
        compression = t * severity * 20
        
        for x in range(x_min, x_max):
            if main_body[x, z]:
                # Radial expansion
                radial = (x - x_mid) / (x_max - x_min + 1) * severity * 6
                deformation[x, z, 0] = radial
                deformation[x, z, 1] = -compression
    
    # Smooth deformation
    deformation[..., 0] = gaussian_filter(deformation[..., 0], sigma=2)
    deformation[..., 1] = gaussian_filter(deformation[..., 1], sigma=2)
    
    # Apply deformation
    from scipy.ndimage import map_coordinates
    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))
    x_new = x_grid + deformation[..., 1]
    y_new = y_grid + deformation[..., 0]
    
    coords_new = np.array([y_new.ravel(), x_new.ravel()])
    deformed_ct = map_coordinates(ct, coords_new, order=1, mode='nearest').reshape(H, W)
    
    # Add fragment-specific features
    # 1. Fragment edges: sharp density drop
    fragment_edge = ndi.binary_dilation(fragment_mask, iterations=1) & ~fragment_mask
    deformed_ct[fragment_edge] *= 0.4
    
    # 2. Fragment interior: fragmentation texture
    fragment_noise = np.random.randn(*ct.shape) * 40 * severity
    deformed_ct[fragment_mask] += fragment_noise[fragment_mask]
    
    # 3. Canal region darkening (simulating epidural space compression)
    canal_region = np.zeros_like(mask, dtype=bool)
    canal_region[:x_min, z_mid-10:z_mid+10] = True
    deformed_ct[canal_region] *= (1 - severity * 0.2)  # Subtle darkening
    
    return np.clip(deformed_ct, -1000, 3000), fragment_mask

# =============================================================================
# CT PHYSICS EFFECTS
# =============================================================================

def apply_cortical_thickness_variation(
    ct: np.ndarray,
    mask: np.ndarray,
    thickness_variation: float = 0.3,
    seed: int = 42
) -> np.ndarray:
    """Apply realistic cortical shell thickness variation.
    
    Real cortical bone has variable thickness:
    - Thinner at endplates
    - Variable around body
    - Subtle local thinning/thickening
    
    Args:
        ct: CT volume
        mask: Bone mask
        thickness_variation: Variation magnitude (0-1)
        seed: Random seed
        
    Returns:
        CT with variable cortical shell
    """
    np.random.seed(seed)
    result = ct.copy().astype(np.float32)
    
    # Create distance from bone surface
    distance_from_surface = ndi.distance_transform_edt(mask)
    
    # Create spatial variation field
    H, W = ct.shape[:2]
    variation_field = generate_perlin_noise_2d((H, W), scale=15.0, seed=seed)
    variation_field = gaussian_filter(variation_field, sigma=5)
    
    # Variable cortical thickness (1-4 voxels)
    base_thickness = 2.0
    thickness_map = base_thickness + variation_field * thickness_variation * 2
    thickness_map = np.clip(thickness_map, 1, 4)
    
    # Create cortical region with variable thickness
    cortical = (distance_from_surface > 0) & (distance_from_surface < thickness_map)
    trabecular = distance_from_surface >= thickness_map
    
    # Smooth transition at cortical-trabecular interface
    transition_zone = (distance_from_surface >= thickness_map - 0.5) & (distance_from_surface < thickness_map + 0.5)
    
    if transition_zone.sum() > 0:
        # Blend values in transition zone
        cortical_val = result[cortical].mean() if cortical.sum() > 0 else 800
        trabecular_val = result[trabecular].mean() if trabecular.sum() > 0 else 250
        
        blend_factor = (distance_from_surface[transition_zone] - (thickness_map[transition_zone] - 0.5))
        result[transition_zone] = cortical_val * (1 - blend_factor) + trabecular_val * blend_factor
    
    # Add subtle endplate thinning
    coords = np.where(mask)
    if len(coords[1]) > 0:
        z_min, z_max = coords[1].min(), coords[1].max()
        z_range = z_max - z_min
        
        # Superior and inferior endplate regions
        for z in range(mask.shape[1]):
            rel_z = (z - z_min) / (z_range + 1e-8)
            # Thinning factor at endplates
            endplate_factor = 1.0 - 0.3 * (rel_z < 0.15 or rel_z > 0.85)
            
            if cortical[:, z].sum() > 0:
                result[:, z][cortical[:, z]] *= endplate_factor
    
    return result


def apply_partial_volume_effect(
    ct: np.ndarray,
    mask: np.ndarray,
    sigma: float = 0.8
) -> np.ndarray:
    """Apply partial volume effect at tissue boundaries.
    
    In real CT, voxels at boundaries contain mixed tissue,
    resulting in intermediate HU values.
    
    Args:
        ct: CT volume
        mask: Tissue mask (bone region)
        sigma: Blur sigma (relates to voxel size)
        
    Returns:
        CT with partial volume effect
    """
    # Find boundary region with graded distances
    distance_outside = ndi.distance_transform_edt(~mask)
    distance_inside = ndi.distance_transform_edt(mask)
    
    # Create smooth boundary zone
    boundary_width = 2.0
    outer_boundary = (distance_outside > 0) & (distance_outside < boundary_width)
    inner_boundary = (distance_inside > 0) & (distance_inside < boundary_width)
    boundary = outer_boundary | inner_boundary
    
    # Apply Gaussian blur with local variance modulation
    blurred = gaussian_filter(ct.astype(np.float32), sigma=sigma)
    
    # Also add local variance for more continuous distribution
    local_var = gaussian_filter((ct - blurred)**2, sigma=sigma*2)
    local_std = np.sqrt(local_var + 1e-8)
    
    result = ct.copy().astype(np.float32)
    
    # Blend at boundaries with distance-weighted mixing
    blend_weight = np.zeros_like(ct)
    blend_weight[outer_boundary] = 1.0 - distance_outside[outer_boundary] / boundary_width
    blend_weight[inner_boundary] = 1.0 - distance_inside[inner_boundary] / boundary_width
    
    result = result * (1 - blend_weight) + blurred * blend_weight
    
    # Add subtle gray-tone variation
    gray_variation = np.random.randn(*ct.shape) * local_std * 0.1
    result[boundary] += gray_variation[boundary]
    
    return result


def apply_poisson_noise(
    ct: np.ndarray,
    dose_level: float = 1.0,
    baseline_photons: int = 10000
) -> np.ndarray:
    """Apply quantum (Poisson) noise to CT.
    
    CT noise follows Poisson statistics based on X-ray photon count.
    Lower dose = fewer photons = more noise.
    
    Args:
        ct: CT volume in HU
        dose_level: Relative dose (1.0 = normal, 0.5 = half dose)
        baseline_photons: Photon count at normal dose
        
    Returns:
        Noisy CT
    """
    # Convert HU to linear attenuation (simplified)
    # HU = 1000 * (mu - mu_water) / mu_water
    # Assuming mu_water ~ 0.02 /mm at typical CT energy
    mu = (ct + 1000) / 1000 * 0.02
    mu = np.clip(mu, 0.001, 0.1)  # Prevent negative/zero
    
    # Simulate photon counts (Beer-Lambert law, simplified)
    # I = I0 * exp(-mu * d)
    # For simplicity, use relative attenuation
    I0 = baseline_photons * dose_level
    transmission = np.exp(-mu * 10)  # Assume 10mm avg path
    expected_photons = I0 * transmission
    
    # Apply Poisson noise
    np.random.seed(None)  # Random for each call
    noisy_photons = np.random.poisson(expected_photons.astype(np.float64))
    noisy_photons = np.maximum(noisy_photons, 1)  # Prevent log(0)
    
    # Convert back to HU
    noisy_transmission = noisy_photons / I0
    noisy_mu = -np.log(noisy_transmission + 1e-10) / 10
    noisy_ct = (noisy_mu / 0.02 * 1000) - 1000
    
    # Blend with original (Poisson model is simplified)
    # Use weighted average to control noise level
    alpha = 0.3  # 30% from noise model
    result = alpha * noisy_ct + (1 - alpha) * ct
    
    return result.astype(np.float32)


def apply_detector_blur(
    ct: np.ndarray,
    psf_sigma: float = 0.5
) -> np.ndarray:
    """Apply detector point spread function (PSF) blur.
    
    Real CT detectors have finite resolution, causing slight blur.
    
    Args:
        ct: CT volume
        psf_sigma: PSF sigma in voxels
        
    Returns:
        Blurred CT
    """
    # Apply isotropic Gaussian blur
    blurred = gaussian_filter(ct.astype(np.float32), sigma=psf_sigma)
    return blurred


def apply_beam_hardening_artifact(
    ct: np.ndarray,
    bone_mask: np.ndarray,
    strength: float = 0.1
) -> np.ndarray:
    """Apply simplified beam hardening artifact near dense bone.
    
    Polychromatic X-rays cause cupping/streaking near bone.
    This is a simplified model using distance-based darkening.
    
    Args:
        ct: CT volume
        bone_mask: Mask of dense bone
        strength: Artifact strength
        
    Returns:
        CT with beam hardening artifacts
    """
    # Create distance map from bone surface
    distance = ndi.distance_transform_edt(~bone_mask)
    
    # Artifact falls off with distance
    artifact = -strength * 100 * np.exp(-distance / 5)
    
    # Apply only near bone (within 10 voxels)
    near_bone = distance < 10
    result = ct.copy().astype(np.float32)
    result[near_bone] += artifact[near_bone]
    
    return result


def calibrate_hu_values(
    ct: np.ndarray,
    mask: np.ndarray,
    target_cortical: float = 1000.0,
    target_trabecular: float = 250.0
) -> np.ndarray:
    """Calibrate HU values to realistic bone ranges.
    
    Args:
        ct: CT volume
        mask: Bone segmentation mask
        target_cortical: Target HU for cortical bone (outer shell)
        target_trabecular: Target HU for trabecular bone (interior)
        
    Returns:
        Calibrated CT
    """
    result = ct.copy().astype(np.float32)
    
    # Identify cortical (outer) vs trabecular (inner)
    eroded = ndi.binary_erosion(mask, iterations=2)
    cortical = mask & ~eroded
    trabecular = eroded
    
    # Get current values
    if cortical.sum() > 0:
        current_cortical = result[cortical].mean()
        cortical_scale = target_cortical / (current_cortical + 1e-8)
        result[cortical] *= cortical_scale
    
    if trabecular.sum() > 0:
        current_trabecular = result[trabecular].mean()
        trabecular_scale = target_trabecular / (current_trabecular + 1e-8)
        result[trabecular] *= trabecular_scale
    
    return result


# =============================================================================
# COMBINED PIPELINE
# =============================================================================

class CTPhysicsSimulator:
    """Combined CT physics simulation pipeline."""
    
    def __init__(
        self,
        apply_trabecular: bool = True,
        apply_partial_volume: bool = True,
        apply_noise: bool = True,
        apply_blur: bool = True,
        apply_beam_hardening: bool = False,
        calibrate_hu: bool = True,
        noise_dose_level: float = 1.0,
        trabecular_intensity: float = 25.0
    ):
        self.apply_trabecular_flag = apply_trabecular
        self.apply_partial_volume_flag = apply_partial_volume
        self.apply_noise_flag = apply_noise
        self.apply_blur_flag = apply_blur
        self.apply_beam_hardening_flag = apply_beam_hardening
        self.calibrate_hu_flag = calibrate_hu
        self.noise_dose_level = noise_dose_level
        self.trabecular_intensity = trabecular_intensity
    
    def process(
        self,
        ct: np.ndarray,
        bone_mask: np.ndarray,
        seed: int = 42
    ) -> np.ndarray:
        """Apply full CT physics simulation pipeline.
        
        Args:
            ct: Input CT volume
            bone_mask: Bone segmentation mask
            seed: Random seed for reproducibility
            
        Returns:
            Processed CT with realistic physics
        """
        result = ct.copy().astype(np.float32)
        
        # 1. Add trabecular texture
        if self.apply_trabecular_flag:
            texture = generate_anisotropic_trabecular_texture(
                result.shape,
                direction='vertical',
                intensity=self.trabecular_intensity,
                seed=seed
            )
            # Apply only inside bone
            result[bone_mask] += texture[bone_mask]
        
        # 2. Calibrate HU
        if self.calibrate_hu_flag:
            result = calibrate_hu_values(result, bone_mask)
        
        # 3. Partial volume
        if self.apply_partial_volume_flag:
            result = apply_partial_volume_effect(result, bone_mask)
        
        # 4. Beam hardening
        if self.apply_beam_hardening_flag:
            result = apply_beam_hardening_artifact(result, bone_mask)
        
        # 5. Detector blur
        if self.apply_blur_flag:
            result = apply_detector_blur(result, psf_sigma=0.3)
        
        # 6. Quantum noise
        if self.apply_noise_flag:
            result = apply_poisson_noise(result, self.noise_dose_level)
        
        return result


# =============================================================================
# TEST
# =============================================================================

def test_ct_physics():
    """Test CT physics simulation on synthetic data."""
    print("Testing CT Physics Simulation...")
    
    # Create synthetic CT slice
    H, W = 200, 200
    ct = np.zeros((H, W), dtype=np.float32) - 1000  # Air background
    
    # Add vertebra-like shape
    y, x = np.ogrid[:H, :W]
    center = (H//2, W//2)
    vertebra = ((y - center[0])**2 / 40**2 + (x - center[1])**2 / 50**2) < 1
    ct[vertebra] = 400  # Bone HU
    
    # Create mask
    mask = vertebra
    
    # Test individual functions
    print("  - Trabecular texture...")
    texture = generate_anisotropic_trabecular_texture((H, W), intensity=30)
    ct_textured = ct.copy()
    ct_textured[mask] += texture[mask]
    
    print("  - Partial volume...")
    ct_pv = apply_partial_volume_effect(ct_textured, mask)
    
    print("  - Poisson noise...")
    ct_noisy = apply_poisson_noise(ct_pv, dose_level=1.0)
    
    print("  - Detector blur...")
    ct_blurred = apply_detector_blur(ct_noisy, psf_sigma=0.3)
    
    print("  - Full pipeline...")
    simulator = CTPhysicsSimulator()
    ct_full = simulator.process(ct, mask)
    
    print(f"  Input range: [{ct.min():.0f}, {ct.max():.0f}]")
    print(f"  Output range: [{ct_full.min():.0f}, {ct_full.max():.0f}]")
    print("CT Physics Simulation test passed!")
    
    return ct, ct_full, mask


# =============================================================================
# ULTRA-ADVANCED: TRABECULAR CONNECTIVITY GRAPH
# =============================================================================

def generate_trabecular_network(
    shape: Tuple[int, int],
    node_density: float = 0.02,
    connection_radius: int = 15,
    thickness_range: Tuple[int, int] = (1, 3),
    seed: int = 42
) -> np.ndarray:
    """Generate graph-based trabecular network with realistic connectivity.
    
    Creates a network of trabeculae with:
    - Node-based topology (trabecular junctions)
    - Connected struts forming load-bearing paths
    - Variable strut thickness
    - Vertical anisotropy (load-bearing direction)
    
    Args:
        shape: Output shape (H, W)
        node_density: Fraction of pixels to use as nodes
        connection_radius: Max distance for node connections
        thickness_range: (min, max) strut thickness
        seed: Random seed
        
    Returns:
        Network mask with intensity proportional to strut thickness
    """
    np.random.seed(seed)
    H, W = shape
    
    # Generate node positions with vertical bias (more nodes vertically aligned)
    num_nodes = int(H * W * node_density)
    node_y = np.random.randint(0, H, num_nodes)
    node_x = np.random.randint(0, W, num_nodes)
    nodes = list(zip(node_y, node_x))
    
    # Create network canvas
    network = np.zeros(shape, dtype=np.float32)
    
    # Build adjacency based on distance and vertical preference
    from scipy.spatial import cKDTree
    tree = cKDTree(nodes)
    
    edges = set()
    for i, node in enumerate(nodes):
        # Find nearby nodes
        nearby_idx = tree.query_ball_point(node, connection_radius)
        
        for j in nearby_idx:
            if i < j:  # Avoid duplicates
                n1, n2 = nodes[i], nodes[j]
                dy = abs(n1[0] - n2[0])
                dx = abs(n1[1] - n2[1])
                
                # Prefer vertical connections (vertical anisotropy)
                vertical_bias = 0.8 if dy > dx else 0.4
                
                if np.random.rand() < vertical_bias:
                    edges.add((i, j))
    
    # Draw struts between connected nodes
    for (i, j) in edges:
        y1, x1 = nodes[i]
        y2, x2 = nodes[j]
        
        # Variable thickness
        thickness = np.random.randint(thickness_range[0], thickness_range[1] + 1)
        
        # Draw line (Bresenham-like)
        length = max(abs(y2 - y1), abs(x2 - x1))
        if length == 0:
            continue
            
        for t in np.linspace(0, 1, length + 1):
            y = int(y1 + t * (y2 - y1))
            x = int(x1 + t * (x2 - x1))
            
            # Apply thickness
            for dy in range(-thickness, thickness + 1):
                for dx in range(-thickness, thickness + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        # Intensity based on distance from center
                        dist = np.sqrt(dy**2 + dx**2)
                        intensity = max(0, 1 - dist / (thickness + 1))
                        network[ny, nx] = max(network[ny, nx], intensity)
    
    # Add nodes as slightly brighter (junctions)
    for y, x in nodes:
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    network[ny, nx] = max(network[ny, nx], 1.2)
    
    return network


def generate_trabecular_with_graph(
    shape: Tuple[int, int],
    mask: np.ndarray,
    intensity: float = 50.0,
    graph_weight: float = 0.4,
    texture_weight: float = 0.6,
    seed: int = 42
) -> np.ndarray:
    """Combine graph-based network with multi-scale texture.
    
    This creates the most realistic trabecular structure by:
    1. Graph network for connectivity/topology
    2. Multi-scale texture for local variation
    3. Blending both for final result
    
    Args:
        shape: Output shape
        mask: Bone region mask
        intensity: Overall HU intensity
        graph_weight: Weight for graph component
        texture_weight: Weight for texture component
        seed: Random seed
        
    Returns:
        Combined trabecular texture with graph connectivity
    """
    # Generate graph-based network
    network = generate_trabecular_network(shape, seed=seed)
    
    # Generate multi-scale texture
    texture = generate_multiscale_trabecular_texture(shape, intensity=intensity, seed=seed)
    
    # Normalize network
    network = (network - network.min()) / (network.max() - network.min() + 1e-8)
    network = network * intensity * 1.5 - intensity * 0.5  # Center around 0
    
    # Combine with weights
    combined = graph_weight * network + texture_weight * texture
    
    # Only apply within mask
    result = np.zeros_like(combined)
    result[mask] = combined[mask]
    
    return result


# =============================================================================
# ULTRA-ADVANCED: 3D COHERENT VOLUME GENERATION
# =============================================================================

def generate_3d_coherent_texture(
    shape_3d: Tuple[int, int, int],
    scale: float = 10.0,
    seed: int = 42
) -> np.ndarray:
    """Generate 3D coherent Perlin noise for cross-slice consistency.
    
    Unlike 2D slice-by-slice generation, this creates a true 3D field
    that ensures smooth transitions between adjacent slices.
    
    Args:
        shape_3d: (H, W, D) volume shape
        scale: Noise scale
        seed: Random seed
        
    Returns:
        3D noise volume with inter-slice coherence
    """
    from scipy.ndimage import zoom
    
    np.random.seed(seed)
    H, W, D = shape_3d
    
    # Generate low-resolution 3D noise grid
    grid_h = max(2, int(H / scale))
    grid_w = max(2, int(W / scale))
    grid_d = max(2, int(D / scale))
    
    # Create random 3D grid
    random_grid = np.random.randn(grid_h, grid_w, grid_d).astype(np.float32)
    
    # Upsample to full resolution (trilinear interpolation)
    zoom_factors = (H / grid_h, W / grid_w, D / grid_d)
    noise_3d = zoom(random_grid, zoom_factors, order=1)
    
    # Ensure correct shape
    noise_3d = noise_3d[:H, :W, :D]
    
    # Normalize
    noise_3d = (noise_3d - noise_3d.mean()) / (noise_3d.std() + 1e-8)
    
    return noise_3d


def generate_3d_trabecular_volume(
    shape_3d: Tuple[int, int, int],
    intensity: float = 40.0,
    scales: list = [20.0, 8.0, 3.0],
    seed: int = 42
) -> np.ndarray:
    """Generate 3D coherent trabecular texture volume.
    
    Multi-scale 3D noise ensuring:
    - Cross-slice coherence
    - Oblique plane consistency
    - Valid 3D topology
    
    Args:
        shape_3d: Volume shape (H, W, D)
        intensity: HU intensity
        scales: List of scales [coarse, medium, fine]
        seed: Random seed
        
    Returns:
        3D trabecular texture volume
    """
    texture = np.zeros(shape_3d, dtype=np.float32)
    weights = [0.35, 0.45, 0.20]
    
    for i, (scale, weight) in enumerate(zip(scales, weights)):
        layer = generate_3d_coherent_texture(shape_3d, scale=scale, seed=seed + i * 100)
        texture += weight * layer
    
    # Apply vertical anisotropy (blur in x-y, keep z sharp)
    texture = gaussian_filter(texture, sigma=(0.5, 2.5, 0.3))
    
    # Normalize and scale
    texture = (texture - texture.mean()) / (texture.std() + 1e-8)
    texture = texture * intensity
    
    return texture


# =============================================================================
# ULTRA-ADVANCED: SOFT TISSUE INTERACTIONS
# =============================================================================

def apply_soft_tissue_context(
    ct: np.ndarray,
    vertebra_mask: np.ndarray,
    add_epidural_fat: bool = True,
    add_muscle_texture: bool = True,
    add_disc_variation: bool = True,
    seed: int = 42
) -> np.ndarray:
    """Apply realistic soft tissue context around vertebra.
    
    Adds:
    - Epidural fat density (-80 to -100 HU)
    - Paraspinal muscle texture (40-80 HU with fiber pattern)
    - Disc hydration variation (50-100 HU gradient)
    
    Args:
        ct: CT image
        vertebra_mask: Vertebra segmentation mask
        add_epidural_fat: Add epidural fat anterior to canal
        add_muscle_texture: Add paraspinal muscle texture
        add_disc_variation: Add disc hydration variation
        seed: Random seed
        
    Returns:
        CT with soft tissue context
    """
    np.random.seed(seed)
    result = ct.copy().astype(np.float32)
    H, W = ct.shape
    
    coords = np.where(vertebra_mask)
    if len(coords[0]) == 0:
        return result
    
    x_min, x_max = coords[0].min(), coords[0].max()
    z_min, z_max = coords[1].min(), coords[1].max()
    
    # === 1. Epidural Fat ===
    if add_epidural_fat:
        # Epidural space: just anterior to spinal canal (posterior to vertebral body)
        epidural_region = np.zeros_like(vertebra_mask, dtype=bool)
        
        # Find canal region (posterior to vertebra)
        canal_x_start = max(0, x_min - 20)
        canal_x_end = x_min
        canal_z_mid = (z_min + z_max) // 2
        canal_z_half = max(5, (z_max - z_min) // 4)
        
        epidural_region[canal_x_start:canal_x_end, 
                       canal_z_mid - canal_z_half:canal_z_mid + canal_z_half] = True
        
        # Apply fat density with subtle texture
        fat_base = -90  # HU for fat
        fat_noise = np.random.randn(*ct.shape) * 10
        result[epidural_region] = fat_base + fat_noise[epidural_region]
        
        # Smooth transition
        result = gaussian_filter(result, sigma=0.5)
    
    # === 2. Paraspinal Muscle Texture ===
    if add_muscle_texture:
        # Muscle regions: lateral to vertebra
        muscle_left = np.zeros_like(vertebra_mask, dtype=bool)
        muscle_right = np.zeros_like(vertebra_mask, dtype=bool)
        
        # Lateral regions
        muscle_width = 40
        muscle_left[x_min:x_max, max(0, z_min - muscle_width):z_min] = True
        muscle_right[x_min:x_max, z_max:min(W, z_max + muscle_width)] = True
        muscle_region = muscle_left | muscle_right
        muscle_region = muscle_region & ~vertebra_mask
        
        # Muscle fiber pattern (parallel lines)
        fiber_pattern = generate_perlin_noise_2d((H, W), scale=15.0, seed=seed+200)
        fiber_pattern = gaussian_filter(fiber_pattern, sigma=(3.0, 0.5))  # Elongated in x
        
        muscle_base = 50  # HU for muscle
        result[muscle_region] = muscle_base + fiber_pattern[muscle_region] * 25
    
    # === 3. Disc Hydration Variation ===
    if add_disc_variation:
        # Disc regions: above and below vertebra
        disc_height = 8
        
        # Superior disc
        disc_sup = np.zeros_like(vertebra_mask, dtype=bool)
        disc_sup[x_min:x_max, max(0, z_min - disc_height):z_min] = True
        disc_sup = disc_sup & ~vertebra_mask
        
        # Inferior disc
        disc_inf = np.zeros_like(vertebra_mask, dtype=bool)
        disc_inf[x_min:x_max, z_max:min(W, z_max + disc_height)] = True
        disc_inf = disc_inf & ~vertebra_mask
        
        disc_region = disc_sup | disc_inf
        
        # Hydration gradient (nucleus pulposus brighter than annulus)
        x_coords = np.arange(H)
        z_coords = np.arange(W)
        xx, zz = np.meshgrid(z_coords, x_coords)
        
        # Center of disc
        center_x = (x_min + x_max) / 2
        
        for disc_mask, disc_z in [(disc_sup, z_min - disc_height//2), 
                                   (disc_inf, z_max + disc_height//2)]:
            if disc_mask.sum() == 0:
                continue
            
            # Radial gradient from center
            dist_from_center = np.sqrt((xx - center_x)**2)
            max_dist = (x_max - x_min) / 2
            
            # Nucleus (center) brighter, annulus (edge) darker
            hydration = 90 - 40 * (dist_from_center / (max_dist + 1e-8))
            hydration = np.clip(hydration, 50, 100)
            
            result[disc_mask] = hydration[disc_mask]
    
    return result


def apply_full_rendering_pipeline(
    ct: np.ndarray,
    vertebra_mask: np.ndarray,
    use_graph_trabecular: bool = True,
    add_soft_tissue: bool = True,
    fracture_type: str = None,
    fracture_severity: float = 0.5,
    seed: int = 42
) -> np.ndarray:
    """Complete ultra-advanced CT rendering pipeline.
    
    Combines all features:
    1. Graph-based trabecular network OR multi-scale texture
    2. Hierarchical fracture surface (if fracture_type specified)
    3. Burst retropulsion (if type is 'burst')
    4. Cortical thickness variation
    5. Soft tissue interactions
    6. CT physics (partial volume, noise, blur)
    
    Args:
        ct: Original CT
        vertebra_mask: Vertebra segmentation
        use_graph_trabecular: Use graph-based network for trabeculae
        add_soft_tissue: Add soft tissue context
        fracture_type: 'compression', 'wedge', 'burst', or None
        fracture_severity: Severity 0-1
        seed: Random seed
        
    Returns:
        Fully processed realistic CT
    """
    result = ct.copy().astype(np.float32)
    
    # 1. Trabecular texture (graph-based or multi-scale)
    if use_graph_trabecular:
        texture = generate_trabecular_with_graph(
            result.shape, vertebra_mask, intensity=40, seed=seed
        )
    else:
        texture = generate_multiscale_trabecular_texture(
            result.shape, intensity=35, local_asymmetry=True, seed=seed
        )
    result[vertebra_mask] += texture[vertebra_mask]
    
    # 2. Apply fracture if specified
    if fracture_type in ['compression', 'wedge']:
        coords = np.where(vertebra_mask)
        z_min, z_max = coords[1].min(), coords[1].max()
        fracture_z = z_min + int((z_max - z_min) * 0.3)
        
        frac_mask, frac_field = generate_hierarchical_fracture_surface(
            result.shape, fracture_z, roughness=fracture_severity * 0.8, seed=seed
        )
        result[frac_mask & vertebra_mask] *= (1 + frac_field[frac_mask & vertebra_mask])
    
    elif fracture_type == 'burst':
        result, _ = simulate_burst_retropulsion(
            result, vertebra_mask, severity=fracture_severity, seed=seed
        )
    
    # 3. Cortical variation
    result = apply_cortical_thickness_variation(
        result, vertebra_mask, thickness_variation=0.35, seed=seed
    )
    
    # 4. Soft tissue context
    if add_soft_tissue:
        result = apply_soft_tissue_context(result, vertebra_mask, seed=seed)
    
    # 5. CT physics
    result = apply_partial_volume_effect(result, vertebra_mask, sigma=0.5)
    result = apply_poisson_noise(result, dose_level=1.0)
    result = apply_detector_blur(result, psf_sigma=0.2)
    
    return np.clip(result, -1000, 3000)


if __name__ == "__main__":
    test_ct_physics()

