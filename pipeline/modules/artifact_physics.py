
import numpy as np
import scipy.ndimage as ndi
from skimage.transform import radon, iradon, rescale
from scipy.ndimage import gaussian_filter

def simulate_metal_artifact_radon(
    ct_slice: np.ndarray,
    metal_mask_slice: np.ndarray,
    num_angles: int = 180,
    noise_level: float = 10000.0,  # Incident photons
    beam_hardening_strength: float = 0.2, # 0.0-1.0
    scatter_sigma: float = 2.0,
    scatter_strength: float = 0.05
) -> np.ndarray:
    """
    Simulate metal artifacts using forward projection (Radon transform).
    
    Physics Model:
    1. Forward Project: CT -> Sinogram (Attenuation Integral)
    2. Beam Hardening: Polynomial hardening of high-attenuation paths.
    3. Scatter: Low-frequency background added to detectors (sinogram).
    4. Photon Starvation: Statistical noise (Poisson) in low-count rays.
    5. Back Project: Reconstruct artifact image.
    
    Args:
        ct_slice: 2D CT image (HU or attenuation)
        metal_mask_slice: 2D Boolean mask of metal
        
    Returns:
        Artifact-corrupted CT slice
    """
    
    # Handle non-square images (Radon/IRadon prefers square or returns square)
    original_shape = ct_slice.shape
    max_dim = max(original_shape)
    
    pad_h = (max_dim - original_shape[0]) // 2
    pad_w = (max_dim - original_shape[1]) // 2
    
    # Pad to square (max_dim, max_dim)
    # Use 'edge' or constant for padding? Air (-1000 HU)
    # But we work in 'mu' space later.
    # Let's simple-pad the input CT and Mask.
    
    if original_shape[0] != original_shape[1]:
        ct_padded = np.full((max_dim, max_dim), -1000.0, dtype=ct_slice.dtype)
        mask_padded = np.zeros((max_dim, max_dim), dtype=bool)
        
        # Center the image
        r0 = (max_dim - original_shape[0]) // 2
        c0 = (max_dim - original_shape[1]) // 2
        
        ct_padded[r0:r0+original_shape[0], c0:c0+original_shape[1]] = ct_slice
        mask_padded[r0:r0+original_shape[0], c0:c0+original_shape[1]] = metal_mask_slice
        
        working_ct = ct_padded
        working_mask = mask_padded
    else:
        working_ct = ct_slice
        working_mask = metal_mask_slice
        r0, c0 = 0, 0
    
    # 1. Prepare Image
    # Convert HU to approximate linear attenuation coefficient (mu)
    # Water = 0 HU = 0.02 /mm
    # Air = -1000 HU = 0.0
    # Bone = 1000 HU = 0.04
    # Metal = 30000 HU = ~0.6 (very high)
    
    # Simple mapping: mu = (HU + 1000) / 1000 * 0.02
    mu_map = (working_ct + 1000) / 1000 * 0.02
    mu_map = np.clip(mu_map, 0, None)
    
    # Set metal mu very high
    mu_map[working_mask] = 2.0  # Titanium/Steel is significantly higher than bone
    
    # 2. Forward Projection (Radon)
    theta = np.linspace(0., 180., num_angles, endpoint=False)
    sinogram = radon(mu_map, theta=theta, circle=False)
    
    # 3. Simulate Physics on Sinogram
    
    # A. Beam Hardening
    # High attenuation paths are hardened (lower apparent attenuation than linear)
    # P_measured = P_true - alpha * P_true^2
    # Metal paths are extremely high attenuation
    sinogram_bh = sinogram.copy()
    # Apply soft polynomial cupping
    sinogram_bh = sinogram_bh - beam_hardening_strength * 0.05 * (sinogram_bh**2)
    
    # B. Scatter (Glare)
    # Scatter is often modeled as convolution on the detector (sinogram columns)
    # It adds a low-frequency bias, reducing contrast
    scatter_field = gaussian_filter(sinogram, sigma=(scatter_sigma, 0)) # Blur along detector channel
    # Intensity = I0 * exp(-P) + Scatter
    # We work in P-space, so this effectively reduces P
    # P_new = -ln( exp(-P) + scatter )
    # Approximation: P_new = P - scatter_strength * scatter_field
    sinogram_bh -= scatter_strength * scatter_field
    
    # C. Photon Starvation (Poisson Noise)
    # I = I0 * exp(-P)
    I0 = noise_level
    transmission = np.exp(-sinogram_bh)
    expected_counts = I0 * transmission
    
    # Add Poisson noise
    rng = np.random.default_rng()
    noisy_counts = rng.poisson(expected_counts)
    noisy_counts = np.maximum(noisy_counts, 1) # Avoid log(0)
    
    # Convert back to Projection
    sinogram_noisy = -np.log(noisy_counts / I0)
    
    # 4. Filtered Back Projection
    reconstruction = iradon(sinogram_noisy, theta=theta, circle=False)
    
    # 5. Convert back to HU
    # mu = ... -> HU = ...
    ct_out = (reconstruction / 0.02 * 1000) - 1000
    
    # Recovery of metal shape (FBP blurs it)
    # Restore original metal pixels to high HU (FBP often creates undershoot/cupping inside metal)
    
    # Crop if needed
    if original_shape[0] != original_shape[1]:
        # Resize/Crop check
        # Iradon output shape matches input max_dim usually
        if ct_out.shape != working_ct.shape:
            # Resize? Usually iradon returns same size if circle=False and input is standard
            # But let's assume it matches working_ct (max_dim, max_dim)
            pass
            
        final_out = ct_out[r0:r0+original_shape[0], c0:c0+original_shape[1]]
    else:
        final_out = ct_out
        
    final_out[metal_mask_slice] = 30000 # Metal HU
    
    return final_out


if __name__ == "__main__":
    # Test
    import matplotlib.pyplot as plt
    phantom = np.zeros((256, 256))
    phantom[50:200, 50:200] = 0 # Water
    phantom[100:150, 100:150] = 1000 # Bone
    
    metal = np.zeros_(256, 256, dtype=bool)
    metal[120:130, 120:130] = True # Screw
    
    out = simulate_metal_artifact_radon(phantom, metal)
    
    plt.figure()
    plt.imshow(out, cmap='gray', vmin=-1000, vmax=3000)
    plt.title("Simulated Metal Artifact (Radon)")
    plt.savefig("test_artifact.png")
