
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter

def generate_lytic_lesion(
    ct_volume: np.ndarray,
    mask_volume: np.ndarray,
    center: tuple,
    radius_mm: float,
    irregularity: float = 0.5,
    margin_width: float = 2.0
):
    """
    Generate osteolytic lesion (bone destruction).
    
    Physics:
    - Reduced HU density (approaching soft tissue/fluid ~30-50 HU)
    - Irregular margins (infiltration)
    - Partial volume effect at boundary
    """
    z, y, x = center
    
    # Create coordinate grid relative to center
    zz, yy, xx = np.ogrid[:ct_volume.shape[0], :ct_volume.shape[1], :ct_volume.shape[2]]
    dist = np.sqrt((zz - z)**2 + (yy - y)**2 + (xx - x)**2)
    
    # Base sphere
    lesion_mask = dist <= radius_mm
    
    # Add irregularity (Perlin-like noise)
    if irregularity > 0:
        noise = np.random.randn(*ct_volume.shape)
        noise = gaussian_filter(noise, sigma=2.0)
        dist_distorted = dist + noise * irregularity * radius_mm
        lesion_mask = dist_distorted <= radius_mm
    
    # Only affect bone
    lesion_mask = lesion_mask & (mask_volume > 0)
    
    # Soft transition at margin
    margin = ndi.binary_dilation(lesion_mask, iterations=int(margin_width)) & ~lesion_mask
    
    # Apply density reduction
    # Lytic center: ~40 HU (soft tissue/fluid)
    # Margin: Blend
    
    ct_out = ct_volume.copy()
    
    # Core destruction
    ct_out[lesion_mask] = 40 + np.random.randn(lesion_mask.sum()) * 10
    
    # Margin blending
    # Simple averaging with valid neighbors? Or just intermediate density?
    # Osteolysis often leaves some trabeculae.
    # Reduce density by 50-80%
    ct_out[margin] = ct_out[margin] * 0.4 + 40 * 0.6
    
    return ct_out, lesion_mask

def generate_blastic_lesion(
    ct_volume: np.ndarray,
    mask_volume: np.ndarray,
    center: tuple,
    radius_mm: float,
    density_increase: float = 800.0,
    texture_scale: float = 2.0
):
    """
    Generate osteoblastic lesion (sclerosis).
    
    Physics:
    - Increased HU density (up to cortical bone levels ~800-1200 HU)
    - Woven bone texture (disorganized)
    """
    z, y, x = center
    zz, yy, xx = np.ogrid[:ct_volume.shape[0], :ct_volume.shape[1], :ct_volume.shape[2]]
    dist = np.sqrt((zz - z)**2 + (yy - y)**2 + (xx - x)**2)
    
    # Sclerotic region
    lesion_mask = (dist <= radius_mm) & (mask_volume > 0)
    
    if lesion_mask.sum() == 0:
        return ct_volume, lesion_mask
        
    ct_out = ct_volume.copy()
    
    # Generate woven bone texture (high freq noise)
    noise = np.random.randn(*ct_volume.shape)
    noise = gaussian_filter(noise, sigma=texture_scale)
    # Normalize to 0-1
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    
    # Add density
    # Blastic lesions are additive
    added_density = density_increase * (0.8 + 0.4 * noise) # Variation
    
    # Fade at edges (Gaussian)
    fade = np.exp(-0.5 * (dist[lesion_mask] / (radius_mm * 0.8))**2)
    
    ct_out[lesion_mask] += added_density[lesion_mask] * fade
    
    # Clip to max bone density (~2000)
    ct_out[lesion_mask] = np.clip(ct_out[lesion_mask], -1000, 2500)
    
    return ct_out, lesion_mask
