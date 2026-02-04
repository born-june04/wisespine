"""
Spine Point Cloud Assembly Models
"""

# New SE(3)-equivariant encoder (e3nn-based)
from .encoder_se3 import SE3PointEncoder, features_to_irreps

# Legacy encoder (kept for reference/comparison)
# from .encoder_legacy import VertebraPointEncoder

from .pretraining import (
    RotationCanonicalizationLoss,
    ContrastiveVertebraTypeLoss,
    MaskedPointModelingLoss,
)

# Assembly transformer
from .assembly import SpineAssemblyTransformer, rot6d_to_matrix, geodesic_distance
from .assembly_spinal_field import SpineAssemblySpinalField
from .assembly_losses import (
    OrderingLoss,
    AssemblyLoss,
    MissingCompletionLoss,
    CombinedAssemblyLoss,
    compute_losses,
)

__all__ = [
    'SE3PointEncoder',
    'features_to_irreps',
    'RotationCanonicalizationLoss',
    'ContrastiveVertebraTypeLoss',
    'MaskedPointModelingLoss',
    'SpineAssemblyTransformer',
    'SpineAssemblySpinalField',
    'rot6d_to_matrix',
    'geodesic_distance',
    'OrderingLoss',
    'AssemblyLoss',
    'MissingCompletionLoss',
    'CombinedAssemblyLoss',
    'compute_losses',
]

