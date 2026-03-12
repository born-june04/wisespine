#!/usr/bin/env python3
"""
Assembly Wrapper for Phase 2: Mask → Mesh pipeline.

This wraps the existing spine_point_cloud_assembly module to provide
a simple interface for adversarial robustness training:
  - Input: Vertebra mask (possibly corrupted by adversary)
  - Output: Mesh (for quality evaluation)

For now, we use a simplified pipeline:
  mask → point cloud → marching cubes → mesh
  
Later, this can be extended to use the full assembly transformer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional
import numpy as np

try:
    import nibabel as nib
except ImportError:
    raise RuntimeError("nibabel required: pip install nibabel")

try:
    from skimage import measure
except ImportError:
    raise RuntimeError("scikit-image required: pip install scikit-image")

try:
    import trimesh
except ImportError:
    raise RuntimeError("trimesh required: pip install trimesh")


class SimpleAssemblyModule:
    """
    Simplified assembly: mask → mesh via marching cubes.
    
    This is a baseline/placeholder for Phase 2.
    Future: integrate the full assembly transformer for learned robust reconstruction.
    """
    
    def __init__(self, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        Args:
            spacing: Voxel spacing (mm) for marching cubes
        """
        self.spacing = spacing
    
    def mask_to_mesh(
        self,
        mask: np.ndarray,
        label: Optional[int] = None,
        step_size: int = 1,
    ) -> Optional[trimesh.Trimesh]:
        """
        Convert a binary mask (or extract a specific label) to a mesh.
        
        Args:
            mask: (H, W, D) uint16 or bool array
            label: If provided, extract this label from multi-label mask
            step_size: Marching cubes step size (higher = faster but coarser)
        
        Returns:
            Trimesh object or None if empty
        """
        # Extract binary mask for this label
        if label is not None:
            binary_mask = (mask == label).astype(np.uint8)
        else:
            binary_mask = (mask > 0).astype(np.uint8)
        
        if binary_mask.sum() == 0:
            return None
        
        # Marching cubes
        try:
            verts, faces, normals, values = measure.marching_cubes(
                binary_mask,
                level=0.5,
                spacing=self.spacing,
                step_size=step_size,
            )
        except Exception as e:
            print(f"Marching cubes failed for label {label}: {e}")
            return None
        
        # Create trimesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        return mesh
    
    def multilabel_mask_to_meshes(
        self,
        mask: np.ndarray,
        step_size: int = 1,
    ) -> dict[int, trimesh.Trimesh]:
        """
        Convert multi-label mask to per-label meshes.
        
        Args:
            mask: (H, W, D) multi-label segmentation
            step_size: Marching cubes step size
        
        Returns:
            Dictionary mapping label → mesh
        """
        labels = sorted(set(np.unique(mask).tolist()) - {0})
        meshes = {}
        
        for label in labels:
            mesh = self.mask_to_mesh(mask, label=label, step_size=step_size)
            if mesh is not None:
                meshes[int(label)] = mesh
        
        return meshes
    
    def compute_mesh_metrics(
        self,
        pred_mesh: trimesh.Trimesh,
        gt_mesh: trimesh.Trimesh,
        num_samples: int = 10000,
    ) -> dict:
        """
        Compute mesh quality metrics (Chamfer distance, Hausdorff, volume error).
        
        Args:
            pred_mesh: Predicted mesh
            gt_mesh: Ground truth mesh
            num_samples: Number of points to sample for distance computation
        
        Returns:
            Dictionary of metrics
        """
        # Sample points from both meshes
        pred_points, _ = trimesh.sample.sample_surface(pred_mesh, num_samples)
        gt_points, _ = trimesh.sample.sample_surface(gt_mesh, num_samples)
        
        # Chamfer distance (symmetric)
        from scipy.spatial import cKDTree
        
        tree_pred = cKDTree(pred_points)
        tree_gt = cKDTree(gt_points)
        
        # GT → pred distances
        dists_gt_to_pred, _ = tree_pred.query(gt_points, k=1)
        # Pred → GT distances
        dists_pred_to_gt, _ = tree_gt.query(pred_points, k=1)
        
        chamfer = (dists_gt_to_pred.mean() + dists_pred_to_gt.mean()) / 2.0
        hausdorff = max(dists_gt_to_pred.max(), dists_pred_to_gt.max())
        
        # Volume error
        vol_pred = pred_mesh.volume if pred_mesh.is_watertight else 0.0
        vol_gt = gt_mesh.volume if gt_mesh.is_watertight else 0.0
        vol_error = abs(vol_pred - vol_gt) / (vol_gt + 1e-6)
        
        return {
            "chamfer_distance": float(chamfer),
            "hausdorff_distance": float(hausdorff),
            "volume_pred": float(vol_pred),
            "volume_gt": float(vol_gt),
            "volume_error": float(vol_error),
        }
    
    def compute_assembly_loss(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        labels: Optional[list[int]] = None,
        step_size: int = 2,
    ) -> dict:
        """
        Compute assembly loss: per-label mesh quality metrics.
        
        Args:
            pred_mask: Predicted (possibly corrupted) mask
            gt_mask: Ground truth mask
            labels: List of labels to evaluate (if None, use all in GT)
            step_size: Marching cubes step size
        
        Returns:
            Dictionary with overall and per-label metrics
        """
        if labels is None:
            labels = sorted(set(np.unique(gt_mask).tolist()) - {0})
        
        # Generate meshes
        pred_meshes = self.multilabel_mask_to_meshes(pred_mask, step_size=step_size)
        gt_meshes = self.multilabel_mask_to_meshes(gt_mask, step_size=step_size)
        
        per_label_metrics = {}
        chamfer_distances = []
        
        for label in labels:
            if label not in gt_meshes:
                continue  # GT doesn't have this label
            
            gt_mesh = gt_meshes[label]
            
            if label not in pred_meshes:
                # Prediction is missing this label → very bad
                per_label_metrics[int(label)] = {
                    "chamfer_distance": float('inf'),
                    "hausdorff_distance": float('inf'),
                    "volume_error": 1.0,
                    "status": "missing",
                }
                chamfer_distances.append(1000.0)  # Large penalty
                continue
            
            pred_mesh = pred_meshes[label]
            
            # Compute metrics
            try:
                metrics = self.compute_mesh_metrics(pred_mesh, gt_mesh, num_samples=5000)
                metrics["status"] = "success"
                per_label_metrics[int(label)] = metrics
                chamfer_distances.append(metrics["chamfer_distance"])
            except Exception as e:
                per_label_metrics[int(label)] = {
                    "status": "failed",
                    "error": str(e),
                }
                chamfer_distances.append(1000.0)
        
        # Overall metrics
        mean_chamfer = float(np.mean(chamfer_distances)) if chamfer_distances else float('inf')
        
        return {
            "mean_chamfer_distance": mean_chamfer,
            "num_labels_evaluated": len(labels),
            "num_labels_success": sum(1 for m in per_label_metrics.values() if m.get("status") == "success"),
            "per_label": per_label_metrics,
        }


# For later: integrate the full assembly transformer
class LearnedAssemblyModule:
    """
    Full learned assembly using the spine_point_cloud_assembly transformer.
    
    This will be implemented in Phase 2-B after the adversary baseline is working.
    """
    
    def __init__(self, checkpoint_path: Optional[Path] = None):
        # TODO: Load pretrained assembly transformer
        raise NotImplementedError("Learned assembly integration coming in Phase 2-B")

