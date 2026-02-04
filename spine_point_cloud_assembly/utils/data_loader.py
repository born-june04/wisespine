"""
Data Loader for Point Cloud Pretraining
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import json


class VertebraPointCloudDataset(Dataset):
    """
    Dataset for vertebra point clouds with features
    
    Expected file structure:
    point_cloud_dir/
        {subject_id}/
            vertebra_{v_id}.npz
                - points: (N, 3) - point positions
                - normals: (N, 3) - surface normals
                - curvature: (N, 2) - k1, k2
                - label: int - vertebra label (1-24)
    """
    
    def __init__(
        self,
        point_cloud_dir: Path,
        split: str = 'train',
        max_points: int = 2048,
        use_curvature: bool = True,
        augment: bool = True,
    ):
        self.point_cloud_dir = Path(point_cloud_dir)
        self.split = split
        self.max_points = max_points
        self.use_curvature = use_curvature
        self.augment = augment and (split == 'train')
        
        # Find all point cloud files
        self.samples = []
        for subject_dir in self.point_cloud_dir.iterdir():
            if not subject_dir.is_dir():
                continue
            
            # Look for .npy point cloud files
            for pc_file in subject_dir.glob('vertebra_*_points.npy'):
                v_id = int(pc_file.stem.split('_')[1])
                self.samples.append({
                    'file': pc_file,
                    'subject_id': subject_dir.name,
                    'vertebra_id': v_id,
                })
        
        print(f"Loaded {len(self.samples)} vertebra point clouds from {split} split")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        pc_file = sample['file']
        subject_id = sample['subject_id']
        v_id = sample['vertebra_id']
        
        # Load point cloud (from .npy file)
        points = np.load(pc_file).astype(np.float32)  # (N, 3)
        label = int(v_id)
        
        # Load features (from .npz file or separate files)
        feature_file = pc_file.parent / f'vertebra_{v_id}_features.npz'
        normals_file = pc_file.parent / f'vertebra_{v_id}_normals.npy'
        curvature_file = pc_file.parent / f'vertebra_{v_id}_curvature.npz'
        
        # Load normals (try features.npz first, then separate file)
        if feature_file.exists():
            features_data = np.load(feature_file)
            if 'normals' in features_data:
                normals = features_data['normals'].astype(np.float32)  # (N, 3)
            elif normals_file.exists():
                normals = np.load(normals_file).astype(np.float32)  # (N, 3)
            else:
                normals = np.zeros((points.shape[0], 3), dtype=np.float32)
        elif normals_file.exists():
            normals = np.load(normals_file).astype(np.float32)  # (N, 3)
        else:
            normals = np.zeros((points.shape[0], 3), dtype=np.float32)
        
        # Load curvature (try features.npz first, then separate file)
        if feature_file.exists():
            features_data = np.load(feature_file)
            if self.use_curvature and 'k1' in features_data and 'k2' in features_data:
                k1 = features_data['k1'].astype(np.float32)
                k2 = features_data['k2'].astype(np.float32)
                curvature = np.stack([k1, k2], axis=-1)  # (N, 2)
            elif curvature_file.exists():
                curvature_data = np.load(curvature_file)
                k1 = curvature_data['k1'].astype(np.float32)
                k2 = curvature_data['k2'].astype(np.float32)
                curvature = np.stack([k1, k2], axis=-1)  # (N, 2)
            else:
                curvature = np.zeros((points.shape[0], 2), dtype=np.float32)
        elif curvature_file.exists():
            curvature_data = np.load(curvature_file)
            k1 = curvature_data['k1'].astype(np.float32)
            k2 = curvature_data['k2'].astype(np.float32)
            curvature = np.stack([k1, k2], axis=-1)  # (N, 2)
        else:
            curvature = np.zeros((points.shape[0], 2), dtype=np.float32)
        
        # Center points by centroid
        centroid = points.mean(axis=0)
        points = points - centroid
        
        # Sample/filter points to max_points
        N = points.shape[0]
        if N > self.max_points:
            # Random sampling
            indices = np.random.choice(N, self.max_points, replace=False)
            points = points[indices]
            normals = normals[indices]
            curvature = curvature[indices]
        elif N < self.max_points:
            # Pad with zeros
            pad_size = self.max_points - N
            points = np.pad(points, ((0, pad_size), (0, 0)), mode='constant')
            normals = np.pad(normals, ((0, pad_size), (0, 0)), mode='constant')
            curvature = np.pad(curvature, ((0, pad_size), (0, 0)), mode='constant')
        
        # Data augmentation (training only)
        if self.augment:
            points, normals = self._augment(points, normals)
        
        # Combine features
        features = np.concatenate([normals, curvature], axis=-1)  # (N, 5)
        
        return {
            'points': torch.from_numpy(points).float(),
            'features': torch.from_numpy(features).float(),
            'label': torch.tensor(label, dtype=torch.long),
            'subject_id': sample['subject_id'],
        }
    
    def _augment(self, points: np.ndarray, normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation"""
        # Random rotation
        if np.random.rand() > 0.5:
            R = self._random_rotation_matrix()
            points = points @ R.T
            normals = normals @ R.T
        
        # Random jitter
        if np.random.rand() > 0.5:
            jitter = np.random.normal(0, 0.01, points.shape)
            points = points + jitter
        
        # Random scaling
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            points = points * scale
        
        return points, normals
    
    def _random_rotation_matrix(self) -> np.ndarray:
        """Generate random 3D rotation matrix"""
        # Random axis-angle
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.uniform(0, 2 * np.pi)
        
        # Rodrigues' rotation formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        return R


def create_dataloader(
    point_cloud_dir: Path,
    split: str = 'train',
    batch_size: int = 32,
    num_workers: int = 4,
    max_points: int = 2048,
    use_curvature: bool = True,
    augment: bool = True,
    shuffle: bool = True,
    return_dataset: bool = False,
) -> DataLoader:
    """Create DataLoader for point cloud dataset
    
    Args:
        return_dataset: If True, return dataset instead of DataLoader (for DDP samplers)
    """
    dataset = VertebraPointCloudDataset(
        point_cloud_dir=point_cloud_dir,
        split=split,
        max_points=max_points,
        use_curvature=use_curvature,
        augment=augment,
    )
    
    if return_dataset:
        return dataset
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )
    
    return dataloader

