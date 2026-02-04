"""
Data Loader for Assembly Tasks

Loads multiple vertebrae per subject for assembly training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json
from collections import defaultdict
import multiprocessing

# Set multiprocessing start method to 'spawn' to avoid CUDA issues
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

from .data_loader import VertebraPointCloudDataset


class AssemblyDataset(Dataset):
    """
    Dataset for assembly tasks.
    
    Loads pre-extracted embeddings for assembly training (faster than encoding on-the-fly).
    """
    
    def __init__(
        self,
        embedding_dir: Path,
        point_cloud_dir: Path,  # Add original point cloud directory
        split: str = 'train',
        max_vertebrae: int = 30,
        augment: bool = True,
    ):
        self.embedding_dir = Path(embedding_dir)
        self.point_cloud_dir = Path(point_cloud_dir)  # Store for loading original points
        self.split = split
        self.max_vertebrae = max_vertebrae
        self.augment = augment and (split == 'train')
        
        # Load metadata
        metadata_file = self.embedding_dir / 'metadata.json'
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_file}\n"
                f"Please run extract_assembly_embeddings.py first to pre-extract embeddings."
            )
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Filter subjects with at least 2 vertebrae
        self.subject_list = [
            subject_id for subject_id, info in metadata.items()
            if info['num_vertebrae'] >= 2
        ]
        
        print(f"Loaded {len(self.subject_list)} subjects with multiple vertebrae from {split} split")
        print(f"Total vertebrae: {sum(metadata[sid]['num_vertebrae'] for sid in self.subject_list)}")
    
    def __len__(self) -> int:
        return len(self.subject_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load pre-extracted embeddings for a subject.
        
        Returns:
            dict with keys:
                - 'embeddings': (N, embed_dim) - encoder embeddings
                - 'vertebra_ids': (N,) - vertebra IDs (0-25 for V1-V26)
                - 'mask': (N,) - boolean mask (True = present)
                - 'subject_id': str
                - 'points': (N, M, 3) - point clouds (for assembly target computation)
                - 'features': (N, M, F) - features (for assembly target computation)
        """
        subject_id = self.subject_list[idx]
        subject_dir = self.embedding_dir / subject_id
        
        # Load all vertebra embeddings for this subject
        embedding_files = sorted(subject_dir.glob('vertebra_*_embedding.npz'))
        
        if len(embedding_files) == 0:
            raise FileNotFoundError(f"No embedding files found for subject {subject_id}")
        
        # Limit to max_vertebrae
        if len(embedding_files) > self.max_vertebrae:
            embedding_files = embedding_files[:self.max_vertebrae]
        
        N = len(embedding_files)
        
        # Load embeddings and data
        embeddings_list = []
        vertebra_ids = []
        points_list = []
        features_list = []
        
        for emb_file in embedding_files:
            v_id = int(emb_file.stem.split('_')[1])
            data = np.load(emb_file)
            
            embedding = torch.from_numpy(data['embedding']).float()  # (embed_dim,)
            # Use centered points for features (encoder expects centered)
            points_centered = torch.from_numpy(data['points']).float()  # (M, 3) - centered
            features = torch.from_numpy(data['features']).float()  # (M, 5)
            
            # Load ORIGINAL (non-centered) points for assembly target computation
            # Original points are in the point cloud directory
            pc_file = self.point_cloud_dir / subject_id / f'vertebra_{v_id}_points.npy'
            if pc_file.exists():
                points_original = torch.from_numpy(np.load(pc_file)).float()  # (M_orig, 3) - original
                # Subsample to match centered points size if needed
                if len(points_original) != len(points_centered):
                    if len(points_original) > len(points_centered):
                        # Use same indices as in extraction (random, but we'll approximate)
                        indices = np.random.choice(len(points_original), len(points_centered), replace=False)
                        points_original = points_original[indices]
                    else:
                        # Pad with zeros (shouldn't happen, but handle it)
                        pad_size = len(points_centered) - len(points_original)
                        points_original = torch.cat([points_original, torch.zeros(pad_size, 3)], dim=0)
            else:
                # Fallback: use centered points (will cause small translation targets)
                print(f"Warning: Original points not found for {pc_file}, using centered points")
                points_original = points_centered
            
            embeddings_list.append(embedding)
            vertebra_ids.append(v_id - 1)  # Convert to 0-indexed (V1=0, V2=1, ...)
            points_list.append(points_original)  # Use ORIGINAL points for target computation
            features_list.append(features)
        
        # Pad to max_vertebrae
        embed_dim = embeddings_list[0].shape[0]
        if N < self.max_vertebrae:
            pad_size = self.max_vertebrae - N
            # Pad with zero embeddings
            zero_embedding = torch.zeros(embed_dim)
            embeddings_list.extend([zero_embedding] * pad_size)
            vertebra_ids.extend([-1] * pad_size)  # -1 for missing
            # Get M from first points tensor
            M = points_list[0].shape[0] if len(points_list) > 0 else 2048
            points_list.extend([torch.zeros(M, 3)] * pad_size)
            features_list.extend([torch.zeros(M, 5)] * pad_size)
        
        # Stack
        embeddings = torch.stack(embeddings_list)  # (max_vertebrae, embed_dim)
        vertebra_ids = torch.tensor(vertebra_ids, dtype=torch.long)  # (max_vertebrae,)
        mask = vertebra_ids != -1  # (max_vertebrae,)
        points = torch.stack(points_list)  # (max_vertebrae, M, 3)
        features = torch.stack(features_list)  # (max_vertebrae, M, 5)
        
        # Compute assembly targets (relative positions and rotations)
        # For now, use centroids as relative positions
        centroids = points.mean(dim=1)  # (max_vertebrae, 3)
        # Relative translation: difference from first vertebra
        if mask.sum() > 0:
            first_centroid = centroids[mask][0]  # (3,)
            relative_translation = centroids - first_centroid.unsqueeze(0)  # (max_vertebrae, 3)
        else:
            relative_translation = torch.zeros(self.max_vertebrae, 3)
        
        # For rotation, use identity quaternion for now
        # TODO: Compute actual relative rotations from point clouds
        relative_rotation = torch.zeros(self.max_vertebrae, 4)
        relative_rotation[:, 3] = 1.0  # Identity quaternion [x, y, z, w] = [0, 0, 0, 1]
        
        return {
            'embeddings': embeddings,  # (max_vertebrae, embed_dim)
            'vertebra_ids': vertebra_ids,  # (max_vertebrae,)
            'mask': mask,  # (max_vertebrae,)
            'subject_id': subject_id,
            'points': points,  # (max_vertebrae, M, 3)
            'features': features,  # (max_vertebrae, M, 5)
            'assembly': {
                'translation': relative_translation,  # (max_vertebrae, 3)
                'rotation': relative_rotation,  # (max_vertebrae, 4)
            },
            'missing_completion': torch.zeros(self.max_vertebrae, embed_dim),  # Placeholder
        }


def create_assembly_dataloader(
    embedding_dir: Path,
    point_cloud_dir: Path,  # Add original point cloud directory
    split: str = 'train',
    batch_size: int = 16,
    num_workers: int = 4,
    max_vertebrae: int = 30,
    augment: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create DataLoader for assembly tasks.
    
    Args:
        embedding_dir: Directory containing pre-extracted embeddings (from extract_assembly_embeddings.py)
        point_cloud_dir: Directory containing original (non-centered) point clouds
        split: Data split ('train', 'val', 'test')
        batch_size: Batch size
        num_workers: Number of data loader workers
        max_vertebrae: Maximum number of vertebrae per subject
        augment: Whether to apply augmentation (training only)
        shuffle: Whether to shuffle data
    """
    dataset = AssemblyDataset(
        embedding_dir=embedding_dir,
        point_cloud_dir=point_cloud_dir,  # Pass original point cloud directory
        split=split,
        max_vertebrae=max_vertebrae,
        augment=augment,
    )
    
    def collate_fn(batch):
        """Collate function for assembly batch"""
        embeddings = torch.stack([item['embeddings'] for item in batch])  # (B, N, embed_dim)
        vertebra_ids = torch.stack([item['vertebra_ids'] for item in batch])  # (B, N)
        mask = torch.stack([item['mask'] for item in batch])  # (B, N)
        subject_ids = [item['subject_id'] for item in batch]
        points = torch.stack([item['points'] for item in batch])  # (B, N, M, 3)
        features = torch.stack([item['features'] for item in batch])  # (B, N, M, 5)
        
        # Assembly targets
        translations = torch.stack([item['assembly']['translation'] for item in batch])  # (B, N, 3)
        rotations = torch.stack([item['assembly']['rotation'] for item in batch])  # (B, N, 4)
        
        # Missing completion targets (zeros for now)
        missing_embeddings = torch.stack([item['missing_completion'] for item in batch])  # (B, N, embed_dim)
        
        return {
            'embeddings': embeddings,
            'vertebra_ids': vertebra_ids,
            'mask': mask,
            'subject_ids': subject_ids,
            'points': points,
            'features': features,
            'targets': {
                'ordering': vertebra_ids,
                'assembly': {
                    'translation': translations,
                    'rotation': rotations,
                },
                'missing_completion': missing_embeddings,
            },
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return dataloader

