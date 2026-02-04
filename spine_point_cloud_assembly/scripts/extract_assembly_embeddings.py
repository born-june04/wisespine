"""
Extract encoder embeddings for all vertebrae and save for assembly training.

This pre-computes embeddings to speed up assembly training.
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
import sys
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import SE3PointEncoder, features_to_irreps
from utils.data_loader import create_dataloader


def load_encoder(encoder_path: Path, device: torch.device, config: dict = None):
    """Load pretrained encoder"""
    print(f"Loading encoder from {encoder_path}")
    
    checkpoint = torch.load(encoder_path, map_location=device)
    
    # Get config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif config is None:
        config = {
            'input_dim': 8,
            'hidden_dim': 256,
            'num_layers': 4,
            'output_dim': 512,
            'use_curvature': True,
        }
    
    # Create encoder
    encoder = SE3PointEncoder(
        irreps_in="2x0e + 1x1o",
        irreps_hidden="32x0e + 16x1o + 8x2e",
        irreps_inv_out="64x0e",
        irreps_eq_out="8x1o",
        out_dim=config.get('output_dim', 512),
        num_layers=config.get('num_layers', 4),
        num_radial=16,
        lmax=2,
        cutoff=5.0,
        max_num_neighbors=32,
        use_curvature=config.get('use_curvature', True),
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['state_dict'])
    else:
        encoder.load_state_dict(checkpoint)
    
    encoder = encoder.to(device)
    encoder.eval()
    
    print(f"✓ Encoder loaded: {config}")
    return encoder, config


def extract_embeddings_for_subject(
    encoder: nn.Module,
    subject_dir: Path,
    device: torch.device,
    max_points: int = 2048,
    use_curvature: bool = True,
):
    """Extract embeddings for all vertebrae in a subject"""
    embeddings_dict = {}
    
    # Find all vertebra files
    vertebrae = []
    for pc_file in subject_dir.glob('vertebra_*_points.npy'):
        v_id = int(pc_file.stem.split('_')[1])
        vertebrae.append({
            'file': pc_file,
            'vertebra_id': v_id,
        })
    
    if len(vertebrae) == 0:
        return None
    
    # Sort by vertebra ID
    vertebrae = sorted(vertebrae, key=lambda x: x['vertebra_id'])
    
    encoder.eval()
    with torch.no_grad():
        for v_data in vertebrae:
            v_id = v_data['vertebra_id']
            pc_file = v_data['file']
            
            # Load point cloud
            points = np.load(pc_file).astype(np.float32)  # (M, 3)
            
            # Load features
            feature_file = pc_file.parent / f'vertebra_{v_id}_features.npz'
            normals_file = pc_file.parent / f'vertebra_{v_id}_normals.npy'
            curvature_file = pc_file.parent / f'vertebra_{v_id}_curvature.npz'
            
            # Load normals
            if feature_file.exists():
                features_data = np.load(feature_file)
                if 'normals' in features_data:
                    normals = features_data['normals'].astype(np.float32)
                elif normals_file.exists():
                    normals = np.load(normals_file).astype(np.float32)
                else:
                    normals = np.zeros((points.shape[0], 3), dtype=np.float32)
            elif normals_file.exists():
                normals = np.load(normals_file).astype(np.float32)
            else:
                normals = np.zeros((points.shape[0], 3), dtype=np.float32)
            
            # Load curvature
            if feature_file.exists():
                features_data = np.load(feature_file)
                if use_curvature and 'k1' in features_data and 'k2' in features_data:
                    k1 = features_data['k1'].astype(np.float32)
                    k2 = features_data['k2'].astype(np.float32)
                    curvature = np.stack([k1, k2], axis=-1)  # (N, 2)
                elif curvature_file.exists():
                    curvature_data = np.load(curvature_file)
                    if 'k1' in curvature_data and 'k2' in curvature_data:
                        k1 = curvature_data['k1'].astype(np.float32)
                        k2 = curvature_data['k2'].astype(np.float32)
                        curvature = np.stack([k1, k2], axis=-1)
                    else:
                        curvature = np.zeros((points.shape[0], 2), dtype=np.float32)
                else:
                    curvature = np.zeros((points.shape[0], 2), dtype=np.float32)
            elif curvature_file.exists():
                curvature_data = np.load(curvature_file)
                if 'k1' in curvature_data and 'k2' in curvature_data:
                    k1 = curvature_data['k1'].astype(np.float32)
                    k2 = curvature_data['k2'].astype(np.float32)
                    curvature = np.stack([k1, k2], axis=-1)
                else:
                    curvature = np.zeros((points.shape[0], 2), dtype=np.float32)
            else:
                curvature = np.zeros((points.shape[0], 2), dtype=np.float32)
            
            # Center points
            centroid = points.mean(axis=0)
            points = points - centroid
            
            # Subsample if needed
            if len(points) > max_points:
                indices = np.random.choice(len(points), max_points, replace=False)
                points = points[indices]
                normals = normals[indices]
                curvature = curvature[indices]
            elif len(points) < max_points:
                pad_size = max_points - len(points)
                points = np.pad(points, ((0, pad_size), (0, 0)), mode='constant')
                normals = np.pad(normals, ((0, pad_size), (0, 0)), mode='constant')
                curvature = np.pad(curvature, ((0, pad_size), (0, 0)), mode='constant')
            
            # Combine features
            features = np.concatenate([normals, curvature], axis=-1)  # (M, 5)
            
            # Convert to tensors
            points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(device)  # (1, M, 3)
            features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)  # (1, M, 5)
            batch_idx = torch.zeros(points_tensor.shape[0] * points_tensor.shape[1], 
                                   dtype=torch.long, device=device)
            
            # Encode
            feat_irreps = features_to_irreps(features_tensor, use_curvature=use_curvature)
            points_flat = points_tensor.view(-1, 3)
            
            output = encoder(points_flat, feat_irreps, batch=batch_idx)
            embedding = output['embedding'].squeeze(0).cpu().numpy()  # (embed_dim,)
            
            embeddings_dict[v_id] = {
                'embedding': embedding,
                'points': points,
                'features': features,
            }
    
    return embeddings_dict


def main():
    parser = argparse.ArgumentParser(description='Extract encoder embeddings for assembly training')
    parser.add_argument('--point_cloud_dir', type=str, required=True,
                        help='Directory containing point cloud data')
    parser.add_argument('--encoder_path', type=str, required=True,
                        help='Path to pretrained encoder checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for extracted embeddings')
    parser.add_argument('--max_points', type=int, default=2048,
                        help='Maximum number of points per vertebra')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    point_cloud_dir = Path(args.point_cloud_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Extracting Encoder Embeddings for Assembly Training")
    print("="*60)
    print(f"Point cloud directory: {point_cloud_dir}")
    print(f"Encoder path: {args.encoder_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print("="*60)
    print()
    
    # Load encoder
    encoder, encoder_config = load_encoder(Path(args.encoder_path), device)
    embed_dim = encoder_config.get('output_dim', 512)
    use_curvature = encoder_config.get('use_curvature', True)
    
    # Find all subjects
    subjects = [d for d in point_cloud_dir.iterdir() if d.is_dir()]
    print(f"Found {len(subjects)} subjects")
    print()
    
    # Extract embeddings for each subject
    metadata = {}
    for subject_dir in tqdm(subjects, desc="Extracting embeddings"):
        subject_id = subject_dir.name
        embeddings_dict = extract_embeddings_for_subject(
            encoder, subject_dir, device, args.max_points, use_curvature
        )
        
        if embeddings_dict is None:
            continue
        
        # Save embeddings for this subject
        subject_output_dir = output_dir / subject_id
        subject_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each vertebra embedding
        for v_id, data in embeddings_dict.items():
            np.savez(
                subject_output_dir / f'vertebra_{v_id}_embedding.npz',
                embedding=data['embedding'],
                points=data['points'],
                features=data['features'],
            )
        
        # Store metadata
        metadata[subject_id] = {
            'vertebra_ids': sorted(embeddings_dict.keys()),
            'num_vertebrae': len(embeddings_dict),
        }
    
    # Save metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print()
    print("="*60)
    print("Extraction Complete!")
    print("="*60)
    print(f"Extracted embeddings for {len(metadata)} subjects")
    print(f"Total vertebrae: {sum(m['num_vertebrae'] for m in metadata.values())}")
    print(f"Embeddings saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

