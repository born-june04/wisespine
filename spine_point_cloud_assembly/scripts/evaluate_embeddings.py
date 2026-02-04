#!/usr/bin/env python3
"""
Evaluate Encoder Embedding Quality

1. t-SNE visualization (Vertebra type clustering)
2. Rotation invariance test
3. Embedding statistics
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import json
import sys
from collections import defaultdict
from contextlib import nullcontext

try:
    from torch.amp import autocast
except ImportError:
    # Fallback for older PyTorch versions
    from torch.cuda.amp import autocast

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import SE3PointEncoder, features_to_irreps
from utils.data_loader import create_dataloader
from models.pretraining import random_rotation_matrix


def load_model(model_path: Path, device: torch.device, config: dict = None):
    """Load pretrained encoder model"""
    print(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model config from checkpoint or use defaults
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif config is None:
        # Default config (should match training config)
        config = {
            'input_dim': 8,
            'hidden_dim': 256,
            'num_layers': 4,
            'output_dim': 512,
        }
    
    # Create model (SE3 encoder)
    model = SE3PointEncoder(
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
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'encoder_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['encoder_state_dict'])
    else:
        # Try direct loading
        try:
            model.load_state_dict(checkpoint)
        except:
            # If that fails, try to find model in checkpoint
            for key in ['model', 'encoder']:
                if key in checkpoint:
                    model.load_state_dict(checkpoint[key])
                    break
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded: {config}")
    return model, config


def extract_embeddings(model: nn.Module, dataloader, device: torch.device, max_samples: int = None, use_amp: bool = False):
    """Extract embeddings from all samples"""
    model.eval()
    
    # Mixed precision context
    amp_context = nullcontext()
    if use_amp and device.type == 'cuda':
        try:
            amp_context = autocast('cuda')
        except TypeError:
            amp_context = autocast()
    
    embeddings = []
    embeddings_raw = []  # Store raw embeddings for collapse check
    labels = []
    subject_ids = []
    vertebra_ids = []
    
    num_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            points = batch['points'].to(device)
            features = batch['features'].to(device)
            label = batch['label'].to(device)
            subject_id = batch['subject_id']
            
            # Forward pass with mixed precision
            # Convert features to IrrepsArray and flatten for batch processing
            B, N, _ = points.shape
            feat_irreps = features_to_irreps(features, use_curvature=True)
            points_flat = points.view(B * N, 3)
            batch_idx = torch.arange(B, device=device).repeat_interleave(N)
            
            with amp_context:
                output = model(points_flat, feat_irreps, batch=batch_idx)
                embedding_raw = output['embedding']  # (B, output_dim)
                # Store raw embedding for collapse check
                embeddings_raw.append(embedding_raw.cpu().numpy())
                # Normalize embeddings to prevent overflow in statistics
                embedding = F.normalize(embedding_raw, p=2, dim=1)
            
            embeddings.append(embedding.cpu().numpy())
            labels.append(label.cpu().numpy())
            subject_ids.extend(subject_id)
            vertebra_ids.extend(label.cpu().numpy().tolist())
            
            num_samples += len(embedding)
            if max_samples and num_samples >= max_samples:
                break
    
    embeddings = np.concatenate(embeddings, axis=0)
    embeddings_raw = np.concatenate(embeddings_raw, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    print(f"✓ Extracted {len(embeddings)} embeddings")
    return embeddings, labels, subject_ids, vertebra_ids, embeddings_raw


def compute_embedding_stats(embeddings: np.ndarray, labels: np.ndarray, output_dir: Path, embeddings_raw: np.ndarray = None):
    """Compute and save embedding statistics"""
    print("\n" + "="*60)
    print("Embedding Statistics")
    print("="*60)
    
    stats = {
        'num_samples': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'num_vertebra_types': len(np.unique(labels)),
        'mean_norm': float(np.linalg.norm(embeddings, axis=1).mean()),
        'std_norm': float(np.linalg.norm(embeddings, axis=1).std()),
        'mean_value': float(embeddings.mean()),
        'std_value': float(embeddings.std()),
        'min_value': float(embeddings.min()),
        'max_value': float(embeddings.max()),
    }
    
    # Per-vertebra-type statistics
    vertebra_stats = {}
    for v_id in np.unique(labels):
        v_mask = labels == v_id
        v_embeddings = embeddings[v_mask]
        
        vertebra_stats[int(v_id)] = {
            'count': int(v_mask.sum()),
            'mean_norm': float(np.linalg.norm(v_embeddings, axis=1).mean()),
            'std_norm': float(np.linalg.norm(v_embeddings, axis=1).std()),
            'mean_embedding': [float(x) for x in v_embeddings.mean(axis=0).tolist()],  # Ensure all floats
        }
    
    stats['vertebra_stats'] = vertebra_stats
    
    # Check for embedding collapse
    # Use raw embeddings if available (before normalization), otherwise use normalized
    if embeddings_raw is not None:
        # Convert to float64 to prevent overflow
        embeddings_raw_safe = embeddings_raw.astype(np.float64)
        embedding_norms = np.linalg.norm(embeddings_raw_safe, axis=1)
        # Also check embedding value variance (more robust)
        embedding_std = embeddings_raw_safe.std(axis=0).mean()  # Mean of per-dimension std
        # Clip very large values for safe statistics
        embedding_norms = np.clip(embedding_norms, 0, 1e10)
    else:
        # If normalized, check embedding value variance instead of norm
    embedding_norms = np.linalg.norm(embeddings, axis=1)
        embedding_std = embeddings.std(axis=0).mean()  # Mean of per-dimension std
    
    norm_std = embedding_norms.std()
    # Collapse check: low norm variance OR low embedding value variance
    # For normalized embeddings, norm_std will be ~0, so use embedding_std
    # Use normalized embedding std for collapse check (more reliable)
    normalized_embedding_std = embeddings.std(axis=0).mean()
    is_collapsed = bool(normalized_embedding_std < 0.01)
    stats['collapse_check'] = {
        'norm_std': float(norm_std) if not np.isinf(norm_std) and not np.isnan(norm_std) else 0.0,
        'embedding_std': float(embedding_std) if not np.isinf(embedding_std) and not np.isnan(embedding_std) else 0.0,
        'normalized_embedding_std': float(normalized_embedding_std),
        'is_collapsed': is_collapsed,
    }
    
    # Print statistics
    print(f"Number of samples: {stats['num_samples']}")
    print(f"Embedding dimension: {stats['embedding_dim']}")
    print(f"Number of vertebra types: {stats['num_vertebra_types']}")
    print(f"\nEmbedding norms:")
    print(f"  Mean: {stats['mean_norm']:.4f}")
    print(f"  Std: {stats['std_norm']:.4f}")
    print(f"\nEmbedding values:")
    print(f"  Mean: {stats['mean_value']:.4f}")
    print(f"  Std: {stats['std_value']:.4f}")
    print(f"  Range: [{stats['min_value']:.4f}, {stats['max_value']:.4f}]")
    print(f"\nCollapse check:")
    if embeddings_raw is not None:
        norm_std_display = norm_std if not np.isinf(norm_std) and not np.isnan(norm_std) else float('inf')
        embedding_std_display = embedding_std if not np.isinf(embedding_std) and not np.isnan(embedding_std) else float('inf')
        print(f"  Raw embedding norm std: {norm_std_display:.4f}" if not np.isinf(norm_std_display) else f"  Raw embedding norm std: inf (very large values)")
        print(f"  Raw embedding value std (mean per-dim): {embedding_std_display:.4f}" if not np.isinf(embedding_std_display) else f"  Raw embedding value std: inf (very large values)")
    print(f"  Normalized embedding value std: {normalized_embedding_std:.4f}")
    print(f"  Is collapsed: {is_collapsed}")
    if is_collapsed:
        print(f"  ⚠️  WARNING: Embedding collapse detected!")
        print(f"     Normalized embedding std: {normalized_embedding_std:.4f} (very low!)")
        print(f"     This suggests all embeddings are very similar. Consider:")
        print(f"     - Checking if model is properly trained")
        print(f"     - Adjusting loss weights")
        print(f"     - Using different pretraining tasks")
    elif embeddings_raw is not None and (np.isinf(norm_std) or np.isinf(embedding_std)):
        print(f"  ℹ️  Note: Raw embedding values are very large (causing overflow), but normalized embeddings are healthy.")
    
    # Save statistics
    stats_file = output_dir / 'embedding_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ Statistics saved to {stats_file}")
    
    return stats


def visualize_tsne(embeddings: np.ndarray, labels: np.ndarray, output_dir: Path, 
                   vertebra_ids: list = None, max_samples: int = 5000):
    """Visualize embeddings using t-SNE"""
    print("\n" + "="*60)
    print("t-SNE Visualization")
    print("="*60)
    
    # Subsample if too many points
    if len(embeddings) > max_samples:
        print(f"Subsampling {len(embeddings)} embeddings to {max_samples} for t-SNE...")
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings_subset = embeddings[indices]
        labels_subset = labels[indices]
    else:
        embeddings_subset = embeddings
        labels_subset = labels
    
    # Reduce dimensionality with PCA first (faster)
    print("Applying PCA...")
    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(embeddings_subset)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Apply t-SNE
    print("Applying t-SNE (this may take a while)...")
    # Use max_iter instead of n_iter (scikit-learn >= 0.23)
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    except TypeError:
        # Fallback for older scikit-learn versions
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings_pca)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get unique labels and colors
    unique_labels = np.unique(labels_subset)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each vertebra type
    for i, v_id in enumerate(unique_labels):
        mask = labels_subset == v_id
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                  c=[colors[i]], label=f'V{v_id}', alpha=0.6, s=20)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('t-SNE Visualization of Vertebra Embeddings\n(Clustering by Vertebra Type)', 
                 fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / 'tsne_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ t-SNE visualization saved to {output_file}")
    
    # Also create region-based visualization (C/T/L)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Map vertebra IDs to regions
    def get_region(v_id):
        if 1 <= v_id <= 7:
            return 'Cervical', 'C'
        elif 8 <= v_id <= 19:
            return 'Thoracic', 'T'
        elif 20 <= v_id <= 24:
            return 'Lumbar', 'L'
        else:
            return 'Other', 'O'
    
    region_colors = {'C': 'red', 'T': 'blue', 'L': 'green', 'O': 'gray'}
    for v_id in unique_labels:
        mask = labels_subset == v_id
        region_name, region_code = get_region(v_id)
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                  c=region_colors[region_code], label=f'{region_name} (V{v_id})',
                  alpha=0.6, s=20)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('t-SNE Visualization by Region\n(Cervical/Thoracic/Lumbar)', 
                 fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file_region = output_dir / 'tsne_visualization_by_region.png'
    plt.savefig(output_file_region, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Region-based t-SNE visualization saved to {output_file_region}")


def test_rotation_invariance(model: nn.Module, dataloader, device: torch.device, 
                            num_samples: int = 100, num_rotations: int = 10, use_amp: bool = False):
    """Test rotation invariance of embeddings"""
    print("\n" + "="*60)
    print("Rotation Invariance Test")
    print("="*60)
    
    model.eval()
    
    similarities = []
    angle_errors = []
    
    sample_count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing rotation invariance"):
            if sample_count >= num_samples:
                break
            
            points = batch['points'].to(device)
            features = batch['features'].to(device)
            
            # Mixed precision context
            amp_context = nullcontext()
            if use_amp and device.type == 'cuda':
                try:
                    amp_context = autocast('cuda')
                except TypeError:
                    amp_context = autocast()
            
            # Original embedding
            B, N, _ = points.shape
            feat_irreps = features_to_irreps(features, use_curvature=True)
            points_flat = points.view(B * N, 3)
            batch_idx = torch.arange(B, device=device).repeat_interleave(N)
            
            with amp_context:
                output = model(points_flat, feat_irreps, batch=batch_idx)
                embedding_original = output['embedding']  # (B, output_dim)
                # Normalize for consistent comparison
                embedding_original = F.normalize(embedding_original, p=2, dim=1)
            
            for b in range(B):
                if sample_count >= num_samples:
                    break
                
                # Original embedding for this sample
                emb_orig = embedding_original[b:b+1]  # (1, output_dim)
                
                # Test multiple rotations
                for rot_idx in range(num_rotations):
                    # Generate random rotation
                    R = random_rotation_matrix(1, device)  # (1, 3, 3)
                    
                    # Rotate points
                    points_rot = torch.bmm(points[b:b+1], R)  # (1, N, 3)
                    
                    # Rotate normals
                    if features.shape[-1] >= 3:
                        normals = features[b:b+1, :, :3]  # (1, N, 3)
                        normals_rot = torch.bmm(normals, R)
                        features_rot = features[b:b+1].clone()
                        features_rot[:, :, :3] = normals_rot
                    else:
                        features_rot = features[b:b+1]
                    
                    # Rotated embedding
                    feat_rot_irreps = features_to_irreps(features_rot, use_curvature=True)
                    points_rot_flat = points_rot.view(-1, 3)
                    batch_idx_rot = torch.zeros(points_rot_flat.shape[0], dtype=torch.long, device=device)
                    
                    with amp_context:
                        output_rot = model(points_rot_flat, feat_rot_irreps, batch=batch_idx_rot)
                        embedding_rotated = output_rot['embedding']  # (1, output_dim)
                        # Normalize for consistent comparison
                        embedding_rotated = F.normalize(embedding_rotated, p=2, dim=1)
                    
                    # Compute cosine similarity
                    cos_sim = torch.nn.functional.cosine_similarity(
                        emb_orig, embedding_rotated, dim=1
                    ).item()
                    similarities.append(cos_sim)
                    
                    # Compute rotation angle error (if we had ground truth)
                    # For now, we just check similarity
                
                sample_count += 1
    
    similarities = np.array(similarities)
    
    print(f"\nRotation Invariance Results:")
    print(f"  Number of tests: {len(similarities)}")
    print(f"  Mean cosine similarity: {similarities.mean():.4f}")
    print(f"  Std cosine similarity: {similarities.std():.4f}")
    print(f"  Min cosine similarity: {similarities.min():.4f}")
    print(f"  Max cosine similarity: {similarities.max():.4f}")
    
    # Good rotation invariance: similarity > 0.9
    high_similarity_ratio = (similarities > 0.9).mean()
    print(f"\n  Similarity > 0.9: {high_similarity_ratio:.2%}")
    
    if high_similarity_ratio > 0.8:
        print("  ✓ Good rotation invariance!")
    elif high_similarity_ratio > 0.6:
        print("  ⚠ Moderate rotation invariance")
    else:
        print("  ✗ Poor rotation invariance")
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(similarities, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(similarities.mean(), color='red', linestyle='--', 
               label=f'Mean: {similarities.mean():.4f}')
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Rotation Invariance: Cosine Similarity Distribution', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return similarities


def main():
    parser = argparse.ArgumentParser(description='Evaluate encoder embedding quality')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--point_cloud_dir', type=str,
                        default='outputs/point_clouds',
                        help='Directory containing point clouds')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/embeddings/evaluation',
                        help='Output directory for evaluation results')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference (default: 1 to match training)')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision (float16) for inference to reduce memory usage')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (None = all)')
    parser.add_argument('--tsne_samples', type=int, default=5000,
                        help='Maximum samples for t-SNE (for speed)')
    parser.add_argument('--rotation_test_samples', type=int, default=100,
                        help='Number of samples for rotation invariance test')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Encoder Embedding Quality Evaluation")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Point cloud directory: {args.point_cloud_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Mixed precision (AMP): {args.use_amp}")
    if args.use_amp:
        print("  ✅ Using float16 for inference (reduces memory usage by ~50%)")
    print()
    
    # Load model
    model, config = load_model(Path(args.model_path), device)
    
    # Create dataloader (no augmentation for evaluation)
    dataloader = create_dataloader(
        point_cloud_dir=Path(args.point_cloud_dir),
        split='train',  # Use all data
        batch_size=args.batch_size,
        num_workers=4,
        max_points=2048,
        use_curvature=True,
        augment=False,  # No augmentation for evaluation
        shuffle=False,
    )
    
    # 1. Extract embeddings
    embeddings, labels, subject_ids, vertebra_ids, embeddings_raw = extract_embeddings(
        model, dataloader, device, max_samples=args.max_samples, use_amp=args.use_amp
    )
    
    # 2. Compute statistics
    stats = compute_embedding_stats(embeddings, labels, output_dir, embeddings_raw=embeddings_raw)
    
    # 3. t-SNE visualization
    visualize_tsne(embeddings, labels, output_dir, vertebra_ids, 
                  max_samples=args.tsne_samples)
    
    # 4. Rotation invariance test
    similarities = test_rotation_invariance(
        model, dataloader, device,
        num_samples=args.rotation_test_samples,
        num_rotations=10,
        use_amp=args.use_amp
    )
    
    # Save rotation invariance results
    rotation_stats = {
        'mean_similarity': float(similarities.mean()),
        'std_similarity': float(similarities.std()),
        'min_similarity': float(similarities.min()),
        'max_similarity': float(similarities.max()),
        'high_similarity_ratio': float((similarities > 0.9).mean()),
    }
    
    rotation_file = output_dir / 'rotation_invariance_stats.json'
    with open(rotation_file, 'w') as f:
        json.dump(rotation_stats, f, indent=2)
    
    # Save histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(similarities, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(similarities.mean(), color='red', linestyle='--', 
               label=f'Mean: {similarities.mean():.4f}')
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Rotation Invariance: Cosine Similarity Distribution', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'rotation_invariance_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Rotation invariance results saved to {rotation_file}")
    
    # Summary
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"\nSummary:")
    print(f"  - Embedding dimension: {stats['embedding_dim']}")
    print(f"  - Number of samples: {stats['num_samples']}")
    print(f"  - Number of vertebra types: {stats['num_vertebra_types']}")
    print(f"  - Embedding collapse: {'⚠️  YES (WARNING!)' if stats['collapse_check']['is_collapsed'] else '✓ No'}")
    if stats['collapse_check']['is_collapsed']:
        print(f"    Norm std: {stats['collapse_check']['norm_std']:.6f} (very low!)")
    print(f"  - Rotation invariance (mean similarity): {rotation_stats['mean_similarity']:.4f}")
    print(f"  - High similarity ratio (>0.9): {rotation_stats['high_similarity_ratio']:.2%}")
    
    # Final assessment
    print(f"\n{'='*60}")
    print("Assessment:")
    print(f"{'='*60}")
    if stats['collapse_check']['is_collapsed']:
        print("⚠️  CRITICAL: Embedding collapse detected!")
        print("   The model may not be learning meaningful representations.")
        print("   Recommendations:")
        print("   1. Check training loss - is it decreasing?")
        print("   2. Verify loss weights are balanced")
        print("   3. Try enabling contrastive or masked tasks")
        print("   4. Check if learning rate is too high")
    else:
        print("✓ Embeddings appear healthy (no collapse detected)")
    
    if rotation_stats['mean_similarity'] > 0.9:
        print("✓ Excellent rotation invariance")
    elif rotation_stats['mean_similarity'] > 0.8:
        print("⚠️  Moderate rotation invariance")
    else:
        print("⚠️  Poor rotation invariance - model may not be rotation-equivariant")


if __name__ == '__main__':
    main()

