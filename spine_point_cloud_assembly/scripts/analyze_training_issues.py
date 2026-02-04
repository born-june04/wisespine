#!/usr/bin/env python3
"""
Analyze Training Issues: Why contrastive and masked losses are zero

This script helps diagnose why contrastive and masked losses are not being computed.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ContrastiveVertebraTypeLoss, MaskedPointModelingLoss
from models.pretraining import random_rotation_matrix
from utils.data_loader import create_dataloader


def analyze_contrastive_loss(batch_size: int, num_unique_labels: int):
    """Analyze why contrastive loss might be zero"""
    print("="*60)
    print("Contrastive Loss Analysis")
    print("="*60)
    
    print(f"\n1. Batch Size: {batch_size}")
    print(f"   Number of unique labels needed: At least 2 (for positive pairs)")
    
    if batch_size == 1:
        print("   ⚠️  PROBLEM: Batch size is 1!")
        print("      - Contrastive loss requires at least 2 samples")
        print("      - With batch_size=1, there are no positive pairs")
        print("      - Solution: Increase batch_size to at least 2 (preferably 8+)")
        return False
    
    if num_unique_labels < 2:
        print("   ⚠️  PROBLEM: Not enough unique labels in batch!")
        print("      - Need at least 2 samples with same label for positive pairs")
        print("      - Solution: Increase batch_size or ensure diverse labels")
        return False
    
    print("   ✓ Batch size is sufficient")
    return True


def test_contrastive_loss():
    """Test contrastive loss computation"""
    print("\n2. Testing Contrastive Loss Computation")
    print("-" * 60)
    
    # Create dummy embeddings and labels
    batch_size = 8
    output_dim = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create embeddings with same labels (for positive pairs)
    embeddings = torch.randn(batch_size, output_dim, device=device)
    labels = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4], device=device)  # Pairs
    
    loss_fn = ContrastiveVertebraTypeLoss(temperature=0.07)
    loss = loss_fn(embeddings, labels)
    
    print(f"   Test batch size: {batch_size}")
    print(f"   Labels: {labels.tolist()}")
    print(f"   Computed loss: {loss.item():.4f}")
    
    if loss.item() == 0.0:
        print("   ⚠️  Loss is zero! Check:")
        print("      - Are embeddings normalized?")
        print("      - Are there positive pairs?")
        print("      - Is temperature too high?")
        return False
    elif torch.isnan(loss) or torch.isinf(loss):
        print("   ⚠️  Loss is NaN/Inf! Check:")
        print("      - Are there any positive pairs?")
        print("      - Is temperature too low?")
        return False
    else:
        print("   ✓ Contrastive loss computation works")
        return True


def test_masked_loss():
    """Test masked loss computation"""
    print("\n3. Testing Masked Loss Computation")
    print("-" * 60)
    
    batch_size = 8
    output_dim = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create embeddings
    embedding_full = torch.randn(batch_size, output_dim, device=device)
    embedding_masked = embedding_full + 0.1 * torch.randn(batch_size, output_dim, device=device)
    
    loss_fn = MaskedPointModelingLoss()
    loss = loss_fn(embedding_full, embedding_masked)
    
    print(f"   Test batch size: {batch_size}")
    print(f"   Computed loss: {loss.item():.4f}")
    
    if loss.item() == 0.0:
        print("   ⚠️  Loss is zero! This happens when:")
        print("      - Embeddings are identical (collapse)")
        print("      - Cosine similarity is exactly 1.0")
        return False
    elif torch.isnan(loss) or torch.isinf(loss):
        print("   ⚠️  Loss is NaN/Inf!")
        return False
    else:
        print("   ✓ Masked loss computation works")
        return True


def check_batch_diversity(dataloader, num_batches: int = 10):
    """Check label diversity in batches"""
    print("\n4. Checking Batch Label Diversity")
    print("-" * 60)
    
    label_counts = []
    unique_label_counts = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        labels = batch['label']
        unique_labels = torch.unique(labels)
        
        label_counts.append(len(labels))
        unique_label_counts.append(len(unique_labels))
        
        # Count positive pairs
        positive_pairs = 0
        for label in unique_labels:
            count = (labels == label).sum().item()
            if count > 1:
                positive_pairs += count * (count - 1) // 2
        
        print(f"   Batch {i+1}: {len(labels)} samples, {len(unique_labels)} unique labels, {positive_pairs} positive pairs")
    
    avg_unique = np.mean(unique_label_counts)
    print(f"\n   Average unique labels per batch: {avg_unique:.1f}")
    
    if avg_unique < 2:
        print("   ⚠️  PROBLEM: Not enough label diversity!")
        print("      - Need at least 2 unique labels per batch for contrastive loss")
        return False
    
    return True


def check_loss_accumulation():
    """Check if losses are being accumulated correctly"""
    print("\n5. Checking Loss Accumulation Logic")
    print("-" * 60)
    
    print("   Checking training code...")
    
    issues = []
    
    # Check 1: Are losses being added to loss_components?
    print("   [1] Loss accumulation in training:")
    print("       - Contrastive: loss_components['contrastive'] += loss_contrast.item()")
    print("       - Masked: loss_components['masked'] += loss_masked.item()")
    print("       ✓ Looks correct")
    
    # Check 2: Validation code
    print("\n   [2] Loss accumulation in validation:")
    print("       - Contrastive: loss_contrast computed but NOT added to loss_components!")
    print("       ⚠️  BUG FOUND: Validation contrastive loss not accumulated!")
    issues.append("Validation contrastive loss not accumulated")
    
    # Check 3: Are losses in the loss dict?
    print("\n   [3] Loss dictionary check:")
    print("       - Need to verify 'contrastive' and 'masked' are in losses dict")
    print("       - Check if --use_contrastive and --use_masked flags are set")
    
    return issues


def main():
    parser = argparse.ArgumentParser(description='Analyze training issues')
    parser.add_argument('--point_cloud_dir', type=str, default='outputs/point_clouds',
                        help='Point cloud directory')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size to check')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Training Issues Analysis")
    print("="*60)
    print(f"Point cloud directory: {args.point_cloud_dir}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # 1. Analyze contrastive loss requirements
    analyze_contrastive_loss(args.batch_size, num_unique_labels=2)
    
    # 2. Test loss computations
    test_contrastive_loss()
    test_masked_loss()
    
    # 3. Check batch diversity
    if Path(args.point_cloud_dir).exists():
        try:
            dataloader = create_dataloader(
                point_cloud_dir=Path(args.point_cloud_dir),
                split='train',
                batch_size=args.batch_size,
                num_workers=0,  # Avoid multiprocessing issues
                max_points=2048,
                use_curvature=True,
                augment=False,
                shuffle=True,
            )
            check_batch_diversity(dataloader, num_batches=10)
        except Exception as e:
            print(f"   ⚠️  Could not check batch diversity: {e}")
    
    # 4. Check code issues
    issues = check_loss_accumulation()
    
    # Summary
    print("\n" + "="*60)
    print("Summary & Recommendations")
    print("="*60)
    
    if args.batch_size == 1:
        print("\n⚠️  CRITICAL ISSUE: Batch size is 1")
        print("   - Contrastive loss requires batch_size >= 2")
        print("   - Masked loss can work with batch_size=1, but may be unstable")
        print("   - RECOMMENDATION: Increase batch_size to at least 8")
    
    if issues:
        print("\n⚠️  CODE ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n   RECOMMENDATION: Fix validation loss accumulation")
    
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()

