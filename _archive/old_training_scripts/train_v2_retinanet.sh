#!/bin/bash
#
# SpineCLUE V2 Training - Using torchvision RetinaNet
# This is the WORKING version!
#

set -e

cd /gscratch/scrubbed/june0604/vindr

echo "============================================================"
echo "SpineCLUE V2 Localization Training"
echo "============================================================"
echo "Using torchvision RetinaNet (no Ultralytics)"
echo "Date: $(date)"
echo ""

# Environment
source /gscratch/ubicomp/june/miniconda3/etc/profile.d/conda.sh
conda activate py311

export PYTHONPATH=/gscratch/scrubbed/june0604/vindr:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# Run training
python3 << 'TRAIN_SCRIPT'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import sys

# Add project root
sys.path.insert(0, '/gscratch/scrubbed/june0604/vindr')

from workspace.networks.spineclue.localization_v2 import SpineCLUELocalizationV2
from workspace.dataset.spineclue_dataset import SpineCLUELocalizationDataset

# Config
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4
SAMPLE_FRACTION = 0.1
SAVE_DIR = 'outputs/spineclue_experiments/v2_retinanet'

os.makedirs(SAVE_DIR, exist_ok=True)

print("="*60)
print("Configuration")
print("="*60)
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LR}")
print(f"Sample fraction: {SAMPLE_FRACTION}")
print(f"Save dir: {SAVE_DIR}")
print()

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create dataset
print("\nLoading dataset...")
train_dataset = SpineCLUELocalizationDataset(
    processed_dir='VerSe/processed',
    split='train',
    sample_fraction=SAMPLE_FRACTION,
    planes=['axial'],  # Just axial for now
)
print(f"  Train samples: {len(train_dataset)}")

# Create dataloader with custom collate
def collate_fn(batch):
    """Custom collate for variable-length bboxes"""
    slices = torch.stack([b['slice'] for b in batch])
    bboxes = [b['bboxes'] for b in batch]
    planes = [b['plane'] for b in batch]
    return {'slice': slices, 'bboxes': bboxes, 'plane': planes}

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True,
)
print(f"  Batches per epoch: {len(train_loader)}")

# Create model
print("\nCreating model...")
model = SpineCLUELocalizationV2(
    num_classes=1,
    pretrained=True,
    score_thresh=0.3,
    nms_thresh=0.4,
)
model = model.to(device)
model.train()

# Optimizer
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Training loop
print("\n" + "="*60)
print("Starting Training")
print("="*60)

best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    epoch_cls_loss = 0.0
    epoch_box_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch_idx, batch in enumerate(pbar):
        # Get data
        slices = batch['slice'].to(device)  # (B, 1, 640, 640)
        bboxes_list = batch['bboxes']
        
        # Prepare targets for RetinaNet
        targets = model.prepare_targets(bboxes_list, device)
        
        # Forward
        optimizer.zero_grad()
        output = model(slices, targets)
        
        loss = output['loss']
        cls_loss = output['loss_classification']
        box_loss = output['loss_bbox_regression']
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track losses
        epoch_loss += loss.item()
        epoch_cls_loss += cls_loss.item()
        epoch_box_loss += box_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{cls_loss.item():.4f}',
            'box': f'{box_loss.item():.4f}',
        })
    
    # Epoch summary
    avg_loss = epoch_loss / len(train_loader)
    avg_cls = epoch_cls_loss / len(train_loader)
    avg_box = epoch_box_loss / len(train_loader)
    
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    
    print(f"\nEpoch {epoch+1}/{EPOCHS} Summary:")
    print(f"  Avg Loss: {avg_loss:.4f} (cls: {avg_cls:.4f}, box: {avg_box:.4f})")
    print(f"  LR: {current_lr:.6f}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        torch.save(checkpoint, f"{SAVE_DIR}/best_model.pt")
        print(f"  ✓ Saved best model (loss: {avg_loss:.4f})")

# Final save
checkpoint = {
    'epoch': EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
}
torch.save(checkpoint, f"{SAVE_DIR}/final_model.pt")

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"Best loss: {best_loss:.4f}")
print(f"Models saved to: {SAVE_DIR}")
TRAIN_SCRIPT

echo ""
echo "============================================================"
echo "Training Complete"
echo "============================================================"
echo "End time: $(date)"

