#!/bin/bash
#
# SpineCLUE V2 Training Script
# Uses torchvision RetinaNet for localization (stable, working!)
#
# Usage:
#   bash scripts/run_spineclue_v2.sh [OPTIONS]
#
# Examples:
#   # Single GPU pilot test
#   bash scripts/run_spineclue_v2.sh --sample_fraction 0.1 --epochs 10
#
#   # Multi-GPU full training
#   bash scripts/run_spineclue_v2.sh --num_gpus 4 --epochs 100 --batch_size 16
#
# Pipeline:
#   Stage 1: Localization - RetinaNet (2D slices → vertebra detection)
#   Stage 2: Segmentation - TransUNet (3D ROI → vertebra mask)
#   Stage 3: Identification - 3D ResNet + Contrastive Learning

set -e

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

# Default parameters
BATCH_SIZE=32
LEARNING_RATE=1e-4
EPOCHS=128
NUM_WORKERS=4
NUM_GPUS=2
SAMPLE_FRACTION=1.0

# Stage selection
TRAIN_STAGE="localization"  # 'localization', 'segmentation', 'identification', 'all'

# Localization (RetinaNet)
LOCALIZATION_EPOCHS=100
LOCALIZATION_LR=1e-4
LOCALIZATION_BATCH_SIZE=4
SCORE_THRESH=0.3
NMS_THRESH=0.4

# Data paths
PROCESSED_DIR="VerSe/processed"
SAVE_DIR="outputs/spineclue_experiments"

# GPU settings
MASTER_ADDR="localhost"
MASTER_PORT=12356
export CUDA_VISIBLE_DEVICES="0,1"

# Experiment name
EXPERIMENT="spineclue_v2"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

# ============================================================================
# Parse command line arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            LOCALIZATION_BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            LOCALIZATION_EPOCHS="$2"
            shift 2
            ;;
        --learning_rate|--lr)
            LEARNING_RATE="$2"
            LOCALIZATION_LR="$2"
            shift 2
            ;;
        --train_stage)
            TRAIN_STAGE="$2"
            shift 2
            ;;
        --processed_dir)
            PROCESSED_DIR="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --sample_fraction)
            SAMPLE_FRACTION="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --cuda_devices)
            export CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --score_thresh)
            SCORE_THRESH="$2"
            shift 2
            ;;
        --nms_thresh)
            NMS_THRESH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Setup save directory
FULL_SAVE_DIR="${SAVE_DIR}/${EXPERIMENT}_${TIMESTAMP}"
mkdir -p "$FULL_SAVE_DIR"

# Setup logging
LOG_DIR="$PROJECT_ROOT/outputs/spineclue_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${EXPERIMENT}_${TIMESTAMP}.log"

# ============================================================================
# Logging
# ============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_section() {
    echo "" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo "$*" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
}

# ============================================================================
# Print Configuration
# ============================================================================

log_section "SpineCLUE V2 Training Configuration"
log "Experiment: $EXPERIMENT"
log "Timestamp: $TIMESTAMP"
log "Save directory: $FULL_SAVE_DIR"
log "Log file: $LOG_FILE"
log ""
log "Stage: $TRAIN_STAGE"
log "Model: RetinaNet (torchvision)"
log ""
log "Training parameters:"
log "  Batch size: $BATCH_SIZE"
log "  Epochs: $EPOCHS"
log "  Learning rate: $LEARNING_RATE"
log "  Sample fraction: $SAMPLE_FRACTION"
log ""
log "GPU settings:"
log "  Number of GPUs: $NUM_GPUS"
log "  CUDA devices: $CUDA_VISIBLE_DEVICES"
log "  Master port: $MASTER_PORT"
log ""
log "Detection settings:"
log "  Score threshold: $SCORE_THRESH"
log "  NMS threshold: $NMS_THRESH"

# ============================================================================
# Check Dependencies
# ============================================================================

log_section "Checking Dependencies"

log "Python: $(python3 --version)"
log "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"

CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" = "True" ]; then
    CUDA_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
    log "CUDA: Available ($CUDA_COUNT devices)"
else
    log "CUDA: Not available"
fi

log "Data directory: $PROCESSED_DIR"

# ============================================================================
# Environment Setup
# ============================================================================

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# ============================================================================
# Training Script
# ============================================================================

TRAIN_SCRIPT=$(cat << 'PYTHON_SCRIPT'
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from pathlib import Path

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--sample_fraction', type=float, default=1.0)
parser.add_argument('--processed_dir', type=str, default='VerSe/processed')
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--score_thresh', type=float, default=0.3)
parser.add_argument('--nms_thresh', type=float, default=0.4)
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()

# DDP setup
distributed = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
if distributed:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
else:
    local_rank = 0
    world_size = 1
    rank = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

is_main = rank == 0

if is_main:
    print("="*60)
    print("SpineCLUE V2 - RetinaNet Localization Training")
    print("="*60)
    print(f"Distributed: {distributed}")
    if distributed:
        print(f"World size: {world_size}")
        print(f"Rank: {rank}")
    print(f"Device: {device}")
    print()

# Import our modules
sys.path.insert(0, os.environ.get('PROJECT_ROOT', '.'))
from workspace.networks.spineclue.localization_v2 import SpineCLUELocalizationV2
from workspace.dataset.spineclue_dataset import SpineCLUELocalizationDataset

# Create dataset
if is_main:
    print("Loading dataset...")

train_dataset = SpineCLUELocalizationDataset(
    processed_dir=args.processed_dir,
    split='train',
    sample_fraction=args.sample_fraction,
    planes=['axial'],
)

if is_main:
    print(f"  Train samples: {len(train_dataset)}")

# Collate function
def collate_fn(batch):
    slices = torch.stack([b['slice'] for b in batch])
    bboxes = [b['bboxes'] for b in batch]
    planes = [b['plane'] for b in batch]
    return {'slice': slices, 'bboxes': bboxes, 'plane': planes}

# Sampler for DDP
if distributed:
    sampler = DistributedSampler(train_dataset, shuffle=True)
else:
    sampler = None

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=(sampler is None),
    sampler=sampler,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
    drop_last=True,
)

if is_main:
    print(f"  Batches per epoch: {len(train_loader)}")

# Create model
if is_main:
    print("\nCreating RetinaNet model...")

model = SpineCLUELocalizationV2(
    num_classes=1,
    pretrained=True,
    score_thresh=args.score_thresh,
    nms_thresh=args.nms_thresh,
)
model = model.to(device)

# Wrap with DDP
if distributed:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model_to_use = model.module
else:
    model_to_use = model

# Optimizer
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

# Save directory
save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

# Training loop
if is_main:
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

best_loss = float('inf')

for epoch in range(args.epochs):
    model.train()
    
    if distributed:
        sampler.set_epoch(epoch)
    
    epoch_loss = 0.0
    epoch_cls_loss = 0.0
    epoch_box_loss = 0.0
    
    # Show progress bar for all ranks
    if distributed:
        desc = f"Epoch {epoch+1}/{args.epochs} [Rank {rank}/{world_size-1}]"
    else:
        desc = f"Epoch {epoch+1}/{args.epochs}"
    
    pbar = tqdm(train_loader, desc=desc, disable=False)
    
    for batch_idx, batch in enumerate(pbar):
        slices = batch['slice'].to(device)
        bboxes_list = batch['bboxes']
        
        # Prepare targets
        targets = model_to_use.prepare_targets(bboxes_list, device)
        
        # Forward
        optimizer.zero_grad()
        output = model(slices, targets)
        
        loss = output['loss']
        cls_loss = output['loss_classification']
        box_loss = output['loss_bbox_regression']
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track
        epoch_loss += loss.item()
        epoch_cls_loss += cls_loss.item()
        epoch_box_loss += box_loss.item()
        
        # Update progress bar for all ranks
        pbar.set_postfix({
            'rank': rank,
            'loss': f'{loss.item():.4f}',
            'cls': f'{cls_loss.item():.4f}',
            'box': f'{box_loss.item():.4f}',
        })
    
    # Epoch summary
    avg_loss = epoch_loss / len(train_loader)
    avg_cls = epoch_cls_loss / len(train_loader)
    avg_box = epoch_box_loss / len(train_loader)
    
    scheduler.step()
    
    if is_main:
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f} (cls: {avg_cls:.4f}, box: {avg_box:.4f})")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model_to_use.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, save_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (loss: {avg_loss:.4f})")

# Final save
if is_main:
    checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model_to_use.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }
    torch.save(checkpoint, save_dir / 'final_model.pt')
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best loss: {best_loss:.4f}")
    print(f"Models saved to: {save_dir}")

# Cleanup
if distributed:
    dist.destroy_process_group()
PYTHON_SCRIPT
)

# ============================================================================
# Run Training
# ============================================================================

log_section "Starting Training"

# Write training script to temp file
TRAIN_SCRIPT_FILE=$(mktemp /tmp/spineclue_v2_train_XXXXXX.py)
echo "$TRAIN_SCRIPT" > "$TRAIN_SCRIPT_FILE"

# Build arguments
TRAIN_ARGS=(
    --batch_size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --lr "$LEARNING_RATE"
    --sample_fraction "$SAMPLE_FRACTION"
    --processed_dir "$PROCESSED_DIR"
    --save_dir "$FULL_SAVE_DIR"
    --num_workers "$NUM_WORKERS"
    --score_thresh "$SCORE_THRESH"
    --nms_thresh "$NMS_THRESH"
)

if [ "$NUM_GPUS" -gt 1 ]; then
    log "Using multi-GPU training with torchrun ($NUM_GPUS GPUs)"
    
    torchrun \
        --nproc_per_node "$NUM_GPUS" \
        --master_addr "$MASTER_ADDR" \
        --master_port "$MASTER_PORT" \
        "$TRAIN_SCRIPT_FILE" \
        "${TRAIN_ARGS[@]}" \
        2>&1 | tee -a "$LOG_FILE"
else
    log "Using single-GPU training"
    
    python3 "$TRAIN_SCRIPT_FILE" \
        "${TRAIN_ARGS[@]}" \
        2>&1 | tee -a "$LOG_FILE"
fi

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

# Cleanup
rm -f "$TRAIN_SCRIPT_FILE"

# ============================================================================
# Summary
# ============================================================================

log_section "Training Complete"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    log "✓ Training completed successfully!"
    log "Results saved to: $FULL_SAVE_DIR"
    log "Log file: $LOG_FILE"
else
    log "✗ Training failed with exit code: $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

