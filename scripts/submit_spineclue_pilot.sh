#!/bin/bash
#SBATCH --job-name=spineclue_pilot
#SBATCH --partition=gpu-l40s
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/gscratch/scrubbed/june0604/vindr/outputs/spineclue_logs/slurm_%j.log
#SBATCH --error=/gscratch/scrubbed/june0604/vindr/outputs/spineclue_logs/slurm_%j.err

# SpineCLUE Pilot Test - SLURM Submission Script
# Usage: sbatch scripts/submit_spineclue_pilot.sh

echo "============================================================"
echo "SpineCLUE Pilot Test - SLURM Job"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo ""

# Setup environment
cd /gscratch/scrubbed/june0604/vindr
source /gscratch/ubicomp/june/miniconda3/etc/profile.d/conda.sh
conda activate py311

# Check GPU health
echo "Checking GPU health..."
nvidia-smi
nvidia-smi -q -d ECC | grep -A5 "Uncorrectable"

# Set environment variables (from run_spineclue.sh)
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/gscratch/ubicomp/june/miniconda3/envs/py311/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/gscratch/scrubbed/june0604/vindr:$PYTHONPATH

# Set Ultralytics home directory to avoid disk quota issues
export ULTRALYTICS_HOME=/gscratch/scrubbed/june0604/vindr/outputs/ultralytics_config
mkdir -p "$ULTRALYTICS_HOME"

# Run pilot test with localization stage
echo ""
echo "Starting localization training..."
echo "Using run_spineclue.sh style execution..."
cd /gscratch/scrubbed/june0604/vindr

bash scripts/run_spineclue.sh \
    --train_stage localization \
    --processed_dir VerSe/processed \
    --sample_fraction 0.1 \
    --localization_epochs 2 \
    --localization_batch_size 2 \
    --localization_lr 1e-4 \
    --use_pretrained_yolo 1 \
    --yolo_pretrained_model yolov8x \
    --yolo_custom_weights /gscratch/scrubbed/june0604/vindr/yolov8x.pt \
    --num_gpus 1 \
    --cuda_devices 0

echo ""
echo "============================================================"
echo "Training Complete"
echo "============================================================"
echo "End time: $(date)"

