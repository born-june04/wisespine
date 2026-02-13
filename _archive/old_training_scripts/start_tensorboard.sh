#!/bin/bash
#
# Start TensorBoard for monitoring training
# Usage: bash scripts/start_tensorboard.sh [experiment_dir]
#

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate conda environment
export LD_LIBRARY_PATH=/gscratch/ubicomp/june/miniconda3/envs/py311/lib:$LD_LIBRARY_PATH

# Find the most recent experiment directory
if [ -z "$1" ]; then
    EXPERIMENT_DIR=$(ls -td outputs/VerSe_COARSE_FINE_* 2>/dev/null | head -1)
    if [ -z "$EXPERIMENT_DIR" ]; then
        echo "Error: No experiment directory found"
        exit 1
    fi
else
    EXPERIMENT_DIR="$1"
fi

TENSORBOARD_DIR="$EXPERIMENT_DIR/tensorboard"

if [ ! -d "$TENSORBOARD_DIR" ]; then
    echo "Error: TensorBoard directory not found: $TENSORBOARD_DIR"
    exit 1
fi

echo "============================================================"
echo "Starting TensorBoard"
echo "============================================================"
echo "Experiment: $EXPERIMENT_DIR"
echo "TensorBoard logdir: $TENSORBOARD_DIR"
echo ""
echo "To access from local machine, use SSH port forwarding:"
echo "  ssh -L 6006:localhost:6006 g3120"
echo ""
echo "Then open in browser: http://localhost:6006"
echo "============================================================"

# Start TensorBoard
tensorboard --logdir "$TENSORBOARD_DIR" --port 6006 --host 0.0.0.0 --reload_interval 30

