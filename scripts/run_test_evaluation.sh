#!/bin/bash
#
# Test Set Evaluation Script for SpineMedNeXt
# MICCAI 2026 Research
#

set -e

# Paths
PROJECT_ROOT="/gscratch/scrubbed/june0604/vindr"
PROCESSED_DIR="$PROJECT_ROOT/VerSe/processed"
OUTPUT_DIR="$PROJECT_ROOT/outputs/test_results"

# Conda setup
source /gscratch/ubicomp/june/miniconda3/bin/activate py311
export LD_LIBRARY_PATH=/gscratch/ubicomp/june/miniconda3/envs/py311/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="1"

cd $PROJECT_ROOT

# Create output directory
mkdir -p $OUTPUT_DIR

echo "============================================================"
echo "SpineMedNeXt Test Set Evaluation"
echo "============================================================"
echo ""

# Define experiments to evaluate
# Format: experiment_dir,model_size,use_vrm,name
declare -a EXPERIMENTS=(
    # Best models from validation
    # "VerSe_COARSE_FINE_12-05_13-22,small,1,Small_VRM1",
    # "VerSe_COARSE_FINE_12-05_20-23,small,1,Small_VRM1_anatomy"
    "VerSe_COARSE_FINE_12-05_23-31,small,1,Small_VRM1_structure"
    "VerSe_COARSE_FINE_12-05_23-35,small,1,Small_VRM1_physics"
)

# Evaluate each experiment
for exp in "${EXPERIMENTS[@]}"; do
    IFS=',' read -r exp_dir model_size use_vrm name <<< "$exp"
    
    exp_path="$PROJECT_ROOT/outputs/$exp_dir"
    
    # Check if experiment exists
    if [ ! -d "$exp_path" ]; then
        echo "⚠️  Skipping $name: Directory not found ($exp_path)"
        continue
    fi
    
    # Check if models exist
    stage1_exists=false
    stage2_exists=false
    
    if [ -f "$exp_path/stage1/best_model.pth" ]; then
        stage1_exists=true
    fi
    
    if [ -f "$exp_path/stage2/best_model.pth" ]; then
        stage2_exists=true
    fi
    
    if [ "$stage1_exists" = false ]; then
        echo "⚠️  Skipping $name: Stage 1 checkpoint not found ($exp_path/stage1/best_model.pth)"
        continue
    fi
    
    if [ "$stage2_exists" = false ]; then
        echo "⚠️  Warning: Stage 2 checkpoint not found, will evaluate Stage 1 only"
    fi
    
    echo ""
    echo "============================================================"
    echo "Evaluating: $name"
    echo "  Model Size: $model_size"
    echo "  VRM: $use_vrm"
    echo "  Experiment: $exp_dir"
    echo "============================================================"
    
    python workspace/evaluation/evaluate_test_set.py \
        --experiment_dir "$exp_path" \
        --processed_dir "$PROCESSED_DIR" \
        --model_size "$model_size" \
        --use_vrm "$use_vrm" \
        --batch_size 4 \
        --num_workers 4 \
        --output_dir "$OUTPUT_DIR/$name" \
        --save_visualizations
    
    echo ""
    echo "✅ $name evaluation complete!"
done

echo ""
echo "============================================================"
echo "All evaluations complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================"

