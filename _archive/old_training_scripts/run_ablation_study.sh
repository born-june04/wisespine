#!/bin/bash
#
# Ablation Study Script
# Runs multiple experiments with different module configurations
#
# Experiments:
# 1. Baseline (all modules off)
# 2. Anatomy only
# 3. Structure only
# 4. Physics only
# 5. Anatomy + Structure
# 6. Anatomy + Physics
# 7. Structure + Physics
# 8. All modules (Anatomy + Structure + Physics)
#

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Common settings (from run_coarse_fine.sh defaults)
BATCH_SIZE=4
LEARNING_RATE=1e-4
EPOCHS=1000
MODEL_SIZE="small"
USE_VRM=1
USE_BFLOAT16=1
NUM_WORKERS=5
TRAIN_STAGE1_ONLY=1

# Loss weights
ORDER_WEIGHT=1.0
LABEL_WEIGHT=1.0
DISTANCE_WEIGHT=1.0
MIN_DISTANCE=12.5
MAX_DISTANCE=125.0

# Module settings
CONSISTENCY_WEIGHT=1.0
NUM_REFINEMENT_ITERATIONS=2
USE_PHYSIOLOGICAL_CONSTRAINTS=1
PHYSICS_LAWS_PATH="physics/discovered_laws/discovered_spine_laws.json"
CONTRASTIVE_WEIGHT=1.0
CONTRASTIVE_TEMPERATURE=0.1

# GPU
NUM_GPUS=2
MASTER_ADDR="localhost"
MASTER_PORT=12355
export CUDA_VISIBLE_DEVICES="1,2"

echo "============================================================"
echo "Ablation Study - Module Comparison"
echo "============================================================"
echo "Model Size: $MODEL_SIZE"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "============================================================"
echo ""

# Function to run a single experiment
run_experiment() {
    local exp_name=$1
    local use_anatomy=$2
    local use_structure=$3
    local use_physics=$4
    
    echo "============================================================"
    echo "Running: $exp_name"
    echo "  Anatomy: $use_anatomy"
    echo "  Structure: $use_structure"
    echo "  Physics: $use_physics"
    echo "============================================================"
    
    echo "Using torchrun for multi-GPU training ($NUM_GPUS GPUs)..."
    torchrun \
        --nnode 1 \
        --nproc_per_node "$NUM_GPUS" \
        --master_addr "$MASTER_ADDR" \
        --master_port "$MASTER_PORT" \
        workspace/trainers/coarse_fine_trainer.py \
            --save_path "$PROJECT_ROOT/outputs" \
            --processed_dir "$PROJECT_ROOT/VerSe/processed" \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --epochs $EPOCHS \
            --model_size $MODEL_SIZE \
            --use_vrm $USE_VRM \
            --use_bfloat16 $USE_BFLOAT16 \
            --num_workers $NUM_WORKERS \
            --save_validation_details 1 \
            --order_weight $ORDER_WEIGHT \
            --label_weight $LABEL_WEIGHT \
            --distance_weight $DISTANCE_WEIGHT \
            --min_distance $MIN_DISTANCE \
            --max_distance $MAX_DISTANCE \
            --use_anatomy_embedding $use_anatomy \
            --consistency_weight $CONSISTENCY_WEIGHT \
            --use_structure_module $use_structure \
            --num_refinement_iterations $NUM_REFINEMENT_ITERATIONS \
            --use_physiological_constraints $USE_PHYSIOLOGICAL_CONSTRAINTS \
            --use_physics_informed $use_physics \
            --physics_laws_path "$PHYSICS_LAWS_PATH" \
            --contrastive_weight $CONTRASTIVE_WEIGHT \
            --contrastive_temperature $CONTRASTIVE_TEMPERATURE \
            --train_stage1_only $TRAIN_STAGE1_ONLY
    
    echo ""
    echo "✅ $exp_name complete!"
    echo ""
}

# Run all ablation experiments
echo "Starting ablation study..."
echo ""

# 1. Baseline (all modules off)
run_experiment "Baseline" 0 0 0

# 2. Anatomy only
run_experiment "Anatomy" 1 0 0

# 3. Structure only
run_experiment "Structure" 0 1 0

# 4. Physics only
run_experiment "Physics" 0 0 1

# 5. Anatomy + Structure
run_experiment "Anatomy+Structure" 1 1 0

# 6. Anatomy + Physics
run_experiment "Anatomy+Physics" 1 0 1

# 7. Structure + Physics
run_experiment "Structure+Physics" 0 1 1

# 8. All modules
run_experiment "All" 1 1 1

echo "============================================================"
echo "All ablation experiments complete!"
echo "============================================================"
