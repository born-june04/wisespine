#!/bin/bash
# ============================================================================
# Anatomy-Aware Module Ablation Experiments
# 
# MICCAI 2026 - SpineMedNeXt
# 
# 실험 구성:
# 1. Individual module ablation (각 모듈 개별 테스트)
# 2. Hybrid combinations (효과적인 조합 찾기)
# 3. Best model selection
# ============================================================================

set -e

# Project paths
PROJECT_ROOT="/gscratch/scrubbed/june0604/vindr"
SCRIPT_DIR="$PROJECT_ROOT/scripts"
OUTPUT_DIR="$PROJECT_ROOT/outputs"
LOG_DIR="$OUTPUT_DIR/logs/anatomy_ablation"

# Create directories
mkdir -p "$LOG_DIR"

# Conda environment
CONDA_ENV="py311"
CONDA_PATH="/gscratch/ubicomp/june/miniconda3"

# Fix library path
export LD_LIBRARY_PATH="$CONDA_PATH/envs/$CONDA_ENV/lib:$LD_LIBRARY_PATH"

# Default parameters
MODEL_SIZE="${MODEL_SIZE:-tiny}"
EPOCHS="${EPOCHS:-256}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
PROCESSED_DIR="$PROJECT_ROOT/VerSe/processed"

# Date for logging
DATE=$(date +%Y-%m-%d)

# ============================================================================
# Helper Functions
# ============================================================================

run_experiment() {
    local exp_name=$1
    local vrm_config=$2
    local gpu=$3
    
    local log_file="$LOG_DIR/${exp_name}_${DATE}.log"
    
    echo "=========================================="
    echo "Starting experiment: $exp_name"
    echo "VRM Config: $vrm_config"
    echo "GPU: $gpu"
    echo "Log: $log_file"
    echo "=========================================="
    
    # Run training
    CUDA_VISIBLE_DEVICES=$gpu python "$PROJECT_ROOT/workspace/trainers/enhanced_coarse_fine_trainer.py" \
        --model_size "$MODEL_SIZE" \
        --vrm_config "$vrm_config" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --processed_dir "$PROCESSED_DIR" \
        --exp_name "$exp_name" \
        2>&1 | tee -a "$log_file"
    
    echo "Experiment $exp_name completed!"
}

run_experiment_background() {
    local exp_name=$1
    local vrm_config=$2
    local gpu=$3
    
    local log_file="$LOG_DIR/${exp_name}_${DATE}.log"
    
    echo "Starting background experiment: $exp_name on GPU $gpu"
    echo "Log file: $log_file"
    
    nohup bash -c "
        source $CONDA_PATH/bin/activate $CONDA_ENV
        export LD_LIBRARY_PATH=$CONDA_PATH/envs/$CONDA_ENV/lib:\$LD_LIBRARY_PATH
        
        CUDA_VISIBLE_DEVICES=$gpu python $PROJECT_ROOT/workspace/trainers/enhanced_coarse_fine_trainer.py \
            --model_size $MODEL_SIZE \
            --vrm_config $vrm_config \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --processed_dir $PROCESSED_DIR \
            --exp_name $exp_name
    " > "$log_file" 2>&1 &
    
    echo "PID: $!"
}

# ============================================================================
# Experiment Configurations
# ============================================================================

# Phase 1: Individual Module Ablation
# - baseline: Original VRM (no new modules)
# - graph_only: Only GraphVertebraRelation
# - phys_only: Only PhysiologicalAttention
# - shape_only: Only AnatomicalShapePrior

# Phase 2: Pairwise Combinations
# - graph_phys: Graph + Physiological
# - graph_shape: Graph + Shape
# - phys_shape: Physiological + Shape

# Phase 3: Full Model
# - full: All modules enabled

# ============================================================================
# Main Execution
# ============================================================================

usage() {
    echo "Usage: $0 [phase|experiment] [gpu]"
    echo ""
    echo "Phases:"
    echo "  phase1    - Run individual module ablation (4 experiments)"
    echo "  phase2    - Run pairwise combinations (3 experiments)"
    echo "  phase3    - Run full model"
    echo "  all       - Run all experiments sequentially"
    echo ""
    echo "Individual experiments:"
    echo "  baseline, graph_only, phys_only, shape_only"
    echo "  graph_phys, graph_shape, phys_shape, full"
    echo ""
    echo "Options:"
    echo "  --parallel - Run experiments in parallel on multiple GPUs"
    echo ""
    echo "Examples:"
    echo "  $0 phase1 0                    # Run phase1 on GPU 0"
    echo "  $0 graph_only 1                # Run graph_only on GPU 1"
    echo "  $0 phase1 --parallel           # Run phase1 in parallel"
}

case "$1" in
    "phase1")
        if [ "$2" == "--parallel" ]; then
            echo "Running Phase 1 in parallel..."
            run_experiment_background "baseline" "baseline" 0
            run_experiment_background "graph_only" "graph_only" 1
            sleep 5
            echo "Waiting for GPU 0 and 1 to complete..."
            wait
            run_experiment_background "phys_only" "phys_only" 0
            run_experiment_background "shape_only" "shape_only" 1
            wait
        else
            GPU="${2:-0}"
            run_experiment "baseline" "baseline" $GPU
            run_experiment "graph_only" "graph_only" $GPU
            run_experiment "phys_only" "phys_only" $GPU
            run_experiment "shape_only" "shape_only" $GPU
        fi
        ;;
    
    "phase2")
        if [ "$2" == "--parallel" ]; then
            echo "Running Phase 2 in parallel..."
            run_experiment_background "graph_phys" "graph_phys" 0
            run_experiment_background "graph_shape" "graph_shape" 1
            wait
            run_experiment_background "phys_shape" "phys_shape" 0
            wait
        else
            GPU="${2:-0}"
            run_experiment "graph_phys" "graph_phys" $GPU
            run_experiment "graph_shape" "graph_shape" $GPU
            run_experiment "phys_shape" "phys_shape" $GPU
        fi
        ;;
    
    "phase3")
        GPU="${2:-0}"
        run_experiment "full" "full" $GPU
        ;;
    
    "all")
        if [ "$2" == "--parallel" ]; then
            echo "Running all phases in parallel..."
            # Phase 1
            run_experiment_background "baseline" "baseline" 0
            run_experiment_background "graph_only" "graph_only" 1
            wait
            run_experiment_background "phys_only" "phys_only" 0
            run_experiment_background "shape_only" "shape_only" 1
            wait
            # Phase 2
            run_experiment_background "graph_phys" "graph_phys" 0
            run_experiment_background "graph_shape" "graph_shape" 1
            wait
            run_experiment_background "phys_shape" "phys_shape" 0
            wait
            # Phase 3
            run_experiment "full" "full" 0
        else
            GPU="${2:-0}"
            # Phase 1
            run_experiment "baseline" "baseline" $GPU
            run_experiment "graph_only" "graph_only" $GPU
            run_experiment "phys_only" "phys_only" $GPU
            run_experiment "shape_only" "shape_only" $GPU
            # Phase 2
            run_experiment "graph_phys" "graph_phys" $GPU
            run_experiment "graph_shape" "graph_shape" $GPU
            run_experiment "phys_shape" "phys_shape" $GPU
            # Phase 3
            run_experiment "full" "full" $GPU
        fi
        ;;
    
    "baseline"|"graph_only"|"phys_only"|"shape_only"|"graph_phys"|"graph_shape"|"phys_shape"|"full")
        GPU="${2:-0}"
        run_experiment "$1" "$1" $GPU
        ;;
    
    "--help"|"-h"|"")
        usage
        ;;
    
    *)
        echo "Unknown option: $1"
        usage
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Experiment(s) completed!"
echo "Logs saved to: $LOG_DIR"
echo "============================================"

