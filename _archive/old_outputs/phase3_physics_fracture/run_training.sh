#!/bin/bash
cd /gscratch/scrubbed/june0604/vindr
source ~/.bashrc
conda activate medgemma

# Create log directory
LOGDIR="/gscratch/scrubbed/june0604/vindr/outputs/phase3_physics_fracture/rl_training/current"
mkdir -p $LOGDIR

# Run training
python spine-rl-sim/train_pybullet_final.py > $LOGDIR/training.log 2>&1
