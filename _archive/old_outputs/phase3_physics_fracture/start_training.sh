#!/bin/bash

# Change to working directory
cd /gscratch/scrubbed/june0604/vindr

# Activate conda
source ~/.bashrc
conda activate medgemma

# Set output log
LOGFILE="/gscratch/scrubbed/june0604/vindr/outputs/phase3_physics_fracture/rl_training/current/training.log"

# Run training (redirecting ALL output)
exec python spine-rl-sim/train_pybullet_final.py >> "$LOGFILE" 2>&1

