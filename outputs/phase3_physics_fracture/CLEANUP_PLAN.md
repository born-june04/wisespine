# Phase 3 Directory Cleanup Plan

## Current Status (2026-02-03)
Total Size: ~4.7GB

### Large Files to Address:
1. **ct_renderings/** (973MB)
   - fractured_ct_manual.nii.gz (973MB) - ONLY file needed
   - Action: Keep this ONE file, it's recent and useful

2. **evaluation/** (1.7GB)
   - rl_fractured.nii.gz (969MB) - Can regenerate
   - ts_manual/, ts_original/, ts_rl/ (TotalSegmentator outputs) - Can regenerate
   - Action: DELETE all, can regenerate in 5 minutes

3. **rl_training/** (11MB)
   - 7 training runs (most failed/aborted)
   - 2026-02-02_18-30-42/ - FINAL successful run (KEEP!)
   - Others: Failed attempts
   - Action: Archive old runs, keep only final

4. **pybullet_models/** (6.7MB)
   - L1_fractured/ - Current working model (KEEP!)
   - pybullet_test/, pybullet_fracture/, pybullet_fracture_v2/ - Old tests
   - Action: Archive old tests

5. **ts_predictions/** (54MB)
   - ts_warped/ - Old prediction results
   - Action: Archive, not critical

## Cleanup Actions:

### DELETE (Save ~1.7GB):
```bash
# evaluation/ - can regenerate quickly
rm -rf evaluation/
```

### ARCHIVE (Move to _archive/):
```bash
mkdir -p _archive/phase3_old_experiments/

# Old RL training runs
mv rl_training/2026-02-02_17-* _archive/phase3_old_experiments/
mv rl_training/2026-02-02_18-03-11 _archive/phase3_old_experiments/
mv rl_training/current _archive/phase3_old_experiments/
mv rl_training/simple _archive/phase3_old_experiments/

# Old pybullet tests
mv pybullet_models/pybullet_test _archive/phase3_old_experiments/
mv pybullet_models/pybullet_fracture _archive/phase3_old_experiments/
mv pybullet_models/pybullet_fracture_v2 _archive/phase3_old_experiments/

# Old predictions
mv ts_predictions/ _archive/phase3_old_experiments/
```

### KEEP:
```bash
# Final successful RL training
rl_training/2026-02-02_18-30-42/

# Current working model
pybullet_models/L1_fractured/

# Manual fracture CT (for comparison)
ct_renderings/fractured_ct_manual.nii.gz

# Documentation
*.md files

# Visualizations (relatively small, useful)
visualizations/
```

## New Directory Structure:

```
outputs/phase3_physics_fracture/
├── README.md
├── LITERATURE_REVIEW.md
├── COMPARISON_PLAN.md
├── PYBULLET_RL_COMPLETE.md
│
├── ct_renderings/
│   └── fractured_ct_manual.nii.gz  (973MB - baseline)
│
├── pybullet_models/
│   └── L1_fractured/               (fragments, visualization)
│
├── rl_training/
│   └── 2026-02-02_18-30-42/        (final model + validation)
│       ├── fracture_agent_final.zip
│       └── validation/
│
├── visualizations/                  (15MB - all plots/images)
│
└── _archive/
    └── phase3_old_experiments/      (archived failed runs)

Expected Size After Cleanup: ~1.0GB (from 4.7GB)
Savings: ~3.7GB
```

## Next Phase Preparation:

Create new directory for surgical artifacts:
```
outputs/phase4_surgical_artifacts/
├── README.md
├── implant_models/      (pedicle screws, rods, etc.)
├── ct_synthesis/        (metal artifact simulation)
├── rl_training/
├── evaluation/
└── visualizations/
```

