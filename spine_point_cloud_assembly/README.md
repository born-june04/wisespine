# 🦴 Spine Point Cloud Assembly Project

> Direction-aware vertebra point representations + SE(3)-equivariant assembly for global anatomical reasoning

## Project Structure

```
spine_point_cloud_assembly/
├── data/                    # Processed data (meshes, point clouds)
├── models/                  # Model definitions
│   ├── encoder.py          # SE(3) equivariant point encoder
│   ├── assembly.py         # Spine assembly transformer
│   └── pretraining.py      # Self-supervised pretraining tasks
├── scripts/                 # Execution scripts
│   ├── extract_meshes.py   # Phase 1: Mask → Mesh
│   ├── sample_points.py    # Phase 1: Mesh → Point Cloud
│   ├── compute_features.py # Phase 1: Directional features
│   ├── pretrain_encoder.py # Phase 2: Encoder pretraining
│   ├── train_assembly.py   # Phase 3: Assembly training
│   └── evaluate.py         # Phase 4: Evaluation
├── utils/                  # Utility functions
│   ├── geometry.py         # Mesh/point cloud operations
│   ├── features.py         # Feature computation (normals, curvature)
│   └── visualization.py    # Visualization tools
└── outputs/                # Output directories
    ├── meshes/            # Extracted meshes
    ├── point_clouds/      # Sampled point clouds
    ├── embeddings/        # Learned embeddings
    └── visualizations/    # Visualization outputs
```

## Quick Start

### Phase 1: Geometry Pipeline

```bash
# 1. Extract meshes from segmentation masks
python scripts/extract_meshes.py \
    --mask_dir VerSe/processed \
    --output_dir outputs/meshes

# 2. Sample point clouds from meshes
python scripts/sample_points.py \
    --mesh_dir outputs/meshes \
    --output_dir outputs/point_clouds \
    --num_points 2048

# 3. Compute directional features
python scripts/compute_features.py \
    --point_cloud_dir outputs/point_clouds \
    --output_dir outputs/point_clouds \
    --compute_normals \
    --compute_curvature
```

### Phase 2: Encoder Pretraining

```bash
python scripts/pretrain_encoder.py \
    --point_cloud_dir outputs/point_clouds \
    --output_dir outputs/embeddings \
    --batch_size 32 \
    --epochs 100
```

### Phase 3: Assembly Training

```bash
python scripts/train_assembly.py \
    --point_cloud_dir outputs/point_clouds \
    --encoder_path outputs/embeddings/best_encoder.pth \
    --output_dir outputs/assembly \
    --batch_size 16 \
    --epochs 50
```

### Phase 4: Evaluation

```bash
python scripts/evaluate.py \
    --assembly_model_path outputs/assembly/best_model.pth \
    --test_data_dir outputs/point_clouds \
    --output_dir outputs/evaluation
```

## Installation

### Quick Install
```bash
cd spine_point_cloud_assembly
pip install scikit-image  # Required for Phase 1
# Or install all dependencies:
pip install -r requirements.txt
```

### Verify Installation
```bash
python scripts/test_geometry_pipeline.py
```

See `INSTALL.md` for detailed installation instructions.

## Requirements

See `requirements.txt` for dependencies.

## Documentation

- **Project Guide**: `../spine_point_cloud_assembly_project_guide_ver_se_2019.md`
- **Previous Results**: `../PREVIOUS_RESULTS_SUMMARY.md`

