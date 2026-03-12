#!/bin/bash
# WiseSpine workspace reorganization script
# Run: bash reorganize.sh

set -e

echo "=== Reorganizing WiseSpine workspace ==="

# 1. Create archive subdirs
mkdir -p _archive/old_demos _archive/old_engines _archive/old_scripts _archive/old_docs

# 2. Archive old demo directories (keep only v8)
for d in fracture_v3_demo fracture_v4_demo fracture_v5_demo fracture_v6_demo fracture_v7_demo 3d_fracture_vis fracture_3d; do
    if [ -d "$d" ]; then
        echo "  Moving $d → _archive/old_demos/"
        mv "$d" _archive/old_demos/
    fi
done

# 3. Archive old engine versions (keep v6 + v8 only)
for f in pipeline/modules/fracture_engine_v3.py \
         pipeline/modules/fracture_engine_v4.py \
         pipeline/modules/fracture_engine_v5.py \
         pipeline/modules/fracture_engine_v7.py \
         pipeline/modules/fracture_simulator_v2.py; do
    if [ -f "$f" ]; then
        echo "  Moving $f → _archive/old_engines/"
        mv "$f" _archive/old_engines/
    fi
done

# 4. Archive old visualization generators
for f in pipeline/modules/_gen_3d_mesh_fracture.py \
         pipeline/modules/_gen_deformation_visuals.py \
         pipeline/modules/_gen_fracture_visuals.py \
         pipeline/modules/_gen_md_style_examples.py \
         pipeline/modules/_gen_real_fracture_visuals.py \
         pipeline/modules/fracture_visualization.py \
         pipeline/modules/visualize_3d_fractures.py; do
    if [ -f "$f" ]; then
        echo "  Moving $f → _archive/old_scripts/"
        mv "$f" _archive/old_scripts/
    fi
done

# 5. Archive old renderers and unrelated modules
for f in pipeline/modules/ct_renderer.py \
         pipeline/modules/ct_renderer_v2.py \
         pipeline/modules/ct_renderer_warping.py \
         pipeline/modules/pybullet_ct_renderer.py \
         pipeline/modules/pybullet_fracture_env.py \
         pipeline/modules/taichi_ct_renderer.py \
         pipeline/modules/adversary_env.py \
         pipeline/modules/physical_adversary_env.py \
         pipeline/modules/validation_callback.py \
         pipeline/modules/assembly_wrapper.py \
         pipeline/modules/test_usecases_v5.py; do
    if [ -f "$f" ]; then
        echo "  Moving $f → _archive/old_scripts/"
        mv "$f" _archive/old_scripts/
    fi
done

# 6. Archive unrelated pipeline scripts
for f in pipeline/simulate_scoliosis.py \
         pipeline/simulate_tumors.py \
         pipeline/simulate_surgery_process.py \
         pipeline/synthesize_artifacts_simple.py \
         pipeline/place_hardware_physics.py \
         pipeline/simulate_causal_response.py; do
    if [ -f "$f" ]; then
        echo "  Moving $f → _archive/old_scripts/"
        mv "$f" _archive/old_scripts/
    fi
done

# 7. Archive unrelated top-level files
for f in cv.tex technical_implementation.md usecase_report.md test_fracture_improvements.py; do
    if [ -f "$f" ]; then
        echo "  Moving $f → _archive/old_docs/"
        mv "$f" _archive/old_docs/
    fi
done

# 8. Archive old directories
for d in analysis extensions usecase_tests outputs scripts evaluation; do
    if [ -d "$d" ]; then
        echo "  Moving $d → _archive/"
        mv "$d" _archive/
    fi
done

echo ""
echo "=== Done! Remaining structure ==="
find . -maxdepth 2 -not -path './_archive/*' -not -path './.git/*' -not -path './.vscode/*' -not -path './__pycache__/*' -not -path './pipeline/__pycache__/*' -not -path './pipeline/modules/__pycache__/*' | sort
