# WiseSpine: Physics-Based Spine Fracture Simulation for Robust Segmentation

**Physically Grounded Fracture Simulation & Adversarial Training for Robust Medical Image Segmentation**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

---

## Motivation

Spine segmentation models (TotalSegmentator, nnU-Net) achieve Dice > 0.95 on normal anatomy but **degrade 10–30% on abnormal cases** — fractures, surgical hardware, and deformities. Existing augmentation approaches (random noise, GAN-based) lack physical plausibility and cannot reproduce the full spectrum of real clinical abnormalities.

## Core Idea

WiseSpine addresses this gap through **physics-based simulation** of spine abnormalities that are grounded in biomechanical principles:

1. **Fracture Simulation** — Grid-based P2G/G2P stress transfer with Continuum Damage Mechanics (CDM), producing AO-classified fracture types (A1–A4) on real VerSe CT data
2. **Surgical Artifact Synthesis** — Pedicle screw placement, metal artifact simulation (streak, blooming, beam hardening)
3. **Adversarial RL** — Reinforcement learning agent that discovers optimal failure-inducing configurations
4. **Causal Graph** — 16-node directed acyclic graph encoding the causal relationships between spine pathologies

### What Makes This Different

| Approach | Physics | Adversarial | Clinical Types | Integration |
|----------|:-------:|:-----------:|:--------------:|:-----------:|
| Random augmentation | - | - | - | - |
| GAN-based synthesis | - | Yes | - | - |
| FEM simulation | Yes (slow) | - | Some | - |
| **WiseSpine** | **Yes (real-time)** | **Yes (RL)** | **AO A1–A4** | **All three** |

---

## Key Results

### Fracture Simulation — AO Classification

Physics-based fracture simulation on real L4 vertebra (VerSe sub-verse503), following AO Spine classification:

| AO Type | Name | Mechanism | TS Dice Degradation | Clinical Stability |
|---------|------|-----------|:-------------------:|-------------------|
| A1 | Wedge Compression | Flexion-compression | ~1–2.5% | Stable |
| A2 | Split Fracture | Coronal separation | ~2–7% | Potentially unstable |
| A3 | Incomplete Burst | Axial compression | ~4–11% | Unstable |
| A4 | Complete Burst | Explosive radial | ~9–23% | Highly unstable |

![AO A1–A4 comparison: 3D mesh + stress diffusion + damage timeline](figs/v2_combined_comparison.png)

### Surgical Artifact Impact on TotalSegmentator

| Configuration | L1 Degradation | L2 Degradation | Average |
|--------------|:--------------:|:--------------:|:-------:|
| Screws only | 3.55% | **14.09%** | 8.82% |
| Screws + Rod | 4.57% | **17.28%** | **10.93%** |
| Multi-level | 2.20% | 3.84% | 3.02% |

**Novel finding — Adjacent vertebra effect**: Hardware placed in L1 causes greater degradation in L2 (17.28%) than L1 itself, due to streak artifact propagation into the adjacent level.

---

## Fracture Physics Engine

The core simulation engine uses a particle-based approach with grid-based stress transfer:

```
Particle Sampling (50K) -> Region Classification (7 zones)
  -> AO-Type Loading -> Grid P2G/G2P Stress (64^3, Jacobi relaxation)
  -> CDM Damage Evolution -> Fragment Detection -> CT HU Mapping
```

Each AO fracture type has biomechanically-specific force configurations:
- **A1 (Wedge)**: Anterior flexion-compression load
- **A2 (Split)**: Coronal plane separation force
- **A3 (Incomplete Burst)**: Axial compression with posterior constraint
- **A4 (Complete Burst)**: 5-point explosive radial loading + retropulsion

For full technical details, see [docs/fracture_physics.md](docs/fracture_physics.md).

---

## Repository Structure

```
wisespine/
├── pipeline/
│   ├── run_batch_pipeline.py          # Full simulation orchestrator
│   ├── simulate_scoliosis.py          # Scoliosis simulation
│   ├── simulate_tumors.py             # Tumor synthesis
│   ├── place_hardware_physics.py      # Pedicle screw placement
│   ├── synthesize_artifacts_simple.py # Metal artifact synthesis
│   ├── causal_graph.py                # Causal DAG (16 nodes, 19 edges)
│   └── modules/
│       ├── fracture_simulator_v2.py   # Core fracture physics engine (CDM)
│       ├── ct_physics.py              # CT imaging physics
│       ├── tumor_synthesis.py         # Lytic/blastic lesion generation
│       ├── artifact_physics.py        # Metal artifact physics
│       └── spine_deformation.py       # Scoliosis deformation
├── analysis/                          # Validation & visualization scripts
├── extensions/                        # Ligament, canal, disc, temporal simulation
├── docs/
│   ├── fracture_physics.md            # Fracture simulator technical docs
│   ├── COMPREHENSIVE_REPORT.md        # Full project history & analysis
│   └── fracture_reports/              # Per-AO-type detailed reports
├── figs/                              # All figures
└── VerSe/                             # VerSe dataset
```

## Documentation

- **[docs/fracture_physics.md](docs/fracture_physics.md)** — Fracture simulator v2 architecture, CDM model, grid-based stress transfer
- **[docs/fracture_reports/](docs/fracture_reports/)** — Per-AO-type reports (A1–A4) with clinical definitions, simulation logic, and visualizations
- **[docs/COMPREHENSIVE_REPORT.md](docs/COMPREHENSIVE_REPORT.md)** — Full project evolution, phase history, and technical deep dive

## Quick Start

```bash
conda activate py311
cd /gscratch/scrubbed/june0604/wisespine

# Generate AO fracture visualizations
python docs/fracture_reports/generate_visualizations.py

# Run fracture simulation on real vertebra
python pipeline/modules/fracture_simulator_v2.py --test-full-physics

# Generate 3D mesh + CT combined views
python pipeline/modules/_gen_3d_mesh_fracture.py

# Run full batch pipeline (scoliosis + tumor + hardware + artifacts)
python pipeline/run_batch_pipeline.py
```

---

**Contact**: june0604@uw.edu
