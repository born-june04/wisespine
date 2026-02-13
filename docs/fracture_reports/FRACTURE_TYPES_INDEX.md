# 🏥 Fracture Type Reports — Index & Next Steps

**Date**: 2026-02-10  
**Repository**: wisespine (Track B: Physics-Based Counterfactual Reasoning)

---

## 📁 Per-Type Reports

| AO Type | Name | Stability | TS Impact | Report |
|---------|------|-----------|-----------|--------|
| **A1** | Wedge Compression | Stable | Low (~1-2.5%) | [AO_A1_Wedge_Compression.md](./AO_A1_Wedge_Compression.md) |
| **A2** | Split Fracture | Potentially Unstable | Moderate (~2-7%) | [AO_A2_Split_Fracture.md](./AO_A2_Split_Fracture.md) |
| **A3** | Incomplete Burst | Unstable | Moderate (~4-11%) | [AO_A3_Incomplete_Burst.md](./AO_A3_Incomplete_Burst.md) |
| **A4** | Complete Burst | Highly Unstable | High (~9-23%) | [AO_A4_Complete_Burst.md](./AO_A4_Complete_Burst.md) |

---

## 📊 Comparative Summary

### Mechanism & Feature Comparison

| Feature | A1 (Wedge) | A2 (Split) | A3 (Inc. Burst) | A4 (Comp. Burst) |
|---------|-----------|-----------|-----------------|-------------------|
| **Force** | Flexion-compression | Axial (coronal) | Axial (centered) | Axial (explosive) |
| **Height Loss** | Anterior only | Minimal | Uniform, moderate | Severe, bilateral |
| **Posterior Wall** | ✅ Intact | ✅ Intact | ⚠️ Fractured (limited) | ❌ Disrupted |
| **Canal Compromise** | None | None | <25% | >25% (retropulsion) |
| **Fragment Count** | 1-2 | 2 | 3-5 | 5-8 |
| **Neurological Risk** | Low | Low | Moderate | **High** |

### Physics Simulation Comparison

```
A1:   ▼▼▼ → ▼      Gradient anterior compression
A2:   ← | →          Coronal plane split
A3:   ▼▼▼▼▼▼▼▼▼      Uniform axial (posterior constrained)
A4:   💥 radial + ↑↑  Explosive + retropulsion
```

### Segmentation Impact Progression

```
TS Dice Degradation (Moderate Severity):

A1  ████░░░░░░░░░░░░░░░░  ~1.5%
A2  ████████░░░░░░░░░░░░  ~4%
A3  ██████████████░░░░░░  ~7%
A4  ████████████████████████████  ~14%
                              Target: 20-30%
```

---

## 🗺️ Next Steps Plan

### Phase 1: Fracture Pipeline Completion (1-2 weeks)

| # | Task | Priority | Status |
|---|------|----------|--------|
| 1 | **Multi-subject fracture generation** — Run A1-A4 on 10+ VerSe subjects | 🔴 High | Not started |
| 2 | **Severity sweep** — Generate mild/moderate/severe for each type | 🔴 High | Not started |
| 3 | **Fracture + Scoliosis combination** — Composite abnormalities | 🟡 Medium | Not started |
| 4 | **TS evaluation on fracture dataset** — Quantify exact Dice degradation | 🔴 High | Not started |

### Phase 2: Track A Integration (2-3 weeks)

| # | Task | Priority | Status |
|---|------|----------|--------|
| 5 | **AO Classification Model** — Train classifier on synthetic A1-A4 data | 🔴 High | Not started |
| 6 | **nnU-Net Fine-tuning** — Use fracture data for robust segmentation | 🔴 High | In progress (parallel) |
| 7 | **Real fracture CT validation** — Compare synthetic vs clinical CTs | 🟡 Medium | Needs data |
| 8 | **Fracture severity assessment** — Predict severity from synthetic data | 🟡 Medium | Not started |

### Phase 3: Track B — Counterfactual Reasoning (3-4 weeks)

| # | Task | Priority | Status |
|---|------|----------|--------|
| 9 | **Counterfactual fracture queries** — "What if bone density was 50% lower?" | 🔴 High | Not started |
| 10 | **Transitivity tests** — A1→A3→A4 progression consistency | 🟡 Medium | Not started |
| 11 | **Physical property estimation** — Extract BV/TV, cortical thickness from CT | 🟡 Medium | Not started |
| 12 | **Surgeon evaluation study** — Realism rating (1-10) by radiologists | 🟢 Low | Planning |

### Phase 4: Publication Preparation (4-6 weeks)

| # | Task | Priority | Status |
|---|------|----------|--------|
| 13 | **MICCAI paper** (Track A) — Robust segmentation + AO classification | 🔴 High | Drafting |
| 14 | **NeurIPS paper** (Track B) — Physics-based counterfactual reasoning | 🟡 Medium | Planning |
| 15 | **Ablation study** — CV aug vs Physics approach comparison | 🔴 High | Not started |
| 16 | **Multi-site generalization** — Test on external datasets | 🟡 Medium | Not started |

---

## 🎯 Immediate Priorities (This Week)

1. **Generate fracture dataset**: Run the pipeline for A1-A4 × 3 severities × 10 subjects = **120 volumes**
2. **Fine-tune nnU-Net** with current fracture data (parallel with wisespine_for_abnormal work)
3. **Validate fracture outputs** — Visual QA + statistical realism checks
4. **AO classification baseline** — Set up classification head on TS features

---

## 📝 Notes

- All fracture simulations use the same underlying physics engine (PyBullet + Taichi) with type-specific force models
- The rendering pipeline (`ct_physics.py`) supports all 4 types via the `fracture_type` parameter
- A4 burst fractures are most clinically relevant and cause the highest segmentation degradation
- Current pipeline generates 2D slices; 3D coherent volume generation (`generate_3d_trabecular_volume()`) is available for full 3D consistency
