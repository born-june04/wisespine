# Fracture Type Reports — Index & Summary

**Repository**: wisespine (Physics-Based Fracture Simulation)

---

## Per-Type Reports

| AO Type | Name | Stability | TS Impact | Report |
|---------|------|-----------|-----------|--------|
| **A1** | Wedge Compression | Stable | Low (~1-2.5%) | [AO_A1_Wedge_Compression.md](./AO_A1_Wedge_Compression.md) |
| **A2** | Split Fracture | Potentially Unstable | Moderate (~2-7%) | [AO_A2_Split_Fracture.md](./AO_A2_Split_Fracture.md) |
| **A3** | Incomplete Burst | Unstable | Moderate (~4-11%) | [AO_A3_Incomplete_Burst.md](./AO_A3_Incomplete_Burst.md) |
| **A4** | Complete Burst | Highly Unstable | High (~9-23%) | [AO_A4_Complete_Burst.md](./AO_A4_Complete_Burst.md) |

---

## Comparative Summary

### Mechanism & Feature Comparison

| Feature | A1 (Wedge) | A2 (Split) | A3 (Inc. Burst) | A4 (Comp. Burst) |
|---------|-----------|-----------|-----------------|-------------------|
| **Force** | Flexion-compression | Axial (coronal) | Axial (centered) | Axial (explosive) |
| **Height Loss** | Anterior only | Minimal | Uniform, moderate | Severe, bilateral |
| **Posterior Wall** | Intact | Intact | Fractured (limited) | Disrupted |
| **Canal Compromise** | None | None | <25% | >25% (retropulsion) |
| **Fragment Count** | 1-2 | 2 | 3-5 | 5-8 |
| **Neurological Risk** | Low | Low | Moderate | **High** |

### Physics Simulation Parameters

```
A1:   Gradient anterior compression      (flexion + axial)
A2:   Coronal plane split               (pure axial, bilateral separation)
A3:   Uniform axial + posterior wall     (axial, center-peaked compression)
A4:   Explosive radial + retropulsion   (5-point loading, posterior fragment)
```

### Segmentation Impact Progression

```
TS Dice Degradation (Moderate Severity):

A1  ####                    ~1.5%
A2  ########                ~4%
A3  ##############          ~7%
A4  ############################  ~14%
                              Target: 20-30%
```

### AO Types Summary Visualization

![AO A1-A4 all types summary comparison](../../figs/AO_all_types_summary.png)
