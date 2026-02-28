# AO Type A2: Split Fracture

## Clinical Definition

AO Type A2 fractures are **coronal or sagittal plane split fractures** that extend through both endplates. The vertebral body is divided into two or more fragments by a **vertical fracture line**. This is caused by **pure axial loading** without significant flexion/extension.

### Key Clinical Features
| Feature | Description |
|---------|-------------|
| **Mechanism** | Pure axial compression (no flexion) |
| **Fracture plane** | Coronal or sagittal split through the body |
| **Endplate involvement** | Both superior and inferior endplates |
| **Fragment pattern** | 2 major fragments separated by fracture line |
| **Stability** | Moderately unstable due to vertical shear |
| **Denis Classification** | Anterior column primarily |

---

## Visualization Results

### Detailed Analysis (Axial + Sagittal + Annotations + Zoomed)

![AO A2 Detailed Analysis](../../figs/AO_A2_Split_Fracture.png)

**Figure interpretation:**
- **Row 1 (AXIAL):** The coronal fracture line is visible as a dark horizontal band through the vertebral body. Yellow arrow marks "Coronal Split Line."
- **Row 2 (SAGITTAL):** Split line traverses the full height of the vertebral body, passing through both endplates.
- **Row 3 (ANALYSIS):** ΔHU maps show bilateral symmetric changes. HU histogram shows leftward shift from cortical disruption at fracture line.

### Severity Progression (Axial + Sagittal)

![AO A2 Severity Gallery](../../figs/AO_A2_Split_Fracture_severity.png)

**Progression notes:**
- **Mild (0.3):** Hairline fracture visible, minimal fragment separation (<1mm equivalent)
- **Moderate (0.6):** Clear dark fracture line, measurable gap between fragments
- **Severe (0.9):** Wide fracture gap, lateral separation of fragments, endplate irregularity

---

## Physics-Based Simulation Logic

### How We Simulate A2

Our simulation models a **coronal plane split** with two key components:

1. **Bilateral Separation**: Opposing displacement fields push left/right halves apart
2. **Fracture Line**: A dark hypodense line at the split location simulates cortical disruption

```
Displacement_left  = -split_gap  (negative x direction)
Displacement_right = +split_gap  (positive x direction)
split_gap = severity × 5 pixels
```

### Step-by-step Simulation Pipeline

1. **Fragment Identification**
   - Divide vertebral body at midline (`x_mid`)
   - Left half: `x < x_mid`, Right half: `x ≥ x_mid`

2. **Separation Deformation**
   - Left fragments displaced in -x direction
   - Right fragments displaced in +x direction
   - Displacement magnitude: `severity × 5` voxels
   - Gaussian smoothing (σ=2) prevents hard boundary artifacts

3. **Fracture Line Rendering**
   - Dark fracture line at `x_mid` with variable width (`2 + severity × 3` px)
   - Random vertical perturbation simulates irregular fracture morphology
   - HU values at fracture line: multiplied by 0.3 (70% reduction — simulates air/fluid/debris gap)

4. **Inverse Warping**
   - `map_coordinates` with bilinear interpolation
   - Maintains CT texture continuity within fragments

### Why This is Physically Correct

| Physical Principle | Our Implementation | Clinical Evidence |
|---|---|---|
| **Vertical fracture line** | Dark line through vertebral body at midline | Magerl (1994): A2 defined by coronal/sagittal split |
| **Both endplates involved** | Fracture line extends full z-range of vertebra | AO definition requires both endplates |
| **Axial loading mechanism** | Symmetric bilateral displacement (no flexion gradient) | Pure axial load produces symmetric split, unlike A1 wedge |
| **Fragment separation** | Opposing displacement vectors for left/right halves | Real splits show measurable inter-fragment gap |
| **Irregular fracture edge** | Random perturbation of split location (±severity×2 px) | Bone fractures are never perfectly straight lines |
| **HU reduction at fracture** | 70% HU reduction at fracture line | Fracture gaps contain low-density material (hematoma, fluid) |
| **Smooth fragment interiors** | Gaussian-filtered displacement within each fragment | Fragments themselves remain structurally intact |

### Source Code Reference
- Simulation: [`generate_visualizations.py → simulate_a2_split()`](generate_visualizations.py)
- Fracture surface: [`ct_physics.py → generate_hierarchical_fracture_surface()`](../../pipeline/modules/ct_physics.py)

---

## Quantitative Validation

From the generated analysis panel:
- **Mean ΔHU**: Negative (density loss from fracture line + trabecular disruption)
- **Max |ΔHU|**: High values at the split line location — cortical bone → fracture gap transition
- **HU distribution**: Clear bimodal shift at fracture zone; intact fragment HU preserved
- **Affected ratio**: Lower than A1 (fracture is more focal — concentrated at the split line)

---

## References

1. **Magerl F, et al.** (1994) "A comprehensive classification of thoracic and lumbar injuries." *Eur Spine J*, 3:184-201.
2. **Vaccaro AR, et al.** (2013) "AOSpine thoracolumbar spine injury classification system." *Spine*, 38(22):2028-2037.
3. **Dai LY, et al.** (2005) "Thoracolumbar fractures in patients with multiple injuries." *J Trauma*, 58(6):1266-1272.
4. **Denis F.** (1983) "The Three Column Spine." *Spine*, 8(8):817-831.
