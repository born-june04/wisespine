
### 11.2 Deep Physical Simulation Review (Honest Assessment)

To rigorously verify the "physics-based" claim, we conducted an exhaustive visual and quantitative inspection of every simulation stage.

#### 1. Hardware & Artifacts
- **Observation**: The `_hardware.nii.gz` file is a binary mask (uint8), not a CT volume. The actual titanium density (3000 HU) and artifacts are composited in the `_artifacts.nii.gz` volume.
- **Verification**: Metal artifacts show correct **bipolar streaks** (bright and dark), confirming the accumulation of Gaussian bloom resembles physical beam hardening, though it is a simplified approximation (no projection-domain simulation).
- **Status**: **REALISTIC (Simplified)**.

#### 2. Scoliosis Deformation
- **Observation**: Cobb 20° to 40° shows clear progression (Peak shift 138px → 162px), but 40° to 60° shows saturation (162px → 163px).
- **Limitation**: The deformation field resolution (0.125 scale) appears to saturate at high curvatures.
- **Status**: **Functionally Valid up to 40°**, plateau at 60°.

#### 3. Tumor Simulation
- **Observation**: Tumor effects are physically modeled (lytic/blastic) but the **affected volume is small** (~286 voxels) because lesions are targeted at pedicle screw locations rather than the vertebral body center.
- **Status**: **Physically Sound but Spatially Limited**.

#### 4. Post-Op Realism
- **Observation**: Laminectomy cleanly removes posterior bone. Hematoma/edema correctly modifies soft tissue density (+50 HU / -20 HU).
- **Status**: **Highly Realistic**.

### Deep Review Visualizations
These panels provide transparency into the simulation quality.

![Review 1: Multi-Plane Inspection](/mmfs1/home/june0604/.gemini/antigravity/brain/92d0520f-43b0-409e-bcfb-e25e827e9d18/review1_multiplane_all_stages.png)
*Fig 11.1: Multi-plane view of all simulation stages. Note the preservation of anatomy across transformations.*

![Review 2: HU Distributions](/mmfs1/home/june0604/.gemini/antigravity/brain/92d0520f-43b0-409e-bcfb-e25e827e9d18/review2_hu_distributions.png)
*Fig 11.2: HU distributions compared to clinical reference bands. Distributions line up well with expectations.*

![Review 3: Hardware Placement](/mmfs1/home/june0604/.gemini/antigravity/brain/92d0520f-43b0-409e-bcfb-e25e827e9d18/review3_hardware_placement.png)
*Fig 11.3: Verification of screw placement within vertebral pedicles.*

![Review 4: Post-Op & Artifacts](/mmfs1/home/june0604/.gemini/antigravity/brain/92d0520f-43b0-409e-bcfb-e25e827e9d18/review4_postop_artifacts.png)
*Fig 11.4: Post-op tissue changes and metal artifacts. Note the realistic streak polarities.*

![Review 5: Scoliosis Quality](/mmfs1/home/june0604/.gemini/antigravity/brain/92d0520f-43b0-409e-bcfb-e25e827e9d18/review5_scoliosis_quality.png)
*Fig 11.5: Scoliosis progression. Note the increasing curvature from 20°.*
