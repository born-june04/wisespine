# TS mask-based reconstruction: limitations (and why Spinal Field is the right abstraction)

This note summarizes what we can and cannot trust when using **TotalSegmentator (TS) vertebra masks** as the starting point for downstream clinical geometry.

The goal is not to argue *“TS is bad”*, but to formalize:

1. **Why voxel-wise segmentation metrics (Dice/IoU) do not guarantee clinical usability**
2. **What structural failure modes appear in practice**
3. **Why we need a structured 3D representation (Spinal Field) as the foundation**

---

## What TS provides (and what it does not)

- **Provides**: per-vertebra binary masks `vertebrae_*.nii.gz`
- **Does not provide**: a globally consistent *spine object* (chain uniqueness, ordering, continuity, physically plausible geometry)

Clinically meaningful tasks (Cobb, alignment, simulation/planning) require the latter.

---

## Structural failure signals we track (B/C alignment)

We explicitly separate **coverage mismatch** from **non-spine spurious**:

- **extra_vs_gt**: predicted label exists but GT does not.
  - This can happen due to **GT/FOV/annotation coverage mismatch** (e.g., S1 present in CT but not annotated).
  - Therefore, **extra_vs_gt is NOT automatically an error**.
- **spurious_non_spine_proxy**: evidence-based proxy for “non-spine structures being segmented as vertebra”.
  - **non_bone**: HU p90 inside the predicted mask is below threshold (default 120)
  - **axis_outlier**: centroid far from the estimated spine axis (PCA on centroids)
  - `spurious_non_spine_proxy = non_bone OR axis_outlier`

---

## Quantitative summary (current eval set)

Auto-generated summary:

- `outputs/evaluation/ts_failure_summary.md`
- `outputs/evaluation/ts_failure_summary.json`

Representative figure:

- `outputs/evaluation/fig1_ts_structure_mismatch.png`

Representative case folders (overlay + chain plot):

- `outputs/evaluation/exemplars/*/*/`

---

## Why this motivates Spinal Field (the “foundation” claim)

The problem is not “segmentation is imperfect”; the problem is:

- **Downstream tasks need a consistent 3D spine object**
- **Segmentation produces disconnected local evidence**
- **Clinical failure is dominated by global structural violations**, which are not visible to Dice/IoU

**Spinal Field** is the abstraction that turns noisy masks into a structured 3D object:

- Continuous centerline + moving frame
- Ordered vertebra attachments with uniqueness/ordering/continuity priors
- Energy-based refinement and physics-aware extensions (simulation/planning)

This is how we move from “mask output” → “clinician-usable virtual spine”.


