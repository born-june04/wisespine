# Spinal Field: Geometry-Aware Vertebra Assembly Beyond Segmentation

> 이 문서는 `spine_point_cloud_assembly/`에서 **우리가 실제로 구현/실험한 내용**(코드/스크립트/체크포인트)과 **최종 결과**를 한 곳에 합쳐둔 프로젝트 “싱글 소스 오브 트루스”입니다.  
> 상세 로그/표는 `docs/BASELINE_EVALUATION.md`, `docs/SPINAL_FIELD_EVALUATION.md`, `docs/ABLATION_SUMMARY.*`를 참고하세요.

---

## 1. Core Motivation

Medical spine analysis pipelines overwhelmingly rely on **segmentation-first** approaches (e.g., nnU-Net). While voxel-wise segmentation performance has largely saturated, **downstream clinical tasks remain fragile** due to:

* Adjacent vertebra ambiguity (e.g., T12–L1, L5–L6)
* Error propagation from imperfect masks
* Lack of global anatomical reasoning

**Key insight**: Segmentation answers *where voxels belong locally*, but does not answer *what the spine is globally*.

Spinal Field reframes the problem:

> Segmentation is a noisy observation. Anatomy is a structured geometric field.

---

## 2. Overall Pipeline (what we actually built)

```
CT Scan (3D Volume)
    ↓
[Segmentation Module – external, e.g., nnU-Net]
    ↓
Per-Vertebra Mask (possibly imperfect)
    ↓
Mesh / Point Cloud Extraction
    ↓
Local Geometry Encoder (SE(3)-equivariant, e3nn)
    ↓
Vertebra Embeddings:
  - z_inv ∈ R^512 (rotation-invariant)
  - z_eq  ∈ R^{K×3} (rotation-equivariant vectors; default K=8)
    ↓
Assembly Model (Baseline SetTransformer OR Spinal Field v2)
    ↓
Global Spine Reconstruction
    ↓
Clinical Measurements (e.g., Cobb angle)
```

---

## 3. Encoder: Geometry-Aware Vertebra Representation (implemented)

**Role**: Abstract each vertebra’s local geometry while preserving orientation information.

### Design / Code pointers

* Input: Point cloud per vertebra (from segmentation)
* Architecture:

  * e3nn-based SE(3)-equivariant GNN (`models/encoder_se3.py::SE3PointEncoder`)
  * Radius graph message passing
  * Spherical harmonics + distance embeddings
* Output:

  * **z_inv**: rotation-invariant embedding (512-dim)
  * **z_eq**: equivariant frame features (K×3)

### Key Properties

* Per-vertebra normalization (centroid + scale)
* Robust to segmentation noise
* No absolute spatial assumptions

### Self-supervised pretraining tasks (implemented + evaluated)

Encoder pretraining은 아래 3가지 self-supervised task 조합을 사용합니다(가중치는 Baseline 문서 기준).

- **Rotation Canonicalization**: SE(3) equivariance 유도
- **Contrastive Learning**: type-discriminative embedding 강화
- **Masked Point Modeling**: dropout/마스킹에 대한 강인성 유도

---

## 4. Assembly: Set Transformer + Spinal Field v2 (implemented)

**Role**: Perform *global anatomical reasoning* over a set of vertebrae.

### Input

* Set of vertebra embeddings {z_i}
* Optional equivariant frame features
* Padding mask (missing vertebrae)
* Mask tokens for completion

### Architecture (Baseline)

* Set Transformer (no absolute positional encoding; permutation-invariant)  
  - `models/assembly.py::SpineAssemblyTransformer`
* Learned [MASK] tokens (BERT-style)
* Multi-head self-attention

### Outputs

For each vertebra i:

* **Ordering**: vertebra type logits
* **Pose**:

  * Translation t_i ∈ R^3
  * Rotation R_i ∈ SO(3) (6D → matrix)
* **Completion**:

  * Predicted embedding for missing vertebrae

### Architecture (Spinal Field v2)

Spinal Field는 baseline 위에 다음 inductive bias를 추가합니다.  
코드: `models/assembly_spinal_field.py::SpineAssemblySpinalField`

- **Global spine field token \(g\)**: set output을 pooling → MLP로 요약한 전역 컨텍스트
- **Continuous spine coordinate \(s_i\in[0,1]\)**: 각 vertebra의 spine-axis 상 연속 좌표를 예측
- **FiLM conditioning**: \(g\)로 토큰 특성을 scale/shift하여 인접 타입 분리를 강화
- **B-spline centerline**: \(g\)로부터 control points를 예측해 centerline/프레임을 만들고  
  local offset을 world translation으로 변환
- **(Optional) neighbor delta-pose head**: \(s\)-정렬 기반 “next” 이웃에 대한 상대 pose 예측

→ 결과적으로 ordering이 “로컬 분류”가 아니라 **spine field 추론 문제**가 되도록 유도합니다.

---

## 5. Key Results (what we achieved so far)

### TL;DR (Test split)

아래 수치는 `docs/BASELINE_EVALUATION.md`와 `docs/SPINAL_FIELD_EVALUATION.md`에서 가져온 “대표” 요약입니다.

| Metric | Baseline (v1) | Spinal Field (v2) |
|---|---:|---:|
| **Ordering Accuracy** | 60.79% | **99.47%** |
| **Translation Error (mean)** | 0.0226 mm | **0.0136 mm** |
| **Rotation Error (mean)** | 0.3836° | **0.1033°** |

추가로 encoder 자체의 회전 불변성은 평균 cosine similarity **0.9484**로 매우 우수했습니다(Baseline 문서의 rotation invariance test).

### Ordering

* Baseline ordering accuracy: **60.79%**
* Failure mode: adjacent vertebra confusion

### Spinal Field v2

* Ordering accuracy: **99.47%**
* Near-complete elimination of adjacent-type confusion

---

## 6. What changed vs Baseline (핵심 인사이트)

- **Baseline의 실패 패턴**: 오류의 대부분이 “인접 타입 혼동” (예: T12↔L1, L2↔L3). 로컬 기하만으로는 구분이 어렵습니다.
- **Spinal Field의 성공 요인**: 전역 토큰 \(g\) + 연속 좌표 \(s_i\) + (centerline 기반) 기하적 구조 편향이 **글로벌 일관성**을 강제해 인접 타입을 분리합니다.

---

## 7. Reproducibility (paths + commands)

### Checkpoints (current best-known)

- **Encoder (pretrained)**: `outputs/embeddings/2026-01-11_17-50-10/best_model.pth`
- **Assembly Baseline (v1)**: `outputs/assembly/2026-01-12_13-38-26/best_model.pth`
- **Assembly Spinal Field (v2)**: `outputs/assembly/2026-01-12_21-07-15/best_model.pth`

### Useful docs

- **Baseline eval**: `docs/BASELINE_EVALUATION.md`
- **Spinal Field eval**: `docs/SPINAL_FIELD_EVALUATION.md`
- **Ablations (auto)**: `docs/ABLATION_SUMMARY.md`, `docs/ABLATION_SUMMARY.csv`

### Typical workflow (Phase 1→4)

구현된 엔드투엔드 파이프라인/스크립트는 `README.md`에 정리되어 있고, 큰 흐름은 다음과 같습니다.

1. **Mask → Mesh**: `scripts/extract_meshes.py`
2. **Mesh → Point cloud**: `scripts/sample_points.py`
3. **Directional features**(normals/curvature): `scripts/compute_features.py`
4. **Encoder pretraining**: `scripts/pretrain_encoder.py`
5. **Assembly training**(baseline/spinal_field): `scripts/train_assembly.py`
6. **Evaluation**: `scripts/evaluate.py` (+ embedding/assembly eval helper scripts)

---

## 8. Clinical Relevance (why this matters)

### Example: Cobb Angle Measurement

Clinical workflow:

* Identify L5
* Reference vertebrae above and below
* Compute angular deviation

**nnU-Net only**:

* Sensitive to segmentation and labeling errors
* Incorrect vertebra ID → incorrect angle

**nnU-Net + Spinal Field**:

* Robust vertebra identification
* Correct adjacency even under noisy masks
* Geometry-consistent reconstruction

→ Enables **reliable clinical measurements beyond voxel accuracy**.

---

## 9. Core Contribution (paper framing)

1. Introduce **Spinal Field**, a geometry-aware global representation of the spine
2. Decouple anatomical reasoning from segmentation accuracy
3. Correct downstream errors without retraining segmentation models
4. Enable robust clinical measurements under imperfect segmentation

> We do not replace segmentation; we correct its limitations using geometry.

---
