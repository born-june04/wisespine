# Previous Results Summary (One-Stage & Two-Stage Approaches)

## Overview

이 문서는 SpineCLUE 프로젝트에서 시도한 **one-stage**와 **two-stage** 접근법의 결과를 요약합니다.  
새로운 **point cloud assembly** 접근법으로 전환하기 전의 baseline 성능을 기록합니다.

---

## 1. Two-Stage Approach: Detection → Segmentation

### 1.1 Localization Stage (Detection)

#### RetinaNet Results
- **Model**: RetinaNet ResNet50-FPN
- **Dataset**: VerSe (Axial view, center slice only)
- **Training samples**: 141
- **Validation samples**: 120
- **Test samples**: 113

**Best Validation Results:**
- Mean IoU: **0.744**
- Precision: **0.869**
- Recall: **0.623**
- F1 Score: **0.724**
- Detection Rate: **0.623**

**Test Results:**
- Mean IoU: **0.771**
- Precision: **0.502**
- Recall: **0.995**
- F1 Score: **0.337**
- Detection Rate: **0.496**

**Key Findings:**
- 높은 Recall (0.995)로 대부분의 vertebra를 탐지하지만, Precision이 낮아 False Positive가 많음
- Detection Rate가 낮음 (0.496) - 실제로 정확히 매칭된 bbox 비율이 낮음

#### Faster R-CNN Results
- **Model**: Faster R-CNN ResNet50-FPN
- **Dataset**: VerSe (Axial view, center slice only)

**Test Results:**
- Mean IoU: **0.872** (RetinaNet보다 높음)
- Precision: **0.776**
- Recall: **0.990**
- F1 Score: **0.456** (계산 오류 수정 후)
- Detection Rate: **0.495**

**Key Findings:**
- RetinaNet보다 높은 IoU와 Precision
- 여전히 Detection Rate는 낮음

#### Physics-Guided Approaches
- **RetinaNet with Physics Loss**: Mean IoU 0.771, Precision 0.538, Recall 0.997
- **Faster R-CNN with Physics Loss**: Mean IoU 0.870, Precision 0.820, Recall 0.987
- Physics-guided NMS는 성능 향상에 도움이 되지 않음

### 1.2 Segmentation Stage

#### Coarse-to-Fine Pipeline
- **Stage 1**: Heatmap-based localization
- **Stage 2**: ROI-based segmentation (TransUNet)

**Limitations:**
- Detection 실패 시 recovery 불가
- Bounding box dependency
- 여전히 voxel-level local processing
- Global anatomical reasoning 부족

---

## 2. One-Stage Approach

### 2.1 Direct Multi-Class Segmentation

**Approach:**
- 전체 CT → multi-class vertebra segmentation
- TransUNet 또는 3D U-Net 사용

**Limitations:**
- Huge memory footprint
- Class imbalance 문제
- Long-range structure modeling 약함
- Vertebra 간 관계, 정렬, 순서를 직접 학습하지 못함

---

## 3. Core Limitations Identified

### 3.1 Representation Limitations
1. **Local Processing**: Voxel-level processing으로 global structure 학습 어려움
2. **No Explicit Geometry**: Surface geometry 정보 부족
3. **No Orientation Awareness**: Directional features 없음
4. **No Equivariance**: Rotation/pose invariance 부족

### 3.2 Pipeline Limitations
1. **Detection Dependency**: Detection 실패 시 전체 pipeline 실패
2. **No Recovery Mechanism**: Partial spine 처리 어려움
3. **No Global Reasoning**: Vertebra 간 관계를 명시적으로 모델링하지 않음

### 3.3 Metric Observations
- **High Recall, Low Precision**: 많은 False Positive
- **Low Detection Rate**: 정확한 매칭 비율 낮음
- **IoU는 높지만 Detection Rate는 낮음**: Bbox는 잘 잡지만 정확한 매칭은 어려움

---

## 4. Motivation for New Approach

### 4.1 Why Point Cloud Assembly?

1. **Explicit Geometry**: Surface mesh와 point cloud로 geometry 정보 활용
2. **Directional Features**: Surface normals, curvature 등 directional features
3. **SE(3) Equivariance**: Rotation/pose invariant representation
4. **Global Reasoning**: Transformer-based assembly로 vertebra 간 관계 모델링
5. **Partial Robustness**: Partial spine에서도 global reasoning 가능

### 4.2 Expected Improvements

- **Better Partial Handling**: Missing vertebrae inference 가능
- **Explicit Ordering**: Anatomical order 학습
- **Geometric Consistency**: Assembly loss로 geometric constraints
- **Robust Representation**: SE(3) equivariant features

---

## 5. Baseline Metrics for Comparison

### Localization Metrics
- **RetinaNet**: Mean IoU 0.771, Precision 0.502, Recall 0.995
- **Faster R-CNN**: Mean IoU 0.872, Precision 0.776, Recall 0.990

### Segmentation Metrics
- (Two-stage segmentation 결과는 별도 평가 필요)

### Target Metrics for New Approach
- **Vertebra Identification Rate**: > 95% (현재 ~50%)
- **Partial Spine Performance**: Robust handling
- **Geometric Accuracy**: Assembly loss로 측정
- **Ordering Accuracy**: Anatomical order prediction

---

## 6. Data Preparation Notes

### Current Data
- **Dataset**: VerSe 2019
- **Format**: 3D CT volumes with segmentation masks
- **Resolution**: 1mm isotropic (preprocessed)
- **Labels**: C1-L5 vertebra labels

### Required for New Approach
- Mesh extraction from masks
- Point cloud sampling (2k-5k points per vertebra)
- Directional features (normals, curvature)
- Local anatomical coordinate frames

---

## 7. Next Steps

1. ✅ **Results Summary** (이 문서)
2. ⏳ **Geometry Pipeline**: Mask → Mesh → Point Cloud
3. ⏳ **Encoder Pretraining**: SE(3) equivariant encoder
4. ⏳ **Assembly Modeling**: Transformer-based assembly
5. ⏳ **Evaluation**: Comparison with baseline

---

## References

- **RetinaNet Results**: `RESULTS.md` - RetinaNet Localization Results
- **Faster R-CNN Results**: `RESULTS.md` - Faster R-CNN Localization Results
- **Two-Stage Pipeline**: `workspace/networks/spineclue/` - CoarseToFinePipeline
- **Project Guide**: `spine_point_cloud_assembly_project_guide_ver_se_2019.md`

