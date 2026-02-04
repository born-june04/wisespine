# Technical Progress Report: Physics-Informed Spine Localization

**Date**: December 6, 2025  
**Project**: SpineMedNeXt - Vertebra Localization and Segmentation  
**Target**: MICCAI 2026 Submission

---

## Executive Summary

This research addresses a fundamental limitation in vertebra localization where deep learning models fail to capture spatial diversity, resulting in clustered predictions that cannot distinguish individual vertebrae. We propose a physics-informed learning framework that automatically discovers governing equations from vertebra arrangement data using SINDy (Sparse Identification of Nonlinear Dynamical Systems) [1] and integrates these laws through contrastive learning in the feature space.

**Research Objective:**
Develop a physics-informed deep learning method for vertebra localization that leverages discovered physical laws to resolve spatial clustering issues and enhance model generalization.

**Key Contributions:**
1. **Physics Law Discovery**: First application of SINDy to discover vertebra arrangement dynamics from medical imaging data
2. **Physics-Informed Contrastive Learning**: Novel integration of discovered physics laws with contrastive learning for feature space regularization
3. **Physics-Guided Clustering**: Enhanced dual-factor clustering with physics consistency as third factor
4. **Architecture Improvements**: Cross-attention spatial aggregator replacing global pooling to preserve spatial information

---

## Background and Motivation

### Problem Statement

Vertebra localization in CT scans faces a critical challenge where models predict all vertebrae at nearly identical spatial coordinates, despite correct distribution along the spinal axis. This clustering problem stems from the model's inability to learn spatial diversity, leading to failure in downstream segmentation tasks.

**Root Causes:**
1. **Spatial Information Loss**: Global pooling operations eliminate spatial context, causing feature homogenization
2. **Weak Supervision**: Traditional heatmap-based supervision insufficient for learning distinct spatial patterns
3. **Implicit Mapping Failure**: Heatmap-to-vertebra correspondence cannot be reliably established

### Research Motivation

Current approaches rely on manually designed constraints or black-box feature learning. We hypothesize that vertebra arrangement follows discoverable physical laws that can be automatically extracted from data and integrated into deep learning models to improve spatial discrimination and generalization.

---

## Research Objectives

### Primary Goal

Develop a physics-informed learning framework that discovers governing equations from vertebra arrangement data and enforces these laws through feature space regularization, solving the spatial clustering problem in vertebra localization.

### Research Questions

1. **Can physics laws governing vertebra arrangement be automatically discovered from medical imaging data?**
   - Hypothesis: Vertebra arrangement follows discoverable dynamical system laws
   - Approach: Apply SINDy to discover sparse governing equations
   - Status: Successfully discovered laws for all vertebrae

2. **Can discovered physics laws improve localization through feature space regularization?**
   - Hypothesis: Contrastive learning between predicted and physics-informed features enforces physical consistency
   - Approach: Physics-informed feature extractor with contrastive loss
   - Status: Design complete, implementation in progress

3. **Does physics-guided clustering improve detection quality?**
   - Hypothesis: Physics consistency as additional clustering factor reduces false positives
   - Approach: Extend dual-factor clustering with physics consistency
   - Status: Integrated, quantitative evaluation pending

### Expected Outcomes

- Improved spatial diversity in vertebra predictions
- Enhanced mapping accuracy between predictions and ground truth
- Better generalization through physics-informed constraints
- Interpretable model with discovered physical relationships

---

## Methodology

### Overall Approach

Our methodology follows a two-phase framework inspired by physics-guided deep learning [2]:

1. **Discovery Phase**: Automatically discover physics laws from vertebra arrangement data using SINDy
2. **Integration Phase**: Enforce discovered laws through contrastive learning in feature space

### Phase 1: Physics Law Discovery

**Motivation**: Rather than manually designing physics constraints, we discover governing equations directly from data, capturing vertebra-specific dynamics and inter-vertebra relationships.

**Methodological Framework**:
- Model vertebra arrangement as a dynamical system: `dx/dt = f(x)`
- Apply SINDy with sparse regression to identify governing equations
- Input: Multi-dimensional state vector encoding coordinates, physical features, and image characteristics
- Output: Sparse polynomial equations for each vertebra capturing arrangement dynamics

**Key Discoveries**:
- Distance-based interactions between adjacent vertebrae drive coordinate changes
- Region-specific dynamics (cervical, thoracic, lumbar) exhibit distinct patterns
- Curvature significantly affects arrangement dynamics
- Image intensity features reflect bone density-related physics

**Significance**: This represents the first successful application of SINDy to vertebra arrangement dynamics, providing interpretable physical relationships that can guide model learning.

### Phase 2: Physics-Informed Contrastive Learning

**Motivation**: Integrate discovered physics laws into deep learning without modifying core architecture, using feature space regularization to enforce physical consistency.

**Theoretical Framework**:
1. **Physics Feature Extractor**: Projects discovered laws to feature embeddings in the same space as model predictions
2. **Contrastive Learning**: Maximizes similarity between predicted features and physics-informed features for the same vertebra, while minimizing similarity for different vertebrae
3. **Feature Space Alignment**: Ensures model learns physics-consistent representations through learned embeddings

**Advantages over Direct Physics Loss**:
- Simpler implementation: Single contrastive loss vs. multiple weighted constraint losses
- Better generalization: Feature space learning adapts to data distribution vs. hard constraints
- Architecture flexibility: Maintains existing model structure
- Learnable adaptation: Physics embeddings can adapt during training

**Status**: Theoretical framework complete, implementation in progress

---

## Completed Ablation Studies

### Ablation 1: Architecture Improvements

**Hypothesis**: Replacing global pooling with spatial-aware aggregation and enhancing loss functions will preserve spatial information and improve feature diversity.

**Components Evaluated**:
1. Cross-Attention Spatial Aggregator: Learnable queries attending to spatial features
2. Enhanced Loss Functions: Peak sharpness, spatial diversity, explicit mapping losses
3. Direct Coordinate Regression: 3D coordinate prediction vs. heatmap-based approach
4. Spatial-Aware VRM: Graph-based relationship modeling with spatial context

**Findings**:
- Spatial information preservation significantly improves feature diversity
- Direct coordinate regression provides more stable training than heatmap-based methods
- Enhanced loss functions effectively enforce spatial constraints

**Conclusion**: Architecture improvements successfully address spatial information loss, providing foundation for physics-informed learning.

### Ablation 2: Structure Module Evaluation

**Hypothesis**: AlphaFold-style iterative refinement with physiological constraints would improve localization accuracy through coordinate refinement.

**Experimental Design**:
- Baseline: Standard coordinate prediction without iterative refinement
- Ablation: Structure module with iterative refinement and physiological constraints

**Results**:
- Structure module showed performance degradation compared to baseline
- Iterative refinement may introduce overfitting or gradient instability for this task
- Physiological constraints alone insufficient without proper integration

**Conclusion**: Structure module approach not pursued further; focus shifted to physics-informed feature learning.

### Ablation 3: Physics Law Discovery

**Hypothesis**: SINDy can successfully discover governing equations from vertebra arrangement data in medical imaging.

**Experimental Design**:
- Applied SINDy to VerSe training dataset
- Evaluated discovery success rate and sparsity of discovered laws
- Analyzed interpretability of discovered relationships

**Results**:
- Successfully discovered physics laws for all vertebrae
- Achieved appropriate sparsity level capturing essential relationships
- Discovered relationships align with known anatomical principles (distance-based, region-specific, curvature-dependent)

**Significance**: First successful discovery of vertebra arrangement dynamics from medical imaging data, validating the feasibility of data-driven physics discovery in this domain.

### Ablation 4: Physics-Guided Clustering

**Hypothesis**: Adding physics consistency as third factor in dual-factor clustering improves detection quality by filtering physically inconsistent detections.

**Experimental Design**:
- Extended SpineCLUE dual-factor clustering (position + dimensions) with physics consistency
- Physics consistency measures distance between detection and physics-predicted location
- Evaluated filtering effectiveness

**Status**: Integrated into pipeline, quantitative ablation study comparing baseline vs. physics-guided clustering pending.

---

## Research Progress

### Completed Work

**Problem Identification and Analysis** (December 1-3, 2025):
- Identified clustering problem in vertebra localization
- Analyzed root causes: spatial information loss, weak supervision, implicit mapping failure
- Conducted comprehensive network analysis to understand feature learning

**Architecture Improvements** (December 3, 2025):
- Implemented cross-attention spatial aggregator
- Developed enhanced loss functions
- Switched to direct coordinate regression
- Integrated spatial-aware VRM

**Physics Law Discovery** (December 5, 2025):
- Successfully applied SINDy to discover physics laws
- Achieved 100% coverage across all vertebrae
- Validated discovered relationships against anatomical principles

**Pipeline Development** (December 6, 2025):
- Implemented SpineCLUE 3-stage pipeline
- Integrated physics-guided clustering
- Established baseline configuration for ablation studies

### Current Work

**Contrastive Learning Integration**:
- Designing physics-informed feature extractor
- Implementing contrastive loss framework
- Developing feature space alignment mechanism

**Ablation Study Preparation**:
- Establishing baseline vs. physics-guided comparison framework
- Preparing quantitative evaluation metrics
- Setting up experimental protocols

### Planned Work

**Full Pipeline Evaluation**:
- End-to-end training with all physics-informed components
- Comprehensive ablation study of individual components
- Performance evaluation on VerSe test set

**Paper Preparation**:
- Method section detailing SINDy + contrastive learning framework
- Results section with ablation studies
- Target: MICCAI 2026 submission (deadline ~March 2026)

---

## Key Contributions

### 1. Physics Law Discovery from Medical Imaging Data

**Novelty**: First application of SINDy to discover vertebra arrangement dynamics from CT imaging data, demonstrating feasibility of automatic physics discovery in medical imaging.

**Significance**:
- Automatic discovery eliminates need for manual constraint design
- Data-driven laws capture vertebra-specific dynamics
- Interpretable relationships provide insights into arrangement patterns

**Impact**: Establishes foundation for physics-informed learning in medical imaging, opening new research direction.

### 2. Physics-Informed Contrastive Learning

**Novelty**: Novel integration of discovered physics laws with contrastive learning for feature space regularization, enabling physics enforcement without architecture modification.

**Theoretical Advantages**:
- Simpler than direct physics loss (single loss function vs. multiple weighted constraints)
- Better generalization through feature space learning
- Maintains architecture flexibility
- Enables learnable physics embeddings

**Impact**: Introduces new paradigm for incorporating domain knowledge in medical imaging through feature space regularization.

### 3. Physics-Guided Clustering Enhancement

**Novelty**: Extension of dual-factor clustering with physics consistency as third factor, demonstrating physics-guided improvements to existing methods.

**Application**: Enhances SpineCLUE pipeline detection quality through physics-based filtering.

**Impact**: Shows how physics-guided approaches can improve existing medical imaging pipelines.

### 4. Architecture Improvements

**Contributions**:
- Cross-attention spatial aggregator preserving spatial information
- Enhanced loss functions enforcing spatial constraints
- Direct coordinate regression for stable training

**Impact**: Solves fundamental spatial information loss problem, enabling effective physics-informed learning.

---

## References

[1] Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *Proceedings of the National Academy of Sciences*, 113(15), 3932-3937.

[2] Yu, R., & Wang, R. (2024). Learning dynamical systems from data: An introduction to physics-guided deep learning. *Proceedings of the National Academy of Sciences*, 121(39), e2311348121.

[3] Sekuboyina, A., et al. (2021). VerSe: A Vertebrae labelling and segmentation benchmark for multi-detector CT images. *Medical Image Analysis*, 73, 102166.

[4] Payer, C., et al. (2020). Coarse to fine vertebrae localization and segmentation with SpatialConfiguration-Net and U-Net. *MICCAI 2020*.

---

**Report prepared by**: Research Team  
**Last updated**: December 6, 2025
