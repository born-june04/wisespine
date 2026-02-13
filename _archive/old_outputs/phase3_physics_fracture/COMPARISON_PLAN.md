# Comparison Analysis Plan

## 목표: RL의 가치 증명

### 비교 대상:

1. **Baseline: Random Corruption**
   - Random erosion/dilation
   - Random displacement
   - No learning

2. **Rule-based: Manual Fracture**
   - Fixed displacement (5-20 voxels)
   - Pre-defined pattern
   - Human-designed

3. **Learned: RL Fracture**
   - Optimized displacement
   - Learned pattern
   - Agent-discovered

### 평가 지표:

#### 1. **TS Performance Degradation**
```
Metric: Dice score
Lower = Better (more challenging for TS)

Expected:
- Baseline: Dice = 0.85-0.90 (too easy)
- Manual: Dice = 0.75-0.85 (moderate)
- RL: Dice = 0.70-0.80 (challenging)
```

#### 2. **Realism**
```
Metric: Clinical plausibility
- Displacement magnitude (reasonable?)
- Pattern (realistic?)
- Visual inspection

Goal: RL should find edge cases that are:
- Challenging for TS
- Still plausible/realistic
```

#### 3. **Diversity**
```
Metric: Variance in deformation patterns
- RL should explore diverse strategies
- Not just one type of deformation

Measure:
- Displacement variance
- Pattern clustering
```

---

## 실험 설계:

### Experiment 1: Dice Score Comparison
```python
Methods:
1. Load trained RL agent
2. Generate N=50 fractured CTs
3. Run TotalSegmentator on all
4. Compare Dice scores

Hypothesis: RL < Manual < Baseline
```

### Experiment 2: Assembly Robustness
```python
Setup:
1. Train assembly with Random corruption
2. Train assembly with RL corruption
3. Test on held-out abnormal cases

Hypothesis: RL-trained assembly performs better
```

### Experiment 3: Ablation Study
```python
Compare:
- RL with physics (current)
- RL without physics (mask-space only)
- Rule-based physics
- Rule-based mask-space

Question: Does physics help?
```

---

## 논문 Story:

### Option A: "Physics는 중요하지 않음"
```
Finding: Mask-space corruption도 충분함
Contribution: Efficient adversarial training
```

### Option B: "Physics가 중요함"
```
Finding: Physics-based가 더 realistic
Contribution: Novel physics-based adversary
```

### Option C: "RL이 중요함"
```
Finding: RL이 optimal deformation 발견
Contribution: Automatic hard example mining
```

---

## 재프레이밍:

### 현재 제목 (X):
"Physics-based RL for Vertebra Fracture Simulation"

### 더 나은 제목 (O):
"Adversarial RL for Robust Spine Assembly: Learning to Generate Challenging Segmentation Failures"

Focus:
- Assembly robustness (main goal)
- RL adversary (method)
- Segmentation failures (attack target)
- NOT "fracture simulation"

