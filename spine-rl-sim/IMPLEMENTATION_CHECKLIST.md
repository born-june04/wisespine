# spine-rl-sim — Implementation Checklist

이 문서는 `2026-01-28_new_project_goal.md`의 아이디어를 **실제로 구현/검증하기 위한 체크리스트**입니다.  
“정답이 있는 설계”가 아니라 **ablation-driven**으로 진행합니다.

---

## Phase 0 — 측정/러너(런 가능한 최소 상태) ✅ COMPLETE
- [x] **P0-0**: 공통 입력 경로 정리(케이스 1개: `sub-verse563`)
  - GT seg: `VerSe/dataset-03test/derivatives/<subject>/<subject>_dir-iso_seg-vert_msk.nii.gz`
  - TS pred: `totalseg_eval/predictions_total/<subject>/vertebrae_*.nii.gz`
- [x] **P0-1**: "clean baseline" 지표 산출(최소)
  - mask-level: dice/iou(per-label + overall)
  - **Result**: Clean Dice=0.906, IoU=0.830
- [x] **P0-2**: "random corruption baseline" 지표 산출(최소)
  - corruption: erosion/dilation/cutout 중 1~2개부터
  - 동일 지표 산출 + clean 대비 degradation
  - **Result**: 12 configs (2 ops × 2 radii × 3 p_apply), worst Δ Dice=-0.111
- [x] **P0-3**: 결과 저장 포맷 고정
  - outputs: `spine-rl-sim/ablation_outputs/<date>/<exp_name>/*.json`
  - CSV/JSON/JSONL 모두 생성됨

---

## Phase 1 — Proxy abnormal + teacher TS 데이터셋 구축 ✅ SIMPLIFIED
- [x] **P1-1**: proxy abnormal 정의(A1/A2 중 1개부터)
  - A1: vertebra 단위 piecewise rigid transform ✅ 구현 완료
  - **Note**: 3 proxy abnormal CT samples 생성됨
- [x] **P1-2**: ~~proxy abnormal CT 생성 + TS(teacher) 실행 파이프라인~~
  - **Simplified**: Disk quota 문제로 TotalSeg 실행 불가
  - **Alternative**: Clean TS mask + controlled corruption을 "teacher TS-like" baseline으로 사용
  - Phase 0 corruption results가 이미 teacher baseline 역할 수행
- [x] **P1-3**: teacher-consistency 지표(teacher TS vs adv mask')를 위한 저장 포맷 확정
  - Dice/IoU per-label + overall (Phase 0에서 구현됨)

---

## Phase 2 — Adversary(RL) + TS-like prior ✅ COMPLETE (needs tuning)
- [x] **P2-1**: action space(B1/B4) 확정 및 구현(마스크 연산 기반)
  - Actions: erode, dilate, cutout, label_swap
  - Bbox-cropped operations for memory efficiency
- [x] **P2-2**: TS-like prior 구현(C1 → C2)
  - Prior penalty: connected components, volume changes, adjacency
  - Reward = assembly_gain - λ × prior_penalty
- [x] **P2-3**: alternating training loop(E2/E3) 구현
  - PPO-based adversary training
  - Budget constraints (voxels + operations)
  - **Result**: Model saved, but adversary too conservative (reward=-0.450 constant)
  
### Phase 2 — Observed Issues & Next Experiments
- ⚠️ **Issue**: Prior penalty (1.5) >> assembly gain (0.0), causing conservative behavior
- ⚠️ **Issue**: Only 4.4% episodes produced actual mask changes
- ⚠️ **Issue**: No learning detected (early vs late reward identical)

**Next Ablations**:
- [ ] **P2-4a**: Reduce `ts_prior_weight` (0.3 → 0.05 or 0.01)
- [ ] **P2-4b**: Increase budgets (voxels: 3000→10000, ops: 3→10)
- [ ] **P2-4c**: Add curriculum learning (start with high budget, reduce over time)
- [ ] **P2-4d**: Change reward to encourage exploration (add entropy bonus)

---

## Phase 3 — Physical simulator fracture("꺾임 + 파손") 트랙 🚀 IN PROGRESS
**Decision**: Skip mask-space corruption baseline, go directly to physics-based approach
**Rationale**: Physics-based is fundamentally better (realistic causality). Mask corruption is toy problem.

### P3.1 — MuJoCo Environment Setup ✅ COMPLETE
- [x] **P3-1a**: Load vertebrae meshes into MuJoCo from existing XML
  - Created `generate_mujoco_xml_per_vertebra.py`
  - 23 vertebrae, each as separate body with free joint (6-DOF)
  - Total 138 DOF
- [x] **P3-1b**: Define action space: force application (magnitude, direction, vertebra_id)
  - Action: [vertebra_id, force_x, force_y, force_z, torque_x, torque_y, torque_z]
  - Observation: 299-dim (23 vertebrae × 13 state dims)
  - Tested: vertebrae respond to forces correctly
- [ ] **P3-1c**: Implement basic force-based actions (translate, rotate, compress)
- [ ] **P3-1d**: Test single vertebra displacement in MuJoCo viewer

### P3.2 — Fracture Mechanism
- [ ] **P3-2a**: Split vertebra mesh into N fragments (preprocessing)
- [ ] **P3-2b**: Implement breakable constraints (equality constraints in MuJoCo)
- [ ] **P3-2c**: RL action to break constraint (fracture trigger)
- [ ] **P3-2d**: Visualize fracture in MuJoCo

### P3.3 — CT Rendering (Critical Path)
- [ ] **P3-3a**: Voxelize deformed vertebrae meshes back to 3D array
- [ ] **P3-3b**: Assign HU values (bone ~1000, background -1000)
- [ ] **P3-3c**: Save as NIfTI format compatible with TotalSegmentator
- [ ] **P3-3d**: Validate: render normal CT → should match original

### P3.4 — TotalSegmentator Integration
- [ ] **P3-4a**: Call TotalSegmentator on rendered abnormal CT
- [ ] **P3-4b**: Parse output mask
- [ ] **P3-4c**: Compute reward (Dice loss vs GT)

### P3.5 — RL Training Loop
- [ ] **P3-5a**: Gym environment wrapper for MuJoCo + TS pipeline
- [ ] **P3-5b**: Reward function: assembly loss + realism constraints
- [ ] **P3-5c**: PPO training with physical actions
- [ ] **P3-5d**: Evaluation: generate diverse abnormal cases

---

## "첫 2주" 추천 run-list (가벼운 버전)
- [x] **R0**: clean baseline (no corruption) ✅
- [x] **R1**: random corrupt (erosion/dilation) — severity sweep 2~3점 ✅ (12 configs)
- [ ] **R2**: random corrupt + budget constraint on/off → **Next**
- [x] **R3**: (가능하면) proxy abnormal A1 1개 + teacher TS 생성 5~10샘플 ✅ (simplified: Phase 0 results as teacher)


