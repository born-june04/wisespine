# Spine RL Sim — New Project Goal (2026-01-28)

## 요약 (한 문장)
**TS는 고정**한 채로, 학습 단계에서 **adversarial agent(RL)** 가 “그럴듯하지만 어려운” 마스크/중간표현 교란을 만들어내고, **assembly 모듈이 그 입력에서도 안정적으로 mesh를 복원**하도록 **min–max(제로섬에 가까운) 학습 루프**를 설계한다.

---

## 배경 / 문제의식
- 현재 TS는 “정상(또는 제한된 분포)” 데이터에서만 안정적으로 마스크를 생성하는 경향이 있음.
- Abnormal(수술/변형/골절/스콜리오시스 등)에서 TS가 어떤 오류를 낼지, 어떤 오류가 assembly를 얼마나 망가뜨릴지 **사전에 규칙으로 다 쓰기 어렵다**.
- 목표는 “TS를 당장 바꾸기”가 아니라, **TS 출력이 불완전/노이즈여도 downstream assembly가 망가지지 않게** 만드는 것.

---

## 제안하는 핵심 아이디어: Adversarial Corruption Game for Assembly Robustness
### 구성요소
- **TS (고정, Frozen)**: CT → mask (혹은 multi-label mask)
- **Adversary (RL policy, 학습됨)**: mask → mask' (교란/변형/오류 주입)
- **Assembly (학습됨)**: mask' → mesh (혹은 mesh + 메타데이터)
- **Evaluator / Loss**: mesh vs GT mesh (또는 GT label/landmark)

### 목적함수(개념)
\[
\min_{\theta\_{\text{assembly}}}\; \max_{\pi\_{\text{adv}}}\; \mathcal{L}(\text{assembly}(T_{\pi}(\text{mask})),\ \text{GT})
\]
- **Assembly**는 loss를 줄이려 하고,
- **Adversary**는 “그럴듯한 범위 안에서” loss를 키우는 교란을 찾는다.

---

## “이게 RL이냐?”에 대한 정리
### RL이 맞는 경우
- Adversary가 **행동(action) 시퀀스**로 교란을 단계적으로 적용하고,
- 환경 상태(state)로 현재 mask/메트릭/제약 위반 정도를 보고,
- reward로 “assembly 실패 유도(+제약 페널티)”를 받으며 정책을 학습하면,
→ 이건 명확히 **RL 기반 adversarial data generation**이다.

### RL이 아닐 수도 있는 경우
- 교란이 단순히 “랜덤 augmentation”으로 고정되어 있고 학습/정책이 없다면,
→ 그건 그냥 **데이터 증강(augmentation)** 이다.

### CT 이미지를 생성하는 건 또 다른 레벨
- CT를 “새로 합성”하는 건 보통 **image generation**(diffusion/GAN 등) 또는 **physics+rendering** 문제로 넘어가며,
→ 여기서 제안하는 1차 목표는 **CT 생성이 아니라 mask/중간표현 교란**이다.

---

## 왜 이게 좋은가 (기대효과)
- **TS를 건드리지 않고도**: downstream(assembly) 강건성 개선 가능.
- “사람이 생각 못 한 실패 모드”를 adversary가 자동으로 찾아낼 수 있음.
- 임상 abnormal CT가 당장 없더라도, “현실적인 오류 분포”를 모사/탐색하는 방향으로 시작 가능.

---

## Adversary가 ‘어디’를 건드릴지: 단계별 옵션
### Stage 1 (가장 현실적/빠름): Mask-space corruption
TS mask(또는 라벨맵)에 아래와 같은 교란을 RL action으로 정의:
- **과소/과대 분할**: erosion/dilation (라벨별 또는 ROI별)
- **구멍/결손**: cutout/holes
- **연결 오류**: 인접 분절 붙이기/떼기(bridging / separation)
- **부분 시야**: crop / FOV 제한
- **라벨 스왑(레벨 오류)**: T12↔L1 같은 경계에서 swap/shift

> 장점: 구현 쉬움, TS 고정 유지, assembly 취약점 탐색에 직결  
> 단점: “실제 CT가 abnormal이라서 TS가 실패”의 원인을 직접 모델링하진 않음

### Stage 2: Mesh/landmark-space corruption (assembly 입력이 mesh/point 기반일 때)
- mask→mesh 파이프라인이 있다면, mesh/landmark를 변형해 assembly robustness를 강화.

### Stage 3 (장기): CT-space 변형/렌더링
- “CT 생성”이 아니라, 정상 CT에 대해 **물리/기하학적 워핑**으로 abnormal을 모사하는 쪽부터 시작 가능:
  - piecewise rigid warp(척추 분절 단위), local deformation field 등
- 이후에야 diffusion/GAN 같은 “진짜 image generation”으로 확장 가능.

---

## 제약(핵심): Adversary가 ‘말도 안 되는 파괴’로만 이기지 못하게
Adversary reward에 아래를 강하게 포함:
- **Plausibility penalty**:
  - 라벨 connected component 수 제한
  - 척추 순서/인접 관계 유지(너무 멀어지면 패널티)
  - 볼륨/두께 범위 유지(너무 얇거나 커지면 패널티)
- **Budget constraint**:
  - 한 에피소드에서 수정 가능한 voxel 수/연산 횟수 제한
  - action magnitude 제한

이 제약이 없으면 adversary는 “완전 랜덤/완전 삭제”로 loss를 키우고 끝나서 학습이 무의미해질 수 있음.

---

## 보상/손실 설계(예시)
### Assembly loss (minimize)
- mesh 품질:
  - Chamfer distance / point-to-surface distance
  - per-vertebra surface Dice(가능하다면)
- 구조 규칙:
  - 인접 척추 간 간격/정렬 regularization
  - self-intersection / topology penalty(가능하면)

### Adversary reward (maximize)
- \(+\) assembly loss 증가
- \(-\) plausibility/budget 위반 패널티

---

## 학습 루프(권장 패턴)
1) **Warm-up**: 랜덤/규칙 기반 corruption으로 assembly를 먼저 안정화
2) **Alternating updates**:
   - assembly 고정 → adversary 업데이트(최악 케이스 찾기)
   - adversary 고정(또는 replay buffer) → assembly 업데이트(강건화)
3) **Curriculum**:
   - 쉬운 교란부터 시작해 점진적으로 난이도/범위를 확대

---

## 평가 전략(성공 기준)
단순히 train loss가 아니라:
- **Clean TS mask** 입력에서도 성능 유지(= degradation 없음)
- **Realistic corruption set**에서 성능 향상
- **Adversary-generated hard set**에서 성능 향상
- 실패 모드 분석: 어떤 교란이 assembly를 망가뜨렸는지 리포팅

---

## 현재 spine-rl-sim과의 관계
- 현재 `spine-rl-sim`은 “freejoint pose를 틀어놓고 복원”하는 **kinematic RL toy env**.
- 새 목표는 “mask→mesh 파이프라인의 강건성”을 위한 **adversarial augmentation RL**로, 환경/state/action/reward가 다름.
- 다만 공통점:
  - “에이전트가 어려운 케이스를 생성/또는 해결”이라는 학습 패턴은 동일.

---

## 오픈 질문 (다음 대화에서 정해야 할 것)
1) Assembly 모듈의 **입력/출력 정의**:
   - 입력이 TS 라벨맵인지, binary mask인지, vertebra별 mask인지?
   - 출력이 OBJ mesh인지, point cloud인지, pose/landmark인지?
2) GT는 무엇으로 둘지:
   - GT mesh를 어떻게 만들지(현재 exporter 기반 가능)
3) 가장 현실적인 “abnormal proxy”는 무엇인지:
   - pose fracture(분절 rigid transform) vs mask morphology vs 둘의 조합
4) 제약(penalty)을 어떻게 두어 “그럴듯한 어려움”을 만들지:
   - 예: 라벨 순서 유지 / connectedness / volume bounds / adjacency bounds

---

## 제안하는 1차 MVP (가장 빨리 가치 확인)
**Mask-space adversary + assembly robustness**로 시작:
- TS 출력 마스크를 입력으로 받고,
- Adversary(RL)가 제한된 연산(erosion/dilation/cutout/label-swap 등)으로 mask'를 만들고,
- Assembly가 mesh 복원을 수행,
- “clean vs corrupted vs adversarial” 3종 평가로 강건성 개선을 확인.

---

## 합의된 진행 방향 업데이트 (중요)
### 목표 재정의
“아무 노이즈”가 아니라 **abnormal CT가 들어왔을 때 TS가 실제로 낼 법한 실패 마스크**를 모사하는 것이 핵심이다.

### 큰 흐름 (2-stage)
#### Stage A: TS-like prior(teacher) 만들기 — RL이 “무엇을 모사해야 하는지” 신호 제공
CT 자체를 ‘생성’하는 문제로 가지 않고(당장은), **proxy abnormal 변형**을 통해 TS 실패 패턴을 수집한다.

- **입력**: 정상 CT (또는 정상 분포에 가까운 CT)
- **Proxy abnormal 변형** (비생성, 제한된 변형/워핑):
  - 분절 단위 piecewise rigid transform(vertebra 단위 이동/회전)
  - 부분 결손/occlusion(시야 제한, crop)
  - 간단 artifact(노이즈, bias, streak-like mask/ROI 수준의 교란부터)
- **Teacher 실행**: 변형된 CT에 **TS를 실제로 돌려** TS 출력 마스크를 얻는다.
- **결과**: “(proxy abnormal CT) → (TS mask)” 데이터셋 = TS failure 분포의 관측치

> 목적: RL이 “TS가 abnormal에서 어떤 실패를 내는지”를 예측하려면 최소한의 supervision/score가 필요함.  
> 여기서 teacher는 TS 자체이고, CT는 “생성”이 아니라 “제한된 변형”으로만 접근한다.

#### Stage B: Adversarial game로 assembly robust 학습
- **Adversary(RL)**: 정상 TS mask를 입력받아, “teacher TS(mask | proxy abnormal)”와 **분포가 비슷**하면서도 assembly를 망가뜨리는 mask'를 생성
- **Assembly**: mask'에서도 mesh 복원이 무너지지 않게 학습

즉, reward는 2항:
- \(+\) assembly 실패 유도 (mesh loss 증가)
- \(-\) TS-like 위반 (teacher 분포에서 너무 벗어나면 패널티)

---

## Evaluation Metrics (필수 지표 세트)
아래는 “우리가 진짜로 원하는 게 개선되었는지”를 확인하기 위한 최소 지표들이다.

### 1) TS-like prior / Adversary realism (TS 실패 분포를 잘 모사하는가?)
**목표**: adversary가 만든 mask'가 “그럴듯한 TS-failure”인지, 혹은 말도 안 되게 파괴적인지 구분.

- **Teacher-consistency (proxy abnormal 기준)**:
  - **Dice / IoU**: mask' vs teacher TS mask (proxy abnormal CT에서 나온 TS 출력)
  - **Boundary F-score (BFScore)** 또는 **surface distance**(mask 경계 기반): TS 오류는 경계에서 치명적이라 경계 지표가 중요
  - **Per-label confusion**: vertebra label swap/shift가 얼마나 잘 재현되는지 (예: T12↔L1)
- **Plausibility / topology sanity** (제약 위반율을 수치화):
  - **Connected components count** (라벨별): 과도한 분해/붙임 감지
  - **Volume change ratio** (라벨별): \(\frac{|mask'|}{|mask|}\) 분포가 현실 범위인지
  - **Adjacency violations**: 인접 척추가 비정상적으로 겹치거나 과도히 멀어진 정도(centroid distance 기반)
- **Budget usage**:
  - 에피소드당 변경 voxel 수 / 적용 연산 횟수 / action magnitude 통계

> 핵심: “assembly를 망가뜨렸냐”만 보면 adversary가 비현실적으로 이기기 쉬움. realism/제약 지표가 꼭 필요.

### 2) Assembly mesh quality (복원 자체가 좋아졌는가?)
**목표**: mask 입력 오류가 있어도 최종 mesh가 GT에 가깝게 나오는지.

- **Geometry distance**:
  - **Chamfer distance** (mesh↔GT mesh 또는 point-sampled surfaces)
  - **Hausdorff distance**(가능하면) / percentile surface distance(95%)
  - **Normal consistency**(가능하면)
- **Per-vertebra**:
  - 라벨별 Chamfer/IoU (특히 경계 vertebra T12/L1 등)
  - **Centroid error** (GT와의 중심 오차)
  - **Volume error**(mesh volume vs GT)
- **Failure rate**:
  - marching cubes 실패/빈 mesh 비율
  - self-intersection/비정상 topology(가능하면 간단한 check)

### 3) Downstream alignment / clinical-ish proxy (선택, 하지만 강력)
**목표**: “형상은 그럴듯한데 정렬이 틀림”을 잡기 위한 지표.

- **Cobb angle error**: \( |Cobb(mesh) - Cobb(GT)| \) 또는 \( |Cobb(mesh) - Cobb(target)| \)
- **Spine curve smoothness**: vertebra centroid spline curvature/second-derivative penalty
- **Relative ordering consistency**: cranio-caudal ordering axis에서 label order 유지율

### 4) End-to-end robustness summary (한 장표로 보여주는 핵심)
**목표**: clean 성능을 유지하면서, corruption/adversarial에서 얼마나 덜 무너지는지.

- **Clean vs Random-corrupt vs Adversarial** 3-way 성능:
  - (clean) mesh metric
  - (random) mesh metric
  - (adv) mesh metric
- **Robustness curve**:
  - corruption severity(예: erosion radius/label swap rate/occlusion %)에 따른 성능 곡선
  - “worst-k” 성능(adv 샘플 중 최악 상위 k% 평균)
- **Regression guardrail**:
  - clean 성능이 baseline 대비 얼마나 떨어졌는지(= degradation 방지)

---

## 실험 리포트에서 최소로 보고해야 할 것(추천)
- Clean 성능(기준)
- Random corrupt에서 성능
- Adversarial corrupt에서 성능
- Adversary realism 점수(teacher-consistency + plausibility violation rate)
- “어떤 실패 모드가 줄었는지” qualitative 사례(가장 중요한 3~5개)

---

## Ablation Plan (실험으로만 답이 나오는 것들)
이 프로젝트는 “정답 설계”가 아니라 **가설 검증**이 핵심이므로, 아래 항목들은 모두 **ablation으로 명시**하고 결과로 결정한다.

### 공통 전제(모든 ablation에서 고정)
- **TS는 frozen** (가중치 업데이트 없음)
- 평가 셋은 최소 3종:
  - **Clean**: 정상 CT에서 TS mask
  - **Proxy abnormal + teacher**: proxy abnormal CT에 TS를 돌린 teacher TS mask
  - **Adversary**: 정상 TS mask에서 adversary가 만든 mask'
- 모든 실험은 동일한 seed set으로 반복(예: 3 seeds)하여 평균±표준편차 보고

### Ablation A: Proxy abnormal 정의(teacher TS-failure 분포의 “원천”)
**가설**: 어떤 proxy abnormal이 “현실적인 TS failure”를 가장 잘 재현하고, assembly robustness에 실제 도움이 되는가?

- **A1 (Rigid)**: vertebra 단위 piecewise rigid transform (translation/rotation 범위)
- **A2 (Occlusion/FOV)**: crop / partial volume (z-truncation, limited FOV)
- **A3 (Artifact-lite)**: noise/bias 같은 약한 intensity 변형(가능하면), 또는 ROI 기반 mask-level artifact proxy
- **A4 (Mix curriculum)**: A1→A2→A3 순서로 난이도 증가

**평가(주요 지표)**:
- teacher-consistency: (proxy abnormal) TS mask 통계 분포(라벨별 volume/CC/경계 지표)
- assembly robustness: clean 대비 degradation 없이, proxy abnormal teacher 기준 mesh metric 개선 여부

### Ablation B: Adversary action space(어떤 교란 연산을 허용할지)
**가설**: adversary가 허용된 조작 집합에 따라 “현실적인 hard case”를 찾는 능력과 학습 안정성이 달라진다.

- **B1 (Morphology)**: erosion/dilation (라벨별, 반경/횟수 제한)
- **B2 (Topology edits)**: hole/cutout/bridging/separation
- **B3 (Label ops)**: label swap/shift(특히 경계 T12↔L1 등)
- **B4 (Mixed)**: B1+B2+B3
- **B5 (Budgeted Mixed)**: B4에 변경 voxel budget 강제(예: ≤1% voxels)

**평가(주요 지표)**:
- adversary realism: plausibility violation rate(연결성/볼륨/adjacency)
- end-to-end: adversarial set에서 mesh metric(worst-k 포함) 개선 여부

### Ablation C: TS-like prior의 형태(“TS처럼 보이게” 강제하는 방법)
**가설**: TS-like prior가 없으면 adversary가 비현실적 파괴로 이기고, prior 형태에 따라 안정성과 성능이 달라진다.

- **C0 (None)**: prior 없이 adversary는 assembly loss만 최대화 (baseline failure mode 확인용)
- **C1 (Handcrafted constraints only)**: CC/volume/adjacency/budget 제약만 사용
- **C2 (Teacher-consistency loss)**: proxy abnormal에서 얻은 teacher TS mask와의 유사도 항 추가
- **C3 (Learned discriminator/energy)**: (teacher TS vs non-TS) 판별기 점수로 prior 구성 (가능할 때)

**평가(주요 지표)**:
- TS-like 점수: (mask' vs teacher TS mask) Dice/Boundary 지표
- 안정성: 학습 붕괴/모드콜랩스 빈도, 제약 위반율

### Ablation D: Reward/penalty 가중치(λ 스윕)
**가설**: 현실성(prior/제약)과 난이도(assembly loss) 사이의 trade-off 최적점이 존재한다.

- D1: λ\_prior ∈ {0.1, 0.3, 1.0}
- D2: λ\_plausibility ∈ {0.1, 0.3, 1.0}
- D3: budget penalty on/off

**평가(주요 지표)**:
- clean 성능 유지(guardrail) + adversarial 성능 개선(worst-k) 동시에 만족하는 구간 탐색

### Ablation E: Training schedule(학습 순서/빈도)
**가설**: alternating 주기/비율에 따라 min–max 학습 안정성이 크게 달라진다.

- **E1 (Warm-up only)**: random corrupt로만 assembly 학습(adv 없음)
- **E2 (Alt 1:1)**: adv 1 epoch ↔ assembly 1 epoch
- **E3 (Alt 5:1)**: adv 1 ↔ assembly 5 (assembly 안정 우선)
- **E4 (Replay buffer)**: adv 샘플 버퍼에서 샘플링하여 assembly 학습(adv는 간헐 업데이트)

**평가(주요 지표)**:
- 학습 안정성(성공적으로 수렴하는 비율)
- 성능(3-way clean/random/adv)

### Ablation F: Curriculum(난이도 스케줄)
**가설**: corruption severity를 점진적으로 올리면 학습이 안정되고, 최종 robust 성능이 좋아진다.

- **F1 (No curriculum)**: 처음부터 max severity
- **F2 (Linear)**: severity 선형 증가
- **F3 (Adaptive)**: assembly 성능이 threshold 넘으면 severity 증가

**평가(주요 지표)**:
- 수렴 속도/안정성
- 최종 worst-k 성능

### Ablation G: Physical simulator fracture 모델링(“꺾임 + 파손”을 물리적으로)
**가설**: fracture를 “정렬/꺾임(buckling)” + “조각 분리(breakage)”의 2-레이어로 모델링하면,
현실성이 올라가면서도 학습 가능한 난이도/속도를 유지할 수 있다.

#### G1: 조각(piece) 개수/분할 방식
> 현실의 미세조각을 그대로 다 넣기보단, **거친 조각(2~6)** + 랜덤화로 시작하고 점진적으로 확장.

- **G1-1 (Fixed 2-piece)**: 항상 2조각 (최소 복잡도)
- **G1-2 (Fixed 4-piece)**: 항상 4조각
- **G1-3 (Random pieces)**: 에피소드마다 조각 수 랜덤(예: 2~6)  
  - 구현 권장: **분할 버전별 MJCF/XML을 여러 개** 만들어두고 `reset()`에서 **모델 리로드로 랜덤 선택**
- **G1-4 (Coarse + jitter)**: 조각 수는 작게 유지하되, break threshold/접촉 파라미터/초기 균열 위치를 랜덤화(“미세조각 효과” 근사)

**평가(주요 지표)**:
- 시뮬 속도(steps/sec), 안정성(폭발/NaN/접촉 발산율)
- 파손 이벤트 분포(발생률, piece 분리 정도)

#### G2: 외력(action) 인터페이스 — “body에 force/torque 주입”
> 사용자 결정: **body wrench**(힘+토크)로 하되, 안전 제한과 에너지 패널티로 현실성을 유지.

- **G2-1 (Force only)**: 3D force만
- **G2-2 (Wrench)**: 3D force + 3D torque (권장 baseline)
- **G2-3 (Single-body vs multi-body)**:
  - 한 번에 하나의 target body에만 적용(현실적 제약)
  - 혹은 여러 body에 분산 적용(학습 난이도↓, 현실성↓)

**평가(주요 지표)**:
- 성공률 vs 파손률 trade-off
- 최대 힘/토크, 누적 work(에너지) 통계

#### G3: Break 이벤트(파손 트리거) 정의
**가설**: “언제 부러졌다고 할지”의 정의가 학습 신호/현실성에 매우 큰 영향을 준다.

- **G3-1 (Force/impulse threshold)**: 접촉력/스프링력/impulse가 임계치 초과 시 break
- **G3-2 (Relative displacement/angle threshold)**: 조각 간 상대 변위/회전이 임계치 초과 시 break
- **G3-3 (Cumulative work threshold)**: 누적 work(∑ F·dx)가 임계치 초과 시 break
- **G3-4 (Hybrid)**: (G3-1 + G3-2) 또는 (G3-2 + G3-3)

**평가(주요 지표)**:
- 파손 이벤트의 재현성/안정성(임계치 스윕에 따른 발생률 곡선)
- “비현실적 파손”(너무 쉽게/너무 자주) 억제 여부

#### G4: Objective(목표) — 파손 이후에 무엇을 최적화할지
**가설**: “정렬 복원”과 “파손 최소화”의 다목적 균형이 실제 사용 시나리오에 가까운 신호를 만든다.

- **G4-1 (Alignment-only)**: Cobb/정렬만 최소화 (파손 패널티 약함)
- **G4-2 (Safety-first)**: 파손 이벤트를 강하게 패널티(회피 학습)
- **G4-3 (Trade-off curve)**: 파손 패널티 가중치 스윕으로 Pareto frontier 산출

**평가(주요 지표)**:
- Cobb error/정렬 지표
- 파손률/분리량/최대 하중

---

## 최소 실험 매트릭스(권장 “첫 2주” 플랜)
전체 조합은 폭발하므로, 아래처럼 **우선순위가 높은 축만** 얕게 스윕한다.

### Phase 0: Baselines(무조건)
- P0-1: clean only (no corrupt)
- P0-2: random corrupt only (handcrafted)
- P0-3: adversary without TS-like prior (C0) → “파괴적 이김” 재현(진단용)

### Phase 1: “가치 검증” 최소 ablation
- Proxy abnormal: A1 vs A2
- Action space: B1 vs B4
- Prior: C1 vs C2

→ 총 2×2×2=8개 + baseline 3개 = 11개 (seed 3이면 33 runs)

### Phase 2: 안정화/성능 최대화
- D(λ) 스윕 + E(schedule) + F(curriculum)에서 각각 2~3개만 추가

---

## 결과 해석 규칙(미리 정해두기)
실험 후 논쟁을 줄이기 위해, “무엇이 더 낫다”의 판정 기준을 미리 고정한다.

- **1순위**: clean 성능이 baseline 대비 크게 악화되면 탈락 (guardrail)
- **2순위**: adversarial worst-k(예: 최악 10%)에서 mesh metric 개선
- **3순위**: teacher-consistency + plausibility violation rate가 허용 범위 내(“그럴듯함” 유지)
- **4순위**: 학습 안정성/재현성(3 seeds에서 분산이 작은 설정 우선)

---

## 다음 대화에서 결정/정리하면 좋은 것들(질문 리스트)
지금 단계에서 “답이 정해진 것”이 아니라, 설계/실험을 위해 합의를 하면 진행이 빨라지는 질문들:

1) **관측(observation)**: 에이전트가 무엇을 보나?
   - vertebra pose(oracle) vs mesh/point features vs mask-derived features
   - 관측 노이즈/부분관측을 넣을지(현실성)
2) **행동(action) 적용 프레임**: force/torque를 world frame으로 줄지, body frame으로 줄지
3) **접촉/충돌 모델링**: collisions on/off, friction 계수 범위, self-collision 허용 여부
4) **조각(piece) 분할 생성 방식**: 수동 분할 vs 자동 분할(클러스터링/plane cut), 어디까지 랜덤화할지
5) **학습 목표의 “실제 활용” 정의**:
   - 교정(align) 정책이 필요한가? (수술/로봇 조작)
   - 아니면 “hard-case 생성기”가 필요한가? (강건성 학습)
6) **평가에서의 ‘현실성’ 기준**: 무엇을 만족해야 “그럴듯한 fracture”라고 할지(제약/임계치/통계)

---

## 추가 Ablation 축 (H ~ N): 설계 답이 정해지지 않은 핵심 변수들
아래 항목들은 “좋아 보이는 직관”만으로 결정하기 어렵고, 실제로는 학습 안정성/일반화/현실성에 큰 영향을 준다.

### Ablation H: Observation design (무엇을 관측으로 줄 것인가?)
**가설**: oracle pose를 주면 너무 쉬워지고, 간접 관측(마스크/랜드마크/센서)로 바꾸면 현실성이 증가하지만 학습이 어려워진다.

- **H1 (Oracle pose)**: (fractured body들의) pos+quat 직접 제공
- **H2 (Mask-derived features)**: TS mask 또는 mask'에서 추출한 요약량 제공
  - 예: 라벨별 centroid(3), PCA axis, bbox size, volume, adjacency 거리
- **H3 (Mesh/point features)**: 현재 mesh에서 샘플 포인트/centroid curve 특징 제공
- **H4 (Partial observability)**: 일부 vertebra만 관측 / 노이즈 추가 / 관측 지연

**평가(주요 지표)**:
- 학습 성공률/수렴 시간
- clean↔corrupt 일반화(관측 노이즈에 대한 강건성)

### Ablation I: Action parameterization (행동을 어떻게 표현/제약할 것인가?)
**가설**: action이 너무 직접적이면 쉬워지고, 너무 간접적이면 학습이 불안정해진다. body wrench는 좋지만 적용 방식이 중요.

- **I1 (Force only)**: 3D force
- **I2 (Wrench)**: 3D force + 3D torque (baseline)
- **I3 (Frame)**:
  - world frame 적용 vs body frame 적용
- **I4 (Targeting)**:
  - single target body per step vs multi-body simultaneous
- **I5 (Temporal smoothing)**:
  - raw action vs low-pass filtered action(액추에이터/손의 관성 근사)

**평가(주요 지표)**:
- 최대 힘/토크 및 누적 work
- 안정성(발산/NaN)
- 목표 달성률 및 파손률 trade-off

### Ablation J: Contact/Collision model (접촉을 어떻게 둘 것인가?)
**가설**: self-collision/마찰/충돌 파라미터는 “현실감”과 “학습 가능성” 모두에 크게 영향.

- **J1 (No collision)**: contype/conaffinity=0 (시각화/가벼운 디버깅용)
- **J2 (Self-collision on)**: vertebra 간 충돌 활성
- **J3 (Friction sweep)**: μ ∈ {0.2, 0.5, 0.8}
- **J4 (Soft contact params)**: solref/solimp 스윕(접촉 강성/감쇠)
- **J5 (Ground/fixture)**: 지면/지지대 유무, 경계조건 랜덤화

**평가(주요 지표)**:
- steps/sec, contact solver 안정성
- 교정 성공률 vs 과도한 접촉력/파손률

### Ablation K: Domain randomization (무엇을 랜덤화할 것인가?)
**가설**: 적절한 랜덤화는 일반화에 도움이 되지만, 과도하면 학습이 안 된다.

- **K1 (No randomization)**: baseline
- **K2 (Material randomization)**: friction, damping, stiffness 랜덤
- **K3 (Break threshold randomization)**: piecewise fracture 임계치 랜덤
- **K4 (Morphology randomization)**: 조각 분할 버전 랜덤(2~6), 초기 crack 위치 랜덤
- **K5 (Sensor noise)**: 관측 노이즈/지연 랜덤

**평가(주요 지표)**:
- unseen 설정에서 성공률(holdout random seeds)
- robustness curve(난이도/랜덤화 강도별)

### Ablation L: Reward shaping / multi-objective trade-offs (보상을 어떻게 줄 것인가?)
**가설**: alignment vs safety(파손/힘 제한) vs smoothness를 어떻게 섞느냐가 정책의 성격을 결정한다.

- **L1 (Sparse success)**: 성공/실패만 보상(학습 난이도↑)
- **L2 (Dense alignment)**: Cobb/pose error 기반 연속 보상(학습 쉬움)
- **L3 (Safety penalty)**: 파손 이벤트/과도 힘/토크에 큰 패널티
- **L4 (Energy/work penalty)**: 누적 work 최소화(현실성)
- **L5 (Pareto sweep)**: (alignment, safety) 가중치 스윕으로 frontier 작성

**평가(주요 지표)**:
- 성공률/파손률/에너지/시간(steps) 4축 trade-off
- worst-k 리스크(최악 케이스에서 파손률)

### Ablation M: RL algorithm / training recipe (학습 레시피)
**가설**: 문제 난이도(접촉/파손/부분관측)에 따라 PPO만으로는 부족할 수 있고, off-policy나 SAC류가 유리할 수 있다.

- **M1 (PPO baseline)**: 현재 recipe
- **M2 (SAC/TD3)**: continuous control에 강한 off-policy
- **M3 (Recurrent policy)**: partial observability(H4)일 때 RNN/GRU policy
- **M4 (Curriculum + buffer)**: 난이도 스케줄 + hard-case replay

**평가(주요 지표)**:
- sample efficiency(동일 step에서 성능)
- 안정성(학습 붕괴 빈도)
- 일반화(랜덤화/노이즈에서 성능 유지)

### Ablation N: Evaluation protocol / splits (평가 설계 자체)
**가설**: "무엇을 holdout으로 두느냐"에 따라 개선이 착시일 수 있으므로, 평가 프로토콜 자체를 ablation으로 명시해야 한다.

- **N1 (Holdout by seed)**: 동일 케이스, 다른 랜덤 시드
- **N2 (Holdout by severity)**: 더 강한 fracture/접촉 난이도
- **N3 (Holdout by morphology)**: 다른 piece 분할(2-piece로 학습, 4-piece로 평가 등)
- **N4 (Holdout by subject)**: sub-verseXXX 다른 환자(장기 목표)

**평가(주요 지표)**:
- 성공률/파손률/에너지의 분포(평균뿐 아니라 quantile)
- worst-k(최악 5~10%) 성능 보고(안전 관점)

---

## Phase 4: Surgical Artifacts (2026-02-03 Update)

### 새로운 방향: Clinical Reality로 Pivot
**핵심 발견**: Phase 3 PyBullet RL 결과 분석 후, **순수 fracture보다 surgical artifacts가 더 중요한 abnormality**임을 확인.

### 실제 Clinical Abnormal 정의
척추 수술 환자에서 흔한 abnormality:

1. **Pedicle screws** (가장 흔함!)
   - Metal hardware로 인한 CT artifact
   - Segmentation 완전 실패 원인
   
2. **Metal rods/plates**
   - Spinal fusion instrumentation
   - Streak artifacts, blooming

3. **Bone cement** (Vertebroplasty/Kyphoplasty)
   - HU 값 변화
   - Vertebra boundary 불명확

4. **Cage implants** (Fusion)
   - Metal/PEEK material
   - Artifact 심각

### Phase 4 목표: Surgical Artifact Simulation
```
Normal CT → Add surgical hardware → Artifact simulation → TS failure → Assembly robustness
```

**장점**:
- 훨씬 더 realistic
- Clinical relevance 명확
- TotalSegmentator 실제 failure mode
- 선행연구: MICCAI 2024 Challenge (Dice 0.65 vs 0.95)

### 구현 계획:
1. **Implant modeling**
   - Pedicle screw 3D models
   - Rod/plate geometry
   - Cement injection patterns

2. **Metal artifact synthesis**
   - Streak artifact rendering
   - Blooming effects
   - HU value corruption

3. **RL adversary for artifact**
   - Learn optimal implant placement
   - Maximize TS failure
   - While maintaining clinical plausibility

4. **Assembly robustness training**
   - Same framework as Phase 3
   - But with surgical artifacts as corruption

### Literature Grounding:
- **"Metal Artifact Reduction in CT"** (Radiology 2023)
- **"Segmentation of Spine with Surgical Hardware"** (MICCAI 2024)
- AO Spine surgical standards

### Why This is Better:
- ❌ Phase 3: Fracture simulation (Dice degradation 0.33-1.0%, too weak)
- ✅ Phase 4: Surgical artifacts (Expected Dice degradation 20-30%, clinically meaningful!)

---

## 현재 상태 요약 (2026-02-03)

### Completed:
- ✅ Phase 0: Baselines (mask corruption, Dice measurement)
- ✅ Phase 1: Proxy abnormal (simplified to mask corruption)
- ✅ Phase 2: Mask-space adversary (RL training working)
- ✅ Phase 3: Physics-based fracture (PyBullet RL complete, but weak results)

### In Progress:
- 🔄 Phase 4: Surgical artifact simulation (NEW DIRECTION!)

### Key Findings:
1. PyBullet fracture works but degradation too weak (0.33%)
2. Manual fracture slightly better (1.0%) but still insufficient
3. **Surgical artifacts are the real target** (expected 20-30% degradation)
4. Need to pivot to clinically-relevant abnormalities

### Next Steps:
1. Literature review on surgical artifact simulation
2. Collect/model pedicle screw geometries
3. Implement metal artifact rendering
4. Adapt RL framework for surgical artifact adversary
5. Validate on real surgical CT cases


