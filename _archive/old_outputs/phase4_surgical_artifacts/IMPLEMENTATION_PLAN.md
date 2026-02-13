# Phase 4: Surgical Artifacts - Implementation Plan

**Start Date**: 2026-02-03  
**Goal**: Physics-based surgical artifact simulation for robust spine assembly

---

## 🎯 Research Goal

**Generate realistic surgical hardware artifacts in CT images to train robust segmentation/assembly systems**

### Why Surgical Artifacts?
- Phase 3 fracture: Only 0.33-1.0% Dice degradation (too weak!)
- Real surgical CTs: 20-30% Dice degradation (MICCAI 2024)
- Clinical relevance: Millions of spine surgery patients worldwide
- **This is the REAL challenge for medical AI!**

---

## 📚 Step 1: Literature Review & Data Collection (Week 1)

### 1.1 Key Papers to Read
```
Priority 1 (Must Read):
- [ ] "Metal Artifact Reduction in CT" (Radiology 2023)
- [ ] "Segmentation of Spine with Surgical Hardware" (MICCAI 2024)
- [ ] "Deep Learning for Metal Artifact Reduction" (IEEE TMI)
- [ ] AO Spine Surgical Standards

Priority 2 (Background):
- [ ] Pedicle screw placement guidelines
- [ ] CT physics: metal artifacts
- [ ] Existing artifact simulation methods
```

### 1.2 Surgical Hardware Types
```
1. Pedicle Screws (Most Common!)
   - Diameter: 4-8mm
   - Length: 30-55mm
   - Material: Titanium/Stainless Steel
   - Placement: Through pedicle into vertebral body

2. Rods
   - Diameter: 5-6mm
   - Length: Variable (connects multiple vertebrae)
   - Material: Titanium/Cobalt-Chromium

3. Bone Cement
   - Vertebroplasty/Kyphoplasty
   - PMMA (Polymethylmethacrylate)
   - HU value: ~800-1200

4. Cage Implants
   - Fusion cages
   - Material: PEEK/Titanium
   - Size: Varies by vertebra level
```

### 1.3 CT Artifact Types
```
Metal Artifacts in CT:
1. Streak Artifacts
   - Radial streaks from metal objects
   - Photon starvation, beam hardening
   
2. Blooming (Volume Averaging)
   - Metal appears larger than actual size
   - Obscures adjacent structures
   
3. HU Value Corruption
   - Surrounding tissue HU changes
   - Can be positive or negative shifts
   
4. Edge Enhancement
   - High-frequency noise at metal edges
```

---

## 🔧 Step 2: Implant Geometry Modeling (Week 1-2)

### 2.1 Collect/Create 3D Models
```
Sources:
- [ ] Open-source surgical hardware CAD models
- [ ] Medical device manufacturer specs
- [ ] Parametric generation (if needed)

Target Models:
- [ ] Pedicle screw (various sizes)
- [ ] Spinal rod
- [ ] Bone cement injection pattern
- [ ] Cage implant (optional)
```

### 2.2 Placement Parameterization
```
Parameters to Model:
- Entry point (pedicle location)
- Trajectory (angle, depth)
- Size (diameter, length)
- Material properties (for artifact simulation)

Constraints:
- Anatomically plausible placement
- Follow AO Spine guidelines
- Avoid critical structures (spinal canal)
```

---

## 💻 Step 3: Artifact Synthesis Pipeline (Week 2-3)

### 3.1 Forward Model (CT Physics)
```python
def synthesize_metal_artifact(ct_volume, implant_mesh, placement):
    """
    Given clean CT + implant geometry + placement,
    generate CT with realistic metal artifacts.
    """
    # Step 1: Rasterize implant to CT voxel space
    implant_mask = rasterize_mesh(implant_mesh, placement, ct_volume.shape)
    
    # Step 2: Set metal HU values
    ct_with_metal = ct_volume.copy()
    ct_with_metal[implant_mask] = METAL_HU  # ~3000-30000 depending on material
    
    # Step 3: Simulate streak artifacts
    ct_with_streaks = add_streak_artifacts(ct_with_metal, implant_mask)
    
    # Step 4: Simulate blooming
    ct_with_blooming = add_blooming_effect(ct_with_streaks, implant_mask)
    
    # Step 5: Simulate HU corruption in surrounding tissue
    ct_final = add_hu_corruption(ct_with_blooming, implant_mask)
    
    return ct_final
```

### 3.2 Implementation Options
```
Option A: Physics-based (Accurate but Slow)
- Simulate X-ray projection
- Forward projection through metal
- Filtered backprojection with artifacts
- Pros: Realistic
- Cons: Computationally expensive

Option B: Empirical Rules (Fast)
- Analytical artifact models
- Parameterized streak patterns
- Gaussian/bilateral filtering
- Pros: Fast, controllable
- Cons: Less realistic

Option C: Hybrid (Recommended!)
- Physics-based for metal rasterization
- Empirical for streak/blooming
- Data-driven calibration
- Pros: Balance of speed & realism
```

---

## 🤖 Step 4: RL Adversary for Artifact Placement (Week 3-4)

### 4.1 Environment Design
```python
class SurgicalArtifactEnv(gym.Env):
    """
    RL environment for adversarial surgical artifact placement.
    
    Goal: Find implant placements that maximize TS failure
          while maintaining clinical plausibility.
    """
    
    def __init__(self, ct_volume, gt_mask):
        self.ct = ct_volume
        self.gt = gt_mask
        self.vertebrae = extract_vertebrae(gt_mask)
        
    def reset(self):
        # Select random vertebra(e) for implant placement
        self.target_vertebra = random.choice(self.vertebrae)
        return self._get_obs()
        
    def step(self, action):
        # action = (vertebra_id, entry_point, trajectory, size)
        
        # 1. Place implant
        implant_mesh = generate_implant(action)
        
        # 2. Synthesize artifact CT
        ct_with_artifact = synthesize_metal_artifact(
            self.ct, implant_mesh, action
        )
        
        # 3. Run TotalSegmentator
        ts_mask = run_totalsegmentator(ct_with_artifact)
        
        # 4. Compute reward
        assembly_loss = compute_assembly_loss(ts_mask, self.gt)
        plausibility_penalty = compute_plausibility(action)
        
        reward = assembly_loss - plausibility_penalty
        
        return self._get_obs(), reward, done, info
```

### 4.2 Reward Design
```
Reward = Assembly_Gain - Plausibility_Penalty

Assembly_Gain:
  + TS segmentation error (Dice decrease)
  + Assembly failure (mesh quality degradation)

Plausibility_Penalty:
  - Anatomically impossible placement
  - Screw penetrates spinal canal
  - Wrong entry point/trajectory
  - Size mismatch (too large/small)
  - Violates AO Spine guidelines
```

---

## 📊 Step 5: Evaluation & Validation (Week 4-5)

### 5.1 Metrics
```
1. Artifact Realism:
   - Visual inspection (qualitative)
   - Comparison with real surgical CTs
   - Radiologist evaluation (if possible)
   
2. TS Performance Degradation:
   - Dice score on synthetic artifacts
   - Target: 20-30% degradation
   - Compare with baseline (clean CT)
   
3. Plausibility:
   - % of anatomically valid placements
   - Compliance with surgical guidelines
   - No critical structure violations
   
4. RL Learning:
   - Reward curve convergence
   - Diversity of learned placements
   - Generalization to different vertebrae
```

### 5.2 Validation on Real Data
```
If Available:
- [ ] Collect real surgical CT dataset
- [ ] Run TS on real vs synthetic
- [ ] Compare artifact patterns
- [ ] Dice score correlation
```

---

## 🗓️ Timeline (6 weeks)

### Week 1: Foundation
- [x] Literature review on surgical artifacts
- [x] Directory setup & planning
- [ ] Collect implant geometries
- [ ] Study CT artifact physics

### Week 2: Modeling
- [ ] Implement implant rasterization
- [ ] Parametric implant generation
- [ ] Placement constraint system
- [ ] Initial artifact synthesis (simple)

### Week 3: Artifact Synthesis
- [ ] Implement streak artifact model
- [ ] Implement blooming effect
- [ ] Implement HU corruption
- [ ] Validate on test cases

### Week 4: RL Environment
- [ ] Design state/action/reward
- [ ] Implement SurgicalArtifactEnv
- [ ] Test with random policy
- [ ] Integrate with TS

### Week 5: RL Training
- [ ] Train PPO adversary
- [ ] Hyperparameter tuning
- [ ] Validation & visualization
- [ ] Compare with baselines

### Week 6: Evaluation & Documentation
- [ ] Comprehensive evaluation
- [ ] Real data validation (if available)
- [ ] Paper writing / documentation
- [ ] Compare with Phase 3 results

---

## 📦 Deliverables

### Code
```
outputs/phase4_surgical_artifacts/
├── implant_models/
│   ├── pedicle_screw.obj
│   ├── spinal_rod.obj
│   └── generate_implant.py
│
├── artifact_synthesis/
│   ├── forward_model.py          (CT physics simulation)
│   ├── streak_artifacts.py       (streak generation)
│   ├── blooming_effect.py        (blooming simulation)
│   └── visualize_artifacts.py
│
├── rl_training/
│   ├── surgical_artifact_env.py  (Gymnasium environment)
│   ├── train_adversary.py        (PPO training)
│   └── validation_callback.py
│
├── evaluation/
│   ├── evaluate_ts_degradation.py
│   ├── compare_with_real.py
│   └── plausibility_check.py
│
└── literature/
    └── [research papers & notes]
```

### Documentation
- Implementation guide
- Artifact synthesis tutorial
- Evaluation report
- Comparison with Phase 3

---

## 🔬 Research Questions to Answer

1. **Can we achieve 20-30% TS Dice degradation?**
   - Phase 3: 0.33-1.0% (fracture only)
   - Phase 4 target: 20-30% (surgical artifacts)

2. **Are synthetic artifacts realistic?**
   - Visual comparison with real surgical CTs
   - Quantitative metrics (if data available)

3. **Does RL find better artifact placements than rules?**
   - Compare learned vs random vs heuristic

4. **Does this improve assembly robustness?**
   - Train assembly on adversarial artifacts
   - Test on held-out surgical CTs

---

## ✅ Success Criteria

**Phase 4 is successful if:**

1. ✅ Synthetic artifacts look realistic (visual inspection)
2. ✅ TS Dice degradation: 20-30% (vs 0.33% in Phase 3)
3. ✅ RL learns diverse, plausible placements
4. ✅ Assembly trained on artifacts is more robust
5. ✅ Clear improvement over Phase 3 approach

---

## 🚀 Next Actions (This Week)

### Day 1-2: Literature & Data
- [ ] Read key papers on metal artifact simulation
- [ ] Find open-source implant CAD models
- [ ] Study AO Spine surgical guidelines

### Day 3-4: Initial Implementation
- [ ] Implement basic implant rasterization
- [ ] Test metal HU value insertion
- [ ] Create simple visualization

### Day 5: First Prototype
- [ ] Generate first synthetic artifact CT
- [ ] Run TotalSegmentator on it
- [ ] Measure Dice degradation
- [ ] Visualize results

---

*Let's make surgical artifacts the RIGHT way! 🔩*

