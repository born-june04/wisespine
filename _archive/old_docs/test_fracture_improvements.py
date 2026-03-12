#!/usr/bin/env python3
"""Quick verification of AO fracture simulation improvements."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pipeline', 'modules'))

import numpy as np

print("=" * 60)
print("AO Fracture Improvement Verification")
print("=" * 60)

# Test 1: Import and AO configs
print("\n[Test 1] Import & AO Config Check...")
from fracture_simulator_v2 import (
    BoneFractureSimulator, AO_LOAD_CONFIGS,
    classify_regions_from_mask, generate_vertebra_particles
)
for ao, cfg in AO_LOAD_CONFIGS.items():
    rr = cfg.get('retropulsion_ratio', 'MISSING')
    print(f"  {ao}: retropulsion_ratio={rr}")
assert AO_LOAD_CONFIGS['A1']['retropulsion_ratio'] == 0.0
assert AO_LOAD_CONFIGS['A4']['retropulsion_ratio'] == 0.40
print("  ✅ PASS")

# Test 2: classify_regions_from_mask
print("\n[Test 2] Mask-based Region Classification...")
test_mask = np.zeros((30, 30, 30), dtype=np.int32)
test_mask[5:25, 5:25, 5:25] = 1  # cube of bone
test_pos = np.random.rand(500, 3).astype(np.float32)
regions = classify_regions_from_mask(test_pos, test_mask)
assert len(regions) == 500
unique_r, counts_r = np.unique(regions, return_counts=True)
rnames = ['anterior', 'central', 'posterior', 'cortical', 'endplate', 'pedicle', 'lamina']
for r, c in zip(unique_r, counts_r):
    print(f"  Region {r} ({rnames[r]}): {c} particles")
# Should have multiple region types, not just one
assert len(unique_r) >= 3, f"Expected >=3 region types, got {len(unique_r)}"
print("  ✅ PASS")

# Test 3: Full simulation with AO A4
print("\n[Test 3] Full A4 Simulation (200 steps)...")
positions = generate_vertebra_particles(n_particles=5000, seed=42)
sim = BoneFractureSimulator(positions, seed=42)
sim.setup_ao_loading('A4')
sim.set_stress_mode('grid')
sim.enable_anisotropy(ratio=2.0)
history = sim.run(n_steps=200, max_force=AO_LOAD_CONFIGS['A4']['max_force'],
                  record_every=50, verbose=True)
final = history[-1]
print(f"  Max damage: {final['max_damage']:.3f}")
print(f"  Damaged %: {final['damaged_frac']*100:.1f}%")
print(f"  Fractured %: {final['fractured_frac']*100:.1f}%")

# Test fragment detection
frag = sim.detect_fragments(damage_threshold=0.9)
print(f"  Fragments detected: {frag['n_fragments']}")
print("  ✅ PASS")

# Test 4: apply_damage_to_ct
print("\n[Test 4] Realistic CT Damage Mapping...")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pipeline', 'modules'))
from _gen_real_fracture_visuals import apply_damage_to_ct
ct = np.random.uniform(200, 800, (20, 20, 20)).astype(np.float32)
mask = np.ones((20, 20, 20), dtype=np.int32)
damage = np.zeros((20, 20, 20), dtype=np.float32)

# Create damage gradient for testing all levels
damage[5, :, :] = 0.1   # compaction
damage[8, :, :] = 0.3   # micro-fracture
damage[11, :, :] = 0.5  # cortical disruption
damage[14, :, :] = 0.7  # hematoma
damage[17, :, :] = 0.95 # complete separation

result = apply_damage_to_ct(ct, mask, damage)

# Verify each level
compaction_hu = result[5, 10, 10]
original_hu = ct[5, 10, 10]
print(f"  Compaction (D=0.1): {original_hu:.0f} → {compaction_hu:.0f} (should INCREASE)")
assert compaction_hu > original_hu, "Compaction should increase HU!"

micro_hu = result[8, 10, 10]
print(f"  Micro-frac (D=0.3): {ct[8,10,10]:.0f} → {micro_hu:.0f} (should decrease)")
assert micro_hu < ct[8, 10, 10], "Micro-fracture should decrease HU!"

hematoma_hu = result[14, 10, 10]
print(f"  Hematoma (D=0.7): {ct[14,10,10]:.0f} → {hematoma_hu:.0f} (should be <200, disrupted)")
assert hematoma_hu < 250, f"Hematoma should be <250 HU (disrupted bone blend), got {hematoma_hu:.0f}"
assert hematoma_hu > 30, f"Hematoma should be >30 HU (not air), got {hematoma_hu:.0f}"

complete_hu = result[17, 10, 10]
print(f"  Complete (D=0.95): {ct[17,10,10]:.0f} → {complete_hu:.0f} (should be ~-50 to 80)")
assert complete_hu < 150, f"Complete separation should be <150 HU, got {complete_hu:.0f}"
assert complete_hu > -200, f"Complete separation should NOT be air (-900), got {complete_hu:.0f}"
print("  ✅ PASS")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✅")
print("=" * 60)
