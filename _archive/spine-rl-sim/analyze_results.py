#!/usr/bin/env python3
import json
import sys

log_file = sys.argv[1] if len(sys.argv) > 1 else "adversary_outputs/2026-01-29_12-27-25/logs/training_log.jsonl"

with open(log_file) as f:
    lines = [json.loads(line) for line in f]

total = len(lines)
with_reward = [l for l in lines if "reward" in l]
with_changes = [l for l in lines if l.get("voxels_changed", 0) > 0]

print("=" * 60)
print("📊 ADVERSARY TRAINING RESULTS")
print("=" * 60)
print(f"Total timesteps logged: {total}")
print(f"Episodes with reward info: {len(with_reward)}")
print(f"Episodes with actual mask changes: {len(with_changes)}")
print()

if with_changes:
    print("✅ Top 10 Corruption Events (by voxel changes):")
    for i, ep in enumerate(sorted(with_changes, key=lambda x: x.get("voxels_changed", 0), reverse=True)[:10], 1):
        ts = ep["timestep"]
        vc = ep["voxels_changed"]
        rw = ep.get("reward", "N/A")
        ops = ep.get("operations_used", "?")
        print(f"  {i}. t={ts:3d}: {vc:6d} voxels changed, ops={ops}, reward={rw}")
    print()

if with_reward:
    rewards = [l["reward"] for l in with_reward]
    losses = [l["assembly_loss"] for l in with_reward]
    penalties = [l["prior_penalty"] for l in with_reward]
    
    print("📈 Training Metrics:")
    print(f"  Reward:        min={min(rewards):7.3f}, max={max(rewards):7.3f}, mean={sum(rewards)/len(rewards):7.3f}")
    print(f"  Assembly Loss: min={min(losses):7.4f}, max={max(losses):7.4f}, mean={sum(losses)/len(losses):7.4f}")
    print(f"  Prior Penalty: min={min(penalties):7.2f}, max={max(penalties):7.2f}, mean={sum(penalties)/len(penalties):7.2f}")
    print()
    
    print("🎯 Last 10 Episodes:")
    for ep in with_reward[-10:]:
        ts = ep["timestep"]
        rw = ep["reward"]
        loss = ep["assembly_loss"]
        pen = ep["prior_penalty"]
        vc = ep["voxels_changed"]
        print(f"  t={ts:3d}: reward={rw:7.3f}, loss={loss:.4f}, penalty={pen:5.2f}, changed={vc:6d} voxels")
    print()

# Check if adversary is learning
if len(with_reward) >= 20:
    early_rewards = [l["reward"] for l in with_reward[:len(with_reward)//4]]
    late_rewards = [l["reward"] for l in with_reward[-len(with_reward)//4:]]
    
    print("🧠 Learning Progress:")
    print(f"  Early episodes (first 25%): mean reward = {sum(early_rewards)/len(early_rewards):.3f}")
    print(f"  Late episodes (last 25%):   mean reward = {sum(late_rewards)/len(late_rewards):.3f}")
    improvement = (sum(late_rewards)/len(late_rewards)) - (sum(early_rewards)/len(early_rewards))
    if improvement > 0.01:
        print(f"  ✅ Improvement: +{improvement:.3f} (adversary getting better!)")
    elif improvement < -0.01:
        print(f"  ⚠️  Degradation: {improvement:.3f} (adversary getting worse)")
    else:
        print(f"  ⚪ Stable: ~{improvement:.3f} (no significant change)")

print("=" * 60)

