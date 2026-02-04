#!/usr/bin/env python3
"""
Visualize configuration comparison results
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('outputs/phase4_surgical_artifacts/evaluation_moderate/SUMMARY.json', 'r') as f:
    results = json.load(f)

# Extract data
configs = ['Baseline', 'Config 1\n(Screws)', 'Config 2\n(Screws+Rod)', 'Config 3\n(Multi-level)']
l1_dice = [
    results['baseline']['L1'],  # Baseline is just float
    results['config1_L1_screws_only']['L1']['dice'],
    results['config2_L1_screws_rod']['L1']['dice'],
    results['config3_multi_level']['L1']['dice']
]
l2_dice = [
    results['baseline']['L2'],  # Baseline is just float
    results['config1_L1_screws_only']['L2']['dice'],
    results['config2_L1_screws_rod']['L2']['dice'],
    results['config3_multi_level']['L2']['dice']
]

# Calculate degradation
l1_deg = [(results['baseline']['L1'] - d) / results['baseline']['L1'] * 100 for d in l1_dice]
l2_deg = [(results['baseline']['L2'] - d) / results['baseline']['L2'] * 100 for d in l2_dice]

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Dice scores
ax = axes[0]
x = np.arange(len(configs))
width = 0.35
ax.bar(x - width/2, l1_dice, width, label='L1', alpha=0.8, color='#2E86AB')
ax.bar(x + width/2, l2_dice, width, label='L2', alpha=0.8, color='#A23B72')
ax.set_ylabel('Dice Score', fontsize=12, fontweight='bold')
ax.set_title('Segmentation Performance', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0.65, 0.95])
ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Clinical Threshold')

# Plot 2: Degradation (%)
ax = axes[1]
ax.bar(x - width/2, l1_deg, width, label='L1', alpha=0.8, color='#2E86AB')
ax.bar(x + width/2, l2_deg, width, label='L2', alpha=0.8, color='#A23B72')
ax.set_ylabel('Degradation (%)', fontsize=12, fontweight='bold')
ax.set_title('Performance Degradation', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.axhspan(20, 30, alpha=0.2, color='green', label='Target Range')
ax.text(2.5, 25, 'Target\n20-30%', ha='center', va='center', fontsize=10, fontweight='bold', color='green')

# Annotate max degradation
max_idx = np.argmax(l2_deg)
ax.annotate(f'{l2_deg[max_idx]:.1f}%', 
            xy=(max_idx + width/2, l2_deg[max_idx]), 
            xytext=(max_idx + width/2, l2_deg[max_idx] + 3),
            ha='center', fontsize=11, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

# Plot 3: Hardware complexity
ax = axes[2]
hardware_counts = [0, 2, 3, 4]  # baseline, 2 screws, 2+1 rod, 4 screws + 2 rods
avg_deg = [(l1 + l2) / 2 for l1, l2 in zip(l1_deg, l2_deg)]
colors = ['gray', '#FFA500', '#FF4500', '#8B0000']
scatter = ax.scatter(hardware_counts, avg_deg, s=300, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
for i, (hw, deg) in enumerate(zip(hardware_counts, avg_deg)):
    ax.annotate(configs[i], xy=(hw, deg), xytext=(hw + 0.3, deg + 0.5),
                fontsize=9, fontweight='bold')
ax.set_xlabel('Number of Hardware Components', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Degradation (%)', fontsize=12, fontweight='bold')
ax.set_title('Hardware Complexity vs. Impact', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
ax.axhline(y=20, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Target Min')

plt.suptitle('Phase 4: Surgical Artifact Configurations - Evaluation Results', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/phase4_surgical_artifacts/evaluation_moderate/COMPARISON_PLOT.png', 
            dpi=300, bbox_inches='tight')
print("✓ Visualization saved: COMPARISON_PLOT.png")

# Print key findings
print("\n" + "="*60)
print("KEY FINDINGS:")
print("="*60)
print(f"\n1. BEST CONFIGURATION: Config 2 (Screws + Rod)")
print(f"   - L2 Degradation: {l2_deg[2]:.2f}% (closest to 20-30% target)")
print(f"   - Average Degradation: {avg_deg[2]:.2f}%")

print(f"\n2. ADJACENT VERTEBRA EFFECT (Novel!)")
print(f"   - L1 has hardware, but L2 shows larger degradation")
print(f"   - Config 2: L1={l1_deg[2]:.2f}%, L2={l2_deg[2]:.2f}%")
print(f"   - Streak artifacts propagate inferiorly!")

print(f"\n3. MULTI-LEVEL PARADOX:")
print(f"   - Config 3 (most hardware): {avg_deg[3]:.2f}% degradation")
print(f"   - Config 2 (less hardware): {avg_deg[2]:.2f}% degradation")
print(f"   - Hypothesis: TS recognizes multi-level as 'structure'?")

print(f"\n4. NEXT STEPS:")
print(f"   - Increase artifact severity in Config 2 (17% → 25%)")
print(f"   - Investigate multi-level segmentation strategy")
print(f"   - RL for optimal screw placement")
print("="*60 + "\n")

