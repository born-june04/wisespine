"""
Visualize Test Set Evaluation Results
Compare multiple models' test performance
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")

MODEL_COLORS = {
    'Small_VRM0': '#1f77b4',
    'Small_VRM1': '#ff7f0e',
    'Tiny_VRM0': '#2ca02c',
    'Tiny_VRM1': '#d62728',
}


def load_test_results(test_results_dir: Path) -> Dict:
    """Load all test result JSON files"""
    results = {}
    
    for model_dir in test_results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Find latest JSON file
        json_files = list(model_dir.glob('test_results_*.json'))
        if not json_files:
            continue
        
        latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_json, 'r') as f:
            data = json.load(f)
            results[model_name] = data
    
    return results


def visualize_comparison(results: Dict, output_path: Path):
    """Create comprehensive comparison visualization"""
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Overall Dice Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    model_names = list(results.keys())
    mean_dices = [results[m]['mean_dice'] for m in model_names]
    std_dices = [results[m]['std_dice'] for m in model_names]
    colors = [MODEL_COLORS.get(m, 'gray') for m in model_names]
    
    bars = ax1.bar(range(len(model_names)), mean_dices,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.set_ylabel('Mean Dice Score', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Test Set Performance', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Adjust y-axis to show differences more clearly - tighter scale
    min_dice = min(mean_dices)
    max_dice = max(mean_dices)
    center = (min_dice + max_dice) / 2
    range_val = max(0.05, (max_dice - min_dice) * 4)  # 4x range for better visibility
    min_dice = max(0, center - range_val / 2)
    max_dice = min(1, center + range_val / 2)
    # Ensure minimum range
    if max_dice - min_dice < 0.05:
        min_dice = center - 0.025
        max_dice = center + 0.025
    ax1.set_ylim([min_dice, max_dice])
    
    # Add value text on top of bars
    for i, (bar, mean) in enumerate(zip(bars, mean_dices)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + (max_dice - min_dice) * 0.01,
                f'{mean:.4f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    # 2. Per-Vertebra Comparison (Heatmap)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Get all vertebrae
    all_vertebrae = set()
    for model_data in results.values():
        for v in model_data['per_vertebra']:
            all_vertebrae.add((v['vertebra_id'], v['vertebra_name']))
    
    sorted_vertebrae = sorted(all_vertebrae, key=lambda x: x[0])
    vertebra_names = [v[1] for v in sorted_vertebrae]
    
    # Create heatmap data
    heatmap_data = []
    for model_name in model_names:
        model_data = results[model_name]
        vertebra_dict = {v['vertebra_id']: v['mean_dice'] 
                        for v in model_data['per_vertebra']}
        row = [vertebra_dict.get(vid, 0) for vid, _ in sorted_vertebrae]
        heatmap_data.append(row)
    
    im = ax2.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(vertebra_names)))
    ax2.set_xticklabels(vertebra_names, rotation=90, fontsize=8)
    ax2.set_yticks(range(len(model_names)))
    ax2.set_yticklabels(model_names)
    ax2.set_title('Per-Vertebra Dice Scores', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Dice Score')
    
    # 3. Per-Vertebra Bar Chart (Top 10)
    ax3 = fig.add_subplot(gs[1, :])
    
    # Average across models for each vertebra
    vertebra_avg = {}
    for vid, vname in sorted_vertebrae:
        dices = []
        for model_data in results.values():
            for v in model_data['per_vertebra']:
                if v['vertebra_id'] == vid:
                    dices.append(v['mean_dice'])
        if dices:
            vertebra_avg[(vid, vname)] = np.mean(dices)
    
    # Sort and get top/bottom
    sorted_by_dice = sorted(vertebra_avg.items(), key=lambda x: x[1], reverse=True)
    top_vertebrae = sorted_by_dice[:10]
    
    top_names = [v[1] for v, _ in top_vertebrae]
    top_dices = [d for _, d in top_vertebrae]
    
    bars = ax3.barh(range(len(top_names)), top_dices, color='steelblue', alpha=0.8, edgecolor='black')
    ax3.set_yticks(range(len(top_names)))
    ax3.set_yticklabels(top_names)
    ax3.set_xlabel('Mean Dice Score', fontsize=12, fontweight='bold')
    ax3.set_title('Top 10 Vertebrae (Average across models)', fontsize=14, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # Adjust x-axis to show differences more clearly
    min_dice = min(top_dices) - 0.05
    max_dice = max(top_dices) + 0.05
    min_dice = max(0, min_dice)  # Don't go below 0
    max_dice = min(1, max_dice)  # Don't go above 1
    # If range is small, use a tighter range
    if max_dice - min_dice < 0.2:
        center = (min(top_dices) + max(top_dices)) / 2
        range_val = max(0.15, (max(top_dices) - min(top_dices)) * 3)
        min_dice = max(0, center - range_val / 2)
        max_dice = min(1, center + range_val / 2)
    ax3.set_xlim([min_dice, max_dice])
    
    for i, (bar, dice) in enumerate(zip(bars, top_dices)):
        ax3.text(dice + (max_dice - min_dice) * 0.01, i, f'{dice:.4f}',
                va='center', fontsize=10)
    
    # 4. Region-wise Performance
    ax4 = fig.add_subplot(gs[2, 0])
    
    regions = {
        'Cervical': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
        'Thoracic': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12'],
        'Lumbar': ['L1', 'L2', 'L3', 'L4', 'L5'],
        'Sacral': ['S1', 'Coccyx'],
    }
    
    region_data = {region: [] for region in regions.keys()}
    region_models = {region: [] for region in regions.keys()}
    
    for model_name, model_data in results.items():
        vertebra_dict = {v['vertebra_name']: v['mean_dice'] 
                        for v in model_data['per_vertebra']}
        
        for region, vertebra_list in regions.items():
            region_dices = [vertebra_dict.get(v, 0) for v in vertebra_list]
            region_dices = [d for d in region_dices if d > 0]  # Remove missing
            if region_dices:
                region_data[region].append(np.mean(region_dices))
                region_models[region].append(model_name)
    
    x = np.arange(len(regions))
    width = 0.2
    
    for i, model_name in enumerate(model_names):
        model_means = []
        for region in regions.keys():
            if model_name in region_models[region]:
                idx = region_models[region].index(model_name)
                model_means.append(region_data[region][idx])
            else:
                model_means.append(0)
        
        ax4.bar(x + i*width, model_means, width, label=model_name,
               color=MODEL_COLORS.get(model_name, 'gray'), alpha=0.8, edgecolor='black')
    
    ax4.set_xlabel('Region', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Mean Dice Score', fontsize=12, fontweight='bold')
    ax4.set_title('Region-wise Performance', fontsize=14, fontweight='bold')
    ax4.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax4.set_xticklabels(regions.keys())
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # 5. Summary Table
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    table_data = []
    for model_name in model_names:
        data = results[model_name]
        table_data.append([
            model_name,
            f"{data['mean_dice']:.4f}",
            f"{data['std_dice']:.4f}",
            f"{data['median_dice']:.4f}",
            f"{data['min_dice']:.4f}",
            f"{data['max_dice']:.4f}",
        ])
    
    headers = ['Model', 'Mean Dice', 'Std Dice', 'Median Dice', 'Min Dice', 'Max Dice']
    table = ax5.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code rows
    for i, model_name in enumerate(model_names):
        color = MODEL_COLORS.get(model_name, 'white')
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(color)
            table[(i+1, j)].set_alpha(0.3)
    
    # Header
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#40466e')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    ax5.set_title('Test Set Results Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('Test Set Evaluation - Model Comparison', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Visualization saved to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize test set results')
    parser.add_argument('--test_results_dir', type=str,
                      default='/gscratch/scrubbed/june0604/vindr/outputs/test_results',
                      help='Directory containing test result JSON files')
    parser.add_argument('--output_path', type=str,
                      default='/gscratch/scrubbed/june0604/vindr/outputs/test_results_comparison.png',
                      help='Output path for visualization')
    
    args = parser.parse_args()
    
    test_results_dir = Path(args.test_results_dir)
    output_path = Path(args.output_path)
    
    print("Loading test results...")
    results = load_test_results(test_results_dir)
    
    if not results:
        print("⚠️  No test results found!")
        return
    
    print(f"Found {len(results)} model results:")
    for name, data in results.items():
        print(f"  {name}: Mean Dice = {data['mean_dice']:.4f}")
    
    print("\nGenerating visualization...")
    visualize_comparison(results, output_path)


if __name__ == '__main__':
    main()

