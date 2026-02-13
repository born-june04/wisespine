"""
Summarize VRM Ablation Results
Read checkpoint files directly to get configuration and metrics
"""

import os
import sys
import torch
from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Add workspace to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def parse_checkpoint(exp_dir: Path) -> Optional[Dict]:
    """Parse checkpoint files to extract configuration and metrics"""
    stage1_checkpoint = exp_dir / 'stage1' / 'best_model.pth'
    stage2_checkpoint = exp_dir / 'stage2' / 'best_model.pth'
    
    if not stage1_checkpoint.exists() or not stage2_checkpoint.exists():
        return None
    
    try:
        # Load Stage 1 checkpoint
        stage1_ckpt = torch.load(stage1_checkpoint, map_location='cpu', weights_only=False)
        stage1_metrics = stage1_ckpt.get('metrics', {})
        stage1_loss = stage1_metrics.get('loss')
        
        # Load Stage 2 checkpoint
        stage2_ckpt = torch.load(stage2_checkpoint, map_location='cpu', weights_only=False)
        stage2_metrics = stage2_ckpt.get('metrics', {})
        stage2_dice = stage2_metrics.get('dice')
        stage2_loss = stage2_metrics.get('loss')
        
        # Get configuration from args
        args = stage1_ckpt.get('args') or stage2_ckpt.get('args')
        if args is None:
            return None
        
        # Extract model_size and use_vrm
        model_size = None
        use_vrm = None
        
        if hasattr(args, 'model_size'):
            model_size = args.model_size
        elif isinstance(args, dict):
            model_size = args.get('model_size')
        
        if hasattr(args, 'use_vrm'):
            use_vrm = int(args.use_vrm) if args.use_vrm else 0
        elif isinstance(args, dict):
            use_vrm = int(args.get('use_vrm', 0))
        
        # If still None, try to infer from model parameters
        if model_size is None:
            stage1_params = sum(p.numel() for p in stage1_ckpt['model_state_dict'].values())
            if stage1_params < 200000:
                model_size = 'tiny'
            elif stage1_params < 2000000:
                model_size = 'small'
            else:
                model_size = 'base'
        
        if use_vrm is None:
            # Check if VRM layers exist in model
            has_vrm = any('vrm' in key.lower() for key in stage1_ckpt['model_state_dict'].keys())
            use_vrm = 1 if has_vrm else 0
        
        # Calculate parameters
        stage1_params = sum(p.numel() for p in stage1_ckpt['model_state_dict'].values())
        stage2_params = sum(p.numel() for p in stage2_ckpt['model_state_dict'].values())
        
        return {
            'model_size': model_size,
            'use_vrm': use_vrm,
            'dice': stage2_dice,
            'loss': stage1_loss,
            'stage2_loss': stage2_loss,
            'epochs': stage2_ckpt.get('epoch'),
            'stage1_params': stage1_params,
            'stage2_params': stage2_params,
        }
    except Exception as e:
        print(f"Error parsing {exp_dir}: {e}")
        return None


def load_all_results(outputs_dir: str) -> Dict:
    """Load all experiment results from checkpoint files"""
    outputs_dir = Path(outputs_dir)
    results = {}
    
    # Scan all VerSe_COARSE_FINE experiment directories
    for exp_dir in outputs_dir.glob('VerSe_COARSE_FINE_*'):
        if not exp_dir.is_dir():
            continue
        
        result = parse_checkpoint(exp_dir)
        if result and result['dice'] is not None:
            key = f"{result['model_size']}_vrm{result['use_vrm']}"
            results[key] = result
    
    return results


def create_comparison_table(results: Dict, output_path: Path):
    """Create comparison table"""
    # Organize by model size
    model_sizes = ['tiny', 'small', 'base']
    
    data = []
    for model_size in model_sizes:
        vrm_key = f"{model_size}_vrm1"
        no_vrm_key = f"{model_size}_vrm0"
        
        vrm_result = results.get(vrm_key, {})
        no_vrm_result = results.get(no_vrm_key, {})
        
        vrm_dice = vrm_result.get('dice', None)
        no_vrm_dice = no_vrm_result.get('dice', None)
        
        improvement = None
        if vrm_dice is not None and no_vrm_dice is not None:
            improvement = (vrm_dice - no_vrm_dice) * 100
        
        # Total params
        vrm_params = None
        no_vrm_params = None
        if vrm_result.get('stage1_params') and vrm_result.get('stage2_params'):
            vrm_params = vrm_result['stage1_params'] + vrm_result['stage2_params']
        if no_vrm_result.get('stage1_params') and no_vrm_result.get('stage2_params'):
            no_vrm_params = no_vrm_result['stage1_params'] + no_vrm_result['stage2_params']
        
        data.append({
            'Model Size': model_size.capitalize(),
            'With VRM Dice': f"{vrm_dice:.4f}" if vrm_dice else "N/A",
            'Without VRM Dice': f"{no_vrm_dice:.4f}" if no_vrm_dice else "N/A",
            'VRM Improvement (%)': f"{improvement:+.2f}%" if improvement is not None else "N/A",
            'With VRM Params': f"{vrm_params/1e6:.2f}M" if vrm_params else "N/A",
            'Without VRM Params': f"{no_vrm_params/1e6:.2f}M" if no_vrm_params else "N/A",
            'With VRM Loss': f"{vrm_result.get('loss', 0):.4f}" if vrm_result.get('loss') else "N/A",
            'Without VRM Loss': f"{no_vrm_result.get('loss', 0):.4f}" if no_vrm_result.get('loss') else "N/A",
        })
    
    # Save as CSV (without pandas)
    csv_path = output_path / 'vrm_ablation_results.csv'
    with open(csv_path, 'w') as f:
        # Header
        if data:
            f.write(','.join(data[0].keys()) + '\n')
            # Data
            for row in data:
                f.write(','.join(str(v) for v in row.values()) + '\n')
    print(f"✓ Results saved to {csv_path}")
    
    # Create DataFrame-like structure for visualization
    columns = list(data[0].keys()) if data else []
    df_data = [[row[col] for col in columns] for row in data] if data else []
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Table
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    table = ax1.table(cellText=df_data, colLabels=columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.12, 0.12, 0.12, 0.15, 0.12, 0.12, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color code rows
    for i in range(len(df_data)):
        for j in range(len(columns)):
            cell = table[(i+1, j)]
            if j == 3:  # Improvement column
                if df_data[i][j] != "N/A":
                    improvement = float(df_data[i][j].replace('%', '').replace('+', ''))
                    if improvement > 0:
                        cell.set_facecolor('#90EE90')  # Light green
                    elif improvement < 0:
                        cell.set_facecolor('#FFB6C1')  # Light pink
                    else:
                        cell.set_facecolor('#F0F0F0')  # Light gray
            else:
                cell.set_facecolor('#FFFFFF')
            cell.set_edgecolor('#CCCCCC')
    
    # Header row
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#40466e')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    ax1.set_title('VRM Ablation Study Results', fontsize=16, fontweight='bold', pad=20)
    
    # Bar chart: Dice comparison
    ax2 = fig.add_subplot(gs[1, 0])
    model_sizes_plot = []
    vrm_dices = []
    no_vrm_dices = []
    
    for model_size in model_sizes:
        vrm_key = f"{model_size}_vrm1"
        no_vrm_key = f"{model_size}_vrm0"
        
        vrm_dice = results.get(vrm_key, {}).get('dice')
        no_vrm_dice = results.get(no_vrm_key, {}).get('dice')
        
        if vrm_dice is not None and no_vrm_dice is not None:
            model_sizes_plot.append(model_size.capitalize())
            vrm_dices.append(vrm_dice)
            no_vrm_dices.append(no_vrm_dice)
    
    if model_sizes_plot:
        x = range(len(model_sizes_plot))
        width = 0.35
        
        bars1 = ax2.bar([i - width/2 for i in x], vrm_dices, width, 
                       label='With VRM', color='#2ca02c', alpha=0.8, edgecolor='black')
        bars2 = ax2.bar([i + width/2 for i in x], no_vrm_dices, width,
                       label='Without VRM', color='#d62728', alpha=0.8, edgecolor='black')
        
        ax2.set_xlabel('Model Size', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Dice Score', fontsize=12, fontweight='bold')
        ax2.set_title('Dice Score Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_sizes_plot)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Improvement chart
    ax3 = fig.add_subplot(gs[1, 1])
    improvements = []
    model_sizes_imp = []
    
    for model_size in model_sizes:
        vrm_key = f"{model_size}_vrm1"
        no_vrm_key = f"{model_size}_vrm0"
        
        vrm_dice = results.get(vrm_key, {}).get('dice')
        no_vrm_dice = results.get(no_vrm_key, {}).get('dice')
        
        if vrm_dice is not None and no_vrm_dice is not None:
            improvement = (vrm_dice - no_vrm_dice) * 100
            improvements.append(improvement)
            model_sizes_imp.append(model_size.capitalize())
    
    if improvements:
        colors = ['#90EE90' if imp > 0 else '#FFB6C1' for imp in improvements]
        bars = ax3.barh(model_sizes_imp, improvements, color=colors, alpha=0.8, edgecolor='black')
        ax3.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
        ax3.set_title('VRM Improvement over Baseline', fontsize=14, fontweight='bold')
        ax3.axvline(0, color='black', linestyle='--', linewidth=2)
        ax3.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            ax3.text(imp + (0.1 if imp >= 0 else -0.1), i, f'{imp:+.2f}%',
                    va='center', ha='left' if imp >= 0 else 'right',
                    fontsize=11, fontweight='bold')
    
    plt.savefig(output_path / 'vrm_ablation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Visualization saved to {output_path / 'vrm_ablation_comparison.png'}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("VRM Ablation Study Summary")
    print("=" * 80)
    if data:
        # Print header
        print(' | '.join(f"{col:20}" for col in columns))
        print("-" * 80)
        # Print data
        for row in data:
            print(' | '.join(f"{str(v):20}" for v in row.values()))
    print("=" * 80 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Summarize VRM ablation results')
    parser.add_argument('--outputs_dir', type=str,
                      default='/gscratch/scrubbed/june0604/vindr/outputs',
                      help='Directory containing experiment outputs')
    parser.add_argument('--output_dir', type=str,
                      default='/gscratch/scrubbed/june0604/vindr/outputs/vrm_ablation_summary',
                      help='Output directory for results')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading results from checkpoint files...")
    results = load_all_results(args.outputs_dir)
    
    print(f"Found {len(results)} experiment results:")
    for key, res in results.items():
        print(f"  {key}: Dice={res.get('dice', 'N/A'):.4f}")
    
    if not results:
        print("⚠️  No results found. Please run experiments first.")
        return
    
    create_comparison_table(results, output_dir)


if __name__ == '__main__':
    main()

