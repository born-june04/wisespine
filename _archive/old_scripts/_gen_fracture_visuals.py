#!/usr/bin/env python3
"""Generate publication-quality fracture visualization with full + zoom-in views.

Uses GRID-BASED stress transfer exclusively (physically correct method).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pathlib import Path
from fracture_simulator_v2 import (
    BoneFractureSimulator, generate_vertebra_particles, classify_regions,
    create_2d_projection, AO_LOAD_CONFIGS, DAMAGE_CMAP,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import imageio

OUTPUT_DIR = Path(__file__).parent.parent.parent / 'docs' / 'fracture_reports' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REGION_COLORS = {
    'anterior_body': '#ff6b6b', 'central_body': '#feca57',
    'posterior_body': '#48dbfb', 'cortical_shell': '#ff9ff3',
    'endplate': '#54a0ff', 'pedicle': '#5f27cd', 'lamina': '#01a3a4',
}
REGION_LABELS = {
    'anterior_body': 'Anterior\nBody', 'central_body': 'Central\nBody',
    'posterior_body': 'Posterior\nBody', 'cortical_shell': 'Cortical\nShell',
    'endplate': 'Endplate', 'pedicle': 'Pedicle', 'lamina': 'Lamina',
}


def run_ao(ao_type, positions, mode='grid', anisotropy=True):
    config = AO_LOAD_CONFIGS[ao_type]
    sim = BoneFractureSimulator(positions.copy(), seed=42)
    sim.setup_ao_loading(ao_type)
    sim.set_stress_mode(mode)
    if anisotropy:
        sim.enable_anisotropy(ratio=2.0)
    history = sim.run(n_steps=200, max_force=config['max_force'],
                      record_every=4, verbose=False)
    return sim, history


def find_damage_bounds(positions, damage, threshold=0.005, pad=0.03):
    """Find bounding box of damaged region with padding."""
    mask = damage > threshold
    if mask.sum() < 10:
        return None
    dp = positions[mask]
    lo = dp.min(axis=0) - pad
    hi = dp.max(axis=0) + pad
    return lo, hi


def create_zoom_projection(positions, damage, axis, bounds, resolution=128):
    """Create 2D projection zoomed into a bounding box region."""
    lo, hi = bounds
    axes = [i for i in range(3) if i != axis]
    ax0, ax1 = axes

    grid = np.zeros((resolution, resolution))
    counts = np.zeros((resolution, resolution))

    for i in range(len(positions)):
        p = positions[i]
        if (p[0] < lo[0] or p[0] > hi[0] or
            p[1] < lo[1] or p[1] > hi[1] or
            p[2] < lo[2] or p[2] > hi[2]):
            continue

        x = int((p[ax0] - lo[ax0]) / (hi[ax0] - lo[ax0] + 1e-10) * (resolution - 1))
        y = int((p[ax1] - lo[ax1]) / (hi[ax1] - lo[ax1] + 1e-10) * (resolution - 1))
        x = np.clip(x, 0, resolution - 1)
        y = np.clip(y, 0, resolution - 1)

        grid[x, y] = max(grid[x, y], damage[i])
        counts[x, y] += 1

    return grid


def generate_ao_card(ao_type, sim, history, positions, regions, output_path):
    """Per-AO detail card: 3 full views + 3 zoom views + bar + timeline."""
    config = AO_LOAD_CONFIGS[ao_type]
    damage = sim.damage

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0a0a12')

    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3,
                  height_ratios=[1, 1, 0.8])

    # Title
    fig.suptitle(f'{config["name"]}',
                 fontsize=22, fontweight='bold', color='white', y=0.97)
    fig.text(0.5, 0.935, f'Mechanism: {config["mechanism"]}  |  '
             f'Stress: Grid-Based P2G/G2P (50 iter, α=0.6)',
             fontsize=11, color='#aaa', ha='center')

    views = [('Axial', 2), ('Sagittal', 0), ('Coronal', 1)]
    bounds = find_damage_bounds(positions, damage, threshold=0.1, pad=0.02)

    # Row 0: Full views
    for vi, (name, axis) in enumerate(views):
        ax = fig.add_subplot(gs[0, vi])
        grid = create_2d_projection(positions, damage, axis=axis)
        ax.imshow(grid.T, origin='lower', cmap=DAMAGE_CMAP, vmin=0, vmax=1)
        ax.set_title(f'{name} (Full)', color='white', fontsize=11, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor('#333')

    # Stats panel (row 0, col 3)
    ax_info = fig.add_subplot(gs[0, 3])
    ax_info.set_facecolor('#0a0a12')
    ax_info.axis('off')
    final = history[-1]
    frag = sim.detect_fragments(damage_threshold=0.9)
    stats = (
        f"Total Damaged: {final['damaged_frac']*100:.1f}%\n"
        f"Total Fractured: {final['fractured_frac']*100:.1f}%\n"
        f"Max Damage: {final['max_damage']:.3f}\n"
        f"Peak Force: {config['max_force']:.1f}\n"
        f"Load Points: {len(config['loading_points'])}\n"
        f"Fragments: {frag['n_fragments']}\n"
        f"\nStress Mode: Grid P2G/G2P\n"
        f"Grid: 64³ ({sim.grid_occupancy.sum()}/{64**3})\n"
        f"Diffusion: 50 iter, α=0.6"
    )
    ax_info.text(0.05, 0.95, stats, fontsize=11, color='white',
                 va='top', fontfamily='monospace', transform=ax_info.transAxes,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', edgecolor='#444'))

    # Row 1: Zoom-in views
    for vi, (name, axis) in enumerate(views):
        ax = fig.add_subplot(gs[1, vi])
        if bounds is not None:
            grid = create_zoom_projection(positions, damage, axis, bounds, resolution=128)
            ax.imshow(grid.T, origin='lower', cmap=DAMAGE_CMAP, vmin=0, vmax=1)
            ax.set_title(f'{name} (Zoom — Damage Zone)', color='#feca57',
                        fontsize=11, fontweight='bold')
        else:
            grid = create_2d_projection(positions, damage, axis=axis)
            ax.imshow(grid.T, origin='lower', cmap=DAMAGE_CMAP, vmin=0, vmax=1)
            ax.set_title(f'{name} (No significant damage)', color='#888', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor('#feca57' if bounds else '#333')

    # Colorbar in row 1, col 3
    ax_cb = fig.add_subplot(gs[1, 3])
    sm = plt.cm.ScalarMappable(cmap=DAMAGE_CMAP, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax_cb, fraction=0.8, pad=0.02)
    cbar.set_label('Damage (D)', color='white', fontsize=11)
    cbar.ax.tick_params(colors='white')
    ax_cb.axis('off')

    # Row 2: Region bar + timeline
    dmg_by_region = sim.get_damage_by_region()
    ax_bar = fig.add_subplot(gs[2, :2])
    ax_bar.set_facecolor('#0d0d1a')
    names = list(dmg_by_region.keys())
    values = [dmg_by_region[n] for n in names]
    colors = [REGION_COLORS.get(n, '#888') for n in names]
    labels = [REGION_LABELS.get(n, n) for n in names]
    bars = ax_bar.barh(range(len(names)), values, color=colors,
                       edgecolor='#333', height=0.7)
    ax_bar.set_yticks(range(len(names)))
    ax_bar.set_yticklabels(labels, fontsize=10, color='white')
    ax_bar.set_xlim(0, 1); ax_bar.set_xlabel('Mean Damage', color='white', fontsize=11)
    ax_bar.set_title('Damage by Region', color='white', fontsize=12, fontweight='bold')
    ax_bar.tick_params(colors='#888')
    for sp in ax_bar.spines.values(): sp.set_edgecolor('#333')
    for bar, val in zip(bars, values):
        ax_bar.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center', color='white', fontsize=9)

    ax_time = fig.add_subplot(gs[2, 2:])
    ax_time.set_facecolor('#0d0d1a')
    steps = [h['step'] for h in history]
    damaged = [h['damaged_frac'] * 100 for h in history]
    fractured = [h['fractured_frac'] * 100 for h in history]
    ax_time.fill_between(steps, damaged, alpha=0.3, color='#ff6b6b')
    ax_time.plot(steps, damaged, color='#ff6b6b', linewidth=2, label='Damaged %')
    ax_time.fill_between(steps, fractured, alpha=0.3, color='#feca57')
    ax_time.plot(steps, fractured, color='#feca57', linewidth=2, label='Fractured %')
    ax_time.set_xlabel('Step', color='white', fontsize=11)
    ax_time.set_ylabel('%', color='white', fontsize=11)
    ax_time.set_title('Damage Progression', color='white', fontsize=12, fontweight='bold')
    ax_time.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    ax_time.tick_params(colors='#888')
    ax_time.set_ylim(0, max(max(damaged), 10) * 1.1)
    for sp in ax_time.spines.values(): sp.set_edgecolor('#333')

    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print(f"  ✅ {ao_type} detail: {output_path}")


def generate_comparison(all_results, positions, output_path):
    """4-column comparison: full views + zoom + timeline."""
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor('#0a0a12')
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.2,
                  height_ratios=[1, 1, 1, 0.7])

    ao_types = ['A1', 'A2', 'A3', 'A4']

    for col, ao_type in enumerate(ao_types):
        sim, history = all_results[ao_type]
        config = AO_LOAD_CONFIGS[ao_type]
        damage = sim.damage

        # Row 0: Axial full
        ax = fig.add_subplot(gs[0, col])
        grid = create_2d_projection(positions, damage, axis=2)
        ax.imshow(grid.T, origin='lower', cmap=DAMAGE_CMAP, vmin=0, vmax=1)
        ax.set_title(f'{config["name"]}\n({config["mechanism"][:35]})',
                     color='white', fontsize=9, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

        # Row 1: Sagittal full
        ax = fig.add_subplot(gs[1, col])
        grid = create_2d_projection(positions, damage, axis=0)
        ax.imshow(grid.T, origin='lower', cmap=DAMAGE_CMAP, vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])

        # Row 2: Axial zoom
        ax = fig.add_subplot(gs[2, col])
        bounds = find_damage_bounds(positions, damage, threshold=0.1, pad=0.02)
        if bounds:
            grid = create_zoom_projection(positions, damage, axis=2, bounds=bounds)
            ax.imshow(grid.T, origin='lower', cmap=DAMAGE_CMAP, vmin=0, vmax=1)
            for sp in ax.spines.values(): sp.set_edgecolor('#feca57')
        else:
            grid = create_2d_projection(positions, damage, axis=2)
            ax.imshow(grid.T, origin='lower', cmap=DAMAGE_CMAP, vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])

        # Row 3: Timeline
        ax = fig.add_subplot(gs[3, col])
        ax.set_facecolor('#0d0d1a')
        steps = [h['step'] for h in history]
        damaged = [h['damaged_frac'] * 100 for h in history]
        fractured = [h['fractured_frac'] * 100 for h in history]
        ax.fill_between(steps, damaged, alpha=0.3, color='#ff6b6b')
        ax.plot(steps, damaged, color='#ff6b6b', linewidth=1.5)
        ax.fill_between(steps, fractured, alpha=0.3, color='#feca57')
        ax.plot(steps, fractured, color='#feca57', linewidth=1.5)
        ax.set_ylim(0, 105)
        ax.set_xlabel('Step', color='#aaa', fontsize=8)
        ax.tick_params(colors='#666', labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('#333')

        # Final stats
        final = history[-1]
        ax.text(0.95, 0.85,
                f"{final['damaged_frac']*100:.0f}%D\n{final['fractured_frac']*100:.0f}%F",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, color='white', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

    # Row labels
    for row, label in enumerate(['Axial (Full)', 'Sagittal (Full)',
                                 'Axial (Zoom)', 'Damage Timeline']):
        fig.text(0.01, 0.88 - row * 0.23, label,
                 fontsize=11, color='#aaa', rotation=90, va='center')

    fig.suptitle('AO Fracture Comparison — Grid-Based Stress Transfer (Full Physics)',
                 fontsize=16, fontweight='bold', color='white', y=0.98)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print(f"  ✅ Comparison: {output_path}")


def generate_progression_gif(ao_type, positions, regions, output_path, fps=8):
    """Animated GIF: full + zoom side by side for one AO type."""
    config = AO_LOAD_CONFIGS[ao_type]
    sim = BoneFractureSimulator(positions.copy(), seed=42)
    sim.setup_ao_loading(ao_type)
    sim.set_stress_mode('grid')
    sim.enable_anisotropy(ratio=2.0)

    frames = []
    n_steps = 200
    for step in range(n_steps + 1):
        force = config['max_force'] * step / n_steps
        sim.compute_stress(force)
        sim.evolve_damage()

        if step % 4 == 0:
            damage = sim.damage.copy()

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            fig.patch.set_facecolor('#0a0a12')

            # Full axial + sagittal
            for vi, (name, axis) in enumerate([('Axial', 2), ('Sagittal', 0)]):
                grid = create_2d_projection(positions, damage, axis=axis)
                axes[vi].imshow(grid.T, origin='lower', cmap=DAMAGE_CMAP, vmin=0, vmax=1)
                axes[vi].set_title(f'{name}', color='white', fontsize=10, fontweight='bold')
                axes[vi].set_xticks([]); axes[vi].set_yticks([])

            # Zoom axial + sagittal
            bounds = find_damage_bounds(positions, damage, threshold=0.05, pad=0.02)
            for vi, (name, axis) in enumerate([('Axial Zoom', 2), ('Sagittal Zoom', 0)]):
                if bounds:
                    grid = create_zoom_projection(positions, damage, axis, bounds)
                else:
                    grid = create_2d_projection(positions, damage, axis=axis-2 if axis > 0 else axis)
                axes[vi+2].imshow(grid.T, origin='lower', cmap=DAMAGE_CMAP, vmin=0, vmax=1)
                axes[vi+2].set_title(f'{name}', color='#feca57', fontsize=10, fontweight='bold')
                axes[vi+2].set_xticks([]); axes[vi+2].set_yticks([])
                for sp in axes[vi+2].spines.values(): sp.set_edgecolor('#feca57')

            for ax in axes[:2]:
                for sp in ax.spines.values(): sp.set_edgecolor('#333')

            damaged_pct = (damage > 0.01).mean() * 100
            fig.suptitle(
                f'{config["name"]} — Step {step}/{n_steps} | '
                f'Force={force:.2f} | Damaged={damaged_pct:.1f}%',
                color='white', fontsize=12, fontweight='bold', y=0.98
            )
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            img = np.asarray(buf)[:, :, :3].copy()
            frames.append(img)
            plt.close(fig)

    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"  ✅ {ao_type} GIF: {output_path} ({len(frames)} frames)")


if __name__ == '__main__':
    print("=" * 70)
    print("Generating Grid-Only Fracture Visualizations (Full + Zoom)")
    print("=" * 70)

    positions = generate_vertebra_particles(n_particles=50000, seed=42)
    regions = classify_regions(positions)

    # Run all AO types with grid stress
    print("\n1. Running AO simulations (grid stress)...")
    results = {}
    for ao in ['A1', 'A2', 'A3', 'A4']:
        print(f"   {ao}...")
        results[ao] = run_ao(ao, positions, mode='grid', anisotropy=True)

    # Per-AO detail cards
    print("\n2. Generating detail cards (full + zoom)...")
    for ao in ['A1', 'A2', 'A3', 'A4']:
        sim, hist = results[ao]
        generate_ao_card(ao, sim, hist, positions, regions,
                         OUTPUT_DIR / f'v2_{ao}_detail.png')

    # 4-column comparison
    print("\n3. Generating comparison grid...")
    generate_comparison(results, positions, OUTPUT_DIR / 'v2_ao_comparison.png')

    # GIFs for A1 and A4
    print("\n4. Generating progression GIFs...")
    for ao in ['A1', 'A4']:
        generate_progression_gif(ao, positions, regions,
                                 OUTPUT_DIR / f'v2_{ao}_progression.gif', fps=8)

    # Print summary
    print("\n" + "=" * 70)
    print("Summary (Grid-Based Stress):")
    print(f"  {'Type':<25} {'Damaged%':>10} {'Fractured%':>12} {'Fragments':>10}")
    print("  " + "-" * 60)
    for ao in ['A1', 'A2', 'A3', 'A4']:
        sim, hist = results[ao]
        h = hist[-1]
        frag = sim.detect_fragments(damage_threshold=0.9)
        print(f"  {AO_LOAD_CONFIGS[ao]['name']:<25} "
              f"{h['damaged_frac']*100:>9.1f}% "
              f"{h['fractured_frac']*100:>11.1f}% "
              f"{frag['n_fragments']:>10d}")

    print("\n✅ Done! Output:", OUTPUT_DIR)
