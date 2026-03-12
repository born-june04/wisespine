#!/usr/bin/env python3
"""Generate vertebra deformation visualizations showing how fractures actually look.

Shows:
1. 3D-like scatter of vertebra particles (intact → fractured)
2. Deformation applied: crack opening, fragment separation
3. Animated GIF for each AO type
4. Before/After comparison
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pathlib import Path
from fracture_simulator_v2 import (
    BoneFractureSimulator, generate_vertebra_particles, classify_regions,
    AO_LOAD_CONFIGS, DAMAGE_CMAP,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import imageio

OUTPUT_DIR = Path(__file__).parent.parent.parent / 'docs' / 'fracture_reports' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def scatter_vertebra(ax, positions, damage, view='axial', title='',
                     deformed_positions=None, point_size=0.3, alpha=0.6):
    """Draw vertebra as 3D-ish scatter with damage coloring.

    Shows actual particle positions (optionally deformed).
    """
    pos = deformed_positions if deformed_positions is not None else positions

    if view == 'axial':
        x, y = pos[:, 0], pos[:, 1]  # top-down
        xlabel, ylabel = 'Anterior ← → Posterior', 'Left ← → Right'
    elif view == 'sagittal':
        x, y = pos[:, 1], pos[:, 2]  # side view
        xlabel, ylabel = 'Left ← → Right', 'Inferior ← → Superior'
    elif view == 'coronal':
        x, y = pos[:, 0], pos[:, 2]  # front view
        xlabel, ylabel = 'Anterior ← → Posterior', 'Inferior ← → Superior'
    else:
        x, y = pos[:, 0], pos[:, 2]
        xlabel, ylabel = 'X', 'Z'

    # Sort by depth so closer particles render on top
    depth_axis = {'axial': 2, 'sagittal': 0, 'coronal': 1}[view]
    depth = pos[:, depth_axis]
    sort_idx = np.argsort(depth)

    # Color: intact=bone color, damaged=yellow→red, fractured=white/black
    sc = ax.scatter(
        x[sort_idx], y[sort_idx],
        c=damage[sort_idx],
        cmap=DAMAGE_CMAP,
        vmin=0, vmax=1,
        s=point_size,
        alpha=alpha,
        edgecolors='none',
        rasterized=True,
    )

    ax.set_facecolor('#0a0a12')
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel, fontsize=8, color='#666')
    ax.set_ylabel(ylabel, fontsize=8, color='#666')
    ax.set_title(title, color='white', fontsize=11, fontweight='bold')
    ax.tick_params(colors='#444', labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')
    return sc


def generate_deformation_card(ao_type, output_path):
    """Before/After card showing intact vs fractured vertebra."""
    config = AO_LOAD_CONFIGS[ao_type]
    positions = generate_vertebra_particles(n_particles=50000, seed=42)

    sim = BoneFractureSimulator(positions.copy(), seed=42)
    sim.setup_ao_loading(ao_type)
    sim.set_stress_mode('grid')
    sim.enable_anisotropy(ratio=2.0)

    # Run to completion
    history = sim.run(n_steps=200, max_force=config['max_force'],
                      record_every=4, verbose=False)

    # Apply deformation (crack opening + fragment separation)
    sim.compute_deformation()
    deformed = sim.positions.copy()
    damage = sim.damage.copy()

    # Amplify deformation for visualization (×5)
    displacement = deformed - positions
    vis_deformed = positions + displacement * 5.0

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#0a0a12')

    fig.suptitle(
        f'{config["name"]}',
        fontsize=22, fontweight='bold', color='white', y=0.97
    )
    fig.text(0.5, 0.935,
             f'{config["mechanism"]}  |  Grid P2G/G2P Stress  |  '
             f'Deformation ×5 amplified',
             fontsize=12, color='#aaa', ha='center')

    views = ['coronal', 'sagittal', 'axial']
    view_names = ['Coronal (Front)', 'Sagittal (Side)', 'Axial (Top)']

    # Row 1: Intact vertebra (damage=0, original positions)
    zeros = np.zeros(len(positions))
    for col, (view, vname) in enumerate(zip(views, view_names)):
        ax = fig.add_subplot(2, 3, col + 1)
        scatter_vertebra(ax, positions, zeros, view=view,
                        title=f'Intact — {vname}', point_size=0.4, alpha=0.7)

    # Row 2: Fractured vertebra (with deformation)
    for col, (view, vname) in enumerate(zip(views, view_names)):
        ax = fig.add_subplot(2, 3, col + 4)
        sc = scatter_vertebra(ax, positions, damage, view=view,
                             title=f'Fractured — {vname}',
                             deformed_positions=vis_deformed,
                             point_size=0.4, alpha=0.7)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.3])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Damage (D)', color='white', fontsize=10)
    cbar.ax.tick_params(colors='white')

    # Stats text
    final = history[-1]
    frag = sim.detect_fragments(damage_threshold=0.9)
    fig.text(0.92, 0.55,
             f"Damaged: {final['damaged_frac']*100:.1f}%\n"
             f"Fractured: {final['fractured_frac']*100:.1f}%\n"
             f"Fragments: {frag['n_fragments']}\n"
             f"Max D: {final['max_damage']:.3f}",
             fontsize=10, color='white', va='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#444'))

    fig.tight_layout(rect=[0, 0, 0.90, 0.92])
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"  ✅ {ao_type} deformation card: {output_path}")


def generate_deformation_gif(ao_type, output_path, fps=6):
    """Animated GIF: vertebra progressively fracturing with visible deformation."""
    config = AO_LOAD_CONFIGS[ao_type]
    positions = generate_vertebra_particles(n_particles=50000, seed=42)

    sim = BoneFractureSimulator(positions.copy(), seed=42)
    sim.setup_ao_loading(ao_type)
    sim.set_stress_mode('grid')
    sim.enable_anisotropy(ratio=2.0)

    n_steps = 200
    frames = []

    for step in range(n_steps + 1):
        force = config['max_force'] * step / n_steps
        sim.compute_stress(force)
        sim.evolve_damage()

        if step % 5 == 0:
            # Apply deformation
            sim.compute_deformation()
            deformed = sim.positions.copy()
            damage = sim.damage.copy()

            # Amplify for visibility
            displacement = deformed - positions
            vis_deformed = positions + displacement * 5.0

            damaged_pct = (damage > sim.material['damage_threshold']).mean() * 100
            fractured_pct = (damage > sim.material['cod_threshold']).mean() * 100

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.patch.set_facecolor('#0a0a12')

            for vi, (view, vname) in enumerate(zip(
                ['coronal', 'sagittal', 'axial'],
                ['Coronal', 'Sagittal', 'Axial']
            )):
                scatter_vertebra(axes[vi], positions, damage, view=view,
                               title=vname,
                               deformed_positions=vis_deformed,
                               point_size=0.5, alpha=0.7)

            fig.suptitle(
                f'{config["name"]} — Step {step}/{n_steps}  |  '
                f'Force={force:.2f}  |  '
                f'Damaged={damaged_pct:.1f}%  |  Fractured={fractured_pct:.1f}%',
                color='white', fontsize=13, fontweight='bold', y=0.98
            )
            fig.tight_layout(rect=[0, 0, 1, 0.93])

            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            img = np.asarray(buf)[:, :, :3].copy()
            frames.append(img)
            plt.close(fig)

    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"  ✅ {ao_type} GIF: {output_path} ({len(frames)} frames)")


def generate_all_ao_deformation(output_path):
    """4-column comparison: intact (top) vs fractured (bottom) for all AO types."""
    positions = generate_vertebra_particles(n_particles=50000, seed=42)

    fig = plt.figure(figsize=(22, 12))
    fig.patch.set_facecolor('#0a0a12')

    ao_types = ['A1', 'A2', 'A3', 'A4']

    for col, ao_type in enumerate(ao_types):
        config = AO_LOAD_CONFIGS[ao_type]
        sim = BoneFractureSimulator(positions.copy(), seed=42)
        sim.setup_ao_loading(ao_type)
        sim.set_stress_mode('grid')
        sim.enable_anisotropy(ratio=2.0)
        history = sim.run(n_steps=200, max_force=config['max_force'],
                          record_every=50, verbose=False)
        sim.compute_deformation()
        deformed = sim.positions.copy()
        damage = sim.damage.copy()
        displacement = deformed - positions
        vis_deformed = positions + displacement * 5.0

        final = history[-1]

        # Row 0: Intact coronal
        ax = fig.add_subplot(2, 4, col + 1)
        scatter_vertebra(ax, positions, np.zeros(len(positions)),
                        view='coronal',
                        title=f'{config["name"]}\n(Intact)',
                        point_size=0.3, alpha=0.6)

        # Row 1: Fractured coronal
        ax = fig.add_subplot(2, 4, col + 5)
        scatter_vertebra(ax, positions, damage, view='coronal',
                        title=f'Fractured ({final["damaged_frac"]*100:.0f}%D, '
                              f'{final["fractured_frac"]*100:.0f}%F)',
                        deformed_positions=vis_deformed,
                        point_size=0.3, alpha=0.6)

    fig.suptitle(
        'AO Fracture Types — Intact vs Fractured Vertebra (Deformation ×5)',
        fontsize=16, fontweight='bold', color='white', y=0.98
    )
    fig.text(0.5, 0.94,
             'Top: Intact  |  Bottom: After grid-based fracture simulation with anisotropy',
             fontsize=11, color='#aaa', ha='center')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"  ✅ All AO deformation comparison: {output_path}")


if __name__ == '__main__':
    print("=" * 70)
    print("Generating Vertebra Deformation Visualizations")
    print("=" * 70)

    # Per-AO deformation cards (before/after)
    print("\n1. Generating deformation cards...")
    for ao in ['A1', 'A2', 'A3', 'A4']:
        generate_deformation_card(ao, OUTPUT_DIR / f'v2_{ao}_deform.png')

    # 4-column comparison
    print("\n2. Generating all-AO deformation comparison...")
    generate_all_ao_deformation(OUTPUT_DIR / 'v2_deform_comparison.png')

    # Animated GIFs
    print("\n3. Generating progression GIFs...")
    for ao in ['A1', 'A2', 'A3', 'A4']:
        generate_deformation_gif(ao, OUTPUT_DIR / f'v2_{ao}_fracture.gif', fps=6)

    print(f"\n✅ Done! Output: {OUTPUT_DIR}")
