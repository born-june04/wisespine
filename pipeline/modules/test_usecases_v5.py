#!/usr/bin/env python3
"""WiseSpine v5 — Use-Case Tests

Three clinical use-case demonstrations:
  1. Multi-patient generalization (3 subjects)
  2. Fracture risk heatmap (Force × BMD matrix)
  3. CT augmentation pipeline (normal → fractured CT)

Usage:
  python pipeline/modules/test_usecases_v5.py --output-dir ./usecase_tests --cuda
"""

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from fracture_engine_v5 import (
    VoxelFEMEngine, CausalParameters, FEMResult,
    _plot_scenario, _plot_fracture_mechanics, _save_progression_gif,
    _AO_COLORS, HAS_MPL
)
from _gen_real_fracture_visuals import load_vertebra

if HAS_MPL:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec


def load_subject(subject_id, verse_root):
    """Load a VerSe subject and return engine-ready data."""
    ct_path = os.path.join(verse_root, 'rawdata', subject_id,
                           f'{subject_id}_dir-ax_ct.nii.gz')
    mask_path = os.path.join(verse_root, 'derivatives', subject_id,
                             f'{subject_id}_dir-ax_seg-vert_msk.nii.gz')
    if not os.path.exists(ct_path) or not os.path.exists(mask_path):
        return None
    ct, mask, label, voxel_spacing = load_vertebra(
        ct_path, mask_path, return_spacing=True)
    mask = (mask > 0).astype(np.int32)
    voxel_size = float(voxel_spacing.mean())
    vol_cm3 = mask.sum() * np.prod(voxel_spacing) / 1000
    return {
        'ct': ct, 'mask': mask, 'label': label,
        'voxel_size': voxel_size, 'voxel_spacing': voxel_spacing,
        'volume_cm3': vol_cm3, 'subject_id': subject_id,
    }


# =============================================================================
#  USE-CASE 1: Multi-Patient Generalization
# =============================================================================

def test_multi_patient(output_dir, verse_root, use_cuda=False):
    """Run same loading on 3 different patients → show patient-specific response."""
    print("\n" + "=" * 70)
    print("USE-CASE 1: Multi-Patient Fracture Risk — Same Load, Different Patients")
    print("=" * 70)

    subjects = ['sub-verse503', 'sub-verse506', 'sub-verse534']
    params = CausalParameters(force_magnitude=5.0, flexion_angle=15.0, bmd_factor=0.7)
    results = []

    for subj in subjects:
        print(f"\n{'─' * 50}")
        print(f"  Patient: {subj}")
        print(f"{'─' * 50}")
        data = load_subject(subj, verse_root)
        if data is None:
            print(f"  ⚠ Skipping {subj} — files not found")
            continue

        ds = max(1, int(np.cbrt(data['mask'].sum() / 200000))) if use_cuda else \
             max(1, int(np.cbrt(data['mask'].sum() / 30000)))

        engine = VoxelFEMEngine(data['mask'], data['ct'],
                                voxel_size_mm=data['voxel_size'],
                                downsample=ds, seed=42, use_cuda=use_cuda)
        engine.set_causal_params(params)
        result = engine.simulate(max_damage_iters=5)
        print(result.summary())

        # Enhanced mechanics visualization per patient
        _plot_fracture_mechanics(engine, output_dir,
                                  filename=f'uc1_{subj}_mechanics.png')

        results.append({
            'subject': subj,
            'ao_type': result.ao_type,
            'confidence': result.confidence,
            'yielded_fraction': result.yielded_fraction,
            'max_displacement': result.max_displacement,
            'max_von_mises': result.max_von_mises,
            'volume_cm3': data['volume_cm3'],
            'n_elements': engine.n_elements,
        })

    # Comparison plot
    if HAS_MPL and len(results) >= 2:
        _plot_patient_comparison(results, output_dir)

    return results


def _plot_patient_comparison(results, output_dir):
    """Bar chart comparing patients side-by-side."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor='#0d1117')
    metrics = [
        ('yielded_fraction', 'Yield Fraction (%)', 100),
        ('max_von_mises', 'Peak σ_vm (MPa)', 1),
        ('max_displacement', 'Max Displacement (mm)', 1),
        ('volume_cm3', 'Vertebra Volume (cm³)', 1),
    ]
    subj_labels = [r['subject'].replace('sub-verse', 'V') for r in results]
    colors = ['#4fc3f7', '#ff7043', '#66bb6a']

    for ax, (key, ylabel, scale) in zip(axes, metrics):
        ax.set_facecolor('#0d1117')
        vals = [r[key] * scale for r in results]
        ao_types = [r['ao_type'] for r in results]
        bars = ax.bar(subj_labels, vals, color=colors[:len(vals)],
                      edgecolor='white', linewidth=0.5, alpha=0.85)
        for bar, ao in zip(bars, ao_types):
            ao_c = _AO_COLORS.get(ao, '#888')
            bar.set_edgecolor(ao_c)
            bar.set_linewidth(2)
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*max(vals),
                    ao, ha='center', va='bottom', fontsize=11,
                    color=ao_c, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#555')
        ax.spines['left'].set_color('#555')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Same Load (5 kN, BMD 0.7) → Different Patients',
                 fontsize=15, color='white', fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(output_dir, 'uc1_patient_comparison.png')
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close(fig)


# =============================================================================
#  USE-CASE 2: Fracture Risk Heatmap (reduced)
# =============================================================================

def test_risk_heatmap(output_dir, verse_root, use_cuda=False):
    """Force × BMD matrix → AO classification heatmap."""
    print("\n" + "=" * 70)
    print("USE-CASE 2: Fracture Risk Heatmap (Force × BMD)")
    print("=" * 70)

    # Load single subject
    data = load_subject('sub-verse503', verse_root)
    if data is None:
        print("  ⚠ sub-verse503 not found")
        return

    ds = max(1, int(np.cbrt(data['mask'].sum() / 200000))) if use_cuda else \
         max(1, int(np.cbrt(data['mask'].sum() / 30000)))

    engine = VoxelFEMEngine(data['mask'], data['ct'],
                            voxel_size_mm=data['voxel_size'],
                            downsample=ds, seed=42, use_cuda=use_cuda)

    forces = [2, 4, 6, 8]
    bmds = [0.3, 0.5, 0.7, 1.0]
    ao_map = {'A0': 0, 'A1': 1, 'A2': 2, 'A3': 3, 'A4': 4}

    matrix = np.zeros((len(bmds), len(forces)))
    yield_matrix = np.zeros((len(bmds), len(forces)))
    ao_labels = [['' for _ in forces] for _ in bmds]

    total = len(forces) * len(bmds)
    count = 0
    t0 = time.time()

    for i, bmd in enumerate(bmds):
        for j, force in enumerate(forces):
            count += 1
            print(f"\n  [{count}/{total}] Force={force}kN, BMD={bmd}")
            engine.set_causal_params(CausalParameters(
                force_magnitude=float(force),
                flexion_angle=10.0,
                bmd_factor=bmd
            ))
            result = engine.simulate(max_damage_iters=5, verbose=False)
            matrix[i, j] = ao_map.get(result.ao_type, 0)
            yield_matrix[i, j] = result.yielded_fraction * 100
            ao_labels[i][j] = result.ao_type
            elapsed = time.time() - t0
            eta = elapsed / count * (total - count)
            print(f"    → {result.ao_type} (yield={result.yielded_fraction*100:.1f}%) "
                  f"[{elapsed/60:.0f}min elapsed, ~{eta/60:.0f}min remaining]")

    # Plot heatmap
    if HAS_MPL:
        _plot_risk_heatmap(matrix, yield_matrix, ao_labels,
                           forces, bmds, output_dir)

    return matrix, ao_labels


def _plot_risk_heatmap(matrix, yield_matrix, ao_labels, forces, bmds, output_dir):
    """Dual heatmap: AO type + yield fraction."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0d1117')

    # Custom colormap for AO types
    from matplotlib.colors import ListedColormap
    ao_cmap = ListedColormap(['#2196F3', '#4CAF50', '#FFC107', '#FF5722', '#9C27B0'])

    # Panel 1: AO Classification
    ax1.set_facecolor('#0d1117')
    im1 = ax1.imshow(matrix, cmap=ao_cmap, vmin=0, vmax=4,
                     aspect='auto', origin='lower')
    for i in range(len(bmds)):
        for j in range(len(forces)):
            ax1.text(j, i, ao_labels[i][j], ha='center', va='center',
                     fontsize=14, fontweight='bold', color='white')
    ax1.set_xticks(range(len(forces)))
    ax1.set_xticklabels([f'{f} kN' for f in forces], color='white')
    ax1.set_yticks(range(len(bmds)))
    ax1.set_yticklabels([f'BMD {b}' for b in bmds], color='white')
    ax1.set_xlabel('Compressive Force', fontsize=12, color='white')
    ax1.set_ylabel('Bone Mineral Density Factor', fontsize=12, color='white')
    ax1.set_title('AO Fracture Classification', fontsize=14,
                  color='white', pad=10)

    # Panel 2: Yield fraction
    ax2.set_facecolor('#0d1117')
    im2 = ax2.imshow(yield_matrix, cmap='YlOrRd', vmin=0, vmax=80,
                     aspect='auto', origin='lower')
    for i in range(len(bmds)):
        for j in range(len(forces)):
            ax2.text(j, i, f'{yield_matrix[i,j]:.0f}%',
                     ha='center', va='center', fontsize=12, color='white')
    ax2.set_xticks(range(len(forces)))
    ax2.set_xticklabels([f'{f} kN' for f in forces], color='white')
    ax2.set_yticks(range(len(bmds)))
    ax2.set_yticklabels([f'BMD {b}' for b in bmds], color='white')
    ax2.set_xlabel('Compressive Force', fontsize=12, color='white')
    ax2.set_title('Yield Fraction', fontsize=14, color='white', pad=10)
    cb = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label('Yield %', fontsize=10, color='white')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

    fig.suptitle('Patient-Specific Fracture Risk Assessment — sub-verse503',
                 fontsize=16, color='white', fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, 'uc2_risk_heatmap.png')
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close(fig)


# =============================================================================
#  USE-CASE 3: CT Augmentation Pipeline
# =============================================================================

def test_augmentation(output_dir, verse_root, use_cuda=False):
    """Generate fractured CT from normal CT using FEM damage."""
    print("\n" + "=" * 70)
    print("USE-CASE 3: CT Augmentation — Normal → Fractured CT")
    print("=" * 70)

    data = load_subject('sub-verse503', verse_root)
    if data is None:
        print("  ⚠ sub-verse503 not found")
        return

    ds = max(1, int(np.cbrt(data['mask'].sum() / 200000))) if use_cuda else \
         max(1, int(np.cbrt(data['mask'].sum() / 30000)))

    engine = VoxelFEMEngine(data['mask'], data['ct'],
                            voxel_size_mm=data['voxel_size'],
                            downsample=ds, seed=42, use_cuda=use_cuda)

    # Generate multiple severities
    scenarios = [
        ('A1_mild', CausalParameters(force_magnitude=3.0, flexion_angle=20.0, bmd_factor=0.8)),
        ('A2_moderate', CausalParameters(force_magnitude=5.0, flexion_angle=10.0, bmd_factor=0.6)),
        ('A4_severe', CausalParameters(force_magnitude=8.0, flexion_angle=5.0, bmd_factor=0.5)),
    ]

    augmented_cts = []
    for name, params in scenarios:
        print(f"\n  Generating: {name}")
        engine.set_causal_params(params)
        result = engine.simulate(max_damage_iters=5)
        print(f"    → {result.ao_type} (yield={result.yielded_fraction*100:.1f}%)")

        # Apply damage to CT: reduce HU in damaged regions
        damage = engine._damage
        dmg_vol = engine._to_3d(damage.astype(np.float32))

        ct_orig = engine._orig_ct if engine.ds > 1 else engine.ct
        mask_orig = engine._orig_mask if engine.ds > 1 else engine.mask

        # Augment: damaged voxels lose bone density (HU drops)
        # d=0 → no change, d=0.5 → 50% HU reduction, d=1 → 90% HU reduction
        ct_augmented = ct_orig.copy()
        bone_region = mask_orig > 0
        hu_reduction = dmg_vol * 0.9  # max 90% reduction
        ct_augmented[bone_region] = ct_orig[bone_region] * (1.0 - hu_reduction[bone_region])

        # Apply displacement warping for geometric deformation
        ux_elem = np.zeros(engine.n_elements)
        uy_elem = np.zeros(engine.n_elements)
        uz_elem = np.zeros(engine.n_elements)
        u = engine._displacement
        for n in range(8):
            ux_elem += u[engine._elem_dofs[:, n*3+0]]
            uy_elem += u[engine._elem_dofs[:, n*3+1]]
            uz_elem += u[engine._elem_dofs[:, n*3+2]]
        ux_elem /= 8.0; uy_elem /= 8.0; uz_elem /= 8.0
        uz_vol = engine._to_3d(uz_elem)

        augmented_cts.append({
            'name': name,
            'ao_type': result.ao_type,
            'ct_augmented': ct_augmented,
            'damage_vol': dmg_vol,
            'uz_vol': uz_vol,
            'yield_frac': result.yielded_fraction,
        })

    # Visualization
    if HAS_MPL:
        _plot_augmentation(ct_orig, mask_orig, augmented_cts, output_dir)

    return augmented_cts


def _plot_augmentation(ct_orig, mask_orig, augmented_cts, output_dir):
    """Show original vs augmented CTs for each severity."""
    n = len(augmented_cts) + 1  # +1 for original
    fig, axes = plt.subplots(3, n, figsize=(5*n, 14), facecolor='#0d1117')
    mid_x = ct_orig.shape[0] // 2
    hu_kw = dict(cmap='bone', origin='lower', vmin=-100, vmax=800, aspect='auto')

    # Row 1: CT sagittal
    axes[0, 0].imshow(ct_orig[mid_x].T, **hu_kw)
    axes[0, 0].set_title('Original CT', fontsize=12, color='#4fc3f7', pad=8)
    axes[0, 0].axis('off')

    # Row 2: CT axial
    mid_z = ct_orig.shape[2] // 2
    axes[1, 0].imshow(ct_orig[:, :, mid_z].T, **hu_kw)
    axes[1, 0].set_title('Axial', fontsize=12, color='white', pad=8)
    axes[1, 0].axis('off')

    # Row 3: Difference (empty for original)
    axes[2, 0].set_facecolor('#0d1117')
    axes[2, 0].text(0.5, 0.5, 'Reference\n(no damage)', ha='center', va='center',
                    fontsize=12, color='#888', transform=axes[2, 0].transAxes)
    axes[2, 0].axis('off')

    for col, aug in enumerate(augmented_cts, 1):
        ao_c = _AO_COLORS.get(aug['ao_type'], '#888')
        ct_aug = aug['ct_augmented']

        # Row 1: Sagittal
        axes[0, col].imshow(ct_aug[mid_x].T, **hu_kw)
        axes[0, col].set_title(f"{aug['name']}\n→ {aug['ao_type']} "
                               f"({aug['yield_frac']*100:.0f}% yield)",
                               fontsize=11, color=ao_c, pad=8)
        axes[0, col].axis('off')

        # Row 2: Axial
        axes[1, col].imshow(ct_aug[:, :, mid_z].T, **hu_kw)
        axes[1, col].axis('off')

        # Row 3: Difference map
        diff = ct_orig[mid_x] - ct_aug[mid_x]
        diff_masked = np.ma.masked_where(mask_orig[mid_x] == 0, diff)
        axes[2, col].imshow(ct_orig[mid_x].T, **hu_kw, alpha=0.2)
        im = axes[2, col].imshow(diff_masked.T, cmap='hot', origin='lower',
                                  vmin=0, vmax=400, aspect='auto', alpha=0.8)
        axes[2, col].set_title('ΔHU (bone density loss)', fontsize=10,
                               color='white', pad=8)
        axes[2, col].axis('off')

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor('#0d1117')

    fig.suptitle('CT Augmentation Pipeline: Normal → Fractured CT',
                 fontsize=16, color='white', fontweight='bold', y=0.98)
    axes[0, 0].annotate('Sagittal', xy=(-0.15, 0.5),
                         xycoords='axes fraction', fontsize=12, color='#888',
                         rotation=90, va='center')
    axes[1, 0].annotate('Axial', xy=(-0.15, 0.5),
                         xycoords='axes fraction', fontsize=12, color='#888',
                         rotation=90, va='center')
    axes[2, 0].annotate('Damage', xy=(-0.15, 0.5),
                         xycoords='axes fraction', fontsize=12, color='#888',
                         rotation=90, va='center')

    plt.tight_layout(rect=[0.02, 0, 1, 0.95])
    out = os.path.join(output_dir, 'uc3_augmentation_pipeline.png')
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close(fig)


# =============================================================================
#  MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='WiseSpine v5 Use-Case Tests')
    parser.add_argument('--output-dir', type=str, default='./usecase_tests')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--skip-patients', action='store_true',
                        help='Skip multi-patient test (~60min)')
    parser.add_argument('--skip-heatmap', action='store_true',
                        help='Skip risk heatmap (~4h)')
    parser.add_argument('--skip-augmentation', action='store_true',
                        help='Skip augmentation test (~60min)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    verse_root = os.path.join(os.path.dirname(__file__), '..', '..',
                              'VerSe', 'dataset-01training')

    t_start = time.time()
    print("=" * 70)
    print("WiseSpine v5 — Clinical Use-Case Tests")
    print("=" * 70)

    if not args.skip_patients:
        test_multi_patient(args.output_dir, verse_root, args.cuda)

    if not args.skip_augmentation:
        test_augmentation(args.output_dir, verse_root, args.cuda)

    if not args.skip_heatmap:
        test_risk_heatmap(args.output_dir, verse_root, args.cuda)

    total = (time.time() - t_start) / 60
    print(f"\n{'=' * 70}")
    print(f"✅ All tests complete! ({total:.0f} min)")
    print(f"   Output: {args.output_dir}/")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
