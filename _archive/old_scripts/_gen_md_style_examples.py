#!/usr/bin/env python3
"""Generate MD-style fracture visualization examples (v3 — fixed coordinates).

Normalized positions for simulation, voxel positions for visualization.
Creates 5 distinct visualization styles for user selection.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pathlib import Path
import nibabel as nib
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection

from fracture_simulator_v2 import BoneFractureSimulator, AO_LOAD_CONFIGS

OUTPUT_DIR = Path(__file__).parent.parent.parent / 'docs' / 'fracture_reports' / 'figures'
VERSE_DIR = Path(__file__).parent.parent.parent / 'VerSe' / 'dataset-01training'
CT_PATH = VERSE_DIR / 'rawdata' / 'sub-verse503' / 'sub-verse503_dir-ax_ct.nii.gz'
MASK_PATH = VERSE_DIR / 'derivatives' / 'sub-verse503' / 'sub-verse503_dir-ax_seg-vert_msk.nii.gz'

FORCE_SCALE = 15.0
AO_TYPES = ['A1', 'A2', 'A3', 'A4']
AO_CLR = {'A1': '#ff4444', 'A2': '#ff8844', 'A3': '#44aaff', 'A4': '#aa44ff'}


def load_and_sample(n_particles=5000):
    """Load vertebra, return normalized and voxel particle positions."""
    ct_nii = nib.load(str(CT_PATH))
    ct = ct_nii.get_fdata().astype(np.float32)
    mask_nii = nib.load(str(MASK_PATH))
    mask_full = mask_nii.get_fdata().astype(np.int16)
    labels = np.unique(mask_full[mask_full > 0])
    label = labels[-2]
    where = np.argwhere(mask_full == label)

    coords = where.astype(np.float32)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(coords), min(n_particles, len(coords)), replace=False)
    voxel = coords[idx]

    # Normalize to [0.05, 0.95] for simulator
    mn, mx = voxel.min(0), voxel.max(0)
    span = mx - mn
    span[span < 1] = 1.0
    norm = 0.05 + 0.9 * (voxel - mn) / span
    return norm, voxel


def run_sim(pos_norm, ao_type, damage_rate=0.05, n_steps=400, record_every=5):
    """Run simulation using normalized positions."""
    config = AO_LOAD_CONFIGS[ao_type]
    sim = BoneFractureSimulator(pos_norm.copy(), seed=42)
    sim.setup_ao_loading(ao_type)
    sim.set_stress_mode('grid')
    sim.enable_anisotropy(ratio=2.0)
    sim._threshold_scale = np.ones(sim.N, dtype=np.float32)
    sim.material['damage_threshold'] = 0.001
    sim.material['damage_rate'] = damage_rate
    sim.material['cod_threshold'] = 0.5
    scaled_force = config['max_force'] * FORCE_SCALE
    hist = sim.run(n_steps=n_steps, max_force=scaled_force,
                   record_every=record_every, verbose=False)
    return hist, config


def ax_style(ax):
    """Apply dark theme to an axes."""
    ax.set_facecolor('#0a0a12')
    ax.tick_params(colors='#666', labelsize=7)
    for sp in ax.spines.values():
        sp.set_color('#333')
    ax.grid(True, alpha=0.06, color='#555')


# ═══════════════════════════════════════════════════════════════
# STYLE A: Phase Portrait — σ-D particle trajectories
# ═══════════════════════════════════════════════════════════════
def style_a(pos_norm, pos_voxel, out):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.patch.set_facecolor('#0a0a12')
    fig.suptitle('Phase Portrait — Particle Stress–Damage Trajectories',
                 fontsize=16, fontweight='bold', color='white', y=0.96)

    rng = np.random.RandomState(42)
    sample = rng.choice(len(pos_norm), 300, replace=False)

    for i, ao in enumerate(AO_TYPES):
        ax = axes[i // 2][i % 2]
        ax_style(ax)
        hist, cfg = run_sim(pos_norm, ao, n_steps=400, record_every=8)

        ms, md = 0, 0
        for pi in sample:
            st = [h['stress'][pi] for h in hist]
            dt = [h['damage'][pi] for h in hist]
            ms, md = max(ms, max(st)), max(md, max(dt))
            fd = dt[-1]
            if fd < 0.005:
                continue
            pts = np.column_stack([st, dt]).reshape(-1, 1, 2)
            if len(pts) < 2:
                continue
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            t = np.linspace(0, 1, len(segs))
            if fd > 0.5:
                c = plt.cm.hot(t); c[:, 3] = 0.7; lw = 1.5
            elif fd > 0.1:
                c = plt.cm.YlOrRd(t * 0.7 + 0.2); c[:, 3] = 0.45; lw = 0.9
            else:
                c = plt.cm.cool(t * 0.5 + 0.3); c[:, 3] = 0.25; lw = 0.5
            ax.add_collection(LineCollection(segs, colors=c, linewidths=lw))

        fs = hist[-1]['stress'][sample]
        fd = hist[-1]['damage'][sample]
        active = fd > 0.01
        if active.any():
            ax.scatter(fs[active], fd[active], s=8, c=fd[active], cmap='hot',
                      vmin=0, vmax=max(md, 0.5), alpha=0.8, zorder=5,
                      edgecolors='white', linewidths=0.3)

        ax.set_xlim(0, max(ms * 1.1, 0.01))
        ax.set_ylim(-0.02, max(md * 1.15, 0.1))
        ax.axhline(y=0.08, color='#ff4444', lw=0.7, ls='--', alpha=0.4)
        ax.axhline(y=0.8, color='#ffaa00', lw=0.7, ls='--', alpha=0.4)
        ax.set_xlabel('Von Mises Stress σ', fontsize=10, color='#ccc')
        ax.set_ylabel('Damage D', fontsize=10, color='#ccc')
        ax.set_title(cfg['name'], fontsize=12, fontweight='bold',
                    color=AO_CLR[ao], pad=8)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=100, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"  ✅ A: {out.name}")


# ═══════════════════════════════════════════════════════════════
# STYLE B: Radial Damage g(r)
# ═══════════════════════════════════════════════════════════════
def style_b(pos_norm, pos_voxel, out):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0a0a12')
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('Radial Damage Distribution g(r) — from Fracture Epicenter',
                 fontsize=15, fontweight='bold', color='white', y=0.97)
    ax_cmp = fig.add_subplot(gs[1, :]); ax_style(ax_cmp)

    for col, ao in enumerate(AO_TYPES):
        hist, cfg = run_sim(pos_norm, ao, n_steps=400, record_every=25)
        fd = hist[-1]['damage']

        # Distances in normalized space
        epi = pos_norm[fd.argmax()]
        dist = np.linalg.norm(pos_norm - epi, axis=1)
        mdist = dist.max()

        ax = fig.add_subplot(gs[0, col]); ax_style(ax)
        bins = np.linspace(0, mdist, 41)
        bc = 0.5 * (bins[:-1] + bins[1:])

        tc = plt.cm.inferno(np.linspace(0.15, 0.95, len(hist)))
        for ti, h in enumerate(hist):
            d = h['damage']
            if d.max() < 0.001:
                continue
            prof = np.array([d[(dist >= bins[j]) & (dist < bins[j+1])].mean()
                            if ((dist >= bins[j]) & (dist < bins[j+1])).any() else 0
                            for j in range(40)])
            a = 0.15 + 0.75 * ti / max(len(hist) - 1, 1)
            ax.plot(bc, prof, color=tc[ti], lw=0.4 + 1.6 * ti / max(len(hist) - 1, 1), alpha=a)

        pf = np.array([fd[(dist >= bins[j]) & (dist < bins[j+1])].mean()
                      if ((dist >= bins[j]) & (dist < bins[j+1])).any() else 0
                      for j in range(40)])
        ax.fill_between(bc, pf, alpha=0.25, color=AO_CLR[ao])
        ax.plot(bc, pf, color=AO_CLR[ao], lw=2.5)
        ax.set_xlabel('r (norm.)', fontsize=8, color='#aaa')
        ax.set_ylabel('D(r)', fontsize=8, color='#aaa')
        ax.set_title(cfg['name'], fontsize=10, fontweight='bold', color=AO_CLR[ao])

        ax_cmp.plot(bc / mdist, pf, color=AO_CLR[ao], lw=2.0, label=cfg['name'])
        ax_cmp.fill_between(bc / mdist, pf, alpha=0.08, color=AO_CLR[ao])

    ax_cmp.set_xlabel('r / R_max', fontsize=10, color='#ccc')
    ax_cmp.set_ylabel('D(r)', fontsize=10, color='#ccc')
    ax_cmp.set_title('All AO Types — Radial Comparison', fontsize=12,
                     fontweight='bold', color='white')
    ax_cmp.legend(fontsize=9, framealpha=0.3, labelcolor='#ccc', facecolor='#1a1a2e')
    fig.savefig(out, dpi=100, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"  ✅ B: {out.name}")


# ═══════════════════════════════════════════════════════════════
# STYLE C: Waterfall Damage Profiles
# ═══════════════════════════════════════════════════════════════
def style_c(pos_norm, pos_voxel, out):
    fig, axes = plt.subplots(1, 4, figsize=(20, 8))
    fig.patch.set_facecolor('#0a0a12')
    fig.suptitle('Damage Propagation — Waterfall (Anterior → Posterior)',
                 fontsize=15, fontweight='bold', color='white', y=0.97)

    # Use normalized Y for profiling
    y = pos_norm[:, 1]

    for col, ao in enumerate(AO_TYPES):
        ax = axes[col]; ax_style(ax)
        hist, cfg = run_sim(pos_norm, ao, n_steps=400, record_every=20)

        bins = np.linspace(y.min(), y.max(), 61)
        bc = 0.5 * (bins[:-1] + bins[1:])
        n_tr = 12
        tidx = np.linspace(0, len(hist) - 1, n_tr).astype(int)
        off_s = 0.08

        for ti_i, ti in enumerate(tidx):
            d = hist[ti]['damage']
            prof = np.array([d[(y >= bins[j]) & (y < bins[j+1])].mean()
                            if ((y >= bins[j]) & (y < bins[j+1])).any() else 0
                            for j in range(60)])
            prof = gaussian_filter1d(prof, sigma=1.2)
            off = ti_i * off_s
            c = plt.cm.inferno(ti_i / max(n_tr - 1, 1) * 0.85 + 0.1)
            ax.fill_between(bc, off, prof + off, alpha=0.65, color=c,
                           zorder=n_tr - ti_i, linewidth=0)
            ax.plot(bc, prof + off, color='white', lw=0.6, alpha=0.9,
                   zorder=n_tr - ti_i + 0.5)
            ax.text(bc[-1] + 0.005, off + prof[-1],
                   f't={hist[ti]["step"]}', fontsize=5, color='#888', va='center')

        ax.set_xlabel('A → P (norm.)', fontsize=9, color='#ccc')
        ax.set_title(cfg['name'], fontsize=11, fontweight='bold',
                    color=AO_CLR[ao], pad=6)
        ax.set_yticks([])
        ax.set_ylabel('Time ↑', fontsize=9, color='#666')

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=100, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"  ✅ C: {out.name}")


# ═══════════════════════════════════════════════════════════════
# STYLE D: Bond-Breaking Network (voxel coords for display)
# ═══════════════════════════════════════════════════════════════
def style_d(pos_norm, pos_voxel, out):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.patch.set_facecolor('#0a0a12')
    fig.suptitle('Bond-Breaking Network — Particle Damage Map',
                 fontsize=15, fontweight='bold', color='white', y=0.96)

    rng = np.random.RandomState(42)
    nv = min(1000, len(pos_norm))
    vi = rng.choice(len(pos_norm), nv, replace=False)
    vp = pos_voxel[vi]

    # Bonds in normalized space but display in voxel space
    tree = cKDTree(pos_norm[vi])
    pairs = list(tree.query_pairs(r=0.05))  # ~5% of unit cube

    x2d, y2d = vp[:, 1], vp[:, 2]

    for i, ao in enumerate(AO_TYPES):
        ax = axes[i // 2][i % 2]; ax_style(ax)
        hist, cfg = run_sim(pos_norm, ao, n_steps=400, record_every=50)
        vd = hist[-1]['damage'][vi]

        # Draw bonds
        for a, b in pairs:
            bd = max(vd[a], vd[b])
            if bd < 0.01:
                ax.plot([x2d[a], x2d[b]], [y2d[a], y2d[b]],
                       color='#334455', lw=0.15, alpha=0.12, zorder=1)
            elif bd < 0.5:
                t = (bd - 0.01) / 0.49
                ax.plot([x2d[a], x2d[b]], [y2d[a], y2d[b]],
                       color=plt.cm.YlOrRd(t * 0.6 + 0.3),
                       lw=0.3 + t * 1.2, alpha=0.2 + t * 0.4, zorder=2)
            else:
                ax.plot([x2d[a], x2d[b]], [y2d[a], y2d[b]],
                       color='#ff1111', lw=1.5, alpha=0.5, ls='--', zorder=3)

        # Particles
        intact = vd < 0.01
        damaged = (vd >= 0.01) & (vd < 0.8)
        broken = vd >= 0.8
        ax.scatter(x2d[intact], y2d[intact], s=2, c='#556677', alpha=0.25, zorder=4)
        if damaged.any():
            ax.scatter(x2d[damaged], y2d[damaged], s=4 + vd[damaged] * 20,
                      c=vd[damaged], cmap='YlOrRd', vmin=0.01, vmax=0.8,
                      alpha=0.7, zorder=5, edgecolors='none')
        if broken.any():
            ax.scatter(x2d[broken], y2d[broken], s=25, c='#ff1111',
                      marker='x', linewidths=1.2, alpha=0.9, zorder=6)

        ni, nd, nb = intact.sum(), damaged.sum(), broken.sum()
        ax.set_title(cfg['name'], fontsize=12, fontweight='bold',
                    color=AO_CLR[ao], pad=8)
        ax.text(0.02, 0.02, f'intact={ni}  damaged={nd}  broken={nb}',
               transform=ax.transAxes, fontsize=7, color='#888')
        ax.set_xlabel('A → P (voxel)', fontsize=9, color='#ccc')
        ax.set_ylabel('I → S (voxel)', fontsize=9, color='#ccc')
        ax.set_aspect('equal')

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=100, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"  ✅ D: {out.name}")


# ═══════════════════════════════════════════════════════════════
# STYLE E: Time-resolved Damage Snapshots (voxel coords)
# ═══════════════════════════════════════════════════════════════
def style_e(pos_norm, pos_voxel, out):
    hist, cfg = run_sim(pos_norm, 'A4', n_steps=400, record_every=10)
    sidx = np.linspace(0, len(hist) - 1, 8).astype(int)

    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor('#0a0a12')
    fig.suptitle(f'{cfg["name"]} — Damage Propagation Snapshots',
                 fontsize=14, fontweight='bold', color='white', y=0.97)
    gs = GridSpec(2, 4, figure=fig, hspace=0.2, wspace=0.12)

    x2d, y2d = pos_voxel[:, 1], pos_voxel[:, 2]
    rng = np.random.RandomState(42)
    nv = min(2500, len(pos_norm))
    vi = rng.choice(len(pos_norm), nv, replace=False)
    gmax = max(h['max_damage'] for h in hist)

    for si, sx in enumerate(sidx):
        h = hist[sx]
        ax = fig.add_subplot(gs[si // 4, si % 4]); ax_style(ax)

        d = h['damage'][vi]
        s = h['stress'][vi]

        bg = d < 0.01
        ax.scatter(x2d[vi[bg]], y2d[vi[bg]], s=0.5, c='#1a1a33', alpha=0.4, zorder=1)

        stressed = (s > 0.001) & (d < 0.05)
        if stressed.any():
            sn = np.clip(s[stressed] / max(s.max(), 0.001), 0, 1)
            ax.scatter(x2d[vi[stressed]], y2d[vi[stressed]], s=1 + sn * 6,
                      c=sn, cmap='Blues', alpha=0.4, vmin=0, vmax=1,
                      zorder=2, edgecolors='none')

        damaged = d >= 0.05
        if damaged.any():
            ax.scatter(x2d[vi[damaged]], y2d[vi[damaged]],
                      s=3 + np.clip(d[damaged] / max(gmax, 0.1), 0, 1) * 20,
                      c=d[damaged], cmap='hot', vmin=0, vmax=max(gmax, 0.1),
                      alpha=0.8, zorder=3, edgecolors='none')

        frac = d >= 0.8
        if frac.any():
            ax.scatter(x2d[vi[frac]], y2d[vi[frac]], s=30, c='#ff0000',
                      marker='x', linewidths=1.2, alpha=0.9, zorder=4)

        pct = (d > 0.01).sum() / len(d) * 100
        ax.set_title(f't={h["step"]}  D_max={h["max_damage"]:.3f}  {pct:.1f}%',
                    fontsize=8, fontweight='bold', color='#ffaa44', pad=3)
        ax.set_aspect('equal')
        if si >= 4: ax.set_xlabel('A → P', fontsize=7, color='#666')
        if si % 4 == 0: ax.set_ylabel('I → S', fontsize=7, color='#666')

    fig.savefig(out, dpi=100, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"  ✅ E: {out.name}")


if __name__ == '__main__':
    print("=" * 60)
    print("MD-Style Visualization Examples (v3)")
    print("=" * 60)

    print("\n1. Loading vertebra...")
    pos_norm, pos_voxel = load_and_sample(5000)
    print(f"   {len(pos_norm)} particles  "
          f"norm range: [{pos_norm.min():.3f}, {pos_norm.max():.3f}]  "
          f"voxel range: [{pos_voxel.min():.0f}, {pos_voxel.max():.0f}]")

    print("\n2. Style A: Phase Portrait...")
    style_a(pos_norm, pos_voxel, OUTPUT_DIR / 'example_style_A_phase.png')

    print("\n3. Style B: Radial Damage g(r)...")
    style_b(pos_norm, pos_voxel, OUTPUT_DIR / 'example_style_B_radial.png')

    print("\n4. Style C: Waterfall...")
    style_c(pos_norm, pos_voxel, OUTPUT_DIR / 'example_style_C_waterfall.png')

    print("\n5. Style D: Bond Network...")
    style_d(pos_norm, pos_voxel, OUTPUT_DIR / 'example_style_D_bonds.png')

    print("\n6. Style E: Snapshots...")
    style_e(pos_norm, pos_voxel, OUTPUT_DIR / 'example_style_E_snapshots.png')

    print("\n✅ All examples generated!")
