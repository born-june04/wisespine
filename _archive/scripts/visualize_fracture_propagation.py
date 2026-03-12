#!/usr/bin/env python3
"""
MD-Style Fracture Propagation Visualizations
=============================================

Inspired by DSSP secondary-structure timeline plots from Molecular Dynamics,
these visualizations show fracture initiation, propagation, and diffusion
across vertebral anatomy using:

  x-axis = vertebral anatomical regions  
  y-axis = time / severity progression

Six prototype styles are generated for user selection.

Author: Wisespine Team
Date: 2026-02-12
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pipeline' / 'modules'))

# ─── Output directory ───────────────────────────────────────────────
OUT_DIR = Path(__file__).resolve().parent.parent / 'docs' / 'fracture_reports' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Constants ───────────────────────────────────────────────────────
# Vertebral regions (x-axis), analogous to residue numbers in MD
VERTEBRA_REGIONS = [
    'Sup. Endplate',
    'Ant. Cortex\n(Upper)',
    'Ant. Cortex\n(Mid)',
    'Ant. Cortex\n(Lower)',
    'Ant. Body\n(Upper)',
    'Ant. Body\n(Mid)',
    'Ant. Body\n(Lower)',
    'Central\nTrabecular',
    'Post. Body\n(Upper)',
    'Post. Body\n(Mid)',
    'Post. Body\n(Lower)',
    'Post. Cortex\n(Upper)',
    'Post. Cortex\n(Mid)',
    'Post. Cortex\n(Lower)',
    'Inf. Endplate',
    'Pedicle\n(Left)',
    'Pedicle\n(Right)',
    'Lamina',
    'Spinous\nProcess',
    'Spinal\nCanal',
]

N_REGIONS = len(VERTEBRA_REGIONS)

# Short labels for tight layouts
SHORT_LABELS = [
    'SupEnd', 'AntCx-U', 'AntCx-M', 'AntCx-L',
    'AntBd-U', 'AntBd-M', 'AntBd-L', 'CenTrb',
    'PstBd-U', 'PstBd-M', 'PstBd-L',
    'PstCx-U', 'PstCx-M', 'PstCx-L',
    'InfEnd', 'PedL', 'PedR', 'Lam', 'SpnPrc', 'Canal'
]

# Severity / time steps (y-axis)
SEVERITY_STEPS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                  0.60, 0.70, 0.80, 0.90, 1.0]
N_STEPS = len(SEVERITY_STEPS)

# ─── Bone state definitions (DSSP-like categories) ──────────────────
STATES = {
    'intact':       0,  # Healthy bone
    'stressed':     1,  # Load accumulation, no fracture yet
    'microfracture': 2, # Trabecular micro-cracks
    'cortical_crack': 3,# Cortical breach begins
    'fracture':     4,  # Clear fracture
    'comminuted':   5,  # Fragmented
    'retropulsed':  6,  # Fragment displaced into canal
    'collapsed':    7,  # Height loss
}

STATE_NAMES = list(STATES.keys())
STATE_COLORS = [
    '#2d5016',  # intact - dark green
    '#f5d742',  # stressed - yellow
    '#ff9800',  # microfracture - orange
    '#e65100',  # cortical crack - dark orange
    '#d32f2f',  # fracture - red
    '#7b1fa2',  # comminuted - purple
    '#1565c0',  # retropulsed - blue
    '#212121',  # collapsed - dark grey
]

STATE_CMAP = ListedColormap(STATE_COLORS)
STATE_NORM = BoundaryNorm(np.arange(-0.5, len(STATE_COLORS) + 0.5, 1), len(STATE_COLORS))


# =============================================================================
# FRACTURE PROPAGATION MODELS
# =============================================================================

def generate_a1_wedge_propagation():
    """A1 Wedge compression: anterior column fails first, posterior preserved."""
    data = np.zeros((N_STEPS, N_REGIONS), dtype=int)
    
    for t_idx, sev in enumerate(SEVERITY_STEPS):
        for r in range(N_REGIONS):
            # Region categories
            is_anterior = r in [1, 2, 3, 4, 5, 6]       # Ant cortex + body
            is_endplate = r in [0, 14]                    # Endplates
            is_central = r == 7                           # Central trabecular
            is_posterior_body = r in [8, 9, 10]           # Post body
            is_posterior_cortex = r in [11, 12, 13]       # Post cortex
            is_posterior_element = r in [15, 16, 17, 18]  # Ped/Lam/SpnPrc
            is_canal = r == 19
            
            if sev < 0.05:
                data[t_idx, r] = STATES['intact']
            elif sev < 0.15:
                # Early: superior endplate stress
                if r == 0:
                    data[t_idx, r] = STATES['stressed']
                elif is_anterior and r <= 3:
                    data[t_idx, r] = STATES['stressed'] if sev > 0.08 else STATES['intact']
                else:
                    data[t_idx, r] = STATES['intact']
            elif sev < 0.30:
                if r == 0:
                    data[t_idx, r] = STATES['microfracture']
                elif is_anterior and r <= 3:
                    data[t_idx, r] = STATES['microfracture']
                elif is_anterior:
                    data[t_idx, r] = STATES['stressed']
                elif is_central:
                    data[t_idx, r] = STATES['stressed']
                else:
                    data[t_idx, r] = STATES['intact']
            elif sev < 0.50:
                if r == 0:
                    data[t_idx, r] = STATES['cortical_crack']
                elif is_anterior and r <= 3:
                    data[t_idx, r] = STATES['cortical_crack']
                elif is_anterior:
                    data[t_idx, r] = STATES['fracture']
                elif is_central:
                    data[t_idx, r] = STATES['microfracture']
                elif is_endplate:
                    data[t_idx, r] = STATES['stressed']
                else:
                    data[t_idx, r] = STATES['intact']
            elif sev < 0.70:
                if is_anterior:
                    data[t_idx, r] = STATES['fracture']
                elif r == 0:
                    data[t_idx, r] = STATES['fracture']
                elif is_central:
                    data[t_idx, r] = STATES['microfracture']
                elif is_posterior_body:
                    data[t_idx, r] = STATES['stressed']
                else:
                    data[t_idx, r] = STATES['intact']
            elif sev < 0.90:
                if is_anterior:
                    data[t_idx, r] = STATES['collapsed']
                elif r == 0:
                    data[t_idx, r] = STATES['fracture']
                elif is_central:
                    data[t_idx, r] = STATES['fracture']
                elif is_posterior_body:
                    data[t_idx, r] = STATES['microfracture']
                elif r == 14:
                    data[t_idx, r] = STATES['stressed']
                else:
                    data[t_idx, r] = STATES['intact']
            else:
                if is_anterior:
                    data[t_idx, r] = STATES['collapsed']
                elif r == 0:
                    data[t_idx, r] = STATES['collapsed']
                elif is_central:
                    data[t_idx, r] = STATES['fracture']
                elif is_posterior_body:
                    data[t_idx, r] = STATES['microfracture']
                elif r == 14:
                    data[t_idx, r] = STATES['microfracture']
                else:
                    data[t_idx, r] = STATES['intact']
    return data


def generate_a2_split_propagation():
    """A2 Split fracture: coronal split, bilateral symmetric."""
    data = np.zeros((N_STEPS, N_REGIONS), dtype=int)
    
    for t_idx, sev in enumerate(SEVERITY_STEPS):
        for r in range(N_REGIONS):
            is_endplate = r in [0, 14]
            is_left_body = r in [4, 5, 6]    # Ant body (left half conceptually)
            is_right_body = r in [8, 9, 10]   # Post body (right half conceptually)
            is_central = r == 7
            is_ant_cortex = r in [1, 2, 3]
            is_post_cortex = r in [11, 12, 13]
            is_posterior_element = r in [15, 16, 17, 18]
            
            if sev < 0.10:
                data[t_idx, r] = STATES['intact']
            elif sev < 0.20:
                if is_central:
                    data[t_idx, r] = STATES['stressed']
                elif is_endplate:
                    data[t_idx, r] = STATES['stressed']
                else:
                    data[t_idx, r] = STATES['intact']
            elif sev < 0.35:
                if is_central:
                    data[t_idx, r] = STATES['microfracture']
                elif is_endplate:
                    data[t_idx, r] = STATES['cortical_crack']
                elif is_left_body or is_right_body:
                    data[t_idx, r] = STATES['stressed']
                else:
                    data[t_idx, r] = STATES['intact']
            elif sev < 0.50:
                if is_central:
                    data[t_idx, r] = STATES['fracture']
                elif is_endplate:
                    data[t_idx, r] = STATES['fracture']
                elif is_left_body or is_right_body:
                    data[t_idx, r] = STATES['microfracture']
                elif is_ant_cortex or is_post_cortex:
                    data[t_idx, r] = STATES['stressed']
                else:
                    data[t_idx, r] = STATES['intact']
            elif sev < 0.70:
                if is_central:
                    data[t_idx, r] = STATES['comminuted']
                elif is_endplate:
                    data[t_idx, r] = STATES['fracture']
                elif is_left_body or is_right_body:
                    data[t_idx, r] = STATES['fracture']
                elif is_ant_cortex or is_post_cortex:
                    data[t_idx, r] = STATES['cortical_crack']
                else:
                    data[t_idx, r] = STATES['intact']
            else:
                if is_central:
                    data[t_idx, r] = STATES['comminuted']
                elif is_endplate:
                    data[t_idx, r] = STATES['fracture']
                elif is_left_body or is_right_body:
                    data[t_idx, r] = STATES['fracture']
                elif is_ant_cortex or is_post_cortex:
                    data[t_idx, r] = STATES['fracture']
                elif r in [15, 16]:  # Pedicles
                    data[t_idx, r] = STATES['stressed']
                else:
                    data[t_idx, r] = STATES['intact']
    return data


def generate_a3_burst_propagation():
    """A3 Incomplete Burst: centrifugal with limited posterior wall damage."""
    data = np.zeros((N_STEPS, N_REGIONS), dtype=int)
    
    for t_idx, sev in enumerate(SEVERITY_STEPS):
        for r in range(N_REGIONS):
            is_endplate = r in [0, 14]
            is_ant_cortex = r in [1, 2, 3]
            is_ant_body = r in [4, 5, 6]
            is_central = r == 7
            is_post_body = r in [8, 9, 10]
            is_post_cortex = r in [11, 12, 13]
            is_pedicle = r in [15, 16]
            is_canal = r == 19
            
            if sev < 0.08:
                data[t_idx, r] = STATES['intact']
            elif sev < 0.15:
                if is_central:
                    data[t_idx, r] = STATES['stressed']
                elif is_endplate:
                    data[t_idx, r] = STATES['stressed']
                else:
                    data[t_idx, r] = STATES['intact']
            elif sev < 0.30:
                if is_central:
                    data[t_idx, r] = STATES['microfracture']
                elif is_endplate:
                    data[t_idx, r] = STATES['cortical_crack']
                elif is_ant_body:
                    data[t_idx, r] = STATES['stressed']
                elif is_post_body:
                    data[t_idx, r] = STATES['stressed']
                else:
                    data[t_idx, r] = STATES['intact']
            elif sev < 0.50:
                if is_central:
                    data[t_idx, r] = STATES['comminuted']
                elif is_endplate:
                    data[t_idx, r] = STATES['fracture']
                elif is_ant_body:
                    data[t_idx, r] = STATES['fracture']
                elif is_ant_cortex:
                    data[t_idx, r] = STATES['cortical_crack']
                elif is_post_body:
                    data[t_idx, r] = STATES['microfracture']
                elif is_post_cortex and r == 12:  # Mid posterior only
                    data[t_idx, r] = STATES['stressed']
                else:
                    data[t_idx, r] = STATES['intact']
            elif sev < 0.70:
                if is_central:
                    data[t_idx, r] = STATES['comminuted']
                elif is_ant_body:
                    data[t_idx, r] = STATES['fracture']
                elif is_ant_cortex:
                    data[t_idx, r] = STATES['fracture']
                elif is_endplate:
                    data[t_idx, r] = STATES['fracture']
                elif is_post_body:
                    data[t_idx, r] = STATES['fracture']
                elif is_post_cortex and r == 12:
                    data[t_idx, r] = STATES['cortical_crack']
                elif is_pedicle:
                    data[t_idx, r] = STATES['stressed']
                else:
                    data[t_idx, r] = STATES['intact']
            else:
                if is_central:
                    data[t_idx, r] = STATES['comminuted']
                elif is_ant_body or is_ant_cortex:
                    data[t_idx, r] = STATES['collapsed']
                elif is_endplate:
                    data[t_idx, r] = STATES['collapsed']
                elif is_post_body:
                    data[t_idx, r] = STATES['fracture']
                elif is_post_cortex and r == 12:
                    data[t_idx, r] = STATES['fracture']
                elif is_post_cortex:
                    data[t_idx, r] = STATES['cortical_crack']
                elif is_pedicle:
                    data[t_idx, r] = STATES['microfracture']
                elif is_canal:
                    data[t_idx, r] = STATES['stressed']  # canal narrowing
                else:
                    data[t_idx, r] = STATES['intact']
    return data


def generate_a4_burst_propagation():
    """A4 Complete Burst: explosive with retropulsion into canal."""
    data = np.zeros((N_STEPS, N_REGIONS), dtype=int)
    
    for t_idx, sev in enumerate(SEVERITY_STEPS):
        for r in range(N_REGIONS):
            is_endplate = r in [0, 14]
            is_ant_cortex = r in [1, 2, 3]
            is_ant_body = r in [4, 5, 6]
            is_central = r == 7
            is_post_body = r in [8, 9, 10]
            is_post_cortex = r in [11, 12, 13]
            is_pedicle = r in [15, 16]
            is_lamina = r == 17
            is_spinous = r == 18
            is_canal = r == 19
            
            if sev < 0.05:
                data[t_idx, r] = STATES['intact']
            elif sev < 0.10:
                if is_central:
                    data[t_idx, r] = STATES['stressed']
                else:
                    data[t_idx, r] = STATES['intact']
            elif sev < 0.20:
                if is_central:
                    data[t_idx, r] = STATES['microfracture']
                elif is_endplate:
                    data[t_idx, r] = STATES['stressed']
                elif is_ant_body or is_post_body:
                    data[t_idx, r] = STATES['stressed']
                else:
                    data[t_idx, r] = STATES['intact']
            elif sev < 0.35:
                if is_central:
                    data[t_idx, r] = STATES['comminuted']
                elif is_endplate:
                    data[t_idx, r] = STATES['cortical_crack']
                elif is_ant_body:
                    data[t_idx, r] = STATES['microfracture']
                elif is_post_body:
                    data[t_idx, r] = STATES['microfracture']
                elif is_ant_cortex:
                    data[t_idx, r] = STATES['stressed']
                elif is_post_cortex:
                    data[t_idx, r] = STATES['stressed']
                elif is_pedicle:
                    data[t_idx, r] = STATES['stressed']
                else:
                    data[t_idx, r] = STATES['intact']
            elif sev < 0.50:
                if is_central:
                    data[t_idx, r] = STATES['comminuted']
                elif is_ant_body or is_ant_cortex:
                    data[t_idx, r] = STATES['fracture']
                elif is_endplate:
                    data[t_idx, r] = STATES['fracture']
                elif is_post_body:
                    data[t_idx, r] = STATES['fracture']
                elif is_post_cortex:
                    data[t_idx, r] = STATES['cortical_crack']
                elif is_pedicle:
                    data[t_idx, r] = STATES['microfracture']
                elif is_canal:
                    data[t_idx, r] = STATES['stressed']
                else:
                    data[t_idx, r] = STATES['intact']
            elif sev < 0.70:
                if is_central:
                    data[t_idx, r] = STATES['comminuted']
                elif is_ant_body or is_ant_cortex:
                    data[t_idx, r] = STATES['collapsed']
                elif is_endplate:
                    data[t_idx, r] = STATES['collapsed']
                elif is_post_body:
                    data[t_idx, r] = STATES['fracture']
                elif is_post_cortex:
                    data[t_idx, r] = STATES['fracture']
                elif is_pedicle:
                    data[t_idx, r] = STATES['fracture']
                elif is_canal:
                    data[t_idx, r] = STATES['retropulsed']
                elif is_lamina:
                    data[t_idx, r] = STATES['stressed']
                else:
                    data[t_idx, r] = STATES['intact']
            else:
                if is_central:
                    data[t_idx, r] = STATES['comminuted']
                elif is_ant_body or is_ant_cortex or is_endplate:
                    data[t_idx, r] = STATES['collapsed']
                elif is_post_body:
                    data[t_idx, r] = STATES['comminuted']
                elif is_post_cortex:
                    data[t_idx, r] = STATES['fracture']
                elif is_pedicle:
                    data[t_idx, r] = STATES['fracture']
                elif is_canal:
                    data[t_idx, r] = STATES['retropulsed']
                elif is_lamina:
                    data[t_idx, r] = STATES['microfracture']
                elif is_spinous:
                    data[t_idx, r] = STATES['stressed']
                else:
                    data[t_idx, r] = STATES['intact']
    return data


# =============================================================================
# PLOT STYLE 1: DSSP-Style Categorical State Map
# =============================================================================

def plot_style1_dssp_state_map(data, title, filename, fracture_info=""):
    """Classic DSSP-like categorical color block plot."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    im = ax.imshow(data, aspect='auto', cmap=STATE_CMAP, norm=STATE_NORM,
                   interpolation='nearest', origin='lower')
    
    # Axes
    ax.set_xticks(range(N_REGIONS))
    ax.set_xticklabels(SHORT_LABELS, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(N_STEPS))
    ax.set_yticklabels([f'{s:.0%}' for s in SEVERITY_STEPS], fontsize=9)
    
    ax.set_xlabel('Vertebral Region', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fracture Severity', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Legend
    patches = [mpatches.Patch(color=STATE_COLORS[i], label=STATE_NAMES[i].replace('_', ' ').title())
               for i in range(len(STATE_NAMES))]
    ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1.01, 1.0),
              fontsize=8, ncol=1, frameon=True, fancybox=True)
    
    # Annotation
    if fracture_info:
        ax.text(0.5, -0.18, fracture_info, transform=ax.transAxes,
                ha='center', fontsize=9, style='italic', color='#555')
    
    # Grid
    ax.set_xticks(np.arange(-0.5, N_REGIONS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N_STEPS, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Style 1 saved: {filename}")


# =============================================================================
# PLOT STYLE 2: HU Diffusion Heatmap (continuous)
# =============================================================================

def generate_hu_diffusion_data(fracture_type='a1'):
    """Generate continuous ΔHU data showing density changes across regions."""
    data = np.zeros((N_STEPS, N_REGIONS))
    
    # Impact epicenter varies by type
    epicenters = {
        'a1': [1, 2, 3, 4, 5, 6],       # Anterior column
        'a2': [0, 7, 14],                 # Central/endplates
        'a3': [7, 4, 5, 6, 8, 9, 10],    # Central radiating
        'a4': list(range(15)),             # Everything
    }
    
    epi = epicenters.get(fracture_type, [7])
    
    for t_idx, sev in enumerate(SEVERITY_STEPS):
        for r in range(N_REGIONS):
            # Distance from epicenter
            min_dist = min(abs(r - e) for e in epi)
            
            # Diffusion with time-delay based on distance
            effective_sev = max(0, sev - min_dist * 0.08)
            
            if r in epi:
                # Epicenter: strong density drop
                data[t_idx, r] = -effective_sev * 600
            elif min_dist <= 2:
                data[t_idx, r] = -effective_sev * 350
            elif min_dist <= 4:
                data[t_idx, r] = -effective_sev * 150
            elif r in [15, 16, 17, 18]:  # Posterior elements
                data[t_idx, r] = -effective_sev * 30
            elif r == 19:  # Canal
                if fracture_type in ['a3', 'a4']:
                    data[t_idx, r] = effective_sev * 200  # Fragment intrusion
                else:
                    data[t_idx, r] = 0
        
        # Add noise for realism
        data[t_idx] += np.random.normal(0, 8, N_REGIONS)
    
    return data


def plot_style2_hu_diffusion(data, title, filename):
    """Continuous heatmap showing ΔHU propagation."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Custom diverging colormap: blue (increase) - white (no change) - red (decrease)
    cmap = plt.cm.RdBu_r
    vmax = max(abs(data.min()), abs(data.max()))
    
    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax,
                   interpolation='bilinear', origin='lower')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('ΔHU from Baseline', fontsize=10)
    
    ax.set_xticks(range(N_REGIONS))
    ax.set_xticklabels(SHORT_LABELS, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(N_STEPS))
    ax.set_yticklabels([f'{s:.0%}' for s in SEVERITY_STEPS], fontsize=9)
    
    ax.set_xlabel('Vertebral Region', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fracture Severity', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Overlay contour lines for fracture front
    contour_data = np.abs(data)
    levels = [50, 100, 200, 400]
    cs = ax.contour(contour_data, levels=levels, colors='black', linewidths=0.5, alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Style 2 saved: {filename}")


# =============================================================================
# PLOT STYLE 3: Fracture Front Kymograph
# =============================================================================

def plot_style3_kymograph(data_states, title, filename):
    """Kymograph showing fracture front propagation as wavefront."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Convert categorical to binary: fractured or not (threshold >= STATES['microfracture'])
    binary = (data_states >= STATES['microfracture']).astype(float)
    
    # Compute "fracture intensity" as weighted sum of state severity
    intensity = data_states.astype(float) / max(STATES.values())
    
    # Use a dramatic fire colormap
    cmap = LinearSegmentedColormap.from_list('fracture_front', 
        ['#1a1a2e', '#16213e', '#0f3460', '#e94560', '#ff6b6b', '#ffd93d', '#ffffff'])
    
    im = ax.imshow(intensity, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                   interpolation='bicubic', origin='lower')
    
    # Overlay fracture front line
    front_line = np.zeros(N_REGIONS)
    for r in range(N_REGIONS):
        col = data_states[:, r]
        fractured = np.where(col >= STATES['microfracture'])[0]
        front_line[r] = fractured[0] if len(fractured) > 0 else N_STEPS
    
    # Plot fracture front as a bold line
    ax.plot(range(N_REGIONS), front_line, 'w-', linewidth=2.5, label='Fracture Front')
    ax.plot(range(N_REGIONS), front_line, 'k--', linewidth=1.0, alpha=0.5)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Fracture Intensity', fontsize=10)
    
    ax.set_xticks(range(N_REGIONS))
    ax.set_xticklabels(SHORT_LABELS, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(N_STEPS))
    ax.set_yticklabels([f'{s:.0%}' for s in SEVERITY_STEPS], fontsize=9)
    
    ax.set_xlabel('Vertebral Region', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fracture Severity', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Style 3 saved: {filename}")


# =============================================================================
# PLOT STYLE 4: Multi-Vertebra Cascade (Inter-vertebral propagation)
# =============================================================================

def plot_style4_cascade(title, filename):
    """Show fracture propagation across multiple vertebrae (L1-L5)."""
    vertebrae = ['T12', 'L1\n(Target)', 'L2', 'L3', 'L4', 'L5']
    n_vert = len(vertebrae)
    time_steps = np.arange(0, 101, 5)  # 0-100 arbitrary time units
    n_t = len(time_steps)
    
    # Generate per-vertebra damage over time
    # Target (L1) gets damaged first, neighbors follow with delay
    damage = np.zeros((n_t, n_vert))
    
    for t_idx, t in enumerate(time_steps):
        # L1 (target): immediate onset
        damage[t_idx, 1] = min(1.0, t / 40.0)
        # T12 (adjacent superior): delayed
        damage[t_idx, 0] = min(0.5, max(0, (t - 30) / 60.0))
        # L2 (adjacent inferior): delayed
        damage[t_idx, 2] = min(0.6, max(0, (t - 25) / 55.0))
        # L3: further delayed, less damage
        damage[t_idx, 3] = min(0.3, max(0, (t - 50) / 80.0))
        # L4, L5: minimal
        damage[t_idx, 4] = min(0.15, max(0, (t - 70) / 100.0))
        damage[t_idx, 5] = min(0.05, max(0, (t - 85) / 120.0))
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [2, 1]})
    
    # Left: Heatmap
    ax1 = axes[0]
    cmap = LinearSegmentedColormap.from_list('cascade', 
        ['#f0f8f0', '#a8e6cf', '#ffd93d', '#ff6b6b', '#c0392b', '#6c3483'])
    
    im = ax1.imshow(damage, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                    interpolation='bilinear', origin='lower')
    
    ax1.set_xticks(range(n_vert))
    ax1.set_xticklabels(vertebrae, fontsize=11, fontweight='bold')
    y_ticks = list(range(0, n_t, 4))
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels([f't={time_steps[i]}' for i in y_ticks], fontsize=9)
    
    ax1.set_xlabel('Vertebral Level', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (arbitrary units)', fontsize=12, fontweight='bold')
    ax1.set_title('Inter-Vertebral Fracture Cascade', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Structural Damage Index', fontsize=10)
    
    # Annotate fracture front wavefront
    for v in range(n_vert):
        col = damage[:, v]
        onset = np.where(col > 0.05)[0]
        if len(onset) > 0:
            ax1.plot(v, onset[0], 'w*', markersize=15, markeredgecolor='black', 
                    markeredgewidth=0.5, zorder=5)
    
    # Right: Line plot showing damage curves
    ax2 = axes[1]
    colors_line = ['#2196F3', '#F44336', '#FF9800', '#4CAF50', '#9C27B0', '#607D8B']
    for v in range(n_vert):
        ax2.plot(time_steps, damage[:, v], '-o', color=colors_line[v],
                linewidth=2, markersize=4, label=vertebrae[v].replace('\n', ' '))
    
    ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Damage Index', fontsize=12, fontweight='bold')
    ax2.set_title('Damage Curves per Vertebra', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.set_ylim(-0.05, 1.1)
    ax2.grid(alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Style 4 saved: {filename}")


# =============================================================================
# PLOT STYLE 5: Stress Transfer Matrix (Contact Map style)
# =============================================================================

def plot_style5_stress_matrix(data_states, title, filename):
    """Contact-map style showing stress transfer between vertebral regions."""
    # Create stress transfer matrix: correlation of damage states between regions
    n = N_REGIONS
    transfer = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Temporal correlation of fracture progression
            series_i = data_states[:, i].astype(float)
            series_j = data_states[:, j].astype(float)
            
            if series_i.std() > 0 and series_j.std() > 0:
                transfer[i, j] = np.corrcoef(series_i, series_j)[0, 1]
            elif series_i.std() == 0 and series_j.std() == 0:
                # Both constant (e.g., both intact)
                transfer[i, j] = 1.0 if np.allclose(series_i, series_j) else 0.0
            else:
                transfer[i, j] = 0.0
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    cmap = LinearSegmentedColormap.from_list('stress',
        ['#1a237e', '#283593', '#4fc3f7', '#ffffff', '#ffab40', '#e65100', '#b71c1c'])
    
    im = ax.imshow(transfer, cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')
    
    ax.set_xticks(range(n))
    ax.set_xticklabels(SHORT_LABELS, rotation=90, ha='center', fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(SHORT_LABELS, fontsize=7)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Temporal Damage Correlation', fontsize=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Vertebral Region', fontsize=11, fontweight='bold')
    ax.set_ylabel('Vertebral Region', fontsize=11, fontweight='bold')
    
    # Annotate high correlations
    for i in range(n):
        for j in range(n):
            if abs(transfer[i, j]) > 0.8 and i != j:
                ax.text(j, i, f'{transfer[i,j]:.1f}', ha='center', va='center',
                       fontsize=5, color='white' if abs(transfer[i,j]) > 0.9 else 'black')
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Style 5 saved: {filename}")


# =============================================================================
# PLOT STYLE 6: Combined 4-panel comparison across all AO types
# =============================================================================

def plot_style6_combined_comparison():
    """4-panel comparison of all AO fracture types in DSSP style."""
    
    generators = {
        'A1 Wedge\n(Flexion-Compression)': generate_a1_wedge_propagation,
        'A2 Split\n(Axial + Coronal)': generate_a2_split_propagation,
        'A3 Inc. Burst\n(High Axial)': generate_a3_burst_propagation,
        'A4 Comp. Burst\n(Explosive)': generate_a4_burst_propagation,
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for idx, (label, gen_fn) in enumerate(generators.items()):
        ax = axes[idx]
        data = gen_fn()
        
        im = ax.imshow(data, aspect='auto', cmap=STATE_CMAP, norm=STATE_NORM,
                       interpolation='nearest', origin='lower')
        
        ax.set_xticks(range(N_REGIONS))
        ax.set_xticklabels(SHORT_LABELS, rotation=60, ha='right', fontsize=6)
        ax.set_yticks(range(N_STEPS))
        ax.set_yticklabels([f'{s:.0%}' for s in SEVERITY_STEPS], fontsize=7)
        
        ax.set_xlabel('Vertebral Region', fontsize=9)
        ax.set_ylabel('Severity', fontsize=9)
        ax.set_title(label, fontsize=12, fontweight='bold', pad=10)
        
        # Grid
        ax.set_xticks(np.arange(-0.5, N_REGIONS, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, N_STEPS, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.3, alpha=0.3)
    
    # Shared legend
    patches = [mpatches.Patch(color=STATE_COLORS[i], label=STATE_NAMES[i].replace('_', ' ').title())
               for i in range(len(STATE_NAMES))]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=10,
              bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=True)
    
    fig.suptitle('AO Fracture Classification — Propagation State Maps\n'
                 '(Inspired by DSSP Secondary Structure Timeline)',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'style6_ao_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Style 6 saved: style6_ao_comparison.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Generating MD-Style Fracture Propagation Visualizations")
    print("=" * 70)
    np.random.seed(42)
    
    # Generate propagation data for all types
    data_a1 = generate_a1_wedge_propagation()
    data_a2 = generate_a2_split_propagation()
    data_a3 = generate_a3_burst_propagation()
    data_a4 = generate_a4_burst_propagation()
    
    # ── Style 1: DSSP State Maps (one per type) ──
    print("\n📊 Style 1: DSSP-Style State Maps")
    plot_style1_dssp_state_map(
        data_a1, 
        'A1 Wedge Compression — Fracture State Map',
        'style1_dssp_A1_Wedge.png',
        'Anterior column fails first under flexion-compression. Posterior wall preserved (Denis 1983).'
    )
    plot_style1_dssp_state_map(
        data_a4,
        'A4 Complete Burst — Fracture State Map',
        'style1_dssp_A4_Burst.png',
        'Explosive axial load → centrifugal propagation → retropulsion into canal (Wilcox 2003).'
    )
    
    # ── Style 2: ΔHU Diffusion Heatmaps ──
    print("\n🔥 Style 2: ΔHU Diffusion Heatmaps")
    for ftype, label in [('a1', 'A1 Wedge'), ('a4', 'A4 Complete Burst')]:
        hu_data = generate_hu_diffusion_data(ftype)
        plot_style2_hu_diffusion(
            hu_data,
            f'{label} — HU Density Change Propagation',
            f'style2_hu_diffusion_{ftype}.png'
        )
    
    # ── Style 3: Kymograph ──
    print("\n🌊 Style 3: Fracture Front Kymographs")
    plot_style3_kymograph(
        data_a1,
        'A1 Wedge — Fracture Front Propagation',
        'style3_kymograph_A1.png'
    )
    plot_style3_kymograph(
        data_a4,
        'A4 Complete Burst — Fracture Front Propagation',
        'style3_kymograph_A4.png'
    )
    
    # ── Style 4: Multi-Vertebra Cascade ──
    print("\n🏗️ Style 4: Multi-Vertebra Cascade")
    plot_style4_cascade(
        'A4 Burst Fracture — Inter-Vertebral Damage Cascade',
        'style4_cascade.png'
    )
    
    # ── Style 5: Stress Transfer Matrix ──
    print("\n🔗 Style 5: Stress Transfer Matrices")
    plot_style5_stress_matrix(
        data_a1,
        'A1 Wedge — Stress Transfer Correlation Matrix',
        'style5_stress_A1.png'
    )
    plot_style5_stress_matrix(
        data_a4,
        'A4 Complete Burst — Stress Transfer Correlation Matrix',
        'style5_stress_A4.png'
    )
    
    # ── Style 6: Combined 4-panel ──
    print("\n🎯 Style 6: Combined AO Comparison")
    plot_style6_combined_comparison()
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ All prototypes generated! Files in:")
    print(f"   {OUT_DIR}")
    print("\nGenerated plots:")
    for f in sorted(OUT_DIR.glob('style*')):
        print(f"   📊 {f.name}")
    print("=" * 70)
