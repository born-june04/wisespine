
import numpy as np
import nibabel as nib
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from scipy.ndimage import label as label_cc

# Import shared modules (co-located in pipeline/)
sys.path.insert(0, str(Path(__file__).parent))
from modules.tumor_synthesis import generate_lytic_lesion, generate_blastic_lesion

def run_tumor_simulation(ct_path, seg_path, hw_path, out_path, angle=60):
    ct_path = Path(ct_path)
    if seg_path: seg_path = Path(seg_path)
    if hw_path: hw_path = Path(hw_path)
    out_path = Path(out_path)
    
    if not ct_path.exists():
        print(f"CT not found: {ct_path}")
        return

    use_hardware = False
    if seg_path and seg_path.exists():
        print(f"Loading Segmentation: {seg_path}")
        nii_mask = nib.load(seg_path)
    elif hw_path and hw_path.exists():
        print(f"Segmentation not found. Using Hardware {hw_path} as proxy.")
        use_hardware = True
        nii_mask = nib.load(hw_path)
    else:
        print("No Segmentation or Hardware mask found. Cannot place tumors.")
        return

    nii_ct = nib.load(ct_path)
    ct_data = np.asanyarray(nii_ct.dataobj).astype(np.float32)
    mask_data = np.asanyarray(nii_mask.dataobj).astype(np.uint8)
    
    print("Simulating Tumors...")
    
    # Labeling
    if use_hardware:
        print("Labeling hardware components...")
        labeled_mask, n_labels = label_cc(mask_data > 0)
        labels = list(range(1, n_labels + 1))
        mask_data = labeled_mask
    else:
        labels = np.unique(mask_data)
        labels = labels[labels > 0]
    
    if len(labels) == 0:
        print("No labels found.")
        return
        
    print(f"Found {len(labels)} potential sites.")
    
    # Select 2 random levels for Lytic, 2 for Blastic
    targets = random.sample(list(labels), min(len(labels), 4))
    lytic_targets = targets[:2]
    blastic_targets = targets[2:] if len(targets) > 2 else []
    
    lesion_locs = []
    
    for label in lytic_targets:
        print(f"Generating Lytic Lesion at Label {label}...")
        coords = np.argwhere(mask_data == label)
        if len(coords) == 0: continue
        c = coords.mean(axis=0)
        center_f = tuple(map(int, c))
        
        try:
            ct_data, diff_mask = generate_lytic_lesion(ct_data, mask_data, center_f, radius_mm=15.0, irregularity=0.6)
            d_coords = np.argwhere(diff_mask)
            if len(d_coords) > 0:
                dc = d_coords.mean(axis=0).astype(int)
                lesion_locs.append(('Lytic', dc, label))
        except Exception as e:
            print(f"Failed Lytic: {e}")
            
    for label in blastic_targets:
        print(f"Generating Blastic Lesion at Label {label}...")
        coords = np.argwhere(mask_data == label)
        if len(coords) == 0: continue
        c = coords.mean(axis=0)
        center_f = tuple(map(int, c))
        
        try:
            ct_data, diff_mask = generate_blastic_lesion(ct_data, mask_data, center_f, radius_mm=12.0, density_increase=900.0)
            d_coords = np.argwhere(diff_mask)
            if len(d_coords) > 0:
                dc = d_coords.mean(axis=0).astype(int)
                lesion_locs.append(('Blastic', dc, label))
        except Exception as e:
            print(f"Failed Blastic: {e}")
            
    # Save Volume
    print(f"Saving {out_path}...")
    nib.save(nib.Nifti1Image(ct_data, nii_ct.affine), out_path)
    
    # Visualization (PNG in same dir as out_path)
    if len(lesion_locs) > 0:
        print("Generating Visualization...")
        n_plots = len(lesion_locs)
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
        if n_plots == 1: axes = [axes]
        
        for i, (l_type, loc, lbl) in enumerate(lesion_locs):
            z_slice = loc[2]
            img = ct_data[:, :, z_slice].T 
            ax = axes[i]
            ax.imshow(img, cmap='gray', origin='lower', vmin=-200, vmax=1500)
            ax.set_title(f"{l_type} Lesion (Label {lbl})\nAxial Z={z_slice}", fontsize=14)
            circ = plt.Circle((loc[1], loc[0]), 30, color='red' if l_type=='Lytic' else 'cyan', fill=False, linewidth=2)
            ax.add_patch(circ)
            
        plt.tight_layout()
        out_png = out_path.parent / f"{out_path.stem}_viz.png"
        plt.savefig(str(out_png), dpi=150)
        print(f"Saved {out_png}")
        plt.close(fig)

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
    angle = 60
    
    ct_path = root / f"scoliosis_cobb{angle}.nii.gz"
    seg_path = root / f"scoliosis_cobb{angle}_seg.nii.gz"
    hw_path = root / f"scoliosis_cobb{angle}_hardware.nii.gz"
    out_path = root / f"scoliosis_cobb{angle}_tumors.nii.gz"
    
    run_tumor_simulation(ct_path, seg_path, hw_path, out_path, angle)

if __name__ == "__main__":
    main()
