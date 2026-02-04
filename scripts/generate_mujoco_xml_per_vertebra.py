#!/usr/bin/env python3
"""
Generate MuJoCo XML with SEPARATE bodies for each vertebra.
Each vertebra gets:
- Its own body
- A free joint (6-DOF)
- Visual mesh
- Collision geometry (optional)

This allows RL to apply forces to individual vertebrae.
"""

import argparse
import sys
from pathlib import Path
import trimesh
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MuJoCo XML with per-vertebra bodies")
    parser.add_argument("--obj", type=str, required=True, help="Input OBJ file with all vertebrae")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for XML and meshes")
    parser.add_argument("--scale", type=float, default=0.001, help="Scale factor (0.001 = mm to m)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    obj_path = Path(args.obj)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading OBJ: {obj_path}")
    scene = trimesh.load(obj_path, force='scene')
    
    # Extract vertebrae by object name
    vertebrae = {}
    for name, geom in scene.geometry.items():
        # Parse vertebra label from name (e.g., "mat_GT_C1", "mat_TS_L5")
        if '_GT_' in name:
            # Get vertebra label (e.g., "mat_GT_C1" → "C1")
            parts = name.split('_')
            if len(parts) >= 3 and parts[1] == 'GT':
                label = parts[2]  # e.g., "C1", "L5", "T12"
                vertebrae[label] = geom
    
    if not vertebrae:
        print("✗ No GT vertebrae found in OBJ file")
        sys.exit(1)
    
    print(f"✓ Found {len(vertebrae)} vertebrae: {sorted(vertebrae.keys())}")
    
    # Export each vertebra mesh
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    
    vertebra_info = []
    for label, mesh in sorted(vertebrae.items()):
        # Scale and center
        mesh.apply_scale(args.scale)
        centroid = mesh.centroid
        
        # Export as OBJ
        mesh_file = mesh_dir / f"{label}.obj"
        mesh.export(str(mesh_file))
        
        vertebra_info.append({
            'label': label,
            'mesh_file': f"meshes/{label}.obj",
            'pos': centroid,
            'mass': mesh.volume * 1000  # Bone density ~1000 kg/m³
        })
        print(f"  {label}: pos={centroid}, mass={vertebra_info[-1]['mass']:.3f} kg")
    
    # Generate MuJoCo XML
    xml_path = out_dir / "spine_per_vertebra.xml"
    
    with open(xml_path, 'w') as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write('<mujoco model="spine_per_vertebra">\n')
        f.write('  <compiler angle="radian" meshdir="." />\n')
        f.write('  <option timestep="0.002" gravity="0 0 -9.81" />\n')
        f.write('\n')
        
        # Assets (meshes)
        f.write('  <asset>\n')
        for info in vertebra_info:
            f.write(f'    <mesh name="{info["label"]}" file="{info["mesh_file"]}" scale="1 1 1" />\n')
        f.write('  </asset>\n')
        f.write('\n')
        
        # Worldbody
        f.write('  <worldbody>\n')
        f.write('    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1" />\n')
        f.write('    <geom type="plane" size="1 1 0.1" rgba=".9 .9 .9 1" />\n')
        f.write('\n')
        
        # Each vertebra as a separate body with free joint
        for info in vertebra_info:
            label = info['label']
            pos = info['pos']
            mass = info['mass']
            
            f.write(f'    <body name="{label}" pos="{pos[0]} {pos[1]} {pos[2]}">\n')
            f.write(f'      <freejoint name="{label}_joint" />\n')
            f.write(f'      <geom name="{label}_geom" type="mesh" mesh="{label}" '
                    f'rgba="0.8 0.8 0.9 1" mass="{mass}" />\n')
            f.write(f'    </body>\n')
            f.write('\n')
        
        f.write('  </worldbody>\n')
        f.write('</mujoco>\n')
    
    print(f"\n✓ MuJoCo XML saved: {xml_path}")
    print(f"  Bodies: {len(vertebra_info)} (1 per vertebra)")
    print(f"  Joints: {len(vertebra_info)} (free joints, 6-DOF each)")
    print(f"  Total DOF: {len(vertebra_info) * 6}")


if __name__ == "__main__":
    main()

