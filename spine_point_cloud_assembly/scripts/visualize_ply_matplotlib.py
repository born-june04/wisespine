#!/usr/bin/env python3
"""
Visualize PLY files using matplotlib (for remote server)
Shows individual vertebrae and complete assembled spine
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import json
import sys

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("WARNING: PyVista not available, will try to load PLY manually")


def load_ply_file(ply_path: Path):
    """Load PLY file and extract points and colors"""
    if PYVISTA_AVAILABLE:
        mesh = pv.read(str(ply_path))
        points = mesh.points  # (N, 3)
        
        # Get colors if available
        if 'RGB' in mesh.point_data:
            colors = mesh.point_data['RGB'] / 255.0  # Normalize to 0-1
        elif 'vertebra_type' in mesh.point_data:
            # Generate colors from vertebra_type
            labels = mesh.point_data['vertebra_type']
            unique_labels = np.unique(labels)
            num_labels = len(unique_labels)
            
            import matplotlib.cm as cm
            colormap = cm.get_cmap('tab20', num_labels)
            label_to_color = {label: colormap(i)[:3] for i, label in enumerate(unique_labels)}
            colors = np.array([label_to_color[label] for label in labels])
        else:
            colors = None
        
        return points, colors, mesh.point_data.get('vertebra_type', None)
    else:
        # Manual PLY parsing (simple ASCII format)
        print("WARNING: Manual PLY parsing not implemented. Install PyVista.")
        return None, None, None


def visualize_vertebrae_separate(points, labels, colors, output_path: Path):
    """Visualize each vertebra separately"""
    unique_labels = np.unique(labels)
    num_vertebrae = len(unique_labels)
    
    # Calculate grid size
    cols = min(4, num_vertebrae)
    rows = (num_vertebrae + cols - 1) // cols
    
    fig = plt.figure(figsize=(cols * 4, rows * 4))
    
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        vertebra_points = points[mask]
        vertebra_colors = colors[mask] if colors is not None else None
        
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        
        if vertebra_colors is not None and len(vertebra_colors) > 0:
            ax.scatter(
                vertebra_points[:, 0],
                vertebra_points[:, 1],
                vertebra_points[:, 2],
                c=vertebra_colors,
                s=1,
                alpha=0.6,
            )
        else:
            # Use a single color for this vertebra
            import matplotlib.cm as cm
            try:
                colormap = plt.colormaps.get_cmap('tab20')
            except AttributeError:
                colormap = cm.get_cmap('tab20')
            color = colormap(idx / max(num_vertebrae - 1, 1))[:3]
            ax.scatter(
                vertebra_points[:, 0],
                vertebra_points[:, 1],
                vertebra_points[:, 2],
                c=[color],
                s=1,
                alpha=0.6,
            )
        
        ax.set_title(f'Vertebra Type {int(label)}', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        max_range = np.array([
            vertebra_points[:, 0].max() - vertebra_points[:, 0].min(),
            vertebra_points[:, 1].max() - vertebra_points[:, 1].min(),
            vertebra_points[:, 2].max() - vertebra_points[:, 2].min()
        ]).max() / 2.0
        mid_x = (vertebra_points[:, 0].max() + vertebra_points[:, 0].min()) * 0.5
        mid_y = (vertebra_points[:, 1].max() + vertebra_points[:, 1].min()) * 0.5
        mid_z = (vertebra_points[:, 2].max() + vertebra_points[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.suptitle('Individual Vertebrae', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    separate_path = output_path.parent / f"{output_path.stem}_individual_vertebrae.png"
    plt.savefig(separate_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved individual vertebrae visualization: {separate_path}")
    plt.close()


def visualize_assembled_spine(points, labels, colors, output_path: Path, use_mesh: bool = True):
    """Visualize complete assembled spine as mesh or point cloud"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if use_mesh and PYVISTA_AVAILABLE:
        # Convert point cloud to mesh using PyVista
        try:
            point_cloud = pv.PolyData(points)
            
            # Try surface reconstruction
            try:
                # Use Delaunay 3D for mesh generation
                mesh = point_cloud.delaunay_3d()
                # Extract surface
                surface = mesh.extract_surface()
                
                # Get face colors from point colors
                if colors is not None:
                    # Map point colors to face colors (average of vertex colors)
                    face_colors = []
                    for i in range(surface.n_faces):
                        face = surface.get_cell(i)
                        point_ids = face.point_ids
                        if len(point_ids) > 0:
                            face_color = colors[point_ids].mean(axis=0)
                            face_colors.append(face_color)
                    if face_colors:
                        surface['colors'] = np.array(face_colors)
                
                # Plot mesh
                vertices = surface.points
                faces = surface.faces.reshape(-1, 4)[:, 1:]  # Remove first column (vertex count)
                
                if colors is not None and 'colors' in surface.cell_data:
                    ax.plot_trisurf(
                        vertices[:, 0],
                        vertices[:, 1],
                        vertices[:, 2],
                        triangles=faces,
                        facecolors=surface['colors'],
                        alpha=0.8,
                        edgecolor='none',
                    )
                else:
                    ax.plot_trisurf(
                        vertices[:, 0],
                        vertices[:, 1],
                        vertices[:, 2],
                        triangles=faces,
                        alpha=0.8,
                        edgecolor='none',
                    )
            except Exception as e:
                print(f"WARNING: Mesh reconstruction failed ({e}), falling back to point cloud")
                use_mesh = False
        
        except Exception as e:
            print(f"WARNING: PyVista mesh conversion failed ({e}), falling back to point cloud")
            use_mesh = False
    
    if not use_mesh:
        # Fallback to point cloud
        if colors is not None:
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=colors,
                s=0.5,
                alpha=0.7,
            )
        elif labels is not None:
            # Use labels for coloring
            unique_labels = np.unique(labels)
            num_labels = len(unique_labels)
            # Use new matplotlib API
            try:
                colormap = plt.colormaps.get_cmap('tab20')
            except AttributeError:
                # Fallback for older matplotlib
                import matplotlib.cm as cm
                colormap = cm.get_cmap('tab20')
            
            # Resample colormap if needed
            if num_labels > 20:
                colormap = colormap.resampled(num_labels)
            
            label_to_color = {label: colormap(i / max(num_labels - 1, 1))[:3] for i, label in enumerate(unique_labels)}
            point_colors = np.array([label_to_color[label] for label in labels])
            
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=point_colors,
                s=0.5,
                alpha=0.7,
            )
        else:
            # No colors or labels, use default
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=0.5,
                alpha=0.7,
            )
    
    ax.set_title('Complete Assembled Spine', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    
    # Set equal aspect ratio
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add legend for vertebra types
    if labels is not None:
        unique_labels = np.unique(labels)
        try:
            colormap = plt.colormaps.get_cmap('tab20')
        except AttributeError:
            import matplotlib.cm as cm
            colormap = cm.get_cmap('tab20')
        
        if len(unique_labels) > 20:
            colormap = colormap.resampled(len(unique_labels))
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=colormap(i / max(len(unique_labels) - 1, 1))[:3], 
                      markersize=8, label=f'Type {int(label)}')
            for i, label in enumerate(unique_labels)
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    
    assembled_path = output_path.parent / f"{output_path.stem}_assembled.png"
    plt.savefig(assembled_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved assembled spine visualization: {assembled_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize PLY files with matplotlib')
    parser.add_argument('--ply_path', type=str, required=True,
                        help='Path to PLY file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for images (default: same as PLY file)')
    parser.add_argument('--show_individual', action='store_true',
                        help='Show individual vertebrae separately')
    parser.add_argument('--show_assembled', action='store_true', default=True,
                        help='Show complete assembled spine (default: True)')
    parser.add_argument('--use_mesh', action='store_true', default=True,
                        help='Convert point cloud to mesh for visualization (default: True)')
    
    args = parser.parse_args()
    
    ply_path = Path(args.ply_path)
    if not ply_path.exists():
        print(f"ERROR: PLY file not found: {ply_path}")
        return
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ply_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / ply_path.stem
    
    print(f"Loading PLY file: {ply_path}")
    points, colors, labels = load_ply_file(ply_path)
    
    if points is None:
        print("ERROR: Failed to load PLY file")
        return
    
    print(f"Loaded {len(points)} points")
    if labels is not None:
        unique_labels = np.unique(labels)
        print(f"Found {len(unique_labels)} vertebra types: {unique_labels}")
    
    # Load metadata if available
    metadata_path = ply_path.with_suffix('.json')
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Subject: {metadata.get('subject_id', 'unknown')}")
        print(f"Total vertebrae: {metadata.get('num_vertebrae', 'unknown')}")
    
    # Visualize
    try:
        if args.show_individual and labels is not None:
            print("\nCreating individual vertebrae visualization...")
            visualize_vertebrae_separate(points, labels, colors, output_path, use_mesh=args.use_mesh)
        else:
            print("\nSkipping individual vertebrae visualization (labels not available or --show_individual not set)")
    except Exception as e:
        print(f"ERROR in individual vertebrae visualization: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        if args.show_assembled:
            print("\nCreating assembled spine visualization...")
            visualize_assembled_spine(points, labels, colors, output_path, use_mesh=args.use_mesh)
        else:
            print("\nSkipping assembled spine visualization (--show_assembled not set)")
    except Exception as e:
        print(f"ERROR in assembled spine visualization: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ Visualization complete!")
    print(f"Output directory: {output_dir}")
    print("Generated files:")
    if args.show_individual and labels is not None:
        individual_file = output_dir / f"{output_path.stem}_individual_vertebrae.png"
        if individual_file.exists():
            print(f"  ✓ {individual_file}")
        else:
            print(f"  ✗ {individual_file} (not found)")
    if args.show_assembled:
        assembled_file = output_dir / f"{output_path.stem}_assembled.png"
        if assembled_file.exists():
            print(f"  ✓ {assembled_file}")
        else:
            print(f"  ✗ {assembled_file} (not found)")


if __name__ == '__main__':
    main()

