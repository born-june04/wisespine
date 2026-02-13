#!/usr/bin/env python3
"""
Generate a MuJoCo MJCF (XML) that references meshes exported as OBJ.

Why we split OBJ by default:
  - MuJoCo's behavior with multi-object OBJs can vary by version/config.
  - Splitting into per-object OBJ files is the most reliable way to ensure
    each vertebra mesh is addressable as a separate <mesh>.

Typical usage (for the combined ALL export):
  python scripts/generate_mujoco_xml_from_obj.py \
    --obj outputs/unity_exports/sub-verse563_ALL_gt_ts.obj \
    --out_dir outputs/mujoco_exports \
    --scale 0.001

Notes:
  - Units: NIfTI world coords are typically in mm. MuJoCo expects meters.
    Use --scale 0.001 (default) to convert mm->m.
  - Colors: we assign per-vertebra colors using the same scheme as the Unity exporter
    (GT brighter, TS darker).
"""

from __future__ import annotations

import argparse
import colorsys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ObjObject:
    name: str
    # faces as list of triples of global vertex indices (1-based, OBJ convention)
    faces: List[Tuple[int, int, int]]
    material: Optional[str] = None


def _parse_face_vertex(token: str) -> int:
    """
    Parse an OBJ face vertex token: 'v', 'v/vt', 'v//vn', or 'v/vt/vn'
    Returns the vertex index as int (OBJ is 1-based).
    """
    if "/" in token:
        token = token.split("/", 1)[0]
    return int(token)


def parse_obj_objects(obj_path: Path) -> Tuple[List[Tuple[float, float, float]], List[ObjObject]]:
    """
    Parse only what we need from an OBJ:
      - global vertex list ('v')
      - objects ('o') and their faces ('f')
      - optional material assignment ('usemtl')
    """
    vertices: List[Tuple[float, float, float]] = []
    objects: List[ObjObject] = []
    current: Optional[ObjObject] = None

    with open(obj_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                _, xs, ys, zs = line.split()[:4]
                vertices.append((float(xs), float(ys), float(zs)))
            elif line.startswith("o "):
                name = line[2:].strip()
                current = ObjObject(name=name, faces=[], material=None)
                objects.append(current)
            elif line.startswith("usemtl "):
                if current is not None:
                    current.material = line[len("usemtl ") :].strip()
            elif line.startswith("f "):
                if current is None:
                    # If OBJ has faces before 'o', create a default object
                    current = ObjObject(name="object_0", faces=[], material=None)
                    objects.append(current)
                parts = line.split()[1:]
                # Triangulate if needed (fan)
                idx = [_parse_face_vertex(p) for p in parts]
                if len(idx) < 3:
                    continue
                v0 = idx[0]
                for i in range(1, len(idx) - 1):
                    current.faces.append((v0, idx[i], idx[i + 1]))

    return vertices, objects


def write_obj_split(
    out_path: Path,
    vertices_global: List[Tuple[float, float, float]],
    obj: ObjObject,
) -> None:
    """
    Write a single-object OBJ with compacted vertices and reindexed faces.
    """
    # Collect used vertices
    used = sorted({i for tri in obj.faces for i in tri})
    # Map old (1-based) -> new (1-based)
    remap: Dict[int, int] = {old: new for new, old in enumerate(used, start=1)}

    lines: List[str] = [
        "# Split from combined OBJ for MuJoCo",
        f"o {obj.name}",
    ]
    for old_idx in used:
        x, y, z = vertices_global[old_idx - 1]
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    for a, b, c in obj.faces:
        lines.append(f"f {remap[a]} {remap[b]} {remap[c]}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def _vertebra_name_to_label(vname: str) -> Optional[int]:
    vname = vname.strip().upper()
    try:
        if vname.startswith("C"):
            i = int(vname[1:])
            return i if 1 <= i <= 7 else None
        if vname.startswith("T"):
            i = int(vname[1:])
            return 7 + i if 1 <= i <= 12 else None
        if vname.startswith("L"):
            i = int(vname[1:])
            return 19 + i if 1 <= i <= 5 else None
        if vname == "S1":
            return 25
    except Exception:
        return None
    return None


def rgba_for_object_name(obj_name: str) -> Tuple[float, float, float, float]:
    """
    Match exporter naming, e.g.:
      sub-verse563_L1_GT
      sub-verse563_T12_TS
    """
    parts = obj_name.split("_")
    kind = None
    vertebra = None
    if parts:
        tail = parts[-1].upper()
        if tail in ("GT", "TS"):
            kind = tail
            if len(parts) >= 2:
                vertebra = parts[-2].upper()

    label = _vertebra_name_to_label(vertebra) if vertebra else None
    if label is None:
        # fallback deterministic hue
        h = (abs(hash(obj_name)) % 360) / 360.0
    else:
        h = (label - 1) / 25.0

    s = 0.75
    v = 0.95 if kind == "GT" else 0.75 if kind == "TS" else 0.85
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return float(r), float(g), float(b), 1.0


def safe_xml_name(name: str) -> str:
    # MuJoCo names are fairly permissive, but keep it simple:
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate MuJoCo MJCF (XML) from OBJ meshes")
    ap.add_argument("--obj", required=True, help="Path to combined OBJ (with multiple 'o' objects)")
    ap.add_argument("--out_dir", required=True, help="Output directory for split OBJs and MJCF XML")
    ap.add_argument("--xml_name", default=None, help="Output XML filename (default: <obj_stem>.xml)")
    ap.add_argument("--scale", type=float, default=0.001, help="Mesh scale (default: 0.001 for mm->m)")
    ap.add_argument(
        "--split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Split combined OBJ into per-object OBJ files (recommended).",
    )
    ap.add_argument(
        "--visual_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, set contype/conaffinity=0 (no collision).",
    )
    args = ap.parse_args()

    obj_path = Path(args.obj).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    verts, objects = parse_obj_objects(obj_path)
    if not objects:
        raise RuntimeError(f"No objects found in OBJ (no 'o' sections?): {obj_path}")

    mesh_entries = []
    geom_entries = []

    meshes_dir = out_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    for obj in objects:
        oname = safe_xml_name(obj.name)
        rgba = rgba_for_object_name(obj.name)
        mesh_name = f"mesh_{oname}"

        if args.split:
            mesh_file = meshes_dir / f"{oname}.obj"
            write_obj_split(mesh_file, verts, obj)
            file_attr = str(mesh_file.relative_to(out_dir))
        else:
            # Reference the combined file; may or may not be split by MuJoCo internally.
            file_attr = str(obj_path.relative_to(out_dir)) if obj_path.is_relative_to(out_dir) else str(obj_path)

        mesh_entries.append(
            f'    <mesh name="{mesh_name}" file="{file_attr}" scale="{args.scale} {args.scale} {args.scale}"/>'
        )
        cont = ' contype="0" conaffinity="0"' if args.visual_only else ""
        geom_entries.append(
            f'      <geom type="mesh" mesh="{mesh_name}" rgba="{rgba[0]:.4f} {rgba[1]:.4f} {rgba[2]:.4f} {rgba[3]:.4f}"{cont}/>'
        )

    xml_name = args.xml_name or f"{obj_path.stem}.xml"
    xml_path = out_dir / xml_name

    xml_lines = [
        "<mujoco model=\"spine_meshes\">",
        "  <compiler angle=\"degree\" coordinate=\"local\"/>",
        "  <option timestep=\"0.002\" gravity=\"0 0 -9.81\"/>",
        "  <asset>",
        *mesh_entries,
        "  </asset>",
        "  <worldbody>",
        "    <light pos=\"0 0 2\" dir=\"0 0 -1\"/>",
        "    <body name=\"spine\" pos=\"0 0 0\">",
        *geom_entries,
        "    </body>",
        "  </worldbody>",
        "</mujoco>",
        "",
    ]
    xml_path.write_text("\n".join(xml_lines))

    print("Wrote MJCF:")
    print(" ", xml_path)
    if args.split:
        print("Wrote split meshes under:")
        print(" ", meshes_dir)


if __name__ == "__main__":
    main()


