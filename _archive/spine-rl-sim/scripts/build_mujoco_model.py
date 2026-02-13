from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def _default_src_xml() -> Path:
    """
    Try to find a sensible default source MJCF path inside this repo.
    Falls back to a common output location used by our exporters.
    """
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "outputs" / "mujoco_exports" / "sub-verse563" / "sub-verse563_ALL_gt_ts.xml"
    return candidate


def _copy_meshes(src_xml: Path, out_mesh_dir: Path, mesh_files: list[str]) -> None:
    out_mesh_dir.mkdir(parents=True, exist_ok=True)
    for rel in mesh_files:
        src = src_xml.parent / rel
        dst = out_mesh_dir / Path(rel).name
        if not dst.exists():
            shutil.copy2(src, dst)


def _build_model(src_xml: Path, out_dir: Path) -> None:
    out_mesh_dir = out_dir / "meshes"
    out_xml = out_dir / "spine_model.xml"

    tree = ET.parse(src_xml)
    root = tree.getroot()

    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise RuntimeError("Invalid MJCF: missing asset or worldbody")

    # Collect mesh defs and geom defs from the original model.
    mesh_elems = asset.findall("mesh")
    if not mesh_elems:
        raise RuntimeError("No mesh assets found")

    mesh_files = [m.attrib["file"] for m in mesh_elems]
    _copy_meshes(src_xml=src_xml, out_mesh_dir=out_mesh_dir, mesh_files=mesh_files)

    spine_body = None
    for body in worldbody.findall("body"):
        if body.attrib.get("name") == "spine":
            spine_body = body
            break
    if spine_body is None:
        raise RuntimeError("Could not find spine body in MJCF")

    geom_elems = spine_body.findall("geom")
    if not geom_elems:
        raise RuntimeError("No spine geoms found")

    # Build new MJCF with per-vertebra bodies (freejoint each).
    new_root = ET.Element("mujoco", attrib={"model": "spine_rl"})
    ET.SubElement(new_root, "compiler", attrib={"angle": "degree", "coordinate": "local"})
    ET.SubElement(new_root, "option", attrib={"timestep": "0.002", "gravity": "0 0 -9.81"})

    new_asset = ET.SubElement(new_root, "asset")
    for m in mesh_elems:
        # Keep mesh name and scale; update path to local assets/meshes.
        attrib = dict(m.attrib)
        attrib["file"] = f"meshes/{Path(attrib['file']).name}"
        ET.SubElement(new_asset, "mesh", attrib=attrib)

    new_worldbody = ET.SubElement(new_root, "worldbody")
    ET.SubElement(new_worldbody, "light", attrib={"pos": "0 0 2", "dir": "0 0 -1"})

    for geom in geom_elems:
        mesh_name = geom.attrib.get("mesh")
        if not mesh_name:
            continue
        body_name = mesh_name.replace("mesh_", "")
        # Free joints must be on top-level bodies.
        body = ET.SubElement(new_worldbody, "body", attrib={"name": body_name, "pos": "0 0 0"})
        ET.SubElement(body, "freejoint", attrib={"name": f"{body_name}_free"})
        geom_attrib = dict(geom.attrib)
        geom_attrib["name"] = f"geom_{body_name}"
        ET.SubElement(body, "geom", attrib=geom_attrib)

    out_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(new_root)
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {out_xml}")
    print(f"Meshes in {out_mesh_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build a per-vertebra MuJoCo model for spine-rl-sim assets/")
    ap.add_argument(
        "--src_xml",
        type=str,
        default=str(_default_src_xml()),
        help="Source MJCF XML (from our exporters).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str((Path(__file__).resolve().parents[1] / "assets").resolve()),
        help="Output assets directory (writes spine_model.xml and meshes/).",
    )
    args = ap.parse_args()

    src_xml = Path(args.src_xml).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not src_xml.exists():
        raise SystemExit(f"Source XML not found: {src_xml}")
    if not (src_xml.parent / "meshes").exists():
        # We only require the referenced files to exist; many exporters place meshes/ next to XML.
        pass

    _build_model(src_xml=src_xml, out_dir=out_dir)
