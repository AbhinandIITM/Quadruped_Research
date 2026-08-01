r"""
Batch-convert COLLADA (.dae) meshes to .obj and/or .stl using trimesh.

trimesh uses pycollada under the hood, which correctly reads and applies
the COLLADA <up_axis> tag (Y_UP / Z_UP) during load, so the exported
mesh comes out in the right orientation without manual axis-flipping.

Install requirements:
    pip install trimesh pycollada lxml networkx

Usage:
    python convert_dae.py
    python convert_dae.py --input-dir .\unitree_go2\meshes --formats obj stl
"""

import os
import sys
import argparse
import trimesh


def convert_file(dae_path, out_dir, formats=("obj",)):
    name = os.path.splitext(os.path.basename(dae_path))[0]
    print(f"Loading {dae_path} ...")

    # force='mesh' merges all geometry in the scene into a single mesh,
    # which matches what a single URDF/MJCF <mesh> asset expects.
    mesh = trimesh.load(dae_path, force="mesh")

    if isinstance(mesh, trimesh.Scene):
        # Fallback: some DAE files contain multiple named parts.
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values()]
        )

    print(f"  -> {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    for fmt in formats:
        out_path = os.path.join(out_dir, f"{name}.{fmt}")
        mesh.export(out_path)
        print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert .dae meshes to .obj/.stl")
    parser.add_argument(
        "--input-dir",
        default=r".\unitree_go2\meshes",
        help="Folder containing the .dae files",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder to write converted meshes to (default: same as input-dir)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["obj"],
        choices=["obj", "stl"],
        help="One or more output formats (default: obj)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    os.makedirs(output_dir, exist_ok=True)

    dae_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".dae")]
    if not dae_files:
        print(f"No .dae files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(dae_files)} .dae files: {dae_files}\n")

    for fname in dae_files:
        dae_path = os.path.join(input_dir, fname)
        try:
            convert_file(dae_path, output_dir, formats=args.formats)
        except Exception as e:
            print(f"  FAILED on {fname}: {e}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()