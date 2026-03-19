# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Sequence

from isaacsim import SimulationApp
from usd_asset_utils import (
    column_transform_to_row_major,
    compute_asset_local_to_scene_matrix,
    is_identity_matrix,
)

# -----------------------------------------------------------------------------
# Async mesh → USD conversion
# -----------------------------------------------------------------------------
async def convert(in_file, out_file, load_materials=False):
    import omni.kit.asset_converter

    def progress_callback(progress, total_steps):
        pass

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    converter_context.ignore_materials = not load_materials
    converter_context.scale = 0.001
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(
        in_file, out_file, progress_callback, converter_context
    )

    while True:
        success = await task.wait_until_finished()
        if success:
            return True
        await asyncio.sleep(0.1)


def asset_convert_pairs(
    input_files: Sequence[str],
    output_files: Sequence[str],
    *,
    load_materials: bool = False,
) -> None:
    if len(input_files) != len(output_files):
        raise ValueError("input_files and output_files must have the same length")

    for in_file, out_file in zip(input_files, output_files):
        input_path = Path(in_file).expanduser().resolve()
        output_path = Path(out_file).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Converting: {input_path}")
        status = asyncio.get_event_loop().run_until_complete(
            convert(str(input_path), str(output_path), load_materials)
        )
        if not status:
            raise RuntimeError(f"Failed converting {input_path} -> {output_path}")
        print(f"[OK] {input_path} -> {output_path}")


def _as_float_matrix4(value) -> list[list[float]]:
    if isinstance(value, list) and len(value) == 4 and all(isinstance(row, list) and len(row) == 4 for row in value):
        return [[float(v) for v in row] for row in value]
    raise ValueError("Expected 4x4 matrix list")


def _column_transform_to_row_major(matrix) -> list[list[float]]:
    col = _as_float_matrix4(matrix)
    return [[float(col[c][r]) for c in range(4)] for r in range(4)]


def assemble_scene_from_manifest(manifest_path: str, scene_output: str) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom

    manifest_file = Path(manifest_path).expanduser().resolve()
    output_path = Path(scene_output).expanduser().resolve()
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    results_root = Path(payload["results_root"]).resolve()
    objects = payload.get("objects", {})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    for prim_path, entry in sorted(objects.items()):
        if not isinstance(entry, dict):
            continue
        usd_rel = entry.get("usd_path")
        usd_transform = entry.get("usd_transform")
        if not isinstance(usd_rel, str) or not usd_rel:
            continue
        if usd_transform is None:
            continue

        asset_path = (results_root / usd_rel).resolve()
        prim = stage.DefinePrim(str(prim_path), "Xform")
        asset_parent = prim
        asset_correction = compute_asset_local_to_scene_matrix(
            asset_path,
            target_up_axis="Z",
            target_meters_per_unit=1.0,
        )
        if not is_identity_matrix(asset_correction):
            asset_parent = stage.DefinePrim(f"{prim_path}/AssetRef", "Xform")
            asset_parent_xform = UsdGeom.Xformable(asset_parent)
            asset_parent_xform.ClearXformOpOrder()
            asset_parent_xform.AddTransformOp(opSuffix="assetNormalization").Set(
                Gf.Matrix4d(column_transform_to_row_major(asset_correction).tolist())
            )

        refs = asset_parent.GetReferences()
        if isinstance(entry.get("usd_prim_path"), str) and entry.get("usd_prim_path"):
            refs.AddReference(str(asset_path), Sdf.Path(entry["usd_prim_path"]))
        else:
            refs.AddReference(str(asset_path))

        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(Gf.Matrix4d(_column_transform_to_row_major(usd_transform)))

    stage.GetRootLayer().Save()
    print(f"[OK] assembled scene USD: {output_path}")


# -----------------------------------------------------------------------------
# Recursively list all files under a folder (Omniverse / local compatible)
# -----------------------------------------------------------------------------
def list_all_files_recursive(root_folder):
    all_files = []

    result, entries = omni.client.list(root_folder)
    if result != omni.client.Result.OK:
        print(f"[WARN] Cannot list folder: {root_folder}")
        return all_files

    for entry in entries:
        full_path = root_folder.rstrip("/") + "/" + entry.relative_path

        if entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            all_files.extend(list_all_files_recursive(full_path))
        else:
            all_files.append(full_path)

    return all_files


# -----------------------------------------------------------------------------
# Main conversion logic
# -----------------------------------------------------------------------------
def asset_convert(args):
    supported_file_formats = ["stl", "obj", "fbx", "glb"]

    scene_graph_ui_root = Path(__file__).resolve().parents[2]
    output_root = str(scene_graph_ui_root / "genmesh")
    omni.client.create_folder(output_root)

    for folder in args.folders:
        print(f"\n[INFO] Recursively scanning: {folder}")

        all_files = list_all_files_recursive(folder)
        print(f"[INFO] Found {len(all_files)} files")

        converted_count = 0

        for file_path in all_files:
            if converted_count >= args.max_models:
                print(f"[INFO] max models ({args.max_models}) reached, stopping.")
                break

            base_name = os.path.basename(file_path)
            name_no_ext, ext = os.path.splitext(base_name)
            fmt = ext.lower()[1:]

            if fmt not in supported_file_formats:
                continue

            output_usd_path = f"{output_root}/{name_no_ext}.usd"

            print(f"[INFO] Converting: {file_path}")
            status = asyncio.get_event_loop().run_until_complete(
                convert(file_path, output_usd_path, args.load_materials)
            )

            if not status:
                print(f"[ERROR] Failed: {file_path}")
            else:
                print(f"[OK] {file_path} → {output_usd_path}")
                converted_count += 1


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    kit = SimulationApp()

    import omni
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("omni.kit.asset_converter")

    parser = argparse.ArgumentParser("Recursively convert meshes to USD (flat output)")
    parser.add_argument(
        "--folders",
        type=str,
        nargs="+",
        default=None,
        help="Folders to recursively scan (Omniverse or local paths)",
    )
    parser.add_argument(
        "--input-files",
        type=str,
        nargs="+",
        default=None,
        help="Explicit input mesh files to convert.",
    )
    parser.add_argument(
        "--output-files",
        type=str,
        nargs="+",
        default=None,
        help="Explicit output USD files; must match --input-files length.",
    )
    parser.add_argument(
        "--assemble-scene-from-manifest",
        type=str,
        default=None,
        help="Assemble a scene USD from a Real2Sim manifest.",
    )
    parser.add_argument(
        "--scene-output",
        type=str,
        default=None,
        help="Output USD path used with --assemble-scene-from-manifest.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=10_000,
        help="Maximum number of meshes to convert",
    )
    parser.add_argument(
        "--load-materials",
        action="store_true",
        help="Load materials from source meshes",
    )

    args, _ = parser.parse_known_args()

    if args.input_files or args.output_files:
        if not args.input_files or not args.output_files:
            raise ValueError("Provide both --input-files and --output-files together")
        asset_convert_pairs(args.input_files, args.output_files, load_materials=args.load_materials)
    elif args.folders:
        asset_convert(args)
    elif not args.assemble_scene_from_manifest:
        raise ValueError("Provide --folders, explicit --input-files/--output-files, or --assemble-scene-from-manifest")

    if args.assemble_scene_from_manifest:
        if not args.scene_output:
            raise ValueError("Provide --scene-output together with --assemble-scene-from-manifest")
        assemble_scene_from_manifest(args.assemble_scene_from_manifest, args.scene_output)

    kit.close()
