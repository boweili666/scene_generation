# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import os
from pathlib import Path

from isaacsim import SimulationApp

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
        required=True,
        help="Folders to recursively scan (Omniverse or local paths)",
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

    asset_convert(args)

    kit.close()
