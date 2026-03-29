#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import time
from pathlib import Path
from typing import Iterable

import requests

if __package__:
    from .manifest import MANIFEST_FILENAME, build_real2sim_asset_manifest
    from .postprocess import postprocess_real2sim_outputs
else:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from pipelines.real2sim.manifest import MANIFEST_FILENAME, build_real2sim_asset_manifest
    from pipelines.real2sim.postprocess import postprocess_real2sim_outputs


DEFAULT_PREDICT_STREAM_SERVER = os.environ.get("PREDICT_STREAM_SERVER", "http://iclspiderman.ri.cmu.edu:8000")
DEFAULT_IMAGE_PATH = Path("runtime/real2sim/masks/image.png")
DEFAULT_MASK_DIR = Path("runtime/real2sim/masks")
DEFAULT_SCENE_GRAPH = Path("runtime/scene_graph/current_scene_graph.json")
DEFAULT_OUTPUT_DIR = Path("runtime/real2sim/scene_results")


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def collect_masks(mask_paths: Iterable[Path], mask_dir: Path | None) -> list[Path]:
    masks = [p for p in mask_paths if p is not None]
    if mask_dir is not None:
        if not mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
        for p in mask_dir.glob("*.png"):
            if p.name.lower() == "image.png":
                continue
            masks.append(p)

    unique = []
    seen = set()
    for p in masks:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(p)

    def sort_key(path: Path):
        stem = path.stem
        if stem.isdigit():
            return (0, int(stem), path.name)
        return (1, stem, path.name)

    unique.sort(key=sort_key)
    return unique


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Minimal client for /predict_stream that saves streamed scene outputs."
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_PREDICT_STREAM_SERVER,
        help=f"Server base URL (default: {DEFAULT_PREDICT_STREAM_SERVER})",
    )
    parser.add_argument(
        "--endpoint",
        default="/predict_stream",
        help="Predict endpoint path (default: /predict_stream)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help=f"Path to image.png (default: {DEFAULT_IMAGE_PATH})",
    )
    parser.add_argument(
        "--masks",
        type=Path,
        nargs="*",
        default=[],
        help="Mask paths (can pass multiple)",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=DEFAULT_MASK_DIR,
        help=f"Directory to auto-load *.png masks, excluding image.png (default: {DEFAULT_MASK_DIR})",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scene-graph",
        type=Path,
        default=DEFAULT_SCENE_GRAPH,
        help=(
            "Scene graph JSON used for support-aware postprocessing when the file exists "
            f"(default: {DEFAULT_SCENE_GRAPH})"
        ),
    )
    parser.add_argument("--texture-baking", type=parse_bool, default=True)
    parser.add_argument("--with-layout-postprocess", type=parse_bool, default=True)
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=20.0,
        help="HTTP connect timeout seconds (default: 20)",
    )
    parser.add_argument(
        "--read-timeout",
        type=float,
        default=1800.0,
        help="HTTP read timeout seconds (default: 1800)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry count on timeout/network error (default: 2)",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=3.0,
        help="Seconds to wait between retries (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for streamed files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--converter-python",
        default=os.environ.get("ISAAC_PYTHON", "python"),
        help="Python executable used to run mesh_to_usd_converter.py",
    )
    parser.add_argument(
        "--asset-converter-script",
        type=Path,
        default=project_root / "pipelines" / "isaac" / "mesh_to_usd_converter.py",
        help="Path to the Isaac mesh-to-USD converter script.",
    )
    return parser


def resolve_scene_graph_path(
    scene_graph_path: Path | None,
    *,
    default_scene_graph: Path = DEFAULT_SCENE_GRAPH,
) -> Path | None:
    if scene_graph_path is None:
        return None
    if scene_graph_path == default_scene_graph and not scene_graph_path.exists():
        return None
    return scene_graph_path


def read_exact(stream, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            raise EOFError(f"Unexpected EOF while reading {size} bytes")
        data += chunk
    return data


def iter_binary_frames(resp: requests.Response):
    raw = resp.raw
    while True:
        prefix = raw.read(4)
        if prefix == b"":
            return
        if len(prefix) != 4:
            raise RuntimeError("Invalid frame prefix length")
        header_len = struct.unpack(">I", prefix)[0]
        header_bytes = read_exact(raw, header_len)
        payload_len = struct.unpack(">Q", read_exact(raw, 8))[0]
        payload = read_exact(raw, payload_len) if payload_len > 0 else b""
        try:
            meta = json.loads(header_bytes.decode("utf-8"))
        except Exception as e:
            raise RuntimeError("Invalid frame header JSON") from e
        yield meta, payload


def clear_previous_outputs(output_dir: Path) -> None:
    objects_dir = output_dir / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)
    usd_objects_dir = output_dir / "usd_objects"

    for object_glb in objects_dir.glob("*.glb"):
        if object_glb.is_file():
            object_glb.unlink()

    if usd_objects_dir.exists() and usd_objects_dir.is_dir():
        for object_usd in usd_objects_dir.glob("*.usd"):
            if object_usd.is_file():
                object_usd.unlink()

    for artifact_name in (
        MANIFEST_FILENAME,
        "scene_merged.glb",
        "scene_merged_pre.glb",
        "scene_merged_post.glb",
        "scene_merged.usd",
        "scene_merged_post.usd",
        "poses.json",
        "poses_partial.json",
        "poses_pre.json",
        "poses_post.json",
    ):
        artifact_path = output_dir / artifact_name
        if artifact_path.exists() and artifact_path.is_file():
            artifact_path.unlink()


def collect_usd_conversion_pairs(output_dir: Path, *, include_scene_glb: bool = False) -> list[tuple[Path, Path]]:
    objects_dir = output_dir / "objects"
    usd_objects_dir = output_dir / "usd_objects"
    pairs: list[tuple[Path, Path]] = []

    if include_scene_glb:
        scene_post_glb = output_dir / "scene_merged_post.glb"
        if scene_post_glb.exists() and scene_post_glb.is_file():
            pairs.append((scene_post_glb, output_dir / "scene_merged_post.usd"))

    for object_glb in sorted(p for p in objects_dir.glob("*.glb") if p.is_file()):
        pairs.append((object_glb, usd_objects_dir / f"{object_glb.stem}.usd"))

    return pairs


def convert_outputs_to_usd(
    output_dir: Path,
    *,
    converter_python: str,
    asset_converter_script: Path,
    include_scene_glb: bool = False,
) -> None:
    conversion_pairs = collect_usd_conversion_pairs(output_dir, include_scene_glb=include_scene_glb)
    if not conversion_pairs:
        return

    converter_script = asset_converter_script.resolve()
    if not converter_script.exists():
        raise FileNotFoundError(f"Asset converter script not found: {converter_script}")

    cmd = [
        converter_python,
        "-u",
        str(converter_script),
        "--input-files",
        *[str(src.resolve()) for src, _ in conversion_pairs],
        "--output-files",
        *[str(dst.resolve()) for _, dst in conversion_pairs],
        "--load-materials",
    ]
    print(f"[POST] convert USD: {len(conversion_pairs)} file(s)")
    subprocess.run(cmd, check=True)


def assemble_scene_usd_from_manifest(
    manifest_path: Path,
    *,
    converter_python: str,
    asset_converter_script: Path,
    scene_output_path: Path,
) -> None:
    converter_script = asset_converter_script.resolve()
    if not converter_script.exists():
        raise FileNotFoundError(f"Asset converter script not found: {converter_script}")

    cmd = [
        converter_python,
        "-u",
        str(converter_script),
        "--assemble-scene-from-manifest",
        str(manifest_path.resolve()),
        "--scene-output",
        str(scene_output_path.resolve()),
    ]
    print(f"[POST] assemble scene USD: {scene_output_path}")
    subprocess.run(cmd, check=True)


def process_stream_response(resp: requests.Response, output_dir: Path) -> None:
    objects_dir = output_dir / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)

    object_count = 0
    got_scene = False
    got_poses = False
    partial_poses: dict[str, object] = {}

    for event, payload in iter_binary_frames(resp):
        event_type = event.get("type")

        if event_type == "object_glb":
            filename = event.get("glb_filename") or f"obj_{event.get('index', object_count):02d}.glb"
            out_path = objects_dir / filename
            out_path.write_bytes(payload)
            object_count += 1
            obj_name = event.get("obj_name")
            if isinstance(obj_name, str) and "pose" in event:
                partial_poses[obj_name] = event["pose"]
                (output_dir / "poses_partial.json").write_text(
                    json.dumps(partial_poses, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            print(f"[STREAM] object {object_count}: saved {out_path}")
            continue

        if event_type == "scene_glb":
            scene_filename = event.get("scene_glb_filename") or "scene_merged.glb"
            scene_path = output_dir / scene_filename
            scene_path.write_bytes(payload)
            got_scene = True
            print(f"[STREAM] scene: saved {scene_path}")
            continue

        if event_type == "poses_json":
            poses_filename = event.get("poses_filename") or "poses.json"
            poses_path = output_dir / poses_filename
            try:
                poses = json.loads(payload.decode("utf-8"))
            except Exception as e:
                raise RuntimeError("poses_json payload is not valid utf-8 json") from e
            poses_path.write_text(json.dumps(poses, indent=2, ensure_ascii=False), encoding="utf-8")
            got_poses = True
            print(f"[STREAM] poses: saved {poses_path}")
            continue

        if event_type == "done":
            print("[STREAM] done")
            continue

        if event_type == "error":
            detail = event.get("detail", "unknown error")
            raise RuntimeError(f"Server stream error: {detail}")

        print(f"[STREAM] unknown event type={event_type}: {event}")

    if not got_scene or not got_poses:
        raise RuntimeError("Stream ended without scene_glb/poses_json final frames")


def build_form_fields(args: argparse.Namespace) -> list[tuple[str, str]]:
    return [
        ("seed", str(args.seed)),
        ("texture_baking", "true" if args.texture_baking else "false"),
        ("with_layout_postprocess", "true" if args.with_layout_postprocess else "false"),
    ]


def open_files_for_request(
    image_path: Path, mask_paths: list[Path]
) -> tuple[list[tuple[str, tuple[str, object, str]]], list[object]]:
    files: list[tuple[str, tuple[str, object, str]]] = []
    opened: list[object] = []

    image_f = image_path.open("rb")
    opened.append(image_f)
    files.append(("image", (image_path.name, image_f, "image/png")))

    for mask in mask_paths:
        mf = mask.open("rb")
        opened.append(mf)
        files.append(("masks", (mask.name, mf, "image/png")))

    return files, opened


def run_once(args: argparse.Namespace, mask_paths: list[Path], output_dir: Path, url: str) -> None:
    form_fields = build_form_fields(args)
    files = None
    opened: list[object] = []

    try:
        files, opened = open_files_for_request(args.image, mask_paths)

        with requests.post(
            url,
            files=files,
            data=form_fields,
            timeout=(args.connect_timeout, args.read_timeout),
            stream=True,
        ) as resp:
            resp.raise_for_status()
            process_stream_response(resp, output_dir)
    finally:
        for f in opened:
            f.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.scene_graph = resolve_scene_graph_path(args.scene_graph)

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    mask_paths = collect_masks(args.masks, args.mask_dir)
    if not mask_paths:
        raise RuntimeError("No masks provided. Use --masks and/or --mask-dir.")

    for m in mask_paths:
        if not m.exists():
            raise FileNotFoundError(f"Mask not found: {m}")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    url = args.server.rstrip("/") + "/" + args.endpoint.lstrip("/")
    print(f"[INFO] POST {url}")
    print(f"[INFO] image={args.image}")
    print(f"[INFO] masks={len(mask_paths)}")
    print("[INFO] mode=binary-multipart + binary-stream")
    print(f"[INFO] output_dir={output_dir}")

    last_error: Exception | None = None
    for attempt in range(1, args.retries + 2):
        try:
            print(
                f"[INFO] attempt={attempt} timeout=(connect={args.connect_timeout}, read={args.read_timeout})"
            )
            clear_previous_outputs(output_dir)
            run_once(args, mask_paths, output_dir, url)
            summary = postprocess_real2sim_outputs(output_dir, scene_graph_path=args.scene_graph)
            print(
                "[POST] scene postprocess:"
                f" objects={summary['objects']}"
                f" support_pairs={summary['support_pairs']}"
                f" forced_upright={summary['forced_upright']}"
                f" snapped_upright={summary['snapped_upright']}"
                f" support_contact_adjustments={summary['support_contact_adjustments']}"
                f" penetration_adjustments={summary['penetration_adjustments']}"
                f" floating_adjustments={summary['floating_adjustments']}"
            )
            convert_outputs_to_usd(
                output_dir,
                converter_python=args.converter_python,
                asset_converter_script=args.asset_converter_script,
                include_scene_glb=args.scene_graph is None,
            )
            if args.scene_graph is not None:
                manifest_path, manifest = build_real2sim_asset_manifest(
                    output_dir,
                    scene_graph_path=args.scene_graph,
                )
                assemble_scene_usd_from_manifest(
                    manifest_path,
                    converter_python=args.converter_python,
                    asset_converter_script=args.asset_converter_script,
                    scene_output_path=output_dir / "scene_merged_post.usd",
                )
                manifest_path, manifest = build_real2sim_asset_manifest(
                    output_dir,
                    scene_graph_path=args.scene_graph,
                )
                print(
                    "[POST] asset manifest:"
                    f" matched={len(manifest.get('objects', {}))}"
                    f" unmatched_scene_paths={len(manifest.get('unmatched_scene_paths', []))}"
                    f" unmatched_outputs={len(manifest.get('unmatched_outputs', []))}"
                    f" scene_usd={manifest.get('scene_usd')}"
                )
                print(f"[POST] manifest saved: {manifest_path}")
            print(f"[DONE] Saved streamed outputs to: {output_dir}")
            return
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_error = e
            if attempt >= args.retries + 1:
                raise
            print(f"[WARN] request failed: {e}. Retrying in {args.retry_wait}s ...")
            time.sleep(args.retry_wait)
        except Exception:
            raise

    if last_error is not None:
        raise last_error


if __name__ == "__main__":
    main()
