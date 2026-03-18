#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path
from typing import Iterable

import requests

from pipelines.real2sim.postprocess import postprocess_real2sim_outputs


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
    parser = argparse.ArgumentParser(
        description="Send one image + multiple masks to /predict_stream and save binary streamed outputs."
    )
    parser.add_argument(
        "--server",
        default="http://128.2.204.110:8000",
        help="Server base URL (default: http://128.2.204.110:8000)",
    )
    parser.add_argument(
        "--endpoint",
        default="/predict_stream",
        help="Predict endpoint path (default: /predict_stream)",
    )
    parser.add_argument("--image", type=Path, required=True, help="Path to image.png")
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
        default=None,
        help="Optional directory to auto-load *.png masks (excluding image.png)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scene-graph",
        type=Path,
        default=None,
        help="Optional scene graph JSON used for support-aware postprocessing.",
    )
    parser.add_argument("--texture-baking", type=parse_bool, default=True)
    parser.add_argument("--pose-optim-backend", default="gs", choices=["gs", "mesh"])
    parser.add_argument("--with-layout-postprocess", type=parse_bool, default=True)
    parser.add_argument("--gs-enable-manual-alignment", type=parse_bool, default=True)
    parser.add_argument("--gs-enable-shape-icp", type=parse_bool, default=True)
    parser.add_argument("--gs-enable-occlusion-check", type=parse_bool, default=False)
    parser.add_argument("--allow-mask-resize", type=parse_bool, default=False)
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
        default=Path("outputs/stream_results"),
        help="Output directory for streamed files.",
    )
    return parser


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
        ("pose_optim_backend", args.pose_optim_backend),
        ("with_layout_postprocess", "true" if args.with_layout_postprocess else "false"),
        ("gs_enable_manual_alignment", "true" if args.gs_enable_manual_alignment else "false"),
        ("gs_enable_shape_icp", "true" if args.gs_enable_shape_icp else "false"),
        ("gs_enable_occlusion_check", "true" if args.gs_enable_occlusion_check else "false"),
        ("allow_mask_resize", "true" if args.allow_mask_resize else "false"),
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
            run_once(args, mask_paths, output_dir, url)
            summary = postprocess_real2sim_outputs(output_dir, scene_graph_path=args.scene_graph)
            print(
                "[POST] scene postprocess:"
                f" objects={summary['objects']}"
                f" support_pairs={summary['support_pairs']}"
                f" forced_upright={summary['forced_upright']}"
                f" snapped_upright={summary['snapped_upright']}"
                f" penetration_adjustments={summary['penetration_adjustments']}"
            )
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
