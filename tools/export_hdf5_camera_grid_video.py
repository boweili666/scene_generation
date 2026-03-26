#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np


CAMERA_ORDER = ("head", "left_hand", "right_hand", "world")


def _infer_fps(timestamps: np.ndarray, default_fps: float = 10.0) -> float:
    if timestamps.ndim != 1 or timestamps.shape[0] < 2:
        return float(default_fps)
    dt = np.diff(timestamps.astype(np.float64))
    dt = dt[np.isfinite(dt) & (dt > 1.0e-6)]
    if dt.size == 0:
        return float(default_fps)
    return float(1.0 / dt.mean())


def _fit_frame(frame: np.ndarray, cell_width: int, cell_height: int, label: str) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    scale = min(cell_width / max(src_w, 1), cell_height / max(src_h, 1))
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
    offset_x = (cell_width - resized_w) // 2
    offset_y = (cell_height - resized_h) // 2
    canvas[offset_y:offset_y + resized_h, offset_x:offset_x + resized_w] = resized
    cv2.putText(
        canvas,
        label,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def export_grid_video(
    dataset_path: Path,
    output_path: Path,
    *,
    episode: str = "demo_0",
    grid_width: int = 1280,
    grid_height: int = 720,
    fps: float | None = None,
) -> Path:
    cell_width = grid_width // 2
    cell_height = grid_height // 2

    with h5py.File(dataset_path, "r") as handle:
        demo_group = handle["data"][episode]
        obs_group = demo_group["obs"]
        timestamps = demo_group["timestamps"][:] if "timestamps" in demo_group else np.empty((0,), dtype=np.float32)
        available = [name for name in CAMERA_ORDER if name in obs_group]
        if len(available) != 4:
            raise ValueError(f"Expected four cameras {CAMERA_ORDER}, got {available}")
        frame_count = min(int(obs_group[name].shape[0]) for name in available)
        out_fps = float(fps) if fps is not None else _infer_fps(timestamps)

        with imageio.get_writer(str(output_path), fps=out_fps) as writer:
            for idx in range(frame_count):
                cells = [
                    _fit_frame(np.asarray(obs_group[name][idx]), cell_width, cell_height, name)
                    for name in CAMERA_ORDER
                ]
                top = np.concatenate((cells[0], cells[1]), axis=1)
                bottom = np.concatenate((cells[2], cells[3]), axis=1)
                writer.append_data(np.concatenate((top, bottom), axis=0))

    print(f"[OK] grid: {output_path} ({frame_count} frames @ {out_fps:.2f} fps)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a 2x2 grid video from teleop HDF5 camera observations.")
    parser.add_argument("dataset", type=Path, help="Input HDF5 path.")
    parser.add_argument("--output", type=Path, default=None, help="Output mp4 path.")
    parser.add_argument("--episode", type=str, default="demo_0", help="Episode name under /data.")
    parser.add_argument("--fps", type=float, default=None, help="Override output fps.")
    parser.add_argument("--width", type=int, default=1280, help="Output grid width.")
    parser.add_argument("--height", type=int, default=720, help="Output grid height.")
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    output_path = (args.output or dataset_path.with_name(f"{dataset_path.stem}_{args.episode}_grid.mp4")).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_grid_video(
        dataset_path,
        output_path,
        episode=args.episode,
        grid_width=max(2, int(args.width)),
        grid_height=max(2, int(args.height)),
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
