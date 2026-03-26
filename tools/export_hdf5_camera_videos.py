#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np


def _infer_fps(timestamps: np.ndarray, default_fps: float) -> float:
    if timestamps.ndim != 1 or timestamps.shape[0] < 2:
        return float(default_fps)
    dt = np.diff(timestamps.astype(np.float64))
    dt = dt[np.isfinite(dt) & (dt > 1.0e-6)]
    if dt.size == 0:
        return float(default_fps)
    return float(1.0 / dt.mean())


def _camera_dataset_names(obs_group: h5py.Group) -> list[str]:
    names: list[str] = []
    for key, value in obs_group.items():
        if not isinstance(value, h5py.Dataset):
            continue
        if value.ndim != 4 or value.shape[-1] != 3:
            continue
        names.append(str(key))
    return sorted(names)


def export_dataset_videos(
    dataset_path: Path,
    output_dir: Path,
    *,
    episode: str | None = None,
    fps: float | None = None,
) -> list[Path]:
    written: list[Path] = []
    with h5py.File(dataset_path, "r") as handle:
        data_group = handle["data"]
        episode_names = [episode] if episode else sorted(data_group.keys())
        for episode_name in episode_names:
            demo_group = data_group[episode_name]
            obs_group = demo_group["obs"]
            timestamps = demo_group["timestamps"][:] if "timestamps" in demo_group else np.empty((0,), dtype=np.float32)
            camera_names = _camera_dataset_names(obs_group)
            if not camera_names:
                continue
            episode_fps = float(fps) if fps is not None else _infer_fps(timestamps, default_fps=10.0)
            for camera_name in camera_names:
                frames = obs_group[camera_name]
                out_path = output_dir / f"{episode_name}_{camera_name}.mp4"
                with imageio.get_writer(str(out_path), fps=episode_fps) as writer:
                    for frame in frames:
                        writer.append_data(frame)
                written.append(out_path)
                print(f"[OK] {camera_name}: {out_path} ({frames.shape[0]} frames @ {episode_fps:.2f} fps)")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RGB camera datasets from a teleop HDF5 file into mp4 videos.")
    parser.add_argument("dataset", type=Path, help="Path to the input HDF5 dataset.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to <dataset_stem>_videos.")
    parser.add_argument("--episode", type=str, default=None, help="Optional episode name under /data, e.g. demo_0.")
    parser.add_argument("--fps", type=float, default=None, help="Override output fps. Defaults to mean timestamp-derived fps.")
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    output_dir = (args.output_dir or dataset_path.with_name(f"{dataset_path.stem}_videos")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    written = export_dataset_videos(dataset_path, output_dir, episode=args.episode, fps=args.fps)
    print(f"[DONE] wrote {len(written)} video(s) to {output_dir}")


if __name__ == "__main__":
    main()
