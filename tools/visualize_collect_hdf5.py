"""Stitch a scene_robot collect hdf5 into a single grid mp4 for quick review.

Layout: one row per demo, three columns (head / left_hand / right_hand).
Each frame is annotated with `demo i | step k/N | phase=p`. The output
file is `<input>.mp4` next to the hdf5 unless --out is given.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CAMERAS = ("head", "left_hand", "right_hand")


def _annotate(rgb: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    # black drop shadow + white text so it reads on any background
    for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        draw.text((10 + dx, 8 + dy), text, fill=(0, 0, 0), font=font)
    draw.text((10, 8), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def stitch(hdf5_path: Path, out_path: Path, fps: int) -> None:
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        demos = sorted(data.keys(), key=lambda k: int(k.split("_")[-1]))
        # Assume all demos share the same length; pad shorter ones by holding the last frame.
        lengths = {d: data[d]["obs"][CAMERAS[0]].shape[0] for d in demos}
        T = max(lengths.values())
        H, W = data[demos[0]]["obs"][CAMERAS[0]].shape[1:3]

        # Preload everything into RAM — for 4 x 28 x 3 x 480x640x3 uint8 this is ~140 MB.
        frames = {
            d: {c: data[d]["obs"][c][:] for c in CAMERAS} for d in demos
        }
        phases = {d: data[d]["obs"]["phase_id"][:] for d in demos}

    print(f"loaded {len(demos)} demos, T={T}, frame={H}x{W}")
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)

    for t in range(T):
        rows = []
        for d in demos:
            n = lengths[d]
            t_eff = min(t, n - 1)
            phase = int(phases[d][t_eff])
            tiles = []
            for cam in CAMERAS:
                tile = frames[d][cam][t_eff]
                label = f"{d} | {cam} | t={t_eff:02d}/{n - 1:02d} | phase={phase}"
                tiles.append(_annotate(tile, label))
            rows.append(np.concatenate(tiles, axis=1))
        grid = np.concatenate(rows, axis=0)
        writer.append_data(grid)

    writer.close()
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()
    out = args.out or args.hdf5.with_suffix(".mp4")
    stitch(args.hdf5, out, args.fps)


if __name__ == "__main__":
    main()
