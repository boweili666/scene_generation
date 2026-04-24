"""Build a 2x3 comparison video: training data (top) vs. closed-loop eval (bottom).

Top row    = training dataset cameras  head / left_hand / right_hand
Bottom row = closed-loop eval cameras  head / left_hand / right_hand

Each subframe is labelled in the top-left corner. The dataset and eval
episodes can be different lengths; the shorter one freezes on its last
frame so both rows stay in sync.

Run in the `lerobot` conda env (needs LeRobotDataset to index the
training dataset's per-episode frame range; torchcodec in env_isaaclab
is broken and cannot decode the chunked video files).

Usage:

    python tools/make_comparison_video.py \
        --dataset-root datasets/lerobot/agibot_bolt_grasp \
        --dataset-repo-id local/agibot_bolt_grasp \
        --eval-dir outputs/eval/diff_agibot_bolt_v1_20runs \
        --dataset-episode 0 \
        --eval-episode 0 \
        --output outputs/eval/diff_agibot_bolt_v1_20runs/comparison_ep00.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


CAMERA_NAMES = ("head", "left_hand", "right_hand")


# =============================================================================
# Frame sources
# =============================================================================
def _load_training_episode_frames(
    *,
    dataset_root: Path,
    repo_id: str,  # unused in the direct-decode path, kept for CLI compatibility
    episode_index: int,
) -> dict[str, np.ndarray]:
    # Bypass LeRobotDataset entirely: read the episode frame range out of the
    # meta/episodes parquet file with pyarrow, then decode the chunked MP4s
    # with imageio. This works even in env_isaaclab where torchcodec cannot
    # load its native libs, because imageio falls back to PyAV / ffmpeg-python
    # via its own ffmpeg backend (`imageio-ffmpeg`).
    del repo_id
    print(f"[merge] loading training dataset from {dataset_root} (direct parquet + imageio)")

    episode_lengths = _read_episode_lengths(dataset_root)
    if episode_index >= len(episode_lengths):
        raise SystemExit(
            f"[merge] episode_index {episode_index} out of range (dataset has {len(episode_lengths)} episodes)"
        )
    episode_from = int(sum(episode_lengths[:episode_index]))
    episode_to = int(episode_from + episode_lengths[episode_index])
    n_frames = episode_to - episode_from
    print(f"[merge] training episode {episode_index}: frames {episode_from}..{episode_to} ({n_frames} frames)")

    buffers: dict[str, np.ndarray] = {}
    for cam in CAMERA_NAMES:
        feature_key = f"observation.images.{cam}"
        video_paths = _find_chunk_videos(dataset_root, feature_key)
        if not video_paths:
            print(f"[merge] WARNING: no chunk videos found for {feature_key}")
            continue
        all_frames = _decode_concatenated_videos(video_paths)
        if all_frames is None or all_frames.shape[0] == 0:
            print(f"[merge] WARNING: chunk videos for {feature_key} decoded 0 frames")
            continue
        total = all_frames.shape[0]
        if episode_to > total:
            print(
                f"[merge] WARNING: {feature_key} decoded {total} frames but episode ends at {episode_to}; "
                f"clipping"
            )
        slice_end = min(episode_to, total)
        sliced = all_frames[episode_from:slice_end]
        if sliced.shape[0] == 0:
            print(f"[merge] WARNING: sliced 0 frames for {feature_key}")
            continue
        buffers[cam] = sliced
    return buffers


def _read_episode_lengths(dataset_root: Path) -> list[int]:
    # Read per-episode `length` from meta/episodes/**.parquet. Works with
    # lerobot 0.4.x directory layout. Falls back to meta/episodes.parquet
    # if the chunked layout is not used.
    candidates = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not candidates:
        single = dataset_root / "meta" / "episodes.parquet"
        if single.exists():
            candidates = [single]
    if not candidates:
        raise SystemExit(
            f"[merge] no episodes parquet found under {dataset_root / 'meta'}"
        )
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "[merge] pyarrow is required to read episodes parquet. "
            "Install with `pip install pyarrow`."
        ) from exc
    lengths: list[int] = []
    for path in candidates:
        table = pq.read_table(str(path))
        if "length" not in table.column_names:
            # Newer schema may store `num_frames` or `frame_count`.
            for alt in ("num_frames", "frame_count"):
                if alt in table.column_names:
                    lengths.extend(int(v) for v in table.column(alt).to_pylist())
                    break
            else:
                raise SystemExit(
                    f"[merge] episodes parquet {path} has no `length`/`num_frames`/`frame_count` column; "
                    f"columns: {table.column_names}"
                )
        else:
            lengths.extend(int(v) for v in table.column("length").to_pylist())
    return lengths


def _find_chunk_videos(dataset_root: Path, feature_key: str) -> list[Path]:
    base = dataset_root / "videos" / feature_key
    if not base.exists():
        return []
    return sorted(base.rglob("*.mp4"))


def _decode_concatenated_videos(paths: list[Path]) -> np.ndarray | None:
    import imageio.v3 as iio

    chunks: list[np.ndarray] = []
    for path in paths:
        try:
            frames = np.stack(list(iio.imiter(str(path))), axis=0)
        except Exception as exc:
            print(f"[merge] imageio failed to decode {path}: {exc}")
            continue
        if frames.shape[0] == 0:
            continue
        chunks.append(frames)
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def _load_eval_episode_frames(eval_dir: Path, episode_index: int) -> dict[str, np.ndarray]:
    import imageio.v3 as iio

    out: dict[str, np.ndarray] = {}
    for cam in CAMERA_NAMES:
        path = eval_dir / f"episode_{episode_index:02d}_{cam}.mp4"
        if not path.exists():
            # Fallback: some older runs may write PNG sequences.
            png_dir = eval_dir / f"episode_{episode_index:02d}" / cam
            if not png_dir.exists():
                print(f"[merge] WARNING: eval clip missing for {cam}: {path}")
                continue
            frames = []
            for png_path in sorted(png_dir.glob("frame_*.png")):
                frames.append(iio.imread(str(png_path)))
            if frames:
                out[cam] = np.stack(frames, axis=0)
                print(f"[merge] eval {cam}: loaded {len(frames)} PNG frames from {png_dir}")
            continue
        frames = np.stack(list(iio.imiter(str(path))), axis=0)
        out[cam] = frames
        print(f"[merge] eval {cam}: loaded {frames.shape[0]} frames from {path}")
    return out


# =============================================================================
# Grid composition
# =============================================================================
def _resize_rgb(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if frame.shape[0] == target_h and frame.shape[1] == target_w:
        return frame
    try:
        import cv2  # type: ignore

        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
    except ImportError:
        pass
    # Pure-numpy nearest-neighbour fallback (cheap, low quality but dependency-free).
    ys = (np.linspace(0, frame.shape[0] - 1, target_h)).astype(np.int32)
    xs = (np.linspace(0, frame.shape[1] - 1, target_w)).astype(np.int32)
    return frame[ys[:, None], xs[None, :]]


def _pad_to_length(frames: np.ndarray, length: int) -> np.ndarray:
    # Freeze the last frame to fill up to `length`. If we have no frames at
    # all, return a black placeholder of the right shape.
    if frames.shape[0] >= length:
        return frames[:length]
    if frames.shape[0] == 0:
        return frames
    pad_count = length - frames.shape[0]
    last_frame = frames[-1:]
    padding = np.repeat(last_frame, pad_count, axis=0)
    return np.concatenate([frames, padding], axis=0)


def _black_placeholder(length: int, h: int, w: int) -> np.ndarray:
    return np.zeros((length, h, w, 3), dtype=np.uint8)


def _annotate(frame: np.ndarray, text: str) -> np.ndarray:
    # Draw a filled background band behind the label so it's readable on
    # every scene colour. Uses cv2 if available, otherwise numpy draw-ops.
    try:
        import cv2  # type: ignore

        out = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        pad = 4
        box_x1, box_y1 = 4, 4
        box_x2 = box_x1 + text_w + 2 * pad
        box_y2 = box_y1 + text_h + 2 * pad + baseline
        cv2.rectangle(out, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), thickness=-1)
        cv2.putText(
            out,
            text,
            (box_x1 + pad, box_y1 + pad + text_h),
            font,
            scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )
        return out
    except ImportError:
        pass
    # Fallback: draw a solid band top-left but no text. Text without cv2 or
    # PIL is painful; better to at least mark the pane visually.
    out = frame.copy()
    out[2:18, 2:100] = 0
    return out


def _build_composite(
    train_frames: dict[str, np.ndarray],
    eval_frames: dict[str, np.ndarray],
    *,
    pane_h: int,
    pane_w: int,
) -> np.ndarray:
    # Work out the target length (max of any source clip).
    all_lengths: list[int] = []
    for d in (train_frames, eval_frames):
        for arr in d.values():
            if arr.shape[0] > 0:
                all_lengths.append(arr.shape[0])
    if not all_lengths:
        raise SystemExit("[merge] no frames found in either training or eval clips")
    target_len = max(all_lengths)
    print(f"[merge] target composite length: {target_len} frames")

    def _prep(source_dict: dict[str, np.ndarray], row_label: str) -> list[np.ndarray]:
        prepared: list[np.ndarray] = []
        for cam in CAMERA_NAMES:
            arr = source_dict.get(cam)
            if arr is None or arr.shape[0] == 0:
                arr = _black_placeholder(target_len, pane_h, pane_w)
            else:
                # Resize per-frame if needed, then pad along time.
                resized = np.stack(
                    [_resize_rgb(frm, pane_h, pane_w) for frm in arr], axis=0
                )
                arr = _pad_to_length(resized, target_len)
            # Annotate label once per frame, in-place over the pane.
            label = f"{row_label} / {cam}"
            annotated = np.stack([_annotate(f, label) for f in arr], axis=0)
            prepared.append(annotated)
        return prepared

    top_panes = _prep(train_frames, "TRAIN")
    bottom_panes = _prep(eval_frames, "EVAL")

    top_row = np.concatenate(top_panes, axis=2)  # (T, H, 3W, 3)
    bottom_row = np.concatenate(bottom_panes, axis=2)
    composite = np.concatenate([top_row, bottom_row], axis=1)  # (T, 2H, 3W, 3)
    return composite


# =============================================================================
# Entry point
# =============================================================================
def _parse_episode_list(spec: str, dataset_root: Path) -> list[int]:
    # Accept: "all", "N", "a,b,c", "a-b", or a mix like "0-5,10,15-19".
    spec = spec.strip().lower()
    if spec == "all":
        lengths = _read_episode_lengths(dataset_root)
        return list(range(len(lengths)))
    episodes: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            episodes.extend(range(start, end + 1))
        else:
            episodes.append(int(chunk))
    return episodes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", type=str, required=True)
    parser.add_argument(
        "--episodes",
        type=str,
        default="0",
        help="Which episodes to include. 'all', a comma list '0,1,2', or a range '0-19'. "
             "Each value is used as BOTH the training episode index and the eval episode index.",
    )
    parser.add_argument(
        "--dataset-episode",
        type=int,
        default=None,
        help="(Legacy) single dataset episode index. Ignored if --episodes is set.",
    )
    parser.add_argument(
        "--eval-episode",
        type=int,
        default=None,
        help="(Legacy) single eval episode index. Ignored if --episodes is set.",
    )
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--pane-height", type=int, default=240, help="Height (px) per sub-pane in the composite")
    parser.add_argument("--pane-width", type=int, default=320, help="Width (px) per sub-pane in the composite")
    parser.add_argument(
        "--gap-frames",
        type=int,
        default=3,
        help="Number of black frames to insert between consecutive episodes in the concatenated output.",
    )
    args = parser.parse_args(argv)

    # Resolve the episode list. Fallback to legacy single-episode args.
    if args.dataset_episode is not None and args.eval_episode is not None and args.episodes == "0":
        episode_pairs = [(int(args.dataset_episode), int(args.eval_episode))]
    else:
        raw_episodes = _parse_episode_list(args.episodes, args.dataset_root)
        episode_pairs = [(idx, idx) for idx in raw_episodes]
    if not episode_pairs:
        raise SystemExit("[merge] no episodes selected")

    pane_h = int(args.pane_height)
    pane_w = int(args.pane_width)
    gap_frames = max(0, int(args.gap_frames))
    composite_segments: list[np.ndarray] = []

    for pair_idx, (train_ep, eval_ep) in enumerate(episode_pairs):
        print(f"\n[merge] ===== pair {pair_idx + 1}/{len(episode_pairs)}: train={train_ep} eval={eval_ep} =====")
        try:
            train = _load_training_episode_frames(
                dataset_root=args.dataset_root,
                repo_id=args.dataset_repo_id,
                episode_index=train_ep,
            )
        except SystemExit as exc:
            print(f"[merge] skip train episode {train_ep}: {exc}")
            continue
        try:
            eval_ = _load_eval_episode_frames(
                eval_dir=args.eval_dir,
                episode_index=eval_ep,
            )
        except Exception as exc:
            print(f"[merge] skip eval episode {eval_ep}: {exc}")
            continue
        if not train and not eval_:
            print(f"[merge] no frames for pair ({train_ep}, {eval_ep}); skipping")
            continue
        segment = _build_composite(
            train,
            eval_,
            pane_h=pane_h,
            pane_w=pane_w,
        )
        # Burn the episode index into each frame of this segment (top-center)
        # so viewers can tell which pair is currently playing when everything
        # is concatenated into one MP4.
        segment = _stamp_episode_label(segment, f"ep {train_ep:02d}")
        composite_segments.append(segment)
        if gap_frames > 0 and pair_idx != len(episode_pairs) - 1:
            gap = np.zeros(
                (gap_frames, pane_h * 2, pane_w * 3, 3), dtype=np.uint8
            )
            composite_segments.append(gap)

    if not composite_segments:
        raise SystemExit("[merge] nothing to write: every pair was skipped")

    composite = np.concatenate(composite_segments, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio

    iio.imwrite(
        str(args.output),
        composite,
        fps=float(args.fps),
        codec="libx264",
        macro_block_size=None,
    )
    print(
        f"\n[merge] wrote {args.output} "
        f"({composite.shape[0]} frames, {composite.shape[2]}x{composite.shape[1]} px, "
        f"{len(episode_pairs)} episode pairs)"
    )
    return 0


def _stamp_episode_label(segment: np.ndarray, text: str) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError:
        return segment
    out = segment.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 6
    frame_w = segment.shape[2]
    center_x = (frame_w - text_w) // 2
    box_x1 = center_x - pad
    box_y1 = 4
    box_x2 = center_x + text_w + pad
    box_y2 = box_y1 + text_h + 2 * pad + baseline
    for i in range(out.shape[0]):
        frame = out[i]
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), thickness=-1)
        cv2.putText(
            frame,
            text,
            (center_x, box_y1 + pad + text_h),
            font,
            scale,
            (0, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )
    return out


if __name__ == "__main__":
    sys.exit(main())
