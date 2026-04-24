"""Convert a scene_auto_grasp_collect HDF5 dataset into a LeRobotDataset.

Usage:
    python tools/convert_hdf5_to_lerobot.py \
        --hdf5 datasets/scene_auto_grasp_collect_agibot_bolt2.hdf5 \
        --repo-id local/agibot_bolt_grasp \
        --output-root datasets/lerobot/agibot_bolt_grasp \
        --task "pick up the bolt from the table"

The source HDF5 layout (written by scene_mouse_collect / scene_auto_grasp_collect):

    data/
        attrs["env_args"]  -> JSON with capture_hz, camera_aliases, ...
        demo_0/
            attrs["num_samples"], attrs["success"]
            actions                 (T, A) float32
            timestamps              (T,)   float32
            obs/joint_pos           (T, J) float32
            obs/joint_vel           (T, J) float32
            obs/ee_pos              (T, 3) float32
            obs/ee_quat             (T, 4) float32
            obs/target_ee_pos       (T, 3) float32
            obs/target_ee_quat      (T, 4) float32
            obs/gripper_joint_pos   (T, G) float32
            obs/root_pose           (T, 7) float32
            obs/active_arm_side     (T,)   int64
            obs/head                (T, H, W, 3) uint8
            obs/left_hand           (T, H, W, 3) uint8
            obs/right_hand          (T, H, W, 3) uint8
        demo_1/
            ...

Each source demo becomes one LeRobot episode. observation.state is
joint_pos (full articulation), action is the HDF5 actions array, and the
three RGB streams map to observation.images.{head,left_hand,right_hand}.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


# LeRobot is imported lazily so `--help` works even without the package.
# The module path moved between 0.2.x (lerobot.common.datasets.*) and 0.4.x
# (lerobot.datasets.*); try both so the script works on either release.
def _load_lerobot():
    last_exc: Exception | None = None
    for path in (
        "lerobot.datasets.lerobot_dataset",
        "lerobot.common.datasets.lerobot_dataset",
    ):
        try:
            import importlib

            module = importlib.import_module(path)
            return module.LeRobotDataset
        except ImportError as exc:
            last_exc = exc
    raise SystemExit(
        "lerobot is not installed (or LeRobotDataset module path is unknown). "
        f"Last import error: {last_exc}. Install with `pip install lerobot`."
    )


CAMERA_KEYS = ("head", "left_hand", "right_hand")


def _read_env_args(h5: h5py.File) -> dict:
    data_group = h5["data"]
    raw = data_group.attrs.get("env_args")
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _iter_demo_names(h5: h5py.File) -> Iterable[str]:
    data_group = h5["data"]
    # Sort demo_N by the integer suffix so episodes land in collection order.
    def _sort_key(name: str) -> int:
        if name.startswith("demo_"):
            try:
                return int(name.split("_", 1)[1])
            except ValueError:
                return 1 << 30
        return 1 << 30

    return sorted(data_group.keys(), key=_sort_key)


def _get(demo_group: h5py.Group, key: str) -> np.ndarray | None:
    if key not in demo_group:
        return None
    return np.asarray(demo_group[key])


def _validate_demo(demo_group: h5py.Group, demo_name: str) -> int | None:
    actions = _get(demo_group, "actions")
    if actions is None:
        print(f"[skip] {demo_name}: missing 'actions'")
        return None
    n_steps = actions.shape[0]
    if n_steps == 0:
        print(f"[skip] {demo_name}: empty episode")
        return None
    for cam in CAMERA_KEYS:
        if f"obs/{cam}" not in demo_group:
            print(f"[skip] {demo_name}: missing camera 'obs/{cam}'")
            return None
        if demo_group[f"obs/{cam}"].shape[0] != n_steps:
            print(
                f"[skip] {demo_name}: camera '{cam}' has "
                f"{demo_group[f'obs/{cam}'].shape[0]} frames but actions has {n_steps}"
            )
            return None
    if "obs/joint_pos" not in demo_group:
        print(f"[skip] {demo_name}: missing 'obs/joint_pos'")
        return None
    return int(n_steps)


def _build_features(
    *,
    state_dim: int,
    action_dim: int,
    image_shapes: dict[str, tuple[int, int, int]],
    use_videos: bool,
) -> dict:
    image_dtype = "video" if use_videos else "image"
    features: dict = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": None,
        },
    }
    for cam, shape in image_shapes.items():
        features[f"observation.images.{cam}"] = {
            "dtype": image_dtype,
            "shape": shape,  # (H, W, 3)
            "names": ["height", "width", "channel"],
        }
    return features


def _probe_shapes(h5: h5py.File) -> tuple[int, int, dict[str, tuple[int, int, int]]]:
    for demo_name in _iter_demo_names(h5):
        demo = h5[f"data/{demo_name}"]
        actions = _get(demo, "actions")
        joint_pos = _get(demo, "obs/joint_pos")
        if actions is None or joint_pos is None:
            continue
        image_shapes: dict[str, tuple[int, int, int]] = {}
        ok = True
        for cam in CAMERA_KEYS:
            ds = demo.get(f"obs/{cam}")
            if ds is None:
                ok = False
                break
            # ds shape is (T, H, W, 3)
            if ds.ndim != 4 or ds.shape[-1] != 3:
                ok = False
                break
            image_shapes[cam] = (int(ds.shape[1]), int(ds.shape[2]), 3)
        if not ok:
            continue
        return int(joint_pos.shape[-1]), int(actions.shape[-1]), image_shapes
    raise SystemExit("Could not find a usable demo group for shape probing.")


def convert(
    *,
    hdf5_path: Path,
    repo_id: str,
    output_root: Path,
    task: str,
    fps: float | None,
    include_failed: bool,
    max_episodes: int | None,
    use_videos: bool,
    overwrite: bool,
) -> None:
    LeRobotDataset = _load_lerobot()

    if not hdf5_path.exists():
        raise SystemExit(f"HDF5 not found: {hdf5_path}")

    if output_root.exists():
        if not overwrite:
            raise SystemExit(
                f"Output root {output_root} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as h5:
        env_args = _read_env_args(h5)
        if fps is None:
            capture_hz = env_args.get("capture_hz")
            if capture_hz is None:
                raise SystemExit(
                    "capture_hz not found in HDF5 env_args. Pass --fps explicitly."
                )
            fps = float(capture_hz)

        state_dim, action_dim, image_shapes = _probe_shapes(h5)
        print(
            f"[info] state_dim={state_dim} action_dim={action_dim} "
            f"cameras={ {k: v for k, v in image_shapes.items()} } fps={fps}"
        )

        features = _build_features(
            state_dim=state_dim,
            action_dim=action_dim,
            image_shapes=image_shapes,
            use_videos=use_videos,
        )

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=int(round(fps)),
            root=output_root,
            features=features,
            use_videos=use_videos,
            image_writer_processes=0,
            image_writer_threads=4 * len(image_shapes),
        )

        total_written = 0
        total_skipped_failed = 0
        total_skipped_bad = 0

        for demo_name in _iter_demo_names(h5):
            if max_episodes is not None and total_written >= max_episodes:
                break
            demo = h5[f"data/{demo_name}"]
            success_attr = demo.attrs.get("success")
            if success_attr is not None:
                success = bool(np.asarray(success_attr).item())
            else:
                success = True
            if not success and not include_failed:
                total_skipped_failed += 1
                continue

            n_steps = _validate_demo(demo, demo_name)
            if n_steps is None:
                total_skipped_bad += 1
                continue

            actions = np.asarray(demo["actions"], dtype=np.float32)
            joint_pos = np.asarray(demo["obs/joint_pos"], dtype=np.float32)
            cam_arrays = {
                cam: np.asarray(demo[f"obs/{cam}"]) for cam in image_shapes
            }
            for cam, arr in cam_arrays.items():
                if arr.dtype != np.uint8:
                    cam_arrays[cam] = arr.astype(np.uint8)

            for t in range(n_steps):
                frame: dict = {
                    "observation.state": joint_pos[t],
                    "action": actions[t],
                }
                for cam, arr in cam_arrays.items():
                    frame[f"observation.images.{cam}"] = arr[t]
                # LeRobot >=0.2 expects task per frame; older versions ignore it.
                frame["task"] = task
                try:
                    dataset.add_frame(frame)
                except TypeError:
                    # Older API without 'task' key support
                    frame.pop("task", None)
                    dataset.add_frame(frame)

            try:
                dataset.save_episode(task=task)
            except TypeError:
                dataset.save_episode()

            total_written += 1
            status = "success" if success else "failed"
            print(f"[write] {demo_name} -> episode {total_written - 1} ({status}, {n_steps} steps)")

        # Newer LeRobot requires consolidate; older versions no-op.
        if hasattr(dataset, "consolidate"):
            try:
                dataset.consolidate()
            except Exception as exc:
                print(f"[warn] consolidate() failed (ignored): {exc}")

    print(
        f"\n[done] wrote {total_written} episodes to {output_root}\n"
        f"       skipped failed: {total_skipped_failed}\n"
        f"       skipped invalid: {total_skipped_bad}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdf5", type=Path, required=True, help="Path to source HDF5 file")
    parser.add_argument("--repo-id", type=str, required=True, help="LeRobot dataset repo id (e.g. local/agibot_bolt_grasp)")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory the LeRobotDataset will be written to")
    parser.add_argument("--task", type=str, default="pick up the target object", help="Task description stored with every episode")
    parser.add_argument("--fps", type=float, default=None, help="Override fps. Defaults to env_args.capture_hz from the HDF5.")
    parser.add_argument("--include-failed", action="store_true", help="Also convert demos whose attrs['success'] is False")
    parser.add_argument("--max-episodes", type=int, default=None, help="Stop after writing N episodes (useful for dry runs)")
    parser.add_argument("--no-videos", action="store_true", help="Store images as PNGs instead of MP4 videos")
    parser.add_argument("--overwrite", action="store_true", help="Delete --output-root first if it exists")
    args = parser.parse_args(argv)

    convert(
        hdf5_path=args.hdf5,
        repo_id=args.repo_id,
        output_root=args.output_root,
        task=args.task,
        fps=args.fps,
        include_failed=args.include_failed,
        max_episodes=args.max_episodes,
        use_videos=not args.no_videos,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
