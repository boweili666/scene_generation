"""Closed-loop evaluation of a LeRobot policy in Isaac Sim.

This reuses the scene builder from `scene_mouse_collect` (same as
`scene_auto_grasp_collect`) to spawn the robot + scene, then swaps the
auto-grasp phase runner for a policy-driven stepping loop:

    obs -> policy.select_action() -> delta EE action -> absolute target
        -> controller.step_pose_target() -> sim.step() -> check lift

The 7D action layout matches what `_build_action_tensor` writes in
`scene_auto_grasp_collect.py`:

    action[0:3] = ee_pos target delta (base frame, m)
    action[3:6] = ee rotation delta as a rotation vector (rad)
    action[6]   = gripper command (0 = open, 1 = closed)

So the eval loop reconstructs `(target_pos_b, target_quat_b)` by adding
the delta to the controller's current EE pose in the base frame, then
feeds it to `controller.step_pose_target`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from .scene_auto_grasp_collect import (
    _build_scene_mouse_collect_args as _reuse_build_args_hidden,
    _read_rigid_body_world_position_z,
    _restore_target_rigid_body_state,
    _robot_forward_xy_world,
    _shifted_target_snapshot,
    _snapshot_target_rigid_body_state,
)
from .scene_mouse_collect import _build_scene_mouse_collect, _reset_scene_to_plan


@dataclass
class SceneEvalArgs:
    device: str
    num_envs: int
    scene_usd_path: str
    scene_graph_path: str
    placements_path: str
    target: str | None
    support: str | None
    plan_output_dir: str
    base_z_bias: float
    arm_side_preference: str
    manifest_path: str | None
    checkpoint: str
    num_episodes: int
    max_steps_per_episode: int
    target_forward_randomization: float
    success_lift_delta: float
    fingertip_distance: float
    agibot_ee_frame_remap: str
    gripper_threshold: float
    random_seed: int | None
    # Path to the LeRobotDataset root (e.g. datasets/lerobot/agibot_bolt_grasp).
    # We read `meta/stats.json` from here at load time to build manual
    # normalize/denormalize helpers around `policy.select_action`. LeRobot
    # 0.4.x stores its normalizer stats OUTSIDE the nn.Module, so
    # `from_pretrained` alone leaves the policy outputting values in
    # normalized space (≈[-1, 1]) instead of real-data space, and closed
    # loop integration would explode without this fix.
    dataset_root: str
    # Override DiffusionPolicy's denoising step count for inference. Default
    # training uses `num_train_timesteps=100`, but a DDIM-style reduced
    # schedule with 10-20 steps gives near-identical actions and makes the
    # closed-loop eval run smoothly (otherwise U-Net denoising every ~10 env
    # steps causes long visible pauses). 0 = keep the model's default.
    num_inference_steps: int
    # Number of physics sub-steps to run between policy.select_action calls.
    # The training data was recorded at `capture_hz=10` with physics at
    # ~60 Hz, i.e. 6 physics steps per dataset frame. The policy's delta
    # action therefore represents 0.1s of motion, not 1/60s. Running the
    # policy once per physics step makes the robot move 6x slower than
    # training intended, so we amortize the command across 6 sim steps by
    # holding the same commanded EE target across a batch of physics
    # updates.
    sim_steps_per_policy_call: int
    # When true, wrap the policy's select_action with an async prefetch
    # runner: a background thread calls `predict_action_chunk` before the
    # internal action queue empties so the main loop never hits a
    # synchronous denoising pause. Eliminates the visible stutter.
    async_inference: bool
    # If non-empty, write per-episode camera videos (or PNG sequences) here.
    # Episode N gets files named episode_NN_<camera>.mp4 (or episode_NN/<camera>/frame_XXXX.png).
    record_dir: str
    record_fps: float


# =============================================================================
# Policy loading (LeRobot Diffusion Policy)
# =============================================================================
def _install_lerobot_import_shims() -> None:
    # env_isaaclab has a broken transformers / huggingface_hub version pair
    # (see `_patch_huggingface_hub_is_offline_mode` below). In addition,
    # `lerobot.policies.__init__` eagerly imports
    # `from .groot.configuration_groot import GrootConfig`, which drags the
    # whole transformers dependency chain in and explodes. We don't need
    # groot for diffusion-policy evaluation, so we pre-populate fake modules
    # for every `lerobot.policies.groot.*` path that `__init__.py` references.
    # Python's import machinery checks `sys.modules` before touching disk,
    # so these shims silently satisfy the offending `from ...` statements.
    import sys
    import types

    def _ensure(name: str) -> types.ModuleType:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        return sys.modules[name]

    groot = _ensure("lerobot.policies.groot")
    groot_cfg = _ensure("lerobot.policies.groot.configuration_groot")
    groot_model = _ensure("lerobot.policies.groot.modeling_groot")
    _ensure("lerobot.policies.groot.groot_n1")

    class _GrootConfigStub:  # pragma: no cover - placeholder only
        pass

    class _GrootPolicyStub:  # pragma: no cover - placeholder only
        pass

    groot_cfg.GrootConfig = _GrootConfigStub
    groot.GrootConfig = _GrootConfigStub
    groot_model.GrootPolicy = _GrootPolicyStub
    groot.GrootPolicy = _GrootPolicyStub


def _patch_huggingface_hub_is_offline_mode() -> None:
    # env_isaaclab ships a newer huggingface_hub where `is_offline_mode` was
    # removed, but the older transformers bundled with isaaclab still does
    # `from huggingface_hub import is_offline_mode`. Since `lerobot.policies`'
    # __init__ eagerly imports `groot` -> `transformers`, touching
    # `lerobot.policies.diffusion.modeling_diffusion` triggers the whole
    # chain. We inject a compatibility shim before importing lerobot so the
    # transformers import goes through. The shim mirrors the behaviour of
    # the old `is_offline_mode` (env flags + HF_HUB_OFFLINE constant).
    import os

    import huggingface_hub

    if hasattr(huggingface_hub, "is_offline_mode"):
        return

    def _is_offline_mode() -> bool:
        if os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1":
            return True
        if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
            return True
        try:
            from huggingface_hub import constants

            return bool(getattr(constants, "HF_HUB_OFFLINE", False))
        except Exception:
            return False

    huggingface_hub.is_offline_mode = _is_offline_mode  # type: ignore[attr-defined]


def _load_policy(checkpoint: str, device: str):
    # Install both shims before any transformers/lerobot import touches
    # this interpreter. Order matters: hf_hub patch first, groot stubs
    # second, then the actual lerobot import.
    _patch_huggingface_hub_is_offline_mode()
    _install_lerobot_import_shims()

    # We only import the diffusion policy module (not lerobot.scripts /
    # lerobot.envs / lerobot.robots) so that this script can run inside
    # env_isaaclab even when those sibling packages have transformers /
    # huggingface_hub version conflicts with isaaclab's pinned deps.
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    ckpt_path = Path(checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    # A LeRobot checkpoint saved via `policy.save_pretrained(...)` lives in
    # `outputs/train/<job>/checkpoints/<step>/pretrained_model`. Accept either
    # that exact dir or a parent that contains it, so users can point at
    # either form on the CLI.
    candidate_dirs = [ckpt_path]
    for sub in ("pretrained_model", "last/pretrained_model", "checkpoints/last/pretrained_model"):
        candidate_dirs.append(ckpt_path / sub)
    resolved = None
    for d in candidate_dirs:
        if (d / "config.json").exists() or (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists():
            resolved = d
            break
    if resolved is None:
        raise SystemExit(
            f"Could not find a loadable pretrained_model folder under {ckpt_path}. "
            f"Tried: {[str(d) for d in candidate_dirs]}"
        )
    print(f"[eval] loading policy from {resolved}")
    policy = DiffusionPolicy.from_pretrained(str(resolved))
    policy.eval()
    policy.to(device)
    return policy


class AsyncPolicyRunner:
    """Async wrapper around DiffusionPolicy.select_action.

    DiffusionPolicy exposes `predict_action_chunk(batch)` which runs a full
    denoising loop and returns `horizon` future actions. `select_action`
    lazily fills its internal `_queues["action"]` deque by calling that
    method whenever the deque empties, so every ~`n_action_steps` env steps
    you see a synchronous pause while U-Net denoises.

    This runner keeps the queue topped up in a background thread:
      - After every `select_action` call, if the deque is at or below
        `prefetch_when_remaining`, kick off a background future that calls
        `predict_action_chunk`.
      - Before the next `select_action` call, if the deque is about to
        empty AND a future is ready, inject its chunk into the deque so
        the main thread never triggers a synchronous denoise.

    Because `predict_action_chunk` reads from `policy._queues` (which the
    main thread keeps populating with fresh observations), the background
    chunk is computed on the LATEST obs, not stale ones.

    Only safe for single-threaded callers against a single policy instance.
    """

    def __init__(self, policy, prefetch_when_remaining: int = 2):
        import concurrent.futures

        self._policy = policy
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._pending_future = None
        self._prefetch_when = int(prefetch_when_remaining)
        self._stats_sync_calls = 0
        self._stats_async_hits = 0

    def select_action(self, batch):
        import concurrent.futures

        action_queue = self._get_action_queue()

        # If a background chunk is ready and the queue is nearly empty,
        # inject it BEFORE letting the policy run (so select_action's own
        # queue-empty check finds pre-filled actions and skips denoising).
        if self._pending_future is not None and action_queue is not None and len(action_queue) <= 1:
            try:
                chunk = self._pending_future.result(timeout=5.0)
                # chunk shape: (B, horizon, action_dim). The policy itself
                # does `self._queues[ACTION].extend(actions.transpose(0, 1))`,
                # i.e. iterate over the horizon axis. Mirror that here.
                action_queue.extend(chunk.transpose(0, 1))
                self._pending_future = None
                self._stats_async_hits += 1
            except concurrent.futures.TimeoutError:
                print("[async] prefetch future timed out; falling back to sync")
                self._pending_future = None

        # Normal policy call. If we just injected a chunk, the action queue
        # is non-empty and this is cheap.
        action = self._policy.select_action(batch)
        if action_queue is not None and len(action_queue) == 0:
            # Queue drained this call (shouldn't happen if prefetch caught it,
            # but count it so we know).
            self._stats_sync_calls += 1

        # Refreshed queue after the pop.
        action_queue = self._get_action_queue()

        # Kick off the next prefetch if the queue is getting low.
        if (
            action_queue is not None
            and len(action_queue) <= self._prefetch_when
            and self._pending_future is None
        ):
            # `predict_action_chunk` only uses the KEYS of the batch arg to
            # decide which observation queues to read; the actual data comes
            # from `self._policy._queues`. IMPORTANT: the policy's
            # select_action stacks the individual per-camera keys (e.g.
            # `observation.images.head`) into a single `observation.images`
            # key BEFORE populating its obs queue, so the queue only has the
            # stacked key. We therefore build the background batch from the
            # actual queue keys (minus the action queue), not from the raw
            # main-loop batch keys — otherwise predict_action_chunk's
            # comprehension `{k: ... for k in batch if k in self._queues}`
            # would drop every image feature and generate_actions would
            # KeyError on `observation.images`.
            queues = getattr(self._policy, "_queues", {}) or {}
            batch_keys = {k: None for k in queues.keys() if k != "action"}
            self._pending_future = self._executor.submit(
                self._policy.predict_action_chunk, batch_keys
            )
        return action

    def _get_action_queue(self):
        queues = getattr(self._policy, "_queues", None)
        if queues is None:
            return None
        return queues.get("action")

    def close(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
        print(
            f"[async] prefetch stats: async_hits={self._stats_async_hits} "
            f"sync_fallbacks={self._stats_sync_calls}"
        )


def _maybe_reduce_inference_steps(policy, num_inference_steps: int) -> None:
    # DiffusionPolicy's denoising loop length is read from `config.num_inference_steps`
    # at select_action time (or `num_train_timesteps` if the former is unset).
    # Setting both to a smaller value (e.g. 10-20) switches to a DDIM-style
    # reduced schedule and cuts the visible stutter between env steps.
    if num_inference_steps <= 0:
        return
    cfg = getattr(policy, "config", None)
    if cfg is None:
        print("[eval] WARNING: policy has no .config; cannot set num_inference_steps")
        return
    # LeRobot 0.4.x DiffusionConfig has `num_inference_steps`; older versions
    # may expose `num_sample_steps` or similar. Try the known names.
    updated = False
    for attr in ("num_inference_steps", "num_sampling_steps", "num_sample_steps"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, int(num_inference_steps))
            updated = True
            print(f"[eval] set policy.config.{attr} = {num_inference_steps}")
    # Some implementations also cache a scheduler with its own num_inference_steps.
    scheduler = getattr(policy, "noise_scheduler", None) or getattr(
        getattr(policy, "diffusion", None), "noise_scheduler", None
    )
    if scheduler is not None and hasattr(scheduler, "set_timesteps"):
        try:
            scheduler.set_timesteps(int(num_inference_steps))
            print(f"[eval] scheduler.set_timesteps({num_inference_steps}) called")
            updated = True
        except Exception as exc:
            print(f"[eval] scheduler.set_timesteps failed: {exc}")
    if not updated:
        print("[eval] WARNING: could not find a num_inference_steps attribute to set")


# =============================================================================
# Observation + action plumbing
# =============================================================================
_CAMERA_NAMES = ("head", "left_hand", "right_hand")


# =============================================================================
# Per-episode camera recording (MP4 via imageio, PNG fallback)
# =============================================================================
class EpisodeRecorder:
    """Buffers camera RGB frames for one episode and flushes to disk.

    Tries to write one MP4 per camera via `imageio.v3` (ffmpeg backend). If
    imageio isn't available or writing fails, falls back to a directory of
    per-frame PNGs. Frames are pulled from the live Isaac Lab camera sensor
    (`cam.data.output["rgb"]`) as uint8 `(H, W, 3)` and stacked in memory —
    for an 80-frame episode at 640x480 that's ~30MB per camera, safe.
    """

    def __init__(self, record_dir: Path | None, episode_idx: int, fps: float) -> None:
        self.record_dir = record_dir
        self.episode_idx = int(episode_idx)
        self.fps = float(fps) if fps > 0 else 10.0
        self._buffers: dict[str, list] = {}
        self._enabled = record_dir is not None

    def capture(self, cameras: dict) -> None:
        if not self._enabled:
            return
        for name in _CAMERA_NAMES:
            cam = cameras.get(name)
            if cam is None:
                continue
            raw = cam.data.output.get("rgb")
            if raw is None:
                continue
            frame = raw if not torch.is_tensor(raw) else raw.detach()
            if frame.ndim == 4:
                frame = frame[0]
            arr = frame.to(dtype=torch.uint8).cpu().numpy()
            self._buffers.setdefault(name, []).append(arr)

    def flush(self) -> None:
        if not self._enabled or self.record_dir is None or not self._buffers:
            return
        import numpy as _np

        self.record_dir.mkdir(parents=True, exist_ok=True)
        for name, frames in self._buffers.items():
            if not frames:
                continue
            stacked = _np.stack(frames, axis=0)  # (T, H, W, 3) uint8
            base = f"episode_{self.episode_idx:02d}_{name}"
            mp4_path = self.record_dir / f"{base}.mp4"
            try:
                import imageio.v3 as iio

                iio.imwrite(
                    str(mp4_path),
                    stacked,
                    fps=float(self.fps),
                    codec="libx264",
                    macro_block_size=None,
                )
                print(f"[record] wrote {mp4_path} ({stacked.shape[0]} frames)")
                continue
            except Exception as exc:
                print(f"[record] imageio mp4 write failed for {name}: {exc}; falling back to PNG")
            # PNG fallback
            png_dir = self.record_dir / f"episode_{self.episode_idx:02d}" / name
            png_dir.mkdir(parents=True, exist_ok=True)
            try:
                import imageio.v3 as iio

                for i, frm in enumerate(stacked):
                    iio.imwrite(str(png_dir / f"frame_{i:04d}.png"), frm)
            except Exception:
                # Final fallback: raw numpy dump
                _np.save(str(png_dir / "frames.npy"), stacked)
            print(f"[record] wrote {png_dir} ({stacked.shape[0]} frames)")
        self._buffers.clear()


# =============================================================================
# Manual normalize/denormalize (LeRobot 0.4.x stores stats outside the model)
# =============================================================================
def _load_dataset_stats_raw(dataset_root: Path) -> dict:
    import json

    stats_path = dataset_root / "meta" / "stats.json"
    if not stats_path.exists():
        raise SystemExit(
            f"Could not find dataset stats at {stats_path}. Make sure "
            f"--dataset-root points at a LeRobotDataset directory produced "
            f"by tools/convert_hdf5_to_lerobot.py."
        )
    with stats_path.open("r") as fh:
        return json.load(fh)


def _build_manual_normalizer(dataset_root: Path, device: torch.device) -> dict:
    # Reads `meta/stats.json` directly (no LeRobotDataset import) and
    # produces the tensor pairs we need to normalize observation.state and
    # unnormalize the policy's action output. Mirrors the logic from the
    # offline eval script so training and eval use the exact same transform.
    raw = _load_dataset_stats_raw(dataset_root)
    out: dict[str, dict[str, torch.Tensor]] = {}
    for key, substats in raw.items():
        if not isinstance(substats, dict):
            continue
        entry: dict[str, torch.Tensor] = {}
        for name in ("mean", "std", "min", "max"):
            value = substats.get(name)
            if value is None:
                continue
            tensor = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(1, -1)
            entry[name] = tensor
        if "std" in entry:
            entry["std"] = entry["std"].clamp_min(1e-8)
        if "min" in entry and "max" in entry:
            entry["range"] = (entry["max"] - entry["min"]).clamp_min(1e-8)
        out[key] = entry
    return out


def _normalize_state_meanstd(x: torch.Tensor, stats: dict) -> torch.Tensor:
    return (x - stats["mean"]) / stats["std"]


def _denormalize_action_minmax(x: torch.Tensor, stats: dict) -> torch.Tensor:
    # DiffusionPolicy's default action normalization is MIN_MAX in [-1, 1].
    return (x + 1.0) * 0.5 * stats["range"] + stats["min"]


def _read_camera_rgb(cam, device: torch.device) -> torch.Tensor:
    # Isaac Lab camera sensors expose the latest RGB frame via
    # `cam.data.output["rgb"]` as (num_envs, H, W, 3) uint8. We take env 0,
    # permute to (C, H, W), cast to float32 in [0, 1], and add the batch dim
    # so the policy sees `(1, 3, H, W)` — the shape its vision encoder expects.
    raw = cam.data.output["rgb"]
    if not torch.is_tensor(raw):
        raw = torch.as_tensor(raw)
    frame = raw[0]
    if frame.dtype != torch.uint8:
        frame = frame.to(torch.uint8)
    frame = frame.to(device=device, dtype=torch.float32).div_(255.0)
    frame = frame.permute(2, 0, 1).contiguous().unsqueeze(0)
    return frame


def _build_observation(controller, cameras: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    joint_pos = controller.robot.data.joint_pos[0].detach().to(device=device, dtype=torch.float32).unsqueeze(0)
    obs: dict[str, torch.Tensor] = {"observation.state": joint_pos}
    for name in _CAMERA_NAMES:
        cam = cameras.get(name)
        if cam is None:
            continue
        obs[f"observation.images.{name}"] = _read_camera_rgb(cam, device)
    return obs


def _parse_action(
    action_1d: torch.Tensor,
    controller,
    *,
    gripper_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    # `_build_action_tensor` in scene_auto_grasp_collect writes:
    #   [pos_delta_b (3), rot_delta_rotvec (3), gripper (1)]
    # where pos_delta_b = target_pos_b - current_ee_pos_b, and rot_delta is
    # `target_rot * current_rot.inv()` as a 3-vector (scipy rotvec). To invert
    # this on the inference side we add the delta back to the current base-
    # frame EE pose.
    action_np = action_1d.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if action_np.shape[0] < 7:
        raise RuntimeError(f"Expected 7D action, got shape {action_np.shape}")
    pos_delta = action_np[0:3]
    rot_delta_rotvec = action_np[3:6]
    gripper_bit = float(action_np[6])

    ee_pos_b_tensor, ee_quat_b_tensor = controller.current_ee_pose_base()
    current_pos_b = ee_pos_b_tensor[0].detach().cpu().numpy().astype(np.float64)
    current_quat_wxyz = ee_quat_b_tensor[0].detach().cpu().numpy().astype(np.float64)

    target_pos_b_np = current_pos_b + pos_delta
    current_quat_xyzw = np.array(
        [current_quat_wxyz[1], current_quat_wxyz[2], current_quat_wxyz[3], current_quat_wxyz[0]],
        dtype=np.float64,
    )
    current_rot = R.from_quat(current_quat_xyzw)
    delta_rot = R.from_rotvec(rot_delta_rotvec)
    target_rot = delta_rot * current_rot
    target_quat_xyzw = target_rot.as_quat()
    target_quat_b_wxyz = np.array(
        [target_quat_xyzw[3], target_quat_xyzw[0], target_quat_xyzw[1], target_quat_xyzw[2]],
        dtype=np.float64,
    )

    device = ee_pos_b_tensor.device
    target_pos_b = torch.tensor(target_pos_b_np, dtype=torch.float32, device=device).unsqueeze(0)
    target_quat_b = torch.tensor(target_quat_b_wxyz, dtype=torch.float32, device=device).unsqueeze(0)
    gripper_closed = gripper_bit >= float(gripper_threshold)
    return target_pos_b, target_quat_b, gripper_closed


# =============================================================================
# Episode rollout
# =============================================================================
def _run_single_episode(
    *,
    scene,
    controller,
    cameras,
    sync_cameras: Callable[[], None] | None,
    plan,
    base_z: float,
    target_prim_path: str,
    target_snapshot: dict[str, Any] | None,
    policy,
    policy_device: torch.device,
    normalizer: dict,
    args: SceneEvalArgs,
    episode_idx: int,
    async_runner: "AsyncPolicyRunner | None" = None,
    recorder: "EpisodeRecorder | None" = None,
) -> dict[str, Any]:
    # Reset robot + base, then teleport target rigid body back to (optionally
    # randomized) snapshot pose. Same recipe as scene_auto_grasp_collect.
    _reset_scene_to_plan(scene, controller, plan, base_z, sync_cameras)

    if args.target_forward_randomization > 0.0 and target_snapshot is not None:
        fwd_x, fwd_y = _robot_forward_xy_world(controller)
        delta = random.uniform(
            -float(args.target_forward_randomization),
            float(args.target_forward_randomization),
        )
        offset_xy = (fwd_x * delta, fwd_y * delta)
        episode_snapshot = _shifted_target_snapshot(target_snapshot, offset_xy)
        print(
            f"[eval] episode {episode_idx}: randomization delta={delta:+.4f}m "
            f"offset_xy=({offset_xy[0]:+.4f}, {offset_xy[1]:+.4f})"
        )
    else:
        episode_snapshot = target_snapshot

    if episode_snapshot is not None:
        _restore_target_rigid_body_state(episode_snapshot)
        scene.write_data_to_sim()
        controller.sim.step()
        scene.update(controller.sim.get_physics_dt())
        if sync_cameras is not None:
            sync_cameras()

    baseline_target_z = _read_rigid_body_world_position_z(target_prim_path)
    if baseline_target_z is None:
        baseline_target_z = 0.0

    # Reset the policy action queue so each episode starts fresh.
    if hasattr(policy, "reset"):
        policy.reset()

    max_reached_rise = 0.0
    success = False
    steps_run = 0

    state_stats = normalizer.get("observation.state", {})
    action_stats = normalizer.get("action", {})
    if "min" not in action_stats or "max" not in action_stats:
        raise SystemExit("action stats missing min/max in meta/stats.json")

    sim_steps_per_policy = max(1, int(args.sim_steps_per_policy_call))

    with torch.no_grad():
        for frame_idx in range(int(args.max_steps_per_episode)):
            obs = _build_observation(controller, cameras, policy_device)
            # Normalize observation.state with the training-time mean/std.
            # Image features are left untouched — the vision encoder inside
            # DiffusionModel applies its own ImageNet mean/std normalization.
            if state_stats and "observation.state" in obs:
                obs["observation.state"] = _normalize_state_meanstd(
                    obs["observation.state"], state_stats
                )
            if async_runner is not None:
                action = async_runner.select_action(obs)
            else:
                action = policy.select_action(obs)
            if action.ndim > 1:
                action = action[0]
            # Denormalize from [-1, 1] back to the real action data space.
            action_denorm = _denormalize_action_minmax(
                action.unsqueeze(0), action_stats
            )[0]

            if frame_idx < 5 or frame_idx % 10 == 0:
                a = action_denorm.detach().cpu().numpy().reshape(-1)
                print(
                    f"[eval frame {frame_idx:3d}] "
                    f"pos_d=({a[0]:+.4f}, {a[1]:+.4f}, {a[2]:+.4f}) "
                    f"rot_d=({a[3]:+.4f}, {a[4]:+.4f}, {a[5]:+.4f}) "
                    f"grip={a[6]:+.3f}"
                )

            target_pos_b, target_quat_b, gripper_closed = _parse_action(
                action_denorm, controller, gripper_threshold=args.gripper_threshold
            )
            # Apply the same commanded EE target across `sim_steps_per_policy`
            # physics sub-steps, matching the training dataset's temporal
            # structure (10 Hz policy frames, ~60 Hz physics). Without this
            # amortization the robot would move 6x slower than the training
            # distribution implies and never reach the bolt.
            for sub_idx in range(sim_steps_per_policy):
                controller.step_pose_target(target_pos_b, target_quat_b, gripper_closed)
                scene.write_data_to_sim()
                controller.sim.step()
                scene.update(controller.sim.get_physics_dt())
                if sync_cameras is not None:
                    sync_cameras()
            # Record one frame per policy call, not per physics sub-step, so
            # the output video plays back at the training capture rate (10 Hz
            # by default) and stays in sync with the action log.
            if recorder is not None:
                recorder.capture(cameras)

            current_z = _read_rigid_body_world_position_z(target_prim_path)
            if current_z is not None:
                rise = float(current_z) - float(baseline_target_z)
                if rise > max_reached_rise:
                    max_reached_rise = rise
                if rise >= float(args.success_lift_delta):
                    success = True
                    steps_run = frame_idx + 1
                    break
            steps_run = frame_idx + 1

    result = {
        "episode": episode_idx,
        "success": bool(success),
        "steps": int(steps_run),
        "max_rise": float(max_reached_rise),
        "baseline_z": float(baseline_target_z),
    }
    tag = "SUCCESS" if success else "FAILED "
    print(f"[eval] episode {episode_idx}: {tag} steps={steps_run} max_rise={max_reached_rise:.4f}m")
    return result


# =============================================================================
# Main entry
# =============================================================================
def run_scene_eval(simulation_app, robot_name: str, args: SceneEvalArgs) -> None:
    import isaaclab.sim as sim_utils
    from .scene_mouse_collect import SceneMouseCollectArgs

    if args.random_seed is not None:
        random.seed(int(args.random_seed))
        torch.manual_seed(int(args.random_seed))

    # `_make_spec_for_scene_collect` expects a concrete side ("left" / "right"),
    # not "auto". Auto-grasp collection defaults to "left" when the user asks
    # for "auto", so we mirror that here.
    initial_arm_side = "left" if args.arm_side_preference == "auto" else args.arm_side_preference

    # Re-use scene_auto_grasp_collect's SceneMouseCollectArgs builder so the
    # scene we eval in is identical to the one we trained on.
    collect_args = SceneMouseCollectArgs(
        device=args.device,
        num_envs=args.num_envs,
        dataset_file="/tmp/scene_eval_unused.hdf5",
        capture_hz=10.0,
        append=False,
        lin_step=0.015,
        ang_step=0.10,
        scene_usd_path=args.scene_usd_path,
        scene_graph_path=args.scene_graph_path,
        placements_path=args.placements_path,
        target=args.target,
        support=args.support,
        object_collision_approx="convex_decomposition",
        target_collision_approx="convex_decomposition",
        convex_decomp_voxel_resolution=1_000_000,
        convex_decomp_max_convex_hulls=64,
        convex_decomp_error_percentage=2.0,
        convex_decomp_shrink_wrap=True,
        plan_output_dir=args.plan_output_dir,
        base_z_bias=args.base_z_bias,
        arm_side=initial_arm_side,
        show_workspace=False,
    )

    sim_cfg = sim_utils.SimulationCfg(device=args.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    (
        scene,
        controller,
        cameras,
        sync_cameras,
        _camera_aliases,
        plan,
        _effective_base_z_bias,
        aligned_base_z,
        _physics_rebind_summary,
        _floor_realign_summary,
        _workspace_visual_summary,
    ) = _build_scene_mouse_collect(sim, robot_name, collect_args)

    scene_root_path = f"{scene.env_prim_paths[0]}/GeneratedScene"
    target_live_prim_path = f"{scene_root_path}/{Path(plan.target_prim).name}"

    # Capture the canonical target pose once, with the scene fully settled,
    # so every episode can snap it back to the same "rest" configuration.
    target_snapshot = _snapshot_target_rigid_body_state(target_live_prim_path)
    if target_snapshot is None:
        print(f"[eval] WARNING: could not snapshot target rigid body at {target_live_prim_path}; "
              f"episodes will rely on _reset_scene_to_plan alone.")

    # Load policy AFTER isaac boot so import failures don't kill scene setup.
    policy_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy = _load_policy(args.checkpoint, str(policy_device))
    _maybe_reduce_inference_steps(policy, int(args.num_inference_steps))

    # Optional async inference wrapper. Uses a single background thread to
    # pre-compute the next action chunk while the current one is being
    # executed, hiding the diffusion denoising latency.
    async_runner: AsyncPolicyRunner | None = None
    if bool(args.async_inference):
        async_runner = AsyncPolicyRunner(policy, prefetch_when_remaining=2)
        print("[eval] async inference wrapper enabled")

    # Build the manual normalize/denormalize pair from the training dataset's
    # stats.json. This is the critical fix that turns nonsense policy
    # outputs (in normalized [-1, 1] space) into real-space delta actions.
    dataset_root_path = Path(args.dataset_root).expanduser().resolve()
    normalizer = _build_manual_normalizer(dataset_root_path, policy_device)
    print(
        f"[eval] loaded normalizer stats for keys: "
        f"{[k for k, v in normalizer.items() if v]}"
    )

    record_dir = Path(args.record_dir).expanduser().resolve() if args.record_dir else None
    if record_dir is not None:
        record_dir.mkdir(parents=True, exist_ok=True)
        print(f"[eval] recording per-episode camera videos to {record_dir}")

    results: list[dict[str, Any]] = []
    for episode_idx in range(int(args.num_episodes)):
        recorder = (
            EpisodeRecorder(record_dir, episode_idx=episode_idx, fps=args.record_fps)
            if record_dir is not None
            else None
        )
        try:
            result = _run_single_episode(
                scene=scene,
                controller=controller,
                cameras=cameras,
                sync_cameras=sync_cameras,
                plan=plan,
                base_z=aligned_base_z,
                target_prim_path=target_live_prim_path,
                target_snapshot=target_snapshot,
                policy=policy,
                policy_device=policy_device,
                normalizer=normalizer,
                args=args,
                episode_idx=episode_idx,
                async_runner=async_runner,
                recorder=recorder,
            )
        except Exception as exc:
            print(f"[eval] episode {episode_idx}: rollout raised: {exc}")
            result = {
                "episode": episode_idx,
                "success": False,
                "steps": 0,
                "max_rise": 0.0,
                "error": str(exc),
            }
        if recorder is not None:
            recorder.flush()
        results.append(result)

    total = len(results)
    n_success = sum(1 for r in results if r.get("success"))
    print("\n" + "=" * 60)
    print(f"[eval] SUMMARY: {n_success}/{total} success "
          f"({100.0 * n_success / max(1, total):.1f}%)")
    mean_rise = sum(float(r.get("max_rise", 0.0)) for r in results) / max(1, total)
    print(f"[eval] mean max_rise: {mean_rise:.4f}m "
          f"(success threshold: {args.success_lift_delta}m)")
    print("=" * 60)
