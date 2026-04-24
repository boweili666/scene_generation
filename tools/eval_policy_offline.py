"""Offline evaluation of a trained LeRobot Diffusion Policy.

Runs the policy on every frame of the training LeRobotDataset and compares
the predicted action against the ground-truth recorded action. This gives
an immediate signal for whether training actually learned the imitation
target (no Isaac Sim required, no closed-loop rollout), and it runs in the
`lerobot` conda env where `lerobot-train` already works.

Usage (in the lerobot env):

    python tools/eval_policy_offline.py \
        --checkpoint outputs/train/diff_agibot_bolt_v1/checkpoints/last/pretrained_model \
        --dataset-repo-id local/agibot_bolt_grasp \
        --dataset-root datasets/lerobot/agibot_bolt_grasp

The per-dim MSE breakdown is the most informative output: the 7D action
layout is [pos_delta(3), rot_delta_rotvec(3), gripper(1)], so large MSE
on dims 0..2 means translation is off, dims 3..5 means rotation is off,
dim 6 means the gripper timing is off.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def _load_policy_and_dataset(checkpoint: Path, repo_id: str, root: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    # Accept either the `pretrained_model` dir itself or a parent that
    # contains it (same convention as the closed-loop eval script).
    candidate_dirs = [checkpoint]
    for sub in ("pretrained_model", "last/pretrained_model", "checkpoints/last/pretrained_model"):
        candidate_dirs.append(checkpoint / sub)
    resolved = None
    for d in candidate_dirs:
        if (d / "config.json").exists() or (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists():
            resolved = d
            break
    if resolved is None:
        raise SystemExit(
            f"Could not find a loadable pretrained_model folder under {checkpoint}. "
            f"Tried: {[str(d) for d in candidate_dirs]}"
        )

    # Load the dataset first so we can hand its per-feature mean/std stats
    # to the policy at construction time. LeRobot's DiffusionPolicy stores
    # Normalize/Unnormalize buffers, but `save_pretrained` does NOT persist
    # the computed stats inside `model.safetensors`, so calling
    # `from_pretrained` alone leaves the buffers at identity and every
    # predicted action comes out in normalized space (≈[-1, 1]) instead of
    # the real data space. Re-supply the stats from the training dataset.
    print(f"[offline_eval] loading dataset {repo_id} from {root}")
    ds = LeRobotDataset(repo_id, root=str(root))
    print(
        f"[offline_eval] dataset loaded: "
        f"episodes={ds.num_episodes} frames={ds.num_frames} fps={ds.fps}"
    )
    dataset_stats = getattr(ds.meta, "stats", None)
    if dataset_stats is None:
        raise SystemExit(
            "Dataset has no `meta.stats` — re-run the dataset converter or re-save the dataset."
        )

    print(f"[offline_eval] loading policy from {resolved}")
    policy = DiffusionPolicy.from_pretrained(str(resolved))
    policy.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    # In LeRobot 0.4.x the diffusion policy's normalizers live in an
    # external processor chain, not inside the nn.Module. `save_pretrained`
    # persists only the diffusion UNet weights and its config, so after
    # `from_pretrained` the policy outputs (and expects) normalized values.
    # We fix that by building manual normalize/denormalize helpers from the
    # training dataset_stats that the user just loaded.
    normalizer = _build_manual_normalizer(dataset_stats, device)
    return policy, ds, device, normalizer


def _extract_stat_tensor(substats: dict, name: str) -> torch.Tensor | None:
    value = substats.get(name)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().clone().float()
    import numpy as np_local

    return torch.as_tensor(np_local.asarray(value), dtype=torch.float32)


def _build_manual_normalizer(dataset_stats: dict, device: torch.device) -> dict:
    # For each feature we may use either MEAN_STD or MIN_MAX normalization.
    # DiffusionPolicy defaults: observation.state=MEAN_STD, action=MIN_MAX,
    # images=MEAN_STD with ImageNet stats applied inside the vision backbone.
    # We compute BOTH sets of (scale, offset) per feature so the caller can
    # pick whichever matches the training config. Values are stored on the
    # target device and broadcast-shaped to (1, D).
    out: dict[str, dict[str, torch.Tensor]] = {}
    for key, substats in dataset_stats.items():
        if not isinstance(substats, dict):
            continue
        entry: dict[str, torch.Tensor] = {}
        mean = _extract_stat_tensor(substats, "mean")
        std = _extract_stat_tensor(substats, "std")
        min_ = _extract_stat_tensor(substats, "min")
        max_ = _extract_stat_tensor(substats, "max")
        if mean is not None and std is not None:
            entry["mean"] = mean.to(device).reshape(1, -1)
            entry["std"] = std.to(device).reshape(1, -1).clamp_min(1e-8)
        if min_ is not None and max_ is not None:
            entry["min"] = min_.to(device).reshape(1, -1)
            entry["max"] = max_.to(device).reshape(1, -1)
            # DiffusionPolicy uses MIN_MAX -> [-1, 1]: x' = 2*(x-min)/(max-min) - 1
            range_ = (entry["max"] - entry["min"]).clamp_min(1e-8)
            entry["range"] = range_
        out[key] = entry
    return out


def _normalize_state_meanstd(x: torch.Tensor, stats: dict) -> torch.Tensor:
    return (x - stats["mean"]) / stats["std"]


def _denormalize_action_meanstd(x: torch.Tensor, stats: dict) -> torch.Tensor:
    return x * stats["std"] + stats["mean"]


def _denormalize_action_minmax(x: torch.Tensor, stats: dict) -> torch.Tensor:
    return (x + 1.0) * 0.5 * stats["range"] + stats["min"]


def _dump_normalizer_structure(policy, after: bool = False) -> None:
    # Introspect ALL submodules of the policy and print any that look like
    # normalizers (type name contains "Normalize" or buffers with stat-like
    # names). LeRobot 0.4.x can bury normalizers one or two levels deep
    # inside policy.model / policy.diffusion etc., and the attribute name
    # varies between releases, so a full walk is more reliable than
    # hand-picking a few names.
    label = "AFTER" if after else "BEFORE"
    print(f"[offline_eval] normalizer scan {label}:")
    found = 0
    for name, module in policy.named_modules():
        tname = type(module).__name__
        has_stat_buffer = any(
            hasattr(module, b) and getattr(module, b) is not None
            for b in ("mean", "std", "min", "max")
        )
        is_normalizer_like = (
            "Normalize" in tname or "Unnormalize" in tname or has_stat_buffer
        )
        if not is_normalizer_like:
            continue
        path = name or "<root>"
        _print_buffer_summary(f"  [{path}] {tname}", module)
        found += 1
    if found == 0:
        print("  (no Normalize-like modules found in policy module tree)")
        # As a fallback also list the top-level children so we can see
        # roughly where things live.
        print("  top-level policy children:")
        for cname, child in policy.named_children():
            print(f"    {cname}: {type(child).__name__}")


def _print_buffer_summary(prefix: str, module) -> None:
    keys_found = []
    for bname in ("mean", "std", "min", "max"):
        buf = getattr(module, bname, None)
        if buf is None:
            continue
        try:
            arr = buf.detach().cpu().numpy().reshape(-1)
            if arr.size == 0:
                keys_found.append(f"{bname}=<empty>")
            else:
                keys_found.append(
                    f"{bname}=[{arr.min():.4f}..{arr.max():.4f}] (shape={list(buf.shape)})"
                )
        except Exception as exc:
            keys_found.append(f"{bname}=<err {exc}>")
    if not keys_found:
        # Maybe the child is itself a ModuleDict of per-feature normalizers.
        if hasattr(module, "items"):
            for key, child in module.items():  # type: ignore[union-attr]
                _print_buffer_summary(f"{prefix}   ↳ [{key}]", child)
            return
        print(f"{prefix}: (no mean/std/min/max buffers)")
        return
    print(f"{prefix}: {' '.join(keys_found)}")


def _apply_dataset_stats_to_policy(policy, dataset_stats: dict) -> None:
    # Walk the policy's normalize_inputs / normalize_targets / unnormalize_outputs
    # submodules and replace their `mean` / `std` / `min` / `max` buffers with
    # the values from the training dataset stats dict. Key names inside the
    # normalizer buffers follow the feature keys (e.g. `observation.state`,
    # `action`).
    targets = []
    for attr_name in ("normalize_inputs", "normalize_targets", "unnormalize_outputs"):
        mod = getattr(policy, attr_name, None)
        if mod is not None:
            targets.append(mod)
    if not targets:
        print("[offline_eval] warning: policy has no normalize_* submodules, stats not applied")
        return

    applied: set[str] = set()
    for mod in targets:
        for key, substats in dataset_stats.items():
            buffer_submod = None
            # lerobot stores one sub-Normalize per feature under mod.buffer_<key_sanitized>
            # but the common pattern is mod[key] OR mod.buffer[key]. Try both.
            try:
                buffer_submod = mod[key]  # type: ignore[index]
            except Exception:
                pass
            if buffer_submod is None:
                continue
            for stat_name in ("mean", "std", "min", "max"):
                if stat_name not in substats:
                    continue
                stat_tensor = substats[stat_name]
                if not torch.is_tensor(stat_tensor):
                    stat_tensor = torch.as_tensor(stat_tensor)
                if hasattr(buffer_submod, stat_name):
                    getattr(buffer_submod, stat_name).data.copy_(
                        stat_tensor.to(getattr(buffer_submod, stat_name).device)
                    )
                    applied.add(f"{key}.{stat_name}")
    print(f"[offline_eval] manually applied {len(applied)} normalizer stats")


def _frame_to_batch(frame: dict, device: torch.device) -> tuple[dict, torch.Tensor]:
    # A LeRobotDataset[i] frame is a dict of tensors (no batch dim). Add a
    # leading batch dim for every observation.* key and move to device. The
    # ground-truth action is returned separately so it can be compared to
    # the policy output.
    obs_batch: dict[str, torch.Tensor] = {}
    for key, value in frame.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key.startswith("observation."):
            obs_batch[key] = value.unsqueeze(0).to(device)
    gt_action = frame["action"].to(device)
    return obs_batch, gt_action


def _episode_index_from_frame(frame: dict) -> int | None:
    idx = frame.get("episode_index")
    if idx is None:
        return None
    if isinstance(idx, torch.Tensor):
        return int(idx.item())
    return int(idx)


def evaluate(
    *,
    checkpoint: Path,
    repo_id: str,
    root: Path,
    max_frames: int | None,
    stride: int,
    action_norm: str,
) -> None:
    policy, ds, device, normalizer = _load_policy_and_dataset(checkpoint, repo_id, root)
    state_stats = normalizer.get("observation.state", {})
    action_stats = normalizer.get("action", {})
    if not state_stats:
        print("[offline_eval] warning: no observation.state stats; state will not be normalized")
    if not action_stats:
        raise SystemExit("[offline_eval] no action stats in dataset; cannot denormalize policy output")
    if action_norm == "mean_std" and "mean" not in action_stats:
        raise SystemExit("[offline_eval] action_norm=mean_std requested but dataset has no mean/std for action")
    if action_norm == "min_max" and "min" not in action_stats:
        raise SystemExit("[offline_eval] action_norm=min_max requested but dataset has no min/max for action")
    print(f"[offline_eval] using action_norm={action_norm}")

    n_total = len(ds)
    if max_frames is not None:
        n_total = min(n_total, int(max_frames))

    prev_episode: int | None = None
    per_frame_mse: list[np.ndarray] = []
    per_dim_sq_error = np.zeros(7, dtype=np.float64)
    counted = 0

    with torch.no_grad():
        for i in range(0, n_total, max(1, int(stride))):
            frame = ds[i]
            episode = _episode_index_from_frame(frame)
            if episode is not None and episode != prev_episode:
                # New episode => clear the Diffusion action queue so the
                # policy starts fresh (consistent with how a real rollout
                # would look).
                if hasattr(policy, "reset"):
                    policy.reset()
                prev_episode = episode

            obs_batch, gt_action = _frame_to_batch(frame, device)
            # Manually normalize `observation.state` before feeding the
            # policy. Image features are left untouched — the vision encoder
            # inside DiffusionModel applies its own ImageNet mean/std
            # normalization.
            if state_stats and "observation.state" in obs_batch:
                obs_batch["observation.state"] = _normalize_state_meanstd(
                    obs_batch["observation.state"], state_stats
                )
            pred = policy.select_action(obs_batch)
            if pred.ndim > 1:
                pred = pred[0]
            # Denormalize the action back to real data space.
            pred_batched = pred.unsqueeze(0)
            if action_norm == "mean_std":
                pred_batched = _denormalize_action_meanstd(pred_batched, action_stats)
            else:
                pred_batched = _denormalize_action_minmax(pred_batched, action_stats)
            pred = pred_batched[0]

            pred_np = pred.detach().cpu().numpy().astype(np.float64).reshape(-1)
            gt_np = gt_action.detach().cpu().numpy().astype(np.float64).reshape(-1)
            if pred_np.shape[0] != gt_np.shape[0]:
                print(
                    f"[offline_eval] frame {i}: action shape mismatch "
                    f"pred={pred_np.shape} gt={gt_np.shape}"
                )
                continue
            sq_err = (pred_np - gt_np) ** 2
            per_dim_sq_error[: sq_err.shape[0]] += sq_err
            per_frame_mse.append(sq_err.mean())
            counted += 1

            if counted <= 5 or counted % 100 == 0:
                print(
                    f"[offline_eval] frame {i} (ep {episode}): "
                    f"pred={pred_np.round(4).tolist()} "
                    f"gt={gt_np.round(4).tolist()} "
                    f"sqerr_mean={sq_err.mean():.6f}"
                )

    if counted == 0:
        print("[offline_eval] no frames evaluated")
        return

    mse_vec = np.asarray(per_frame_mse, dtype=np.float64)
    per_dim_mse = per_dim_sq_error / counted

    print("\n" + "=" * 60)
    print(f"[offline_eval] frames evaluated: {counted}")
    print(f"[offline_eval] mean per-frame MSE: {mse_vec.mean():.6f}")
    print(f"[offline_eval] median per-frame MSE: {np.median(mse_vec):.6f}")
    print(f"[offline_eval] p95 per-frame MSE: {np.quantile(mse_vec, 0.95):.6f}")
    print(f"[offline_eval] max per-frame MSE: {mse_vec.max():.6f}")
    print("[offline_eval] per-dim MSE (action[0:3]=pos, [3:6]=rotvec, [6]=gripper):")
    for dim, val in enumerate(per_dim_mse):
        tag = ("pos" if dim < 3 else ("rot" if dim < 6 else "grip"))
        print(f"    action[{dim}] ({tag}): {val:.6f}  rmse={np.sqrt(val):.6f}")
    print("=" * 60)
    print(
        "\n[hint] pos rmse in meters, rot rmse in radians, gripper rmse unitless."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the LeRobot pretrained_model dir (or a parent containing it)")
    parser.add_argument("--dataset-repo-id", type=str, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after evaluating N frames")
    parser.add_argument("--stride", type=int, default=1, help="Evaluate every Nth frame (1 = every frame)")
    parser.add_argument(
        "--action-norm",
        type=str,
        default="min_max",
        choices=["min_max", "mean_std"],
        help="Unnormalization scheme for the policy's action output. DiffusionPolicy "
             "defaults to MIN_MAX (maps [-1, 1] -> [min, max]) but some configs use "
             "MEAN_STD. If min_max gives garbage try switching to mean_std.",
    )
    args = parser.parse_args(argv)

    evaluate(
        checkpoint=args.checkpoint,
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        max_frames=args.max_frames,
        stride=args.stride,
        action_norm=args.action_norm,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
