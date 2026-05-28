"""Re-layout an existing LeRobotDataset's observation.state in place (copy).

Why this exists: the 20-episode `agibot_bolt_grasp` bolt dataset only survives
in LeRobot v3.0 format — its source HDF5 is gone. To retrain a real-robot
deployable policy we need observation.state reduced from the raw 34-D Isaac
joint_pos to the 20-D real-robot state, WITHOUT losing any of the 20
episodes / 919 frames or re-encoding the AV1 videos.

So instead of HDF5 -> LeRobot (impossible, no HDF5), this does
LeRobot -> LeRobot: copy the dataset directory verbatim, then surgically
rewrite only the state-dependent parts:

  1. data/**.parquet      : observation.state column 34-D -> 20-D
  2. meta/info.json       : features.observation.state.shape [34] -> [20]
  3. meta/stats.json      : observation.state global stats recomputed (20-D)
  4. meta/episodes/**.parquet : per-episode stats/observation.state/* (20-D)

Videos, images, action, timestamps and all indices are byte-identical to the
source. The 34->20 reduction reuses the verified, official-Genie-Sim-aligned
`joint_pos_to_slim20` from convert_hdf5_to_lerobot.py (G1 dof_order slice +
omnipicker_sim_to_real grippers), so this dataset is consistent with the
deploy-time state builder.

Usage:
    python tools/relayout_lerobot_state.py \
        --src datasets/lerobot/agibot_bolt_grasp \
        --dst datasets/lerobot/agibot_bolt_grasp_slim20 \
        [--overwrite]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_THIS = Path(__file__).resolve()


def _load_slim20():
    """Reuse joint_pos_to_slim20 / SLIM20_DIM from the HDF5 converter so the
    34->20 reduction is defined in exactly one place."""
    conv_path = _THIS.parent / "convert_hdf5_to_lerobot.py"
    spec = importlib.util.spec_from_file_location("_conv_slim20", conv_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.joint_pos_to_slim20, int(mod.SLIM20_DIM)


def _stats_block(arr: np.ndarray) -> dict:
    """LeRobot per-feature stats block. arr: (N, D) float."""
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q10": np.quantile(arr, 0.10, axis=0).tolist(),
        "q50": np.quantile(arr, 0.50, axis=0).tolist(),
        "q90": np.quantile(arr, 0.90, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def _rewrite_data_parquets(dst: Path, reduce_fn) -> dict[int, np.ndarray]:
    """Rewrite observation.state in every data parquet. Returns
    {episode_index: stacked 20-D state} for per-episode stats."""
    data_files = sorted((dst / "data").rglob("*.parquet"))
    if not data_files:
        raise SystemExit(f"No data parquet found under {dst/'data'}")
    per_ep: dict[int, list[np.ndarray]] = {}
    for pf in data_files:
        table = pq.read_table(pf)
        state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
        if state.ndim != 2 or state.shape[1] < 34:
            raise SystemExit(
                f"{pf}: observation.state is {state.shape}, expected (N, >=34). "
                "This dataset is not raw 34-D joint_pos; aborting to avoid corruption."
            )
        new_state = np.stack([reduce_fn(row) for row in state]).astype(np.float32)
        ep_idx = np.asarray(table.column("episode_index").to_pylist())
        for e in np.unique(ep_idx):
            per_ep.setdefault(int(e), []).append(new_state[ep_idx == e])

        # Replace the column, preserving every other column + schema order.
        col_i = table.column_names.index("observation.state")
        new_col = pa.array(list(new_state), type=pa.list_(pa.float32()))
        table = table.set_column(col_i, "observation.state", new_col)
        pq.write_table(table, pf)
        print(f"[data] {pf.relative_to(dst)}: state {state.shape} -> {new_state.shape}")
    return {e: np.concatenate(chunks) for e, chunks in per_ep.items()}


def _patch_info(dst: Path, new_dim: int) -> None:
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    feat = info["features"]["observation.state"]
    old = list(feat.get("shape", []))
    feat["shape"] = [new_dim]
    info_path.write_text(json.dumps(info, indent=4))
    print(f"[meta] info.json observation.state.shape {old} -> [{new_dim}]")


def _patch_global_stats(dst: Path, all_state: np.ndarray) -> None:
    sp = dst / "meta" / "stats.json"
    stats = json.loads(sp.read_text())
    stats["observation.state"] = _stats_block(all_state)
    sp.write_text(json.dumps(stats, indent=4))
    print(f"[meta] stats.json observation.state recomputed over {all_state.shape}")


def _patch_episode_stats(dst: Path, per_ep: dict[int, np.ndarray]) -> None:
    ep_files = sorted((dst / "meta" / "episodes").rglob("*.parquet"))
    if not ep_files:
        print("[meta] no meta/episodes parquet (older layout); skipped")
        return
    prefix = "stats/observation.state/"
    for pf in ep_files:
        table = pq.read_table(pf)
        cols = table.column_names
        ep_index = np.asarray(table.column("episode_index").to_pylist())
        # Build replacement arrays per stat key, row-aligned to this parquet.
        stat_keys = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]
        new_vals: dict[str, list] = {k: [] for k in stat_keys}
        for e in ep_index:
            block = _stats_block(per_ep[int(e)])
            for k in stat_keys:
                new_vals[k].append(block[k])
        for k in stat_keys:
            cname = prefix + k
            if cname not in cols:
                continue
            ci = cols.index(cname)
            ref = table.column(ci)
            arr = pa.array(new_vals[k], type=ref.type)
            table = table.set_column(ci, cname, arr)
        pq.write_table(table, pf)
        print(f"[meta] episodes {pf.relative_to(dst)}: per-episode state stats rewritten ({len(ep_index)} eps)")


def relayout(src: Path, dst: Path, overwrite: bool) -> None:
    reduce_fn, new_dim = _load_slim20()
    if not (src / "meta" / "info.json").exists():
        raise SystemExit(f"{src} is not a LeRobotDataset (no meta/info.json)")
    if dst.exists():
        if not overwrite:
            raise SystemExit(f"{dst} exists. Pass --overwrite to replace it.")
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"[copy] {src} -> {dst}")
    shutil.copytree(src, dst)

    per_ep = _rewrite_data_parquets(dst, reduce_fn)
    all_state = np.concatenate([per_ep[e] for e in sorted(per_ep)])
    _patch_info(dst, new_dim)
    _patch_global_stats(dst, all_state)
    _patch_episode_stats(dst, per_ep)

    print(
        f"\n[done] {dst}\n"
        f"       episodes={len(per_ep)} frames={all_state.shape[0]} "
        f"state_dim={all_state.shape[1]} (was 34)"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="Source LeRobotDataset root (34-D state)")
    p.add_argument("--dst", type=Path, required=True, help="Destination root (20-D slim state)")
    p.add_argument("--overwrite", action="store_true", help="Delete --dst first if it exists")
    args = p.parse_args(argv)
    relayout(args.src, args.dst, args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
