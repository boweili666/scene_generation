"""Export a slim20 LeRobotDataset's normalization stats to norm_stats.json.

LeRobot 0.4.x `DiffusionPolicy.from_pretrained()` does NOT persist the
dataset normalization stats inside the checkpoint, so a deployed policy
loaded only from the checkpoint would normalize observation.state and
de-normalize the action with identity buffers — every predicted action
would come out in normalized [-1, 1] space and the robot would misbehave.

This dumps the `observation.state` and `action` mean/std/min/max from the
training dataset's meta/stats.json into a small JSON that ships next to the
checkpoint. diffusion_policy_deploy.py reads it to rebuild the exact
MEAN_STD (state) / MIN_MAX (action) transforms used in training, matching
the offline and sim-eval normalizer paths.

Usage (lerobot env):
    python tools/export_norm_stats.py \
        --dataset-root datasets/lerobot/agibot_bolt_grasp_slim20 \
        --out outputs/train/diff_agibot_bolt_slim20/checkpoints/last/pretrained_model/norm_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_FEATURES = ("observation.state", "action")
_KEYS = ("mean", "std", "min", "max")


def export(dataset_root: Path, out: Path) -> None:
    stats_path = dataset_root / "meta" / "stats.json"
    if not stats_path.exists():
        raise SystemExit(f"{stats_path} not found — is this a LeRobotDataset root?")
    stats = json.loads(stats_path.read_text())

    payload: dict = {}
    for feat in _FEATURES:
        sub = stats.get(feat)
        if not isinstance(sub, dict):
            raise SystemExit(f"stats.json has no '{feat}' block")
        block = {}
        for k in _KEYS:
            if k not in sub:
                raise SystemExit(f"stats.json['{feat}'] missing '{k}'")
            block[k] = list(sub[k])
        payload[feat] = block

    sdim = len(payload["observation.state"]["mean"])
    adim = len(payload["action"]["mean"])
    if sdim != 20:
        raise SystemExit(
            f"observation.state is {sdim}-D, expected 20 (slim20). "
            "Point --dataset-root at the slim20 dataset."
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[done] wrote {out}  (state {sdim}-D, action {adim}-D)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    export(args.dataset_root, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
