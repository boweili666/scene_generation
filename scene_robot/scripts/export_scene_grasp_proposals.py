#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.services.grasp_scene_adapter import build_scene_grasp_proposals, default_scene_grasp_proposals_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export world-frame grasp proposals for a scene USD.")
    parser.add_argument("--scene-usd-path", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()
    output_path = args.output_path or default_scene_grasp_proposals_path(args.scene_usd_path)

    payload = build_scene_grasp_proposals(
        args.scene_usd_path,
        args.manifest_path,
        annotation_root=args.annotation_root,
        output_path=output_path,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
