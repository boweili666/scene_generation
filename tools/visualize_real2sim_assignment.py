#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.backend.services.real2sim_assignment_visualization import build_assignment_visualization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a static HTML and bbox overlay for a Real2Sim assignment.json file.",
    )
    parser.add_argument("assignment_json", help="Path to assignment.json")
    parser.add_argument(
        "--html-output",
        help="Optional output path for the generated HTML review. Defaults to assignment_review.html next to assignment.json.",
    )
    parser.add_argument(
        "--bbox-output",
        help="Optional output path for the generated bbox overlay image. Defaults to assignment_bbox_overlay.png next to assignment.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = build_assignment_visualization(
        args.assignment_json,
        html_output_path=args.html_output,
        bbox_output_path=args.bbox_output,
    )
    print(f"assignment_json: {result['assignment_path']}")
    print(f"bbox_overlay: {result['bbox_output_path']}")
    print(f"html_review: {result['html_output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
