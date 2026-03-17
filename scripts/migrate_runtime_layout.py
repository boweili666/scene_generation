#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

from app.backend.config.settings import (
    DEFAULT_RENDER_PATH,
    LATEST_INPUT_IMAGE,
    LEGACY_SCENE_GRAPH_PATH,
    LOGS_DIR,
    PROJECT_ROOT,
    RUNTIME_DIR,
    SCENE_GRAPH_PATH,
    ensure_runtime_layout,
)


def _copy_file(src: Path, dst: Path) -> bool:
    if not src.exists() or dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _copy_tree(src: Path, dst: Path) -> int:
    if not src.exists() or not src.is_dir():
        return 0
    copied = 0
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        target = dst / rel
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def _copy_matching_files(src_dir: Path, pattern: str, dst_dir: Path) -> int:
    if not src_dir.exists():
        return 0
    copied = 0
    for path in src_dir.glob(pattern):
        if not path.is_file():
            continue
        target = dst_dir / path.name
        if target.exists():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def main() -> None:
    ensure_runtime_layout()

    copied = []
    copied.append(("real2sim_log", _copy_file(PROJECT_ROOT / "real2sim.log", LOGS_DIR / "real2sim.log")))
    copied.append(("scene_service_log", _copy_file(PROJECT_ROOT / "scene_service.log", LOGS_DIR / "scene_service.log")))
    copied.append(("latest_input", _copy_file(PROJECT_ROOT / "web" / "assets" / "uploads" / "latest_input.jpg", LATEST_INPUT_IMAGE)))
    copied.append(("render_png", _copy_file(PROJECT_ROOT / "web" / "assets" / "renders" / "render.png", DEFAULT_RENDER_PATH)))
    copied.append(("scene_graph", _copy_file(LEGACY_SCENE_GRAPH_PATH, Path(SCENE_GRAPH_PATH))))
    runtime_count = _copy_tree(PROJECT_ROOT / "option2_pipeline" / "runtime", RUNTIME_DIR / "real2sim")
    usd_count = _copy_matching_files(
        PROJECT_ROOT / "isaac_local" / "my_viewer",
        "generated_room.scene_service*.usd",
        RUNTIME_DIR / "scene_service" / "usd",
    )

    print("Migration summary:")
    for label, ok in copied:
        print(f"  {label}: {'copied' if ok else 'skipped'}")
    print(f"  real2sim_runtime_files: {runtime_count}")
    print(f"  scene_service_usd_files: {usd_count}")


if __name__ == "__main__":
    main()
