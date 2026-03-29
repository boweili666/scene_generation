from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAM3D_ROOT = Path(
    os.environ.get("SAM3D_OBJECTS_ROOT", PROJECT_ROOT / "third_party" / "sam-3d-objects")
).resolve()


def default_config_path(sam3d_root: Path) -> Path:
    return sam3d_root.resolve() / "checkpoints" / "hf" / "pipeline.yaml"


def ensure_sam3d_imports(sam3d_root: Path) -> tuple[Path, Path]:
    root = sam3d_root.resolve()
    notebook_dir = root / "notebook"

    for path in (root, notebook_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return root, notebook_dir


def validate_sam3d_layout(
    sam3d_root: Path,
    *,
    config_path: Path | None = None,
) -> tuple[Path, Path]:
    root = sam3d_root.resolve()
    config = (config_path or default_config_path(root)).resolve()

    if not root.exists():
        raise FileNotFoundError(
            f"SAM3D root not found: {root}. "
            "Initialize third_party/sam-3d-objects or set SAM3D_OBJECTS_ROOT to a populated checkout."
        )

    required_paths = [
        root / "notebook" / "inference.py",
        root / "sam3d_objects" / "__init__.py",
    ]
    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required SAM3D file not found: {required_path}. "
                "Make sure third_party/sam-3d-objects is initialized with the expected layout."
            )

    if not config.exists():
        raise FileNotFoundError(
            f"SAM3D pipeline config not found: {config}. "
            "Pass --config or ensure checkpoints/hf/pipeline.yaml exists in the SAM3D checkout."
        )

    return root, config
