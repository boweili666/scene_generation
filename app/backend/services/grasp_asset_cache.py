from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

from app.backend.config import PROJECT_ROOT


SCENE_ROBOT_SRC = PROJECT_ROOT / "scene_robot" / "src"
if str(SCENE_ROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(SCENE_ROBOT_SRC))

from grasp_annotator import PipelineConfig, annotate_single_object


ASSET_GRASP_CACHE_SCHEMA_VERSION = "grasp_asset_cache_v1"
ASSET_GRASP_CACHE_MANIFEST_SCHEMA_VERSION = "asset_grasp_cache_manifest_v1"
DEFAULT_GRASP_ANNOTATIONS_DIRNAME = "grasp_annotations"
DEFAULT_CACHE_ENTRIES_DIRNAME = "annotations"
DEFAULT_CACHE_ARTIFACTS_DIRNAME = "artifacts"
DEFAULT_CACHE_MANIFEST_FILENAME = "asset_grasp_cache_manifest.json"

_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True)
class ManifestObjectRecord:
    asset_id: str
    prim_path: str
    output_name: str | None
    class_name: str | None
    caption: str | None
    source: str | None
    glb_path: Path
    usd_path: Path | None


@dataclass(frozen=True)
class ManifestAssetRecord:
    asset_id: str
    output_name: str | None
    class_name: str | None
    caption: str | None
    source: str | None
    glb_path: Path
    usd_path: Path | None
    prim_paths: tuple[str, ...]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _relative_to(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _sanitize_asset_id(value: str) -> str:
    normalized = _NON_ALNUM_RE.sub("_", value.strip())
    normalized = normalized.strip("_")
    return normalized or "asset"


def _resolve_manifest_asset_path(manifest_path: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _asset_id_for_object(
    *,
    prim_path: str,
    output_name: str | None,
    glb_path: Path,
) -> str:
    if output_name:
        return _sanitize_asset_id(output_name)
    rel = str(glb_path.with_suffix("")).replace("/", "_")
    if rel:
        return _sanitize_asset_id(rel)
    return _sanitize_asset_id(Path(prim_path).name)


def default_grasp_annotation_root(manifest_path: str | Path) -> Path:
    manifest = Path(manifest_path).resolve()
    return manifest.parent.parent / DEFAULT_GRASP_ANNOTATIONS_DIRNAME


def cache_entries_dir(output_root: str | Path) -> Path:
    return Path(output_root).resolve() / DEFAULT_CACHE_ENTRIES_DIRNAME


def cache_artifacts_dir(output_root: str | Path) -> Path:
    return Path(output_root).resolve() / DEFAULT_CACHE_ARTIFACTS_DIRNAME


def cache_entry_path(output_root: str | Path, asset_id: str) -> Path:
    return cache_entries_dir(output_root) / f"{asset_id}.json"


def load_manifest_object_records(manifest_path: str | Path) -> list[ManifestObjectRecord]:
    manifest = Path(manifest_path).resolve()
    payload = _load_json(manifest)
    objects = payload.get("objects", {})
    if not isinstance(objects, dict):
        raise ValueError(f"Manifest objects must be a JSON object: {manifest}")

    records: list[ManifestObjectRecord] = []
    for prim_path in sorted(objects):
        meta = objects.get(prim_path)
        if not isinstance(meta, dict):
            continue
        if not isinstance(prim_path, str) or not prim_path:
            continue
        glb_path = _resolve_manifest_asset_path(manifest, meta.get("glb_path"))
        if glb_path is None:
            continue
        output_name = str(meta.get("output_name")).strip() if meta.get("output_name") is not None else None
        asset_id = _asset_id_for_object(
            prim_path=prim_path,
            output_name=output_name,
            glb_path=glb_path,
        )
        records.append(
            ManifestObjectRecord(
                asset_id=asset_id,
                prim_path=prim_path,
                output_name=output_name,
                class_name=str(meta.get("class")).strip() if meta.get("class") else None,
                caption=str(meta.get("caption")).strip() if meta.get("caption") else None,
                source=str(meta.get("source")).strip() if meta.get("source") else None,
                glb_path=glb_path,
                usd_path=_resolve_manifest_asset_path(manifest, meta.get("usd_path")),
            )
        )
    return records


def find_manifest_object_record(manifest_path: str | Path, prim_path: str) -> ManifestObjectRecord | None:
    needle = str(prim_path).strip()
    if not needle:
        return None
    for record in load_manifest_object_records(manifest_path):
        if record.prim_path == needle:
            return record
    return None


def cache_entry_path_for_prim(
    manifest_path: str | Path,
    prim_path: str,
    *,
    output_root: str | Path | None = None,
) -> Path | None:
    record = find_manifest_object_record(manifest_path, prim_path)
    if record is None:
        return None
    resolved_output_root = Path(output_root).resolve() if output_root is not None else default_grasp_annotation_root(manifest_path)
    return cache_entry_path(resolved_output_root, record.asset_id)


def unique_manifest_assets(
    manifest_path: str | Path,
    *,
    selected_prim_paths: Iterable[str] | None = None,
) -> list[ManifestAssetRecord]:
    selected = {str(value).strip() for value in (selected_prim_paths or []) if str(value).strip()}
    grouped: dict[str, list[ManifestObjectRecord]] = {}
    for record in load_manifest_object_records(manifest_path):
        if selected and record.prim_path not in selected:
            continue
        grouped.setdefault(str(record.glb_path), []).append(record)

    assets: list[ManifestAssetRecord] = []
    for key in sorted(grouped):
        items = grouped[key]
        first = items[0]
        assets.append(
            ManifestAssetRecord(
                asset_id=first.asset_id,
                output_name=first.output_name,
                class_name=first.class_name,
                caption=first.caption,
                source=first.source,
                glb_path=first.glb_path,
                usd_path=first.usd_path,
                prim_paths=tuple(item.prim_path for item in items),
            )
        )
    return assets


def load_cached_grasp_annotation(cache_path: str | Path) -> dict[str, Any]:
    payload = _load_json(Path(cache_path).resolve())
    schema_version = str(payload.get("schema_version") or "").strip()
    if schema_version not in {ASSET_GRASP_CACHE_SCHEMA_VERSION, "grasp_primitives_v1"}:
        raise ValueError(f"Unsupported grasp annotation schema '{schema_version}' in {cache_path}")
    return payload


def _build_cache_entry(
    asset: ManifestAssetRecord,
    *,
    annotation_payload: dict[str, Any],
    annotation_path: Path,
    object_dir: Path,
    output_root: Path,
    annotator_status: str,
) -> dict[str, Any]:
    return {
        "schema_version": ASSET_GRASP_CACHE_SCHEMA_VERSION,
        "annotation_schema_version": annotation_payload.get("schema_version"),
        "asset_id": asset.asset_id,
        "source": asset.source,
        "output_name": asset.output_name,
        "class": asset.class_name,
        "caption": asset.caption,
        "prim_paths": list(asset.prim_paths),
        "source_glb": str(asset.glb_path),
        "source_usd": str(asset.usd_path) if asset.usd_path else None,
        "category": annotation_payload.get("category"),
        "object_name": annotation_payload.get("object_name"),
        "grasp_primitives": annotation_payload.get("grasp_primitives", []),
        "artifacts": annotation_payload.get("artifacts", {}),
        "annotator_status": annotator_status,
        "annotator_object_dir": _relative_to(output_root, object_dir),
        "annotator_annotation_path": _relative_to(output_root, annotation_path),
    }


def build_asset_grasp_cache(
    manifest_path: str | Path,
    *,
    output_root: str | Path | None = None,
    pipeline_config: PipelineConfig | None = None,
    selected_prim_paths: Iterable[str] | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    resolved_output_root = Path(output_root).resolve() if output_root is not None else default_grasp_annotation_root(manifest)
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    cache_entries_dir(resolved_output_root).mkdir(parents=True, exist_ok=True)
    cache_artifacts_dir(resolved_output_root).mkdir(parents=True, exist_ok=True)

    pipeline = pipeline_config or PipelineConfig()
    asset_records = unique_manifest_assets(manifest, selected_prim_paths=selected_prim_paths)
    objects_payload: list[dict[str, Any]] = []

    for asset in asset_records:
        stable_cache_path = cache_entry_path(resolved_output_root, asset.asset_id)
        if resume and stable_cache_path.exists():
            cached_payload = load_cached_grasp_annotation(stable_cache_path)
            objects_payload.append(
                {
                    "asset_id": asset.asset_id,
                    "status": "skipped_existing",
                    "cache_path": _relative_to(resolved_output_root, stable_cache_path),
                    "category": cached_payload.get("category"),
                    "primitive_count": len(cached_payload.get("grasp_primitives", [])),
                    "prim_paths": list(asset.prim_paths),
                    "source_glb": _relative_to(resolved_output_root, asset.glb_path),
                }
            )
            continue

        result = annotate_single_object(
            asset.glb_path,
            cache_artifacts_dir(resolved_output_root),
            pipeline,
            resume=resume,
        )
        annotation_path = Path(result["annotation"]).resolve()
        annotation_payload = _load_json(annotation_path)
        cache_entry = _build_cache_entry(
            asset,
            annotation_payload=annotation_payload,
            annotation_path=annotation_path,
            object_dir=Path(result["object_dir"]).resolve(),
            output_root=resolved_output_root,
            annotator_status=str(result.get("status", "ok")),
        )
        _write_json(stable_cache_path, cache_entry)
        objects_payload.append(
            {
                "asset_id": asset.asset_id,
                "status": str(result.get("status", "ok")),
                "cache_path": _relative_to(resolved_output_root, stable_cache_path),
                "category": cache_entry.get("category"),
                "primitive_count": len(cache_entry.get("grasp_primitives", [])),
                "prim_paths": list(asset.prim_paths),
                "source_glb": _relative_to(resolved_output_root, asset.glb_path),
            }
        )

    manifest_payload = {
        "schema_version": ASSET_GRASP_CACHE_MANIFEST_SCHEMA_VERSION,
        "manifest_path": str(manifest),
        "output_root": str(resolved_output_root),
        "cache_entries_dir": str(cache_entries_dir(resolved_output_root)),
        "cache_artifacts_dir": str(cache_artifacts_dir(resolved_output_root)),
        "pipeline_config": {
            **asdict(pipeline),
            "graspnet_repo": str(pipeline.graspnet_repo),
            "graspnet_checkpoint": str(pipeline.graspnet_checkpoint),
        },
        "objects": objects_payload,
    }
    _write_json(resolved_output_root / DEFAULT_CACHE_MANIFEST_FILENAME, manifest_payload)
    return manifest_payload


def ensure_asset_grasp_cache_for_prim(
    manifest_path: str | Path,
    prim_path: str,
    *,
    output_root: str | Path | None = None,
    pipeline_config: PipelineConfig | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    record = find_manifest_object_record(manifest_path, prim_path)
    if record is None:
        raise ValueError(f"Target prim '{prim_path}' was not found in manifest {manifest_path}.")
    resolved_output_root = Path(output_root).resolve() if output_root is not None else default_grasp_annotation_root(manifest_path)
    stable_cache_path = cache_entry_path(resolved_output_root, record.asset_id)
    cache_existed = stable_cache_path.exists()
    manifest_payload = build_asset_grasp_cache(
        manifest_path,
        output_root=resolved_output_root,
        pipeline_config=pipeline_config,
        selected_prim_paths=[record.prim_path],
        resume=resume,
    )
    return {
        "prim_path": record.prim_path,
        "asset_id": record.asset_id,
        "cache_path": str(stable_cache_path),
        "cache_existed": bool(cache_existed),
        "cache_exists_now": stable_cache_path.exists(),
        "manifest": manifest_payload,
    }
