from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from app.backend.config.settings import PROJECT_ROOT
from app.backend.services.robot_placement import DEFAULT_OUTPUT_DIR, RobotPlacementPlan, plan_to_payload


DEFAULT_SCENE_USD_PATH = PROJECT_ROOT / "runtime" / "scene_service" / "usd" / "scene_latest.usd"
DEFAULT_ROBOT_SCENE_PATH = DEFAULT_OUTPUT_DIR / "scene_with_robot.usd"
DEFAULT_ROBOT_SCENE_META_PATH = DEFAULT_OUTPUT_DIR / "robot_scene_result.json"

ROBOT_ASSET_CANDIDATES: Dict[str, tuple[Path, ...]] = {
    "kinova": (
        PROJECT_ROOT
        / "scene_robot"
        / "src"
        / "scene_robot_assets"
        / "GEN3-7DOF-VISION_ROBOTIQ-2F85_COMBINED"
        / "GEN3-7DOF-VISION_ROBOTIQ-2F85_COMBINED.usd",
    ),
    "agibot": (
        PROJECT_ROOT
        / "scene_robot"
        / "src"
        / "scene_robot_assets"
        / "agibot"
        / "G1_omnipicker"
        / "robot.usda",
        PROJECT_ROOT
        / "scene_robot"
        / "src"
        / "scene_robot_assets"
        / "agibot"
        / "G1_omnipicker"
        / "robot.usd",
    ),
    "r1lite": (
        PROJECT_ROOT / "scene_robot" / "src" / "scene_robot_assets" / "r1lite" / "robot" / "robot.usd",
    ),
}


@dataclass(frozen=True)
class RobotSceneResult:
    robot: str
    robot_asset_path: str
    robot_prim_path: str
    robot_floor_offset_z: float
    scene_input_usd: str
    scene_output_usd: str
    base_pose: tuple[float, float, float, float]


def _import_pxr():
    from pxr import Gf, Usd, UsdGeom

    return Gf, Usd, UsdGeom


def resolve_robot_asset_path(robot: str) -> Path:
    candidates = ROBOT_ASSET_CANDIDATES.get(robot)
    if candidates is None:
        raise ValueError(f"Unsupported robot '{robot}'. Choose from: {sorted(ROBOT_ASSET_CANDIDATES)}")
    for path in candidates:
        if path.exists():
            return path
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Robot USD not found for '{robot}'. Tried: {joined}")


def compute_robot_floor_offset_z(robot: str) -> float:
    _, Usd, UsdGeom = _import_pxr()

    asset_path = resolve_robot_asset_path(robot)
    stage = Usd.Stage.Open(str(asset_path))
    if stage is None:
        raise ValueError(f"Failed to open robot USD: {asset_path}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise ValueError(f"Robot USD has no valid default prim: {asset_path}")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"], useExtentsHint=True)
    bbox = bbox_cache.ComputeLocalBound(default_prim)
    aligned = bbox.ComputeAlignedRange()
    if aligned.IsEmpty():
        return 0.0
    min_z = float(aligned.GetMin()[2])
    return max(0.0, -min_z)


def embed_robot_in_scene_usd(
    scene_usd_path: str | Path,
    plan: RobotPlacementPlan,
    *,
    robot: str | None = None,
    output_usd_path: str | Path = DEFAULT_ROBOT_SCENE_PATH,
    robot_prim_path: str = "/World/RobotPlacement",
    robot_asset_prim_name: str = "RobotAsset",
    floor_z: float = 0.0,
    overwrite_existing: bool = True,
) -> RobotSceneResult:
    Gf, Usd, UsdGeom = _import_pxr()

    robot_name = robot or plan.robot
    robot_asset_path = resolve_robot_asset_path(robot_name)
    floor_offset_z = compute_robot_floor_offset_z(robot_name)

    source_path = Path(scene_usd_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Scene USD does not exist: {source_path}")

    stage = Usd.Stage.Open(str(source_path))
    if stage is None:
        raise ValueError(f"Failed to open scene USD: {source_path}")

    world = stage.GetPrimAtPath("/World")
    if not world or not world.IsValid():
        world = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(world)

    if overwrite_existing and stage.GetPrimAtPath(robot_prim_path).IsValid():
        stage.RemovePrim(robot_prim_path)

    wrapper = UsdGeom.Xform.Define(stage, robot_prim_path)
    wrapper_prim = wrapper.GetPrim()
    asset_prim_path = f"{robot_prim_path.rstrip('/')}/{robot_asset_prim_name}"
    asset_xform = UsdGeom.Xform.Define(stage, asset_prim_path)
    asset_prim = asset_xform.GetPrim()
    asset_prim.GetReferences().AddReference(str(robot_asset_path))

    translate = (
        float(plan.base_pose[0]),
        float(plan.base_pose[1]),
        float(floor_z) + float(floor_offset_z),
    )
    rotate = (0.0, 0.0, float(plan.base_pose[3]))

    api = UsdGeom.XformCommonAPI(wrapper_prim)
    api.SetTranslate(Gf.Vec3d(*translate))
    api.SetRotate(Gf.Vec3f(*rotate), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    output_path = Path(output_usd_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(output_path))

    return RobotSceneResult(
        robot=robot_name,
        robot_asset_path=str(robot_asset_path),
        robot_prim_path=asset_prim_path,
        robot_floor_offset_z=floor_offset_z,
        scene_input_usd=str(source_path),
        scene_output_usd=str(output_path),
        base_pose=(
            float(plan.base_pose[0]),
            float(plan.base_pose[1]),
            float(floor_z) + float(floor_offset_z),
            float(plan.base_pose[3]),
        ),
    )


def save_robot_scene_result(
    result: RobotSceneResult,
    plan: RobotPlacementPlan,
    *,
    output_meta_path: str | Path = DEFAULT_ROBOT_SCENE_META_PATH,
) -> Path:
    output = Path(output_meta_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "scene": {
            "robot": result.robot,
            "robot_asset_path": result.robot_asset_path,
            "robot_prim_path": result.robot_prim_path,
            "robot_floor_offset_z": result.robot_floor_offset_z,
            "scene_input_usd": result.scene_input_usd,
            "scene_output_usd": result.scene_output_usd,
            "base_pose": list(result.base_pose),
        },
        "placement_plan": plan_to_payload(plan),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
