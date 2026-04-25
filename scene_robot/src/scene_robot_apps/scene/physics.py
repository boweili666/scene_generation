"""Physics setup, BBox queries, and reset helpers for scene-based pipelines.

Originally bundled inside `scene_mouse_collect.py`; pulled out so the
mouse-collect orchestrator stays focused on scene assembly + episode loop.
This module concentrates everything that pokes at PhysX / USD physics
state:

* Collision approximation aliases + per-prim convex-decomposition settings.
* World-AABB queries via `UsdGeom.BBoxCache` (`_compute_world_prim_*_z`).
* PhysX rigid-body tensor views for live world-pose reads on GPU scenes
  (`_get_rigid_body_view` / `_read_rigid_body_world_position_z`).
* Robot root alignment to the floor (`_align_robot_root_to_floor`).
* Generated-scene physics rebinding: stripping nested PhysicsScenes,
  applying CollisionAPI / RigidBodyAPI / MassAPI to imported props, and
  re-grounding visually-floating objects (`_rebind_generated_scene_physics`,
  `_realign_generated_scene_floor_objects`,
  `_prepare_scene_usd_without_physics_scenes`).
* Episode boundary helpers (`_reset_scene_to_plan`, `_settle_dynamic_scene`).

`scene_mouse_collect.py` re-exports the public names so external imports
that already say `from ..pipelines.scene_mouse_collect import _reset_scene_to_plan`
keep working unchanged.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from isaaclab.scene import InteractiveScene

from app.backend.services.robot_placement import RobotPlacementPlan

from ..control.robot_controller import RobotController


# =============================================================================
# Collision approximation
# =============================================================================
_DYNAMIC_MESH_COLLISION_APPROX = "convexHull"
_COLLISION_APPROX_ALIASES = {
    "default": None,
    "triangle_mesh": "none",
    "convex_hull": "convexHull",
    "convex_decomposition": "convexDecomposition",
    "mesh_simplification": "meshSimplification",
    "bounding_cube": "boundingCube",
    "bounding_sphere": "boundingSphere",
    "sdf": "sdf",
    "sphere_fill": "sphereFill",
}


@dataclass(frozen=True)
class ConvexDecompositionSettings:
    voxel_resolution: int
    max_convex_hulls: int
    error_percentage: float
    shrink_wrap: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "voxel_resolution": self.voxel_resolution,
            "max_convex_hulls": self.max_convex_hulls,
            "error_percentage": self.error_percentage,
            "shrink_wrap": self.shrink_wrap,
        }


def _resolve_collision_approx(option: str | None) -> str | None:
    key = str(option or "default").strip().lower()
    if key not in _COLLISION_APPROX_ALIASES:
        raise ValueError(
            f"Unsupported collision approximation '{option}'. "
            f"Expected one of: {', '.join(sorted(_COLLISION_APPROX_ALIASES))}."
        )
    return _COLLISION_APPROX_ALIASES[key]


def _apply_convex_decomposition_settings(prim, settings: ConvexDecompositionSettings) -> None:
    from pxr import PhysxSchema

    api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
    api.CreateVoxelResolutionAttr().Set(int(settings.voxel_resolution))
    api.CreateMaxConvexHullsAttr().Set(int(settings.max_convex_hulls))
    api.CreateErrorPercentageAttr().Set(float(settings.error_percentage))
    api.CreateShrinkWrapAttr().Set(bool(settings.shrink_wrap))


# =============================================================================
# Robot root pose helpers (orientation + alignment)
# =============================================================================
def _yaw_quat_wxyz(yaw_deg: float) -> tuple[float, float, float, float]:
    half = math.radians(yaw_deg) * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _apply_root_pose(controller: RobotController, plan: RobotPlacementPlan, base_z: float) -> None:
    root_state = controller.robot.data.default_root_state.clone()
    root_state[:, 0] = float(plan.base_pose[0])
    root_state[:, 1] = float(plan.base_pose[1])
    root_state[:, 2] = float(base_z)
    quat = torch.tensor(_yaw_quat_wxyz(plan.base_pose[3]), device=controller.sim.device, dtype=torch.float32)
    root_state[:, 3:7] = quat.unsqueeze(0).repeat(controller.scene.num_envs, 1)
    controller.robot.write_root_pose_to_sim(root_state[:, :7])
    controller.robot.write_root_velocity_to_sim(root_state[:, 7:])
    controller.robot.reset()


# =============================================================================
# USD BBox queries (static scenery only — see _read_rigid_body_world_position_z
# for live dynamic body reads)
# =============================================================================
def _compute_world_prim_min_z(stage, prim_path: str) -> float | None:
    # For static scenery (support/table), the classic `UsdGeom.BBoxCache` is
    # fine: nothing moves it so the authored xformOps reflect the truth. This
    # helper is no longer used for the dynamic grasp target — for that, see
    # `_read_rigid_body_world_position_z` below, which goes through the
    # physx backend via `SingleXFormPrim.get_world_pose()`.
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["render", "default"], useExtentsHint=False)
    aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if aligned.IsEmpty():
        return None
    return float(aligned.GetMin()[2])


def _compute_world_prim_max_z(stage, prim_path: str) -> float | None:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["render", "default"], useExtentsHint=False)
    aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if aligned.IsEmpty():
        return None
    return float(aligned.GetMax()[2])


# =============================================================================
# PhysX rigid-body tensor views (live world-pose reads on GPU scenes)
# =============================================================================
_RIGID_BODY_VIEW_CACHE: dict[str, object] = {}


def _get_rigid_body_view(prim_path: str):
    # Create (once) and cache a physx-backed rigid-body view for the given
    # prim path. Isaac Lab's own `RigidObject` uses exactly this mechanism
    # (`SimulationManager.get_physics_sim_view().create_rigid_body_view(...)`)
    # to read live rigid body state on GPU physics scenes. We can't reuse
    # isaaclab's `RigidObject` because our target lives inside an untracked
    # `AssetBaseCfg` USD reference, but we can create a parallel view here.
    cached = _RIGID_BODY_VIEW_CACHE.get(prim_path)
    if cached is not None:
        return cached
    from isaacsim.core.simulation_manager import SimulationManager

    physics_sim_view = SimulationManager.get_physics_sim_view()
    if physics_sim_view is None:
        return None
    view = physics_sim_view.create_rigid_body_view(prim_path)
    if view is None or getattr(view, "_backend", None) is None:
        return None
    _RIGID_BODY_VIEW_CACHE[prim_path] = view
    return view


def _read_rigid_body_world_position_z(prim_path: str) -> float | None:
    # Live read of a rigid body's world Z from the physx tensor backend.
    # `get_transforms()` returns `(N, 7)` with `[px, py, pz, qx, qy, qz, qw]`
    # per instance, reflecting the true physics state — unlike classic USD
    # `xformOp` reads, which lag behind on GPU physics scenes.
    view = _get_rigid_body_view(prim_path)
    if view is None:
        return None
    transforms = view.get_transforms()
    if transforms is None or transforms.shape[0] == 0:
        return None
    return float(transforms[0, 2].item())


def _clear_rigid_body_view_cache() -> None:
    _RIGID_BODY_VIEW_CACHE.clear()


# =============================================================================
# Floor alignment for robot + generated-scene props
# =============================================================================
def _align_robot_root_to_floor(
    scene: InteractiveScene,
    controller: RobotController,
    *,
    floor_z: float = 0.0,
    max_passes: int = 3,
) -> float:
    robot_prim_path = f"{scene.env_prim_paths[0]}/Robot"
    applied_root_z = float(controller.robot.data.root_pose_w[0, 2].item())

    for _ in range(max_passes):
        scene.write_data_to_sim()
        controller.sim.step()
        scene.update(controller.sim.get_physics_dt())
        min_z = _compute_world_prim_min_z(scene.stage, robot_prim_path)
        if min_z is None:
            return applied_root_z
        delta_z = float(floor_z) - float(min_z)
        if abs(delta_z) <= 1.0e-3:
            return applied_root_z
        root_pose = controller.robot.data.root_pose_w.clone()
        root_pose[:, 2] += delta_z
        root_velocity = controller.robot.data.root_vel_w.clone()
        controller.robot.write_root_pose_to_sim(root_pose)
        controller.robot.write_root_velocity_to_sim(root_velocity)
        controller.robot.reset()
        applied_root_z = float(root_pose[0, 2].item())

    return applied_root_z


# =============================================================================
# Generated-scene physics rebinding
# =============================================================================
def _assign_simulation_owner(api_schema, physics_scene_path: str | None) -> None:
    if not physics_scene_path:
        return
    rel = api_schema.CreateSimulationOwnerRel()
    rel.SetTargets([physics_scene_path])


def _set_disable_gravity_attr(prim, enabled: bool) -> None:
    from pxr import Sdf

    attr = prim.GetAttribute("physxRigidBody:disableGravity")
    if not attr.IsValid():
        attr = prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool, False)
    attr.Set(not bool(enabled))


def _iter_collision_prims(root_prim):
    from pxr import Usd, UsdGeom

    if not root_prim or not root_prim.IsValid():
        return []
    return [prim for prim in Usd.PrimRange(root_prim) if prim.IsA(UsdGeom.Gprim)]


def _iter_visual_collision_prims(root_prim):
    from pxr import UsdGeom

    visual_prims = []
    for prim in _iter_collision_prims(root_prim):
        purpose = UsdGeom.Imageable(prim).GetPurposeAttr().Get()
        if purpose == UsdGeom.Tokens.guide:
            continue
        visual_prims.append(prim)
    return visual_prims


def _supported_scene_object_paths(scene_graph: dict) -> set[str]:
    supported_paths: set[str] = set()
    for edge in (scene_graph.get("edges") or {}).get("obj-obj", []):
        if not isinstance(edge, dict):
            continue
        relation_tokens = {token.strip().lower() for token in str(edge.get("relation") or "").split(",") if token.strip()}
        source = edge.get("source")
        target = edge.get("target")
        if "supported by" in relation_tokens and isinstance(source, str):
            supported_paths.add(source)
        if "supports" in relation_tokens and isinstance(target, str):
            supported_paths.add(target)
    return supported_paths


def _compute_bounds_for_prims(stage, prims) -> object | None:
    from pxr import Gf, Usd, UsdGeom

    bounds_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True)
    combined = Gf.Range3d()
    found = False
    for prim in prims:
        if not prim or not prim.IsValid():
            continue
        rng = bounds_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        if not found:
            combined = Gf.Range3d(rng.GetMin(), rng.GetMax())
            found = True
            continue
        combined.UnionWith(rng)
    return combined if found and not combined.IsEmpty() else None


def _realign_generated_scene_floor_objects(
    stage,
    scene_root_path: str,
    scene_graph: dict,
    *,
    floor_z: float = 0.0,
) -> dict[str, int]:
    from pxr import Gf, UsdGeom

    supported_paths = _supported_scene_object_paths(scene_graph)
    grounded_roots = 0
    visual_roots = 0
    for prim_path in (scene_graph.get("obj") or {}):
        if prim_path in supported_paths:
            continue
        prim_name = Path(str(prim_path)).name
        live_prim = stage.GetPrimAtPath(f"{scene_root_path}/{prim_name}")
        if not live_prim.IsValid():
            continue

        visual_prims = _iter_visual_collision_prims(live_prim)
        if not visual_prims:
            continue
        visual_roots += 1

        rng = _compute_bounds_for_prims(stage, visual_prims)
        if rng is None:
            continue
        delta_z = float(floor_z) - float(rng.GetMin()[2])
        if abs(delta_z) <= 1.0e-4:
            continue

        xform = UsdGeom.Xformable(live_prim)
        ops = xform.GetOrderedXformOps()
        if not ops:
            continue

        mat = Gf.Matrix4d(ops[0].Get())
        translate = mat.ExtractTranslation()
        mat.SetTranslateOnly(translate + Gf.Vec3d(0.0, 0.0, delta_z))
        ops[0].Set(mat)
        grounded_roots += 1

    return {"visual_roots": visual_roots, "grounded_roots": grounded_roots}


def _find_live_physics_scene_path(stage, scene_root_path: str) -> str | None:
    preferred = None
    fallback = None
    scene_root_prefix = scene_root_path.rstrip("/") + "/"
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsScene":
            continue
        prim_path = str(prim.GetPath())
        if fallback is None:
            fallback = prim_path
        if not prim_path.startswith(scene_root_prefix):
            preferred = prim_path
            break
    return preferred or fallback


def _remove_nested_physics_scenes(stage, scene_root_path: str) -> list[str]:
    removed_paths: list[str] = []
    scene_root_prefix = scene_root_path.rstrip("/") + "/"
    to_remove: list[str] = []
    for prim in stage.Traverse():
        prim_name = prim.GetName().lower()
        if prim.GetTypeName() != "PhysicsScene" and prim_name != "physicsscene":
            continue
        prim_path = str(prim.GetPath())
        if prim_path.startswith(scene_root_prefix):
            to_remove.append(prim_path)
    for prim_path in sorted(to_remove, key=len, reverse=True):
        stage.RemovePrim(prim_path)
        removed_paths.append(prim_path)
    return removed_paths


def _prepare_scene_usd_without_physics_scenes(scene_usd_path: str) -> str:
    from pxr import Usd

    source_path = Path(scene_usd_path).resolve()
    stage = Usd.Stage.Open(str(source_path))
    if stage is None:
        return str(source_path)

    to_remove: list[str] = []
    for prim in stage.Traverse():
        prim_name = prim.GetName().lower()
        if prim.GetTypeName() == "PhysicsScene" or prim_name == "physicsscene":
            to_remove.append(str(prim.GetPath()))

    if not to_remove:
        return str(source_path)

    for prim_path in sorted(to_remove, key=len, reverse=True):
        stage.RemovePrim(prim_path)

    sanitized_path = Path(tempfile.gettempdir()) / f"{source_path.stem}.scene_collect_nophysics.usd"
    stage.GetRootLayer().Export(str(sanitized_path))
    return str(sanitized_path)


def _rebind_generated_scene_physics(
    stage,
    scene_root_path: str,
    scene_graph: dict,
    *,
    object_collision_approx: str | None = None,
    target_prim: str | None = None,
    target_collision_approx: str | None = None,
    convex_decomposition_settings: ConvexDecompositionSettings | None = None,
) -> dict[str, object]:
    from pxr import UsdGeom, UsdPhysics

    physics_scene_path = _find_live_physics_scene_path(stage, scene_root_path)
    if physics_scene_path is None:
        return {
            "physics_scene_path": None,
            "dynamic_roots": 0,
            "room_colliders": 0,
            "object_colliders": 0,
            "object_collision_approx": None,
            "target_collision_approx": None,
            "convex_decomposition_prim_count": 0,
            "convex_decomposition_settings": None,
        }

    room_colliders = 0
    room_root = stage.GetPrimAtPath(f"{scene_root_path}/GeneratedRoom/Room")
    for prim in _iter_collision_prims(room_root):
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        collision_api.CreateCollisionEnabledAttr(True)
        _assign_simulation_owner(collision_api, physics_scene_path)
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.none)
        room_colliders += 1

    dynamic_roots = 0
    object_colliders = 0
    applied_object_collision_approx = None
    applied_target_collision_approx = None
    convex_decomposition_prim_count = 0
    for prim_path in (scene_graph.get("obj") or {}):
        prim_name = Path(str(prim_path)).name
        live_prim = stage.GetPrimAtPath(f"{scene_root_path}/{prim_name}")
        if not live_prim.IsValid():
            continue

        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(live_prim)
        rigid_body_api.CreateRigidBodyEnabledAttr(True)
        rigid_body_api.CreateKinematicEnabledAttr(False)
        rigid_body_api.CreateStartsAsleepAttr(False)
        _assign_simulation_owner(rigid_body_api, physics_scene_path)
        _set_disable_gravity_attr(live_prim, True)

        mass_api = UsdPhysics.MassAPI.Apply(live_prim)
        mass_api.CreateMassAttr(1.0)

        visual_collision_prims = _iter_visual_collision_prims(live_prim)
        active_collision_prims = visual_collision_prims

        for collision_prim in active_collision_prims:
            collision_api = UsdPhysics.CollisionAPI.Apply(collision_prim)
            collision_api.CreateCollisionEnabledAttr(True)
            _assign_simulation_owner(collision_api, physics_scene_path)
            if collision_prim.IsA(UsdGeom.Mesh):
                is_target_prim = str(prim_path) == str(target_prim)
                approximation = (
                    object_collision_approx
                    if object_collision_approx is not None
                    else (
                        target_collision_approx
                        if target_collision_approx is not None and is_target_prim
                        else _DYNAMIC_MESH_COLLISION_APPROX
                    )
                )
                UsdPhysics.MeshCollisionAPI.Apply(collision_prim).CreateApproximationAttr(
                    approximation
                )
                if approximation == "convexDecomposition" and convex_decomposition_settings is not None:
                    _apply_convex_decomposition_settings(collision_prim, convex_decomposition_settings)
                    convex_decomposition_prim_count += 1
                if object_collision_approx is not None:
                    applied_object_collision_approx = approximation
                elif target_collision_approx is not None and is_target_prim:
                    applied_target_collision_approx = approximation
            object_colliders += 1
        dynamic_roots += 1

    return {
        "physics_scene_path": physics_scene_path,
        "dynamic_roots": dynamic_roots,
        "room_colliders": room_colliders,
        "object_colliders": object_colliders,
        "object_collision_approx": applied_object_collision_approx,
        "target_collision_approx": applied_target_collision_approx,
        "convex_decomposition_prim_count": convex_decomposition_prim_count,
        "convex_decomposition_settings": (
            None if convex_decomposition_prim_count == 0 or convex_decomposition_settings is None
            else convex_decomposition_settings.as_dict()
        ),
    }


# =============================================================================
# Episode boundary helpers
# =============================================================================
def _settle_dynamic_scene(
    scene: InteractiveScene,
    controller: RobotController,
    sync_cameras: Callable[[], None] | None,
    *,
    settle_steps: int = 90,
) -> None:
    for _ in range(max(0, settle_steps)):
        scene.write_data_to_sim()
        controller.sim.step()
        scene.update(controller.sim.get_physics_dt())
        if sync_cameras is not None:
            sync_cameras()


def _reset_scene_to_plan(
    scene: InteractiveScene,
    controller: RobotController,
    plan: RobotPlacementPlan,
    base_z: float,
    sync_cameras: Callable[[], None] | None,
    *,
    settle_steps: int = 90,
) -> float:
    controller.reset()
    scene.reset()
    _apply_root_pose(controller, plan, base_z)
    if hasattr(controller, "_apply_reset_joint_state"):
        controller._apply_reset_joint_state()
    scene.write_data_to_sim()
    controller.sim.step()
    scene.update(controller.sim.get_physics_dt())
    aligned_base_z = _align_robot_root_to_floor(scene, controller, floor_z=0.0)
    _settle_dynamic_scene(scene, controller, sync_cameras, settle_steps=settle_steps)
    aligned_base_z = _align_robot_root_to_floor(scene, controller, floor_z=0.0)
    if sync_cameras is not None:
        sync_cameras()
    return aligned_base_z
