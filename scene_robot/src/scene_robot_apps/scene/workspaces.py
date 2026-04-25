from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from app.backend.services.robot_placement import ROBOT_PROFILES


@dataclass(frozen=True)
class WorkspaceBox:
    name: str
    center_xy: tuple[float, float]
    size_xy: tuple[float, float]
    color: str
    alpha: float

    def bounds_xy(self) -> tuple[float, float, float, float]:
        half_x = float(self.size_xy[0]) * 0.5
        half_y = float(self.size_xy[1]) * 0.5
        return (
            float(self.center_xy[0]) - half_x,
            float(self.center_xy[0]) + half_x,
            float(self.center_xy[1]) - half_y,
            float(self.center_xy[1]) + half_y,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "center_xy": [float(self.center_xy[0]), float(self.center_xy[1])],
            "size_xy": [float(self.size_xy[0]), float(self.size_xy[1])],
            "color": self.color,
            "alpha": float(self.alpha),
        }


@dataclass(frozen=True)
class RobotWorkspaceSpec:
    robot: str
    title: str
    base_radius: float
    body_half_extents_xy: tuple[float, float]
    working_area: WorkspaceBox

    def to_payload(self) -> dict[str, Any]:
        return {
            "robot": self.robot,
            "title": self.title,
            "base_radius": float(self.base_radius),
            "body_half_extents_xy": [float(self.body_half_extents_xy[0]), float(self.body_half_extents_xy[1])],
            "working_area": self.working_area.to_payload(),
        }


def default_robot_workspace_specs() -> dict[str, RobotWorkspaceSpec]:
    return {
        "kinova": RobotWorkspaceSpec(
            robot="kinova",
            title="Kinova",
            base_radius=float(ROBOT_PROFILES["kinova"]["base_radius"]),
            body_half_extents_xy=(0.16, 0.18),
            working_area=WorkspaceBox(
                name="working_area",
                center_xy=(0.56, 0.0),
                size_xy=(0.52, 0.64),
                color=str(ROBOT_PROFILES["kinova"]["color"]),
                alpha=0.24,
            ),
        ),
        "agibot": RobotWorkspaceSpec(
            robot="agibot",
            title="Agibot G1 Omnipicker",
            base_radius=float(ROBOT_PROFILES["agibot"]["base_radius"]),
            body_half_extents_xy=(0.22, 0.20),
            working_area=WorkspaceBox(
                name="working_area",
                center_xy=(0.53, 0.0),
                size_xy=(0.50, 0.92),
                color=str(ROBOT_PROFILES["agibot"]["color"]),
                alpha=0.24,
            ),
        ),
        "r1lite": RobotWorkspaceSpec(
            robot="r1lite",
            title="R1Lite",
            base_radius=float(ROBOT_PROFILES["r1lite"]["base_radius"]),
            body_half_extents_xy=(0.20, 0.19),
            working_area=WorkspaceBox(
                name="working_area",
                center_xy=(0.48, 0.0),
                size_xy=(0.48, 0.80),
                color=str(ROBOT_PROFILES["r1lite"]["color"]),
                alpha=0.24,
            ),
        ),
    }


def save_robot_workspace_overview(
    *,
    output_image_path: str | Path,
    output_json_path: str | Path | None = None,
    robots: tuple[str, ...] | None = None,
    dpi: int = 180,
) -> dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

    specs = default_robot_workspace_specs()
    selected = list(robots or tuple(specs.keys()))
    for robot in selected:
        if robot not in specs:
            raise ValueError(f"Unsupported robot '{robot}'. Choose from: {sorted(specs)}")

    fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 5), constrained_layout=True)
    if len(selected) == 1:
        axes = [axes]

    x_limits = (-0.20, 1.00)
    y_limits = (-0.75, 0.75)

    for axis, robot in zip(axes, selected):
        spec = specs[robot]
        body = Rectangle(
            (-spec.body_half_extents_xy[0], -spec.body_half_extents_xy[1]),
            spec.body_half_extents_xy[0] * 2.0,
            spec.body_half_extents_xy[1] * 2.0,
            linewidth=1.4,
            edgecolor="#202020",
            facecolor="#f2f2f2",
            zorder=2,
        )
        base = Circle((0.0, 0.0), radius=spec.base_radius, linewidth=1.6, edgecolor="#202020", facecolor="none", zorder=3)
        axis.add_patch(body)
        axis.add_patch(base)
        axis.add_patch(
            FancyArrowPatch(
                (0.0, 0.0),
                (0.22, 0.0),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.8,
                color="#202020",
                zorder=4,
            )
        )

        working_area = spec.working_area
        xmin, xmax, ymin, ymax = working_area.bounds_xy()
        axis.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=2.0,
                edgecolor=working_area.color,
                facecolor=working_area.color,
                alpha=working_area.alpha,
                zorder=1,
            )
        )
        axis.text(
            float(working_area.center_xy[0]),
            float(working_area.center_xy[1]),
            working_area.name,
            fontsize=9,
            ha="center",
            va="center",
            color=working_area.color,
        )

        axis.axhline(0.0, color="#d0d0d0", linewidth=0.8, zorder=0)
        axis.axvline(0.0, color="#d0d0d0", linewidth=0.8, zorder=0)
        axis.set_title(spec.title)
        axis.set_xlabel("x in robot base frame (m)")
        axis.set_ylabel("y in robot base frame (m)")
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.20)

    output_image = Path(output_image_path).resolve()
    output_image.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_image, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)

    payload = {
        "robots": [specs[robot].to_payload() for robot in selected],
        "output_image_path": str(output_image),
    }
    if output_json_path is not None:
        output_json = Path(output_json_path).resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["output_json_path"] = str(output_json)
    return payload


def _hex_to_rgb(color_hex: str) -> tuple[float, float, float]:
    value = str(color_hex).strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB color, got: {color_hex}")
    return (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
    )


def _rotate_xy(x: float, y: float, yaw_deg: float) -> tuple[float, float]:
    import math

    yaw_rad = math.radians(float(yaw_deg))
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    return (
        (cos_yaw * float(x)) - (sin_yaw * float(y)),
        (sin_yaw * float(x)) + (cos_yaw * float(y)),
    )


def project_workspace_box_to_support(
    *,
    robot: str,
    base_pose: tuple[float, float, float, float],
    support_center_xy: tuple[float, float],
    support_half_extents_xy: tuple[float, float],
    support_yaw_deg: float,
) -> dict[str, Any] | None:
    specs = default_robot_workspace_specs()
    if robot not in specs:
        raise ValueError(f"Unsupported robot '{robot}'. Choose from: {sorted(specs)}")

    working_area = specs[robot].working_area
    half_x = float(working_area.size_xy[0]) * 0.5
    half_y = float(working_area.size_xy[1]) * 0.5
    center_local_x = float(working_area.center_xy[0])
    center_local_y = float(working_area.center_xy[1])
    base_x, base_y, _base_z, base_yaw_deg = (float(value) for value in base_pose)

    corner_offsets = (
        (-half_x, -half_y),
        (-half_x, half_y),
        (half_x, half_y),
        (half_x, -half_y),
    )
    support_local_corners: list[tuple[float, float]] = []
    for offset_x, offset_y in corner_offsets:
        corner_local_x = center_local_x + offset_x
        corner_local_y = center_local_y + offset_y
        world_dx, world_dy = _rotate_xy(corner_local_x, corner_local_y, base_yaw_deg)
        corner_world_x = base_x + world_dx
        corner_world_y = base_y + world_dy
        rel_x = corner_world_x - float(support_center_xy[0])
        rel_y = corner_world_y - float(support_center_xy[1])
        support_local_x, support_local_y = _rotate_xy(rel_x, rel_y, -float(support_yaw_deg))
        support_local_corners.append((support_local_x, support_local_y))

    projected_min_x = max(min(point[0] for point in support_local_corners), -float(support_half_extents_xy[0]))
    projected_max_x = min(max(point[0] for point in support_local_corners), float(support_half_extents_xy[0]))
    projected_min_y = max(min(point[1] for point in support_local_corners), -float(support_half_extents_xy[1]))
    projected_max_y = min(max(point[1] for point in support_local_corners), float(support_half_extents_xy[1]))

    if projected_max_x <= projected_min_x or projected_max_y <= projected_min_y:
        return None

    projected_center_local_x = (projected_min_x + projected_max_x) * 0.5
    projected_center_local_y = (projected_min_y + projected_max_y) * 0.5
    projected_center_world_dx, projected_center_world_dy = _rotate_xy(
        projected_center_local_x,
        projected_center_local_y,
        support_yaw_deg,
    )

    return {
        "name": working_area.name,
        "center_xy": (
            float(support_center_xy[0]) + projected_center_world_dx,
            float(support_center_xy[1]) + projected_center_world_dy,
        ),
        "size_xy": (
            float(projected_max_x - projected_min_x),
            float(projected_max_y - projected_min_y),
        ),
        "yaw_deg": float(support_yaw_deg),
        "color": working_area.color,
        "alpha": working_area.alpha,
    }


def add_projected_workspace_visual_to_stage(
    stage,
    *,
    robot: str,
    root_prim_path: str,
    projected_workspace: dict[str, Any],
    top_z: float,
    thickness: float = 0.015,
) -> dict[str, Any]:
    from pxr import Gf, UsdGeom

    root = UsdGeom.Xform.Define(stage, root_prim_path).GetPrim()
    rgb = _hex_to_rgb(str(projected_workspace["color"]))
    box_path = f"{root_prim_path.rstrip('/')}/{projected_workspace['name']}"
    box = UsdGeom.Cube.Define(stage, box_path)
    box.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(box.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(
        Gf.Vec3d(
            float(projected_workspace["center_xy"][0]),
            float(projected_workspace["center_xy"][1]),
            float(top_z) + (float(thickness) * 0.5),
        )
    )
    xform.AddRotateZOp().Set(float(projected_workspace["yaw_deg"]))
    xform.AddScaleOp().Set(
        Gf.Vec3f(
            max(1e-4, float(projected_workspace["size_xy"][0])),
            max(1e-4, float(projected_workspace["size_xy"][1])),
            max(1e-4, float(thickness)),
        )
    )
    box.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([rgb])
    box.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set([float(projected_workspace["alpha"])])
    UsdGeom.Imageable(root).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
    return {
        "workspace_root_path": root_prim_path,
        "box_path": box_path,
    }


def add_robot_workspace_visuals_to_stage(
    stage,
    *,
    robot: str,
    robot_prim_path: str,
    plane_z: float = 0.02,
) -> dict[str, Any]:
    from pxr import Gf, UsdGeom

    specs = default_robot_workspace_specs()
    if robot not in specs:
        raise ValueError(f"Unsupported robot '{robot}'. Choose from: {sorted(specs)}")

    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        raise ValueError(f"Robot prim does not exist: {robot_prim_path}")

    spec = specs[robot]
    root_path = f"{robot_prim_path.rstrip('/')}/WorkspaceVisuals"
    visuals_root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    created_paths: list[str] = []

    working_area = spec.working_area
    rgb = _hex_to_rgb(working_area.color)
    working_area_path = f"{root_path}/{working_area.name}"
    working_area_cube = UsdGeom.Cube.Define(stage, working_area_path)
    working_area_cube.CreateSizeAttr(1.0)
    working_area_xform = UsdGeom.Xformable(working_area_cube.GetPrim())
    working_area_xform.ClearXformOpOrder()
    working_area_xform.AddTranslateOp().Set(
        Gf.Vec3d(float(working_area.center_xy[0]), float(working_area.center_xy[1]), float(plane_z * 0.5))
    )
    working_area_xform.AddScaleOp().Set(
        Gf.Vec3f(
            max(1e-4, float(working_area.size_xy[0])),
            max(1e-4, float(working_area.size_xy[1])),
            max(1e-4, float(plane_z)),
        )
    )
    working_area_cube.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([rgb])
    working_area_cube.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set([float(working_area.alpha)])
    created_paths.append(working_area_path)

    body_path = f"{root_path}/RobotBodyFootprint"
    body = UsdGeom.Cube.Define(stage, body_path)
    body.CreateSizeAttr(1.0)
    body_xform = UsdGeom.Xformable(body.GetPrim())
    body_xform.ClearXformOpOrder()
    body_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(plane_z * 0.5)))
    body_xform.AddScaleOp().Set(
        Gf.Vec3f(
            max(1e-4, float(spec.body_half_extents_xy[0]) * 2.0),
            max(1e-4, float(spec.body_half_extents_xy[1]) * 2.0),
            max(1e-4, float(plane_z)),
        )
    )
    body.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([(0.95, 0.95, 0.95)])
    body.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set([0.25])
    created_paths.append(body_path)

    arrow_path = f"{root_path}/ForwardArrow"
    arrow = UsdGeom.Mesh.Define(stage, arrow_path)
    arrow_points = [
        Gf.Vec3f(0.02, 0.0, float(plane_z + 0.008)),
        Gf.Vec3f(0.18, 0.05, float(plane_z + 0.008)),
        Gf.Vec3f(0.18, -0.05, float(plane_z + 0.008)),
    ]
    arrow.CreatePointsAttr(arrow_points)
    arrow.CreateFaceVertexCountsAttr([3])
    arrow.CreateFaceVertexIndicesAttr([0, 1, 2])
    arrow.CreateDoubleSidedAttr(True)
    arrow.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([(0.1, 0.1, 0.1)])
    arrow.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set([0.85])
    created_paths.append(arrow_path)

    visibility_attr = UsdGeom.Imageable(visuals_root).CreateVisibilityAttr()
    visibility_attr.Set(UsdGeom.Tokens.inherited)

    return {
        "robot": robot,
        "robot_prim_path": robot_prim_path,
        "workspace_root_path": root_path,
        "created_paths": created_paths,
    }
