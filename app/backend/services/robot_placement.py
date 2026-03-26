from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from app.backend.config.settings import DEFAULT_PLACEMENTS_PATH, RUNTIME_DIR, SCENE_GRAPH_PATH


DEFAULT_OUTPUT_DIR = RUNTIME_DIR / "robot_placement"
TARGET_DISTANCE_TIE_TOLERANCE = 0.10

SUPPORT_EDGE_SOURCE = "supported by"
SUPPORT_EDGE_TARGET = "supports"
DEFAULT_OBJECT_RADIUS = 0.2

TARGET_CLASS_PRIORITY = [
    "mug",
    "cup",
    "bottle",
    "can",
    "book",
    "laptop",
    "phone",
    "glasses",
    "case",
    "lamp",
    "plant",
]

SURFACE_SIZE_PRIORS: Dict[str, Tuple[float, float]] = {
    "table": (0.95, 0.70),
    "desk": (1.20, 0.70),
    "workbench": (1.40, 0.80),
    "counter": (1.50, 0.70),
    "nightstand": (0.52, 0.42),
    "cabinet": (0.90, 0.50),
    "shelf": (0.90, 0.35),
    "dresser": (1.00, 0.45),
}

OBJECT_RADIUS_PRIORS: Dict[str, float] = {
    "bed": 0.80,
    "book": 0.12,
    "bottle": 0.08,
    "cabinet": 0.55,
    "case": 0.11,
    "chair": 0.38,
    "cup": 0.08,
    "desk": 0.70,
    "dresser": 0.60,
    "glasses": 0.08,
    "lamp": 0.18,
    "laptop": 0.19,
    "mug": 0.09,
    "nightstand": 0.30,
    "phone": 0.08,
    "plant": 0.18,
    "shelf": 0.45,
    "sofa": 0.95,
    "table": 0.55,
    "workbench": 0.80,
}

ROBOT_PROFILES: Dict[str, Dict[str, float | str]] = {
    "kinova": {
        "base_radius": 0.28,
        "anchor_gap": 0.10,
        "wall_weight": 0.35,
        "target_weight": 0.15,
        "color": "#2d5bff",
    },
    "agibot": {
        "base_radius": 0.35,
        "anchor_gap": 0.10,
        "wall_weight": 0.35,
        "target_weight": 0.12,
        "color": "#0d9d7a",
    },
    "r1lite": {
        "base_radius": 0.33,
        "anchor_gap": 0.10,
        "wall_weight": 0.35,
        "target_weight": 0.12,
        "color": "#7a49ff",
    },
}


@dataclass(frozen=True)
class CandidatePose:
    side: str
    base_xy: Tuple[float, float]
    yaw_deg: float
    target_distance: float
    overlap_free: bool
    overlap_margin: float
    obstacle_clearance: float
    wall_clearance: float
    target_alignment: float
    tie_break_score: float


@dataclass(frozen=True)
class RobotPlacementPlan:
    robot: str
    target_prim: str
    support_prim: str
    support_center_xy: Tuple[float, float]
    support_z: float
    support_yaw_deg: float
    support_half_extents_xy: Tuple[float, float]
    support_shape: str
    chosen_side: str
    base_pose: Tuple[float, float, float, float]
    room_bounds: Tuple[float, float, float, float] | None
    supported_objects: List[str]
    floor_obstacles: List[str]
    candidates: List[CandidatePose]


PlacementPayload = Dict[str, Any]
PlacementMap = Dict[str, PlacementPayload]


def _read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize_aabb(raw: Any) -> Dict[str, List[float]] | None:
    if not isinstance(raw, dict):
        return None
    normalized: Dict[str, List[float]] = {}
    for key in ("min", "max", "center", "size"):
        value = raw.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            normalized[key] = [float(value[0]), float(value[1]), float(value[2])]
    return normalized or None


def _normalize_placements(payload: Dict[str, Any]) -> PlacementMap:
    normalized: PlacementMap = {}
    for prim, raw in (payload or {}).items():
        if not isinstance(prim, str):
            continue
        if isinstance(raw, list) and len(raw) >= 3:
            item: PlacementPayload = {"x": float(raw[0]), "y": float(raw[1]), "z": float(raw[2])}
            if len(raw) >= 4:
                item["yaw"] = float(raw[3])
            normalized[prim] = item
        elif isinstance(raw, dict):
            if not all(key in raw for key in ("x", "y", "z")):
                continue
            item = {"x": float(raw["x"]), "y": float(raw["y"]), "z": float(raw["z"])}
            if "yaw" in raw and raw["yaw"] is not None:
                item["yaw"] = float(raw["yaw"])
            aabb = _normalize_aabb(raw.get("aabb"))
            if aabb is not None:
                item["aabb"] = aabb
            normalized[prim] = item
    return normalized


def load_scene_state(
    scene_graph_path: str | Path | None = SCENE_GRAPH_PATH,
    placements_path: str | Path | None = DEFAULT_PLACEMENTS_PATH,
) -> tuple[Dict[str, Any], PlacementMap]:
    scene_graph_resolved = SCENE_GRAPH_PATH if scene_graph_path is None else scene_graph_path
    placements_resolved = DEFAULT_PLACEMENTS_PATH if placements_path is None else placements_path
    return _read_json(scene_graph_resolved), _normalize_placements(_read_json(placements_resolved))


def _placement_xy(payload: PlacementPayload) -> Tuple[float, float]:
    if isinstance(payload, (list, tuple)) and len(payload) >= 2:
        return (float(payload[0]), float(payload[1]))
    return (float(payload["x"]), float(payload["y"]))


def _placement_xyz(payload: PlacementPayload) -> Tuple[float, float, float]:
    if isinstance(payload, (list, tuple)) and len(payload) >= 3:
        return (float(payload[0]), float(payload[1]), float(payload[2]))
    return (float(payload["x"]), float(payload["y"]), float(payload["z"]))


def _placement_yaw_deg(payload: PlacementPayload) -> float:
    if isinstance(payload, (list, tuple)):
        return float(payload[3]) if len(payload) >= 4 else 0.0
    value = payload.get("yaw")
    return float(value) if value is not None else 0.0


def _placement_aabb(payload: PlacementPayload) -> Dict[str, List[float]] | None:
    if not isinstance(payload, dict):
        return None
    raw = payload.get("aabb")
    if isinstance(raw, dict):
        return raw
    return None


def _placement_center_xy(payload: PlacementPayload) -> Tuple[float, float]:
    aabb = _placement_aabb(payload)
    if aabb and "center" in aabb:
        center = aabb["center"]
        return (float(center[0]), float(center[1]))
    return _placement_xy(payload)


def _placement_center_z(payload: PlacementPayload) -> float:
    aabb = _placement_aabb(payload)
    if aabb and "center" in aabb:
        return float(aabb["center"][2])
    return _placement_xyz(payload)[2]


def _placement_half_size_xy(payload: PlacementPayload) -> Tuple[float, float] | None:
    aabb = _placement_aabb(payload)
    if not aabb or "size" not in aabb:
        return None
    size = aabb["size"]
    return (max(0.0, float(size[0]) * 0.5), max(0.0, float(size[1]) * 0.5))


def _placement_planar_radius(payload: PlacementPayload) -> float | None:
    half_size_xy = _placement_half_size_xy(payload)
    if half_size_xy is None:
        return None
    return math.hypot(half_size_xy[0], half_size_xy[1])


def _relation_tokens(relation: str) -> set[str]:
    return {token.strip().lower() for token in str(relation or "").split(",") if token.strip()}


def _class_name(scene_graph: Dict[str, Any], prim: str) -> str:
    meta = scene_graph.get("obj", {}).get(prim, {})
    return str(meta.get("class") or meta.get("class_name") or "").strip().lower()


def _caption(scene_graph: Dict[str, Any], prim: str) -> str:
    meta = scene_graph.get("obj", {}).get(prim, {})
    return str(meta.get("caption") or "").strip().lower()


def _object_radius(scene_graph: Dict[str, Any], prim: str, placements: PlacementMap | None = None) -> float:
    if placements is not None and prim in placements:
        aabb_radius = _placement_planar_radius(placements[prim])
        if aabb_radius is not None:
            return aabb_radius
    cls = _class_name(scene_graph, prim)
    return float(OBJECT_RADIUS_PRIORS.get(cls, DEFAULT_OBJECT_RADIUS))


def _is_surface(scene_graph: Dict[str, Any], prim: str) -> bool:
    cls = _class_name(scene_graph, prim)
    return cls in SURFACE_SIZE_PRIORS


def _match_query(query: str, scene_graph: Dict[str, Any], placements: PlacementMap) -> List[str]:
    needle = query.strip().lower()
    if not needle:
        return []
    matches: List[str] = []
    for prim in scene_graph.get("obj", {}):
        if prim not in placements:
            continue
        cls = _class_name(scene_graph, prim)
        caption = _caption(scene_graph, prim)
        if needle == prim.lower() or needle == cls or needle in prim.lower() or needle in caption:
            matches.append(prim)
    return sorted(set(matches))


def _target_priority(scene_graph: Dict[str, Any], prim: str) -> tuple[int, str]:
    cls = _class_name(scene_graph, prim)
    try:
        idx = TARGET_CLASS_PRIORITY.index(cls)
    except ValueError:
        idx = len(TARGET_CLASS_PRIORITY)
    return (idx, prim)


def list_supported_objects(scene_graph: Dict[str, Any], support_prim: str) -> List[str]:
    supported: set[str] = set()
    for edge in scene_graph.get("edges", {}).get("obj-obj", []):
        if not isinstance(edge, dict):
            continue
        rels = _relation_tokens(edge.get("relation", ""))
        source = edge.get("source")
        target = edge.get("target")
        if source == support_prim and SUPPORT_EDGE_TARGET in rels and isinstance(target, str):
            supported.add(target)
        if target == support_prim and SUPPORT_EDGE_SOURCE in rels and isinstance(source, str):
            supported.add(source)
    return sorted(supported)


def find_support_prim(scene_graph: Dict[str, Any], target_prim: str) -> str:
    for edge in scene_graph.get("edges", {}).get("obj-obj", []):
        if not isinstance(edge, dict):
            continue
        rels = _relation_tokens(edge.get("relation", ""))
        source = edge.get("source")
        target = edge.get("target")
        if source == target_prim and SUPPORT_EDGE_SOURCE in rels and isinstance(target, str):
            return target
        if target == target_prim and SUPPORT_EDGE_TARGET in rels and isinstance(source, str):
            return source
    raise ValueError(f"Unable to find a support object for target '{target_prim}'.")


def resolve_target_prim(
    scene_graph: Dict[str, Any],
    placements: PlacementMap,
    target: str | None = None,
) -> str:
    if target:
        matches = _match_query(target, scene_graph, placements)
        if not matches:
            raise ValueError(f"Unable to resolve target '{target}'.")
        matches.sort(key=lambda prim: (_target_priority(scene_graph, prim), prim))
        return matches[0]

    candidates: List[str] = []
    for prim in scene_graph.get("obj", {}):
        if prim not in placements:
            continue
        try:
            support_prim = find_support_prim(scene_graph, prim)
        except ValueError:
            continue
        if _is_surface(scene_graph, support_prim):
            candidates.append(prim)

    if not candidates:
        raise ValueError("No supported tabletop object was found in the current scene.")
    candidates.sort(key=lambda prim: (_target_priority(scene_graph, prim), prim))
    return candidates[0]


def _surface_prior(scene_graph: Dict[str, Any], support_prim: str) -> Tuple[float, float]:
    cls = _class_name(scene_graph, support_prim)
    caption = _caption(scene_graph, support_prim)
    width, depth = SURFACE_SIZE_PRIORS.get(cls, SURFACE_SIZE_PRIORS["table"])
    if "round" in caption:
        diameter = max(width, depth)
        return (diameter, diameter)
    return (width, depth)


def _rotate_xy(x: float, y: float, yaw_deg: float) -> Tuple[float, float]:
    yaw = math.radians(yaw_deg)
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (c * x - s * y, s * x + c * y)


def _world_to_local(dx: float, dy: float, yaw_deg: float) -> Tuple[float, float]:
    return _rotate_xy(dx, dy, -yaw_deg)


def _estimate_surface_geometry(
    scene_graph: Dict[str, Any],
    placements: PlacementMap,
    support_prim: str,
) -> tuple[Tuple[float, float], str]:
    base_width, base_depth = _surface_prior(scene_graph, support_prim)
    caption = _caption(scene_graph, support_prim)
    support = placements[support_prim]
    center_x, center_y = _placement_center_xy(support)
    yaw_deg = _placement_yaw_deg(support)
    supported_objects = list_supported_objects(scene_graph, support_prim)

    aabb_half_size_xy = _placement_half_size_xy(support)
    if aabb_half_size_xy is not None:
        if "round" in caption:
            radius = max(aabb_half_size_xy)
            return ((radius, radius), "circle")
        c = abs(math.cos(math.radians(yaw_deg)))
        s = abs(math.sin(math.radians(yaw_deg)))
        det = (c * c) - (s * s)
        if abs(det) > 1e-4:
            local_half_x = ((c * aabb_half_size_xy[0]) - (s * aabb_half_size_xy[1])) / det
            local_half_y = ((c * aabb_half_size_xy[1]) - (s * aabb_half_size_xy[0])) / det
            if local_half_x > 1e-4 and local_half_y > 1e-4:
                return ((abs(local_half_x), abs(local_half_y)), "rect")
        return (aabb_half_size_xy, "rect")

    if "round" in caption:
        radius = max(base_width, base_depth) * 0.5
        for prim in supported_objects:
            if prim not in placements:
                continue
            px, py = _placement_center_xy(placements[prim])
            dx, dy = px - center_x, py - center_y
            radius = max(radius, math.hypot(dx, dy) + _object_radius(scene_graph, prim, placements) + 0.08)
        return ((radius, radius), "circle")

    half_x = base_width * 0.5
    half_y = base_depth * 0.5
    for prim in supported_objects:
        if prim not in placements:
            continue
        px, py = _placement_center_xy(placements[prim])
        local_x, local_y = _world_to_local(px - center_x, py - center_y, yaw_deg)
        radius = _object_radius(scene_graph, prim, placements) + 0.08
        half_x = max(half_x, abs(local_x) + radius)
        half_y = max(half_y, abs(local_y) + radius)
    return ((half_x, half_y), "rect")


def _room_bounds_from_scene(scene_graph: Dict[str, Any]) -> Tuple[float, float, float, float] | None:
    dims = scene_graph.get("scene", {}).get("dimensions", {})
    if not isinstance(dims, dict):
        return None
    length = dims.get("length")
    width = dims.get("width")
    try:
        half_x = float(length) * 0.5
        half_y = float(width) * 0.5
    except (TypeError, ValueError):
        return None
    return (-half_x, half_x, -half_y, half_y)


def _candidate_side_vectors() -> List[tuple[str, Tuple[float, float]]]:
    return [
        ("right", (0.0, 1.0)),
        ("left", (0.0, -1.0)),
        ("front", (1.0, 0.0)),
        ("back", (-1.0, 0.0)),
    ]


def _world_rect_extent_from_local_half_extents(
    half_extents_xy: Tuple[float, float],
    yaw_deg: float,
    normal_xy: Tuple[float, float],
) -> float:
    half_x, half_y = half_extents_xy
    yaw = math.radians(yaw_deg)
    c = math.cos(yaw)
    s = math.sin(yaw)
    local_x_axis = (c, s)
    local_y_axis = (-s, c)
    return (
        abs((normal_xy[0] * local_x_axis[0]) + (normal_xy[1] * local_x_axis[1])) * half_x
        + abs((normal_xy[0] * local_y_axis[0]) + (normal_xy[1] * local_y_axis[1])) * half_y
    )


def _iter_floor_obstacles(
    scene_graph: Dict[str, Any],
    placements: PlacementMap,
    *,
    support_prim: str,
    supported_objects: Iterable[str],
) -> Iterable[str]:
    excluded = set(supported_objects)
    excluded.add(support_prim)
    for prim in placements:
        if prim not in excluded:
            yield prim


def _wall_clearance(room_bounds: Tuple[float, float, float, float] | None, base_xy: Tuple[float, float], radius: float) -> float:
    if room_bounds is None:
        return 2.0
    xmin, xmax, ymin, ymax = room_bounds
    x, y = base_xy
    return min(x - xmin, xmax - x, y - ymin, ymax - y) - radius


def _obstacle_clearance(
    scene_graph: Dict[str, Any],
    placements: PlacementMap,
    base_xy: Tuple[float, float],
    robot_radius: float,
    obstacles: Iterable[str],
) -> float:
    clearance = math.inf
    for prim in obstacles:
        aabb_bounds_xy = _placement_aabb_bounds_xy(placements[prim])
        if aabb_bounds_xy is not None:
            signed_distance = _signed_distance_to_aabb_xy(base_xy, aabb_bounds_xy)
            margin = signed_distance - (robot_radius + 0.08)
        else:
            px, py = _placement_center_xy(placements[prim])
            distance = math.hypot(base_xy[0] - px, base_xy[1] - py)
            margin = distance - (robot_radius + _object_radius(scene_graph, prim, placements) + 0.08)
        clearance = min(clearance, margin)
    return 2.0 if math.isinf(clearance) else clearance


def _target_alignment(
    support_center_xy: Tuple[float, float],
    target_xy: Tuple[float, float],
    normal_xy: Tuple[float, float],
) -> float:
    dx = target_xy[0] - support_center_xy[0]
    dy = target_xy[1] - support_center_xy[1]
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        return 0.0
    return ((dx / norm) * normal_xy[0]) + ((dy / norm) * normal_xy[1])


def _target_distance(base_xy: Tuple[float, float], target_xy: Tuple[float, float]) -> float:
    return math.hypot(base_xy[0] - target_xy[0], base_xy[1] - target_xy[1])


def _yaw_to_target(source_xy: Tuple[float, float], target_xy: Tuple[float, float]) -> float:
    return math.degrees(math.atan2(target_xy[1] - source_xy[1], target_xy[0] - source_xy[0]))


def plan_robot_base_pose(
    scene_graph: Dict[str, Any],
    placements: PlacementMap,
    *,
    target_prim: str | None = None,
    support_prim: str | None = None,
    robot: str = "agibot",
) -> RobotPlacementPlan:
    profile = ROBOT_PROFILES.get(robot)
    if profile is None:
        raise ValueError(f"Unsupported robot '{robot}'. Choose from: {sorted(ROBOT_PROFILES)}")

    resolved_target = resolve_target_prim(scene_graph, placements, target_prim)
    resolved_support = support_prim or find_support_prim(scene_graph, resolved_target)
    if resolved_support not in placements:
        raise ValueError(f"Support object '{resolved_support}' has no placement.")

    support = placements[resolved_support]
    support_center_xy = _placement_center_xy(support)
    support_z = _placement_center_z(support)
    support_yaw_deg = _placement_yaw_deg(support)
    target_xy = _placement_center_xy(placements[resolved_target])
    support_half_extents_xy, support_shape = _estimate_surface_geometry(scene_graph, placements, resolved_support)
    supported_objects = list_supported_objects(scene_graph, resolved_support)
    floor_obstacles = sorted(_iter_floor_obstacles(scene_graph, placements, support_prim=resolved_support, supported_objects=supported_objects))
    room_bounds = _room_bounds_from_scene(scene_graph)

    robot_radius = float(profile["base_radius"])
    anchor_gap = float(profile.get("anchor_gap", profile.get("stand_off", 0.10)))
    wall_weight = float(profile["wall_weight"])
    target_weight = float(profile["target_weight"])

    candidates: List[CandidatePose] = []
    for side, normal_xy in _candidate_side_vectors():
        if support_shape == "circle":
            support_extent = max(support_half_extents_xy)
        else:
            support_aabb_half_size_xy = _placement_half_size_xy(support)
            if support_aabb_half_size_xy is not None:
                support_extent = (
                    abs(normal_xy[0]) * support_aabb_half_size_xy[0]
                    + abs(normal_xy[1]) * support_aabb_half_size_xy[1]
                )
            else:
                support_extent = _world_rect_extent_from_local_half_extents(
                    support_half_extents_xy,
                    support_yaw_deg,
                    normal_xy,
                )
        base_distance = support_extent + robot_radius + anchor_gap
        base_xy = (
            support_center_xy[0] + normal_xy[0] * base_distance,
            support_center_xy[1] + normal_xy[1] * base_distance,
        )
        obstacle_clearance = _obstacle_clearance(scene_graph, placements, base_xy, robot_radius, floor_obstacles)
        wall_clearance = _wall_clearance(room_bounds, base_xy, robot_radius)
        alignment = _target_alignment(support_center_xy, target_xy, normal_xy)
        target_distance = _target_distance(base_xy, target_xy)
        overlap_margin = min(obstacle_clearance, wall_clearance)
        overlap_free = obstacle_clearance >= 0.0 and wall_clearance >= 0.0
        tie_break_score = obstacle_clearance + (wall_weight * wall_clearance) + (target_weight * alignment)
        candidates.append(
            CandidatePose(
                side=side,
                base_xy=base_xy,
                yaw_deg=_yaw_to_target(base_xy, support_center_xy),
                target_distance=target_distance,
                overlap_free=overlap_free,
                overlap_margin=overlap_margin,
                obstacle_clearance=obstacle_clearance,
                wall_clearance=wall_clearance,
                target_alignment=alignment,
                tie_break_score=tie_break_score,
            )
        )

    overlap_free_candidates = [item for item in candidates if item.overlap_free]
    if overlap_free_candidates:
        viable = overlap_free_candidates
    else:
        best_overlap_margin = max(item.overlap_margin for item in candidates)
        viable = [item for item in candidates if item.overlap_margin >= best_overlap_margin - 1e-6]

    best_target_distance = min(item.target_distance for item in viable)
    shortlisted = [
        item for item in viable if item.target_distance <= best_target_distance + TARGET_DISTANCE_TIE_TOLERANCE
    ]
    shortlisted.sort(
        key=lambda item: (
            item.tie_break_score,
            item.obstacle_clearance,
            item.wall_clearance,
            -item.target_distance,
        ),
        reverse=True,
    )
    chosen = shortlisted[0]
    return RobotPlacementPlan(
        robot=robot,
        target_prim=resolved_target,
        support_prim=resolved_support,
        support_center_xy=support_center_xy,
        support_z=support_z,
        support_yaw_deg=support_yaw_deg,
        support_half_extents_xy=support_half_extents_xy,
        support_shape=support_shape,
        chosen_side=chosen.side,
        base_pose=(chosen.base_xy[0], chosen.base_xy[1], 0.0, chosen.yaw_deg),
        room_bounds=room_bounds,
        supported_objects=supported_objects,
        floor_obstacles=floor_obstacles,
        candidates=candidates,
    )


def _polygon_points(center_xy: Tuple[float, float], half_extents_xy: Tuple[float, float], yaw_deg: float) -> List[Tuple[float, float]]:
    cx, cy = center_xy
    hx, hy = half_extents_xy
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    points: List[Tuple[float, float]] = []
    for local_x, local_y in corners:
        dx, dy = _rotate_xy(local_x, local_y, yaw_deg)
        points.append((cx + dx, cy + dy))
    return points


def _world_bounds(scene_graph: Dict[str, Any], placements: PlacementMap, plan: RobotPlacementPlan) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    if plan.room_bounds is not None:
        xmin, xmax, ymin, ymax = plan.room_bounds
        xs.extend([xmin, xmax])
        ys.extend([ymin, ymax])
    for payload in placements.values():
        aabb = _placement_aabb(payload)
        if aabb and "min" in aabb and "max" in aabb:
            xs.extend([float(aabb["min"][0]), float(aabb["max"][0])])
            ys.extend([float(aabb["min"][1]), float(aabb["max"][1])])
        else:
            x, y = _placement_center_xy(payload)
            xs.append(x)
            ys.append(y)
    xs.append(plan.base_pose[0])
    ys.append(plan.base_pose[1])
    padding = 0.9
    return (min(xs) - padding, max(xs) + padding, min(ys) - padding, max(ys) + padding)


def _placement_aabb_bounds_xy(payload: PlacementPayload) -> Tuple[float, float, float, float] | None:
    aabb = _placement_aabb(payload)
    if not aabb or "min" not in aabb or "max" not in aabb:
        return None
    return (
        float(aabb["min"][0]),
        float(aabb["max"][0]),
        float(aabb["min"][1]),
        float(aabb["max"][1]),
    )


def _signed_distance_to_aabb_xy(point_xy: Tuple[float, float], bounds_xy: Tuple[float, float, float, float]) -> float:
    x, y = point_xy
    min_x, max_x, min_y, max_y = bounds_xy
    dx = max(min_x - x, 0.0, x - max_x)
    dy = max(min_y - y, 0.0, y - max_y)
    if dx > 0.0 or dy > 0.0:
        return math.hypot(dx, dy)
    return -min(x - min_x, max_x - x, y - min_y, max_y - y)


def _project(bounds: Tuple[float, float, float, float], width: int, height: int, x: float, y: float) -> Tuple[float, float, float]:
    xmin, xmax, ymin, ymax = bounds
    scale = min(width / max(xmax - xmin, 1e-6), height / max(ymax - ymin, 1e-6))
    px = (x - xmin) * scale
    py = height - ((y - ymin) * scale)
    return (px, py, scale)


def render_plan_svg(
    scene_graph: Dict[str, Any],
    placements: PlacementMap,
    plan: RobotPlacementPlan,
    output_path: str | Path,
    *,
    width: int = 1080,
    height: int = 820,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    bounds = _world_bounds(scene_graph, placements, plan)
    support_color = "#b8c7ff"
    target_color = "#ff6b6b"
    robot_color = str(ROBOT_PROFILES[plan.robot]["color"])
    other_color = "#7f8da8"
    supported_color = "#1b8b6b"

    def project_xy(x: float, y: float) -> Tuple[float, float]:
        px, py, _ = _project(bounds, width, height, x, y)
        return px, py

    def project_aabb_rect(aabb_bounds_xy: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        min_x, max_x, min_y, max_y = aabb_bounds_xy
        x0, y0 = project_xy(min_x, max_y)
        x1, y1 = project_xy(max_x, min_y)
        return (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))

    _, _, scale = _project(bounds, width, height, 0.0, 0.0)
    content: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#12305b"/>',
        "</marker>",
        "</defs>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#f8fbff"/>',
    ]

    if plan.room_bounds is not None:
        xmin, xmax, ymin, ymax = plan.room_bounds
        x0, y0 = project_xy(xmin, ymax)
        x1, y1 = project_xy(xmax, ymin)
        content.append(
            f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{(x1 - x0):.1f}" height="{(y1 - y0):.1f}" '
            'fill="none" stroke="#d7e3f7" stroke-width="3" rx="24"/>'
        )

    support_payload = placements.get(plan.support_prim)
    support_aabb_bounds_xy = None if support_payload is None else _placement_aabb_bounds_xy(support_payload)
    if support_aabb_bounds_xy is not None:
        rect_x, rect_y, rect_w, rect_h = project_aabb_rect(support_aabb_bounds_xy)
        content.append(
            f'<rect x="{rect_x:.1f}" y="{rect_y:.1f}" width="{rect_w:.1f}" height="{rect_h:.1f}" '
            f'fill="{support_color}" fill-opacity="0.55" stroke="#5c78d6" stroke-width="3" rx="14"/>'
        )
    elif plan.support_shape == "circle":
        radius_world = max(plan.support_half_extents_xy)
        cx, cy = project_xy(*plan.support_center_xy)
        content.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius_world * scale:.1f}" fill="{support_color}" '
            'fill-opacity="0.55" stroke="#5c78d6" stroke-width="3"/>'
        )
    else:
        points = _polygon_points(plan.support_center_xy, plan.support_half_extents_xy, plan.support_yaw_deg)
        svg_points = " ".join(f"{project_xy(x, y)[0]:.1f},{project_xy(x, y)[1]:.1f}" for x, y in points)
        content.append(
            f'<polygon points="{svg_points}" fill="{support_color}" fill-opacity="0.55" '
            'stroke="#5c78d6" stroke-width="3"/>'
        )

    for candidate in plan.candidates:
        cx, cy = project_xy(*candidate.base_xy)
        content.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{float(ROBOT_PROFILES[plan.robot]["base_radius"]) * scale:.1f}" '
            'fill="#ffffff" fill-opacity="0.75" stroke="#8aa0c6" stroke-width="2" stroke-dasharray="10 7"/>'
        )
        status = "clear" if candidate.overlap_free else "overlap"
        content.append(
            f'<text x="{cx + 14:.1f}" y="{cy - 14:.1f}" font-family="monospace" font-size="16" fill="#4d617f">'
            f'{candidate.side} {status} {candidate.target_distance:.2f}m</text>'
        )

    for prim, payload in placements.items():
        x, y = _placement_center_xy(payload)
        px, py = project_xy(x, y)
        radius = max(_object_radius(scene_graph, prim, placements) * scale, 10.0)
        fill = other_color
        stroke = "#ffffff"
        opacity = 0.75
        if prim == plan.support_prim:
            continue
        if prim == plan.target_prim:
            fill = target_color
            stroke = "#8b1e1e"
            opacity = 0.95
        elif prim in plan.supported_objects:
            fill = supported_color
            stroke = "#0d5f49"
            opacity = 0.90
        aabb_bounds_xy = _placement_aabb_bounds_xy(payload)
        if aabb_bounds_xy is not None:
            rect_x, rect_y, rect_w, rect_h = project_aabb_rect(aabb_bounds_xy)
            content.append(
                f'<rect x="{rect_x:.1f}" y="{rect_y:.1f}" width="{rect_w:.1f}" height="{rect_h:.1f}" '
                f'fill="{fill}" fill-opacity="{opacity:.2f}" stroke="{stroke}" stroke-width="2" rx="8"/>'
            )
            label_anchor_x = rect_x + rect_w
            label_anchor_y = rect_y + (rect_h * 0.5)
        else:
            content.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius:.1f}" fill="{fill}" fill-opacity="{opacity:.2f}" '
                f'stroke="{stroke}" stroke-width="2"/>'
            )
            label_anchor_x = px + radius
            label_anchor_y = py
        label = _class_name(scene_graph, prim) or prim.split("/")[-1]
        content.append(
            f'<text x="{label_anchor_x + 8:.1f}" y="{label_anchor_y + 5:.1f}" font-family="Arial, sans-serif" font-size="16" fill="#23314f">'
            f"{label}</text>"
        )

    robot_x, robot_y = project_xy(plan.base_pose[0], plan.base_pose[1])
    robot_radius_px = float(ROBOT_PROFILES[plan.robot]["base_radius"]) * scale
    content.append(
        f'<circle cx="{robot_x:.1f}" cy="{robot_y:.1f}" r="{robot_radius_px:.1f}" fill="{robot_color}" '
        'fill-opacity="0.82" stroke="#12305b" stroke-width="3"/>'
    )
    arrow_length = robot_radius_px + 34.0
    yaw = math.radians(plan.base_pose[3])
    end_x = robot_x + math.cos(yaw) * arrow_length
    end_y = robot_y - math.sin(yaw) * arrow_length
    content.append(
        f'<line x1="{robot_x:.1f}" y1="{robot_y:.1f}" x2="{end_x:.1f}" y2="{end_y:.1f}" '
        'stroke="#12305b" stroke-width="4" marker-end="url(#arrow)"/>'
    )
    title = (
        f"{plan.robot} | target={plan.target_prim.split('/')[-1]} | support={plan.support_prim.split('/')[-1]} "
        f"| side={plan.chosen_side} | yaw={plan.base_pose[3]:.1f} deg"
    )
    content.append(
        f'<text x="36" y="42" font-family="Arial, sans-serif" font-size="26" font-weight="700" fill="#14233d">{title}</text>'
    )
    content.append(
        '<text x="36" y="72" font-family="Arial, sans-serif" font-size="17" fill="#4d617f">'
        "Blue shape: support surface. Red: target object. Green: tabletop objects. Dashed: candidate base poses labeled by overlap status and target distance.</text>"
    )
    content.append("</svg>")
    output.write_text("\n".join(content), encoding="utf-8")
    return output


def plan_to_payload(plan: RobotPlacementPlan) -> Dict[str, Any]:
    payload = asdict(plan)
    payload["candidates"] = [asdict(item) for item in plan.candidates]
    return payload


def save_plan_outputs(
    scene_graph: Dict[str, Any],
    placements: PlacementMap,
    plan: RobotPlacementPlan,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "robot_base_plan.json"
    svg_path = root / "robot_base_plan.svg"
    json_path.write_text(json.dumps(plan_to_payload(plan), ensure_ascii=False, indent=2), encoding="utf-8")
    render_plan_svg(scene_graph, placements, plan, svg_path)
    return {"json": str(json_path), "svg": str(svg_path)}
