import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from isaacsim import SimulationApp
from PIL import Image
from room_usd_builder import generate_room_usd_from_scene

SCENE_GRAPH_UI_ROOT = Path(__file__).resolve().parents[2]

OBJ_OBJ_RELATIONS = {
    "left",
    "right",
    "in front of",
    "behind",
    "face to",
    "face same as",
    "supported by",
    "supports",
    "center aligned",
    "adjacent",
}

OBJ_WALL_RELATIONS = {
    "against wall",
    "in corner",
}


def load_scene_graph_json(json_path: Path) -> Dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_and_validate_scene_graph(json_path: Path) -> Dict:
    return validate_and_prepare_scene_graph(load_scene_graph_json(json_path))


def validate_and_prepare_scene_graph(data: Dict) -> Dict:
    """
    Validate and prepare the single accepted scene-graph schema:
    {
      "scene": {...},
      "obj": {"/World/Foo_0": {...}, ...},
      "edges": {"obj-obj": [...], "obj-wall": [...]}
    }
    `obj` must be a dict keyed by prim path.
    """
    if not isinstance(data, dict):
        raise ValueError("Scene graph must be a JSON object.")
    if not isinstance(data.get("scene"), dict):
        raise ValueError("Scene graph must contain a 'scene' object.")
    obj_map_raw = data.get("obj")
    if not isinstance(obj_map_raw, dict):
        raise ValueError("Scene graph must contain 'obj' as a dict keyed by usd path.")
    edges = data.get("edges")
    if not isinstance(edges, dict):
        raise ValueError("Scene graph must contain 'edges' as an object.")
    if not isinstance(edges.get("obj-obj"), list) or not isinstance(edges.get("obj-wall"), list):
        raise ValueError("Scene graph edges must include list fields: 'obj-obj' and 'obj-wall'.")

    obj_map: Dict[str, Dict] = {}
    for prim, item in obj_map_raw.items():
        if not isinstance(prim, str) or not prim:
            raise ValueError("Each key in 'obj' must be a non-empty usd path string.")
        if not isinstance(item, dict):
            raise ValueError("Each value in 'obj' must be an object.")
        meta = dict(item)
        meta.setdefault("usd_path", prim)
        meta.setdefault("class", item.get("class") or item.get("caption") or Path(prim).name)
        obj_map[prim] = meta

    prepared = dict(data)
    prepared["obj"] = obj_map
    prepared["edges"] = {"obj-obj": list(edges["obj-obj"]), "obj-wall": list(edges["obj-wall"])}

    for edge in prepared["edges"]["obj-obj"]:
        if not isinstance(edge, dict):
            raise ValueError("Each edge in 'obj-obj' must be an object.")
        rels = _relation_tokens(edge.get("relation", ""))
        invalid = [rel for rel in rels if rel not in OBJ_OBJ_RELATIONS]
        if invalid:
            raise ValueError(
                f"Unsupported obj-obj relation(s): {invalid}. "
                f"Allowed: {sorted(OBJ_OBJ_RELATIONS)}"
            )

    for edge in prepared["edges"]["obj-wall"]:
        if not isinstance(edge, dict):
            raise ValueError("Each edge in 'obj-wall' must be an object.")
        rels = _relation_tokens(edge.get("relation", ""))
        invalid = [rel for rel in rels if rel not in OBJ_WALL_RELATIONS]
        if invalid:
            raise ValueError(
                f"Unsupported obj-wall relation(s): {invalid}. "
                f"Allowed: {sorted(OBJ_WALL_RELATIONS)}"
            )

    return prepared

def discover_usd_assets(asset_root: Path) -> List[Path]:
    """Recursively gather all USD files under the given asset root."""
    return list(asset_root.rglob("*.usd"))


def find_asset_matches_by_name(name: str, usd_paths: Iterable[Path]) -> List[Path]:
    """Return USD paths that contain the given object name (case-insensitive)."""
    needle = name.lower()
    matches: List[Path] = []
    for p in usd_paths:
        if needle in p.name.lower():
            matches.append(p)
    return matches


def first_asset_match(name: str, usd_paths: Iterable[Path]) -> Optional[Path]:
    matches = find_asset_matches_by_name(name, usd_paths)
    return matches[0] if matches else None


def _object_label(meta: Dict, prim: str) -> str:
    return meta.get("class") or meta.get("class_name") or Path(prim).name


def _is_support_relation(relation: str) -> bool:
    rels = _relation_tokens(relation)
    return "supported by" in rels or "supports" in rels


def _supported_sources(edges: List[Dict]) -> set:
    return {edge["source"] for edge in edges if _is_support_relation(edge.get("relation", ""))}


def _compute_usd_bbox_info(usd_path: Path):
    """
    Return (size, center) of a prim's local bbox from the referenced USD root.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        return None
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"], useExtentsHint=True)
    bbox = bbox_cache.ComputeLocalBound(root)
    rng = bbox.ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    bmin = rng.GetMin()
    bmax = rng.GetMax()
    size = (bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2])
    center = ((bmin[0] + bmax[0]) * 0.5, (bmin[1] + bmax[1]) * 0.5, (bmin[2] + bmax[2]) * 0.5)
    return {"size": size, "center": center}


def _add_ground_plane(stage, size: float, height: float) -> None:
    """Add a simple quad mesh as a ground plane."""
    from pxr import Gf, UsdGeom

    half = size * 0.5
    plane = UsdGeom.Mesh.Define(stage, "/World/GroundPlane")
    plane.CreatePointsAttr(
        [
            (-half, -half, height),
            (-half, half, height),
            (half, half, height),
            (half, -half, height),
        ]
    )
    plane.CreateFaceVertexCountsAttr([4])
    plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    plane.CreateNormalsAttr([(0.0, 0.0, 1.0)])
    plane.SetNormalsInterpolation("constant")
    plane.CreateExtentAttr([(-half, -half, height), (half, half, height)])
    plane.CreateDisplayColorAttr([(0.6, 0.6, 0.6)])
    plane.CreateDoubleSidedAttr(True)


def _add_default_lighting(stage) -> None:
    """Add a dome light to avoid a completely dark scene."""
    from pxr import UsdLux

    dome = UsdLux.DomeLight.Define(stage, "/World/DefaultDomeLight")
    dome.CreateIntensityAttr(1000.0)
    dome.CreateExposureAttr(0.0)
    dome.CreateDiffuseAttr(1.0)
    dome.CreateSpecularAttr(1.0)


def build_stage_from_entries(
    object_entries: List[Dict],
    usd_paths: List[Path],
    edges: List[Dict],
    save_usd: Optional[Path],
    plane_size: float,
    plane_height: float,
    default_ground_z_offset: float = 0.0,
    asset_match_lookup: Optional[Dict[str, Path]] = None,
    room_usd: Optional[Path] = None,
    use_default_ground: bool = True,
) -> None:
    """
    Create a fresh stage, add references for each object, and apply transforms captured in the JSON.
    """
    # Deferred imports that require SimulationApp
    import omni.usd
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Gf, UsdGeom

    ctx = omni.usd.get_context()
    ctx.new_stage()  # start from an empty stage
    stage = ctx.get_stage()
    stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))
    if room_usd and room_usd.exists():
        add_reference_to_stage(str(room_usd), "/World/GeneratedRoom")
        print(f"[ROOM] referenced room USD: {room_usd}")

    if use_default_ground:
        # Use Isaac Sim default ground plane style and allow a slight Z offset.
        from isaacsim.core.api import World

        world_for_ground = World(stage_units_in_meters=1.0)
        world_for_ground.scene.add_default_ground_plane(
            z_position=plane_height + float(default_ground_z_offset)
        )
    elif not (room_usd and room_usd.exists()):
        _add_ground_plane(stage, plane_size, plane_height)
    _add_default_lighting(stage)

    for entry in object_entries:
        name = entry["name"]
        prim = entry["prim"]
        transform = entry["transform"]

        usd_match = asset_match_lookup.get(prim) if asset_match_lookup else None
        if usd_match is None:
            usd_match = first_asset_match(name, usd_paths)
        if usd_match is None:
            print(f"[MISS] {name} (prim: {prim}) -> NOT FOUND")
            continue

        print(f"[HIT ] {name} (prim: {prim}) -> {usd_match}")
        add_reference_to_stage(str(usd_match), prim)

        if transform is not None:
            prim_obj = stage.GetPrimAtPath(prim)
            xform = UsdGeom.Xformable(prim_obj)
            xform.ClearXformOpOrder()

            transform_op = xform.AddTransformOp()
            transform_op.Set(Gf.Matrix4d(transform))
        else:
            print("  (no transform found in JSON; leaving default)")

    _align_support_bottom_to_obb_top(stage, edges)

    if save_usd:
        stage.GetRootLayer().Export(str(save_usd))
        print(f"Stage saved to {save_usd}")


def intersect_optional_ranges(r1: Tuple[Optional[float], Optional[float]], r2: Tuple[Optional[float], Optional[float]]):
    """
    r1, r2 are (min, max) where min or max can be None.
    Returns the intersection range.
    """
    a1, b1 = r1
    a2, b2 = r2

    a = a1 if a1 is not None else a2
    if a2 is not None and (a1 is None or a2 > a1):
        a = a2

    b = b1 if b1 is not None else b2
    if b2 is not None and (b1 is None or b2 < b1):
        b = b2

    if a is not None and b is not None and a > b:
        return None  # no intersection
    return (a, b)


def _relation_tokens(relation: str) -> List[str]:
    tokens: List[str] = []
    for rel in str(relation).split(","):
        if not rel:
            continue
        tokens.append(rel)
    return tokens


def _merge_sector(base: Dict, new: Dict) -> Dict:
    base_center = base.get("sector_center")
    new_center = new.get("sector_center")
    if new_center is None:
        return base
    if base_center is None:
        out = dict(base)
        out["sector_center"] = new_center
        out["sector_half_angle"] = new.get("sector_half_angle", math.pi)
        return out
    out = dict(base)
    c = math.atan2(math.sin(base_center) + math.sin(new_center), math.cos(base_center) + math.cos(new_center))
    out["sector_center"] = c
    out["sector_half_angle"] = min(
        math.pi,
        max(base.get("sector_half_angle", math.pi), new.get("sector_half_angle", math.pi)),
    )
    return out


def _point_in_region(x: float, y: float, region: Dict) -> bool:
    xmin, xmax = region["x_range"]
    ymin, ymax = region["y_range"]
    if xmin is not None and x < xmin:
        return False
    if xmax is not None and x > xmax:
        return False
    if ymin is not None and y < ymin:
        return False
    if ymax is not None and y > ymax:
        return False
    cx, cy = region["center"]
    dist = math.hypot(x - cx, y - cy)
    dmin = region.get("dist_min")
    dmax = region.get("dist_max")
    if dmin is not None and dist < dmin:
        return False
    if dmax is not None and dist > dmax:
        return False
    return True


def infer_region_from_relation(target_coord, relation, target_object_size, spread_scale: float):
    """
    target_coord: j 的 bbox (xmin, xmax, ymin, ymax, zmin, zmax)
    relation: e.g. "left,adjacent"
    target_object_size: (dx, dy, dz)
    返回：一个 dict，包含 i 的 x/y/z 可行范围 + distance 约束
    """
    xmin, xmax, ymin, ymax, zmin, zmax = target_coord
    dx, dy, dz = target_object_size

    # 初始没有约束 → (-inf, +inf)
    region = {
        "x_range": (None, None),
        "y_range": (None, None),
        "z_range": (None, None),
        "dist_min": None,
        "dist_max": None,
        "center": ((xmin + xmax) / 2, (ymin + ymax) / 2),
        "sector_center": None,
        "sector_half_angle": math.pi,
    }

    relations = _relation_tokens(relation)
    dir_x, dir_y = 0.0, 0.0
    has_lr = any(rel in ("left", "right") for rel in relations)
    has_fb = any(rel in ("in front of", "behind") for rel in relations)
    for rel in relations:
        # ---------------- Left / Right ---------------------
        if rel == "left":
            r_y = (None, ymin - dy)
            region["y_range"] = intersect_optional_ranges(region["y_range"], r_y) or region["y_range"]
            dir_y -= 1.0

        if rel == "right":
            r_y = (ymax + dy, None)
            region["y_range"] = intersect_optional_ranges(region["y_range"], r_y) or region["y_range"]
            dir_y += 1.0

        # ---------------- Front / Behind ---------------------
        if rel == "in front of":
            r_x = (xmax + dx, None)
            region["x_range"] = intersect_optional_ranges(region["x_range"], r_x) or region["x_range"]
            dir_x += 1.0

        if rel == "behind":
            r_x = (None, xmin - dx)
            region["x_range"] = intersect_optional_ranges(region["x_range"], r_x) or region["x_range"]
            dir_x -= 1.0

    # ---------------- Alignment ---------------------
    if "center aligned" in relations:
        cx, cy = region["center"]
        # left/right relation -> align in front/behind axis (x)
        if has_lr and not has_fb:
            region["x_range"] = intersect_optional_ranges(region["x_range"], (cx, cx)) or (cx, cx)
        # front/behind relation -> align in left/right axis (y)
        elif has_fb and not has_lr:
            region["y_range"] = intersect_optional_ranges(region["y_range"], (cy, cy)) or (cy, cy)
        else:
            region["x_range"] = intersect_optional_ranges(region["x_range"], (cx, cx)) or (cx, cx)
            region["y_range"] = intersect_optional_ranges(region["y_range"], (cy, cy)) or (cy, cy)

    if dir_x != 0.0 or dir_y != 0.0:
        region["sector_center"] = math.atan2(dir_y, dir_x)
        region["sector_half_angle"] = math.radians(35.0)

    return region


def _look_at_to_euler(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Convert an eye/target pair into Euler angles (roll, pitch, yaw) that point the camera at the target.
    Assumes Z-up; roll is fixed at 0.
    """
    delta = target - eye
    dx, dy, dz = delta
    yaw = math.degrees(math.atan2(dy, dx))
    dist_xy = math.hypot(dx, dy)
    pitch = -math.degrees(math.atan2(dz, dist_xy))
    return np.array([0.0, pitch, yaw])


def sample_position_in_region(region, target_object_size, spread_scale: float):
    _ = target_object_size

    # Resolve bounds.
    xmin, xmax = region["x_range"]
    if xmin is None:
        xmin = -10.0 * spread_scale
    if xmax is None:
        xmax = 10.0 * spread_scale
    ymin, ymax = region["y_range"]
    if ymin is None:
        ymin = -10.0 * spread_scale
    if ymax is None:
        ymax = 10.0 * spread_scale

    # Sample z once; XY is sampled with "align-first + fan expansion".
    zmin, zmax = region["z_range"]
    if zmin is None:
        zmin = 0
    if zmax is None:
        zmax = 2.0
    z = random.uniform(zmin, zmax)

    cx, cy = region["center"]
    # 1) align-first sample
    align_x = min(max(cx, xmin), xmax)
    align_y = min(max(cy, ymin), ymax)
    if _point_in_region(align_x, align_y, region):
        return (align_x, align_y, z)

    # 2) sector fan expansion sample
    sector_center = region.get("sector_center")
    sector_half = region.get("sector_half_angle", math.pi)
    dist_min = region.get("dist_min")
    dist_max = region.get("dist_max")
    if dist_min is None:
        dist_min = 0.0
    if dist_max is None:
        dist_max = 10.0 * spread_scale
    if dist_max < dist_min:
        dist_max = dist_min + 0.1

    for expand in (1.0, 1.5, 2.0, 3.0):
        half = min(math.pi, sector_half * expand)
        r_hi = max(dist_min + 0.1, dist_max * expand)
        for _ in range(40):
            if sector_center is None:
                angle = random.uniform(-math.pi, math.pi)
            else:
                angle = random.uniform(sector_center - half, sector_center + half)
            radius = random.uniform(dist_min, r_hi)
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            x = min(max(x, xmin), xmax)
            y = min(max(y, ymin), ymax)
            if _point_in_region(x, y, region):
                return (x, y, z)

    # 3) fallback: uniform in allowed rectangle
    return (random.uniform(xmin, xmax), random.uniform(ymin, ymax), z)


def _clamp_pos_to_region(pos: Tuple[float, float, float], region: Dict) -> Tuple[float, float, float]:
    """Clamp a sampled position to the region bounds."""
    x, y, z = pos
    xmin, xmax = region["x_range"]
    ymin, ymax = region["y_range"]
    zmin, zmax = region["z_range"]
    if xmin is not None:
        x = max(x, xmin)
    if xmax is not None:
        x = min(x, xmax)
    if ymin is not None:
        y = max(y, ymin)
    if ymax is not None:
        y = min(y, ymax)
    if zmin is not None:
        z = max(z, zmin)
    if zmax is not None:
        z = min(z, zmax)
    return (x, y, z)


def _apply_edge_range_constraint(
    region: Dict,
    edge_range: Dict,
    target_center: Tuple[float, float, float],
) -> Dict:
    """
    Constrain region using optional per-axis ranges.
    Range values are interpreted as offsets from the target center.
    """
    out = dict(region)
    for axis, idx in zip(("x", "y", "z"), (0, 1, 2)):
        if axis not in edge_range:
            continue
        r = edge_range[axis]
        if not isinstance(r, (list, tuple)) or len(r) != 2:
            continue
        lo, hi = r
        lo_w = None if lo is None else target_center[idx] + lo
        hi_w = None if hi is None else target_center[idx] + hi
        key = f"{axis}_range"
        out[key] = intersect_optional_ranges(out[key], (lo_w, hi_w)) or (lo_w, hi_w)
    return out


def _merge_range(base_range: Tuple[Optional[float], Optional[float]], new_range: Tuple[Optional[float], Optional[float]]):
    """Intersect two ranges, falling back to the new range if they do not overlap."""
    merged = intersect_optional_ranges(base_range, new_range)
    return merged if merged is not None else new_range


def _merge_dist_min(base: Optional[float], new: Optional[float]) -> Optional[float]:
    """Use the stricter (larger) minimum distance."""
    if new is None:
        return base
    if base is None:
        return new
    return max(base, new)


def _merge_dist_max(base: Optional[float], new: Optional[float]) -> Optional[float]:
    """Use the stricter (smaller) maximum distance."""
    if new is None:
        return base
    if base is None:
        return new
    return min(base, new)


def _aggregate_source_constraints(
    source: str,
    locked: Dict[str, Tuple[float, float, float]],
    edges: List[Dict],
    size_lookup: Dict[str, Tuple[float, float, float]],
    plane_height: float,
    spread_scale: float,
):
    """
    Collect constraints for `source` from all edges whose target is already locked.
    Returns (region, constraint_count).
    """
    region = {
        "x_range": (None, None),
        "y_range": (None, None),
        # default to sitting on the ground if no constraint touches z
        "z_range": (plane_height + size_lookup[source][2] * 0.5, plane_height + size_lookup[source][2] * 0.5),
        "dist_min": None,
        "dist_max": None,
        "center": (0.0, 0.0),
        "sector_center": None,
        "sector_half_angle": math.pi,
    }
    centers = []
    constraint_count = 0

    for edge in edges:
        if edge["source"] != source:
            continue
        target = edge["target"]
        if target not in locked:
            continue

        constraint_count += 1
        target_center = locked[target]
        target_size = size_lookup[target]
        target_coord = _bbox_from_center(target_center, target_size)
        candidate = infer_region_from_relation(target_coord, edge["relation"], size_lookup[source], spread_scale)
        if "range" in edge:
            candidate = _apply_edge_range_constraint(candidate, edge["range"], target_center)

        region["x_range"] = _merge_range(region["x_range"], candidate["x_range"])
        region["y_range"] = _merge_range(region["y_range"], candidate["y_range"])
        region["z_range"] = _merge_range(region["z_range"], candidate["z_range"])
        region["dist_min"] = _merge_dist_min(region["dist_min"], candidate.get("dist_min"))
        region["dist_max"] = _merge_dist_max(region["dist_max"], candidate.get("dist_max"))
        region = _merge_sector(region, candidate)
        centers.append(candidate["center"])

    if centers:
        cx = sum(c[0] for c in centers) / len(centers)
        cy = sum(c[1] for c in centers) / len(centers)
        region["center"] = (cx, cy)

    return region, constraint_count


def _build_default_sampling_region(
    size: Tuple[float, float, float],
    plane_height: float,
    spread_scale: float,
    center: Tuple[float, float] = (0.0, 0.0),
) -> Dict:
    return {
        "x_range": (-10.0 * spread_scale, 10.0 * spread_scale),
        "y_range": (-10.0 * spread_scale, 10.0 * spread_scale),
        "z_range": (plane_height + size[2] * 0.5, plane_height + size[2] * 0.5),
        "dist_min": None,
        "dist_max": None,
        "center": center,
        "sector_center": None,
        "sector_half_angle": math.pi,
    }


def _has_vertical_support(prim: str, edges: List[Dict]) -> bool:
    """Return True if prim is the supported (top) object in any support edge."""
    for edge in edges:
        pair = _support_pair(edge)
        if not pair:
            continue
        top, _base = pair
        if top == prim:
            return True
    return False


def _enforce_support_alignment(
    placements: Dict[str, Tuple[float, float, float]],
    edges: List[Dict],
    size_lookup: Dict[str, Tuple[float, float, float]],
) -> None:
    """
    For each supported-by style relation, snap the source so its bbox min z == target bbox max z.
    """
    for edge in edges:
        pair = _support_pair(edge)
        if not pair:
            continue
        source, target = pair
        if source not in placements or target not in placements:
            continue
        tx, ty, tz = placements[target]
        _, _, tdz = size_lookup[target]
        _, _, sdz = size_lookup[source]
        target_top = tz + tdz * 0.5
        placements[source] = (placements[source][0], placements[source][1], target_top + sdz * 0.5)


def _align_support_bottom_to_obb_top(stage, edges: List[Dict]) -> None:
    """
    Move supported objects so their bottom center lies within the supporter top OBB.
    Uses world-space oriented bounding boxes from UsdGeom.BBoxCache.
    """
    from pxr import Gf, Usd, UsdGeom

    processed_sources = set()
    for edge in edges:
        pair = _support_pair(edge)
        if not pair:
            continue

        source_path, target_path = pair

        # Avoid double-processing the same source.
        if source_path in processed_sources:
            continue

        source_prim = stage.GetPrimAtPath(source_path)
        target_prim = stage.GetPrimAtPath(target_path)
        if not source_prim.IsValid() or not target_prim.IsValid():
            continue

        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"], useExtentsHint=True)
        source_bbox = cache.ComputeWorldBound(source_prim)
        target_bbox = cache.ComputeWorldBound(target_prim)

        s_range = source_bbox.GetRange()
        t_range = target_bbox.GetRange()
        if s_range.IsEmpty() or t_range.IsEmpty():
            continue

        s_mat = source_bbox.GetMatrix()
        t_mat = target_bbox.GetMatrix()

        s_min = s_range.GetMin()
        s_max = s_range.GetMax()
        # Bottom face center in source OBB local space, then lifted to world.
        bottom_local = Gf.Vec3d((s_min[0] + s_max[0]) * 0.5, (s_min[1] + s_max[1]) * 0.5, s_min[2])
        bottom_world = s_mat.Transform(bottom_local)

        t_min = t_range.GetMin()
        t_max = t_range.GetMax()
        t_inv = t_mat.GetInverse()
        # Express the source bottom center in the target OBB frame.
        bottom_in_target = t_inv.Transform(bottom_world)

        # Keep the whole source bottom footprint inside target OBB top footprint.
        bottom_corners = [
            Gf.Vec3d(sx, sy, s_min[2])
            for sx in (s_min[0], s_max[0])
            for sy in (s_min[1], s_max[1])
        ]
        corner_offsets = []
        for c in bottom_corners:
            c_world = s_mat.Transform(c)
            c_target = t_inv.Transform(c_world)
            corner_offsets.append((c_target[0] - bottom_in_target[0], c_target[1] - bottom_in_target[1]))

        inset_ratio = 0.1  # keep contact point away from edges
        tx_extent = (t_max[0] - t_min[0]) * 0.5
        ty_extent = (t_max[1] - t_min[1]) * 0.5
        tx_center = (t_min[0] + t_max[0]) * 0.5
        ty_center = (t_min[1] + t_max[1]) * 0.5
        x_inset = tx_extent * inset_ratio
        y_inset = ty_extent * inset_ratio
        x_low = t_min[0] + x_inset
        x_high = t_max[0] - x_inset
        y_low = t_min[1] + y_inset
        y_high = t_max[1] - y_inset
        # If the OBB is too thin, fall back to center.
        if x_low > x_high:
            x_low = x_high = tx_center
        if y_low > y_high:
            y_low = y_high = ty_center

        # Build feasible ranges for bottom-center placement so all four bottom corners stay in-bounds.
        cx_low, cx_high = x_low, x_high
        cy_low, cy_high = y_low, y_high
        for ox, oy in corner_offsets:
            cx_low = max(cx_low, x_low - ox)
            cx_high = min(cx_high, x_high - ox)
            cy_low = max(cy_low, y_low - oy)
            cy_high = min(cy_high, y_high - oy)

        # If infeasible (object footprint larger than support top), clamp to center.
        if cx_low > cx_high:
            cx_low = cx_high = tx_center
        if cy_low > cy_high:
            cy_low = cy_high = ty_center

        # Sample a feasible point on the target top surface.
        sample_x = random.uniform(cx_low, cx_high)
        sample_y = random.uniform(cy_low, cy_high)
        top_z = t_max[2]

        target_top_world = t_mat.Transform(Gf.Vec3d(sample_x, sample_y, top_z))
        delta = target_top_world - bottom_world

        xform = UsdGeom.Xformable(source_prim)
        ops = xform.GetOrderedXformOps()
        if not ops:
            continue

        mat = Gf.Matrix4d(ops[0].Get())
        translate = mat.ExtractTranslation()
        mat.SetTranslateOnly(translate + delta)
        ops[0].Set(mat)

        processed_sources.add(source_path)

def _resolve_placement_overlaps(
    placements: Dict[str, Tuple[float, float, float]],
    size_lookup: Dict[str, Tuple[float, float, float]],
    exclude_prims: Optional[set] = None,
    separation: float = 0.01,
    max_iters: int = 50,
) -> None:
    """
    Push objects in the horizontal plane so their axis-aligned bounding boxes do not overlap.
    Touching (volume zero) is allowed.
    """

    def bbox(center, size):
        return _bbox_from_center(center, size)

    def overlap_amount(a_min, a_max, b_min, b_max):
        return min(a_max, b_max) - max(a_min, b_min)

    excluded = exclude_prims or set()
    prims = [prim for prim in placements.keys() if prim not in excluded]

    for _ in range(max_iters):
        changed = False
        for i in range(len(prims)):
            for j in range(i + 1, len(prims)):
                p1, p2 = prims[i], prims[j]
                c1, c2 = placements[p1], placements[p2]
                s1, s2 = size_lookup[p1], size_lookup[p2]
                b1 = bbox(c1, s1)
                b2 = bbox(c2, s2)

                ox = overlap_amount(b1[0], b1[1], b2[0], b2[1])
                oy = overlap_amount(b1[2], b1[3], b2[2], b2[3])
                oz = overlap_amount(b1[4], b1[5], b2[4], b2[5])

                # Overlap exists only if all three have positive overlap.
                if ox <= 0 or oy <= 0 or oz <= 0:
                    continue

                # Resolve along the axis with smaller overlap to minimize movement.
                if ox < oy:
                    axis = 0
                    delta = ox / 2 + separation
                else:
                    axis = 1
                    delta = oy / 2 + separation

                dir_vec = c1[axis] - c2[axis]
                if dir_vec == 0.0:
                    dir_vec = 1.0 if random.random() > 0.5 else -1.0
                sign = 1.0 if dir_vec > 0 else -1.0

                c1_list = list(c1)
                c2_list = list(c2)
                c1_list[axis] += delta * sign
                c2_list[axis] -= delta * sign
                placements[p1] = (c1_list[0], c1_list[1], c1_list[2])
                placements[p2] = (c2_list[0], c2_list[1], c2_list[2])
                changed = True

        if not changed:
            break


def _estimate_object_size(meta: Dict) -> Tuple[float, float, float]:
    """Approximate axis-aligned bbox size from world_3d_bbox or 3d_bbox."""
    bbox = meta.get("world_3d_bbox") or meta.get("3d_bbox")
    if not bbox:
        return (1.0, 1.0, 1.0)
    xs, ys, zs = zip(*bbox)
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    dz = max(zs) - min(zs)
    # Avoid zero-size boxes that break sampling.
    return (dx or 1.0, dy or 1.0, dz or 1.0)


def _bbox_from_center(center: Tuple[float, float, float], size: Tuple[float, float, float]):
    """Create (xmin, xmax, ymin, ymax, zmin, zmax) from center + size."""
    cx, cy, cz = center
    dx, dy, dz = size
    hx, hy, hz = dx * 0.5, dy * 0.5, dz * 0.5
    return (cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz)


def _circular_mean(angles: List[float]) -> float:
    if not angles:
        return 0.0
    return math.atan2(sum(math.sin(a) for a in angles), sum(math.cos(a) for a in angles))


def _support_pair(edge: Dict) -> Optional[Tuple[str, str]]:
    """Return (top_object, base_supporter) if edge expresses support."""
    source = edge.get("source")
    target = edge.get("target")
    if not source or not target:
        return None
    rels = _relation_tokens(edge.get("relation", ""))
    if "supported by" in rels:
        return source, target
    if "supports" in rels:
        return target, source
    return None


def _room_bounds_from_scene(data: Dict, spread_scale: float) -> Tuple[float, float, float, float]:
    dims = data.get("scene", {}).get("dimensions", {})
    length = dims.get("length")
    width = dims.get("width")
    if isinstance(length, (int, float)) and isinstance(width, (int, float)) and length > 0 and width > 0:
        return (-float(length) * 0.5, float(length) * 0.5, -float(width) * 0.5, float(width) * 0.5)
    r = 10.0 * spread_scale
    return (-r, r, -r, r)


def _collect_anchor_priority(
    objs: Dict[str, Dict],
    obj_obj_edges: List[Dict],
    obj_wall_edges: List[Dict],
    size_lookup: Dict[str, Tuple[float, float, float]],
) -> Tuple[List[str], List[str], Dict[str, str]]:
    hard_wall_objects = set()
    for edge in obj_wall_edges:
        source = edge.get("source")
        if source not in objs:
            continue
        rels = _relation_tokens(edge.get("relation", ""))
        if edge.get("hard") is True or "against wall" in rels or "in corner" in rels:
            hard_wall_objects.add(source)

    support_top_to_base: Dict[str, str] = {}
    supporter_objects = set()
    for edge in obj_obj_edges:
        pair = _support_pair(edge)
        if not pair:
            continue
        top, base = pair
        if top in objs and base in objs:
            support_top_to_base[top] = base
            supporter_objects.add(base)

    def _volume(prim: str) -> float:
        sx, sy, sz = size_lookup[prim]
        return sx * sy * sz

    prioritized = sorted(
        objs.keys(),
        key=lambda p: (
            1 if p in hard_wall_objects else 0,
            1 if p in supporter_objects else 0,
            _volume(p),
            p,
        ),
        reverse=True,
    )

    largest_by_volume = sorted(objs.keys(), key=lambda p: (_volume(p), p), reverse=True)[: min(5, len(objs))]
    anchored_set = set(hard_wall_objects) | set(supporter_objects) | set(largest_by_volume)
    anchored = [p for p in prioritized if p in anchored_set]
    general = [p for p in prioritized if p not in anchored]
    return anchored, general, support_top_to_base


def _build_grid_points(xmin: float, xmax: float, ymin: float, ymax: float, grid_step: float) -> List[Tuple[float, float]]:
    xs: List[float] = []
    ys: List[float] = []
    x = xmin
    while x <= xmax + 1e-8:
        xs.append(round(x, 6))
        x += grid_step
    y = ymin
    while y <= ymax + 1e-8:
        ys.append(round(y, 6))
        y += grid_step
    return [(xv, yv) for xv in xs for yv in ys]


def _xy_overlap(center_a, size_a, center_b, size_b, eps: float = 1e-4) -> bool:
    ax0, ax1, ay0, ay1, _, _ = _bbox_from_center(center_a, size_a)
    bx0, bx1, by0, by1, _, _ = _bbox_from_center(center_b, size_b)
    return (min(ax1, bx1) - max(ax0, bx0) > eps) and (min(ay1, by1) - max(ay0, by0) > eps)


def _parse_wall_side(edge: Dict) -> Optional[str]:
    text = str(edge.get("target", "")).lower()
    if "left wall" in text:
        return "left"
    if "right wall" in text:
        return "right"
    if "front wall" in text:
        return "front"
    if "back wall" in text or "behind wall" in text:
        return "behind"
    return None


def _solve_discrete_yaw_from_edges(
    placements: Dict[str, Tuple[float, float, float]],
    edges: List[Dict],
) -> Dict[str, float]:
    discrete = [0.0, 90.0, 180.0, 270.0]
    yaws_deg = {prim: 0.0 for prim in placements}
    for _ in range(4):
        updated = dict(yaws_deg)
        for prim in placements:
            goals = []
            for edge in edges:
                if edge.get("source") != prim:
                    continue
                target = edge.get("target")
                if target not in placements:
                    continue
                rels = _relation_tokens(edge.get("relation", ""))
                sx, sy, _ = placements[prim]
                tx, ty, _ = placements[target]
                for rel in rels:
                    if rel == "face to":
                        goals.append(math.degrees(math.atan2(ty - sy, tx - sx)))
                    elif rel == "face same as":
                        goals.append(yaws_deg.get(target, 0.0))
            if goals:
                avg = sum(goals) / len(goals)
                best = min(discrete, key=lambda d: min(abs(d - avg), 360.0 - abs(d - avg)))
                updated[prim] = best
        yaws_deg = updated
    return {k: math.radians(v) for k, v in yaws_deg.items()}


def _build_entries_from_scene_edges(
    data: Dict,
    plane_height: float,
    spread_scale: float,
    grid_step: float,
    asset_bbox_lookup: Optional[Dict[str, Dict[str, Tuple[float, float, float]]]] = None,
) -> Tuple[List[Dict], Dict[str, Tuple[float, float, float]]]:
    objs = data.get("obj", {})
    edges_payload = data.get("edges", {})
    edges: List[Dict] = edges_payload.get("obj-obj", [])
    wall_edges: List[Dict] = edges_payload.get("obj-wall", [])
    if not objs:
        return [], {}

    size_lookup: Dict[str, Tuple[float, float, float]] = {}
    center_lookup: Dict[str, Tuple[float, float, float]] = {}
    for prim, meta in objs.items():
        if asset_bbox_lookup and prim in asset_bbox_lookup:
            info = asset_bbox_lookup[prim]
            raw_size = info["size"]
            raw_center = info["center"]
        else:
            raw_size = _estimate_object_size(meta)
            raw_center = (0.0, 0.0, 0.0)

        size_lookup[prim] = raw_size
        center_lookup[prim] = raw_center
    room_xmin, room_xmax, room_ymin, room_ymax = _room_bounds_from_scene(data, spread_scale)
    grid_points = _build_grid_points(room_xmin, room_xmax, room_ymin, room_ymax, max(0.05, grid_step))

    anchored, general, support_top_to_base = _collect_anchor_priority(objs, edges, wall_edges, size_lookup)
    placement_order = anchored + general
    print(f"[ANCHOR] anchored={anchored}")
    non_adjacent_clearance = 0.1

    adjacent_pairs = set()
    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src not in objs or tgt not in objs:
            continue
        rels = _relation_tokens(edge.get("relation", ""))
        if "adjacent" in rels:
            adjacent_pairs.add(frozenset((src, tgt)))

    # Pre-compute extra wall clearance based on incoming directional relations.
    incoming_clearance = {prim: {"left": 0.0, "right": 0.0, "front": 0.0, "behind": 0.0} for prim in objs}
    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src not in objs or tgt not in objs:
            continue
        src_size = size_lookup[src]
        tgt_size = size_lookup[tgt]
        for rel in _relation_tokens(edge.get("relation", "")):
            if rel == "left":
                incoming_clearance[tgt]["left"] = max(incoming_clearance[tgt]["left"], 0.5 * (src_size[1] + tgt_size[1]))
            elif rel == "right":
                incoming_clearance[tgt]["right"] = max(incoming_clearance[tgt]["right"], 0.5 * (src_size[1] + tgt_size[1]))
            elif rel == "in front of":
                incoming_clearance[tgt]["front"] = max(incoming_clearance[tgt]["front"], 0.5 * (src_size[0] + tgt_size[0]))
            elif rel == "behind":
                incoming_clearance[tgt]["behind"] = max(incoming_clearance[tgt]["behind"], 0.5 * (src_size[0] + tgt_size[0]))

    placements: Dict[str, Tuple[float, float, float]] = {}
    unresolved = list(placement_order)

    while unresolved:
        progressed = False
        for prim in list(unresolved):
            base = support_top_to_base.get(prim)
            if base and base not in placements:
                continue

            sx, sy, sz = size_lookup[prim]
            hx, hy = sx * 0.5, sy * 0.5
            z = plane_height + sz * 0.5
            x_min = room_xmin + hx + incoming_clearance[prim]["behind"]
            x_max = room_xmax - hx - incoming_clearance[prim]["front"]
            y_min = room_ymin + hy + incoming_clearance[prim]["left"]
            y_max = room_ymax - hy - incoming_clearance[prim]["right"]

            adjacent_targets: List[str] = []
            center_targets_xy: List[str] = []
            center_targets_x: List[str] = []
            center_targets_y: List[str] = []
            lr_targets: List[str] = []
            fb_targets: List[str] = []

            for edge in edges:
                if edge.get("source") != prim:
                    continue
                tgt = edge.get("target")
                if tgt not in placements:
                    continue
                tx, ty, _ = placements[tgt]
                tsx, tsy, _ = size_lookup[tgt]
                rels = _relation_tokens(edge.get("relation", ""))
                has_lr = any(rel in ("left", "right") for rel in rels)
                has_fb = any(rel in ("in front of", "behind") for rel in rels)
                for rel in rels:
                    if rel == "left":
                        y_max = min(y_max, ty - 0.5 * (sy + tsy))
                        lr_targets.append(tgt)
                    elif rel == "right":
                        y_min = max(y_min, ty + 0.5 * (sy + tsy))
                        lr_targets.append(tgt)
                    elif rel == "in front of":
                        x_min = max(x_min, tx + 0.5 * (sx + tsx))
                        fb_targets.append(tgt)
                    elif rel == "behind":
                        x_max = min(x_max, tx - 0.5 * (sx + tsx))
                        fb_targets.append(tgt)
                    elif rel == "adjacent":
                        adjacent_targets.append(tgt)
                if "center aligned" in rels:
                    # left/right relation -> align x; front/behind relation -> align y
                    if has_lr and not has_fb:
                        center_targets_x.append(tgt)
                    elif has_fb and not has_lr:
                        center_targets_y.append(tgt)
                    else:
                        center_targets_xy.append(tgt)

            for edge in wall_edges:
                if edge.get("source") != prim:
                    continue
                rels = _relation_tokens(edge.get("relation", ""))
                is_against_wall = "against wall" in rels
                is_in_corner = "in corner" in rels
                if not is_against_wall and not is_in_corner:
                    continue

                side = _parse_wall_side(edge)
                if side is None and is_against_wall:
                    # If edge does not specify a wall side, pin to a deterministic default wall.
                    side = "behind"
                hard = edge.get("hard") is True or is_against_wall or is_in_corner
                band = 0.15 if hard else 0.5

                if side == "left":
                    y_min = max(y_min, room_ymin + hy)
                    y_max = min(y_max, room_ymin + hy + band)
                elif side == "right":
                    y_min = max(y_min, room_ymax - hy - band)
                    y_max = min(y_max, room_ymax - hy)
                elif side == "front":
                    x_min = max(x_min, room_xmax - hx - band)
                    x_max = min(x_max, room_xmax - hx)
                elif side == "behind":
                    x_min = max(x_min, room_xmin + hx)
                    x_max = min(x_max, room_xmin + hx + band)

                if is_in_corner:
                    text = str(edge.get("target", "")).lower()
                    if "left" in text:
                        y_min = max(y_min, room_ymin + hy)
                        y_max = min(y_max, room_ymin + hy + band)
                    elif "right" in text:
                        y_min = max(y_min, room_ymax - hy - band)
                        y_max = min(y_max, room_ymax - hy)

                    if "front" in text:
                        x_min = max(x_min, room_xmax - hx - band)
                        x_max = min(x_max, room_xmax - hx)
                    elif "back" in text or "behind" in text:
                        x_min = max(x_min, room_xmin + hx)
                        x_max = min(x_max, room_xmin + hx + band)

            if base and base in placements:
                bx, by, bz = placements[base]
                _, _, bsz = size_lookup[base]
                z = bz + 0.5 * bsz + 0.5 * sz
                x_min = max(x_min, bx - 0.5)
                x_max = min(x_max, bx + 0.5)
                y_min = max(y_min, by - 0.5)
                y_max = min(y_max, by + 0.5)

            if x_min > x_max:
                x_min, x_max = x_max, x_min
            if y_min > y_max:
                y_min, y_max = y_max, y_min

            candidates = [(x, y) for (x, y) in grid_points if x_min <= x <= x_max and y_min <= y <= y_max]
            if not candidates:
                # If constraints are over-constrained, fall back to valid room grid points.
                candidates = list(grid_points)

            valid = []
            for x, y in candidates:
                center = (x, y, z)
                overlap = False
                for other, ocenter in placements.items():
                    # Allow vertical support overlap with its support base.
                    if base == other:
                        continue
                    if _xy_overlap(center, size_lookup[prim], ocenter, size_lookup[other]):
                        overlap = True
                        break
                    # If there is no adjacent relation, keep a minimum horizontal clearance.
                    if frozenset((prim, other)) not in adjacent_pairs:
                        osx, osy, _ = size_lookup[other]
                        dx = abs(center[0] - ocenter[0])
                        dy = abs(center[1] - ocenter[1])
                        touch_x = 0.5 * (sx + osx)
                        touch_y = 0.5 * (sy + osy)
                        gap_x = dx - touch_x
                        gap_y = dy - touch_y
                        if max(gap_x, gap_y) < non_adjacent_clearance:
                            overlap = True
                            break
                if not overlap:
                    valid.append((x, y))
            candidates = valid or candidates

            def _score(x: float, y: float) -> float:
                score = 0.0
                for tgt in adjacent_targets:
                    tx, ty, _ = placements[tgt]
                    tsx, tsy, _ = size_lookup[tgt]
                    dx = abs(x - tx)
                    dy = abs(y - ty)
                    touch_x = 0.5 * (sx + tsx)
                    touch_y = 0.5 * (sy + tsy)
                    score += min(abs(dx - touch_x), abs(dy - touch_y))
                for tgt in center_targets_xy:
                    tx, ty, _ = placements[tgt]
                    score += abs(x - tx) + abs(y - ty)
                for tgt in center_targets_x:
                    tx, _, _ = placements[tgt]
                    score += abs(x - tx)
                for tgt in center_targets_y:
                    _, ty, _ = placements[tgt]
                    score += abs(y - ty)
                # Keep orthogonal-axis drift small for directional relations.
                for tgt in lr_targets:
                    tx, _, _ = placements[tgt]
                    score += 0.5 * abs(x - tx)
                for tgt in fb_targets:
                    _, ty, _ = placements[tgt]
                    score += 0.5 * abs(y - ty)
                return score

            scored = [(p, _score(p[0], p[1])) for p in candidates]
            best_score = min(s for _, s in scored)
            ties = [p for p, s in scored if abs(s - best_score) <= 1e-9]
            best_xy = random.choice(ties)
            placements[prim] = (best_xy[0], best_xy[1], z)
            unresolved.remove(prim)
            progressed = True

        if not progressed:
            prim = unresolved.pop(0)
            sx, sy, sz = size_lookup[prim]
            placements[prim] = (0.0, 0.0, plane_height + sz * 0.5)

    yaw_lookup = _solve_discrete_yaw_from_edges(placements, edges)
    entries = []
    for prim, meta in objs.items():
        label = _object_label(meta, prim)
        cx, cy, cz = placements[prim]
        yaw = yaw_lookup.get(prim, 0.0)
        c = math.cos(yaw)
        s = math.sin(yaw)
        rot = [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]

        # 保持对象几何中心落在采样位置：T = placement - R * center
        center_vec = np.array(center_lookup.get(prim, (0.0, 0.0, 0.0)), dtype=float)
        placement_vec = np.array((cx, cy, cz), dtype=float)
        rot_mat = np.array(rot)
        tx, ty, tz = (placement_vec - rot_mat @ center_vec).tolist()

        transform = [
            [rot[0][0], rot[0][1], rot[0][2], 0.0],
            [rot[1][0], rot[1][1], rot[1][2], 0.0],
            [rot[2][0], rot[2][1], rot[2][2], 0.0],
            [tx, ty, tz, 1.0],
        ]
        entries.append({"prim": prim, "name": label, "transform": transform, "center": center_vec.tolist()})

    entries.sort(key=lambda e: e["name"])
    return entries, placements


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild a scene from a scene-graph JSON by locating USD assets and applying transforms."
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=SCENE_GRAPH_UI_ROOT / "isaac_local" / "my_viewer" / "full_scene_graph.json",
        help="Path to the scene-graph JSON file.",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=SCENE_GRAPH_UI_ROOT / "isaac_local" / "my_viewer" / "test_usd",
        help="Root directory to search for USD files.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run SimulationApp headless (default).",
    )
    parser.add_argument(
        "--windowed",
        dest="headless",
        action="store_false",
        help="Run with a UI window (disables headless).",
    )
    parser.add_argument(
        "--save-usd",
        type=Path,
        default=None,
        help="If set, export the rebuilt stage to this USD path.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=10000,
        help="Number of frames to step before closing.",
    )
    parser.add_argument(
        "--plane-size",
        type=float,
        default=50.0,
        help="Width/length of the ground plane (meters).",
    )
    parser.add_argument(
        "--plane-height",
        type=float,
        default=0.0,
        help="Z height to place the ground plane.",
    )
    parser.add_argument(
        "--generate-room",
        action="store_true",
        help="Generate a standalone room USD from scene dimensions and reference it in the final stage.",
    )
    parser.add_argument(
        "--room-usd",
        type=Path,
        default=None,
        help="Path for generated room USD (used with --generate-room).",
    )
    parser.add_argument(
        "--room-include-ceiling",
        action="store_true",
        help="Include a ceiling mesh when generating room USD.",
    )
    parser.add_argument(
        "--room-include-front-wall",
        action="store_true",
        help="Include a front wall mesh when generating room USD.",
    )
    parser.add_argument(
        "--room-texture-dir",
        type=Path,
        default=SCENE_GRAPH_UI_ROOT / "third_party" / "stable_material",
        help=(
            "Directory containing StableMaterials textures with names like "
            "floor_basecolor.png, floor_normal.png, floor_metallic_roughness.png "
            "(and same for walls/ceiling)."
        ),
    )
    parser.add_argument(
        "--no-default-ground",
        action="store_true",
        help="Do not add Isaac default ground plane.",
    )
    parser.add_argument(
        "--spread-scale",
        type=float,
        default=0.5,
        help="Scale positional sampling bounds; <1 makes layout tighter.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.25,
        help="Grid resolution in meters for room discretization during placement.",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=SCENE_GRAPH_UI_ROOT / "web" / "assets" / "renders" / "render.png",
        help="Capture the viewport to this PNG path after layout.",
    )
    parser.add_argument(
        "--capture-frame",
        type=int,
        default=10,
        help="Frame index (0-based) at which to capture the screenshot.",
    )
    parser.add_argument(
        "--camera-eye",
        type=float,
        nargs=3,
        default=[30.0, 0, 30.0],
        help="Camera position used for the capture (x y z).",
    )
    parser.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
        help="Camera look-at target used for the capture (x y z).",
    )
    parser.add_argument(
        "--camera-euler",
        type=float,
        nargs=3,
        default=[0.0, 90.0, 0.0],
        help="Camera orientation as Euler angles in degrees (roll pitch yaw). Ignored if --camera-target is used.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[1920, 1080],
        help="Camera sensor resolution (width height).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed; defaults to a random system seed.",
    )
    parser.set_defaults(headless=False)
    return parser.parse_args()


def initialize_random_seed(seed_arg: Optional[int]) -> int:
    """Seed Python and NumPy RNGs and return the effective seed."""
    system_rng = random.SystemRandom()
    seed = seed_arg if seed_arg is not None else system_rng.randrange(0, 2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    print(f"[seed] {seed}")
    return seed


def build_asset_bbox_lookup(
    data: Dict,
    usd_paths: List[Path],
    asset_match_lookup: Optional[Dict[str, Path]] = None,
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """Precompute local bbox size/center from matched USD assets for each object."""
    if asset_match_lookup is None:
        asset_match_lookup = build_asset_match_lookup(data, usd_paths)
    asset_bbox_lookup: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    for prim, usd_match in asset_match_lookup.items():
        if not usd_match:
            continue
        info = _compute_usd_bbox_info(usd_match)
        if info:
            asset_bbox_lookup[prim] = info
    return asset_bbox_lookup


def build_asset_match_lookup(data: Dict, usd_paths: List[Path]) -> Dict[str, Path]:
    """Choose one USD match per prim so bbox center and final reference stay consistent."""
    asset_match_lookup: Dict[str, Path] = {}
    for prim, meta in data.get("obj", {}).items():
        label = _object_label(meta, prim)
        usd_match = first_asset_match(label, usd_paths) or first_asset_match(Path(prim).name, usd_paths)
        if usd_match:
            asset_match_lookup[prim] = usd_match
    return asset_match_lookup


def print_layout_matches(
    object_entries: List[Dict],
    placements: Dict[str, Tuple[float, float, float]],
    usd_paths: List[Path],
) -> None:
    print("\nMatches (edge-based placement):")
    for entry in object_entries:
        name = entry["name"]
        prim = entry["prim"]
        transform = entry["transform"]
        matches = find_asset_matches_by_name(name, usd_paths)

        print(f"\n{name} (prim: {prim})")
        print(f"transform: {transform}")
        if prim in placements:
            print(f"approx center: {placements[prim]}")
        if matches:
            for m in matches:
                print(f"-> {m}")
        else:
            print("-> NOT FOUND")


def render_and_save_image(args: argparse.Namespace, simulation_app: SimulationApp) -> None:
    """Render via sensor camera and save one RGB PNG."""
    _ = simulation_app
    import isaacsim.core.utils.numpy.rotations as rot_utils
    from isaacsim.core.api import World
    try:
        from isaacsim.sensors.camera import Camera
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to import isaacsim.sensors.camera.Camera. "
            "This Isaac Sim camera pipeline depends on Pillow (PIL) via replicator/syntheticdata. "
            "Install Pillow in the Isaac Python environment (./python.sh -m pip install Pillow) and retry."
        ) from exc

    eye = np.array(args.camera_eye, dtype=np.float64)
    target = np.array(args.camera_target, dtype=np.float64)
    euler = _look_at_to_euler(eye, target)
    orientation = rot_utils.euler_angles_to_quats(euler, degrees=True)

    world = World(stage_units_in_meters=1.0)
    world.reset()

    camera = Camera(
        prim_path="/World/RenderCamera",
        position=eye,
        frequency=30,
        resolution=tuple(args.resolution),
        orientation=orientation,
    )
    camera.initialize()

    total_frames = max(args.frames, args.capture_frame + 1)
    for _ in range(max(1, total_frames)):
        world.step(render=True)

    camera.get_current_frame()
    rgba = camera.get_rgba()
    if rgba is None or rgba.shape[-1] < 3:
        raise RuntimeError("Camera did not return a valid RGBA frame.")
    rgb = np.asarray(rgba[:, :, :3])
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0.0, 1.0) * 255.0 if np.issubdtype(rgb.dtype, np.floating) else np.clip(rgb, 0, 255)
        rgb = rgb.astype(np.uint8)
    print(f"Saving the rendered image to: {args.screenshot}")
    Image.fromarray(rgb).save(args.screenshot)


def main() -> None:
    args = parse_args()
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)

    initialize_random_seed(args.seed)

    simulation_app = SimulationApp({"headless": args.headless})

    data = load_and_validate_scene_graph(args.json)
    edges = data.get("edges", {}).get("obj-obj", [])
    print(f"Scanning USD files under: {args.asset_root}")
    usd_paths = discover_usd_assets(args.asset_root)
    print(f"Total USD files found: {len(usd_paths)}")
    import time
    start_time = time.time()
    asset_match_lookup = build_asset_match_lookup(data, usd_paths)
    asset_bbox_lookup = build_asset_bbox_lookup(data, usd_paths, asset_match_lookup)
    room_usd_path: Optional[Path] = None
    if args.generate_room:
        room_usd_path = args.room_usd or (args.json.parent / "generated_room.usd")
        generate_room_usd_from_scene(
            scene_data=data,
            output_usd=room_usd_path,
            floor_z=args.plane_height,
            include_ceiling=args.room_include_ceiling,
            include_front_wall=args.room_include_front_wall,
            texture_dir=args.room_texture_dir,
        )
        print(f"[ROOM] generated room USD: {room_usd_path}")

    object_entries, placements = _build_entries_from_scene_edges(
        data, args.plane_height, args.spread_scale, args.grid_step, asset_bbox_lookup
    )
    end_time = time.time()
    print(f"Time to build entries: {end_time - start_time:.2f}s")
    print_layout_matches(object_entries, placements, usd_paths)

    build_stage_from_entries(
        object_entries,
        usd_paths,
        edges,
        args.save_usd,
        args.plane_size,
        args.plane_height,
        asset_match_lookup=asset_match_lookup,
        room_usd=room_usd_path,
        use_default_ground=(not args.no_default_ground and room_usd_path is None),
    )
    render_and_save_image(args, simulation_app)

    simulation_app.close()


if __name__ == "__main__":
    main()
