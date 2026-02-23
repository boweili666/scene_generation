import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from isaacsim import SimulationApp
from PIL import Image

SCENE_GRAPH_UI_ROOT = Path(__file__).resolve().parents[2]


def load_scene_graph(json_path: Path) -> Dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_scene_graph(data: Dict) -> Dict:
    """
    Normalize legacy scene-graph formats (list-based objects/edges) into the
    dict layout expected by the rest of this script.
    """
    normalized = dict(data)

    # Convert ["objects": [...]] into {"obj": {...}} keyed by prim path.
    obj = normalized.get("obj")
    if not isinstance(obj, dict):
        objects = normalized.get("objects", [])
        obj_map = {}
        if isinstance(objects, list):
            for item in objects:
                if not isinstance(item, dict):
                    continue
                prim = item.get("path") or item.get("prim") or item.get("name")
                if not prim:
                    continue
                meta = {}
                meta["class"] = item.get("class") or item.get("class_name") or item.get("name")
                for key in ("world_3d_bbox", "3d_bbox", "transform"):
                    if key in item:
                        meta[key] = item[key]
                obj_map[prim] = meta
        normalized["obj"] = obj_map

    # Convert edge lists into {"obj-obj": [...]}; keep existing dicts intact.
    edges = normalized.get("edges")
    if isinstance(edges, list):
        normalized["edges"] = {"obj-obj": edges}
    elif isinstance(edges, dict):
        normalized["edges"] = edges
    else:
        normalized["edges"] = {"obj-obj": []}

    return normalized

def collect_usd_paths(asset_root: Path) -> List[Path]:
    """Recursively gather all USD files under the given asset root."""
    return list(asset_root.rglob("*.usd"))


def find_matches(name: str, usd_paths: Iterable[Path]) -> List[Path]:
    """Return USD paths that contain the given object name (case-insensitive)."""
    needle = name.lower()
    matches: List[Path] = []
    for p in usd_paths:
        if needle in p.name.lower():
            matches.append(p)
    return matches


def first_match(name: str, usd_paths: Iterable[Path]) -> Optional[Path]:
    matches = find_matches(name, usd_paths)
    return matches[0] if matches else None


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


def rebuild_stage(
    object_entries: List[Dict],
    usd_paths: List[Path],
    edges: List[Dict],
    save_usd: Optional[Path],
    plane_size: float,
    plane_height: float,
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
    # Use Isaac Sim World helper to add a physics-ready ground plane instead of a custom quad.
    from isaacsim.core.api import World

    world_for_ground = World(stage_units_in_meters=1.0)
    world_for_ground.scene.add_default_ground_plane(z_position=plane_height)
    _add_default_lighting(stage)

    for entry in object_entries:
        name = entry["name"]
        prim = entry["prim"]
        transform = entry["transform"]

        usd_match = first_match(name, usd_paths)
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


def intersect_range(r1: Tuple[Optional[float], Optional[float]], r2: Tuple[Optional[float], Optional[float]]):
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


def infer_position_region(target_coord, relation, target_object_size, spread_scale: float):
    """
    target_coord: j 的 bbox (xmin, xmax, ymin, ymax, zmin, zmax)
    relation: e.g. "left,near"
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
        "near_dist": None,
        "far_dist": None,
        "center": ((xmin + xmax) / 2, (ymin + ymax) / 2),
    }

    relations = [r.strip() for r in relation.split(",")]

    for rel in relations:
        # ---------------- Left / Right ---------------------
        if rel == "left":
            r_y = (None, ymin - dy)
            region["y_range"] = intersect_range(region["y_range"], r_y)

        if rel == "right":
            r_y = (ymax + dy, None)
            region["y_range"] = intersect_range(region["y_range"], r_y)

        # ---------------- Front / Behind ---------------------
        if rel == "front":
            r_x = (xmax + dx, None)
            region["x_range"] = intersect_range(region["x_range"], r_x)

        if rel == "behind":
            r_x = (None, xmin - dx)
            region["x_range"] = intersect_range(region["x_range"], r_x)

        # ---------------- Near / Far ---------------------
        if rel == "near":
            region["near_dist"] = 5 * max(dx, dy) * spread_scale  # 可调整

        if rel == "far":
            region["far_dist"] = 10 * max(dx, dy) * spread_scale

    return region


def _eye_target_to_euler(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
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


def sample_position_from_region(region, target_object_size, spread_scale: float):
    dx, dy, dz = target_object_size

    # Sample x
    xmin, xmax = region["x_range"]
    if xmin is None:
        xmin = -10.0 * spread_scale
    if xmax is None:
        xmax = 10.0 * spread_scale
    x = random.uniform(xmin, xmax)

    # Sample y
    ymin, ymax = region["y_range"]
    if ymin is None:
        ymin = -10.0 * spread_scale
    if ymax is None:
        ymax = 10.0 * spread_scale
    y = random.uniform(ymin, ymax)

    # Sample z
    zmin, zmax = region["z_range"]
    if zmin is None:
        zmin = 0
    if zmax is None:
        zmax = 2.0
    z = random.uniform(zmin, zmax)

    # distance constraint
    cx, cy = region["center"]
    dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    if region["near_dist"] is not None:
        if dist > region["near_dist"]:
            # resample toward center
            x = cx + (x - cx) * 0.3 * spread_scale
            y = cy + (y - cy) * 0.3 * spread_scale

    if region["far_dist"] is not None:
        if dist < region["far_dist"]:
            # push away
            x = cx + (x - cx) * 2 * spread_scale
            y = cy + (y - cy) * 2 * spread_scale

    return (x, y, z)


def _clamp_pos_to_region(pos: Tuple[float, float, float], region: Dict) -> Tuple[float, float, float]:
    """Clamp a sampled position to the region bounds (ignores near/far)."""
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


def _apply_range_constraint(
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
        out[key] = intersect_range(out[key], (lo_w, hi_w)) or (lo_w, hi_w)
    return out


def _merge_range(base_range: Tuple[Optional[float], Optional[float]], new_range: Tuple[Optional[float], Optional[float]]):
    """Intersect two ranges, falling back to the new range if they do not overlap."""
    merged = intersect_range(base_range, new_range)
    return merged if merged is not None else new_range


def _merge_near(base: Optional[float], new: Optional[float]) -> Optional[float]:
    """Use the stricter (larger) near constraint."""
    if new is None:
        return base
    if base is None:
        return new
    return max(base, new)


def _merge_far(base: Optional[float], new: Optional[float]) -> Optional[float]:
    """Use the stricter (smaller) far constraint."""
    if new is None:
        return base
    if base is None:
        return new
    return min(base, new)


def _aggregate_constraints(
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
        "near_dist": None,
        "far_dist": None,
        "center": (0.0, 0.0),
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
        candidate = infer_position_region(target_coord, edge["relation"], size_lookup[source], spread_scale)
        if "range" in edge:
            candidate = _apply_range_constraint(candidate, edge["range"], target_center)

        region["x_range"] = _merge_range(region["x_range"], candidate["x_range"])
        region["y_range"] = _merge_range(region["y_range"], candidate["y_range"])
        region["z_range"] = _merge_range(region["z_range"], candidate["z_range"])
        region["near_dist"] = _merge_near(region["near_dist"], candidate["near_dist"])
        region["far_dist"] = _merge_far(region["far_dist"], candidate["far_dist"])
        centers.append(candidate["center"])

    if centers:
        cx = sum(c[0] for c in centers) / len(centers)
        cy = sum(c[1] for c in centers) / len(centers)
        region["center"] = (cx, cy)

    return region, constraint_count


def _has_vertical_support(prim: str, edges: List[Dict]) -> bool:
    """Return True if any edge for prim implies it should sit on top of another object."""
    for edge in edges:
        if edge["source"] != prim:
            continue
        rel = edge["relation"].lower()
        if "supported by" in rel or "on top" in rel or "on " in rel:
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
        rel = edge["relation"].lower()
        if "supported by" not in rel and "on top" not in rel and "on " not in rel:
            continue
        source = edge["source"]
        target = edge["target"]
        if source not in placements or target not in placements:
            continue
        tx, ty, tz = placements[target]
        print(placements[target])
        _, _, tdz = size_lookup[target]
        print(size_lookup[target])
        _, _, sdz = size_lookup[source]
        print(size_lookup[source])
        target_top = tz + tdz * 0.5
        placements[source] = (placements[source][0], placements[source][1], target_top + sdz * 0.5)
        print(placements[source])


def _align_support_bottom_to_obb_top(stage, edges: List[Dict]) -> None:
    """
    Move supported objects so their bottom center lies within the supporter top OBB.
    Uses world-space oriented bounding boxes from UsdGeom.BBoxCache.
    """
    from pxr import Gf, Usd, UsdGeom

    processed_sources = set()
    for edge in edges:
        rel = edge["relation"].lower()
        if "supported by" not in rel and "on top" not in rel and "on " not in rel:
            continue

        source_path = edge["source"]
        target_path = edge["target"]

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
        # Express the source bottom center in the target OBB frame.
        bottom_in_target = t_mat.GetInverse().Transform(bottom_world)
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
        # Sample a point on the target top surface inside the inset OBB footprint.
        sample_x = random.uniform(x_low, x_high)
        sample_y = random.uniform(y_low, y_high)
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

def _resolve_overlaps(
    placements: Dict[str, Tuple[float, float, float]],
    size_lookup: Dict[str, Tuple[float, float, float]],
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

    prims = list(placements.keys())

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


def _compute_size(meta: Dict) -> Tuple[float, float, float]:
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


def _compute_unit_scale(size: Tuple[float, float, float]) -> float:
    """
    Return a uniform scale that fits the bbox inside a 10 x 10 x 10 cube
    (i.e., longest side becomes 10 units).
    """
    longest = max(size)
    target = 1
    return target / longest if longest > 0.0 else 1.0


def _bbox_from_center(center: Tuple[float, float, float], size: Tuple[float, float, float]):
    """Create (xmin, xmax, ymin, ymax, zmin, zmax) from center + size."""
    cx, cy, cz = center
    dx, dy, dz = size
    hx, hy, hz = dx * 0.5, dy * 0.5, dz * 0.5
    return (cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz)


def _build_entries_from_edges(
    data: Dict,
    plane_height: float,
    spread_scale: float,
    asset_bbox_lookup: Optional[Dict[str, Dict[str, Tuple[float, float, float]]]] = None,
) -> Tuple[List[Dict], Dict[str, Tuple[float, float, float]]]:
    """
    Use edge relations to approximate object positions instead of exact transforms,
    anchoring one node at a time. Start from a central anchor, sample floating
    positions for the rest based on currently-locked neighbors, then lock the next
    most-constrained node and repeat.

    Returns object entries (prim, name, transform) and a dict of final centers.
    """
    objs = data.get("obj", {})
    edges: List[Dict] = data.get("edges", {}).get("obj-obj", [])
    if not objs:
        return [], {}

    # All objects are normalized to fit inside a 1x1x1 box before sampling/placement.
    size_lookup: Dict[str, Tuple[float, float, float]] = {}  # scaled sizes
    center_lookup: Dict[str, Tuple[float, float, float]] = {}  # scaled local centers
    scale_lookup: Dict[str, float] = {}
    for prim, meta in objs.items():
        if asset_bbox_lookup and prim in asset_bbox_lookup:
            info = asset_bbox_lookup[prim]
            raw_size = info["size"]
            raw_center = info["center"]
        else:
            raw_size = _compute_size(meta)
            raw_center = (0.0, 0.0, 0.0)

        unit_scale = _compute_unit_scale(raw_size)
        scale_lookup[prim] = unit_scale
        size_lookup[prim] = tuple(s * unit_scale for s in raw_size)
        center_lookup[prim] = tuple(c * unit_scale for c in raw_center)
    placements: Dict[str, Tuple[float, float, float]] = {}

    # Anchor the first object at the world center, resting on the ground plane.
    anchor_prim = next(iter(objs))
    anchor_size = size_lookup[anchor_prim]
    placements[anchor_prim] = (0.0, 0.0, plane_height + anchor_size[2] * 0.5)

    unlocked = set(objs.keys()) - {anchor_prim}
    max_iters = max(1, len(objs) * 2)

    # Iteratively lock one more node at a time, resampling floating objects using
    # edges that point to already-locked nodes.
    last_floating: Dict[str, Tuple[float, float, float]] = {}
    while unlocked and max_iters > 0:
        floating: Dict[str, Tuple[float, float, float]] = {}
        constraint_counts: Dict[str, int] = {}

        for prim in unlocked:
            region, count = _aggregate_constraints(prim, placements, edges, size_lookup, plane_height, spread_scale)
            if count == 0:
                size = size_lookup[prim]
                region = {
                    "x_range": (-10.0 * spread_scale, 10.0 * spread_scale),
                    "y_range": (-10.0 * spread_scale, 10.0 * spread_scale),
                    "z_range": (plane_height + size[2] * 0.5, plane_height + size[2] * 0.5),
                    "near_dist": None,
                    "far_dist": None,
                    "center": (0.0, 0.0),
                }

            print(
                f"[REGION] {prim} x:{region['x_range']} y:{region['y_range']} "
                f"z:{region['z_range']} near:{region['near_dist']} far:{region['far_dist']} "
                f"center:{region['center']}"
            )
            raw_pos = sample_position_from_region(region, size_lookup[prim], spread_scale)
            pos = _clamp_pos_to_region(raw_pos, region)
            print(f"  sampled: {raw_pos} -> clamped: {pos}")
            floating[prim] = (pos[0], pos[1], pos[2])
            constraint_counts[prim] = count

        last_floating = floating
        # Choose the next node to lock: prefer the one with the most constraints
        # to already-locked nodes to stabilize the layout sooner.
        prim_to_lock = sorted(unlocked, key=lambda p: (-constraint_counts.get(p, 0), p))[0]
        pos_to_lock = floating.get(prim_to_lock) or last_floating.get(prim_to_lock)
        if pos_to_lock is None:
            size = size_lookup[prim_to_lock]
            pos_to_lock = (0.0, 0.0, plane_height + size[2] * 0.5)
        placements[prim_to_lock] = pos_to_lock
        unlocked.remove(prim_to_lock)
        max_iters -= 1

    # If something went wrong and nodes remain unlocked, fall back to their last
    # sampled floating pose.
    for prim in list(unlocked):
        fallback_pos = last_floating.get(prim)
        if fallback_pos is None:
            size = size_lookup[prim]
            fallback_pos = (0.0, 0.0, plane_height + size[2] * 0.5)
        placements[prim] = fallback_pos
        unlocked.remove(prim)

    # Clamp to ground for objects without vertical support relations.
    for prim in placements:
        if _has_vertical_support(prim, edges):
            continue
        size = size_lookup[prim]
        ground_z = plane_height + size[2] * 0.5
        cx, cy, cz = placements[prim]
        placements[prim] = (cx, cy, ground_z)

    # Snap supported objects so bbox min z == target bbox max z.
    # _enforce_support_alignment(placements, edges, size_lookup)
    # Ensure no bounding boxes overlap; push apart in XY as needed.
    _resolve_overlaps(placements, size_lookup)

    entries = []
    for prim, meta in objs.items():
        label = meta.get("class") or meta.get("class_name") or Path(prim).name
        cx, cy, cz = placements[prim]
        ox, oy, oz = center_lookup.get(prim, (0.0, 0.0, 0.0))
        # Randomize initial yaw so objects do not all face the same direction.
        scale = scale_lookup.get(prim, 1.0)
        yaw = random.uniform(-math.pi, math.pi)
        c = math.cos(yaw)
        s = math.sin(yaw)
        rot = [
            [c * scale, -s * scale, 0.0],
            [s * scale, c * scale, 0.0],
            [0.0, 0.0, scale],
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


def main() -> None:
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
        default=Path("/home/lbw/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/manu_converted_wrapped"),
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
        default=10,
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
        "--spread-scale",
        type=float,
        default=0.5,
        help="Scale positional sampling and near/far distances; <1 makes layout tighter.",
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
    parser.set_defaults(headless=True)
    args = parser.parse_args()
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)

    # Seed randomness; default to a system-derived seed for non-deterministic runs.
    system_rng = random.SystemRandom()
    seed = args.seed if args.seed is not None else system_rng.randrange(0, 2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    print(f"[seed] {seed}")

    simulation_app = SimulationApp({"headless": args.headless})

    # Imports that rely on SimulationApp initialization.
    import isaacsim.core.utils.numpy.rotations as rot_utils
    from isaacsim.core.api import World
    from isaacsim.sensors.camera import Camera

    data = load_scene_graph(args.json)
    data = normalize_scene_graph(data)
    edges = data.get("edges", {}).get("obj-obj", [])
    print(f"Scanning USD files under: {args.asset_root}")
    usd_paths = collect_usd_paths(args.asset_root)
    print(f"Total USD files found: {len(usd_paths)}")

    # Precompute asset bbox size/center for more accurate placement and baked recenter.
    asset_bbox_lookup: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    for prim, meta in data.get("obj", {}).items():
        label = meta.get("class") or meta.get("class_name") or Path(prim).name
        usd_match = first_match(label, usd_paths) or first_match(Path(prim).name, usd_paths)
        if not usd_match:
            continue
        info = _compute_usd_bbox_info(usd_match)
        if info:
            asset_bbox_lookup[prim] = info

    object_entries, placements = _build_entries_from_edges(
        data, args.plane_height, args.spread_scale, asset_bbox_lookup
    )

    print("\nMatches (edge-based placement):")
    for entry in object_entries:
        name = entry["name"]
        prim = entry["prim"]
        transform = entry["transform"]
        matches = find_matches(name, usd_paths)

        print(f"\n{name} (prim: {prim})")
        print(f"transform: {transform}")
        if prim in placements:
            print(f"approx center: {placements[prim]}")
        if matches:
            for m in matches:
                print(f"-> {m}")
        else:
            print("-> NOT FOUND")

    
    rebuild_stage(object_entries, usd_paths, edges, args.save_usd, args.plane_size, args.plane_height)

    # Set up a sensor camera (no viewport capture). Uses a look-at orientation derived from camera_eye/target.
    eye = np.array(args.camera_eye, dtype=np.float64)
    target = np.array(args.camera_target, dtype=np.float64)
    euler = _eye_target_to_euler(eye, target) if args.camera_target else np.array(args.camera_euler, dtype=np.float64)
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
    img = Image.fromarray(camera.get_rgba()[:, :, :3])
    print(f"Saving the rendered image to: {args.screenshot}")
    img.save(args.screenshot)

    simulation_app.close()


if __name__ == "__main__":
    main()
