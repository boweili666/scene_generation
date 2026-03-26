"""
FastAPI service wrapper: keep Isaac Sim alive and rebuild/sample scenes on demand
without restarting the simulator each time.
Run example:
    python -m app.backend.services.scene_service --host 0.0.0.0 --port 8001 --headless
Request example:
    curl -X POST http://localhost:8001/scene \
         -H "Content-Type: application/json" \
         -d '{"scene_graph_path":"/path/to/graph.json","save_usd":"/tmp/out.usd"}'
"""

import argparse
import asyncio
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from ..config import (
    BASE_DIR,
    DEFAULT_PLACEMENTS_PATH as CFG_DEFAULT_PLACEMENTS_PATH,
    DEFAULT_RENDER_PATH as CFG_DEFAULT_RENDER_PATH,
    GENMESH_ROOT as CFG_GENMESH_ROOT,
    ISAAC_ASSET_ROOT as CFG_ISAAC_ASSET_ROOT,
    REAL2SIM_MANIFEST_PATH as CFG_REAL2SIM_MANIFEST_PATH,
    RETRIEVAL_ASSET_ROOT as CFG_RETRIEVAL_ASSET_ROOT,
    SCENE_GRAPH_PATH as CFG_SCENE_GRAPH_PATH,
    SCENE_SERVICE_USD_DIR as CFG_SCENE_SERVICE_USD_DIR,
)

# Import pipelines/isaac/scene_renderer.py
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(BASE_DIR).resolve()
ISAAC_SCRIPT_DIR = PROJECT_ROOT / "pipelines" / "isaac"
import sys

for candidate in (THIS_DIR, ISAAC_SCRIPT_DIR):
    if str(candidate) not in sys.path:
        sys.path.append(str(candidate))

# pylint: disable=wrong-import-position
from isaacsim import SimulationApp
import scene_renderer as sf
import resample_modes as rm

# Default paths aligned with this repo's config.py.
DEFAULT_SCENE_GRAPH_PATH = str(CFG_SCENE_GRAPH_PATH)
DEFAULT_PLACEMENTS_PATH: Optional[str] = str(CFG_DEFAULT_PLACEMENTS_PATH)
DEFAULT_SCREENSHOT_PATH: Optional[str] = str(CFG_DEFAULT_RENDER_PATH)
DEFAULT_SCENE_USD_PATH: str = str((CFG_SCENE_SERVICE_USD_DIR / "scene_latest.usd").resolve())
DEFAULT_ROOM_USD_PATH: str = str((CFG_SCENE_SERVICE_USD_DIR / "generated_room.scene_service.usd").resolve())


class SceneRequest(BaseModel):
    scene_graph_path: Optional[str] = Field(
        default=DEFAULT_SCENE_GRAPH_PATH,
        description="Local JSON file path; provide either this or scene_graph. Uses built-in default if omitted.",
    )
    scene_graph: Optional[Dict] = Field(
        None, description="Inline scene graph object; provide either this or scene_graph_path."
    )
    asset_root: str = Field(
        default=str(CFG_ISAAC_ASSET_ROOT),
        description="Legacy fallback USD asset root for class-name matching when source is missing.",
    )
    retrieval_asset_root: str = Field(
        default=str(CFG_RETRIEVAL_ASSET_ROOT),
        description="USD asset root directory used for source=retrieval matching.",
    )
    real2sim_manifest_path: Optional[str] = Field(
        default=str(CFG_REAL2SIM_MANIFEST_PATH),
        description="Manifest generated from the latest Real2Sim outputs for source=real2sim matching.",
    )
    resample_mode: Literal["joint", "lock_real2sim"] = Field(
        default="joint",
        description="Scene resampling mode: joint samples all objects; lock_real2sim keeps real2sim support chains rigid and only samples their unsupported roots.",
    )
    plane_size: float = 10.0
    plane_height: float = 0.0
    default_ground_z_offset: float = -0.01
    spread_scale: float = 0.45
    use_default_ground: bool = True
    generate_room: bool = False
    room_include_back_wall: bool = True
    room_include_left_wall: bool = True
    room_include_right_wall: bool = True
    room_include_front_wall: bool = False
    unit: str = Field(default="m", description="Scene graph length unit: cm or m.")
    placements: Optional[Dict[str, List[float]]] = Field(
        default=None,
        description="Optional pose overrides for selected prims (prim_path -> [x,y,z] or [x,y,z,yaw_deg]).",
    )
    placements_path: Optional[str] = Field(
        default=DEFAULT_PLACEMENTS_PATH,
        description="Optional JSON path to load placements (same format as placements, supports yaw).",
    )
    save_usd: Optional[str] = Field(
        default=DEFAULT_SCENE_USD_PATH,
        description="Export the full generated stage to this path.",
    )
    frames: int = 15
    capture_frame: int = 5
    camera_eye: List[float] = Field(default_factory=lambda: [18.0, 0.0, 18.0])
    camera_target: List[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0])
    camera_euler: List[float] = Field(default_factory=lambda: [0.0, 90.0, 0.0])
    resolution: List[int] = Field(default_factory=lambda: [1280, 720])
    screenshot: Optional[str] = Field(
        default=DEFAULT_SCREENSHOT_PATH, description="Save PNG screenshot to this path if provided."
    )
    seed: Optional[int] = None
    max_layout_attempts: int = Field(default=5, ge=1, le=20)


class SampleRequest(BaseModel):
    scene_graph: Dict
    plane_height: float = 0.0
    spread_scale: float = 0.45
    unit: str = "m"


class ApplyRequest(BaseModel):
    usd_path: str = Field(..., description="Input USD path.")
    placements: Optional[Dict[str, List[float]]] = Field(
        default=None, description="prim_path -> [x,y,z] or [x,y,z,yaw_deg]; provide either this or placements_path."
    )
    placements_path: Optional[str] = Field(
        default=None, description="Load placements from file; provide either this or placements."
    )
    out_path: Optional[str] = Field(
        default=None, description="Output USD path; overwrites input if omitted."
    )
    missing_ok: bool = Field(
        default=True, description="Whether to skip missing prims; otherwise raise an error."
    )


def _create_app(sim_app: SimulationApp) -> FastAPI:
    """
    Build FastAPI app and keep Isaac Sim state inside closure variables.
    """
    import isaacsim.core.utils.numpy.rotations as rot_utils
    from isaacsim.core.api import World
    from isaacsim.sensors.camera import Camera
    import omni.usd

    app = FastAPI(title="IsaacSim Scene Service", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    state_lock = asyncio.Lock()
    usd_cache: Dict[str, List[Path]] = {}
    start_time = time.time()
    world: Optional[World] = None

    def _get_usd_paths(root: Path) -> List[Path]:
        key = str(root.resolve())
        if key not in usd_cache:
            usd_cache[key] = sf.discover_usd_assets(root)
        return usd_cache[key]

    def _scale_scene_dimensions_to_m(data: Dict) -> Dict:
        dims = data.get("scene", {}).get("dimensions", {})
        for k in ("length", "width", "height"):
            v = dims.get(k)
            if isinstance(v, (int, float)):
                dims[k] = float(v) / 100.0
        dims["unit"] = "m"
        return data

    def _load_scene_graph(req: SceneRequest) -> Dict:
        if not req.scene_graph_path and not req.scene_graph:
            raise HTTPException(status_code=400, detail="Provide at least one of scene_graph_path or scene_graph.")
        if req.scene_graph_path:
            data = sf.load_and_validate_scene_graph(Path(req.scene_graph_path))
        else:
            data = sf.validate_and_prepare_scene_graph(req.scene_graph)
        if req.unit.lower() == "cm":
            data = _scale_scene_dimensions_to_m(data)
        return data

    def _resolve_scene_save_path(req: SceneRequest) -> Path:
        save_path = Path(req.save_usd) if req.save_usd else Path(DEFAULT_SCENE_USD_PATH)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        return save_path

    def _resolve_room_usd_path() -> Path:
        room_root = Path(DEFAULT_ROOM_USD_PATH)
        room_root.parent.mkdir(parents=True, exist_ok=True)
        for legacy_path in room_root.parent.glob("generated_room.scene_service*.usd"):
            if legacy_path.is_file():
                legacy_path.unlink()
        return room_root.parent / f"{room_root.stem}.{time.time_ns()}.usd"

    def _load_placements_dict(path: Optional[str], inline: Optional[Dict]) -> Dict[str, Tuple[float, float, float, Optional[float]]]:
        if not path and not inline:
            return {}
        payload = inline
        if path:
            p = Path(path)
            if not p.exists():
                # If file does not exist, create an empty one and continue with empty overrides.
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("{}", encoding="utf-8")
                return {}
            with p.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        placements: Dict[str, Tuple[float, float, float, Optional[float]]] = {}
        if isinstance(payload, dict):
            for prim, val in payload.items():
                if isinstance(val, (list, tuple)) and len(val) in (3, 4):
                    yaw = float(val[3]) if len(val) == 4 else None
                    placements[prim] = (float(val[0]), float(val[1]), float(val[2]), yaw)
                elif isinstance(val, dict) and all(k in val for k in ("x", "y", "z")):
                    yaw = float(val["yaw"]) if "yaw" in val else None
                    placements[prim] = (float(val["x"]), float(val["y"]), float(val["z"]), yaw)
        return placements

    def _apply_yaw_about_center(
        mat: List[List],
        yaw_deg: float,
        placement_center: Optional[Tuple[float, float, float]],
        base_center: Optional[Tuple[float, float, float]],
    ) -> List[List]:
        """
        When updating yaw, keep the mesh center at placement_center to avoid orbiting
        around the world origin.

        Args:
        - mat: Original 4x4 transform matrix (includes previous rotation + translation).
        - yaw_deg: New yaw in degrees.
        - placement_center: Target center position (usually xyz from override placements).
        - base_center: Original center position (from sampled placements), used to infer local pivot.
        """

        mat_np = np.array(mat, dtype=float)
        R0 = mat_np[:3, :3]
        t0 = mat_np[3, :3]

        # Desired center: prefer override xyz, otherwise fallback to original center.
        p_target = np.array(placement_center if placement_center is not None else base_center, dtype=float)

        # Extract uniform scale (assumes same scale on X/Y). Fallback to diagonal value on failure.
        try:
            scale = float(np.linalg.norm(R0[:, 0]))
            if scale == 0:
                raise ValueError
        except Exception:
            scale = mat[2][2] if len(mat) >= 3 and len(mat[2]) >= 3 else 1.0

        # Infer local geometric center c_local such that p_base = R0 * c_local + t0.
        c_local = None
        if base_center is not None and p_target is not None:
            p_base = np.array(base_center, dtype=float)
            try:
                c_local = np.linalg.solve(R0, p_base - t0)
            except Exception:
                pass

        yaw_rad = math.radians(yaw_deg)
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        R1 = np.array([[c * scale, -s * scale, 0.0], [s * scale, c * scale, 0.0], [0.0, 0.0, scale]])

        if c_local is not None and p_target is not None:
            t1 = p_target - R1 @ c_local
        else:
            # If pivot inference fails, keep translation and only replace rotation (legacy behavior).
            t1 = t0

        mat_np[:3, :3] = R1
        mat_np[3, :3] = t1
        return mat_np.tolist()

    def _inject_yaw_from_transforms(
        placements: Dict[str, Tuple[float, float, float, Optional[float]]],
        object_entries: List[Dict],
    ) -> None:
        """
        If placements do not include yaw, infer yaw from transform matrix (Z-axis rotation, degrees).
        Only updates when yaw is missing to avoid overriding external inputs.
        """
        for entry in object_entries:
            prim = entry.get("prim")
            if prim not in placements:
                continue
            # Skip when yaw already exists.
            existing = placements[prim]
            if len(existing) >= 4 and existing[3] is not None:
                continue
            mat = entry.get("transform")
            if not mat or len(mat) < 2 or len(mat[0]) < 2 or len(mat[1]) < 1:
                continue
            # yaw = atan2(sin, cos) = atan2(mat[1][0], mat[0][0])
            try:
                yaw_deg = math.degrees(math.atan2(mat[1][0], mat[0][0]))
            except Exception:
                continue
            tx, ty, tz = existing[:3]
            placements[prim] = (tx, ty, tz, yaw_deg)

    def _apply_placement_overrides(
        object_entries: List[Dict],
        overrides: Dict[str, Tuple[float, float, float, Optional[float]]],
        base_placements: Dict[str, Tuple[float, float, float, Optional[float]]],
    ) -> Dict[str, Tuple[float, float, float, Optional[float]]]:
        """
        Override sampled placements with external placements. If yaw is provided,
        rotate around object center to avoid orbiting around world origin.
        """
        updated = {}
        for entry in object_entries:
            prim = entry["prim"]
            if prim not in overrides:
                continue
            tx, ty, tz, yaw = overrides[prim]

            # New/original center used for pivot-preserving rotation.
            base_center = base_placements.get(prim)[:3] if prim in base_placements else None
            placement_center = (tx, ty, tz)

            mat = entry["transform"]
            if yaw is not None:
                mat = _apply_yaw_about_center(mat, yaw, placement_center, base_center)
            else:
                mat = [row[:] for row in mat]
                mat[3][0], mat[3][1], mat[3][2] = tx, ty, tz

            entry["transform"] = mat
            updated[prim] = (tx, ty, tz, yaw)
        return updated

    def _support_realign_skip_prims(
        layout_debug: Dict[str, object],
        placement_overrides: Dict[str, Tuple[float, float, float, Optional[float]]],
    ) -> set[str]:
        skip_prims = {
            str(prim)
            for prim in (layout_debug.get("locked_real2sim_children") or [])
            if isinstance(prim, str)
        }
        skip_prims.update(str(prim) for prim in placement_overrides.keys())
        return skip_prims

    def _apply_placements_to_usd(
        usd_path: Path,
        placements: Dict[str, Tuple[float, float, float, Optional[float]]],
        out_path: Optional[Path],
        missing_ok: bool,
    ) -> Path:
        from pxr import Gf, Usd, UsdGeom

        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise HTTPException(status_code=400, detail=f"Failed to open USD: {usd_path}")

        for prim_path, payload in placements.items():
            tx, ty, tz, yaw = payload
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                if missing_ok:
                    print(f"[SKIP] prim not found: {prim_path}")
                    continue
                raise HTTPException(status_code=400, detail=f"Prim does not exist: {prim_path}")

            xformable = UsdGeom.Xformable(prim)
            ops = xformable.GetOrderedXformOps()
            if yaw is None:
                if ops:
                    mat = Gf.Matrix4d(ops[0].Get())
                    mat.SetTranslateOnly(Gf.Vec3d(tx, ty, tz))
                    ops[0].Set(mat)
                else:
                    op = xformable.AddTransformOp()
                    op.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(tx, ty, tz)))
                print(f"[SET] {prim_path} -> ({tx}, {ty}, {tz})")
            else:
                rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw)
                mat = Gf.Matrix4d(rot)
                mat.SetTranslateOnly(Gf.Vec3d(tx, ty, tz))
                if ops:
                    ops[0].Set(mat)
                else:
                    op = xformable.AddTransformOp()
                    op.Set(mat)
                print(f"[SET] {prim_path} -> ({tx}, {ty}, {tz}, yaw={yaw})")

        save_path = out_path or usd_path
        stage.GetRootLayer().Export(str(save_path))
        return save_path

    def _range_to_aabb_payload(rng) -> Dict[str, List[float]]:
        bmin = rng.GetMin()
        bmax = rng.GetMax()
        min_xyz = [float(bmin[0]), float(bmin[1]), float(bmin[2])]
        max_xyz = [float(bmax[0]), float(bmax[1]), float(bmax[2])]
        return {
            "min": min_xyz,
            "max": max_xyz,
            "center": [
                float((min_xyz[0] + max_xyz[0]) * 0.5),
                float((min_xyz[1] + max_xyz[1]) * 0.5),
                float((min_xyz[2] + max_xyz[2]) * 0.5),
            ],
            "size": [
                float(max_xyz[0] - min_xyz[0]),
                float(max_xyz[1] - min_xyz[1]),
                float(max_xyz[2] - min_xyz[2]),
            ],
        }

    def _collect_world_aabbs(stage, prim_paths: List[str]) -> Dict[str, Dict[str, List[float]]]:
        from pxr import Usd, UsdGeom

        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "guide"], useExtentsHint=True)
        aabb_lookup: Dict[str, Dict[str, List[float]]] = {}
        for prim_path in prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if rng.IsEmpty():
                mins: Optional[List[float]] = None
                maxs: Optional[List[float]] = None
                for child in Usd.PrimRange(prim):
                    child_rng = cache.ComputeWorldBound(child).ComputeAlignedRange()
                    if child_rng.IsEmpty():
                        continue
                    child_min = child_rng.GetMin()
                    child_max = child_rng.GetMax()
                    if mins is None:
                        mins = [float(child_min[0]), float(child_min[1]), float(child_min[2])]
                        maxs = [float(child_max[0]), float(child_max[1]), float(child_max[2])]
                    else:
                        mins = [
                            min(mins[0], float(child_min[0])),
                            min(mins[1], float(child_min[1])),
                            min(mins[2], float(child_min[2])),
                        ]
                        maxs = [
                            max(maxs[0], float(child_max[0])),
                            max(maxs[1], float(child_max[1])),
                            max(maxs[2], float(child_max[2])),
                        ]
                if mins is None or maxs is None:
                    continue

                class _FallbackRange:
                    def __init__(self, min_xyz: List[float], max_xyz: List[float]) -> None:
                        self._min = tuple(min_xyz)
                        self._max = tuple(max_xyz)

                    def GetMin(self):
                        return self._min

                    def GetMax(self):
                        return self._max

                rng = _FallbackRange(mins, maxs)
            aabb_lookup[prim_path] = _range_to_aabb_payload(rng)
        return aabb_lookup

    def _collect_world_placements(
        stage,
        prim_paths: List[str],
    ) -> Dict[str, Tuple[float, float, float, Optional[float]]]:
        from pxr import Gf, UsdGeom

        placements: Dict[str, Tuple[float, float, float, Optional[float]]] = {}
        for prim_path in prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            xformable = UsdGeom.Xformable(prim)
            ops = xformable.GetOrderedXformOps()
            if not ops:
                continue
            mat = Gf.Matrix4d(ops[0].Get())
            translate = mat.ExtractTranslation()
            yaw = math.degrees(math.atan2(mat[1][0], mat[0][0]))
            placements[prim_path] = (
                float(translate[0]),
                float(translate[1]),
                float(translate[2]),
                float(yaw),
            )
        return placements

    def _serialize_placements_payload(
        placements: Dict[str, Tuple[float, float, float, Optional[float]]],
        aabb_lookup: Optional[Dict[str, Dict[str, List[float]]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        serializable: Dict[str, Dict[str, Any]] = {}
        for prim in sorted(placements):
            payload = placements[prim]
            item: Dict[str, Any] = {
                "x": float(payload[0]),
                "y": float(payload[1]),
                "z": float(payload[2]),
            }
            if len(payload) >= 4 and payload[3] is not None:
                item["yaw"] = float(payload[3])
            if aabb_lookup and prim in aabb_lookup:
                item["aabb"] = aabb_lookup[prim]
            serializable[prim] = item
        return serializable

    def _calc_asset_bbox(data: Dict, usd_paths: List[Path]) -> Dict[str, Dict]:
        lookup: Dict[str, Dict] = {}
        for prim, meta in data.get("obj", {}).items():
            label = meta.get("class") or meta.get("class_name") or Path(prim).name
            match = sf.first_asset_match(label, usd_paths) or sf.first_asset_match(Path(prim).name, usd_paths)
            if not match:
                continue
            info = sf._compute_usd_bbox_info(match)  # noqa: SLF001
            if info:
                lookup[prim] = info
        return lookup

    def _ensure_camera(stage, eye: np.ndarray, target: np.ndarray, euler: np.ndarray, resolution: Tuple[int, int]):
        """
        Reuse and reset /World/RenderCamera if it already exists; otherwise create it.
        """
        import omni.usd
        from pxr import Gf, UsdGeom

        ctx = omni.usd.get_context()
        prim_path = "/World/RenderCamera"
        prim = ctx.get_stage().GetPrimAtPath(prim_path)
        if prim.IsValid():
            xform = UsdGeom.Xformable(prim)
        else:
            camera = Camera(
                prim_path=prim_path,
                position=eye,
                frequency=30,
                resolution=resolution,
                orientation=rot_utils.euler_angles_to_quats(euler, degrees=True),
            )
            camera.initialize()
            return camera

        quat = rot_utils.euler_angles_to_quats(euler, degrees=True)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(Gf.Matrix4d().SetTranslate(eye))
        xform.AddOrientOp().Set(quat)
        return Camera(
            prim_path=prim_path,
            position=eye,
            frequency=30,
            resolution=resolution,
            orientation=quat,
        )

    async def _rebuild(req: SceneRequest, use_overrides: bool = True) -> Dict:
        nonlocal world
        data = _load_scene_graph(req)
        resample_mode = rm.validate_resample_mode(req.resample_mode)
        room_closed_walls = {
            "behind": req.room_include_back_wall,
            "left": req.room_include_left_wall,
            "right": req.room_include_right_wall,
            "front": req.room_include_front_wall,
        }
        fallback_usd_paths = _get_usd_paths(Path(req.asset_root))
        retrieval_usd_paths = _get_usd_paths(Path(req.retrieval_asset_root))
        combined_usd_paths = list(dict.fromkeys([*retrieval_usd_paths, *fallback_usd_paths]))
        real2sim_manifest = sf.load_real2sim_manifest(
            Path(req.real2sim_manifest_path) if req.real2sim_manifest_path else None
        )
        real2sim_scale_lookup = sf.build_real2sim_uniform_scale_lookup(real2sim_manifest)
        asset_match_lookup = sf.build_asset_match_lookup(
            data,
            fallback_usd_paths,
            retrieval_usd_paths=retrieval_usd_paths,
            real2sim_manifest=real2sim_manifest,
        )
        asset_bbox = sf.build_asset_bbox_lookup(
            data,
            combined_usd_paths,
            asset_match_lookup=asset_match_lookup,
            object_scale_lookup=real2sim_scale_lookup,
        )
        edges = data.get("edges", {}).get("obj-obj", [])
        placement_overrides = _load_placements_dict(req.placements_path, req.placements) if use_overrides else {}

        room_usd_path: Optional[Path] = None
        if req.generate_room:
            room_usd_path = _resolve_room_usd_path()
            room_texture_dir = PROJECT_ROOT / "third_party" / "stable_material"
            try:
                sf.generate_room_usd_from_scene(
                    scene_data=data,
                    output_usd=room_usd_path,
                    floor_z=req.plane_height,
                    include_ceiling=False,
                    include_back_wall=req.room_include_back_wall,
                    include_left_wall=req.room_include_left_wall,
                    include_right_wall=req.room_include_right_wall,
                    include_front_wall=req.room_include_front_wall,
                    texture_dir=room_texture_dir,
                )
            except Exception as e:
                print(f"[WARN] generate_room_usd_from_scene failed, fallback without room: {e}")
                room_usd_path = None

        seed_base = req.seed or random.SystemRandom().randrange(0, 2**32 - 1)
        room_bounds = sf._room_bounds_from_scene(data, req.spread_scale)  # noqa: SLF001
        grid_step = 0.25
        scene_save_path = _resolve_scene_save_path(req)
        seed = seed_base
        object_entries: List[Dict[str, Any]] = []
        placements: Dict[str, Tuple[float, float, float, Optional[float]]] = {}
        layout_debug: Dict[str, object] = {}
        last_layout_error: Optional[sf.LayoutCollisionError] = None
        attempts_used = 0

        for attempt_idx in range(req.max_layout_attempts):
            seed = int((seed_base + attempt_idx) % (2**32 - 1))
            random.seed(seed)
            np.random.seed(seed)

            object_entries, placements = sf._build_entries_from_scene_edges(  # noqa: SLF001
                data,
                req.plane_height,
                req.spread_scale,
                grid_step,
                asset_bbox,
                object_scale_lookup=real2sim_scale_lookup,
                room_closed_walls=room_closed_walls,
            )
            layout_debug = {
                "mode_applied": False,
                "reason": None,
                "real2sim_roots": [],
                "locked_real2sim_children": [],
                "skipped_real2sim_prims": [],
                "missing_manifest_transforms": [],
            }
            if resample_mode == "lock_real2sim":
                object_entries, placements, layout_debug = rm.apply_lock_real2sim_relative_transforms(
                    data,
                    object_entries,
                    placements,
                    asset_bbox_lookup=asset_bbox,
                    real2sim_manifest=real2sim_manifest,
                )
            if placement_overrides:
                updated = _apply_placement_overrides(object_entries, placement_overrides, placements)
                placements.update(updated)
            _inject_yaw_from_transforms(placements, object_entries)
            support_realign_skip_prims = _support_realign_skip_prims(layout_debug, placement_overrides)

            try:
                sf.build_stage_from_entries(
                    object_entries,
                    combined_usd_paths,
                    edges,
                    scene_save_path,
                    req.plane_size,
                    req.plane_height,
                    default_ground_z_offset=req.default_ground_z_offset,
                    asset_match_lookup=asset_match_lookup,
                    room_usd=room_usd_path,
                    use_default_ground=req.use_default_ground,
                    skip_support_realign_prims=support_realign_skip_prims,
                    room_bounds=room_bounds,
                    room_closed_walls=room_closed_walls,
                )
                attempts_used = attempt_idx + 1
                break
            except sf.LayoutCollisionError as e:
                last_layout_error = e
                print(
                    f"[WARN] layout collision on attempt {attempt_idx + 1}/{req.max_layout_attempts} "
                    f"(seed={seed}): {e}"
                )
        else:
            detail: Dict[str, Any] = {
                "message": "failed to generate a collision-free layout after all attempts",
                "attempts": req.max_layout_attempts,
                "seed_base": seed_base,
            }
            if last_layout_error is not None:
                detail["collisions"] = last_layout_error.collisions
            raise HTTPException(status_code=409, detail=detail)

        stage = omni.usd.get_context().get_stage()
        prim_paths = [entry["prim"] for entry in object_entries if isinstance(entry.get("prim"), str)]
        placements = _collect_world_placements(stage, prim_paths)
        placement_aabbs = _collect_world_aabbs(
            stage,
            prim_paths,
        )

        # Camera and rendering.
        eye = np.array(req.camera_eye, dtype=np.float64)
        target = np.array(req.camera_target, dtype=np.float64)
        euler = sf._look_at_to_euler(eye, target) if req.camera_target else np.array(req.camera_euler)  # noqa: SLF001
        resolution = (int(req.resolution[0]), int(req.resolution[1]))

        world = World(stage_units_in_meters=1.0)
        world.reset()
        camera = _ensure_camera(world.stage, eye, target, euler, resolution)
        camera.initialize()

        total_frames = max(req.frames, req.capture_frame + 1)
        for _ in range(total_frames):
            world.step(render=True)

        screenshot_saved = False
        if req.screenshot:
            Path(req.screenshot).parent.mkdir(parents=True, exist_ok=True)
            raw = camera.get_rgba()
            if raw is None or (hasattr(raw, "size") and raw.size == 0):
                print("[WARN] camera.get_rgba() returned empty; skip saving screenshot")
            else:
                img = np.array(raw)
                if img.ndim == 1:
                    expected = resolution[1] * resolution[0] * 4
                    if img.size != expected:
                        print(f"[WARN] camera.get_rgba() size {img.size} != expected {expected}; skip screenshot")
                        img = None
                    else:
                        img = img.reshape((resolution[1], resolution[0], 4))
                if img is not None:
                    if img.shape[-1] == 4:
                        img = img[:, :, :3]
                    from PIL import Image

                    Image.fromarray(img).save(req.screenshot)
                    screenshot_saved = True

        # Persist final placements to default file for easy reuse.
        if DEFAULT_PLACEMENTS_PATH:
            out_path = Path(DEFAULT_PLACEMENTS_PATH)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            serializable = _serialize_placements_payload(placements, placement_aabbs)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)

        resolved_assets = []
        resolved_counts = {"real2sim": 0, "retrieval": 0, "fallback": 0, "missing": 0}
        locked_children = set(layout_debug.get("locked_real2sim_children", []))
        for prim, meta in data.get("obj", {}).items():
            asset = asset_match_lookup.get(prim)
            resolved_source = asset.source if asset is not None else "missing"
            resolved_counts[resolved_source] = resolved_counts.get(resolved_source, 0) + 1
            resolved_assets.append(
                {
                    "prim": prim,
                    "class": meta.get("class"),
                    "source": meta.get("source"),
                    "resolved_source": resolved_source,
                    "asset_path": str(asset.asset_path) if asset is not None else None,
                    "reference_prim_path": asset.reference_prim_path if asset is not None else None,
                    "sampled": prim not in locked_children,
                    "locked_relative": prim in locked_children,
                }
            )
        resolved_assets.sort(key=lambda item: item["prim"])

        return {
            "seed": seed,
            "resample_mode": resample_mode,
            "placements": _serialize_placements_payload(placements, placement_aabbs),
            "objects": object_entries,
            "usd_files": len(combined_usd_paths),
            "edges": len(edges),
            "saved_usd": str(scene_save_path),
            "screenshot_saved": screenshot_saved,
            "debug": {
                "resample_mode": resample_mode,
                "asset_resolution_counts": resolved_counts,
                "asset_resolution": resolved_assets,
                "layout": layout_debug,
                "layout_attempts": attempts_used,
            },
        }

    @app.get("/health")
    async def health():
        return {"status": "ok", "uptime_sec": time.time() - start_time}

    @app.post("/scene")
    async def scene(req: SceneRequest):
        async with state_lock:
            return await _rebuild(req)

    @app.post("/scene_new")
    async def scene_new(req: SceneRequest):
        """
        Same as /scene, but ignores placements/placements_path and always resamples new placements.
        """
        async with state_lock:
            return await _rebuild(req, use_overrides=False)

    @app.post("/sample")
    async def sample(req: SampleRequest):
        async with state_lock:
            data = sf.validate_and_prepare_scene_graph(req.scene_graph)
            if req.unit.lower() == "cm":
                data = _scale_scene_dimensions_to_m(data)
            objs = data.get("obj", {})
            if not objs:
                raise HTTPException(status_code=400, detail="scene_graph.obj is empty")
            _, placements = sf._build_entries_from_scene_edges(  # noqa: SLF001
                data,
                req.plane_height,
                req.spread_scale,
                0.25,
                None,
                room_closed_walls={"behind": True, "left": True, "right": True, "front": False},
            )
            return {"placements": placements}

    @app.post("/apply_placements")
    async def apply_placements(req: ApplyRequest):
        async with state_lock:
            placements = _load_placements_dict(req.placements_path, req.placements)
            if not placements:
                raise HTTPException(
                    status_code=400,
                    detail="placements is empty; provide placements or placements_path",
                )
            usd_path = Path(req.usd_path)
            out_path = Path(req.out_path) if req.out_path else None
            saved = _apply_placements_to_usd(usd_path, placements, out_path, req.missing_ok)
            return {"updated_usd": str(saved), "count": len(placements)}

    return app


def main():
    parser = argparse.ArgumentParser(description="Keep Isaac Sim alive and drive it via FastAPI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--headless", action="store_true", help="Start SimulationApp in headless mode.")
    parser.add_argument("--windowed", dest="headless", action="store_false", help="Start in windowed mode.")
    parser.set_defaults(headless=True)
    args = parser.parse_args()

    sim_app = SimulationApp({"headless": args.headless})
    app = _create_app(sim_app)

    # Uvicorn blocks; close SimulationApp explicitly on shutdown.
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
