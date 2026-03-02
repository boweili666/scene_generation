"""
FastAPI service wrapper: keep Isaac Sim alive and rebuild/sample scenes on demand
without restarting the simulator each time.
Run example:
    python server/scene_service.py --host 0.0.0.0 --port 8001 --headless
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
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Read project config to avoid hard-coded absolute paths.
try:
    from .config import (
        BASE_DIR,
        DEFAULT_PLACEMENTS_PATH as CFG_DEFAULT_PLACEMENTS_PATH,
        DEFAULT_RENDER_PATH as CFG_DEFAULT_RENDER_PATH,
        GENMESH_ROOT as CFG_GENMESH_ROOT,
        SCENE_GRAPH_PATH as CFG_SCENE_GRAPH_PATH,
    )
except ImportError:
    from config import (
        BASE_DIR,
        DEFAULT_PLACEMENTS_PATH as CFG_DEFAULT_PLACEMENTS_PATH,
        DEFAULT_RENDER_PATH as CFG_DEFAULT_RENDER_PATH,
        GENMESH_ROOT as CFG_GENMESH_ROOT,
        SCENE_GRAPH_PATH as CFG_SCENE_GRAPH_PATH,
    )

# Import isaac_local/scripts/save_figure.py
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(BASE_DIR).resolve()
ISAAC_SCRIPT_DIR = PROJECT_ROOT / "isaac_local" / "scripts"
import sys

for candidate in (THIS_DIR, ISAAC_SCRIPT_DIR):
    if str(candidate) not in sys.path:
        sys.path.append(str(candidate))

# pylint: disable=wrong-import-position
from isaacsim import SimulationApp
import save_figure as sf

# Default paths aligned with this repo's config.py.
DEFAULT_SCENE_GRAPH_PATH = str(CFG_SCENE_GRAPH_PATH)
DEFAULT_PLACEMENTS_PATH: Optional[str] = str(CFG_DEFAULT_PLACEMENTS_PATH)
DEFAULT_SCREENSHOT_PATH: Optional[str] = str(CFG_DEFAULT_RENDER_PATH)


class SceneRequest(BaseModel):
    scene_graph_path: Optional[str] = Field(
        default=DEFAULT_SCENE_GRAPH_PATH,
        description="Local JSON file path; provide either this or scene_graph. Uses built-in default if omitted.",
    )
    scene_graph: Optional[Dict] = Field(
        None, description="Inline scene graph object; provide either this or scene_graph_path."
    )
    asset_root: str = Field(
        default="/home/lbw/3dgen-project/scene_graph_ui_test/isaac_local/my_viewer/test_usd",
        description="USD asset root directory for matching object names.",
    )
    plane_size: float = 10.0
    plane_height: float = 0.0
    spread_scale: float = 0.45
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
        default=None, description="Export stage to this path if provided."
    )
    frames: int = 15
    capture_frame: int = 5
    camera_eye: List[float] = Field(default_factory=lambda: [30.0, 0.0, 30.0])
    camera_target: List[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0])
    camera_euler: List[float] = Field(default_factory=lambda: [0.0, 90.0, 0.0])
    resolution: List[int] = Field(default_factory=lambda: [1280, 720])
    screenshot: Optional[str] = Field(
        default=DEFAULT_SCREENSHOT_PATH, description="Save PNG screenshot to this path if provided."
    )
    seed: Optional[int] = None


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
            usd_cache[key] = sf.collect_usd_paths(root)
        return usd_cache[key]

    def _load_scene_graph(req: SceneRequest) -> Dict:
        if not req.scene_graph_path and not req.scene_graph:
            raise HTTPException(status_code=400, detail="Provide at least one of scene_graph_path or scene_graph.")
        data = (
            sf.load_scene_graph(Path(req.scene_graph_path))
            if req.scene_graph_path
            else req.scene_graph
        )
        data = sf.normalize_scene_graph(data)
        if req.unit.lower() == "cm":
            data = sf.scale_scene_graph_units(data, "cm")
        return data

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

    def _calc_asset_bbox(data: Dict, usd_paths: List[Path]) -> Dict[str, Dict]:
        lookup: Dict[str, Dict] = {}
        for prim, meta in data.get("obj", {}).items():
            label = meta.get("class") or meta.get("class_name") or Path(prim).name
            match = sf.first_match(label, usd_paths) or sf.first_match(Path(prim).name, usd_paths)
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
        usd_paths = _get_usd_paths(Path(req.asset_root))
        asset_bbox = _calc_asset_bbox(data, usd_paths)
        edges = data.get("edges", {}).get("obj-obj", [])
        placement_overrides = _load_placements_dict(req.placements_path, req.placements) if use_overrides else {}

        seed = req.seed or random.SystemRandom().randrange(0, 2**32 - 1)
        random.seed(seed)
        np.random.seed(seed)

        object_entries, placements = sf._build_entries_from_edges(  # noqa: SLF001
            data, req.plane_height, req.spread_scale, asset_bbox
        )
        if placement_overrides:
            updated = _apply_placement_overrides(object_entries, placement_overrides, placements)
            placements.update(updated)
        # Backfill yaw from transform when missing to keep exported placements complete.
        _inject_yaw_from_transforms(placements, object_entries)

        sf.rebuild_stage(
            object_entries,
            usd_paths,
            edges,
            Path(req.save_usd) if req.save_usd else None,
            req.plane_size,
            req.plane_height,
        )

        # Camera and rendering.
        eye = np.array(req.camera_eye, dtype=np.float64)
        target = np.array(req.camera_target, dtype=np.float64)
        euler = sf._eye_target_to_euler(eye, target) if req.camera_target else np.array(req.camera_euler)  # noqa: SLF001
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
            # JSON does not support tuples; convert to lists. Include yaw when present.
            serializable = {}
            for k, v in placements.items():
                if len(v) >= 4 and v[3] is not None:
                    serializable[k] = [v[0], v[1], v[2], v[3]]
                else:
                    serializable[k] = [v[0], v[1], v[2]]
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)

        return {
            "seed": seed,
            "placements": placements,
            "objects": object_entries,
            "usd_files": len(usd_paths),
            "edges": len(edges),
            "screenshot_saved": screenshot_saved,
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
            data = sf.normalize_scene_graph(req.scene_graph)
            if req.unit.lower() == "cm":
                data = sf.scale_scene_graph_units(data, "cm")
            objs = data.get("obj", {})
            if not objs:
                raise HTTPException(status_code=400, detail="scene_graph.obj is empty")
            _, placements = sf._build_entries_from_edges(  # noqa: SLF001
                data, req.plane_height, req.spread_scale, None
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
