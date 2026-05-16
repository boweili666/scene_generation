"""Per-episode domain randomization for the auto-grasp collection loop.

The auto-grasp pipeline reuses one ranked grasp candidate across many
episodes (see `grasp/target_state.py` for the target-jitter helpers it
already ships). This module adds three further per-episode perturbations
so the recorded dataset covers more of the sim-to-real gap:

* **Light** — multiply the `/World/Light` dome intensity, jitter its
  RGB colour, and (when direction randomization is on) spawn a
  `/World/SunLight` distant light whose orientation + intensity + colour
  are re-sampled every episode.
* **Camera extrinsics** — an independent SE(3) jitter per robot camera
  (head / left_hand / right_hand). Agibot cameras are body-mounted and
  re-posed every physics step by the `_sync` closure in
  `scene_mouse_collect`, so the jitter is published into shared state
  here and *composed inside that closure* — a direct `set_world_poses`
  would be overwritten on the next step.
* **Robot base** — an XY (disk) + yaw offset on the placement plan,
  gated by a workspace-box reachability pre-check so the cached
  world-frame grasp waypoints stay inside the arm's working area. The
  grasp candidate is intentionally *not* shifted: the object stays put
  in the world and the arm must reach it from the new base pose.

All knobs default to 0.0 (disabled) — with a zeroed config every helper
here is a no-op and collection behaves exactly as before.

`pxr` is imported lazily inside the light helper (same convention as
`scene/physics.py`) so this module imports fine without Isaac Sim.
"""

from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from .workspaces import default_robot_workspace_specs

_DOME_LIGHT_PRIM = "/World/Light"
_SUN_LIGHT_PRIM = "/World/SunLight"
_SUN_DEFAULT_INTENSITY = 2000.0
_SUN_DEFAULT_ANGLE_DEG = 1.0
_CAMERA_NAMES = ("head", "left_hand", "right_hand")
_IDENTITY_JITTER = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
_BASE_RESAMPLE_TRIES = 24


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """Per-episode randomization ranges (all 0.0 → that axis disabled)."""

    # Dome + sun intensity is multiplied by a uniform sample in
    # [1 - r, 1 + r]. Colour channels get an additive uniform jitter in
    # [-r, +r] around the base colour, clamped to [0, 1]. Direction adds
    # a `/World/SunLight` distant light re-oriented by ± this many degrees
    # in pitch and yaw each episode.
    light_intensity_randomization: float = 0.0
    light_color_randomization: float = 0.0
    light_direction_randomization_deg: float = 0.0
    # Independent per-camera jitter: translation drawn in a sphere of this
    # radius (m), rotation drawn as a small random rotation of up to this
    # many degrees, expressed in the camera mount's local frame.
    camera_extrinsics_pos_randomization: float = 0.0
    camera_extrinsics_rot_randomization_deg: float = 0.0
    # Robot base offset: XY drawn uniformly in a disk of this radius (m),
    # yaw drawn uniform in [-deg, +deg]. Applied to the placement plan
    # only if the reachability pre-check passes.
    robot_base_xy_randomization: float = 0.0
    robot_base_yaw_randomization_deg: float = 0.0
    # 0 → use the process-global RNG (matches `target_forward_randomization`);
    # non-zero → a dedicated seeded `random.Random` for reproducible runs.
    domain_randomization_seed: int = 0

    @property
    def any_light(self) -> bool:
        return (
            self.light_intensity_randomization > 0.0
            or self.light_color_randomization > 0.0
            or self.light_direction_randomization_deg > 0.0
        )

    @property
    def any_camera(self) -> bool:
        return (
            self.camera_extrinsics_pos_randomization > 0.0
            or self.camera_extrinsics_rot_randomization_deg > 0.0
        )

    @property
    def any_base(self) -> bool:
        return (
            self.robot_base_xy_randomization > 0.0
            or self.robot_base_yaw_randomization_deg > 0.0
        )

    @property
    def enabled(self) -> bool:
        return self.any_light or self.any_camera or self.any_base


def make_rng(cfg: DomainRandomizationConfig) -> random.Random:
    """Dedicated RNG when seeded, else the shared module RNG."""
    if int(cfg.domain_randomization_seed) != 0:
        return random.Random(int(cfg.domain_randomization_seed))
    return random.Random()


# =============================================================================
# Camera extrinsic jitter — shared state consumed by the agibot `_sync` closure
# =============================================================================
_CAMERA_JITTER: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {}


def reset_camera_extrinsic_jitter() -> None:
    _CAMERA_JITTER.clear()


def set_camera_extrinsic_jitter(
    jitter: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]],
) -> None:
    """Publish per-camera (pos_xyz, quat_wxyz) jitter in the mount-local frame."""
    _CAMERA_JITTER.clear()
    _CAMERA_JITTER.update(jitter)


def camera_jitter_active() -> bool:
    return bool(_CAMERA_JITTER)


def get_camera_extrinsic_jitter(
    camera_name: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Return the mount-local jitter for `camera_name` (identity if none)."""
    return _CAMERA_JITTER.get(camera_name, _IDENTITY_JITTER)


# =============================================================================
# Per-episode sampling
# =============================================================================
def _sample_unit_disk(rng: random.Random) -> tuple[float, float]:
    # Uniform over the disk (sqrt keeps the area density flat).
    r = math.sqrt(rng.random())
    theta = rng.uniform(0.0, 2.0 * math.pi)
    return (r * math.cos(theta), r * math.sin(theta))


def _sample_unit_sphere(rng: random.Random) -> tuple[float, float, float]:
    v = np.array([rng.gauss(0.0, 1.0) for _ in range(3)], dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return (0.0, 0.0, 0.0)
    r = rng.random() ** (1.0 / 3.0)
    v = v / n * r
    return (float(v[0]), float(v[1]), float(v[2]))


def _small_random_quat_wxyz(
    rng: random.Random, max_deg: float
) -> tuple[float, float, float, float]:
    if max_deg <= 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    axis = np.array([rng.gauss(0.0, 1.0) for _ in range(3)], dtype=float)
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    axis = axis / n
    angle = math.radians(rng.uniform(-max_deg, max_deg))
    quat_xyzw = R.from_rotvec(axis * angle).as_quat()
    return (
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    )


def _sample_base_offset(
    rng: random.Random, cfg: DomainRandomizationConfig
) -> tuple[float, float, float]:
    base_dx = base_dy = base_dyaw = 0.0
    if cfg.robot_base_xy_randomization > 0.0:
        ux, uy = _sample_unit_disk(rng)
        radius = float(cfg.robot_base_xy_randomization)
        base_dx, base_dy = ux * radius, uy * radius
    if cfg.robot_base_yaw_randomization_deg > 0.0:
        d = float(cfg.robot_base_yaw_randomization_deg)
        base_dyaw = rng.uniform(-d, d)
    return base_dx, base_dy, base_dyaw


@dataclass(frozen=True)
class EpisodeRandomization:
    """The values sampled for one episode (kept around for logging)."""

    light_intensity_factor: float = 1.0
    light_color_delta: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sun_pitch_deg: float = 0.0
    sun_yaw_deg: float = 0.0
    camera_jitter: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = (
        dataclasses.field(default_factory=dict)
    )
    base_dx: float = 0.0
    base_dy: float = 0.0
    base_dyaw_deg: float = 0.0
    base_resampled_tries: int = 0
    base_fell_back: bool = False


def sample_episode_randomization(
    rng: random.Random, cfg: DomainRandomizationConfig
) -> EpisodeRandomization:
    intensity_factor = 1.0
    if cfg.light_intensity_randomization > 0.0:
        r = float(cfg.light_intensity_randomization)
        intensity_factor = rng.uniform(max(0.0, 1.0 - r), 1.0 + r)

    color_delta = (0.0, 0.0, 0.0)
    if cfg.light_color_randomization > 0.0:
        r = float(cfg.light_color_randomization)
        color_delta = (
            rng.uniform(-r, r),
            rng.uniform(-r, r),
            rng.uniform(-r, r),
        )

    sun_pitch = sun_yaw = 0.0
    if cfg.light_direction_randomization_deg > 0.0:
        d = float(cfg.light_direction_randomization_deg)
        sun_pitch = rng.uniform(-d, d)
        sun_yaw = rng.uniform(-d, d)

    camera_jitter: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {}
    if cfg.any_camera:
        pos_r = float(cfg.camera_extrinsics_pos_randomization)
        rot_d = float(cfg.camera_extrinsics_rot_randomization_deg)
        for name in _CAMERA_NAMES:
            ux, uy, uz = _sample_unit_sphere(rng)
            pos = (ux * pos_r, uy * pos_r, uz * pos_r)
            quat = _small_random_quat_wxyz(rng, rot_d)
            camera_jitter[name] = (pos, quat)

    return EpisodeRandomization(
        light_intensity_factor=intensity_factor,
        light_color_delta=color_delta,
        sun_pitch_deg=sun_pitch,
        sun_yaw_deg=sun_yaw,
        camera_jitter=camera_jitter,
    )


# =============================================================================
# Light randomization (USD writes on the live stage)
# =============================================================================
_NOMINAL_LIGHT: dict[str, Any] = {}


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _capture_nominal_light(stage) -> None:
    # Read the dome's authored intensity/colour once so multiplicative /
    # additive jitter always references the baseline, not last episode's
    # already-jittered value.
    if _NOMINAL_LIGHT:
        return
    from pxr import UsdLux

    prim = stage.GetPrimAtPath(_DOME_LIGHT_PRIM)
    intensity = 3000.0
    color = (0.75, 0.75, 0.75)
    if prim and prim.IsValid():
        light = UsdLux.DomeLight(prim)
        attr_i = light.GetIntensityAttr()
        if attr_i and attr_i.HasAuthoredValue():
            intensity = float(attr_i.Get())
        attr_c = light.GetColorAttr()
        if attr_c and attr_c.HasAuthoredValue():
            c = attr_c.Get()
            color = (float(c[0]), float(c[1]), float(c[2]))
    _NOMINAL_LIGHT["dome_intensity"] = intensity
    _NOMINAL_LIGHT["dome_color"] = color


def _ensure_sun_light(stage):
    from pxr import UsdLux

    prim = stage.GetPrimAtPath(_SUN_LIGHT_PRIM)
    if prim and prim.IsValid():
        return UsdLux.DistantLight(prim)
    sun = UsdLux.DistantLight.Define(stage, _SUN_LIGHT_PRIM)
    sun.CreateAngleAttr(_SUN_DEFAULT_ANGLE_DEG)
    sun.CreateIntensityAttr(_SUN_DEFAULT_INTENSITY)
    return sun


def apply_light_randomization(
    stage, er: EpisodeRandomization, cfg: DomainRandomizationConfig
) -> None:
    """Write the sampled light state onto the live USD stage."""
    if not cfg.any_light or stage is None:
        return
    from pxr import Gf, UsdGeom, UsdLux

    _capture_nominal_light(stage)

    dome_prim = stage.GetPrimAtPath(_DOME_LIGHT_PRIM)
    if dome_prim and dome_prim.IsValid():
        dome = UsdLux.DomeLight(dome_prim)
        base_i = float(_NOMINAL_LIGHT.get("dome_intensity", 3000.0))
        dome.GetIntensityAttr().Set(base_i * float(er.light_intensity_factor))
        bc = _NOMINAL_LIGHT.get("dome_color", (0.75, 0.75, 0.75))
        cd = er.light_color_delta
        dome.GetColorAttr().Set(
            Gf.Vec3f(
                _clamp01(bc[0] + cd[0]),
                _clamp01(bc[1] + cd[1]),
                _clamp01(bc[2] + cd[2]),
            )
        )

    if cfg.light_direction_randomization_deg > 0.0:
        sun = _ensure_sun_light(stage)
        sun.GetIntensityAttr().Set(_SUN_DEFAULT_INTENSITY * float(er.light_intensity_factor))
        bc = _NOMINAL_LIGHT.get("dome_color", (0.75, 0.75, 0.75))
        cd = er.light_color_delta
        sun.GetColorAttr().Set(
            Gf.Vec3f(
                _clamp01(bc[0] + cd[0]),
                _clamp01(bc[1] + cd[1]),
                _clamp01(bc[2] + cd[2]),
            )
        )
        # Nominal sun points straight down (-Z); re-orient by the sampled
        # pitch/yaw. Rebuild the xform op cleanly each episode.
        xform = UsdGeom.Xformable(sun.GetPrim())
        xform.ClearXformOpOrder()
        rot = R.from_euler(
            "xyz", [-90.0 + er.sun_pitch_deg, 0.0, er.sun_yaw_deg], degrees=True
        )
        q = rot.as_quat()  # xyzw
        xform.AddOrientOp().Set(Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2])))


# =============================================================================
# Robot base randomization (placement-plan offset + reachability pre-check)
# =============================================================================
def _world_to_base_xy(
    base_pose: tuple[float, float, float, float],
    point_world: tuple[float, float, float],
) -> tuple[float, float]:
    # Same derivation as grasp.ranking._world_to_base_xy_simple — inlined
    # so this module doesn't depend on the grasp package.
    base_x, base_y, _bz, base_yaw_deg = (float(v) for v in base_pose)
    dx = float(point_world[0]) - base_x
    dy = float(point_world[1]) - base_y
    yaw = math.radians(-float(base_yaw_deg))
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    return (cos_y * dx - sin_y * dy, sin_y * dx + cos_y * dy)


def _waypoints_world(candidate) -> list[tuple[float, float, float]]:
    return [
        tuple(candidate.pre_grasp_pos_world),
        tuple(candidate.grasp.position_world),
        tuple(candidate.lift_pos_world),
        tuple(candidate.retreat_pos_world),
    ]


def _base_pose_reaches(
    base_pose: tuple[float, float, float, float],
    candidate,
    robot_name: str,
    workspace_margin: float,
) -> bool:
    specs = default_robot_workspace_specs()
    spec = specs.get(robot_name)
    if spec is None:
        # No workspace model for this robot → can't pre-check; accept.
        return True
    min_x, max_x, min_y, max_y = spec.working_area.bounds_xy()
    m = float(workspace_margin)
    min_x, max_x = min_x + m, max_x - m
    min_y, max_y = min_y + m, max_y - m
    for pt in _waypoints_world(candidate):
        bx, by = _world_to_base_xy(base_pose, pt)
        if not (min_x <= bx <= max_x and min_y <= by <= max_y):
            return False
    return True


def resolve_base_randomization(
    rng: random.Random,
    cfg: DomainRandomizationConfig,
    plan,
    candidate,
    robot_name: str,
    workspace_margin: float,
) -> tuple[Any, EpisodeRandomization]:
    """Return a (possibly base-offset) plan plus the sampled info.

    Resamples up to `_BASE_RESAMPLE_TRIES` times until the cached grasp
    waypoints all fall inside the robot's working-area box from the new
    base pose; falls back to the nominal plan if none qualify.
    """
    if not cfg.any_base:
        return plan, EpisodeRandomization()

    bx, by, bz, byaw = (float(v) for v in plan.base_pose)
    last = (0.0, 0.0, 0.0)
    for attempt in range(1, _BASE_RESAMPLE_TRIES + 1):
        dx, dy, dyaw = _sample_base_offset(rng, cfg)
        new_pose = (bx + dx, by + dy, bz, byaw + dyaw)
        if _base_pose_reaches(new_pose, candidate, robot_name, workspace_margin):
            episode_plan = dataclasses.replace(plan, base_pose=new_pose)
            return episode_plan, EpisodeRandomization(
                base_dx=dx,
                base_dy=dy,
                base_dyaw_deg=dyaw,
                base_resampled_tries=attempt,
            )
        last = (dx, dy, dyaw)

    return plan, EpisodeRandomization(
        base_dx=last[0],
        base_dy=last[1],
        base_dyaw_deg=last[2],
        base_resampled_tries=_BASE_RESAMPLE_TRIES,
        base_fell_back=True,
    )
