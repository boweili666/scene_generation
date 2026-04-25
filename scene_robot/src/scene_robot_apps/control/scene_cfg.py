"""InteractiveSceneCfg factories for stack-cube-style scenes.

`build_single_robot_scene_cfg` and `build_merged_scene_cfg` assemble
ground / dome light / robot / table / cube prims into the
`InteractiveSceneCfg` types Isaac Lab expects. They were originally
defined alongside the per-robot stack specs in `stack_cube.py`, but
they're general-purpose scene factories — anything that wants the
"robot + table + two cubes" reference scene goes through them — so
they live in their own module now.

`_make_cube_cfg` is the shared helper for translating a `CuboidSpec`
into a `RigidObjectCfg`.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .robot_spec import CuboidSpec, RobotStackSpec


def _make_material(spec: CuboidSpec):
    physics_material = None
    if spec.friction is not None:
        physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=spec.friction[0],
            dynamic_friction=spec.friction[1],
        )
    return physics_material


def _make_cube_cfg(prim_path: str, spec: CuboidSpec) -> RigidObjectCfg:
    rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=spec.kinematic)
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=spec.size,
            rigid_props=rigid_props,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_make_material(spec),
            mass_props=None if spec.mass is None else sim_utils.MassPropertiesCfg(mass=spec.mass),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=spec.color, metallic=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=spec.pos),
    )


def build_single_robot_scene_cfg(spec: RobotStackSpec):
    @configclass
    class _SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, spec.ground_z)),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=spec.light_intensity, color=(0.75, 0.75, 0.75)),
        )
        robot = spec.robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

        if spec.table is not None:
            table = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.CuboidCfg(
                    size=spec.table.size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=spec.table.color, metallic=0.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=spec.table.pos),
            )

        if spec.cube_base is not None:
            cube_base = _make_cube_cfg("{ENV_REGEX_NS}/CubeBase", spec.cube_base)
        if spec.cube_pick is not None:
            cube_pick = _make_cube_cfg("{ENV_REGEX_NS}/CubePick", spec.cube_pick)

    return _SceneCfg


def build_merged_scene_cfg(spec_items: list[tuple[str, RobotStackSpec]]):
    attrs: dict[str, object] = {
        "ground": AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        ),
        "dome_light": AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        ),
    }
    for prefix, spec in spec_items:
        attrs[f"{prefix}_robot"] = spec.robot_cfg.replace(prim_path=f"/World/{prefix}/Robot")
        if spec.table is not None:
            attrs[f"{prefix}_table"] = RigidObjectCfg(
                prim_path=f"/World/{prefix}/Table",
                spawn=sim_utils.CuboidCfg(
                    size=spec.table.size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=spec.table.color, metallic=0.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=spec.table.pos),
            )
        attrs[f"{prefix}_cube_base"] = _make_cube_cfg(f"/World/{prefix}/CubeBase", spec.cube_base)
        attrs[f"{prefix}_cube_pick"] = _make_cube_cfg(f"/World/{prefix}/CubePick", spec.cube_pick)
    return configclass(type("MergedStackSceneCfg", (InteractiveSceneCfg,), attrs))
