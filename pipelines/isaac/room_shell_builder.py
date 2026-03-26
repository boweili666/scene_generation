from pathlib import Path
from typing import Dict, Optional


def _add_quad(stage, prim_path: str, points, color=(0.75, 0.75, 0.75), material_path: str = ""):
    from pxr import UsdGeom, UsdPhysics, UsdShade

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([(0.0, 0.0, 1.0)])
    mesh.SetNormalsInterpolation("constant")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    mesh.CreateExtentAttr([(min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))])
    mesh.CreateDisplayColorAttr([color])
    mesh.CreateDoubleSidedAttr(True)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
        UsdPhysics.Tokens.none
    )
    if material_path:
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(UsdShade.Material.Get(stage, material_path))


def _make_preview_material(stage, path: str, color=(0.75, 0.75, 0.75), roughness=0.8, metallic=0.0):
    from pxr import Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _make_textured_preview_material(
    stage,
    path: str,
    color=(0.75, 0.75, 0.75),
    roughness=0.8,
    metallic=0.0,
    basecolor_tex: Optional[Path] = None,
    normal_tex: Optional[Path] = None,
    metallic_roughness_tex: Optional[Path] = None,
):
    from pxr import Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))

    st_reader = UsdShade.Shader.Define(stage, f"{path}/PrimvarReader_st")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    if basecolor_tex and basecolor_tex.exists():
        tex = UsdShade.Shader.Define(stage, f"{path}/BaseColorTex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(basecolor_tex))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")

    if normal_tex and normal_tex.exists():
        tex = UsdShade.Shader.Define(stage, f"{path}/NormalTex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(normal_tex))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(tex.ConnectableAPI(), "rgb")

    if metallic_roughness_tex and metallic_roughness_tex.exists():
        tex = UsdShade.Shader.Define(stage, f"{path}/MetalRoughTex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(metallic_roughness_tex))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        # StableMaterials packed map: G=roughness, B=metallic.
        tex.CreateOutput("g", Sdf.ValueTypeNames.Float)
        tex.CreateOutput("b", Sdf.ValueTypeNames.Float)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(tex.ConnectableAPI(), "g")
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(tex.ConnectableAPI(), "b")

    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _find_surface_texture(texture_dir: Path, surface: str, kind: str) -> Optional[Path]:
    p = texture_dir / f"{surface}_{kind}.png"
    return p if p.exists() else None


def _color_from_material_text(text: str, default):
    t = str(text or "").lower()
    if "hardwood" in t or "wood" in t:
        return (0.55, 0.40, 0.28)
    if "carpet" in t:
        return (0.45, 0.48, 0.52)
    if "concrete" in t:
        return (0.55, 0.55, 0.55)
    if "tile" in t:
        return (0.70, 0.70, 0.70)
    if "drywall" in t or "paint" in t:
        return (0.90, 0.90, 0.90)
    return default


def generate_room_usd_from_scene(
    scene_data: Dict,
    output_usd: Path,
    floor_z: float = 0.0,
    include_ceiling: bool = False,
    include_back_wall: bool = True,
    include_left_wall: bool = True,
    include_right_wall: bool = True,
    include_front_wall: bool = False,
    texture_dir: Optional[Path] = None,
) -> Path:
    """
    Generate a standalone USD room shell from scene dimensions.
    Axis convention is Z-up:
      - X: front/back
      - Y: left/right
      - Z: height
    """
    from pxr import Usd

    dims = scene_data.get("scene", {}).get("dimensions", {})
    length = float(dims.get("length", 10.0))
    width = float(dims.get("width", 10.0))
    height = float(dims.get("height", 3.0))

    half_l = length * 0.5
    half_w = width * 0.5
    z0 = float(floor_z)
    z1 = z0 + height

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_usd))
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)
    room_mats = scene_data.get("scene", {}).get("materials", {})
    floor_color = _color_from_material_text(room_mats.get("floor"), (0.65, 0.65, 0.65))
    wall_color = _color_from_material_text(room_mats.get("walls"), (0.85, 0.85, 0.85))
    ceil_color = _color_from_material_text(room_mats.get("ceiling"), (0.78, 0.78, 0.78))

    floor_base = floor_nrm = floor_mr = None
    wall_base = wall_nrm = wall_mr = None
    ceil_base = ceil_nrm = ceil_mr = None
    if texture_dir and texture_dir.exists():
        floor_base = _find_surface_texture(texture_dir, "floor", "basecolor")
        floor_nrm = _find_surface_texture(texture_dir, "floor", "normal")
        floor_mr = _find_surface_texture(texture_dir, "floor", "metallic_roughness")
        wall_base = _find_surface_texture(texture_dir, "walls", "basecolor")
        wall_nrm = _find_surface_texture(texture_dir, "walls", "normal")
        wall_mr = _find_surface_texture(texture_dir, "walls", "metallic_roughness")
        ceil_base = _find_surface_texture(texture_dir, "ceiling", "basecolor")
        ceil_nrm = _find_surface_texture(texture_dir, "ceiling", "normal")
        ceil_mr = _find_surface_texture(texture_dir, "ceiling", "metallic_roughness")

    _make_textured_preview_material(
        stage,
        "/World/RoomLooks/FloorMat",
        color=floor_color,
        roughness=0.65,
        basecolor_tex=floor_base,
        normal_tex=floor_nrm,
        metallic_roughness_tex=floor_mr,
    )
    _make_textured_preview_material(
        stage,
        "/World/RoomLooks/WallMat",
        color=wall_color,
        roughness=0.85,
        basecolor_tex=wall_base,
        normal_tex=wall_nrm,
        metallic_roughness_tex=wall_mr,
    )
    _make_textured_preview_material(
        stage,
        "/World/RoomLooks/CeilingMat",
        color=ceil_color,
        roughness=0.9,
        basecolor_tex=ceil_base,
        normal_tex=ceil_nrm,
        metallic_roughness_tex=ceil_mr,
    )

    # Floor
    _add_quad(
        stage,
        "/World/Room/floor",
        [
            (-half_l, -half_w, z0),
            (half_l, -half_w, z0),
            (half_l, half_w, z0),
            (-half_l, half_w, z0),
        ],
        color=floor_color,
        material_path="/World/RoomLooks/FloorMat",
    )

    # Back wall (x = -half_l)
    if include_back_wall:
        _add_quad(
            stage,
            "/World/Room/back_wall",
            [
                (-half_l, -half_w, z0),
                (-half_l, half_w, z0),
                (-half_l, half_w, z1),
                (-half_l, -half_w, z1),
            ],
            color=wall_color,
            material_path="/World/RoomLooks/WallMat",
        )

    # Left wall (y = -half_w)
    if include_left_wall:
        _add_quad(
            stage,
            "/World/Room/left_wall",
            [
                (-half_l, -half_w, z0),
                (half_l, -half_w, z0),
                (half_l, -half_w, z1),
                (-half_l, -half_w, z1),
            ],
            color=wall_color,
            material_path="/World/RoomLooks/WallMat",
        )

    # Right wall (y = +half_w)
    if include_right_wall:
        _add_quad(
            stage,
            "/World/Room/right_wall",
            [
                (-half_l, half_w, z0),
                (half_l, half_w, z0),
                (half_l, half_w, z1),
                (-half_l, half_w, z1),
            ],
            color=wall_color,
            material_path="/World/RoomLooks/WallMat",
        )

    if include_front_wall:
        _add_quad(
            stage,
            "/World/Room/front_wall",
            [
                (half_l, -half_w, z0),
                (half_l, half_w, z0),
                (half_l, half_w, z1),
                (half_l, -half_w, z1),
            ],
            color=wall_color,
            material_path="/World/RoomLooks/WallMat",
        )

    if include_ceiling:
        _add_quad(
            stage,
            "/World/Room/ceiling",
            [
                (-half_l, -half_w, z1),
                (half_l, -half_w, z1),
                (half_l, half_w, z1),
                (-half_l, half_w, z1),
            ],
            color=ceil_color,
            material_path="/World/RoomLooks/CeilingMat",
        )

    stage.SetMetadata("upAxis", "Z")
    stage.GetRootLayer().Save()
    return output_usd
