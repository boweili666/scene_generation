import tempfile
import unittest
import json
from pathlib import Path
from unittest import mock

from pipelines.real2sim.streaming_generation_client import (
    DEFAULT_IMAGE_PATH,
    DEFAULT_MASK_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PREDICT_STREAM_SERVER,
    DEFAULT_SCENE_GRAPH,
    assemble_scene_usd_from_manifest,
    build_parser,
    collect_usd_conversion_pairs,
    convert_outputs_to_usd,
    resolve_scene_graph_path,
    select_masks_from_assignment,
)


class Real2SimStreamingGenerationClientTest(unittest.TestCase):
    def test_select_masks_from_assignment_filters_to_assigned_outputs_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask_paths = []
            for name in ("0.png", "1.png", "2.png", "3.png"):
                path = root / name
                path.write_bytes(b"mask")
                mask_paths.append(path)

            assignment_path = root / "assignment.json"
            assignment_path.write_text(
                json.dumps(
                    {
                        "assignments": [
                            {"output_name": "2"},
                            {"output_name": "0"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            selected = select_masks_from_assignment(mask_paths, assignment_path)

            self.assertEqual([path.name for path in selected], ["2.png", "0.png"])

    def test_build_parser_uses_runtime_defaults(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.server, DEFAULT_PREDICT_STREAM_SERVER)
        self.assertEqual(args.image, DEFAULT_IMAGE_PATH)
        self.assertEqual(args.mask_dir, DEFAULT_MASK_DIR)
        self.assertEqual(args.scene_graph, DEFAULT_SCENE_GRAPH)
        self.assertEqual(args.output_dir, DEFAULT_OUTPUT_DIR)

    def test_resolve_scene_graph_path_skips_missing_default_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            default_path = Path(tmp) / "current_scene_graph.json"
            custom_path = Path(tmp) / "custom_scene_graph.json"

            self.assertIsNone(resolve_scene_graph_path(default_path, default_scene_graph=default_path))
            self.assertEqual(
                resolve_scene_graph_path(custom_path, default_scene_graph=default_path),
                custom_path,
            )

            default_path.write_text("{}", encoding="utf-8")
            self.assertEqual(
                resolve_scene_graph_path(default_path, default_scene_graph=default_path),
                default_path,
            )

    def test_collect_usd_conversion_pairs_defaults_to_objects_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            objects_dir = root / "objects"
            objects_dir.mkdir(parents=True)

            (root / "scene_merged_post.glb").write_bytes(b"post")
            (root / "scene_merged.glb").write_bytes(b"canonical")
            (objects_dir / "obj_01.glb").write_bytes(b"1")
            (objects_dir / "obj_00.glb").write_bytes(b"0")

            pairs = collect_usd_conversion_pairs(root)

            self.assertEqual(
                pairs,
                [
                    (objects_dir / "obj_00.glb", root / "usd_objects" / "obj_00.usd"),
                    (objects_dir / "obj_01.glb", root / "usd_objects" / "obj_01.usd"),
                ],
            )

    def test_collect_usd_conversion_pairs_can_include_post_scene(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            objects_dir = root / "objects"
            objects_dir.mkdir(parents=True)

            (root / "scene_merged_post.glb").write_bytes(b"post")
            (objects_dir / "obj_00.glb").write_bytes(b"0")

            pairs = collect_usd_conversion_pairs(root, include_scene_glb=True)

            self.assertEqual(
                pairs,
                [
                    (root / "scene_merged_post.glb", root / "scene_merged_post.usd"),
                    (objects_dir / "obj_00.glb", root / "usd_objects" / "obj_00.usd"),
                ],
            )

    def test_convert_outputs_to_usd_invokes_converter_once_with_object_pairs_and_materials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            objects_dir = root / "objects"
            objects_dir.mkdir(parents=True)
            converter_script = root / "mesh_to_usd_converter.py"
            converter_script.write_text("# stub\n", encoding="utf-8")

            (objects_dir / "obj_00.glb").write_bytes(b"0")

            with mock.patch("pipelines.real2sim.streaming_generation_client.subprocess.run") as run_mock:
                convert_outputs_to_usd(
                    root,
                    converter_python="/fake/python",
                    asset_converter_script=converter_script,
                )

            run_mock.assert_called_once()
            cmd = run_mock.call_args.args[0]
            self.assertEqual(cmd[:4], ["/fake/python", "-u", str(converter_script.resolve()), "--input-files"])
            self.assertIn(str((objects_dir / "obj_00.glb").resolve()), cmd)
            self.assertIn("--output-files", cmd)
            self.assertIn(str((root / "usd_objects" / "obj_00.usd").resolve()), cmd)
            self.assertIn("--load-materials", cmd)

    def test_assemble_scene_usd_from_manifest_invokes_converter_in_assembly_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            converter_script = root / "mesh_to_usd_converter.py"
            manifest_path = root / "real2sim_asset_manifest.json"
            scene_output = root / "scene_merged_post.usd"
            converter_script.write_text("# stub\n", encoding="utf-8")
            manifest_path.write_text("{}", encoding="utf-8")

            with mock.patch("pipelines.real2sim.streaming_generation_client.subprocess.run") as run_mock:
                assemble_scene_usd_from_manifest(
                    manifest_path,
                    converter_python="/fake/python",
                    asset_converter_script=converter_script,
                    scene_output_path=scene_output,
                )

            run_mock.assert_called_once()
            cmd = run_mock.call_args.args[0]
            self.assertEqual(cmd[:3], ["/fake/python", "-u", str(converter_script.resolve())])
            self.assertIn("--assemble-scene-from-manifest", cmd)
            self.assertIn(str(manifest_path.resolve()), cmd)
            self.assertIn("--scene-output", cmd)
            self.assertIn(str(scene_output.resolve()), cmd)


if __name__ == "__main__":
    unittest.main()
