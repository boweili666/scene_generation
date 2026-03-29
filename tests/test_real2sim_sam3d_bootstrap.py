import sys
import tempfile
import unittest
from pathlib import Path

from pipelines.real2sim.sam3d_bootstrap import (
    default_config_path,
    ensure_sam3d_imports,
    validate_sam3d_layout,
)


class Real2SimSam3dBootstrapTest(unittest.TestCase):
    def test_default_config_path_matches_expected_layout(self) -> None:
        root = Path("/tmp/example-sam3d")
        self.assertEqual(
            default_config_path(root),
            root / "checkpoints" / "hf" / "pipeline.yaml",
        )

    def test_ensure_sam3d_imports_adds_root_and_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notebook = root / "notebook"
            notebook.mkdir(parents=True)

            resolved_root, resolved_notebook = ensure_sam3d_imports(root)

            self.assertEqual(resolved_root, root.resolve())
            self.assertEqual(resolved_notebook, notebook.resolve())
            self.assertIn(str(root.resolve()), sys.path)
            self.assertIn(str(notebook.resolve()), sys.path)

    def test_validate_sam3d_layout_accepts_expected_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inference_path = root / "notebook" / "inference.py"
            package_init = root / "sam3d_objects" / "__init__.py"
            config = root / "checkpoints" / "hf" / "pipeline.yaml"
            inference_path.parent.mkdir(parents=True)
            package_init.parent.mkdir(parents=True)
            config.parent.mkdir(parents=True)
            inference_path.write_text("# stub\n", encoding="utf-8")
            package_init.write_text("# stub\n", encoding="utf-8")
            config.write_text("pipeline: {}\n", encoding="utf-8")

            resolved_root, resolved_config = validate_sam3d_layout(root)

            self.assertEqual(resolved_root, root.resolve())
            self.assertEqual(resolved_config, config.resolve())

    def test_validate_sam3d_layout_requires_inference_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_init = root / "sam3d_objects" / "__init__.py"
            config = root / "checkpoints" / "hf" / "pipeline.yaml"
            package_init.parent.mkdir(parents=True)
            config.parent.mkdir(parents=True)
            package_init.write_text("# stub\n", encoding="utf-8")
            config.write_text("pipeline: {}\n", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                validate_sam3d_layout(root)


if __name__ == "__main__":
    unittest.main()
