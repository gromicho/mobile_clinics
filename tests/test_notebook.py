import json
from pathlib import Path
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import nbformat
from build_notebook import make_notebook


class NotebookTests(unittest.TestCase):
    def test_colab_package_replacement_requires_real_restart(self):
        setup = next(c.source for c in make_notebook().cells if c.id == "setup")
        imports = next(c.source for c in make_notebook().cells if c.id == "imports")
        state = {}
        replacements = ["numpy"]

        def install(command):
            report = Path(command[command.index("--report") + 1])
            report.write_text(json.dumps({"install": [
                {"metadata": {"name": name}} for name in replacements]}))

        with patch.dict("sys.modules", {"google.colab": SimpleNamespace()}), \
             patch("importlib.metadata.distributions", return_value=[
                 SimpleNamespace(metadata={"Name": "numpy"})]), \
             patch("subprocess.check_call", side_effect=install), \
             patch("subprocess.check_output", return_value="test-commit"):
            with self.assertRaisesRegex(RuntimeError, "Restart session"):
                exec(setup, state)
            replacements.clear()
            # Running setup again must not permit use of the stale kernel.
            with self.assertRaisesRegex(RuntimeError, "Restart session"):
                exec(setup, state)
            with self.assertRaisesRegex(RuntimeError, "Restart session"):
                exec(imports, state)
            # A new kernel with satisfied requirements can continue.
            restarted = {}
            exec(setup, restarted)
            self.assertFalse(restarted["_COLAB_RESTART_REQUIRED"])

    def test_colab_new_package_does_not_require_restart(self):
        setup = next(c.source for c in make_notebook().cells if c.id == "setup")

        def install(command):
            Path(command[command.index("--report") + 1]).write_text(
                json.dumps({"install": [{"metadata": {"name": "pandana"}}]}))

        with patch.dict("sys.modules", {"google.colab": SimpleNamespace()}), \
             patch("importlib.metadata.distributions", return_value=[
                 SimpleNamespace(metadata={"Name": "numpy"})]), \
             patch("subprocess.check_call", side_effect=install), \
             patch("subprocess.check_output", return_value="test-commit"):
            state = {}
            exec(setup, state)
            self.assertFalse(state["_COLAB_RESTART_REQUIRED"])

    def test_python_sources_compile(self):
        for path in Path(".").rglob("*.py"):
            if "build" not in path.parts and ".venv" not in path.parts:
                compile(path.read_text(encoding="utf8"), str(path), "exec")

    def test_source_matches_generator(self):
        actual = nbformat.read("mobile_clinic_routing.ipynb", as_version=4)
        nbformat.validate(actual)
        self.assertEqual(actual, make_notebook())
        for cell in actual.cells:
            if cell.cell_type == "code":
                compile(cell.source, cell.id, "exec")
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])

    def test_colab_launcher(self):
        cell = make_notebook().cells[0].source
        self.assertIn("https://colab.research.google.com/github/gromicho/mobile_clinics/blob/main/mobile_clinic_routing.ipynb", cell)

    def test_no_gadm_boundaries_tracked(self):
        paths = subprocess.check_output(["git", "ls-files", "data"], text=True).splitlines()
        self.assertFalse(any("gadm" in p.lower() for p in paths), paths)


if __name__ == "__main__":
    unittest.main()
