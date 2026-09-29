import json
from pathlib import Path
import subprocess
import unittest

import nbformat
from build_notebook import make_notebook


class NotebookTests(unittest.TestCase):
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
