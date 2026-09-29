"""Rebuild and validate notebook results, figures and the Beamer PDF from the root."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
for command in [["build_notebook.py"], ["scripts/notebook_preflight.py", "mobile_clinic_routing.ipynb"],
                ["-m", "unittest", "discover", "-s", "tests", "-v"], ["scripts/execute_notebook.py"],
                ["scripts/check_results.py"], ["slides/make_facilities_compare.py"], ["scripts/build_deck.py"]]:
    subprocess.run([sys.executable, *command], cwd=root, check=True)
