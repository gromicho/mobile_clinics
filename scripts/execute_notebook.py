"""Execute every cell in a fresh kernel, saving outputs only under ignored build/."""
from pathlib import Path
import argparse
import sys

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--kernel", default="python3")
args = parser.parse_args()
nb = nbformat.read(root / "mobile_clinic_routing.ipynb", as_version=4)
client = NotebookClient(nb, timeout=1800, kernel_name=args.kernel, resources={"metadata": {"path": str(root)}})
client.execute()
build = root / "build"
build.mkdir(exist_ok=True)
nbformat.write(nb, build / "mobile_clinic_routing.executed.ipynb")
errors = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
if errors: raise RuntimeError(errors)
print("Fresh-kernel execution passed; maintained notebook remains output-free.")
