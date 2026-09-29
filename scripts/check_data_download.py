"""Exercise the GADM acquisition path in a new, ignored cache."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clinic_data import gadm_file
import geopandas as gpd

cache = Path("build/download-check")
path = gadm_file(cache)
wards = gpd.read_file(path)
assert (wards.NAME_1 == "Nyamira").sum() == 20
print("GADM download/extraction passed:", path)
