"""Compare two historical facility selections, without asserting clinical capability."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import pandana as pdna
from clinic_data import load_nyamira

wards, facilities, hospitals, graph, node_ids, *_ = load_nyamira()
nodes, edges = ox.graph_to_gdfs(graph)
edges = edges.reset_index()
network = pdna.Network(nodes.x, nodes.y, edges.u, edges.v, edges[["length"]], twoway=False)
osm = gpd.read_file("data/Nyamira_health_osm.geojson")
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, points, key, label in [(axes[0], osm, "osm", "OSM tagged facilities"),
                                (axes[1], hospitals, "maina", "Maina hospital labels")]:
    network.set_pois(key, maxdist=80000, maxitems=1, x_col=points.geometry.x, y_col=points.geometry.y)
    km = network.nearest_pois(80000, key, num_pois=1).loc[node_ids, 1].to_numpy() / 1000
    wards.assign(distance=km).plot(column="distance", cmap="Blues", vmin=0, vmax=30,
        edgecolor="grey", linewidth=.5, ax=ax, legend=True,
        legend_kwds={"label": "Road km to nearest selected record", "shrink": .65})
    points.plot(ax=ax, color="crimson", marker="P", markersize=60)
    ax.set_title(f"{label} ({len(points)} records)\nMean {km.mean():.1f} km; maximum {km.max():.1f} km", fontsize=11)
    ax.set_axis_off()
fig.text(.02, .01, "Boundaries: GADM; roads/facilities: © OpenStreetMap contributors (ODbL); facilities: Maina et al. (2019).", fontsize=7)
fig.tight_layout(rect=[0, .025, 1, 1])
fig.savefig("slides/figs/facilities_compare.png", dpi=160, bbox_inches="tight")
print("Saved the facility-source comparison.")
