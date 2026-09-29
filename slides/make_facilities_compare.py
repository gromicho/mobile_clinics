"""Regenerates slides/figs/facilities_compare.png: road distance to the nearest vaccine point per ward,
under the OpenStreetMap facility tags versus the hospitals of Maina et al. (2019).
Run from the repository root:  py -3 slides/make_facilities_compare.py"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd, osmnx as ox, pandana as pdna
import matplotlib.pyplot as plt
import contextily as cx

D = "data"
wards = gpd.read_file(f"{D}/gadm41_KEN_3.json")
wards = wards[wards.NAME_1 == "Nyamira"].reset_index(drop=True)
n = len(wards)
G = ox.load_graphml(f"{D}/Nyamira_drive.graphml")
nodes, edges = ox.graph_to_gdfs(G)
edges = edges.reset_index()
net = pdna.Network(nodes["x"], nodes["y"], edges["u"], edges["v"], edges[["length"]], twoway=False)
cent = wards.to_crs(32736).centroid.to_crs(4326)
nid = net.get_node_ids(cent.x.values, cent.y.values).values

osm = gpd.read_file(f"{D}/Nyamira_health_osm.geojson")          # what OpenStreetMap tags as hospital or clinic
mf = pd.read_csv(f"{D}/Nyamira_facilities_maina2019.csv")
mf = gpd.GeoDataFrame(mf, geometry=gpd.points_from_xy(mf.lon, mf.lat), crs=4326)
hosp = mf[mf["type"].str.contains("Hospital")]


def km_to_nearest(fac, key):
    net.set_pois(key, maxdist=80000, maxitems=1, x_col=fac.geometry.x, y_col=fac.geometry.y)
    return net.nearest_pois(80000, key, num_pois=1).loc[nid, 1].values / 1000


ko, km = km_to_nearest(osm, "osm"), km_to_nearest(hosp, "maina")
print(f"OSM: mean {ko.mean():.1f} km, max {ko.max():.1f} | Maina: mean {km.mean():.1f} km, max {km.max():.1f}"
      f" | rank correlation {pd.Series(ko).corr(pd.Series(km), method='spearman'):.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
for ax, k, fac, title, extra in [(axes[0], ko, osm, "OpenStreetMap: 6 tagged facilities", None),
                                 (axes[1], km, hosp, "Maina et al. 2019: 6 hospitals of 98 facilities", mf)]:
    wards.assign(k=k).plot(column="k", cmap="Blues", vmin=0, vmax=30, alpha=0.65, edgecolor="grey", linewidth=0.5, ax=ax,
                           legend=True, legend_kwds={"label": "road km to nearest", "shrink": 0.55})
    try:
        cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.Esri.WorldTopoMap, zoom=11, attribution_size=5)
    except Exception as e:
        print("basemap skipped:", type(e).__name__)
    if extra is not None:
        extra.plot(ax=ax, color="dimgrey", markersize=6, zorder=2)
    fac.plot(ax=ax, color="crimson", marker="P", markersize=90, edgecolor="white", zorder=3)
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
plt.tight_layout()
plt.savefig("slides/figs/facilities_compare.png", dpi=150)
print("saved slides/figs/facilities_compare.png")
