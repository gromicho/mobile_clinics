"""Nyamira teaching inputs. GADM is downloaded locally, never redistributed."""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import pandana as pdna
import rasterio
import rasterio.mask

GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_KEN_3.json.zip"
WORLDPOP_URL = "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/2020/KEN/ken_ppp_2020_1km_Aggregated_UNadj.tif"
MAINA_URL = "https://ndownloader.figshare.com/files/14379593"


def download(url, path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {path.name}")
        request = urllib.request.Request(url, headers={"User-Agent": "mobile-clinics-teaching/1.0"})
        temp = path.with_suffix(path.suffix + ".part")
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            temp.replace(path)
        except Exception:
            temp.unlink(missing_ok=True)
            raise
    return path


def gadm_file(cache="data"):
    """Academic/noncommercial use: https://gadm.org/license.html."""
    cache = Path(cache)
    path = cache / "gadm41_KEN_3.json"
    if not path.exists():
        archive = download(GADM_URL, cache / "gadm41_KEN_3.json.zip")
        # Extract only the known member, never arbitrary paths from an archive.
        with zipfile.ZipFile(archive) as zipped:
            content = zipped.read("gadm41_KEN_3.json")
        json.loads(content)  # reject an invalid download before caching it
        path.write_bytes(content)
    return path


def load_nyamira(data="data", seed=42):
    data = Path(data)
    boundary = gadm_file(data)
    wards = gpd.read_file(boundary)
    wards = wards.loc[wards.NAME_1 == "Nyamira", ["NAME_2", "NAME_3", "geometry"]].rename(
        columns={"NAME_2": "subcounty", "NAME_3": "ward"}).reset_index(drop=True)
    if len(wards) != 20 or wards.ward.duplicated().any():
        raise ValueError("This example expects the 20 distinct GADM 4.1 wards of Nyamira.")
    projected = wards.estimate_utm_crs()
    poly = wards.geometry.union_all()
    raster = download(WORLDPOP_URL, data / "ken_ppp_2020_1km_Aggregated_UNadj.tif")
    with rasterio.open(raster) as src:
        populations = []
        for geom in wards.to_crs(src.crs).geometry:
            arr, _ = rasterio.mask.mask(src, [geom.__geo_interface__], crop=True, filled=False)
            populations.append(float(np.ma.clip(arr[0], 0, None).sum()))
    wards["pop"] = populations
    if not np.isfinite(populations).all() or min(populations) <= 0:
        raise ValueError("Every ward must have positive finite population.")
    graph_path = data / "Nyamira_drive.graphml"
    if graph_path.exists():
        graph = ox.load_graphml(graph_path)
    else:
        buffered = gpd.GeoSeries([poly], crs=4326).to_crs(projected).buffer(2500).to_crs(4326).iloc[0]
        graph = ox.graph_from_polygon(buffered, network_type="drive")
        ox.save_graphml(graph, graph_path)
    facilities_path = data / "Nyamira_facilities_maina2019.csv"
    if facilities_path.exists():
        facilities = pd.read_csv(facilities_path)
    else:
        book = download(MAINA_URL, data / "ssa_mfl_maina2019.xlsx")
        frame = pd.read_excel(book)
        frame = frame.loc[frame.Country.str.strip().eq("Kenya") & frame.Lat.notna() & frame.Long.notna()]
        points = gpd.GeoDataFrame(frame, geometry=gpd.points_from_xy(frame.Long, frame.Lat), crs=4326)
        facilities = pd.DataFrame(points.loc[points.within(poly)].drop(columns="geometry")).rename(
            columns={"Facility name": "name", "Facility type": "type", "Ownership": "ownership", "Lat": "lat", "Long": "lon"})
        facilities = facilities[["name", "type", "ownership", "lat", "lon"]].sort_values(["type", "name"])
        facilities.to_csv(facilities_path, index=False)
    facilities = gpd.GeoDataFrame(facilities, geometry=gpd.points_from_xy(facilities.lon, facilities.lat), crs=4326)
    hospitals = facilities.loc[facilities["type"].str.contains("Hospital", case=False, na=False)].copy()
    if hospitals.empty:
        raise ValueError("No hospitals found for the accessibility proxy.")
    nodes, edges = ox.graph_to_gdfs(graph)
    edges = edges.reset_index()
    net = pdna.Network(nodes.x, nodes.y, edges.u, edges.v, edges[["length"]], twoway=False)
    centroids = wards.to_crs(projected).centroid.to_crs(4326)
    node_ids = net.get_node_ids(centroids.x, centroids.y).to_numpy()
    wards["lon"], wards["lat"] = centroids.x.to_numpy(), centroids.y.to_numpy()
    snapped = gpd.GeoSeries(gpd.points_from_xy(nodes.loc[node_ids, "x"], nodes.loc[node_ids, "y"]), crs=4326)
    wards["snap_km"] = centroids.to_crs(projected).distance(snapped.to_crs(projected)).to_numpy() / 1000
    n = len(wards)
    distance = np.asarray(net.shortest_path_lengths(np.repeat(node_ids, n), np.tile(node_ids, n))).reshape(n, n) / 1000
    if not np.isfinite(distance).all() or distance.max() > 1000:
        raise ValueError("Disconnected or implausible ward road distances; inspect the graph.")
    if len(set(node_ids)) != n:
        raise ValueError("Two wards snap to the same road node; choose distinct service locations.")
    net.set_pois("hospitals", maxdist=80000, maxitems=1, x_col=hospitals.lon, y_col=hospitals.lat)
    wards["km_to_hospital_proxy"] = net.nearest_pois(80000, "hospitals", num_pois=1).loc[node_ids, 1].to_numpy() / 1000
    largest = wards.km_to_hospital_proxy.max()
    if not np.isfinite(largest) or not 0 < largest < 80:
        raise ValueError("Hospital accessibility is missing, zero or beyond the search radius.")
    # This fixed synthetic ward profile is independent of fitting and evaluation.
    wards["unvaccinated"] = np.random.default_rng(seed).beta(4, 6, n)
    wards["inaccessibility"] = wards.km_to_hospital_proxy / largest
    base = (0.02 * wards["pop"] * wards.unvaccinated * (0.5 + wards.inaccessibility)).to_numpy()
    features = np.column_stack([np.log(wards["pop"]), wards.unvaccinated, wards.inaccessibility])
    matches = np.flatnonzero(wards.ward.eq("Township"))
    if len(matches) != 1:
        raise ValueError("Expected exactly one Township ward for the teaching depot.")
    provenance = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [boundary, raster, graph_path, facilities_path]}
    return wards, facilities, hospitals, graph, node_ids, distance, base, features, int(matches[0]), provenance


def road_coordinates(graph, node_ids, order):
    coords = []
    if len(order) == 1:
        v = graph.nodes[node_ids[order[0]]]
        return np.array([[v["x"], v["y"]]])
    for a, b in zip(order[:-1], order[1:]):
        path = ox.routing.shortest_path(graph, node_ids[a], node_ids[b], weight="length")
        if path is None:
            raise ValueError("A displayed route has no directed road path.")
        coords.extend((graph.nodes[v]["x"], graph.nodes[v]["y"]) for v in path)
    return np.asarray(coords)
