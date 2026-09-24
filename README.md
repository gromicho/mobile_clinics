# Mobile clinic routing with learned constraints

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gromicho/mobile_clinics/blob/main/mobile_clinic_routing.ipynb)

Companion notebook to Chapter 6 of Mayukh Ghosh's PhD thesis (Domain-Driven Mobile Clinic Routing). It shows why a predict-then-optimize planner mis-plans when the demand met at one ward depends on which other wards the clinic visits, and how embedding the demand model in the optimization model (`gurobi-machinelearning`) fixes it. Both signs of spillover are shown: cannibalisation and mobilisation.

- `mobile_clinic_routing.ipynb`: the executed notebook.
- `build_notebook.py`: regenerates the notebook from source cells (`py -3 build_notebook.py`, then execute with Jupyter).
- `data/`: cached downloads (GADM wards, WorldPop raster, OSM road graph and facilities for Nyamira). Delete to re-download.

## Running

Locally: Python 3.12 with osmnx, pandana, geopandas, rasterio, folium, scikit-learn, gurobipy and gurobi-machinelearning. The first cell installs anything missing.

Colab: click the badge above and run the cells top to bottom. The first cell installs the packages and clones this repository so the cached data is used and the 2 to 3 minute OpenStreetMap download is skipped; `pip install gurobipy` provides a size-limited licence (2000 variables, 2000 constraints) and every model in the notebook stays at or under 1000 of each, with subtour cuts generated lazily outside that count.

For a session, do not "Run all": the sweep and the answer table take about a minute together. Rerun only the hands-on parameter cell and the scenario cells, which take a few seconds each.

## Design choices

- One county, Nyamira, 20 wards from GADM level 3, so the routing instance stays small.
- Road distances between ward centroids from pandana contraction hierarchies on the osmnx drive graph; no API keys.
- Demand is simulated with an explicit exposure term, and the training set is a replayed history of random deployments, which makes the "decision variable inside the predictor" step explicit.
- Objective: expected doses delivered minus a per-km driving cost, at most 8 stops, open path from Nyamira Township.
- Subtours are eliminated with Dantzig-Fulkerson-Johnson cuts generated lazily in a Gurobi callback, so they add nothing to the model size the Community Edition checks.
