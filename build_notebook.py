"""Generate the output-free maintained notebook: python build_notebook.py."""
from pathlib import Path
import nbformat as nbf


def make_notebook():
    cells = []
    def md(key, source):
        cells.append(nbf.v4.new_markdown_cell(source.strip(), id=key))
    def code(key, source):
        cells.append(nbf.v4.new_code_cell(source.strip(), id=key))

    md("introduction", r"""
# Mobile clinic routing: predicting while optimising

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gromicho/mobile_clinics/blob/main/mobile_clinic_routing.ipynb)

A SIKS teaching companion to Chapter 6 of Mayukh Ghosh's thesis, joint work with
Chintan Amrit and Joaquim Gromicho. Nyamira's geography is real; **vaccination
outcomes are synthetic**. This illustrates an idea, not a replication of the thesis
experiments or evidence of an effect in an AMREF deployment.

We compare each predictor used before optimisation with the **same predictor**
embedded inside it. A historical-mean forecast and a simulator benchmark provide
additional comparisons. The objective is expected doses served minus driving cost.

Run top to bottom. If setup asks for a restart after replacing packages, choose
**Runtime → Restart session**, then **Run all**. Do not factory-reset the runtime:
the installed packages should stay in place. Internet is needed for the initial GADM download and map
backgrounds. GADM is downloaded locally for academic use under
[its terms](https://gadm.org/license.html), not redistributed. See
[DATA_SOURCES.md](https://github.com/gromicho/mobile_clinics/blob/main/DATA_SOURCES.md)
for versions, provenance and licences.
""")
    code("setup", r"""
import os, subprocess, sys
from pathlib import Path

# Colab gets the published main branch and latest stable runtime dependencies.
REPO_REF = "main"
if not Path("clinic_routing.py").exists():
    checkout = Path("_mobile_clinics_main")
    if not checkout.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_REF,
                        "https://github.com/gromicho/mobile_clinics.git", str(checkout)], check=True)
    os.chdir(checkout)
if "google.colab" in sys.modules:
    import json, tempfile
    from importlib.metadata import distributions
    # Colab preloads libraries. Replacing them on disk does not unload their
    # old Python modules or compiled extensions from this running kernel.
    installed_before = {d.metadata["Name"].lower().replace("_", "-") for d in distributions()
                        if d.metadata["Name"]}
    with tempfile.TemporaryDirectory() as temp:
        report = Path(temp) / "install-report.json"
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                               "--report", str(report), "-r", "requirements.txt"])
        replaced = sorted({item["metadata"]["name"].lower().replace("_", "-")
                           for item in json.loads(report.read_text())["install"]}
                          & installed_before)
    # Keep this flag set if setup is rerun without actually restarting.
    _COLAB_RESTART_REQUIRED = globals().get("_COLAB_RESTART_REQUIRED", False) or bool(replaced)
    if _COLAB_RESTART_REQUIRED:
        raise RuntimeError("Setup replaced installed packages. Choose Runtime > Restart session, "
                           "then Run all. Do not factory-reset the runtime. "
                           "A restart is needed before importing NumPy and the geographic libraries.")
sys.path.insert(0, str(Path.cwd()))
print("Source commit:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
""")
    code("imports", r"""
if globals().get("_COLAB_RESTART_REQUIRED", False):
    raise RuntimeError("Choose Runtime > Restart session, then Run all before importing packages.")
from dataclasses import replace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import contextily as cx
import folium
from IPython.display import display
from clinic_data import load_nyamira, road_coordinates
from clinic_routing import Settings, run_scenario
from reporting import environment_snapshot, save_results

FIGS = Path("slides/figs"); FIGS.mkdir(parents=True, exist_ok=True)
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)

def save_figure(fig, name):
    fig.savefig(FIGS / name, dpi=160, bbox_inches="tight")
    plt.show()

def basemap(ax):
    try:
        cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.Esri.WorldTopoMap,
                       zoom=11, attribution_size=5)
    except Exception as error:
        print(f"Basemap unavailable ({error}); displaying the ward and route layers.")
""")
    md("parameters-text", """
## Parameters
After changing this cell, rerun the two scenario cells. Each call rebuilds models
from deterministic data; there is no hidden model cache. Try range 5 or 20 km,
4 or 12 stops, or `hidden=(2,)`. Predict expected doses, distance and the weighted
objective separately. Changing architecture keeps the training history fixed.
""")
    code("parameters", """
cfg = Settings(range_km=10.0, stops=8, hidden=(8,), seed=42)
# Capacity: 250 doses/stop. Driving cost: 1 dose-equivalent/km. Noise sigma: .15.
# Benchmark interpolation error: at most .5 dose-equivalent over the whole route.
""")
    md("data-text", """
## 1. Geography
GADM 4.1 supplies 20 wards, WorldPop supplies 2020 population, and a cached OSM
drive graph supplies roads. Maina et al. (2019) supplies historical facility
locations. Its hospital-labelled records are **accessibility proxies**, not verified
vaccine-storage locations. The county filter excludes facilities outside the county.

Service points are ward centroids snapped to roads. Township's centroid is the
teaching depot and counts as a serviced stop; it is not the hospital's actual
coordinate. Snap offsets are reported, and off-network access distance is excluded.
Travel uses directed distances. The synthetic interaction uses their symmetric mean.
""")
    code("load-data", """
wards, facilities, hospitals, graph, node_ids, distance, base, features, start, hashes = load_nyamira()
print(f"{len(wards)} wards; {wards['pop'].sum():,.0f} people; {len(facilities)} facilities; {len(hospitals)} hospital proxies")
display(wards[["ward", "pop", "km_to_hospital_proxy", "snap_km"]].round(2))
""")
    code("geography-map", """
fig, ax = plt.subplots(figsize=(8, 7))
wards.plot(column="pop", cmap="YlOrRd", alpha=.55, edgecolor="grey", ax=ax, legend=True,
           legend_kwds={"label": "Population (WorldPop 2020)", "shrink": .6})
basemap(ax)
facilities.plot(ax=ax, color="dimgray", markersize=7, label="Facility record")
hospitals.plot(ax=ax, color="tab:blue", marker="P", markersize=65, label="Hospital proxy")
ax.scatter(wards.lon[start], wards.lat[start], marker="*", s=130, color="red", label="Starting point")
ax.set_title("Nyamira: population and historical facility locations")
ax.set_axis_off(); ax.legend(loc="lower left", fontsize=8)
fig.text(.02, .01, "Boundaries: GADM 4.1; population: WorldPop; facilities: Maina et al. 2019", fontsize=7)
save_figure(fig, "map.png")
""")
    md("simulator", r"""
## 2. A simulator with both signs of spillover
For visit indicators $y_j$, exposure is
$$E_i(y)=\min\left(1,\sum_{j\ne i}e^{-\bar d_{ij}/R}y_j\right).$$
Mean uncapped demand is $\mu_i(y)=b_i(1+\gamma E_i(y))$. Service is
$$S_i(y)=\min(C,\mu_i(y)\epsilon_i),\qquad
\epsilon_i\sim\operatorname{LogNormal}(-\sigma^2/2,\sigma).$$
The noise has **mean one**. Negative gamma represents cannibalisation; positive
gamma represents mobilisation. Base demand uses population, a simulated unvaccinated
share and the hospital-distance proxy.

This is a **set-based** effect. Visit order changes travel but not turnout. There is
no chronological patient movement or population-conservation model.

We evaluate expected capped service analytically. With standard normal CDF $\Phi$ and
$a=(\log(C/\mu)+\sigma^2/2)/\sigma$,
$$h(\mu)=\mu\Phi(a-\sigma)+C\Phi(-a).$$
Predictors learn expected **service** from noisy capped observations. Capped mean
demand and mean capped demand are different quantities.
""")
    md("learning", """
## 3. Training and held-out validation
The synthetic history contains 1500 random deployments of 0–10 wards, including
explicit zero-exposure cases. It supplies potential service at **every ward**, even
unvisited ones. This is deliberately richer than actual clinic records. Another
400 independent periods are held out.

Linear regression and a small ReLU network use log population, unvaccinated share,
accessibility and exposure. Static and embedded policies use the same fitted model.
The historical baseline averages each ward's training observations. Reported errors
use held-out outcomes and the simulator's known expectation. These are deployment
holdouts, not new-county validation. Fitting uses one numerical-library thread.
""")
    code("environment", """
environment = environment_snapshot(cfg, hashes)
display(pd.Series({k: environment[k] for k in ["cpu", "physical_cores", "logical_cores", "ram_gib",
                                              "os", "python", "gurobi", "threads", "background_load"]}))
print("Solve limit:", cfg.time_limit, "seconds; relative MIP tolerance:", cfg.mip_gap)
print("Wall times separate fitting, model construction and Gurobi solving.")
""")
    md("formulation", r"""
## 4. Routing
Choose an open path from Township with at most $K$ serviced wards, including the
depot. A one-stop route is allowed. The weighted objective differs from the
demand-threshold formulation in the thesis.
$$\max\sum_i z_i-\lambda\sum_{i\ne j}d_{ij}x_{ij},\qquad
0\le z_i\le Cy_i,\quad z_i\le\widehat h_i.$$
Binary $x$ selects arcs; binary $y$ selects wards; $z$ is service credit. No arcs enter
the depot. Each other visited ward has one incoming arc; each visited ward has at
most one outgoing arc; there is one fewer arc than stops. Disconnected cycles receive
DFJ cuts in a callback:
$$\sum_{i,j\in S:i\ne j}x_{ij}\le\sum_{i\in S}y_i-y_k,
\quad S\subseteq N\setminus\{s\},\ k\in S.$$

| Policy | Service credit |
|---|---|
| Static linear / MLP | Predictor with exposure fixed at zero |
| Embedded linear / MLP | Same predictor, exposure determined by the selected wards |
| Historical mean | Mean service at each ward over the training history |
| Simulator benchmark | Known expected service, with bounded interpolation error |

Predictions are clipped to $[0,C]$ consistently. The benchmark uses concave secants
with total error at most `benchmark_error`. Adding this error to the solver bound
gives an upper bound on the optimal expected objective. Every returned route is
evaluated using the **exact expectation**. Status, gap, model sizes and timings remain
visible. [Implementation](https://github.com/gromicho/mobile_clinics/blob/main/clinic_routing.py).
""")
    code("scenario-function", """
def scenario(gamma, settings):
    result, validation, example = run_scenario(distance, base, features, start, gamma, settings)
    display(validation.round(3))
    display(result[["policy", "expected_service", "km", "expected_objective", "doses_per_km",
                    "status", "mip_gap", "expected_objective_upper_bound"]].round(3))
    display(result[["policy", "variables", "linear_constraints", "general_constraints", "lazy_cuts",
                    "build_seconds", "solve_seconds"]].round(3))
    return result, validation, example
""")
    md("scenario-a-text", "### Cannibalisation: gamma = -0.5\nCompare each embedded policy with its static counterpart. What service gain requires extra travel?")
    code("scenario-a", """
resA, validationA, exampleA = scenario(-0.5, cfg)
display(exampleA.round(3))  # actual training rows, not invented examples
""")
    md("scenario-b-text", "### Mobilisation: gamma = +0.5\nCompare distance and service together. Embedding a fitted model does not guarantee an improved expected objective.")
    code("scenario-b", "resB, validationB, _ = scenario(+0.5, cfg)")
    code("comparison-chart", """
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for row, (frame, title) in enumerate([(resA, "Cannibalisation"), (resB, "Mobilisation")]):
    for ax, column, unit in [(axes[row, 0], "expected_service", "Expected doses"),
                             (axes[row, 1], "expected_objective", "Expected doses minus driving cost")]:
        ax.barh(frame.policy, frame[column], color=["#a7b5c5", "#277f8e", "#a7b5c5", "#277f8e", "#c8b886", "#444444"])
        ax.set_title(title); ax.set_xlabel(unit); ax.invert_yaxis()
fig.tight_layout(); save_figure(fig, "bars.png")
""")
    code("route-maps", """
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for row, (frame, title) in enumerate([(resA, "Cannibalisation"), (resB, "Mobilisation")]):
    for ax, policy in zip(axes[row], ["static MLP", "embedded MLP", "benchmark"]):
        result = frame.set_index("policy").loc[policy]
        wards.plot(color="#f5e9c9", alpha=.5, edgecolor="grey", linewidth=.5, ax=ax)
        basemap(ax)
        xy = road_coordinates(graph, node_ids, result.order)
        ax.plot(xy[:, 0], xy[:, 1], color="#145f73", linewidth=2)
        ax.scatter(wards.lon.iloc[result.order], wards.lat.iloc[result.order], color="#145f73", s=28)
        ax.scatter(wards.lon[start], wards.lat[start], color="red", marker="*", s=100)
        ax.set_title(f"{title}: {policy}\\n{result.expected_service:.0f} expected doses; {result.km:.1f} km", fontsize=10)
        ax.set_axis_off()
fig.text(.01, .005, "GADM boundaries; roads © OpenStreetMap contributors (ODbL). Synthetic outcomes.", fontsize=7)
fig.tight_layout(rect=[0, .02, 1, 1]); save_figure(fig, "routes.png")
""")
    md("sweep-text", """
## 5. Spillover sensitivity
All strengths use the same underlying deployment/noise draws. Each predictor gets
the same histories, and evaluation is analytic. The benchmark band includes
interpolation and solver uncertainty. Results are not assumed to coincide at zero
spillover: estimation error can still affect decisions.
""")
    code("sweep", """
sweep_rows, sweep_validation = [], []
for gamma in [-.8, -.5, -.25, 0, .25, .5, .8]:
    frame, validation, _ = run_scenario(distance, base, features, start, gamma, cfg)
    sweep_rows.append(frame); sweep_validation.append(validation.assign(gamma=gamma))
sweep = pd.concat(sweep_rows, ignore_index=True)
fig, ax = plt.subplots(figsize=(10, 5))
for policy, frame in sweep.groupby("policy", sort=False):
    ax.plot(frame.gamma, frame.expected_objective, marker="o", label=policy)
bench = sweep[sweep.policy.eq("benchmark")]
ax.fill_between(bench.gamma, bench.expected_objective, bench.expected_objective_upper_bound,
                alpha=.2, color="black", label="Benchmark bound interval")
ax.set(ylabel="Expected doses minus driving cost")
for x, label in [(-.8, "Cannibalisation"), (0, "No interaction"), (.8, "Mobilisation")]:
    ax.text(x, -.12, label, transform=ax.get_xaxis_transform(), ha="center", va="top")
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.2); fig.tight_layout(); save_figure(fig, "sweep.png")
display(sweep.pivot(index="gamma", columns="policy", values="expected_objective").round(2))
""")
    md("experiments-text", """
## 6. Parameter experiments and training-history sensitivity
These tables report the run's outcomes without hardcoded answers. Architecture
changes retain the same training data. Three history seeds test training variability
on this fixed synthetic ward profile; they are a small teaching check.
""")
    code("parameter-experiments", """
experiments = {"baseline": cfg, "range 5": replace(cfg, range_km=5), "range 20": replace(cfg, range_km=20),
               "4 stops": replace(cfg, stops=4), "12 stops": replace(cfg, stops=12), "2 hidden units": replace(cfg, hidden=(2,))}
answer_frames = []
for experiment, settings in experiments.items():
    frame, _, _ = run_scenario(distance, base, features, start, -.5, settings)
    answer_frames.append(frame.assign(experiment=experiment))
answers = pd.concat(answer_frames, ignore_index=True)
display(answers.pivot(index="experiment", columns="policy", values="expected_objective").round(2))
""")
    code("history-sensitivity", """
replicates = []
for seed in [42, 43, 44]:
    for gamma in [-.5, .5]:
        frame, _, _ = run_scenario(distance, base, features, start, gamma, replace(cfg, seed=seed))
        replicates.append(frame.assign(history_seed=seed))
replicates = pd.concat(replicates, ignore_index=True)
display(replicates.groupby(["gamma", "policy"]).expected_objective.agg(["mean", "std", "min", "max"]).round(2))
""")
    md("online-map-text", """
## 7. Online route map
This uses Esri tiles and normal Folium web assets. Toggle routes in the layer
control. It includes markers and OSM road paths, without embedding GADM boundaries.
""")
    code("online-map", """
route_map = folium.Map(location=[wards.lat.mean(), wards.lon.mean()], zoom_start=11,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles © Esri; routes © OpenStreetMap contributors (ODbL)")
for policy, color in [("static MLP", "red"), ("embedded MLP", "green"), ("benchmark", "blue")]:
    result = resA.set_index("policy").loc[policy]
    layer = folium.FeatureGroup(name=f"{policy}: {result.expected_service:.0f} expected doses, {result.km:.1f} km")
    xy = road_coordinates(graph, node_ids, result.order)
    if len(xy) > 1:
        folium.PolyLine(xy[:, ::-1].tolist(), color=color, weight=4).add_to(layer)
    for stop, i in enumerate(result.order):
        folium.CircleMarker([wards.lat[i], wards.lon[i]], radius=5, color=color,
                            tooltip=f"{stop}: {wards.ward[i]}").add_to(layer)
    layer.add_to(route_map)
folium.LayerControl().add_to(route_map)
route_map
""")
    code("save-results", """
save_results(RESULTS, environment, cfg, {"scenario_a": resA, "scenario_b": resB,
    "validation_a": validationA, "validation_b": validationB, "training_example": exampleA,
    "sweep": sweep, "sweep_validation": pd.concat(sweep_validation),
    "parameter_experiments": answers, "history_sensitivity": replicates})
print("Saved full-precision results and environment metadata to", RESULTS)
""")
    md("limitations", """
## What this supports
Decision-dependent outcomes can change routing decisions. Matched comparisons
separate embedding from predictor choice. Better forecasts, better doses/km and
better weighted objectives remain different claims; embedding a misspecified
forecast can also worsen the decision.

Observed-record applications must address missing unvisited outcomes, capacity
censoring, non-random historical assignments, supported exposure and spillover
identification. New geography requires validation and actual service/depot locations.
Cold-chain operations, time budgets and fairness require explicit constraints.

The simulator benchmark supports conclusions only about this synthetic problem,
within reported interpolation and solver bounds. It does not estimate real causal effects.
""")
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


if __name__ == "__main__":
    nbf.write(make_notebook(), Path(__file__).with_name("mobile_clinic_routing.ipynb"))
    print("Wrote mobile_clinic_routing.ipynb")
