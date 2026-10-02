# Mobile clinic routing with learned constraints

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gromicho/mobile_clinics/blob/main/mobile_clinic_routing.ipynb)
[![Notebook validation](https://github.com/gromicho/mobile_clinics/actions/workflows/notebook.yml/badge.svg)](https://github.com/gromicho/mobile_clinics/actions/workflows/notebook.yml)

A SIKS teaching companion to Chapter 6 of Mayukh Ghosh's thesis, joint work with
Chintan Amrit and Joaquim Gromicho. The county geography is real; the vaccination
outcomes, unvaccinated shares and spillover mechanism are synthetic. This example
illustrates decision-dependent prediction, rather than reproducing the thesis's
experiments or demonstrating a real AMREF treatment effect.

The notebook compares **static and embedded versions of the same linear and neural
network predictors**, a historical-mean baseline, and a simulator benchmark. Every
route is evaluated using exact expected service after applying the capacity limit.
The benchmark's solver bound includes an explicit interpolation-error allowance.
Embedding can improve a decision, but the example does not assume it always wins.

## Run

In Colab, open the badge and run top to bottom. Setup obtains the published `main`
branch and installs the latest stable runtime packages, with only API minimums.
It leaves Colab's notebook kernel tools in place. If setup replaces existing
packages, it stops with a restart instruction: choose **Runtime → Restart session**,
then **Run all**. This keeps newly installed packages while clearing stale NumPy
and other compiled-library imports. Do not factory-reset the runtime. GADM boundaries download
automatically from their publisher into an ignored local cache. Internet access is
required for that initial download and the online map backgrounds; no API keys are
needed. The interactive map uses normal Folium/Esri online loading.

Locally, use Python 3.12 or 3.13 in a virtual environment:

```sh
python -m pip install --upgrade -r requirements-dev.txt
python scripts/execute_notebook.py
```

For the exact direct-package versions used for the recorded slides, use Python
3.12 and `python -m pip install -r requirements-reproducible.txt` instead.
Those versions are preserved for reproduction; normal setup does not force them.
New runs record their actual versions in `results/environment.json` and may differ
from the saved presentation results. CI checks the latest stable dependencies on
both Python 3.12 and 3.13.

The maintained notebook has stable cell IDs and no saved outputs. Full execution
writes its executed copy under ignored `build/`, full-precision results under
`results/`, and static figures under `slides/figs/`. Committed figures, JSON results
and [the slide PDF](slides/intro.pdf) provide a reviewable snapshot. The executed
notebook is not published as a data bundle.

The slides include the complete model in their backup section. A
[slide-by-slide presenter narrative](slides/presenter_narrative.md) accompanies
the main talk and the technical backup.

For the live demo, run the setup once, then change `Settings(...)` and rerun the two
scenario cells. The sweep, parameter table and repeated-history check take longer.
There is no hidden model cache and no shared evaluation RNG; the same inputs produce
the same training history, independent of which other scenarios ran first.

## Scientific choices

- Objective: expected doses served minus `cost_per_km * distance`; capacity is per
  stop, not a total carried vaccine inventory. Doses/km is a separate metric.
- Service points are road-snapped ward centroids. The Township centroid is the
  teaching depot, counts towards the stop budget, and receives service. A one-stop
  route is supported. Operational use requires actual sites and collection points.
- Directed distances determine travel. Symmetric mean distance defines the
  assumed set-based interaction. The model does not track chronological patient
  movement, vaccine conservation or cold-chain operations.
- Hospital-labelled Maina records are accessibility proxies, not verified vaccine
  stores. The historical county subset omits facilities outside the boundary.
- Training histories supply potential capped service at every ward, including
  unvisited wards, with explicit zero-exposure cases. This is a full-information
  simulation assumption. Real records require handling missingness, censoring,
  selection and spillover identification before applying this method.
- Validation uses independent held-out deployment periods. A three-seed teaching
  check measures training-history sensitivity on the same synthetic ward profile.
  It is not geographic validation or a broad empirical benchmark.
- Predictors learn capped service, not uncapped demand. Mean-one lognormal noise
  and analytic evaluation keep the statistical and optimisation objectives aligned.
- The simulator benchmark uses concave secants with total objective error at most
  0.5 dose-equivalent by default. Its reported upper bound adds that error to the
  solver bound. Solved routes retain status, gap and unrounded metrics.

## Rebuild and check

```sh
python scripts/notebook_preflight.py mobile_clinic_routing.ipynb
python -m unittest discover -s tests -v
python scripts/check_data_download.py
python scripts/execute_notebook.py
python scripts/check_results.py
```

`python scripts/rebuild.py` runs the complete validation, numerical/figure export,
facility comparison and Beamer build; install `pdflatex` to rebuild the PDF.
`slides/content.tex` contains editable slide content; `scripts/build_deck.py`
generates its tables and environment statement from the notebook's actual results.

The GitHub workflow is configured for Linux/Python 3.12 and 3.13 and the restricted licence bundled with
`pip install gurobipy`. It first verifies that this licence really rejects a model
above its size limit, then runs the small-instance tests and every notebook cell.
The test is not a substitute for opening the actual Google Colab UI. An existing
local academic licence is also supported. Variables, linear constraints, general
constraints and callback cuts are reported separately; requests exceeding the
teaching model-size budget receive a clear error.

`results/environment.json` records hardware, package versions, source/input hashes,
seeds, solver settings and timing scope. Each run records its own environment;
historical timings are not reassigned to today's machine. No cross-machine speedup
claim is made. Normal setup permits latest stable dependencies; exact versions for
the recorded slides are retained in requirements-reproducible.txt. Each new run
records the versions actually used in its manifest.

## Files and licences

- `mobile_clinic_routing.ipynb`: maintained teaching notebook.
- `build_notebook.py`: deterministic notebook source, including stable cell IDs.
- `clinic_routing.py`: simulator, validation, predictors and routing models.
- `clinic_data.py`: downloads, geospatial preparation and route coordinates.
- `reporting.py`: environment and result export.
- `tests/`, `scripts/`: mathematical, execution and publication checks.
- `data/`: redistributable caches. **GADM ZIP/JSON files are ignored and downloaded
  locally**, subject to [GADM's terms](https://gadm.org/license.html).
- `results/`, `slides/`: numerical evidence and the presentation.

Original software/documentation uses the [MIT licence](LICENSE). Data, map tiles,
logos and trademarks retain their own terms: see [DATA_SOURCES.md](DATA_SOURCES.md).
