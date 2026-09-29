# Data sources and reuse

The teaching outcomes, unvaccinated shares and spillover mechanism are simulated.
They are not observed AMREF results. Original code is MIT-licensed; the following
third-party materials have separate terms. Each execution records SHA-256 hashes
of its actual inputs in `results/environment.json`.

| Input | Version and source | Terms and treatment |
|---|---|---|
| Ward boundaries | GADM 4.1 Kenya, level 3; [download](https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_KEN_3.json.zip) | [GADM terms](https://gadm.org/license.html): local academic/noncommercial use; redistribution requires permission. Downloaded on demand and ignored by Git. No boundary coordinates in the interactive map or published notebook outputs. Academic static maps follow GADM's map-publication allowance. |
| Population | WorldPop 2020, Kenya, 1 km, UN-adjusted; [GeoTIFF](https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/2020/KEN/ken_ppp_2020_1km_Aggregated_UNadj.tif) | WorldPop / University of Southampton and contributing institutions; [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cached input; ward totals use raster pixel-centre inclusion. |
| Roads and older comparison facilities | OpenStreetMap contributors, Nyamira graph created 2026-09-24 with OSMnx 2.0.3; graph metadata records the creation time | [ODbL](https://opendatacommons.org/licenses/odbl/1-0/) and [OSM attribution](https://www.openstreetmap.org/copyright). Cached GraphML and GeoJSON retain these terms. Derived route coordinates use OSM data. |
| Facility records | Maina et al. (2019), *A spatial database of health facilities managed by the public health sector in sub Saharan Africa*, Scientific Data 6:134; [paper](https://doi.org/10.1038/s41597-019-0142-2), [dataset](https://doi.org/10.6084/m9.figshare.7725374), [workbook](https://ndownloader.figshare.com/files/14379593) | CC0 dataset. The cached CSV is the Nyamira point-in-polygon subset, renamed to five columns. Hospital labels are an accessibility proxy, not verified cold-chain capability. The historical list does not establish current facility status. |
| Online backgrounds | Esri World Topographic Map via contextily and Folium | Provider attribution appears on maps; tiles are fetched online. Follow [Esri's terms](https://www.esri.com/en-us/legal/terms/full-master-agreement). No offline tile bundle is distributed. |
| UvA and Analytics for a Better World logos | Existing supplied deck assets | Institutional logos/trademarks are outside the code licence. |

The original WorldPop/facility cache retrieval timestamps were not recorded and
are not reconstructed from the present machine's clock. Their exact bytes are
identified by the run manifest and the repository commit. Newly downloaded GADM
files remain local under `data/`; its ZIP and JSON are both ignored. CI downloads
them for execution but excludes both the data cache and the executed notebook from
uploaded artifacts. Earlier Git commits are not rewritten by this maintenance change.

The example is intentionally specific to Nyamira. Other counties need a deliberate
choice of service points, depot, graph extent, facilities, projection and model size.
