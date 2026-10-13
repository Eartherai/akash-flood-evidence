# Akash Flood

## New bounded area/date workflow, 12 October

The [local execution console and multimodal workflow](release/MULTIMODAL-WORKFLOW-20261012.md) now connects raw radar processing, fixed S2 observation screening, generic observation-aware graph bounds and combined reports. A disclosed cached historical/radar-source case completed 14 stages in 171.07 seconds; new S2 acquisition was real. Optical coverage was only 11.10%, and live historical OSM attempts failed. This is a working bounded processing tier, not arbitrary-world support, hazard-accuracy validation or full Track B eligibility. Older kits and scientific results remain unchanged.


## Observation-aware hardening - 11 October 2026

The new [dated evidence note](release/HARDENING-EVIDENCE-2026-10-11.md) records the completed fixed optical selection (269,363 assessed cells, 38.10%, with 437,628 unknown) and a separate graph scenario that makes unknown/outside road support uncertain. All ten formerly robust Rasuwa statements become ambiguous; Kathmandu changes from 69 candidate-only robust statements to 28 scenario-connected and 41 ambiguous. Independent proofs retain those assumptions; they do not establish closures or safe routes. Both checking coverage gates still fail. The new [six-page report](output/pdf/akash-technical-evidence-report-2026-10-11.pdf) and [one-page evidence brief](output/pdf/akash-situation-evidence-brief-2026-10-11.pdf) have source bindings and rendered-page QA. Earlier results below retain their original dates and scope; human validation and public publication remain pending.

## Complete-cohort and native-setup upgrade - 10 October 2026

The new optical composite improves actual assessed support from **11.25% to 37.49%**, retaining all 79,555 original assessed cells and adding 185,492. Its 3,572 exposed-surface/vegetation-loss candidates are provisional; 441,944 cells remain unknown. A frozen SCL-only per-cell date selection uses 13 sources; invalid or unacquired selected reflectance remains unknown. The separate CPU run took 100.38 s, with 121.36 MB new body bytes and 246.0 MB peak RSS. [Method and independent audit](research/optical-composite-2026-10-10/review.md).

The [native-cell date viewer](outputs/optical-composite-viewer-2026-10-10/index.html), served at `http://127.0.0.1:8798/`, shows exact source IDs/dates, per-cell ages and gained support. Pre-event assessed ages are 2/4/14 days and post-event ages 1/6/21 days. The composite is separate from the preserved original radar, optical pair and road scenario. Browser verification found correct candidate/added/unknown states and source dates; no acquisition or model call occurs in this saved viewer.

A separately frozen checking-only comparison intersects **16.90% of the whole mapped EMSR927 reference extent**, with **61.66% checking-domain** and **27.44% reference coverage**. Both unchanged 90%/95% gates fail. This is a known-event geometric comparison between different classes, not debris precision, recall, IoU, completeness or accuracy. An independent per-cell area check agreed within 4.38e-8 m2. [Checking review](research/evaluation-only/optical-composite-2026-10-10/review.md). Damage maps remain excluded from production and copilot inputs.

The [two-case CPU copilot](research/multi-case-copilot-review-2026-10-10.md) at `http://127.0.0.1:8797/` explicitly switches between Rasuwa and Kathmandu, uses source-bound English/Nepali clauses and marks unavailable Kathmandu buildings correctly. Twenty measured calls cover 12 distinct known development questions: v1 status/inclusion 8/12, source/case 12/12; corrective v2 status/inclusion/source 6/6, strict relevance 4/6; focused v3 repeated 2/2 known relevance failures. Runs are not pooled as blind validation. Three extra real browser questions verified English Rasuwa and Nepali Kathmandu source citations; human Nepali review remains pending. The copilot deliberately keeps the dated original Rasuwa optical pair separate from the new composite.

The new [macOS arm64 native bootstrap](release/native-setup/README.md) passed actual fresh-source installation, relocation, independent SNAP/JVM proof and frozen-model synthetic inference with outbound networking and original-workspace reads denied. It uses exact official cached SNAP payloads, pinned auxiliaries and separately supplied unchanged checkpoint; no vendor binary/weights are bundled in the source ZIP. Corrected DEM max error 8.33e-10 m across 116 samples per JVM; deterministic 64x64 model inference retained 32 unknown cells. This establishes prerequisite portability, not a cold real-satellite replay, absolute geolocation or hazard accuracy. [Native review](research/native-setup-review-2026-10-10.json).

The stable source suite passed **883 tests in 24.66 s**, with 2,241 existing/fixture warnings. New code retains previous scientific artifacts and failures. The latest [six-page report](output/pdf/akash-technical-evidence-report-2026-10-10.pdf) has rendered-page visual verification and exact source bindings. Public GitHub/YouTube/Devpost links, real human eligibility and complete unrestricted raw execution are not claimed. Original project code is now MIT-licensed; upstream assets retain their separate terms. Lumen Track A remains the stronger track candidate for this single human team; these local alternatives do not permit two entries from one person.


## Kathmandu access and submission update - 8 October 2026

The second case now has a separate, verified historical-road/access view at `http://127.0.0.1:8794/`, exported to `outputs/kathmandu-access-report-final-2026-10-08/index.html`. It preserves the original 361.09-second water-only execution and its unchanged rasters. A later CPU mapping stage took **487.77 seconds**, with no new network or inference in that stage. Buildings remain **unavailable**, not zero; this is a category-limited supplement, not a complete second-case building/debris assessment.

After two size-limited failures, one separately declared driving-road/node-POI query acquired the **1 September 2024** historical snapshot. Its successful body was **29,375,561 bytes** in **29.20 seconds**; the three size-bounded attempts together delivered **69,381,129 body bytes** in **109.81 seconds** of summed request time. Earlier failed all-category requests and metadata probes are separate. There is no current-OSM fallback or independent historical-server authenticity claim.

The source graph contains **253,451 nodes, 519,044 directed edges and 261,690 physical road segments**. Exactly **293 physical segments / 586 directed edges** intersect new-water candidates. All **69 source settlements** retain a saved path when those segments are hypothetically closed: **36** settlements lie inside the water AOI, **68** inside the extraction context, and one arrives as an incidental way-node dependency outside it. The **128 destinations** are mapped node-tagged town/city/hospital candidates, with operating status and physical entrances unknown. All other baseline edges, including unknown/outside-AOI portions, are assumed available. These results do not establish safe passage or observed closures.

Independent review streamed all feature records, recomputed all 261,690 native-cell corridor intersections, and separately checked every nearest fixed destination and all 69 paths. The renderer rechecks source bindings, water counts, source groups, nearest targets and witnesses before exporting. The map draws source centerlines and a selected saved path clipped to exact AOI cells; it omits the full baseline network from the view while retaining it in the analysis. See [the source and graph review](research/kathmandu-core-access-review-2026-10-08.md).

The complete suite passed **814 tests in 23.94 seconds** with **2,241 warnings**, comprising existing Affine deprecations and an intentional duplicate-ZIP fixture warning:

```sh
uv run --with pyshp==2.3.1 --with scikit-learn==1.7.2 python -m pytest -q
```

These are software checks, not hazard accuracy. The matched three-minute English-subtitled demonstration project is in `../videos/akash-track-b-demo`; its own source/capture and render receipts record the actual assets and encoding outcome. It uses actual local model/UI footage and labels saved screenshots explicitly. Public GitHub/YouTube/Devpost URLs, human eligibility/account information, complete raw-runtime portability and validated local flood/debris skill remain outstanding.

The updated six-page submission report is `output/pdf/akash-technical-evidence-report-2026-10-08.pdf`. It includes this later Kathmandu access stage and the dated 814-test receipt. Its adjacent provenance file binds the sources and records review of all six rendered pages. The 7 October PDF and saved-case bundle remain preserved at their original paths.

The source-only local review candidate is documented in [release/README-SOURCE.md](release/README-SOURCE.md). It excludes satellite/checking assets, model weights, generated vendor artifacts and prior case outputs. Its relocated synthetic fixture checks software installation and the declared path/cut algorithm; it does not reproduce either real case from raw data. On 11 October, the human owner confirmed that the original code is theirs and selected the [MIT license](LICENSE). Public repository publication is still pending; third-party assets retain their separate terms.

## Checking source audit - 8 October 2026

The planned Kathmandu reference comparison could not be executed: both GIS archives linked by UNOSAT product 3993 lack its September 30 Spot-6 Kathmandu flood layer. Their metadata contains regional Sentinel-1 layers instead. The acquired archives, hashes, schema inventories and failed first GDB parser guard are preserved separately under checking-only research; no regional source was substituted and no accuracy/overlap result is claimed. See [the source outcome](research/kathmandu-checking-source-outcome-2026-10-08.md).

A separate native-cell area checker and bounded source transport are implemented, with 22 focused geometry/transport tests. At this earlier source-checking milestone the complete suite passed **715 tests** in 31.35 seconds using `uv run --with pyshp==2.3.1 --with scikit-learn==1.7.2 python -m pytest -q`. Its 2,211 warnings comprise existing deprecations and an intentional duplicate-ZIP fixture warning. These are software checks, not flood validation. Original model, Kathmandu outputs, access algorithms and portable judge bundle remain unchanged; raw checking archives/extracted files are excluded from Git and the production bundle.

## Submission artifacts and latest checks - 7 October 2026

The latest case-aware English/Nepali copilot is at `http://127.0.0.1:8790/`. It uses the real pinned Qwen model on CPU to select verified facts from the current water, optical and conditional-access artifacts. Its first measured run passed 13/16 inclusion/status checks and had one timeout, omitted facts and irrelevant extra clauses. After a separately documented retrieval/prompt correction, a repeated six-question development set passed 6/6 with measured request times of 4.92-10.08 seconds. One unsupported safety request needed an explicit policy correction. These are known, AI-authored development questions, not blind evaluation or human Nepali QA. Both failed earlier runs are preserved. See [the current copilot guide](research/current-case-copilot.md) and [first-run review](research/fresh-copilot-first-run-review.md).

The portable judge viewer is `output/release/akash-saved-case-replay-2026-10-07.zip` (2,741,950 bytes). It preserves 22 original artifacts, verifies relative-path checksums and runs with Python's standard library. Actual extraction outside this workspace passed with original workspace access and outbound/model operations blocked. It displays saved evidence and reports; it does not run the copilot or reproduce raw satellite processing. See [the release review and launch instructions](research/portable-release-review-2026-10-07.md).

The six-page technical evidence report is now at `output/pdf/akash-technical-evidence-report.pdf`, with its source/output SHA256 and visual-review record beside it. It reports the actual model benchmark, both CPU case executions, infrastructure/access assumptions, failed reference comparison and release limitations. It is a working technical deliverable; completing the report does not establish flood/debris accuracy or a finished submission.

The current one-page situation report is `output/pdf/akash-attributed-situation-report.pdf`. Both PDFs were rendered and visually reviewed. The updated offline dashboard is `outputs/fresh-report-attributed/report.html`; these exports add the full WorldDEM/OSM notices and the actually used Sen1Floods11 training citation. Earlier outputs and execution receipts are preserved.

The later historical-access adapter verifies completed water against its original processing plan and binds a separate pre-event OSM source without editing that plan or its elapsed-time receipt. Its source/transport and supplement checks passed independently. Both actual Kathmandu historical-map acquisition attempts failed, so no Kathmandu access result is claimed. See [the independent supplement review](research/supplemental-access-review.md).

A separate SCL-only observation gate was implemented and tested, but the Rasuwa acquisition was deliberately not repeated: the existing 11-source raw-clear union bounds the new eligible eroded-land gate above by **59.36%**. It cannot satisfy a 70% scene-coverage claim. This is an observation opportunity bound, not a new debris prediction; frozen optical outputs remain unchanged.

The complete suite recorded at this 7 October milestone passed **693 tests**, with deprecation warnings only. It includes the source adapters, HTTP boundaries, focused retrieval/policy corrections and saved-case packaging. None of these software checks measures hazard accuracy. The technical PDF preserves its earlier 563-test generation state; the [readiness audit](research/submission-readiness-2026-10-07.md) also captures an earlier inspection. These paragraphs record the subsequently completed report, attribution, copilot and portable viewer work without backdating those artifacts.

Local CPU prototype for satellite water-change screening and road-access scenarios. **This is not yet a complete Track B submission.** No flood accuracy, real-world road closure, or rescue capability has been established.

## Reproducible case workflow — 7 October 2026

One command now completes cached native Sentinel-1 acquisition/import, Copernicus DEM preparation, local terrain correction, unchanged-model inference, historical feature intersections, conditional road-access proofs and an offline report. The measured Rasuwa replay took **48.055 seconds on CPU**, with **zero new network requests or body bytes**. Its eight saved water TIFFs match the preceding generic replay byte for byte. Cold download time, model fitting and independent event accuracy are separate. See the [runner guide](research/fresh-case-runner.md) and saved `outputs/fresh-one-command-replay/execution-receipt.json`.

```sh
uv run --with scikit-learn==1.7.2 python scripts/run_fresh_case.py \
  --bbox 85.30 28.12 85.43 28.32 \
  --context-bbox 85.27 28.08 85.46 28.36 \
  --event-date 2026-08-26 --osm-snapshot 2026-07-27 \
  --catalog outputs/fresh-public-catalog/catalog.json \
  --osm data/rasuwa-pilot/osm.json \
  --osm-provenance data/rasuwa-pilot/osm.provenance.json \
  --source-cache data/s1-snap-native --with-report \
  --out outputs/example-new-case --execute
```

The prepared portable SNAP runtime, independently proved local DEM-reader overlay, pinned geoid and frozen model must already be available. These large/derived assets are excluded from Git; this checkout is not yet a complete fresh-install distribution. Omitting `--execute` writes an offline execution plan and starts no subprocess or network. Every attempt requires a new output path. New source acquisition can omit the catalogue, OSM and native cache arguments, within explicit download/time/grid limits. `--omit-osm` supports water inference alone. See the runner guide for supported geography, pairing, resource caps and prerequisites.

The generic replay assesses **1,869,789/2,827,950 cells (66.12%)**, leaving **958,161 unknown**. It produces **30,193 new-water** and **32,551 lost-water classification candidates**. Its explicit context and generic observation policy differ from the original Rasuwa application below; the small count changes are not an accuracy improvement. New-water candidates intersect **313 physical road segments and 14 mapped building polygons**, with zero bridge intersections. Using those roads as hypothetical open/closed variables yields **22 ambiguous, 10 robust-connected, 20 unmapped and 3 baseline-disconnected settlement cases**. **32 paths and 22 cuts** passed independent source-edge checks. These certify the declared graph scenarios, not damage, closure or safe routing. Details are in the [exposure/report review](research/fresh-exposure-report-review.md).

A separate Sentinel-2 screen now reads actual blue/red/NIR/SWIR reflectance, applies each asset's declared scale/offset, aggregates only exact all-valid 2x2 native samples to 20 m and excludes cloud/shadow/snow/water/invalid neighborhoods. The fixed NDVI-loss/BSI-change thresholds are untuned engineering hypotheses. It assesses **79,555/706,991 cells (11.25%)**, leaves **627,436 unknown**, and flags **1,098 exposed-surface/vegetation-loss candidate cells (43.92 ha of classifications)**. Both original 70% single-scene optical gates still fail. Agriculture, vegetation removal, riverbank change and sensor errors can trigger candidates; no debris accuracy is established. Six completed crops were explicitly reused from a preserved failed transfer, with retrospective crop hashes and incomplete whole-object provenance. The acquisition campaign recorded **66,934,035 body bytes**; 60 MB was a per-new-run cap. Request receipts count successful ranges, not every failed HTTP attempt. See the [independent optical review](research/optical-surface-change-review.md).

Checking-only comparison against the preserved EMSR927 mass-movement inventory **rejects this optical screen as a sufficient debris map**. Exact native-cell polygon intersection assesses only **25.10% of the checking domain and 0.94% of the mapped reference extent**. Candidates intersect **0.30% of the whole mapped reference extent**. Both unchanged 90%/95% coverage gates fail. Just nine candidate cells fall inside the checking domain; the apparently high fraction of that tiny candidate area inside the inventory is not classifier precision or accuracy. Candidate and reference classes differ, prior reference outcomes were already known, and the inventory is not an exhaustive vegetation-change label. No thresholds were changed after this check. Protocol/result are isolated in `research/evaluation-only/optical-extent-protocol.frozen.json` and `optical-extent-checking-result.json`; they are excluded from production and the copilot.

The combined offline report keeps the radar and optical native grids separate. Optional optical candidates do not modify the road scenario. Each settlement button draws saved routes and cuts clipped to the exact AOI cells while the expanded graph remains in the analysis. All displayed raster counts, source hashes, grid signatures and optical SCL coverage are checked before export. The earlier one-page PDF `output/pdf/akash-multisensor-case-report.pdf` and HTML `outputs/fresh-report-current/report.html` are preserved. The current attributed exports are `output/pdf/akash-attributed-situation-report.pdf` and `outputs/fresh-report-attributed/report.html`, as recorded above.

The unchanged water model also completed a **second application case in Kathmandu**, with inputs and policy frozen before new pixel acquisition. Using September 20 / October 2, 2024 Sentinel-1A descending orbit 19 observations, the acquisition-through-water run took **361.09 seconds on CPU**, including **2,702,627,011 new radar-source body bytes** and **44,874,244 new DEM body bytes**. The prepared runtime, geoid and checkpoint were reused; catalogue discovery time/bytes are separate. The case assesses **2,054,876/2,185,368 cells (94.03%)**, leaves **130,492 unknown**, and produces **4,565 new-water and 4,350 lost-water classification candidates**. No checkpoint fit, retuning or reference labels were used. This demonstrates transfer of the processing/application workflow, not flood accuracy. The October 2 observation is four days after the chosen September 28 peak, and this water-only run omits historical OSM, exposure and a second-case access report. Source freeze: `research/kathmandu-inference-protocol.json`; actual receipt: `outputs/kathmandu-unseen-water/execution-receipt.json`.

Its verified water-only application export is `outputs/kathmandu-water-report/index.html`, served at `http://127.0.0.1:8787/`. The renderer verifies the protocol timestamp, source/code hashes, unchanged checkpoint, exact receipt-to-water-output binding, canonical historical cutoff and exclusion of the multi-day event window. It independently recounts the water states before displaying them; no model inference or accuracy evaluation is repeated for rendering. This application page is separate from the Rasuwa exposure/access report.

```sh
uv run python scripts/build_fresh_report.py \
  --water-manifest outputs/fresh-one-command-replay/water/manifest.json \
  --exposure-manifest outputs/fresh-one-command-replay/exposure/source-and-output-manifest.json \
  --optical-manifest outputs/s2-surface-change-resumed/manifest.json \
  --out outputs/example-new-report
uv run python -m http.server 8786 --bind 127.0.0.1 --directory outputs/fresh-report-current
```

Then open `http://127.0.0.1:8786/report.html`. This report has no CDN or model dependency. The separate older dashboard and CPU copilot remain at `http://localhost:8783/`. The verified complete suite passed **509 tests** with deprecation warnings only:

```sh
uv run --with scikit-learn==1.7.2 python -m pytest -q
```

## Preserved radar/corrector milestone — 6 October 2026

Anonymous public Sentinel-1 pixels now work on this Mac without a Copernicus account or GPU. The native path wraps public provider COGs and their matching annotations in a checksum-recorded **derived** SAFE, removes thermal noise without clipping negative corrected powers, calibrates sigma0, and performs geometric terrain correction in a project-local ESA SNAP runtime. It uses embedded orbit vectors; precise-orbit correction and radiometric terrain flattening have not been established.

Track B lists **Copernicus DEM**, not SRTM. The earlier SRTM run is preserved as engineering evidence. A fresh Copernicus GLO-30 run uses explicit `h = H_EGM2008 + N_EGM2008` with a checksum-pinned, mandatory local NGA geoid grid. Independent review reproduced a half-source-pixel sampling error in the installed SNAP external DEM reader. A narrowly documented local overlay corrected it while preserving vendor jars and source geographic centers. Actual JVM tests covered 116 analytic samples, and 453 nominal-center samples per date in the fresh terrain outputs agreed with the declared DEM within 0.000243 m. This verifies numerical DEM sampling, not absolute radar geolocation or hazard accuracy. The correction is not an ESA-endorsed release. See `research/copdem-processing-review.md`.

The unchanged candidate screen performs poorly against the evaluation-only landslide map: the final numerical corrective comparison intersected **1.46%** of mapped event cells and failed the predeclared coverage gates. This is radar-change overlap, not classifier accuracy. Both prior outcomes were known before the numerical corrective repeat; it is not a blind holdout. No thresholds or screening code were changed from the original freeze. Reference labels and metrics are excluded from the production pipeline and assistant.

The corrected real screen assesses 1,865,827 of 2,827,950 AOI cells (65.98%), leaving 962,123 unknown. It flags 1,602 change candidates and 71 water-screen candidates. Geometric change intersections include 71 road physical segments, 4 bridge-tagged segments and 59 OSM building polygon features. Road association uses an explicit 10 m half-width corridor; these are **not** damage or closure counts. Source rasters, GeoJSONs and SHA bindings are saved under `outputs/radar-evidence-copdem-pixel-corrected` and `outputs/radar-exposure-copdem-pixel-corrected`.

The actual CPU Qwen assistant passed 11/12 AI-authored development cases, with all 12 displayed-clause checks matching source facts. One Nepali candidate-count question selected irrelevant facts; one unsafe raw model answer required the explicit policy guard. Both four-section report languages completed. These results are development smoke checks, not independent human QA or a Nepali language review. The one-page English PDF in `output/pdf/akash-rasuwa-situation-report.pdf` was rendered and visually checked.

The access proof now also has a separate nearest-baseline-destination adapter. Each settlement's destination stays fixed after a closure, using directed baseline road distance plus fixed node associations. This remains a hypothetical bridge-state analysis; facilities, entrances and actual road passability are unverified. The original any-destination proof and frozen three-region inspection benchmark are preserved. The benchmark did **not** consistently achieve its predeclared 20% benefit over the strongest baseline.

## Trained CPU water baseline — 7 October IST

The frozen Sen1Floods11 baseline trained on 263 chips from six geographic collections and 1,050,468 sampled pixels in 8.01 seconds on CPU. Its compressed checkpoint is 54,300 bytes. The complete training/evaluation process recorded a maximum resident size of 569 MB; dataset transfer was 732 MB and is separate from the fitting time. No GPU or paid API was used.

Final evaluation opened 111 chips from India, Pakistan and Bolivia only after the model and source hashes were frozen. Micro water IoU was **43.89%**, precision **79.99%**, recall **49.30%**, and F1 **61.00%**, with **99.70%** of label-valid pixels assessed. The conservative full-label water recall was 49.13%. Per-collection IoU was India 48.62%, Pakistan 28.44%, Bolivia 43.58%; this variation and low recall prevent a strong accuracy claim. Permanent and flood water are inseparable in the training labels. These results are geographic-collection water benchmarks, not Rasuwa or debris accuracy, and they are not directly comparable with the paper's random-chip split. See `outputs/sen1-cpu-v1` and `research/sen1-cpu-training-protocol.json`.

The unchanged checkpoint was applied once to raw, corrected Copernicus-DEM SNAP sigma0 powers using a separately frozen observation policy. The model layer assesses 1,863,073 of 2,827,950 Rasuwa cells (65.88%), leaving 964,877 unknown. It predicts 111,922 before-water cells and 109,605 after-water cells on the identical joint domain; 30,189 gain-water and 32,506 lost-water classification candidates are **not confirmed flooding**. Inference and initial rendering took 15.11 seconds. A later display-only revision improved map labels without repeating inference or changing any raster values.

`http://localhost:8783/` displays this layer separately from the radar-change screen and assistant. The local endpoint checks every recorded output SHA, raster grid, 0/1/255 domain, temporal states and all pixel counts. Training evaluation data and results are not loaded by the production endpoint. Neither classifier scores nor temporal candidates establish calibrated confidence, debris, damage or route safety.

Reproduce the declared experiment only in a fresh research run with its own evaluation receipt; the saved final test must not be silently rescored or retuned. The application command for the preserved model is:

```sh
uv run --with scikit-learn==1.7.2 --with matplotlib==3.10.8 python scripts/infer_trained_water.py --out outputs/sen1-water-new-run
uv run --with scikit-learn==1.7.2 python -m pytest -q
```

The latter command includes the optional training-loader checks; invoking the base-environment `pytest` executable can skip that module even when uv overlays scikit-learn. The original model-integration milestone passed 383 tests; the latest complete suite is recorded above.

## What works

- Live Copernicus Sentinel-1/2 catalogue discovery for a bounding box and event date.
- Same relative orbit, pass, instrument mode and polarization pairing. Both scene footprints must cover the requested area. No fallback to unrelated dates.
- Prepared, calibrated, terrain-corrected single-band dB raster ingestion with strict grid/CRS checks.
- Deterministic new-water candidate screening, nodata/terrain masks, GeoTIFF and GeoJSON output.
- OSM motor-road topology, one-way direction, parallel edges, alternate destinations and baseline-versus-scenario connectivity. Bridges and tunnels are not automatically closed by water intersections.
- Exact optimistic/pessimistic access bounds for explicitly supplied uncertain physical road segments, with route/cut witnesses and separate open/closed inspection outcomes. These certify the modeled graph only, not physical road passability.
- Building-footprint exposure, including multipolygon relations, courtyard holes, relation-member deduplication and explicit geometry exclusions. Counts describe mapped features, not confirmed damaged buildings.
- Native anonymous Sentinel-1 processing and a separate optional openEO processing route. The anonymous route removes the account requirement for the current case study.
- An actual local Qwen model that selects allowlisted source facts for English/Nepali answers. Displayed numbers come from computed output manifests. Unsupported damage, casualty and passability claims are declined. Nepali text has not received human language review.
- Input checksums, computed metrics, route evidence and an interactive HTML report.
- Explicitly labeled synthetic fixtures for engineering tests. Their numbers are not real-event results.

## Setup and run

Python 3.12 and [uv](https://docs.astral.sh/uv/) are used. Dependency versions are pinned in `uv.lock`.

```sh
uv sync --frozen
uv run pytest -q
uv run python scripts/make_fixture.py
uv run akash-flood run data/synthetic-fixture/run.json --out outputs/engineering-demo
```

Output directories must be empty; choose a new name for repeat runs. Open `outputs/engineering-demo/report.html` through a local server to inspect the report and evidence links:

```sh
uv run python -m http.server 8765 --bind 127.0.0.1 --directory outputs
```

Then visit `http://127.0.0.1:8765/engineering-demo/report.html`. The map currently uses CDN Leaflet resources, so the interactive view needs internet; JSON/GeoTIFF evidence remains local.

Search actual scenes without an account:

```sh
uv run akash-flood discover --bbox 85.30 28.12 85.43 28.32 --event-date 2026-08-26 --out data/rasuwa-pilot/catalog.json
```

The catalogue search retrieves metadata, not satellite pixels. Original Sentinel-1 amplitude measurements require calibration, terrain correction and co-registration before they can enter the detector. Source metadata declarations are mandatory but are not an independent verification of preprocessing.

Historical OSM extractor:

```sh
uv run akash-flood fetch-osm --bbox 85.25 28.04 85.48 28.38 --snapshot 2026-07-27 --event-date 2026-08-26 --out data/rasuwa-pilot/osm.json
```

It requests historical data and records the exact query and response checksum. The endpoint rejected `out meta`, so the working route uses `out body`; historical provenance relies on the server executing the saved date-filtered query, not independent per-element edit timestamps. It never silently retries with current OSM. The `timestamp_osm_base` in an Overpass response describes the server database, not the requested historical snapshot.

The extractor now requests building nodes, ways and relations. The first saved Rasuwa extraction predates that change and selected building ways only: it yielded **9,251 valid polygon features and one point-only building exclusion**. Missing relations and OSM coverage gaps remain possible. These are baseline inventory counts across the expanded road-network area, not flood exposures.

## Historical optional openEO connection

This route was prepared before the anonymous native path worked. It remains optional; account authentication is not a prerequisite for the current native case study.

The prepared pilot lives in `data/rasuwa-pilot/s1-jobs`. To reproduce preparation in a fresh directory:

```sh
uv sync --frozen --extra satellite
uv run akash-flood prepare-s1 data/rasuwa-pilot/catalog.json --out data/rasuwa-pilot/s1-jobs-new
uv run --extra satellite python scripts/process_s1.py data/rasuwa-pilot/s1-jobs-new --scene before
```

The last command only validates the graph. Both saved pilot graphs passed the live backend validator on 5 October 2026; this does not prove the requested pixels are available or the filters were applied. The recipe requests VV sigma0 in **linear power**, Copernicus DEM 30 m and a 10 m UTM grid. It does not request a GPU. [CDSE's radar example](https://documentation.dataspace.copernicus.eu/notebook-samples/openeo/Radar_ARD.html) documents the calibration parameters; [processing notes](https://documentation.dataspace.copernicus.eu/APIs/openEO/openeo_processing.html) explain backend limitations.

With a Copernicus Data Space account ready, run this from the project directory:

```sh
uv run --extra satellite python scripts/process_s1.py data/rasuwa-pilot/s1-jobs --scene before --action start
```

Open the device-login URL printed by the command and sign in in your browser. No password belongs in the script or chat. The command starts one remote processing job using the account's service quota. Refresh tokens are not persisted. Use `--action status`, then `--action download` when finished; each invocation uses a new device login. Repeat with `--scene after` for the second observation. Job IDs and graph checksums are recorded, duplicate starts are refused, and downloads preserve result metadata.

Before detector ingestion, verify actual source scene identities, co-registration and pixel validity; convert valid positive linear power with `10 * log10(power)`; and create a defensible terrain-quality mask. Geometric orthorectification is not radiometric terrain flattening. No conversion or metadata declaration is automatically treated as scientific validation.

## Public terrain preparation

```sh
uv run akash-flood fetch-terrain --bbox 85.30 28.12 85.43 28.32 --out data/rasuwa-pilot/terrain-new
uv run akash-flood align-slope data/rasuwa-pilot/terrain/slope-30m.tif path/to/processed-radar.tif --out path/to/aligned-slope.tif
```

The public Copernicus GLO-30 crop, projected elevation, Horn 3×3 slope and checksums are already saved in `data/rasuwa-pilot/terrain`. This is a digital **surface** model. Derivatives use a 30 m metric grid with complete neighborhoods only. Later sampling onto a 10 m radar grid does not add finer terrain information. A slope filter is not a physical radar shadow/layover mask.

The pilot terrain grid (including its 90 m halo) has a median slope of **35.1°**. Only **0.32%** of valid cells meet the current 5° cutoff. This makes the generic threshold especially restrictive here; a sensitivity assessment and independent validation are needed before choosing final thresholds. Excluded pixels remain unassessed, not classified as flood-free. [Dataset documentation](https://copernicus-dem-30m.s3.amazonaws.com/readme.html) describes source resolution and coverage.

## Prepared input contract

The synthetic `run.json` is also a schema example. Supply `before` and `after` source records with `path`, `scene_id`, `datetime` (timezone required), `relative_orbit`, `orbit_state`, `polarization`, `backscatter` (`sigma0` or `gamma0`), `units` (`dB`), `calibrated`, `terrain_corrected` and `processing_record`. Do not relabel a raw amplitude TIFF as dB.

All rasters must use the same projected metre grid and single-band layout. Optional `slope` uses degrees; optional `radar_invalid` has nonzero values for radar shadow/layover or otherwise invalid pixels. A slope threshold is not a substitute for a physical radar quality mask.

Optional `osm` points to an Overpass JSON containing nodes and ways, with `osm_snapshot` before the event. `settlements` and `destinations` are lists of objects with `id`, `name`, `lon`, and `lat`; `max_snap_m` limits nearest-node snapping. Snap distance is an association, not proof of an off-road connection. Polygon facilities are not automatically converted into accessible road entrances.

Outputs:

| File | Contents |
|---|---|
| `run_manifest.json` | Source metadata, checksums, configuration, limitations |
| `hazard_candidates.tif` | 0 no candidate, 1 new-water candidate, 2 pre-existing dark pixel, 255 unassessed |
| `hazards.geojson` | New-water candidate polygons |
| `exposed_roads.geojson` | Intersections and closure-scenario flags |
| `exposed_buildings.geojson` | Intersecting building footprints, overlap areas and fractions; no damage claim |
| `building_geometry_audit.json` | Invalid/incomplete building exclusions and suppressed relation members |
| `settlement_access.json` | Before/after reachable destinations and route IDs |
| `metrics.json` | Computed pixel/area/access totals; no model-generated numbers |
| `report.html`, `map.html` | Inspectable engineering output |

## Initial engineering milestone (superseded by the measured status above)

The initial synthetic raster-to-report milestone had 40 passing tests on the Apple M5 MacBook Air with 24 GB memory. These were software checks, not a speed or accuracy benchmark for a real flood. The expanded engineering demonstration is at `outputs/engineering-buildings-demo/report.html`; the historical-input audit is at `outputs/rasuwa-input-audit/index.html`. Subsequent native processing, source adapters, model and numerical-review work added further tests.

A live catalogue query for Rasuwa returned a compatible S1 pair from **16 and 28 August 2026**, relative orbit **85**, ascending. The anonymous public native path now processes those source assets locally. A real Sentinel-2 scene-classification crop was read through public COG range access. It contained extensive missing/cloud-covered pixels; no flood result has been inferred from it.

## Measured access-bounds proof

```sh
uv run pytest -q tests/test_access_bounds.py
uv run python scripts/prove_access_bounds.py
```

The proof declares all 28 bridge-tagged physical segments uncertain in the real historical Rasuwa motor-road graph (13,347 nodes / 26,688 directed edges), with no fixed closures and no satellite hazard input. The latest analysis took 0.888 s on local CPU. Nine settlements retain a modeled route even with all uncertain segments closed; 23 are ambiguous, three were already disconnected in the baseline, and 20 are unmapped under the 500 m node-snap rule.

One segment's inspection resolves three ambiguous settlement cases as connected if open, or 21 as disconnected if closed, conditional on these artificial scenarios and access to any mapped destination. The algorithm returns distinct branch counts, not an expected value. Monotone bounds cover all allowed assignments without enumerating them; the independent tests exhaust 320 small-network assignments and all one-link branches. Complete paths and cut certificates are saved in `outputs/access-bounds-proof/proof.json`. `inspection-proof.json` supplies both branch proofs for the segment with the highest closed-branch resolution count. All saved certificates were checked directly against source edges.

This is a feasibility result, not damage mapping or field validation. A separate nearest-baseline-destination proof is now saved in `outputs/nearest-access-proof`; the interactive experiment above still uses any mapped destination. Correlated hazards, facility entrances and operability, network boundaries and real flood evidence remain unresolved. Detailed scope and the decision gate are in `../research/access-bounds-proof.md`.

An optional read-only CDSE account check is also available:

```sh
uv run --extra satellite python scripts/check_cdse_access.py
```

Complete the printed Copernicus device-login URL locally. This check reads account quota metadata and creates no processing job. Tokens remain in the process; no credentials or account identity are written into project evidence. Authenticating alone does not establish pixel access, source identity, processing cost or flood quality.

## Work still required

1. Improve sensor coverage and hazard mapping beyond the weak candidate heuristic; numerical DEM sampling correction is now independently checked.
2. Validate historical OSM topology and facility coverage for an expanded road-network area; extraction itself is now working.
3. Validate a debris layer beyond the completed partial optical candidate screen; current coverage is inadequate for a complete map.
4. Validate building coverage and add facility entrance handling, uncertainty scenarios and boundary checks.
5. Expand independent English/Nepali model and report evaluation beyond the measured 12-case development check; adapter unit tests do not establish model comprehension or linguistic quality.
6. Improve hazard evidence using a declared training/development protocol and an unseen evaluation case. Current one-event comparisons are weak and insufficient; test another area/date from a fresh runtime.
7. Complete the six-page report, three-minute demo, public repository and Devpost entry.

## Provenance and AI assistance

The team's Akash archive was inspected for reusable satellite-processing patterns. No archived training script or checkpoint has been executed by this prototype. New code in this directory was authored with OpenAI Codex, including implementation, tests, documentation and research assistance. Human reviewers must verify scientific outputs and be able to explain the system. Do not claim the existing cloud-removal paper's PSNR as flood performance.

The original project code is MIT-licensed following the human owner's explicit selection on 11 October. This is an owner assertion, not an independent ownership investigation or a license grant over third-party assets. The Hugging Face archive's `other` label is not a blanket open-source grant for every file. Third-party library, satellite, elevation, OSM and model-weight terms remain applicable; see [source boundaries](release/THIRD-PARTY-BOUNDARIES.md).

## Data attribution for real runs

Contains modified Copernicus Sentinel data 2026.

Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.

© OpenStreetMap contributors. OSM data are under ODbL.

Training data actually used: **Sen1Floods11**, Bonafilia, Tellman, Anderson and Issenberg, *Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1*, CVPR Workshops 2020. The organizer permits educational use; the original repository has no standalone license file. Mirror license declarations are not asserted as verified upstream licensing.

When reference EMS products are evaluated: European Union, Copernicus Emergency Management Service data. The separate Kathmandu application also contains modified Copernicus Sentinel data 2024. Synthetic engineering fixtures do not contain satellite or OSM observations.
