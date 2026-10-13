# Akash: Flood Evidence and Access Proofs

Akash turns radar and optical observations into inspectable flood evidence, then asks how missing road observations change a settlement's modeled access to a fixed destination. Unknown pixels stay unknown. A graph connection never becomes a claim that a road is physically safe.

**[Open the public saved-evidence demo](https://eartherai.github.io/akash-flood-evidence/)** · **[Submission story](docs/submission/about-project.md)**

The public website runs the saved case explorer and a separate historical two-case geographic viewer entirely in the browser. Python processing, SNAP and the local Qwen copilot run locally after setup; GitHub Pages does not execute those services. The saved-fact lookup on the public website is deterministic and is not a live language model.

## What to inspect first

1. Switch between the completed case's radar and optical maps. They retain separate 10 m and 20 m native grids. Compare the assessed and unknown support, rather than comparing raw pixel counts across resolutions.
2. Inspect settlement statuses and conditional path/cut evidence. Unknown within-area and outside-area roads are uncertain; destinations remain fixed. This case establishes no robust new settlement isolation.
3. Open the historical Rasuwa/Kathmandu viewer separately. It contains dated route scenarios, not the newer 14-stage run's results.
4. Download the saved facts, run manifest and source. The actual completed case and current source have explicitly separate version identities.

## Measured results and limits

| Evidence | Recorded result | Meaning |
|---|---|---|
| Completed bounded case | 14 stages, 171.07 seconds on one Mac | Cached radar, elevation and pre-event OSM; real new Sentinel-2 acquisition. This is not arbitrary-area cold success. |
| Radar support | 1,666,783 assessed / 2,480,036 cells; 813,253 unknown | 25,237 new-water candidate cells. Binary water labels do not distinguish flood water from permanent water. |
| Optical support | 68,801 assessed / 620,008 cells, 11.10% | 551,207 unknown and 1,060 provisional surface-change candidates. These are not validated debris or damage labels. |
| All 55 source settlement points | 32 undetermined, 20 without a mapped association, 3 baseline disconnected | Expanded uncertainty scenario; no robust new isolation. Historical source points are not a census or population estimate. |
| Frozen water-model benchmark | 43.89% micro water IoU; 79.99% precision; 49.30% recall on 111 held-out chips | Separate geographic-collection benchmark, not Rasuwa, debris or road-passability accuracy. |

The known-family completed case is evidence of bounded engineering execution. Cold historical OSM provider failures, limited optical coverage, unvalidated local damage mapping, unblinded assistant checks and lack of a judge-selected cold run remain material limitations. The project does not establish casualties, damaged buildings, observed closures, safe routes or hospital operation.

## Run the browser demonstration

Python 3.11–3.13 is sufficient to serve the static site:

```sh
python3 -m http.server 8834 --bind 127.0.0.1 --directory docs/site
```

Open http://127.0.0.1:8834/. No GPU, paid AI API, model download or account is required to explore these saved results. The browser uses the included assets; it performs no satellite processing.

## Install and run the analysis source

Use Python 3.12 and uv 0.13.0. Dependencies are pinned in `uv.lock`.

```sh
uv sync --frozen --python 3.12
uv run --no-sync python -m pytest -q
uv run --no-sync python release/smoke_source.py --out output/synthetic-smoke
```

The smoke run generates a clearly labeled synthetic fixture and recomputes its conditional graph proof. Its guards prohibit external networking and model imports. It is a software portability check, not a satellite replay or accuracy experiment. Choose a fresh output directory for every run.

For the native area/date console, read [the workflow guide](release/MULTIMODAL-WORKFLOW-20261012.md), [trusted prerequisite activation](release/CONSOLE-NATIVE-ACTIVATION-20261012.md) and [macOS arm64 runtime setup](release/native-setup/README.md). These give the exact installed asset contract, model/protocol inputs, DEM/geoid proof and activation arguments required by the actual planner. This repository excludes raw observations, native binaries and trained weights; these must be supplied separately under their applicable terms. Colab/Kaggle can run Python experiments, but this native runtime setup has not been demonstrated there.

For optional frozen-model loader tests:

```sh
uv run --with scikit-learn==1.7.2 python -m pytest -q
```

CI tests software and saved-demo consistency. It does not fetch satellite imagery, run a local language model, validate hazard mapping or assert Linux SNAP execution.

## Source, licenses and assistance

Original project code is [MIT licensed](LICENSE) following the human owner's explicit ownership assertion. [Upstream asset boundaries](release/THIRD-PARTY-BOUNDARIES.md) remain separate. The website includes attributed, derived OSM geographic evidence and Copernicus-derived figures; see its [data and attribution notice](docs/site/NOTICES.md). No vendor binary, checkpoint or raw satellite archive is redistributed.

OpenAI Codex assisted research, implementation, debugging, verification, UI, documentation and submission preparation. AI agents are tools, not human team members or repository collaborators. Git commits use the owner's existing Eartherai identity and contain no Codex coauthor trailer.

The [original imported README](docs/publication/imported-readme-2026-10-13.md) preserves dated experiments and earlier limitations. Some historical scratch/output links are not included in this source-only import. `release/SOURCE-MANIFEST.json` binds the original 217 source payloads before publication; [source-origin.json](docs/publication/source-origin.json) explains the current publication additions. Earlier scientific outputs and frozen model evaluations were not modified for this release.

The hackathon permits one submitting track per team and one submitting team per person. Publishing both projects does not create two eligible submissions. No official judging score, winning guarantee or completed Devpost receipt is claimed.
