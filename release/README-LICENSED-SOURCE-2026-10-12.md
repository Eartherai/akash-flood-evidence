# Local MIT source export — 12 October 2026

The human owner stated “Entirely my original code; use MIT”. This export adds the approved root `LICENSE` to the original positive source allowlist. MIT applies to original project code. Dependencies, model weights, satellite/elevation/map data and other third-party assets retain separate terms; see `THIRD-PARTY-BOUNDARIES.md`. The owner assertion is not an independent ownership investigation.

The ZIP root remains `akash-source-candidate`. Its manifest uses the distinct `akash-source-mit-review-v1` schema and binds every payload, including the source policies and approved license. Two exact new module paths, `web/observation-geography-core-20261012.mjs` and `web/observation-geography-ui-20261012.mjs`, are added explicitly; this does not admit arbitrary `.mjs` files. Older archives and their original verification policy remain unchanged. Verify this new export with its own policy:

```sh
python scripts/package_licensed_source_20261012.py --verify /path/to/akash-source-mit-2026-10-12.zip
```

On a compatible Python 3.11–3.13 system with uv and an available dependency cache, enter the extracted root, then run:

```sh
uv sync --frozen --offline
uv run --no-sync python release/smoke_source.py --out 'output/synthetic smoke'
```

The existing smoke generates synthetic rasters and a small directed road graph. It checks prepared-raster report generation, scenario access change and path/cut witnesses while blocking network/model operations. It does not reproduce raw satellite acquisition, native SNAP processing, trained model inference, arbitrary area/date selection, hazard accuracy, physical road safety or human usability.

Only bounded UTF-8 source is included. Raw/reference/checking data and geometry, imagery, DEM/geoid grids, model weights/checkpoints, SNAP/vendor binaries/generated source, virtual environments, archives and execution outputs are excluded. Viewer source may be present without its generated evidence data. Full-checkout README links can point outside this limited source tier. Five data-dependent tests remain excluded with their reasons in the manifest; do not report a full-checkout test count for this tier.

The dated geographic viewer's source files are `scripts/build_observation_geography_20261012.py` and `web/observation-geography-20261012.html` plus the two explicitly admitted modules. Its builder requires separately supplied, preserved observation-access JSON, protocol/checker receipts, exact R2 compact data and source settlement/destination GeoJSON. It verifies those identities and writes a complete output folder containing saved data and `core.mjs`/`ui.mjs` aliases. Those generated data/aliases are not present in this source-only ZIP. Serve the complete separately packaged viewer output over localhost; opening the source HTML alone does not reproduce it. Derived historical OSM data retains ODbL and other applicable upstream terms. The viewer displays saved context, with no new raw-data replay, model inference, hazard classifications or graph solve.

This is a local review export. Public GitHub, public/unlisted YouTube, real human Devpost details and a submission receipt remain separate delivery requirements.
