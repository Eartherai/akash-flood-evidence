# Local source review candidate — 8 October 2026

Current license update, 11 October 2026: the human owner confirmed ownership of the original code and selected MIT. New licensed exports include the root `LICENSE`; older source archives remain unchanged historical snapshots. Third-party assets retain their separate terms. This remains a local source review artifact; nothing here uploads or publishes a repository. Read `THIRD-PARTY-BOUNDARIES.md` before redistribution.

The positive allowlist contains UTF-8 Python under `src`, `scripts` and `tests`, HTML/CSS/JavaScript under `web`, `pyproject.toml`, `uv.lock`, the project README and these release notes. `release/SOURCE-MANIFEST.json` records every included source file's size and SHA-256. No imagery, native DEM/geoid grid, reference/checking geometry, checkpoint, model weight, SNAP binary or generated upstream Java source, virtual environment or old execution output is included. Source scripts referencing excluded assets are retained for inspection; those real-input workflows cannot run from this ZIP alone. Existing README links describe the full working checkout and may target artifacts absent from this source-only tier.

On a compatible system with Python 3.11–3.13 and uv available, extract the ZIP and enter its `akash-source-candidate` folder. Installation obtains third-party dependencies from the configured package index, or a populated uv cache; it does not download satellite/model assets.

```sh
uv sync --frozen
uv run --no-sync python release/smoke_source.py --out 'output/synthetic smoke'
uv run --no-sync python -m pytest -q
```

The smoke explicitly generates 64×64 synthetic rasters and a four-node road network. It verifies the installed source comes from this extraction, produces the prepared-raster report, checks a baseline-to-scenario access change, and independently checks a path and cut witness for uncertain-link bounds. It blocks outbound socket operations while executing, optionally rejects reads of a supplied original workspace path, and records outputs. These checks demonstrate source/dependency relocation and synthetic software behavior. They do not reproduce the trained water model, SNAP preprocessing, real Kathmandu/Rasuwa input acquisition, hazard accuracy or physical road safety.

Five tests dependent on deliberately excluded data/research files are excluded from this tier: `test_checking_native_water_extent.py`, `test_kathmandu_checking_transport.py`, `test_research_server.py`, `test_research_radar_server.py` and `test_trained_water_server.py`. Their exact dependency reasons are in the source manifest. Optional training-loader tests skip when sklearn is absent. Any test count from this extraction must be reported as its own observed count, never the full-checkout suite count.

Frozen-model training/inference uses the separate pinned overlay **`scikit-learn==1.7.2`**. It is not a base dependency in `pyproject.toml`/`uv.lock`, and its checkpoint, processing runtime, geoid and scientific protocols are not included here. Rendering some research figures separately requires Matplotlib; PDF authoring separately requires ReportLab and pypdf. The synthetic smoke needs none of these overlays.

Verify this candidate without installing it:

```sh
python scripts/package_source_candidate_20261008.py --verify /path/to/akash-source-candidate-2026-10-08.zip
```

This verifies the fixed allowlist and per-member checksums, not author identity, source history, upstream rights or scientific validity.
