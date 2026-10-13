# Final source review candidate — 10 October 2026

This candidate captures current UTF-8 source through the project's existing positive allowlist. The dated source manifest and ZIP timestamps use 10 October; the original 8 October candidate and all native-setup rehearsals remain preserved. Exact membership, sizes, SHA256 identities and observed verification/smoke results are recorded in `output/release/akash-source-candidate-2026-10-10.zip.verification.json` and `release/native-setup/source-final-review-2026-10-10.json`.

The allowlist includes Python under `src`, `scripts`, `tests`, Markdown/Python release guidance, web HTML/CSS/JavaScript, README, pyproject and the lock file. It excludes every imagery/reference/checking geometry, model checkpoint, copilot weight, DEM/geoid payload, SNAP binary/generated upstream Java implementation, old execution output, cache, virtual environment and archive. Five tests requiring deliberately excluded research/data inputs remain outside the source tier. Source code may reference omitted assets; including that code does not make its real-input workflow runnable from this ZIP alone.

Read `release/THIRD-PARTY-BOUNDARIES.md` before redistribution. This is a local source review artifact, with no public upload, account action or project/model license assignment. The original project/inherited asset rights remain unresolved. External components retain their own terms; source inclusion is not a grant for excluded data or model weights.

Installation and the prepared-raster/graph fixture smoke use base dependencies only:

```sh
uv sync --frozen
uv run --no-sync python release/smoke_source.py --out 'output/synthetic smoke'
```

The relocated verification uses a populated uv cache and blocks socket operations and original-workspace reads during smoke execution. Its generated 64×64 fixture and path/cut witnesses verify source installation and declared software behavior. It loads no learned model, native SNAP runtime or satellite/checking input, and reports no hazard accuracy. The previously completed macOS arm64 native/JVM/frozen-model rehearsal remains separate in the native setup review; this final source check does not repeat it.

Frozen-model inference separately requires scikit-learn 1.7.2 and exact externally supplied model/protocol assets. Native setup separately requires the supported macOS arm64 runtime, pinned auxiliary files and independently proved overlay, as described in `release/native-setup/README.md`. PDF generators in source require separately installed ReportLab/pypdf and the omitted bound scientific outputs. The new copilot execution receipts are preserved independently in the working checkout, outside this source-only tier.

Verify the fixed allowlist and per-member checksum identity without installing:

```sh
python scripts/package_source_candidate_20261008.py --verify /path/to/akash-source-candidate-2026-10-10.zip
```

The dated wrapper retained in this candidate produces the final local capture and adjacent receipt. Re-running it requires a new source/output location because it refuses existing candidate/review/log files. Any relocated smoke results apply to the exact archive bound by that receipt; the complete-checkout 883-test result is a separate existing receipt and is not represented as a test count from source extraction.
