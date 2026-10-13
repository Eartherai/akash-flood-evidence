# Native raw-workflow prerequisites — macOS arm64

This optional local setup tier prepares the exact SNAP 14.0.0 runtime, geodetic auxiliary files, local external-DEM correction and unchanged CPU checkpoint contract. The source ZIP still contains no vendor binary, upstream Java implementation, geoid, imagery, checking labels or model weights. Setup does not grant project/model redistribution rights, accept an installer agreement, publish anything or establish flood/debris accuracy. Linux, Windows and macOS Intel have not been supported or tested by this adapter.

Read `release/THIRD-PARTY-BOUNDARIES.md` before redistribution. The upstream correction source is GPL-3.0-or-later; setup preserves complete corresponding Java source, COPYING and NOTICE with the overlay. Vendor third-party notices and JRE legal files are retained. [Pinned SNAP source](https://github.com/senbox-org/snap-engine/tree/059c4901869eacd2136590a1c21530abbd31d2f3) and [NGA/PROJ geoid notices](https://cdn.proj.org/us_nga_README.txt) apply to those components. The original code/model license remains the human owner's decision.

Extract a freshly generated source candidate, enter its folder, and use Python 3.11–3.13 plus uv. SNAP's native executable is macOS arm64. Keep at least 4 GB free disk for setup; a full new raw case needs additional space, RAM and its own transfer budgets.

```sh
uv sync --frozen
uv run --with scikit-learn==1.7.2 --with jpype1==1.6.0 python scripts/setup_native_runtime.py plan
```

The read-only plan starts no Java, model, subprocess or network request. Supply these separately held assets: the exact locally trained `water-model.joblib` (54,300 bytes, SHA256 `8b8b5795aa890f01fa611d4bd46ed0b97b9e8b5974ebd6dfed5e27e0f8d54d2d`), original `model-evaluation-freeze.json`, and original `sen1-cpu-training-protocol.json`. Their exact bytes and frozen feature/training-code/lock hashes are required. No model is fetched or fitted. This minimal tier preserves historical training-report/ledger bindings inside the unchanged original freeze; those files and training labels are absent from the runtime tier.

Prefer existing verified caches. Replace the example absolute cache paths below with paths to separately held assets. Installation refuses every existing runtime/model destination, including failed attempts. Use a new source extraction to retry.

```sh
uv run --with scikit-learn==1.7.2 --with jpype1==1.6.0 python scripts/setup_native_runtime.py install \
  --dmg /path/to/esa-snap_sentinel_macos_arm-14.0.0.dmg \
  --geoid /path/to/us_nga_egm08_25.tif \
  --egm96 /path/to/ww15mgh_b.zip \
  --dem-source /path/to/FileElevationModel-14.0.0.java \
  --checkpoint /path/to/water-model.joblib \
  --freeze /path/to/model-evaluation-freeze.json \
  --protocol /path/to/sen1-cpu-training-protocol.json --prove
```

For uncached vendor/auxiliary/source files, omit their cache flags and explicitly add `--allow-download`. The script permits only four exact HTTPS URLs and pinned byte/hash identities. It makes one attempt per missing file, allows no redirect to a different URL, caps the official DMG at 1,125,044,359 bytes and the total successful body budget at 1,210,000,000 bytes, and keeps failed partial files and attempt receipts. The expected four-file total is 1,208,242,576 bytes. Python dependency installation is separate from this asset budget. The live official [ESA checksum list](https://step.esa.int/downloads/14.0/installers/sha256sums) matched the pinned DMG during the 10 October review.

The installer mounts the verified DMG read-only and extracts only the pinned core, optical, microwave and bundled JRE payloads. It starts no installer, requests no account and submits no legal assent. It retains the EGM96 object required by SNAP initialization; the application still uses externally converted EGM2008 heights with `externalDEMApplyEGM=false`. EGM96 is a runtime dependency and does not replace the declared EGM2008 conversion.

Installation generates `data/native-setup/activation-1/activation.json`. A new inference-only freeze and overlay manifest contain current source-root paths and bind preserved bytes. The original freeze and original generated overlay manifest remain byte-identical; new files explicitly record their derivation. The independent JVM proof must complete before the doctor passes.

```sh
uv run --with scikit-learn==1.7.2 --with jpype1==1.6.0 python scripts/setup_native_runtime.py doctor \
  --activation data/native-setup/activation-1/activation.json
uv run --with scikit-learn==1.7.2 --with jpype1==1.6.0 python scripts/setup_native_runtime.py smoke \
  --activation data/native-setup/activation-1/activation.json --out output/native-synthetic-smoke
```

The doctor rehashes all installed regular files and checks Python overlays, bundled Java, model/feature-source contracts and the independent proof. It starts no model, JVM or network request. The synthetic smoke starts separate vendor/corrected JVMs with 116 analytic DEM samples each and actually loads the frozen checkpoint for deterministic 64×64 synthetic inference. It checks preserved unknown observations. These operations prove setup and numerical/model software execution, not an end-to-end raw satellite replay, geolocation accuracy, hazard accuracy or a complete submission. The smoke blocks Python socket connections; the stronger dated rehearsal additionally applies macOS sandbox restrictions to all descendant JVM/Python processes.

After moving an installed extraction, create a new activation and rerun its proof; old absolute receipts are preserved as historical evidence:

```sh
uv run --with scikit-learn==1.7.2 --with jpype1==1.6.0 python scripts/setup_native_runtime.py activate \
  --out data/native-setup/activation-after-move --prove
```

A future raw run must pass the current activation's `--model-freeze`, `--dem-reader-patch` and `--dem-reader-proof` paths explicitly to `scripts/run_fresh_case.py`, plus `--requirements /path/to/Track-B.txt` for the separately supplied organizer text. The existing runtime remains at the processor's supported `data/snap-runtime/portable` layout. Raw acquisition still requires eligible public metadata/pixels, valid AOI/date/paired scenes, DEM coverage, resource caps and optional checked historic OSM. The setup rehearsal acquires none of those case inputs and does not claim their future success.

The dated local source ZIP and rehearsal record are new artifacts. The preserved 8 October ZIP and its review remain unchanged. Rehearsal builds a new source candidate using the existing positive allowlist, installs from separately verified cached files, moves the extraction, regenerates its proof, and runs the smoke with outbound networking and reads of the original project denied. The copied model/protocol are supplied only to that temporary local test; they are never added to the source ZIP. Reproduce with:

```sh
uv run --with scikit-learn==1.7.2 --with jpype1==1.6.0 python release/native-setup/rehearse.py \
  --out release/native-setup/rehearsal-new
```
