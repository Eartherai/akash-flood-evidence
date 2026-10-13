# Bounded live multimodal workflow - 12 October 2026

This additive implementation connects operator-selected area/date inputs to actual source discovery, native radar processing, a fixed optical pair, observation-aware road scenarios and source-bound reports. It does not change the original runner, scientific model, inference threshold, original graph module or old evidence packages. Execution success and hazard accuracy are separate.

The local console is `http://127.0.0.1:8825/` while its server runs. Start from the project root:

```sh
uv run --offline --with scikit-learn==1.7.2 python scripts/serve_fresh_console_20261012.py --port 8825
```

Here `--offline` concerns dependency resolution only. An explicit **Run this plan** action contacts the providers declared in the plan. Trusted startup `--source-cache PATH` can reuse checksum-bound native source files; reuse is disclosed and is not a new remote download. Browser callers cannot enter file paths, endpoints or commands. Plans run once in fresh server-generated directories; failures remain and cannot be overwritten or automatically retried.

For a fresh installed or relocated source extraction, bind the native setup tier's proved activation and the separately supplied organizer text using trusted startup `--activation` and `--requirements`. [Exact setup and metadata-check scope](CONSOLE-NATIVE-ACTIVATION-20261012.md). The original-checkout defaults remain available for historical local review; they are not assumed to exist in a new installation.

The fixed multimodal entry point is `scripts/run_multimodal_fresh_20261012.py`. It accepts base area/date/OSM/time/resource flags plus `--with-optical`, `--with-observation-access`, `--optical-budget-bytes` and `--optical-timeout-seconds`. The console selects those options when its multimodal workflow is chosen.

The current optical v3 adapter accepts a complete context catalogue while screening the exact AOI. It freezes source IDs, dates, assets, radiometry, code and native grid before acquisition. Optical source windows are at most 24 days per side, with a default 60 MB body budget, a configurable 100 MB maximum and a 600 s maximum deadline, plus explicit unknown masks. The recorded completed case declared a 90 MB cap before its acquisition. Partial/cloudy coverage remains visible; no coverage-based reselection or threshold tuning occurs. Its surface-change rules are hypotheses, not validated debris.

The generic observation adapter binds the current exposure directory rather than the two older demonstration cases. It preserves fixed destinations, makes candidate/unknown/outside support uncertain, verifies all-open/all-closed extrema with independent forward traversal, and retains null fractions for empty witnesses. These are graph scenarios, not observed closures or safe passage.

The combined report retains separate native radar/optical grids. It rejects mismatched AOI, context, event date or normalized historical cutoff. Its title reflects which observation layers are included; case facts, source/output hashes and situation note are downloadable. The console serves a report only after the actual stage and output hashes pass.

The worker, stages and process inspections use direct POSIX spawn with an independent session and explicit descriptor controls. This fixes a Mac PROJ/SQLite child-side-fork crash found during the first actual launch. Warmed-library/thread, cancellation, resistant descendants, logs and source identity are tested. See the Python [POSIX spawn documentation](https://docs.python.org/3.12/library/os.html#os.posix_spawn).

Historical OSM uses one explicitly selected provider per new plan. The preserved FOSSGIS request returned HTTP 504, and a separate Private.coffee attempt hit the original 20-second first-read timeout. The new helper aligns its initial read with the declared 90-second query while the worker enforces a hard 120-second stage ceiling and 30 MB body cap. It never substitutes current OSM, suppresses a provider failure or automatically retries. Provider selection is recorded before each new attempt. The OSM community [public instance directory](https://wiki.openstreetmap.org/wiki/Overpass_API#Public_Overpass_API_instances) lists historical-data support.

The operator test is a small new AOI `[85.29,27.75,85.32,27.78]`, event `2024-09-28`, cutoff `2024-09-01`, inside the already known Kathmandu event family. It is not blind, judge-selected, unrestricted-world or new accuracy validation. A separate integration attempt uses a smaller AOI in the known Rasuwa event with explicit verified pre-event OSM and raw-source cache reuse after live-map failures. It tests the connected processing tier; it does not establish live historical retrieval. Current attempts and actual status remain in `outputs/fresh-console-2026-10-12`. Source snapshots and failed attempts are preserved alongside this note.

Targeted fixture checks cover scientific contracts and real process/HTTP behavior. They are not satellite validation. All 70 prior frozen science bindings were checked unchanged before execution. `full_track_b_requirements_satisfied` and local hazard accuracy remain false. The earlier subjective rubric scores have not been increased for code written or partial execution.


## Actual integration result

The disclosed cached-source Rasuwa run `case-9668b96689efb7b0ead66f5c` completed all 14 stages in 171.07 seconds. Its new optical acquisition received 60,069,247 bytes with a 90 MB cap after the first 60 MB attempt stopped before the final SWIR crop. The same area, dates, source pair and thresholds were retained. Optical jointly assessed 68,801/620,008 cells (11.10%); 551,207 remain unknown and 1,060 are provisional surface-change candidates. This weak coverage is not presented as a quality improvement or a passing optical gate.

Radar assessed 1,666,783/2,480,036 cells; 813,253 remain unknown, with 25,237 new-water candidates. Native radar acquisition reused 28 verified source assets, read 2,679,575,201 measurement-source bytes and received zero new remote radar bytes. Historic OSM was reused with exact pre-event provenance. New catalogue queries and S2 range requests are real. Expanded graph results remain 32 undetermined, 20 unmapped and 3 baseline disconnected in the whole source inventory; inside this new AOI the corresponding counts are 14/6/2. No confirmed closures are supplied.

The completed run's R6 code stayed unchanged through execution and was independently rehashed by the root. Exact source snapshots are retained. A subsequent R7 presentation-service change raised the hash-verified output serving cap to 64 MB so the 25.4 MB route-facts download is accessible; it did not alter the report, masks, routes, model or completed stage receipt. Source export R7 and raw-run R6 identities have distinct scopes. Reports and server source do not imply parity with a published Git commit.

Actual report checks at 375, 768 and 1440 pixels show no horizontal overflow and both native image layers load. All six report outputs return exact checked bytes. These are AI/software checks on one Mac, not human field or Nepali-language validation. Some implementation agents reached their usage limit; the root completed the actual case, browser and final artifact checks. There is no independent human review or new hazard-accuracy evaluation.
