# Akash hardening evidence - 11 October 2026

This update exposes how observation gaps change conditional access statements and completes previously frozen optical acquisition. The trained water model, radar classifications, optical source selection, engineering thresholds and earlier outputs remain unchanged. It strengthens the evidence workflow; it does not establish local hazard accuracy or a guaranteed competition outcome.

## Observation-aware access

The preserved earlier experiment varied road groups intersecting new-water candidates and assumed other baseline edges available, including unknown and outside-AOI portions. A separate frozen experiment now makes each physical road group with unsupported saved-centreline length uncertain open/closed as well. Every combination remains permitted without a failure probability. The open and closed extremes provide exact reachability bounds for that declared graph scenario; unsupported coverage is not observed damage.

| All source settlements | Rasuwa candidate-only / observation-aware | Kathmandu candidate-only / observation-aware |
|---|---:|---:|
| Connected in both scenario extremes | 10 / 0 | 69 / 28 |
| Connected only in the open extreme | 22 / 32 | 0 / 41 |
| Unmapped under the saved association rule | 20 / 20 | 0 / 0 |
| Baseline-disconnected | 3 / 3 | 0 / 0 |
| Uncertain physical road groups | 313 / 8,434 | 293 / 122,715 |

All ten formerly robust Rasuwa statements become ambiguous when unsupported road portions are included. In Kathmandu, 41 of 69 become ambiguous. Within the water AOI, Rasuwa has 15 ambiguous, six unmapped and two baseline-disconnected settlements; Kathmandu has 28 scenario-connected and eight ambiguous. Whole-source denominators include incidental settlements outside the extraction context.

The 28 Kathmandu scenario-connected outcomes comprise 27 nonempty, fully assessed centreline witnesses and one empty graph route whose source and destination share a node. The physical point/facility connection is unvalidated. Neither those graph outcomes nor assessed nonwater establishes passability, facility operation or safe travel. Baseline destinations, source direction, physical grouping and saved association rules remain fixed. No inspection ranking, travel-time estimate or new routing objective was computed.

An independent forward-DFS and cut/source-support checker verified **101 original witnesses, 101 new paths, 73 cuts and 202 reachability extremes** across both cases. The original source-to-OSM and nearest-target provenance boundary is reused; this is graph/coverage verification, not an independent field investigation. An initial strict arithmetic failure was preserved, and the review records the subsequently justified numerical tolerance.

| CPU stage | Wall time | Peak process RSS | Scope |
|---|---:|---:|---|
| Rasuwa graph experiment | 1.40 s | 120.2 MB | Saved graph and support only |
| Kathmandu graph experiment | 13.52 s | 609.5 MB | Saved graph and support only |
| Independent combined graph check | 11.95 s | 720.4 MB | Forward reachability, cuts and support arithmetic |

These stages used no network, GPU, model inference, training or reference geometry. Their costs are separate from earlier radar acquisition, terrain correction, inference and historical mapping.

Exact records:

- [Frozen observation-access protocol](../research/observation-access-2026-10-11/protocol.frozen.json)
- [Rasuwa receipt](../outputs/observation-access-2026-10-11/rasuwa-v1/receipt.json) and [Kathmandu receipt](../outputs/observation-access-2026-10-11/kathmandu-v1/receipt.json)
- [Independent verification](../research/observation-access-2026-10-11/independent-verification.json) and [review](../research/observation-access-2026-10-11/review.md)

## Completed optical inputs

Completion acquired the remaining reflectance for the same frozen 13-source Sentinel-2 SCL selection. Source dates, per-cell date indices and thresholds were not changed in response to checking outcomes. Invalid/cloudy/out-of-selection cells remain unknown; no spectral-date fallback or temporal interpolation is used.

| Native 20 m cells, AOI denominator 706,991 | Original pair | 10 October composite | 11 October completion |
|---|---:|---:|---:|
| Jointly assessed | 79,555 (11.25%) | 265,047 (37.49%) | 269,363 (38.10%) |
| Unknown | 627,436 | 441,944 | 437,628 |
| Provisional exposed-surface/vegetation-loss candidates | 1,098 | 3,572 | 3,724 |
| Candidate classification area | 43.92 ha | 142.88 ha | 148.96 ha |

The new completion adds 4,316 assessed cells, retains all 265,047 earlier composite cells, changes zero earlier assessed classifications and retains all 79,555 original pair cells. These are support and processing measurements, not accuracy gains. The known partial acquisition and earlier checking outcomes were already available; this is not blind validation.

Assessed pre-event ages are 2/4/14/19/22 days and post-event ages 1/6/11/16/18/21 days. Different cells do not describe one simultaneous event-time image. Fixed NDVI-loss/BSI-increase thresholds can respond to agriculture, clearing, moisture, season or sensor effects. The optical candidates remain separate from radar inference and the road scenario.

The separate completion run recorded **105.24 s wall time, 5.93 s CPU time, 129.28 MB new delivered body bytes and 258.9 MB peak RSS**, with no GPU. The earlier composite's 121.36 MB body bytes and 100.38 s wall time remain separate receipts; the recorded combined acquisition campaign totals 250,642,015 body bytes. Crop/range hashes and ETags do not establish verification of complete provider objects.

The independent audit recomputed native selection, DN-to-reflectance conversion, candidates, date/age histograms and retention, and verified every declared output hash. It used no network, reference labels, threshold retuning or GPU.

- [Completion manifest](../outputs/optical-completion-2026-10-11/manifest.json)
- [Independent optical audit](../research/optical-completion-2026-10-11/independent-audit.json) and [review](../research/optical-completion-2026-10-11/review.md)

## Checking-only comparison: both gates still fail

EMSR927 is a photo-interpreted mass-movement inventory, whereas these candidates indicate exposed-surface/vegetation-loss. It is not an exhaustive label for all such changes. Prior checking outcomes were known. Reference geometries and damage interpretation remain excluded from production inference, training, tuning and copilot inputs.

| Whole checking denominator | Completion result | Unchanged gate |
|---|---:|---|
| Checking-domain assessment | 63.34% | 90%: fail |
| Mapped-reference assessment | 30.85% | 95%: fail |
| Whole mapped reference intersected by candidates | 18.70% | Geometric intersection only |

Of 1,110,777.28 m2 mapped reference extent, 768,107.92 m2 remain unknown. Candidate intersection is 207,753.74 m2. The independent 27,432-cell area calculation agrees with polygon-union sums within 3.82e-8 m2. Full-reference and unknown denominators are retained; covered-reference overlap is not substituted for the whole-reference result.

These numbers establish neither detector precision, recall, IoU, completeness, flood/debris accuracy nor unseen-event skill. Checking and candidate classes, dates and resolutions differ; absolute co-registration remains unvalidated. The Kathmandu checking comparison remains unavailable because the acquired archives did not contain the stated Spot-6 Kathmandu layer; no substitute accuracy result is claimed.

- [Checking result](../research/evaluation-only/optical-completion-2026-10-11/result.json)
- [Independent area and static source-separation audit](../research/evaluation-only/optical-completion-2026-10-11/independent-area-audit.json)

## Reports, preserved evidence and remaining work

The new [six-page technical report](../output/pdf/akash-technical-evidence-report-2026-10-11.pdf) and [one-page Rasuwa brief](../output/pdf/akash-situation-evidence-brief-2026-10-11.pdf) use saved aggregates only. Adjacent provenance and QA records bind their exact inputs and record individual visual inspection of all seven rendered pages. The 10 October PDFs, earlier optical results, water executions and candidate-only graph statements remain at their original paths and retain their original dates and assumptions.

This note does not claim new copilot, video or human language validation. Earlier copilot questions were known, AI-authored development checks, and their archived results retain that scope. Human Nepali review and final human signoff remain pending. The frozen water benchmark still misses approximately half of labeled water, and local hazard skill is unestablished. Raw replay supports bounded compatible inputs and prepared prerequisites; unrestricted judge-selected geography, a clean cold replay of real scenes and complete portability are not demonstrated.

Public GitHub, YouTube and Devpost links, real human eligibility/account details and the human submitter's explanation of the work remain outstanding. No project-wide license grant is assigned by this note; source datasets, OSM, vendor code and model dependencies retain their respective licensing boundaries. A complete report or passing software proof does not certify operational disaster readiness or competition placement. The same person/team cannot submit both local alternatives under the current one-track rule.

## Optional observation copilot and final software checks

The separately activated observation-aware wrapper preserves the original service and all 70 earlier source bindings. Two source-bound stores expose the new graph counts, exact source-point scope, named witness support and null empty-route fractions. Local Qwen selects source-clause IDs; quantities and wording are never generated by the language model. Activation took 11.318 s with peak Python-process RSS 1.421 GB, excluding the separate CPU model process. All 142 frozen development records remained unchanged.

Six fixed EN/NE development questions were frozen before one actual CPU-model pass: 5/6 displayed-response expectations and 4/6 raw-model expectations passed. A supported English Rasuwa counts/scope request wrongly declined; one raw Nepali cross-case request needed correction by the explicit case guard. All raw responses and failures are retained, with no retries or prompt/question retuning. This is not blind relevance or human-reviewed Nepali. [Full review](../research/observation-copilot-2026-10-11/review.md) and [completion receipt](../research/observation-copilot-2026-10-11/completion.json).

A separate actual browser interaction repeated the known Kathmandu empty-route question in Nepali, opened the visible citations and downloaded the answer. The null fractions and source pointer matched the bound graph output. [Browser proof](../research/viewer-hardening-2026-10-11/copilot-browser-verification.json). This UI check does not change the six-question results.

The full source checkout passed 927 software tests with 2,242 warnings in 33.64 s (35.60 s including command startup/overlay setup). The optional overlay pinned pyshp 2.3.1 and scikit-learn 1.7.2. [Exact command and log binding](../research/release-hardening-2026-10-11/full-suite.json). Software tests do not establish hazard accuracy.

## Reproduction tiers

The UTF-8 source ZIP uses the original positive allowlist and contains no raw imagery, checking geometry, model weights, vendor runtime or scientific execution outputs. Synthetic source smoke can run after extracting/installing base dependencies. Actual source-bound copilot and scientific reruns need their separately verified source closure and runtime prerequisites; saved browser viewers reproduce the saved audit only. The full review kit carries dated aggregate checking evidence, raw development-answer records and graph witnesses separately. Original source licensing remains unset; packaging does not assign rights or publish a submission.
