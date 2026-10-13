## Inspiration

The Track B challenge asks three urgent questions: where did the flood hit, what infrastructure may be affected, and who may have lost road access? A colored satellite map alone cannot answer all three. Cloud, mountain terrain, missing map features and uncertain road observations can turn an apparently precise answer into a misleading one.

Akash was built around a more demanding question: **what does the available evidence actually support, and which conclusions still depend on things we cannot observe?** Its focus is traceable satellite change and conditional access analysis, designed to run with modest local computing resources.

## What Akash does

Akash connects before/after Sentinel-1 radar processing, a fixed Sentinel-2 optical pair, Copernicus elevation data and pre-event OpenStreetMap. It produces water-change candidates, provisional optical surface-change candidates, infrastructure intersections and a source-bound situation note.

The distinctive feature is the road-access analysis. Akash fixes each mapped settlement's nearest baseline town or hospital target, then examines the road graph under declared uncertain-edge scenarios. Candidate, unobserved and outside-observation road support remains uncertain. The analysis checks both all-open and all-closed extremes and retains witness paths or cut certificates that can be checked independently.

This separates a conclusion supported throughout the declared scenario from one that changes when uncertain roads change. An unmapped point and a settlement already disconnected in the baseline receive separate statuses. Neither becomes a newly isolated village by default.

The public website presents saved evidence and downloads. The separate local execution console accepts bounded area/date inputs, displays an explicit processing plan and starts acquisition and processing only after an operator runs that plan. GitHub Pages does not run the satellite backend or local language model.

## How it was built

The processing code uses Python, ESA SNAP and a small frozen scikit-learn water classifier trained with Sen1Floods11. Rasterio, NumPy and geospatial libraries preserve observation masks and native grids; NetworkX supports the access analysis. Radar and optical evidence retain separate resolutions and denominators rather than implying that resampling creates new observations.

The console binds source identities, historical dates, resource limits and output hashes. Failed attempts remain recorded. Original project code is MIT-licensed; datasets, derived map data, weights and vendor runtimes retain their upstream terms and are supplied separately where required.

A separate optional CPU Qwen/llama.cpp copilot selects verified source facts for English/Nepali answers. Deterministic source clauses supply displayed measurements, and guards check case and source scope. Its development evaluations and failures are retained; it has not been established as a validated interpreter for the latest completed case.

## Results we can show

One connected Rasuwa case completed **14 stages in 171.07 seconds** for the supplied August 2026 event. This was a known-case integration run with verified cached radar, elevation and pre-event OSM inputs. New Sentinel-2 requests acquired **60,069,247 bytes**. The timing is therefore a bounded integration measurement, not a cold-download or arbitrary-area promise.

| Evidence in the completed case | Saved result |
|---|---:|
| Radar jointly assessed cells | 1,666,783 / 2,480,036 |
| Radar unknown cells | 813,253 |
| New-water candidate cells | 25,237 |
| Optical jointly assessed cells | 68,801 / 620,008 (11.10%) |
| Optical unknown cells | 551,207 |
| Provisional optical-change candidates | 1,060 |

Within the new radar AOI, access statuses were **14 undetermined, 6 unmapped and 2 baseline disconnected**. No confirmed closures were supplied. These counts describe graph evidence, not physical road passability.

The separate frozen Sen1Floods11 collection benchmark achieved **43.89% water IoU, 79.99% precision and 49.30% recall**. It measures binary water classification on that benchmark; it does not establish local Himalayan flood, debris or damage accuracy. Earlier EMSR927 comparisons were checking-only, involved different classes and failed internal coverage gates. Published damage maps do not feed production predictions or the copilot.

## Challenges and what we learned

Historical OSM providers returned timeouts and HTTP 504 errors. A declared optical byte limit stopped one attempt; a subsequent, separately declared limit allowed acquisition without changing the source pair or thresholds. A Mac child-process crash required a process-launch correction. These failures reinforced the need for explicit prerequisites, bounded execution and retained receipts.

The largest lesson was scientific: successful execution does not imply successful mapping. Most optical cells in the completed case remained unknown. An independently checked graph proof certifies its assumptions; it cannot turn a satellite candidate into an observed closure.

## What comes next

The next priorities are a separately frozen cold area/date run, compatible independent local checking labels, stronger debris validation and human English/Nepali usability review. Arbitrary judge-selected raw-data reliability, unseen Himalayan accuracy and operational rescue use remain unproved. Akash is an educational evidence prototype, not an early-warning or safe-routing system.

## Credits and AI assistance

Contains modified Copernicus Sentinel data 2026. Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. © OpenStreetMap contributors; ODbL applies to OSM data. Training: Bonafilia, Tellman, Anderson and Issenberg, *Sen1Floods11*, CVPR Workshops 2020. Checking-only credit: European Union, Copernicus Emergency Management Service data.

OpenAI Codex assisted with implementation, review and submission preparation. Qwen is the separately run optional project copilot. AI tools are disclosed in Built With and the README; they are not human teammates. Demo-media preparation used HyperFrames and GSAP.
