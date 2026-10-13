"""Create a six-page technical evidence report from saved results only.

The report includes checking-only aggregate comparisons. It never loads the
reference GeoPackage, labels, a model checkpoint, or invokes inference.
"""
from datetime import datetime, timezone
from html import escape
import hashlib
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/pdf/akash-technical-evidence-report-2026-10-11.pdf'
PROVENANCE = OUT.with_suffix('.provenance.json')
W = A4[0]-84
INK = colors.HexColor('#17353e')
TEAL = colors.HexColor('#087e82')
STYLES = {name:ParagraphStyle(name,fontName='Helvetica-Bold' if name in ('title','heading','kicker') else 'Helvetica',
    fontSize=size,leading=leading,textColor=INK,spaceAfter=space)
    for name,size,leading,space in [('title',24,28,16),('heading',13,17,9),('body',9.5,13,8),('small',8,10.5,7),('kicker',8,11,9)]}


def record(path):
    path=Path(path).resolve()
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def saved_water(path):
    water=json.loads(path.read_text())
    if (water.get('schema')!='akash-fresh-water-v1' or water.get('available') is not True
            or water.get('reference_labels_loaded') is not False or water.get('gpu_used') is not False):
        raise ValueError('Require previously verified, completed reference-free water evidence')
    return water


def p(text,style='body'):
    return Paragraph(escape(text),STYLES[style])


def table(rows,widths):
    result = Table([[p(str(x),'small') for x in row] for row in rows],colWidths=widths)
    result.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5efeb')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('LINEBELOW',(0,0),(-1,-1),.25,colors.HexColor('#cbd9d3'))]))
    return result


def heading(number,title):
    return [p('AKASH / TRACK B / TECHNICAL EVIDENCE / 11 OCTOBER 2026','kicker'),p(title,'title')]


def main():
    if OUT.exists() or PROVENANCE.exists():
        raise FileExistsError('Preserve prior report artifacts')
    paths = {
        'rasuwa_water': ROOT/'outputs/fresh-one-command-replay/water/manifest.json',
        'rasuwa_exposure': ROOT/'outputs/fresh-one-command-replay/exposure/source-and-output-manifest.json',
        'rasuwa_exposure_summary': ROOT/'outputs/fresh-one-command-replay/exposure/summary.json',
        'rasuwa_execution': ROOT/'outputs/fresh-one-command-replay/execution-receipt.json',
        'kathmandu_water': ROOT/'outputs/kathmandu-unseen-water/water/manifest.json',
        'kathmandu_execution': ROOT/'outputs/kathmandu-unseen-water/execution-receipt.json',
        'kathmandu_protocol': ROOT/'research/kathmandu-inference-protocol.json',
        'kathmandu_access': ROOT/'outputs/kathmandu-node-drive-access-2026-10-08/summary.json',
        'kathmandu_access_audit': ROOT/'research/kathmandu-node-drive-access-audit-2026-10-08.json',
        'kathmandu_view': ROOT/'outputs/kathmandu-access-report-final-2026-10-08/source-and-output-manifest.json',
        'copilot_development': ROOT/'outputs/focused-copilot-development-v2/summary.json',
        'kathmandu_reference_source': ROOT/'research/evaluation-only/kathmandu/source-compatibility-result.json',
        'model_result': ROOT/'outputs/sen1-cpu-v1/final_test-evaluation.json',
        'model_summary': ROOT/'outputs/sen1-cpu-v1/summary.json',
        'model_training': ROOT/'outputs/sen1-cpu-v1/training-report.json',
        'model_freeze': ROOT/'outputs/sen1-cpu-v1/model-evaluation-freeze.json',
        'optical': ROOT/'outputs/s2-surface-change-resumed/manifest.json',
        'optical_checking': ROOT/'research/evaluation-only/optical-extent-checking-result.json',
        'optical_independent': ROOT/'research/optical-reference-checking-review/independent-area-audit.json',
        'composite': ROOT/'outputs/optical-completion-2026-10-11/manifest.json',
        'composite_audit': ROOT/'research/optical-completion-2026-10-11/independent-audit.json',
        'composite_checking': ROOT/'research/evaluation-only/optical-completion-2026-10-11/result.json',
        'completion_area_audit': ROOT/'research/evaluation-only/optical-completion-2026-10-11/independent-area-audit.json',
        'rasuwa_observation_access': ROOT/'outputs/observation-access-2026-10-11/rasuwa-v1/receipt.json',
        'kathmandu_observation_access': ROOT/'outputs/observation-access-2026-10-11/kathmandu-v1/receipt.json',
        'observation_access_protocol': ROOT/'research/observation-access-2026-10-11/protocol.frozen.json',
        'observation_access_audit': ROOT/'research/observation-access-2026-10-11/independent-verification.json',
        'multi_case': ROOT/'outputs/multi-case-copilot-2026-10-10/final-summary.json',
        'native_setup': ROOT/'research/native-setup-review-2026-10-10.json',
        'prior_post_scl': ROOT/'research/low-compute-hazard-s2-composite-coverage.json',
        'rasuwa_map': ROOT/'outputs/fresh-report-current/map.png',
        'kathmandu_map': ROOT/'outputs/kathmandu-water-report/map.png',
    }
    inputs = {key:record(path) for key,path in paths.items()}
    multi_case=json.loads(paths['multi_case'].read_text()); native_setup=json.loads(paths['native_setup'].read_text())
    rasuwa=saved_water(paths['rasuwa_water'])
    kathmandu=saved_water(paths['kathmandu_water'])
    exposure_binding=json.loads(paths['rasuwa_exposure'].read_text())
    exposure=json.loads(paths['rasuwa_exposure_summary'].read_text())
    summary_binding=exposure_binding['outputs']['summary.json']
    if (exposure_binding['status']!='completed' or exposure['status']!='completed'
            or summary_binding['sha256']!=inputs['rasuwa_exposure_summary']['sha256']
            or exposure['water_source']!=inputs['rasuwa_water']
            or exposure['temporal_counts']!=rasuwa['counts'] or exposure['case']!=rasuwa['case']):
        raise ValueError('Saved exposure aggregate must bind exactly to this verified water case')
    r_execution=json.loads(paths['rasuwa_execution'].read_text())
    k_execution=json.loads(paths['kathmandu_execution'].read_text())
    r_access=json.loads(paths['rasuwa_observation_access'].read_text())
    k_access=json.loads(paths['kathmandu_observation_access'].read_text())
    access_audit=json.loads(paths['observation_access_audit'].read_text())
    optical_audit=json.loads(paths['composite_audit'].read_text())
    area_audit=json.loads(paths['completion_area_audit'].read_text())
    if (access_audit['status'] != 'passed' or any(row['status'] != 'passed' for row in access_audit['results'])
            or not optical_audit['all_output_hashes_verified'] or not optical_audit['native_source_selection_exactly_recomputed']
            or not area_audit['same_original_coverage_gates_verified'] or not area_audit['all_original_gates_still_fail']
            or not area_audit['production_bound_paths_exclude_reference_and_checking_files']):
        raise ValueError('Require successful numerical/source-separation audits, retaining failed coverage gates')
    if (optical_audit['manifest']['sha256'] != inputs['composite']['sha256']
            or area_audit['result']['sha256'] != inputs['composite_checking']['sha256']):
        raise ValueError('Audit bindings do not match the exact displayed optical inputs')
    for case, key in [('rasuwa','rasuwa_observation_access'),('kathmandu','kathmandu_observation_access')]:
        audited=next(row for row in access_audit['results'] if row['case']==case)
        saved=next(item for item in audited['inputs'] if item['path'].endswith(f'{case}-v1/receipt.json'))
        if saved['sha256'] != inputs[key]['sha256']:
            raise ValueError('Access audit binding does not match displayed receipt')
    totals={key:sum(row[key] for row in access_audit['results']) for key in
        ['original_witnesses_checked','new_paths_checked','new_cuts_checked','independent_forward_dfs_extrema_checked']}
    if list(totals.values()) != [101,101,73,202]:
        raise ValueError('Report needs this explicit independent graph-check version')
    for receipt in (r_access,k_access):
        if (receipt['status'] != 'completed' or not receipt['all_frozen_inputs_unchanged']
                or any(receipt['claims'].values())):
            raise ValueError('Require completed, source-preserving observation-only graph checks')
    if (r_access['expanded_status_counts'] != {'ambiguous':32,'unmapped':20,'baseline_disconnected':3}
            or k_access['expanded_status_counts'] != {'ambiguous':41,'robust_connected':28}):
        raise ValueError('Report requires this explicit observation-aware scenario version')
    for execution in (r_execution,k_execution):
        if execution['status']!='completed' or execution['gpu_used'] is not False or execution['reference_labels_loaded'] is not False:
            raise ValueError('Require completed CPU-only source executions')
    o=json.loads(paths['optical'].read_text())
    if o['reference_labels_loaded'] is not False or o['local_debris_accuracy_established'] is not False:
        raise ValueError('Exploratory optical layer must retain its scope')
    r=rasuwa['counts'];k=kathmandu['counts'];s=exposure['access_scenario']
    if (r['assessed_pixels'],k['assessed_pixels'],s['n_uncertain_physical_segments'],o['counts']['candidate_pixels'])!=(1869789,2054876,313,1098):
        raise ValueError('Report text belongs to this explicit saved evidence version')
    latest_access=json.loads(paths['kathmandu_access'].read_text())
    latest_audit=json.loads(paths['kathmandu_access_audit'].read_text())
    copilot=json.loads(paths['copilot_development'].read_text())
    if (latest_access['access_scenario']['status_counts'] != {'robust_connected':69}
            or latest_access['buildings'] is not None
            or latest_access['roads']['mapped_feature_count'] != 261690
            or latest_access['access_scenario']['n_uncertain_physical_segments'] != 293
            or latest_audit['fixed_nearest_targets_checked'] != 69
            or latest_audit['paths_checked'] != 69 or latest_audit['settlements']['inside_water_aoi'] != 36
            or copilot['passes'] != 6 or copilot['cases'] != 6 or copilot['guard_corrections'] != 1):
        raise ValueError('Latest report text differs from its bound saved results')
    story=[]
    story += heading(1,'Evidence before action')
    story += [p('Akash is a CPU research prototype that links satellite water-state classifications to pre-event mapped infrastructure and checkable conditional road-access statements. Its current contribution is a reproducible evidence chain, not validated disaster damage detection.'),
        p('The map keeps unobserved pixels unknown. Candidate intersections never become observed road closures. Each conditional route or cut is checked against the directed source graph; the selected baseline destination stays fixed. The dashboard exposes source dates, counts and proof geometry.'),
        table([['Processing stage','Recorded behavior'],
            ['Public satellite inputs','Native Sentinel-1 VV/VH measurements and annotations; selected same platform/orbit/pass.'],
            ['Terrain and observations','Copernicus DEM height conversion and local SNAP terrain correction; unsupported geometry is excluded.'],
            ['Frozen water model','Small Sen1Floods11-trained classifier, unchanged threshold and features.'],
            ['Exposure and access','Historical OSM geometry intersections; declared open/closed uncertainty scenarios with source-edge certificates.'],
            ['Report','Deterministic saved-map numbers; optical candidates displayed separately.']], [126,W-126]),
        Spacer(1,13),p('Two actual application runs','heading'),
        table([['Case','Elapsed CPU run','What it establishes'],
            ['Rasuwa, August 2026',f'{r_execution["elapsed_seconds"]:.2f} s; cached sources','Eight stages through offline report. Zero new network requests; all eight water TIFFs match the earlier generic replay.'],
            ['Kathmandu, September 2024',f'{k_execution["elapsed_seconds"]:.2f} s; new imagery','Six stages through water, with 2.75 GB new radar/DEM bytes. A separate later historical access supplement is reported on page 4; its time is not included here.']], [126,111,W-237]),
        Spacer(1,12),p('The 11 October update makes unknown/outside road support explicitly uncertain and completes previously frozen optical selections. Radar inputs, trained water model, thresholds and prior classifications remain unchanged. Local hazard accuracy is unestablished. Public links, unrestricted judge-area support and human language review remain incomplete.','small'),PageBreak()]

    story += heading(2,'Inputs and numerical provenance')
    story += [p('Production sources are Sentinel-1, Sentinel-2, Copernicus DEM and pre-event OpenStreetMap. Sen1Floods11 supplies training labels. EMSR927 is confined to checking; its vectors, labels and post-event damage interpretation are excluded from training, threshold selection, production inference and the copilot.'),
        p('Sentinel-1 processing','heading'),
        p('The anonymous source path checks provider object sizes, SHA256 and derived-wrapper identities. Matching annotations and amplitude measurements are wrapped in a derived SAFE; it is not an original ESA-delivered SAFE. SNAP removes thermal noise, calibrates linear Sigma0 and performs geometric terrain correction. Embedded orbit vectors are used; precise-orbit updating and radiometric terrain flattening are absent.'),
        p('Classifier input requires finite positive linear powers, supported geometry and local feature support. Powers are converted once to dB. Negative or missing values are not replaced with epsilon. Before and after dates share an exact integer-overlap grid with no date-alignment resampling.'),
        p('Copernicus DEM','heading'),
        p('WorldDEM-30 orthometric heights are converted with the NGA EGM2008 geoid: ellipsoidal h = H + N. A local external-DEM sampling correction addresses a half-source-pixel offset; vendor jars are preserved and a separate overlay is hash-bound. JVM analytic checks and saved nominal-center samples verify numerical sampling, not absolute geolocation or ESA endorsement.'),
        p('Kathmandu review retained a failed whole-grid checker at a boundary rather than reporting a false pass. Eligible AOI height samples agree within 0.001 m. Exhaustive checks found valid interpolation support for every AOI center and the conservative 40 m feature halo.'),
        p('Historical map and source controls','heading'),
        p('The Rasuwa map cutoff is 27 July 2026, midnight UTC. Historical server-query provenance is retained, without per-object edit timestamps or independent server-authenticity proof. The older extract requested building ways only; relation completeness is unknown. Polygon/mapping exclusions remain audited.'),
        p('Bounded acquisition records source hashes, grid identity and failure outcomes. It rejects reference-source paths and mismatched timestamps. Supported live processing is limited to one UTM zone and compatible full-context scenes; mosaics, equator-crossing/polar areas and unrestricted global execution are not demonstrated.'),PageBreak()]

    story += heading(3,'Measured model performance')
    story += [p('The HistGradientBoosting water classifier was trained from scratch using 263 geographic-collection chips, with 72 validation chips and 111 final chips from India, Pakistan and Bolivia. Features and a 0.5 threshold were frozen before final scoring. The final set was evaluated once; no later retuning or repeated holdout score is claimed.'),
        table([['Final binary-water metric','Saved result'],['Micro IoU','43.89%'],['Precision','79.99%'],['Recall','49.30%'],['F1','61.00%'],['Supported labeled-pixel coverage','99.70%'],['Recall with unsupported water counted undetected','49.13%']], [W*.68,W*.32]),
        Spacer(1,11),p('The model misses approximately half of labeled water. Its final geographic IoUs are 48.62% India, 28.44% Pakistan and 43.58% Bolivia (macro 40.21%). Dataset labels merge permanent and flood water; collection names alone do not establish Himalayan mountain accuracy. Classifier scores are uncalibrated.'),
        p('The saved 54,300-byte checkpoint fits on CPU. Training used 1,050,468 sampled pixels; recorded fitting wall time was 8.013 seconds, excluding acquisition and preparation. Ten features use VV/VH dB, their difference, fixed local dB statistics and valid-neighbor fraction. These statistical dB means are not mean-power backscatter.'),
        p('Unchanged-model scene applications','heading'),
        table([['Native 10 m cells','Rasuwa','Kathmandu'],
            ['AOI',f'{r["aoi_pixels"]:,}',f'{k["aoi_pixels"]:,}'],
            ['Assessed',f'{r["assessed_pixels"]:,} (66.12%)',f'{k["assessed_pixels"]:,} (94.03%)'],
            ['Unknown',f'{r["unknown_pixels"]:,}',f'{k["unknown_pixels"]:,}'],
            ['New-water candidates',f'{r["new_water_candidate_pixels"]:,}',f'{k["new_water_candidate_pixels"]:,}'],
            ['Lost-water candidates',f'{r["lost_water_candidate_pixels"]:,}',f'{k["lost_water_candidate_pixels"]:,}']], [W*.48,W*.26,W*.26]),
        Spacer(1,10),p('These are classification counts, not measured event flood areas. Kathmandu inputs were frozen before new pixels: S1A 20 September / 2 October 2024, descending orbit 19. The after image is four days after the chosen 28 September peak. No local flood accuracy is established. The original water-only receipt omitted OSM; a separate later source-bound access supplement preserves that receipt unchanged.','small'),PageBreak()]

    story += heading(4,'Infrastructure and access bounds')
    story += [p('Native candidate-cell polygons are intersected with all retained mapped source features. Roads use a declared 10 m half-width, flat-cap corridor; this is a geometric association, not measured road width. A bridge-tagged physical segment is also a road. Buildings are OSM polygons, not verified damaged structures.'),
        table([['Mapped unit','Total','New overlap','Lost overlap','Some unknown'],
            *[[name.title(),str(exposure[name]['mapped_feature_count']),str(exposure[name]['features_with_new_water_candidate_overlap']),str(exposure[name]['features_with_lost_water_candidate_overlap']),str(exposure[name]['features_with_unknown_water_state'])] for name in ('roads','bridges','buildings')]], [103,85,100,100,W-388]),
        Spacer(1,10),p('No confirmed damage or closure is inferred. Candidate absence and unknown radar coverage do not establish passability. Feature counts include context outside the AOI; assessed and unknown categories can overlap within one feature.'),
        p('Observation-aware conditional access','heading'),
        p('The old experiment varied candidate-overlapping road segments only. It left unknown and outside-AOI roads assumed open. The new, separately frozen graph experiment adds any physical group with unsupported saved-centreline length to the same unrestricted open/closed scenario set. This is uncertainty about observation support, not observed damage or a failure probability.'),
        table([['All source settlements','Rasuwa: old / new','Kathmandu: old / new'],
            ['Connected in both scenario extremes','10 / 0','69 / 28'],
            ['Connected only in the open extreme','22 / 32','0 / 41'],
            ['Unmapped','20 / 20','0 / 0'],
            ['Baseline disconnected','3 / 3','0 / 0'],
            ['Uncertain physical road groups','313 / 8,434','293 / 122,715']], [W*.48,W*.26,W*.26]),
        Spacer(1,9),p('All 10 previously robust Rasuwa statements become ambiguous; 41 of 69 Kathmandu statements do. Within the water AOI, Rasuwa has 15 ambiguous, 6 unmapped and 2 baseline-disconnected settlements; Kathmandu has 28 scenario-connected and 8 ambiguous. Whole-source counts include incidental settlements outside the extraction context. None of these outcomes establishes road passability.'),
        p('Baseline destinations remain fixed using saved point-association rules and deterministic ties. Direction, physical grouping and geometry are preserved. Kathmandu has 27 nonempty fully assessed witnesses and one empty graph route sharing a source/destination node; the physical point/facility connection is unvalidated. No new ranking or travel-time estimate is made.','small'),
        p(f'Independent checks verified 101 old witnesses, 101 new paths, 73 cuts and 202 reachability extremes. The graph-only runs took {r_access["wall_seconds"]:.2f} s / {k_access["wall_seconds"]:.2f} s wall and {r_access["peak_process_rss_bytes"]/1e6:.1f} / {k_access["peak_process_rss_bytes"]/1e6:.1f} MB peak RSS, Rasuwa/Kathmandu. No network, GPU, model inference or reference geometry was used.','small'),
        p('The earlier Kathmandu acquisition/mapping supplement and Rasuwa paths/cuts remain archived under their original candidate-only assumptions. Buildings are unavailable for Kathmandu. Assessed nonwater also cannot establish physical passability; unmapped and unspecified failures remain excluded. The earlier inspection benchmark failed its 20% benefit target.','small'),PageBreak()]

    composite=json.loads(paths['composite'].read_text()); composite_check=json.loads(paths['composite_checking'].read_text())
    if composite['counts']!={'aoi_pixels':706991,'joint_assessed_pixels':269363,'unknown_pixels':437628,'candidate_pixels':3724} or composite_check['all_coverage_gates_pass']:
        raise ValueError('Composite version or failed-gate scope differs')
    if (composite['retention']['lost_assessed_cells'] != 0
            or composite['retention']['previous_classifications_changed'] != 0
            or composite['source_dates_and_pixel_selection_changed']
            or composite['surface_change_thresholds_retuned']
            or composite['reference_labels_loaded'] or composite['gpu_used']):
        raise ValueError('Completion must preserve selection, old classifications and threshold scope')
    story += heading(5,'More observation, honest boundaries')
    story += [p('The new 13-source Sentinel-2 composite selects the latest eligible pre-event and earliest eligible post-event land observation for each native 20 m cell. Source choice is frozen from SCL alone before new reflectance. Invalid/cloud/shadow/snow/water neighborhoods are excluded with a 60 m buffer. Exact all-valid 2x2 aggregation and each asset scale/offset preserve the native grid; no temporal interpolation or spectral-date fallback is used.'),
        table([['Optical observation audit','Original pair','10 Oct / 11 Oct'],
            ['Assessed cells / 706,991','79,555 (11.25%)','265,047 (37.49%) / 269,363 (38.10%)'],
            ['Unknown cells','627,436','441,944 / 437,628'],
            ['Provisional surface-change cells','1,098','3,572 / 3,724'],
            ['Candidate classification area','43.92 ha','142.88 / 148.96 ha']], [W*.44,W*.23,W*.33]),
        Spacer(1,9),p(f'Fixed-selection completion adds 4,316 assessed cells and retains all 265,047 earlier composite cells with zero prior classification changes. All 79,555 original pair cells remain assessed. The completion run took {composite["wall_seconds"]:.2f} s wall time, {composite["cpu_seconds"]:.2f} s CPU, {composite["new_body_bytes"]/1e6:.2f} MB new body bytes and {composite["peak_process_rss_bytes"]/1e6:.1f} MB peak RSS. Earlier composite acquisition costs are separate. This is completion of known partial acquisition, not blind validation.'),
        p('Different dates can create apparent change. Assessed pre-event ages are 2/4/14/19/22 days and post-event ages 1/6/11/16/18/21 days. The source dates and SCL pixel choices are unchanged. Fixed NDVI-loss/BSI-increase thresholds remain engineering hypotheses; farming, clearing, moisture or sensor errors can cause flags. The cell viewer exposes dates, age and support; this optical layer does not enter radar inference or road scenarios.','small'),
        p('Checking-only EMSR927 comparison','heading'),
        p('EMSR927 AOI01 is photo-interpreted landslide/mass-movement extent, not water or an exhaustive vegetation-change label. Prior checking outcomes were known. A separate frozen evaluation binds the reference file and exact loader; no damage-map inputs enter production or the copilot.'),
        table([['Whole checking denominator','Original pair','Completed composite'],
            ['Checking domain assessed (90% gate)','25.10% / FAIL',f'{composite_check["fractions"]["checking_domain_assessed"]*100:.2f}% / FAIL'],
            ['Mapped extent assessed (95% gate)','0.94% / FAIL',f'{composite_check["fractions"]["reference_extent_assessed"]*100:.2f}% / FAIL'],
            ['Whole mapped extent intersected','0.3007%',f'{composite_check["fractions"]["reference_extent_intersected_by_candidates"]*100:.2f}%']], [W*.52,W*.24,W*.24]),
        Spacer(1,9),p(f'Both unchanged gates still fail; {composite_check["areas_m2"]["unknown_reference_extent"]:,.2f} m2 of {composite_check["areas_m2"]["mapped_reference_extent"]:,.2f} m2 mapped extent remain unknown. An independent {area_audit["native_cell_count_inspected"]:,}-cell area audit agrees within {area_audit["maximum_absolute_area_difference_m2"]:.2e} m2. More observation does not establish debris precision, recall, IoU, completeness or unseen-event skill. Conditional overlap never replaces the whole-reference denominator.','small'),
        p('The earlier radar-change screen intersects 1.459% of mapped mass movement and retains its failed gates. Kathmandu checking remains blocked: both archives linked by UNOSAT 3993 lacked its stated Spot-6 layer. No regional substitute or fabricated accuracy is claimed.','small'),PageBreak()]

    story += heading(6,'Limits, reproduction and credits')
    story += [p('Release limitations','heading'),
        p('This educational prototype cannot warn minutes ahead of a glacier collapse. Satellite revisit times, cloudy optical pixels, radar terrain/urban effects, unvalidated geolocation and delayed after images limit detection. New-water transitions can reflect classifier error, season or registration rather than event inundation. Bare or small debris can be missed entirely.'),
        p('The raw runner defaults to an offline plan. Explicit execution needs prepared SNAP, geoid, patch/proof, checkpoint and scikit-learn 1.7.2; a clean extracted macOS arm64 prerequisite setup passed after relocation, using the official cached SNAP DMG and pinned auxiliaries. Separate vendor/corrected JVMs each checked 116 analytic samples; corrected max error 8.33e-10 m. Frozen-model inference assessed 4,064/4,096 synthetic cells with 32 unknown, with deterministic repeat and outbound/original-workspace access denied. This is prerequisite portability, not a clean cold replay of real satellite cases. A relocated 2.74 MB standard-library viewer serves saved maps/proofs/PDFs without workspace, network or model access. It does not reproduce raw processing or the copilot. Public repository and YouTube links remain pending.'),
        p('The archived 10 October CPU Qwen copilot switches between Rasuwa and Kathmandu, with source-bound English/Nepali clauses and selected-case guards. Its first 12-known-question run passed 8/12 status/inclusion and 12/12 source/case checks, with supported-question over-refusals. A corrective 6-question repeat passed 6/6 status/inclusion and source checks, but 4/6 strict relevance; a final focused repeat passed 2/2 known relevance failures. Twenty calls cover 12 distinct known questions; results are not pooled as a new test. These archived copilot checks do not validate the new observation-aware scenario. AI-authored questions and repeated prompt tests are not blind evaluation; human Nepali review is pending.'),
        p('Verification and evidence','heading'),
        p('The archived 10 October baseline suite passed 883 tests in 24.66 s with 2,241 existing/fixture warnings. Separate 11 October receipts verify fixed optical selection and unchanged earlier classifications, and both observation-aware graph scenarios. Original water/model/helper hashes remain unchanged. Earlier Kathmandu checks recomputed road intersections and nearest targets. These software and numerical checks do not measure hazard accuracy.'),
        p('Required data credits','heading'),
        p('Contains modified Copernicus Sentinel data 2026. The separate Kathmandu application also contains modified Copernicus Sentinel data 2024.','small'),
        p('Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.','small'),
        p('© OpenStreetMap contributors. Historical OSM data and any published modified OSM database remain subject to ODbL.','small'),
        p('Sen1Floods11: Bonafilia, Tellman, Anderson and Issenberg, 2020, CVPR Workshops. The organizer permits educational use; the original repository lacks a standalone license file. No mirror license is presented as verified upstream licensing.','small'),
        p('Checking: European Union, Copernicus Emergency Management Service data; UNOSAT product 3993 source audit. NGA EGM2008 supplies vertical conversion; ESA SNAP/local GPL-3.0-or-later overlay supplies processing. Qwen2.5 (Apache-2.0), llama.cpp (MIT). AI-assisted code/report preparation; AI is not a human teammate.','small'),
        p('Primary sources and reproducible source records','heading')]
    links=[
        ('Sen1Floods11 dataset and authors','https://github.com/cloudtostreet/Sen1Floods11'),
        ('Sentinel-2 products and radiometry','https://sentiwiki.copernicus.eu/web/s2-products'),
        ('EMSR927 checking activation','https://mapping.emergency.copernicus.eu/activations/EMSR927'),
        ('Nepal DHM: 27-29 September 2024 event report','https://dhm.gov.np/uploads/dhm/downloads/Situational_Report_on_Extreme_Precipitation_and_Flooding_Event_of_27-29_September_2024.pdf'),
    ]
    for label,url in links:
        story.append(Paragraph(f'<link href="{escape(url)}" color="#087e82">{escape(label)}</link>',STYLES['small']))
    story.append(p('Every displayed result is tied to the saved JSON inputs and SHA256 manifest accompanying this PDF. Protocols, source receipts, checking-only results and limitations are available in the repository evidence folders.','small'))

    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#cbd9d3'));canvas.line(42,38,A4[0]-42,38)
        canvas.setFont('Helvetica',7);canvas.setFillColor(INK)
        canvas.drawString(42,25,'Working technical evidence - educational prototype, not operational advice')
        canvas.drawRightString(A4[0]-42,25,f'{doc.page} / 6')
    OUT.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=35,bottomMargin=50,
        title='Akash technical evidence and limitations',author='Akash project - AI-assisted preparation').build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(OUT)
    if len(reader.pages)!=6:
        raise ValueError(f'Expected exactly six pages, got {len(reader.pages)}; repair layout before delivery')
    text=' '.join(' '.join(page.extract_text().split()) for page in reader.pages)
    for phrase in ('43.89%','49.30%','0.3007%','38.10%','122,715','8,434','437,628','Bonafilia','all rights reserved','public repository'):
        if phrase.lower() not in text.lower():
            raise ValueError('Missing substantive report evidence or credit: '+phrase)
    for key,path in paths.items():
        if record(path)!=inputs[key]:
            raise ValueError('Source changed during report authoring')
    write = {
        'schema':'akash-six-page-technical-report-v4','created_at':datetime.now(timezone.utc).isoformat(),
        'inputs':inputs,'code':record(__file__),'output':record(OUT),'pages':len(reader.pages),
        'model_invoked':False,'training_labels_loaded':False,'reference_geometries_loaded':False,
        'satellite_pixels_loaded':False,'saved_aggregates_only':True,
        'checking_only_aggregates_displayed':True,'full_track_b_requirements_satisfied':False,
        'visual_review':'pending',
    }
    PROVENANCE.write_text(json.dumps(write,indent=2)+'\n')
    print(json.dumps({'pdf':record(OUT),'pages':len(reader.pages),'visual_review':'pending'},indent=2))


if __name__=='__main__':
    main()
