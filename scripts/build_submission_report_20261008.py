"""Create a six-page technical evidence report from saved results only.

The report includes checking-only aggregate comparisons. It never loads the
reference GeoPackage, labels, a model checkpoint, or invokes inference.
"""
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from pypdf import PdfReader

from build_fresh_report import record, read_water, read_exposure

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/pdf/akash-technical-evidence-report-2026-10-08.pdf'
PROVENANCE = OUT.with_suffix('.provenance.json')
W = A4[0]-84
INK = colors.HexColor('#17353e')
TEAL = colors.HexColor('#087e82')
STYLES = {name:ParagraphStyle(name,fontName='Helvetica-Bold' if name in ('title','heading','kicker') else 'Helvetica',
    fontSize=size,leading=leading,textColor=INK,spaceAfter=space)
    for name,size,leading,space in [('title',24,28,16),('heading',13,17,9),('body',9.5,13,8),('small',8,10.5,7),('kicker',8,11,9)]}


def p(text,style='body'):
    return Paragraph(escape(text),STYLES[style])


def table(rows,widths):
    result = Table([[p(str(x),'small') for x in row] for row in rows],colWidths=widths)
    result.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5efeb')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('LINEBELOW',(0,0),(-1,-1),.25,colors.HexColor('#cbd9d3'))]))
    return result


def heading(number,title):
    return [p('AKASH / TRACK B / TECHNICAL EVIDENCE / 8 OCTOBER 2026','kicker'),p(title,'title')]


def main():
    if OUT.exists() or PROVENANCE.exists():
        raise FileExistsError('Preserve prior report artifacts')
    paths = {
        'rasuwa_water': ROOT/'outputs/fresh-one-command-replay/water/manifest.json',
        'rasuwa_exposure': ROOT/'outputs/fresh-one-command-replay/exposure/source-and-output-manifest.json',
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
        'prior_post_scl': ROOT/'research/low-compute-hazard-s2-composite-coverage.json',
        'rasuwa_map': ROOT/'outputs/fresh-report-current/map.png',
        'kathmandu_map': ROOT/'outputs/kathmandu-water-report/map.png',
    }
    inputs = {key:record(path) for key,path in paths.items()}
    rasuwa,_,_ = read_water(paths['rasuwa_water'])
    kathmandu,_,_ = read_water(paths['kathmandu_water'])
    _,exposure,proof,_ = read_exposure(paths['rasuwa_exposure'],paths['rasuwa_water'],rasuwa)
    r_execution=json.loads(paths['rasuwa_execution'].read_text())
    k_execution=json.loads(paths['kathmandu_execution'].read_text())
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
        Spacer(1,12),p('Local hazard accuracy is unestablished and the debris map is insufficient. The case-aware bilingual copilot and small offline viewer work; the matching video has separate capture/render receipts. Public links, unrestricted judge-area support and fresh-machine raw processing remain incomplete.','small'),PageBreak()]

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
        p('Conditional graph experiment','heading'),
        p('All 313 new-water-overlapping physical road segments (626 directed edges) are jointly uncertain open/closed. All other baseline edges stay available in every scenario, including unknown and outside-AOI coverage. Every combination is allowed, without probabilities. Unspecified failures and debris are excluded.'),
        table([['Settlement status under that scenario set','Count'],['Connected even when every selected segment is closed','10'],['Connected with all selected segments open, disconnected with them closed','22'],['Not mapped within the 500 m point-association rule','20'],['Disconnected in the baseline graph','3']], [W-70,70]),
        Spacer(1,11),p('There are 55 node-tagged settlements and 3 mapped town/hospital destinations. The nearest baseline-reachable destination is selected by directed road length plus fixed point-to-node association distances, with deterministic ties. The destination never switches after closures. This is not a travel-time or operational-hospital estimate.'),
        p(f'{s["certificate_checks"]["paths"]} saved path witnesses and {s["certificate_checks"]["cuts"]} cut certificates passed independent source-edge checks. The dashboard draws this evidence. Connected means connected under the declared graph assumptions; it does not mean safe to drive.'),
        p('Later Kathmandu supplement: Sep 1 pre-event driving roads and node POIs; 261,690 physical segments, 293 uncertain; all 69 source settlements retain a path with those segments closed (36 inside the AOI). All 69 fixed targets and paths were independently checked. Buildings are unavailable; other baseline edges remain assumed open. CPU mapping took 487.77 s after acquisition; neither stage is included in the original 361.09 s water receipt. The earlier inspection benchmark failed its 20% benefit target.','small'),PageBreak()]

    story += heading(5,'Checking exposed important failures')
    story += [p('EMSR927 AOI01 supplies a photo-interpreted landslide / mass-movement extent, not an open-water inundation label. Comparing it with water predictions would not measure flood accuracy. Reference outcomes were already known before corrective processing and the optical experiment, so these are descriptive checking comparisons, not blind tests.'),
        p('Radar-change screen','heading'),
        p('The corrected physical radar-change screen intersects only 1.459% of rasterized mapped mass-movement extent. Domain and event coverage gates fail. This screen is separate from the trained water classifier and cannot be used as its IoU, recall or landslide accuracy.'),
        p('Exploratory optical surface-change screen','heading'),
        p('24 / 27 August 2026 native Sentinel-2 bands use declared reflectance scale/offset. Exact all-valid 2x2 aggregation brings 10 m bands to 20 m. Joint land SCL support excludes cloud, shadow, snow, water and invalid neighborhoods with a 60 m buffer. Fixed NDVI-loss/BSI-change thresholds are untuned hypotheses; no cloud reconstruction is used.'),
        table([['Optical scope / checking denominator','Result'],
            ['Full optical AOI assessed','79,555 / 706,991 cells (11.25%)'],
            ['Full optical AOI unknown','627,436 cells (88.75%)'],
            ['Surface-change candidates over full AOI','1,098 cells; 43.92 ha of classifications'],
            ['Reference checking-domain area assessed','25.098% (90% gate fails)'],
            ['Mapped mass-movement extent assessed','0.9351% (95% gate fails)'],
            ['Whole mapped extent intersected','0.3007%']], [W*.59,W*.41]),
        Spacer(1,11),p('Only nine candidate cells lie inside the checking domain. The independent exact-polygon area audit agrees within 5.10 x 10^-8 m2. A high conditional overlap on nine cells would not be precision or accuracy. The different class meanings and incomplete observation support reject this layer as a sufficient debris map.'),
        p('An earlier SCL-only union through day 21 reaches only 59.36% raw clear support before buffering/paired checks; it does not overturn the failed gate. Kathmandu checking was also blocked: both GIS archives linked by UNOSAT product 3993 lacked its stated September 30 Spot-6 Kathmandu layer. Regional Sentinel-1 layers were not substituted; no overlap/accuracy result was produced.','small'),PageBreak()]

    story += heading(6,'Limits, reproduction and credits')
    story += [p('Release limitations','heading'),
        p('This educational prototype cannot warn minutes ahead of a glacier collapse. Satellite revisit times, cloudy optical pixels, radar terrain/urban effects, unvalidated geolocation and delayed after images limit detection. New-water transitions can reflect classifier error, season or registration rather than event inundation. Bare or small debris can be missed entirely.'),
        p('The raw runner defaults to an offline plan. Explicit execution needs prepared SNAP, geoid, patch/proof, checkpoint and scikit-learn 1.7.2; clean raw setup is unproved. A relocated 2.74 MB standard-library viewer serves saved maps/proofs/PDFs without workspace, network or model access. It does not reproduce raw processing or the copilot. Public repository and YouTube links remain pending.'),
        p('The real CPU Qwen copilot selects source fact IDs; verified artifacts supply numbers and clauses. The first 16 known development questions had omissions, irrelevant facts and one timeout. After a disclosed retrieval/prompt correction, six repeated cases passed source/relevance/status checks in 4.92-10.08 s; one safety request required a policy correction. This is nonblind AI-authored development, not independent language QA. Human Nepali review is pending.'),
        p('Verification and evidence','heading'),
        p('The 8 October complete suite passed 814 tests in 23.94 s with 2,241 existing/fixture warnings. Kathmandu independent checks recomputed every road corridor intersection and nearest target; the renderer rechecks source identities and witnesses. Original water/model/helper hashes remain unchanged. These software and numerical checks do not measure hazard accuracy.'),
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
    for phrase in ('43.89%','49.30%','0.3007%','0.9351%','313','Bonafilia','all rights reserved','public repository'):
        if phrase.lower() not in text.lower():
            raise ValueError('Missing substantive report evidence or credit: '+phrase)
    for key,path in paths.items():
        if record(path)!=inputs[key]:
            raise ValueError('Source changed during report authoring')
    write = {
        'schema':'akash-six-page-technical-report-v2','created_at':datetime.now(timezone.utc).isoformat(),
        'inputs':inputs,'code':record(__file__),'output':record(OUT),'pages':len(reader.pages),
        'model_invoked':False,'training_labels_loaded':False,'reference_geometries_loaded':False,
        'checking_only_aggregates_displayed':True,'full_track_b_requirements_satisfied':False,
        'visual_review':'pending',
    }
    PROVENANCE.write_text(json.dumps(write,indent=2)+'\n')
    print(json.dumps({'pdf':record(OUT),'pages':len(reader.pages),'visual_review':'pending'},indent=2))


if __name__=='__main__':
    main()
