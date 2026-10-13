"""One-page English brief from the saved observation-aware 11 October evidence.

Reads completed saved aggregates only. Does not load models, patient rows,
satellite pixels or checking reference geometry. Earlier PDFs remain unchanged.
"""
from datetime import datetime, timezone
from html import escape
import hashlib
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/pdf/akash-situation-evidence-brief-2026-10-11.pdf'


def record(path):
    path=Path(path)
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}


def main():
    if OUT.exists() or OUT.with_suffix('.provenance.json').exists():
        raise FileExistsError('Preserve prior evidence brief')
    paths={
        'water':ROOT/'outputs/fresh-one-command-replay/water/manifest.json',
        'optical':ROOT/'outputs/optical-completion-2026-10-11/manifest.json',
        'optical_audit':ROOT/'research/optical-completion-2026-10-11/independent-audit.json',
        'exposure':ROOT/'outputs/fresh-one-command-replay/exposure/summary.json',
        'access':ROOT/'outputs/observation-access-2026-10-11/rasuwa-v1/receipt.json',
        'access_audit':ROOT/'research/observation-access-2026-10-11/independent-verification.json',
        'checking':ROOT/'research/evaluation-only/optical-completion-2026-10-11/result.json',
        'checking_area_audit':ROOT/'research/evaluation-only/optical-completion-2026-10-11/independent-area-audit.json',
    }
    sources={name:record(path) for name,path in paths.items()}
    water=json.loads(paths['water'].read_text())['counts']
    optical=json.loads(paths['optical'].read_text())
    access=json.loads(paths['access'].read_text())
    checking=json.loads(paths['checking'].read_text())
    access_audit=json.loads(paths['access_audit'].read_text())
    area_audit=json.loads(paths['checking_area_audit'].read_text())
    optical_audit=json.loads(paths['optical_audit'].read_text())
    c=optical['counts']
    if (water['new_water_candidate_pixels']!=30193 or c['joint_assessed_pixels']!=269363
            or c['candidate_pixels']!=3724 or checking['all_coverage_gates_pass']
            or access['expanded_status_counts']!={'ambiguous':32,'unmapped':20,'baseline_disconnected':3}
            or optical['surface_change_thresholds_retuned'] or optical['source_dates_and_pixel_selection_changed']
            or optical['reference_labels_loaded'] or any(access['claims'].values())):
        raise ValueError('Unexpected evidence version or changed claim boundaries')
    if (access_audit['status']!='passed' or not optical_audit['all_output_hashes_verified']
            or not area_audit['all_original_gates_still_fail']):
        raise ValueError('Require successful source-bound checking audits')
    audited=next(row for row in access_audit['results'] if row['case']=='rasuwa')
    receipt_binding=next(item for item in audited['inputs'] if item['path'].endswith('rasuwa-v1/receipt.json'))
    if (receipt_binding['sha256']!=sources['access']['sha256']
            or optical_audit['manifest']['sha256']!=sources['optical']['sha256']
            or area_audit['result']['sha256']!=sources['checking']['sha256']):
        raise ValueError('Checking audit bindings differ from displayed results')
    width=A4[0]-84
    ink=colors.HexColor('#183d36');teal=colors.HexColor('#087e82')
    body=ParagraphStyle('body',fontName='Helvetica',fontSize=9,leading=12,textColor=ink,spaceAfter=6)
    small=ParagraphStyle('small',parent=body,fontSize=7.5,leading=10)
    head=ParagraphStyle('head',parent=body,fontName='Helvetica-Bold',fontSize=19,leading=23,spaceAfter=8)
    def p(text,style=body):return Paragraph(escape(text),style)
    def table(rows,widths):
        t=Table([[p(str(value),small) for value in row] for row in rows],colWidths=widths)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e1ece4')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f5f0')]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        return t
    story=[p('AKASH / RASUWA / SAVED EVIDENCE BRIEF / 11 OCTOBER 2026',small),
        p('Unknown changes the access claim.',head),
        p('Event: 26 August 2026. Educational research evidence; not an operational damage assessment or safe-route instruction. Radar water states, provisional optical surface changes and conditional road scenarios remain distinct.'),
        table([['Saved layer / denominator','Recorded result'],
            ['Radar native 10 m water states',f'{water["new_water_candidate_pixels"]:,} new-water and {water["lost_water_candidate_pixels"]:,} lost-water candidates'],
            ['Radar assessment support',f'{water["assessed_pixels"]:,}/{water["aoi_pixels"]:,} cells (66.12%); {water["unknown_pixels"]:,} unknown'],
            ['Completed optical native 20 m support',f'{c["joint_assessed_pixels"]:,}/{c["aoi_pixels"]:,} cells (38.10%); {c["unknown_pixels"]:,} unknown'],
            ['Provisional optical change',f'{c["candidate_pixels"]:,} cells; {optical["candidate_area_m2"]/10000:.2f} ha of classifications, not confirmed debris']],
            [width*.46,width*.54]),Spacer(1,8),
        p('Fixed optical selection. The source dates, SCL choices and surface-change thresholds are unchanged. Completion adds 4,316 assessed cells, retains all 265,047 earlier composite cells and changes zero earlier classifications. Pre-event ages are 2/4/14/19/22 days; post-event ages are 1/6/11/16/18/21 days. Different cells are not observed at the same time.'),
        table([['Association with new-water candidates','Observation-aware graph result'],
            ['313 physical road segments, 0 bridge-tagged segments and 14 mapped building polygons. These are geometric associations, not measured damage.',
             '55 source settlements: 0 connected in both scenario extremes; 32 ambiguous; 20 unmapped; 3 baseline-disconnected. All 10 previously robust statements become ambiguous.']],
            [width*.48,width*.52]),Spacer(1,8),
        p('Unknown road support is now explicit. Candidate overlap and unsupported saved-centreline portions, including outside the AOI, make 8,434 physical groups uncertain open/closed. Open and closed extremes give exact bounds under that unrestricted scenario set. Destinations remain fixed at the original nearest baseline-reachable mapped town/hospital. No observed closure, failure probability, safe driving or facility-operation claim follows.'),
        p('Within the water AOI: 15 ambiguous, 6 unmapped and 2 baseline-disconnected settlements. Whole-source counts include incidental settlements outside the extraction context. Assessed nonwater still cannot prove passability. Unmapped and unspecified failures remain excluded.'),
        p(f'Checking-only comparison. Completed optical candidates intersect {checking["fractions"]["reference_extent_intersected_by_candidates"]*100:.2f}% of the whole mapped EMSR927 mass-movement extent. Domain/reference assessment is {checking["fractions"]["checking_domain_assessed"]*100:.2f}%/{checking["fractions"]["reference_extent_assessed"]*100:.2f}%; unchanged 90%/95% gates fail. Classes differ and previous checking outcomes were known. No local flood/debris accuracy, completeness or blind validation is established.'),
        p('Limits. Cloud, shadow, terrain, image age, unvalidated co-registration, land-use changes and classifier error can hide or mimic change. Unknown is not undamaged. Optical candidates do not enter radar inference or road scenarios. This system cannot warn minutes before a glacier collapse.'),
        p(f'Separate CPU costs: optical completion {optical["wall_seconds"]:.2f} s wall, {optical["cpu_seconds"]:.2f} s CPU, {optical["new_body_bytes"]/1e6:.2f} MB new body bytes, {optical["peak_process_rss_bytes"]/1e6:.1f} MB peak RSS; graph audit {access["wall_seconds"]:.2f} s wall, {access["peak_process_rss_bytes"]/1e6:.1f} MB peak RSS. The earlier Rasuwa cached raw-to-report run is separate. No new model fit or GPU use.',small),
        p('Contains modified Copernicus Sentinel data 2026. Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. © OpenStreetMap contributors. Sen1Floods11: Bonafilia et al., CVPR Workshops 2020. Checking: European Union, Copernicus Emergency Management Service data. AI-assisted preparation; sources and hashes accompany this brief.',small)]
    def footer(canvas,doc):
        canvas.setFont('Helvetica',7);canvas.setFillColor(teal)
        canvas.drawString(42,24,'Source-bound research evidence. Counts are not confirmed damage.')
        canvas.drawRightString(A4[0]-42,24,'1 / 1')
    OUT.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=32,bottomMargin=42,
        title='Akash Rasuwa observation-aware evidence brief',author='Akash project - AI-assisted').build(story,onFirstPage=footer)
    pages=len(PdfReader(OUT).pages)
    if pages!=1:raise ValueError(f'Brief must remain exactly one page, got {pages}')
    for name,path in paths.items():
        if record(path)!=sources[name]:raise ValueError('Saved evidence changed during authoring')
    proof={'schema':'akash-observation-aware-brief-v1','created_at':datetime.now(timezone.utc).isoformat(),
        'sources':sources,'code':record(__file__),'output':record(OUT),'pages':pages,
        'model_invoked':False,'reference_geometry_loaded':False,'training_labels_loaded':False,
        'human_nepali_review':False,'visual_review':'pending'}
    OUT.with_suffix('.provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'output':record(OUT),'pages':pages},indent=2))


if __name__=='__main__':main()
