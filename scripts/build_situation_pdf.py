"""One-page English artifact from saved actual-model report and map facts.

The model selects the four displayed radar clauses. The separate exposure
table is computed from the bound mapped-feature summary, never reference maps.
"""
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    radar=ROOT/'outputs/radar-evidence-copdem-pixel-corrected/manifest.json'
    exposure=ROOT/'outputs/radar-exposure-copdem-pixel-corrected/summary.json'
    binding=exposure.parent/'source-and-output-manifest.json'
    response_path=ROOT/'outputs/radar-copilot-development/en_report.json'
    frozen=ROOT/'outputs/radar-copilot-development/freeze.json'
    row=json.loads(response_path.read_text());response=row['response']
    if not row['passes'] or response.get('report_complete') is not True or response['language']!='en':
        raise ValueError('A successful actual English model report is required; no template fallback')
    if response['source_sha256']!=sha(radar):raise ValueError('Radar source changed after report generation')
    frozen_sources=json.loads(frozen.read_text())['sources']
    for path in (radar,exposure,binding):
        if frozen_sources[str(path.resolve())]!=sha(path):raise ValueError('Source differs from actual model run')
    required=['radar_acquisition','radar_coverage','radar_candidates','radar_limits']
    if [claim['fact_id'] for claim in response['claims']]!=required:
        raise ValueError('All four source report clauses, including limits, are required')
    summary=json.loads(exposure.read_text());manifest=json.loads(radar.read_text())
    out=ROOT/'output/pdf';out.mkdir(parents=True,exist_ok=True)
    destination=out/'akash-rasuwa-situation-report.pdf'
    if destination.exists():raise FileExistsError('Preserve earlier report artifacts')
    ink=colors.HexColor('#183647');teal=colors.HexColor('#176F68');muted=colors.HexColor('#53646B')
    styles={
        'title':ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=23,leading=26,textColor=ink),
        'kicker':ParagraphStyle('kicker',fontName='Helvetica-Bold',fontSize=9,leading=12,textColor=teal,spaceAfter=5),
        'heading':ParagraphStyle('heading',fontName='Helvetica-Bold',fontSize=10,leading=13,textColor=ink,spaceAfter=4),
        'body':ParagraphStyle('body',fontName='Helvetica',fontSize=9.2,leading=12.6,textColor=ink,spaceAfter=8),
        'small':ParagraphStyle('small',fontName='Helvetica',fontSize=7.6,leading=10,textColor=muted,spaceAfter=4),
        'hash':ParagraphStyle('hash',fontName='Courier',fontSize=6.9,leading=8.6,textColor=muted,splitLongWords=True),
        'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=8.5,leading=11,textColor=ink),
    }
    paragraph=lambda text,style='body':Paragraph(escape(text),styles[style])
    story=[paragraph('AKASH / MEASURED RADAR SCREEN / RESEARCH PROTOTYPE','kicker'),
           paragraph('Rasuwa situation report','title'),Spacer(1,8),
           paragraph('Sentinel-1D: 16 August 2026 before / 28 August 2026 after. No confirmed flood, debris, damage or passability conclusion.','small')]
    headings=['Acquisition','Assessed coverage and unknowns','Screening candidates','Interpretation limits']
    for title,claim in zip(headings,response['claims']):
        story.append(KeepTogether([paragraph(title,'heading'),paragraph(claim['text'])]))
    story.extend([Spacer(1,3),paragraph('Mapped-feature intersections - separate computed facts','heading')])
    table=[['Historical mapped feature','Mapped total','Some assessed','Change overlap','Water overlap']]
    for key,label in [('roads','Road physical segments'),('bridges','Bridge-tagged segments'),('buildings','OSM building polygons')]:
        value=summary[key]
        table.append([label,*[f"{value[k]:,}" for k in ('mapped_feature_count','features_with_assessed_radar','features_with_change_candidate_overlap','features_with_water_candidate_overlap')]])
    table=[[paragraph(str(value),'cell') for value in line] for line in table]
    grid=Table(table,colWidths=[162,71,78,80,76],hAlign='LEFT')
    grid.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8F0EE')),
        ('LINEBELOW',(0,0),(-1,0),.7,teal),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#D6DFDD')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),
        ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    story.extend([grid,Spacer(1,6),paragraph('Road overlap uses a chosen 10 m half-width (20 m total) corridor, not measured road width. Bridge segments are a subset of roads. Candidate categories may overlap. Counts are not damaged structures or closed roads; partial and unassessed coverage remains unknown.','small'),
        paragraph('Processing: Copernicus GLO-30; explicit EGM2008-to-ellipsoid height conversion; local SNAP external-DEM sampling correction. Embedded orbit vectors; no precise-orbit update or radiometric terrain flattening. No GPU or account was used for the radar run. Qwen runs separately on CPU.','small'),
        Spacer(1,5),paragraph('Audit and attribution','heading'),
        paragraph('The actual local Qwen model selected the four radar fact IDs; source clauses supply the text and numbers. The exposure table is a separate bound map calculation. No evaluation labels or reference-derived metrics enter this report.','small'),
        paragraph('Radar manifest SHA256','small'),paragraph(sha(radar),'hash'),
        paragraph('Exposure summary SHA256','small'),paragraph(sha(exposure),'hash'),Spacer(1,6),
        paragraph('Contains modified Copernicus Sentinel data 2026. Copernicus WorldDEM-30: DLR e.V. 2010-2014 and Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the EU and ESA. OpenStreetMap contributors, ODbL. AI assistance: OpenAI Codex and local Qwen.','small')])
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#D6DFDD'));canvas.line(42,35,A4[0]-42,35)
        canvas.setFont('Helvetica',7);canvas.setFillColor(muted)
        canvas.drawString(42,23,'Candidate-screen report; not an operational safety assessment')
        canvas.drawRightString(A4[0]-42,23,'1 / 1')
    doc=SimpleDocTemplate(str(destination),pagesize=A4,rightMargin=42,leftMargin=42,
        topMargin=36,bottomMargin=45,title='Akash - Rasuwa radar screening situation report',author='Akash project')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(destination)
    if len(reader.pages)!=1:raise RuntimeError('Situation report exceeded one page; repair layout before delivery')
    text=reader.pages[0].extract_text()
    for value in ('1,865,827','962,123','1,602','No confirmed','71','59'):
        if value not in text:raise RuntimeError('Missing source value in rendered report: '+value)
    evidence={'created_at':datetime.now(timezone.utc).isoformat(),'pages':1,
        'report_pdf':{'path':str(destination),'sha256':sha(destination)},
        'actual_model_report':{'path':str(response_path),'sha256':sha(response_path)},
        'inputs':{str(p):sha(p) for p in (radar,exposure,binding,frozen)},
        'builder_sha256':sha(Path(__file__)),'evaluation_labels_loaded':False,
        'visual_review':'Required after generation; page-count and text checks alone are insufficient'}
    (out/'akash-rasuwa-situation-report.provenance.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'path':str(destination),'pages':1,'sha256':sha(destination)},indent=2))


if __name__=='__main__':main()
