"""One-page English evidence brief derived only from verified saved maps.

No model, language model, new pixels or reference geometry is loaded here.
"""
from pathlib import Path
import hashlib,json
from html import escape
from datetime import datetime,timezone
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf/akash-situation-evidence-brief-2026-10-10.pdf'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if OUT.exists():raise FileExistsError('Preserve earlier evidence brief')
 paths={'water':ROOT/'outputs/fresh-one-command-replay/water/manifest.json',
        'optical':ROOT/'outputs/optical-composite-2026-10-10/manifest.json',
        'exposure':ROOT/'outputs/fresh-one-command-replay/exposure/summary.json',
        'checking':ROOT/'research/evaluation-only/optical-composite-2026-10-10/result.json'}
 sources={name:{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for name,p in paths.items()}
 water=json.loads(paths['water'].read_text())['counts'];optical=json.loads(paths['optical'].read_text());exposure=json.loads(paths['exposure'].read_text());checking=json.loads(paths['checking'].read_text())
 if water['new_water_candidate_pixels']!=30193 or optical['counts']['joint_assessed_pixels']!=265047 or checking['all_coverage_gates_pass']:raise ValueError('Unexpected evidence version')
 W=A4[0]-84;ink=colors.HexColor('#183d36');teal=colors.HexColor('#087e82')
 body=ParagraphStyle('body',fontName='Helvetica',fontSize=9,leading=12,textColor=ink,spaceAfter=6)
 small=ParagraphStyle('small',parent=body,fontSize=7.3,leading=9.8)
 head=ParagraphStyle('head',parent=body,fontName='Helvetica-Bold',fontSize=19,leading=23,spaceAfter=8)
 def p(text,style=body):
  for a,b in {'native10m':'native 10 m','native20m':'native 20 m','142.88ha':'142.88 ha','roads,0':'roads, 0','and14':'and 14'}.items():text=text.replace(a,b)
  return Paragraph(escape(text),style)
 def table(rows,widths):
  t=Table([[p(str(v),small) for v in row] for row in rows],colWidths=widths)
  t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e1ece4')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f5f0')]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));return t
 c=optical['counts'];s=exposure['access_scenario'];status=s['status_counts']
 story=[p('AKASH / RASUWA / SAVED EVIDENCE BRIEF / 10 OCTOBER 2026',small),p('Observe. Inspect. Preserve unknown.',head),
 p('Event: 26 August 2026. Educational research evidence; not an operational damage assessment or safe-route instruction. Native radar water states, optical surface-change candidates and conditional road scenarios remain distinct.'),
 table([['Saved layer / denominator','Measured result'],
 ['Radar native10m water states',f"{water['new_water_candidate_pixels']:,} new-water and {water['lost_water_candidate_pixels']:,} lost-water candidates"],
 ['Radar assessment support',f"{water['assessed_pixels']:,}/{water['aoi_pixels']:,} cells (66.12%); {water['unknown_pixels']:,} unknown"],
 ['New optical native20m support',f"{c['joint_assessed_pixels']:,}/{c['aoi_pixels']:,} cells (37.49%); {c['unknown_pixels']:,} unknown"],
 ['Provisional optical surface change',f"{c['candidate_pixels']:,} cells; 142.88ha of classifications, not confirmed debris area"]], [W*.45,W*.55]),Spacer(1,9),
 p('Observation dates matter. Radar uses the same orbit/pass before and after. The optical composite selects per-cell dates from13 source scenes using land SCL alone; assessed pre-event ages are2/4/14days and post-event ages1/6/21days. All79,555 original optical assessed cells are retained, with185,492 additional cells. Fixed thresholds are unchanged; a failed source request leaves selected cells unknown.'),
 table([['Mapped potential exposure','Conditional graph outcome'],
 ['313 physical roads,0 bridge-tagged segments and14 mapped building polygons intersect new-water candidates. These are geometric associations, not damage counts.',f"55 source settlements: {status['robust_connected']} connected in both scenario extremes; {status['ambiguous']} depend on selected road states; {status['unmapped']} unmapped; {status['baseline_disconnected']} disconnected before the scenario."]], [W*.48,W*.52]),Spacer(1,8),
 p('All313 candidate-overlapping physical road segments are uncertain open/closed. Other baseline edges are assumed available, including unknown/outside-AOI portions. Destinations stay fixed at the nearest baseline-reachable mapped town/hospital. Independently checked32paths/22cuts certify those graph assumptions, not observed closures, facility operation or safe travel.'),
 p('Checking-only result. New optical candidates intersect16.90% of the whole mapped EMSR927 mass-movement extent. Domain/reference assessment is61.66%/27.44%; unchanged90%/95% coverage gates fail. Classes differ and prior reference outcomes were known. No flood/debris accuracy, precision, recall, completeness or prospective validity is established.'),
 p('Limits. Cloud, shadow, terrain, image age, unvalidated co-registration, land-use changes and classifier error can hide or mimic change. Unknown is not nonwater or undamaged. The system cannot warn minutes before a glacier collapse. The new composite does not change the original radar/access scenario.'),
 p('Measured costs: separate optical acquisition/composition100.38s wall,121.36MB new body bytes,246.0MB peak RSS, CPU only. The earlier Rasuwa cached raw-to-report replay48.055s and its source versions remain archived; these elapsed times are separate.',small),
 p('Contains modified Copernicus Sentinel data2026. Produced using Copernicus WorldDEM-30 © DLR e.V.2010-2014 and © Airbus Defence and Space GmbH2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. © OpenStreetMap contributors. Sen1Floods11: Bonafilia et al., CVPR Workshops2020. Checking: European Union, Copernicus Emergency Management Service data. AI-assisted preparation; sources/hashes accompany this brief.',small)]
 # Preserve ordinary spacing in derived numerical prose.
 replacements={'native10m':'native 10 m','native20m':'native 20 m','142.88ha':'142.88 ha','from13':'from 13','are2/4/14days':'are 2/4/14 days','ages1/6/21days':'ages 1/6/21 days','All79,555':'All 79,555','with185,492':'with 185,492','roads,0':'roads, 0','and14':'and 14','All313':'All 313','checked32paths/22cuts':'checked 32 paths/22 cuts','intersect16.90%':'intersect 16.90%','is61.66%/27.44%':'is 61.66%/27.44%','unchanged90%/95%':'unchanged 90%/95%','composition100.38s wall,121.36MB':'composition 100.38 s wall, 121.36 MB','bytes,246.0MB':'bytes, 246.0 MB','replay48.055s':'replay 48.055 s','data2026':'data 2026','e.V.2010':'e.V. 2010','GmbH2014':'GmbH 2014','Workshops2020':'Workshops 2020'}
 for item in story:
  if isinstance(item,Paragraph):
   text=item.text
   for a,b in replacements.items():text=text.replace(a,b)
   if text!=item.text:item.__init__(text,item.style)
 def footer(canvas,doc):
  canvas.setFont('Helvetica',7);canvas.setFillColor(teal);canvas.drawString(42,24,'Saved source-bound evidence. Counts are not confirmed damage.');canvas.drawRightString(A4[0]-42,24,'1 / 1')
 SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=32,bottomMargin=42,title='Akash Rasuwa evidence brief',author='Akash project - AI-assisted').build(story,onFirstPage=footer)
 pages=len(PdfReader(OUT).pages)
 if pages!=1:raise ValueError('Brief must remain exactly one page')
 for name,pth in paths.items():
  if sha(pth)!=sources[name]['sha256']:raise ValueError('Source changed during rendering')
 proof={'created_at':datetime.now(timezone.utc).isoformat(),'sources':sources,'output':{'path':str(OUT),'sha256':sha(OUT),'bytes':OUT.stat().st_size},'pages':pages,'code_sha256':sha(__file__),'model_invoked':False,'reference_geometry_loaded':False,'human_nepali_review':False,'visual_review':'pending'}
 OUT.with_suffix('.provenance.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof['output'],indent=2))
if __name__=='__main__':main()
