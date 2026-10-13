"""Export a verified water-only application without implying access results."""
import argparse
from datetime import datetime,timezone
from html import escape
import json
from pathlib import Path

from build_fresh_report import read_water,map_image,record,bound


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--water-manifest',type=Path,required=True)
    p.add_argument('--execution-receipt',type=Path,required=True)
    p.add_argument('--application-protocol',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Preserve previous application exports')
    water,arrays,grid=read_water(a.water_manifest)
    receipt=json.loads(a.execution_receipt.read_text());protocol=json.loads(a.application_protocol.read_text())
    if (receipt.get('status')!='completed' or receipt.get('gpu_used') is not False
            or receipt.get('retraining_or_retuning') is not False or receipt.get('reference_labels_loaded') is not False
            or protocol.get('frozen_before_raw_pixel_acquisition_and_inference') is not True
            or protocol.get('reference_labels_loaded') is not False
            or protocol['reuse_checkpoint']['sha256']!=water['model']['sha256']
            or protocol['case']!=water['case']):
        raise ValueError('Application receipt, source freeze and unchanged checkpoint must agree')
    if datetime.fromisoformat(protocol['frozen_at'])>=datetime.fromisoformat(receipt['started_at']):
        raise ValueError('Application protocol must precede raw acquisition execution')
    actual_water=record(a.water_manifest)
    stages=[x for x in receipt['stages'] if x['name']=='infer_water' and x['status']=='completed']
    if len(stages)!=1 or not any(all(item.get(key)==value for key,value in actual_water.items())
                                 for item in stages[0]['artifacts']):
        raise ValueError('Execution receipt does not bind this exact saved water output')
    for key in ('bbox','context_bbox','event_date'):
        if receipt['case'][key]!=water['case'][key]:
            raise ValueError('Execution timing belongs to another application case')
    def snapshot(value):
        x=datetime.fromisoformat(value.replace('Z','+00:00'))
        return x.replace(tzinfo=timezone.utc) if x.tzinfo is None else x.astimezone(timezone.utc)
    if snapshot(receipt['case']['osm_snapshot'])!=snapshot(water['case']['osm_snapshot']):
        raise ValueError('Execution historical cutoff differs from saved application')
    for item in protocol['source_code']+[protocol['reuse_checkpoint'],protocol['model_freeze'],protocol['catalog'],protocol['selected_plan']]:
        bound(item,a.application_protocol.parent)
    before=water['source_dates']['before'];after=water['source_dates']['after']
    window=protocol['event_window_utc_conservative']
    parse=lambda x:datetime.fromisoformat(x.replace('Z','+00:00'))
    if parse(before)>=parse(window['start']) or parse(after)<parse(window['end_exclusive']):
        raise ValueError('Selected source dates overlap the declared multi-day event')
    a.out.mkdir();view=map_image(a.out,arrays,grid);c=water['counts']
    for name,source in [('water-manifest.json',a.water_manifest),('execution-receipt.json',a.execution_receipt),('application-protocol.json',a.application_protocol)]:
        (a.out/name).write_bytes(source.read_bytes())
    legend=''.join(f'<span><i style="background:{color}"></i>{label}</span>' for color,label in [('#c1c8cf','Unknown'),('#edf2ee','Assessed, no candidate'),('#99bbb8','Water on both dates'),('#087e82','New-water candidate'),('#d87c43','Lost-water candidate')])
    html=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Akash | Second water application</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#eff3f1;color:#17353e;font:15px/1.55 system-ui}}main{{max-width:1160px;margin:auto;padding:32px}}h1{{font-size:34px;line-height:1.2}}h2{{font-size:21px}}.kicker{{font-size:12px;letter-spacing:.14em;color:#087e82;font-weight:700}}.notice{{background:#fff3d9;border-left:3px solid #a56223;padding:14px 18px}}.grid{{display:grid;grid-template-columns:1fr 1.1fr;gap:22px;margin:24px 0}}.panel{{background:white;border:1px solid #d4dfdc;border-radius:10px;padding:23px}}img{{width:100%;height:570px;object-fit:contain}}.legend{{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;margin:15px 0}}.legend span{{display:flex;align-items:center;gap:6px}}i{{width:12px;height:12px;display:inline-block}}.metric{{font-size:30px;font-weight:700;margin:4px 0}}td,th{{padding:8px 3px;text-align:left;border-bottom:1px solid #dbe3e0}}table{{width:100%}}a{{color:#096c75}}code{{word-break:break-all;font-size:12px}}@media(max-width:800px){{main{{padding:18px}}.grid{{grid-template-columns:1fr}}}}</style><main><div class="kicker">AKASH / UNCHANGED-MODEL APPLICATION</div><h1>Second case: Kathmandu water-state candidates</h1><p>Event peak date {escape(water['case']['event_date'])} | {escape(water['case']['case_id'])}</p><p class="notice">This tests transfer of the acquisition and model application workflow. Local flood accuracy is unvalidated. The after image is four days beyond the chosen peak. Water-only output supplies no debris, damage, road closure or access result.</p><div class="grid"><section class="panel"><h2>Native-grid water-state evidence</h2><img src="map.png" alt="Native10m water-state candidate classifications, with unknowns preserved"><div class="legend">{legend}</div><p>North is approximately up in the UTM grid. Source cells retain their values; the browser scales the image for viewing. AOI west/south/east/north: {escape(', '.join(str(x) for x in water['case']['bbox']))}.</p></section><section class="panel"><h2>Independent application, fixed model</h2><div class="metric">{c['assessed_pixels']/c['aoi_pixels']*100:.2f}% assessed</div><p>{c['assessed_pixels']:,} / {c['aoi_pixels']:,} native 10 m cells; {c['unknown_pixels']:,} remain unknown.</p><table><tr><th>State</th><th>Cells</th></tr>{''.join(f'<tr><td>{name}</td><td>{c[key]:,}</td></tr>' for name,key in [('Before water','before_water_pixels'),('After water','after_water_pixels'),('Water on both dates','unchanged_water_pixels'),('New-water candidates','new_water_candidate_pixels'),('Lost-water candidates','lost_water_candidate_pixels')])}</table><h2>Actual execution</h2><div class="metric">{receipt['elapsed_seconds']:.2f} s / CPU</div><p>Includes new raw-source acquisition, DEM preparation, terrain correction and unchanged-model inference. Prepared runtime, geoid and checkpoint were reused; catalogue discovery is separate.</p><p>Before: {escape(before)}<br>After: {escape(after)}</p><p>Both selected observations lie outside the conservatively excluded multi-day event window. Inputs and policy were frozen before new pixel acquisition. No reference labels, retraining or retuning were used.</p><p>Permanent and flood water are inseparable. Unknowns remain unassessed. New/lost cells are classification candidates, not measured event flood area.</p><p>Checkpoint SHA256:<br><code>{water['model']['sha256']}</code></p><p><a href="water-manifest.json" download>Water source receipt</a> · <a href="execution-receipt.json" download>Stage execution receipt</a> · <a href="application-protocol.json" download>Pre-acquisition source freeze</a></p></section></div><p>Contains modified Copernicus Sentinel data 2024. Copernicus WorldDEM-30: DLR e.V. 2010-2014 and Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by EU/ESA. AI-assisted research implementation.</p></main></html>'''
    (a.out/'index.html').write_text(html)
    binding={'schema':'akash-water-transfer-render-v1','recorded_at':datetime.now(timezone.utc).isoformat(),
             'inputs':[record(a.water_manifest),record(a.execution_receipt),record(a.application_protocol)],
             'code':[record(Path(__file__)),record(Path(__file__).with_name('build_fresh_report.py'))],
             'outputs':{x.name:record(x) for x in a.out.iterdir() if x.is_file()},'rendering':view,
             'local_accuracy_established':False,'model_invoked_by_renderer':False}
    (a.out/'source-and-output-manifest.json').write_text(json.dumps(binding,indent=2)+'\n')
    print(json.dumps({'status':'completed','html':record(a.out/'index.html'),'counts':c},indent=2))


if __name__=='__main__':main()
