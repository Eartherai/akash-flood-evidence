"""Build an offline case report from bound water and exposure artifacts.

This deterministic report does not invoke a model, read reference labels, or
turn candidate intersections into observed damage. The trained water layer
and the explicitly hypothetical road scenario remain separate evidence.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import struct
import time
import zlib

import numpy as np
from pyproj import CRS, Transformer
import rasterio
from rasterio.features import shapes

from akash_flood.fresh_case import reject_source_path
from akash_flood.water_inference import temporal_layers

ROOT = Path(__file__).resolve().parents[1]
STATUS_LABELS = {
    'robust_connected': 'Route in both scenario extremes',
    'ambiguous': 'Access depends on candidate-road states',
    'baseline_disconnected': 'No route in baseline graph',
    'robust_newly_disconnected': 'No route even in optimistic scenario',
    'unmapped': 'Settlement or destination association unavailable',
}
COLORS = {'unknown': '#C1C8CF', 'assessed': '#EDF2EE',
          'unchanged': '#99BBB8', 'new': '#087E82', 'lost': '#D87C43'}


def record(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest}


def bound(item, parent):
    path = Path(item['path'])
    path = (path if path.is_absolute() else parent / path).resolve()
    reject_source_path(path)
    actual = record(path)
    if actual['sha256'] != item['sha256'] or actual['bytes'] != item['bytes']:
        raise ValueError('Source checksum or size mismatch: ' + path.name)
    return path


def read_json(path):
    reject_source_path(path)
    if path.stat().st_size > 100_000_000:
        raise ValueError('JSON evidence exceeds 100 MB report limit')
    return json.loads(path.read_text())


def read_water(path):
    manifest = read_json(path)
    if (manifest.get('schema') != 'akash-fresh-water-v1'
            or manifest.get('available') is not True
            or manifest.get('reference_labels_loaded') is not False
            or manifest.get('gpu_used') is not False):
        raise ValueError('Require completed reference-free fresh water evidence')
    files = {name: bound(item, path.parent) for name, item in manifest['outputs'].items()}
    bound(manifest['policy'], path.parent)
    bound(manifest['processing_record'], path.parent)
    arrays, grid = {}, None
    for name in ('aoi-mask.tif', 'valid-mask.tif', 'before-water.tif',
                 'after-water.tif', 'new-water-candidates.tif', 'lost-water-candidates.tif'):
        with rasterio.open(files[name]) as src:
            signature = (src.crs, src.transform, src.width, src.height)
            if src.count != 1 or src.crs is None or (grid is not None and signature != grid):
                raise ValueError('Report requires exactly matching single-band grids')
            grid = signature
            if src.width * src.height > 20_000_000:
                raise ValueError('Raster exceeds the declared 20 million cell report limit')
            arrays[name] = src.read(1)
    crs = CRS.from_user_input(grid[0])
    if not crs.is_projected or any(axis.unit_name != 'metre' for axis in crs.axis_info):
        raise ValueError('Metric projected evidence required')
    declared = manifest['grid']
    if (declared['epsg'] != crs.to_epsg() or declared['shape'] != [grid[3], grid[2]]
            or declared['transform'] != list(grid[1])
            or declared.get('resolution_m') != 10
            or grid[1].a != 10 or grid[1].e != -10 or grid[1].b != 0 or grid[1].d != 0
            or declared.get('resampling_used_for_date_alignment') is not False):
        raise ValueError('Declared water grid differs from saved raster grid')
    aoi, valid = arrays['aoi-mask.tif'], arrays['valid-mask.tif']
    if not np.isin(aoi, (0, 1)).all() or not np.isin(valid, (0, 1)).all() or not aoi.any():
        raise ValueError('Require nonempty binary AOI and assessment masks')
    new, lost, counts = temporal_layers(arrays['before-water.tif'], arrays['after-water.tif'], aoi == 1)
    if (not np.array_equal(valid == 1, arrays['before-water.tif'] != 255)
            or not np.array_equal(new, arrays['new-water-candidates.tif'])
            or not np.array_equal(lost, arrays['lost-water-candidates.tif'])
            or counts != manifest['counts']):
        raise ValueError('Temporal state, assessment mask or recounted totals disagree')
    return manifest, arrays, grid


def read_exposure(path, water_path, water):
    manifest = read_json(path)
    if manifest.get('schema') != 'akash-fresh-exposure-binding-v1' or manifest.get('status') != 'completed':
        raise ValueError('Require completed bound fresh exposure evidence')
    files = {name: bound(item, path.parent) for name, item in manifest['outputs'].items()}
    summary = read_json(files['summary.json'])
    if (summary.get('schema') != 'akash-fresh-exposure-v1' or summary.get('status') != 'completed'
            or summary['case'] != water['case'] or summary['temporal_counts'] != water['counts']
            or summary['water_source'] != record(water_path)):
        raise ValueError('Exposure is not bound to this exact water case')
    historic = summary['historical_osm']
    bound(historic['source'], path.parent)
    bound(historic['provenance'], path.parent)
    if historic['snapshot'] != water['case']['osm_snapshot']:
        raise ValueError('Historical snapshot differs between case artifacts')
    scenario = summary['access_scenario']
    if scenario.get('scenario_is_hypothetical') is not True or scenario.get('confirmed_closures') != 0:
        raise ValueError('Only explicitly hypothetical scenarios are supported')
    if scenario.get('n_uncertain_physical_segments', -1) < 0 or not scenario.get('certificate_checks'):
        raise ValueError('Scenario membership and certificate verification are required')
    proof = read_json(files['access-proof.json'])
    actual_counts = dict(Counter(row['status'] for row in proof['settlements']))
    if actual_counts != scenario['status_counts'] or len(proof['settlements']) != scenario['n_settlements']:
        raise ValueError('Access summary differs from proof rows')
    for row in proof['settlements']:
        selection = row.get('nearest_baseline_destination')
        for key in ('path_witness', 'optimistic_path_witness'):
            if row.get(key) and (selection is None or row[key]['destination_id'] != selection['destination_id']):
                raise ValueError('Route switched away from selected baseline destination')
    return manifest, summary, proof, files


def read_optical(path, water):
    """Verify a separate native optical layer, without resampling onto radar."""
    path=Path(path).resolve();manifest=read_json(path)
    if (manifest.get('schema')!='akash-s2-surface-change-v1' or manifest.get('status')!='completed'
            or manifest.get('reference_labels_loaded') is not False or manifest.get('gpu_used') is not False
            or manifest.get('local_debris_accuracy_established') is not False
            or manifest.get('partial_exploration_is_original_gate_pass') is not False
            or manifest.get('scene_date_alignment_resampling_used') is not False):
        raise ValueError('Require explicitly partial, reference-free optical candidate evidence')
    if manifest['bbox']!=water['case']['bbox'] or manifest['event_date']!=water['case']['event_date']:
        raise ValueError('Optical layer belongs to a different case')
    files={name:bound(item,path.parent) for name,item in manifest['outputs'].items()}
    protocol=read_json(bound(manifest['protocol'],path.parent))
    transport=read_json(bound(manifest['transport'],path.parent))
    bound(protocol['catalog'],path.parent)
    for item in protocol['source_items'].values():bound(item,path.parent)
    for group in manifest['source_crops'].values():
        for item in group.values():bound(item['crop'],path.parent)
    grid=None;arrays={}
    for name in ('surface-change-candidates.tif','aoi-mask.tif','valid-mask.tif'):
        with rasterio.open(files[name]) as ds:
            signature=(ds.crs,ds.transform,ds.width,ds.height)
            if ds.count!=1 or ds.dtypes[0]!='uint8' or ds.width*ds.height>4_000_000 or (grid and grid!=signature):
                raise ValueError('Optical report requires exact matching bounded native grids')
            grid=signature;arrays[name]=ds.read(1)
    declared=manifest['grid'];crs=CRS.from_user_input(grid[0])
    if (not crs.is_projected or any(axis.unit_name!='metre' for axis in crs.axis_info)
            or crs.to_epsg()!=water['grid']['epsg'] or grid[1].a!=20 or grid[1].e!=-20
            or grid[1].b!=0 or grid[1].d!=0 or declared['shape']!=[grid[3],grid[2]]
            or declared['transform']!=list(grid[1]) or CRS.from_user_input(declared['crs'])!=crs):
        raise ValueError('Optical evidence is not on its declared native20m grid')
    labels=arrays['surface-change-candidates.tif'];aoi=arrays['aoi-mask.tif'];valid=arrays['valid-mask.tif']
    if (not np.isin(labels,(0,1,255)).all() or not np.isin(aoi,(0,1)).all() or not aoi.any()
            or not np.isin(valid,(0,1)).all() or not np.array_equal(valid==1,labels!=255)
            or np.any((valid==1)&(aoi==0))):
        raise ValueError('Optical unknown/assessed/native-AOI domains disagree')
    counts={'aoi_pixels':int(aoi.sum()),'joint_assessed_pixels':int(valid.sum()),
            'unknown_pixels':int(((aoi==1)&(valid==0)).sum()),'candidate_pixels':int((labels==1).sum())}
    for name in ('before','after'):
        source=bound(manifest['source_crops'][name]['scl']['crop'],path.parent)
        with rasterio.open(source) as ds:
            if (ds.crs,ds.transform,ds.width,ds.height)!=grid:
                raise ValueError('Optical source SCL is not on the saved native grid')
            clear=int(((aoi==1)&np.isin(ds.read(1),(4,5,6))).sum())
        actual={'raw_scl_clear_pixels':clear,'fraction':clear/counts['aoi_pixels'],
                'original70percentgate_passes':clear/counts['aoi_pixels']>=.7}
        if actual!=manifest['raw_scl_single_scene_clear'][name]:
            raise ValueError('Recounted optical source SCL coverage disagrees')
    if (counts!=manifest['counts'] or manifest['candidate_area_m2']!=counts['candidate_pixels']*400
            or manifest['joint_assessed_fraction']!=counts['joint_assessed_pixels']/counts['aoi_pixels']
            or sum(r['received_body_bytes'] for r in transport['requests'])!=manifest['network_received_body_bytes']):
        raise ValueError('Recounted optical totals or transport receipt disagree')
    expected=np.full((*aoi.shape,3),255,dtype='uint8');expected[aoi==1]=[193,200,207]
    expected[valid==1]=[232,240,230];expected[labels==1]=[147,78,35]
    facts={key:manifest[key] for key in ('counts','source_dates','joint_assessed_fraction','candidate_area_m2','raw_scl_single_scene_clear','scope','limitations')}
    facts['manifest']=record(path)
    return facts,expected


def write_png(path, rgb):
    """Lossless RGB PNG using the standard library; no plotting dependency."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected RGB uint8 native-grid image')
    def chunk(kind, payload):
        return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))
    height, width = rgb.shape[:2]
    rows = b''.join(b'\0' + row.tobytes() for row in rgb)
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(rows, 6)) + chunk(b'IEND', b''))


def map_image(out, arrays, grid):
    aoi = arrays['aoi-mask.tif'] == 1
    rows, cols = np.where(aoi)
    top, bottom, left, right = rows.min(), rows.max()+1, cols.min(), cols.max()+1
    sl = np.s_[top:bottom, left:right]
    rgb = np.full((bottom-top, right-left, 3), 255, dtype='uint8')
    color = lambda key: tuple(bytes.fromhex(COLORS[key][1:]))
    rgb[aoi[sl]] = color('unknown')
    rgb[(arrays['valid-mask.tif'] == 1)[sl]] = color('assessed')
    rgb[((arrays['before-water.tif'] == 1) & (arrays['after-water.tif'] == 1))[sl]] = color('unchanged')
    rgb[(arrays['new-water-candidates.tif'] == 1)[sl]] = color('new')
    rgb[(arrays['lost-water-candidates.tif'] == 1)[sl]] = color('lost')
    write_png(out / 'map.png', rgb)
    return {'row_start': int(top), 'col_start': int(left), 'width': int(right-left), 'height': int(bottom-top),
            'display_resampling': False, 'native_spacing_m': 10, 'legend': COLORS}


def evidence_svg(out, grid, view, files, proof, aoi_mask):
    width, height = view['width'], view['height']
    tx = Transformer.from_crs(4326, grid[0], always_xy=True)
    inv = ~grid[1]
    def xy(lon, lat):
        col, row = inv * tx.transform(lon, lat)
        return (round(col-view['col_start'], 2), round(row-view['row_start'], 2))
    road = read_json(files['roads.geojson'])
    segments, overlays = {}, []
    for f in road['features']:
        points = [xy(*coordinate) for coordinate in f['geometry']['coordinates']]
        for edge in f['properties']['source_directed_edges']:
            segments[edge['edge_id']] = points
        if len(points) < 2:
            continue
        # The viewport clips the expanded historical network. It is not pruned
        # from the analysis or represented as a complete physical road map.
        if (max(p[0] for p in points) < 0 or min(p[0] for p in points) > width
                or max(p[1] for p in points) < 0 or min(p[1] for p in points) > height):
            continue
        uncertain = f['properties']['coverage']['new_water_candidate'] > 1e-6
        coords = ' '.join(f'{x},{y}' for x, y in points)
        overlays.append(f'<polyline points="{coords}" fill="none" stroke="{"#9C3929" if uncertain else "#4D5962"}" stroke-width="{2.1 if uncertain else .8}" opacity="{.85 if uncertain else .38}"/>')
    data = {}
    for row in proof['settlements']:
        paths, cuts = {}, []
        for key in ('path_witness', 'optimistic_path_witness'):
            if row.get(key):
                paths[key] = [segments[e] for e in row[key]['path_edge_ids']]
        certificate = row.get('pessimistic_cut_certificate') or row.get('unreachable_certificate')
        if certificate:
            cuts = [segments[e] for e in certificate['removed_edge_ids']]
        data[row['id']] = {'paths': paths, 'cuts': cuts}
    encoded = base64.b64encode((out / 'map.png').read_bytes()).decode()
    # SVG overflow clips its CSS box, which can include letterboxing outside
    # the viewBox. Clip overlays to the exact native AOI cells as well.
    cropped_aoi = aoi_mask[view['row_start']:view['row_start']+height,
                           view['col_start']:view['col_start']+width]
    paths = []
    for geometry, value in shapes(cropped_aoi.astype('uint8'), mask=cropped_aoi == 1):
        if value != 1:
            continue
        rings = geometry['coordinates']
        paths.append(' '.join('M'+' L'.join(f'{x},{y}' for x,y in ring)+' Z' for ring in rings))
    clip = ''.join(f'<path d="{path}" clip-rule="evenodd"/>' for path in paths)
    svg = (f'<svg id="evidence-map" role="img" aria-label="Native-grid water-state candidates and historical road scenario" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
           f'<defs><clipPath id="native-aoi-clip" clipPathUnits="userSpaceOnUse">{clip}</clipPath></defs>'
           f'<image href="data:image/png;base64,{encoded}" width="{width}" height="{height}" style="image-rendering:pixelated"/>'
           f'<g id="road-layer" clip-path="url(#native-aoi-clip)">{"".join(overlays)}</g><g id="route-layer" clip-path="url(#native-aoi-clip)"></g></svg>')
    return svg, data


def report_facts(water, summary, proof):
    counts = water['counts']
    pixel_area = 100.0
    return {'schema': 'akash-fresh-report-v1', 'case': water['case'], 'source_dates': water['source_dates'],
            'counts': counts, 'assessed_fraction': counts['assessed_pixels']/counts['aoi_pixels'],
            'new_water_classification_candidate_area_km2': counts['new_water_candidate_pixels']*pixel_area/1e6,
            'lost_water_classification_candidate_area_km2': counts['lost_water_candidate_pixels']*pixel_area/1e6,
            'mapped_features': {kind: summary[kind] for kind in ('roads', 'bridges', 'buildings')},
            'access_scenario': summary['access_scenario'], 'settlements': proof['settlements'],
            'model_sha256': water['model']['sha256'], 'model_bytes': water['model']['bytes'],
            'historical_snapshot': summary['historical_osm']['snapshot'],
            'ai_role': 'Frozen Sen1-trained water classifier; report text and all numbers are deterministic source clauses.',
            'scope': 'Experimental water-state candidates and conditional graph scenarios. Local hazard accuracy unvalidated.',
            'debris_mapping': 'not implemented', 'confirmed_closures': 0, 'reference_labels_loaded': False,
            'limits': [
                'Permanent and flood water are inseparable; new/lost classifications are provisional and are not measured flood area.',
                'Masked coverage is unknown. Uncalibrated classifier scores and 10 m grid spacing do not establish accuracy.',
                'Candidate-road states form a hypothetical scenario. Remaining baseline edges are assumed available; unspecified failures and radar-unknown roads are not certified passable.',
                'Mapped node-tagged destinations lack entrance and operating-status validation. Nearest means directed baseline road length plus fixed node associations, not travel time.',
                'OSM incompleteness, network boundaries, missing facilities and unverified restrictions limit modeled routes.',
                'No debris, physical damage, casualties, operational rescue routing or independent unseen-event validation is established.',
            ]}


def html_report(facts, svg, routes):
    c = facts['counts']; scenario = facts['access_scenario']
    feature_rows = ''.join('<tr><th>'+escape(kind.title())+'</th>'+''.join('<td>'+f'{facts["mapped_features"][kind][key]:,}'+'</td>' for key in
        ('mapped_feature_count', 'features_with_assessed_water_state', 'features_with_unknown_water_state', 'features_with_new_water_candidate_overlap', 'features_with_lost_water_candidate_overlap'))+'</tr>' for kind in ('roads', 'bridges', 'buildings'))
    status_rows = ''.join(f'<div class="status"><strong>{value:,}</strong><span>{escape(STATUS_LABELS[key])}</span></div>' for key,value in scenario['status_counts'].items())
    settlement_rows = ''
    for row in facts['settlements']:
        nearest = row.get('nearest_baseline_destination')
        target = str(nearest['destination_id']) if nearest else 'Unavailable'
        distance = f'{nearest["baseline_distance_including_association_m"]/1000:.2f} km' if nearest else '-'
        settlement_rows += f'<tr><td><button class="route" data-id="{escape(row["id"], quote=True)}">{escape(row["name"])}</button></td><td>{escape(STATUS_LABELS[row["status"]])}</td><td>{escape(target)}</td><td>{distance}</td></tr>'
    legend = ''.join(f'<span><i style="background:{value}"></i>{escape({"unknown":"Unknown","assessed":"Assessed, no water-state candidate","unchanged":"Water on both dates","new":"New water-state candidate","lost":"Lost water-state candidate"}[key])}</span>' for key,value in COLORS.items())
    route_json = json.dumps(routes, ensure_ascii=True, separators=(',', ':')).replace('<', '\\u003c')
    limits = ''.join('<li>'+escape(text)+'</li>' for text in facts['limits'])
    case = facts['case']
    optical=''
    if facts.get('optical'):
        o=facts['optical'];oc=o['counts']
        gate_text=('The original 70% optical coverage gate fails on at least one source date.'
                   if not all(x['original70percentgate_passes'] for x in o['raw_scl_single_scene_clear'].values())
                   else 'Both source dates pass the raw 70% SCL coverage gate; the joint land-support mask is stricter.')
        optical=f'''<section class="panel wide"><h2>Optical vegetation-loss and exposed-surface candidates</h2><div class="main-grid"><div><img src="optical-overview.png" alt="Native20m optical candidates: brown candidates, pale green assessed cells, gray unknown" style="width:100%;height:370px;object-fit:contain"><div class="legend"><span><i style="background:#c1c8cf"></i>Unknown</span><span><i style="background:#e8f0e6"></i>Assessed, no candidate</span><span><i style="background:#934e23"></i>Surface-change candidate</span></div></div><div><strong>{o['joint_assessed_fraction']*100:.2f}% jointly assessed</strong><p>{oc['joint_assessed_pixels']:,} of {oc['aoi_pixels']:,} native 20 m cells. {oc['unknown_pixels']:,} remain unknown.</p><p><strong>{oc['candidate_pixels']:,} candidate cells</strong> ({o['candidate_area_m2']/10000:.2f} ha of classifications).</p><p>Before: {escape(o['source_dates']['before'])}<br>After: {escape(o['source_dates']['after'])}</p><p>Both dates must have supported land reflectance outside the fixed 60 m cloud/shadow/invalid neighborhood. Candidates combine vegetation loss with an exposed-surface spectral change.</p><p class="notice">Numeric thresholds are untuned hypotheses. Agriculture, riverbank change, vegetation removal and sensor errors can trigger candidates. This is a partial review layer; local debris accuracy is unvalidated. {gate_text}</p><p>No optical candidate assigns road closures or changes the radar scenario. The layers keep their own native grids.</p><p class="links"><a href="optical-manifest.json" download>Optical source receipt</a><a href="optical-protocol.json" download>Fixed optical protocol</a></p></div></div></section>'''
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Akash | Case evidence</title>
<style>:root{{--ink:#17353e;--muted:#526770;--teal:#087e82}}*{{box-sizing:border-box}}body{{margin:0;background:#eff3f1;color:var(--ink);font:15px/1.55 system-ui}}main{{max-width:1180px;margin:auto;padding:32px}}h1{{font-size:36px;line-height:1.15;margin:5px 0 12px}}h2{{font-size:21px;margin:0 0 14px}}p{{margin:8px 0}}.kicker{{color:var(--teal);font-size:12px;font-weight:700;letter-spacing:.14em}}.notice{{background:#fff3d9;border-left:3px solid #a56223;padding:13px 17px;margin:20px 0}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:22px 0}}.card,.panel{{background:#fff;border:1px solid #d4dfdc;border-radius:10px;padding:22px}}.card strong{{display:block;font-size:27px;line-height:1.2;margin:7px 0}}.muted,small{{color:var(--muted)}}.main-grid{{display:grid;grid-template-columns:1fr 1.15fr;gap:20px}}#evidence-map{{display:block;width:100%;height:560px;background:#fff;overflow:hidden}}.legend{{display:flex;gap:10px;flex-wrap:wrap;font-size:12px;margin:14px 0}}.legend span{{display:flex;align-items:center;gap:6px}}.legend i{{width:13px;height:13px;display:inline-block}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:9px 6px;border-bottom:1px solid #dbe3e0;text-align:left;vertical-align:top}}.statuses{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:15px 0}}.status{{padding:12px;background:#f0f5f3;border-radius:6px}}.status strong{{font-size:25px;display:block}}.status span{{font-size:12px}}button{{cursor:pointer;color:#075f65;background:#e5f0ed;border:1px solid #bfd2cc;border-radius:5px;padding:6px 9px}}button:hover{{background:#cfdfdb}}.table-scroll{{max-height:400px;overflow:auto}}.wide{{margin-top:20px}}a{{color:#096c75}}code{{word-break:break-all;font-size:12px}}.links{{display:flex;gap:18px;flex-wrap:wrap}}@media(max-width:850px){{main{{padding:18px}}.cards{{grid-template-columns:1fr 1fr}}.main-grid{{grid-template-columns:1fr}}#evidence-map{{height:480px}}}}@media print{{main{{padding:0}}button{{border:0;background:none}}.table-scroll{{max-height:none}}.cards{{grid-template-columns:repeat(4,1fr)}}.main-grid{{grid-template-columns:1fr 1fr}}#evidence-map{{height:420px}}}}
</style><main><div class="kicker">AKASH / SOURCE-BOUND CASE EVIDENCE</div><h1>Water-state changes and road-access scenarios</h1><p class="muted">Event date {escape(case['event_date'])} | {escape(case['case_id'])} | EPSG native grid preserved</p>
<p class="notice">Experimental candidate map. Local flood accuracy is unvalidated; debris mapping is incomplete. Road overlap and scenario connectivity are not observed damage, closure or safe-route findings.</p>
<div class="cards"><div class="card"><small>Assessed AOI</small><strong>{facts['assessed_fraction']*100:.2f}%</strong><span>{c['assessed_pixels']:,} / {c['aoi_pixels']:,} cells</span></div><div class="card"><small>Unknown coverage</small><strong>{c['unknown_pixels']:,}</strong><span>Unsupported 10 m cells</span></div><div class="card"><small>New water-state candidates</small><strong>{c['new_water_candidate_pixels']:,}</strong><span>{facts['new_water_classification_candidate_area_km2']:.4f} km2 of classifications</span></div><div class="card"><small>Lost water-state candidates</small><strong>{c['lost_water_candidate_pixels']:,}</strong><span>{facts['lost_water_classification_candidate_area_km2']:.4f} km2 of classifications</span></div></div>
<div class="main-grid"><section class="panel"><h2>Native-grid classification evidence</h2>{svg}<div class="legend">{legend}</div><small>North is approximately up in the UTM grid. Gray lines: historical mapped roads. Red lines: road centerlines whose corridors intersect new-water candidates. Map clips to the AOI; access analysis retains the expanded network.</small><p><button id="toggle-roads">Hide roads</button> <button id="clear-route">Clear selected proof</button></p><p id="route-note" aria-live="polite" class="muted">Choose a settlement to draw its saved route and cut evidence.</p></section>
<section class="panel"><h2>Mapped-feature intersections</h2><table><thead><tr><th>Feature</th><th>Mapped</th><th>Some assessed</th><th>Some unknown</th><th>New overlap</th><th>Lost overlap</th></tr></thead><tbody>{feature_rows}</tbody></table><p class="muted">Physical road segments use a chosen 10 m half-width corridor. Bridges are a subset of roads. Building counts are OSM polygon features. Some-assessed and some-unknown counts can overlap; these are not damage counts.</p>
<h2 style="margin-top:28px">Conditional access bounds</h2><p>{scenario['n_uncertain_physical_segments']:,} physical road segments have two hypothetical states, open or closed. No observed closures are supplied. Each settlement keeps its nearest baseline-reachable mapped town/hospital destination.</p><div class="statuses">{status_rows}</div><p class="muted">{scenario['certificate_checks']['paths']:,} saved paths and {scenario['certificate_checks']['cuts']:,} cuts independently checked against source edge identities. The certificate proves the stated graph scenario, not real-world passability.</p></section></div>
{optical}<section class="panel wide"><h2>Inspect a settlement's proof</h2><div class="table-scroll"><table><thead><tr><th>Settlement</th><th>Conditional graph result</th><th>Fixed mapped destination</th><th>Baseline length + associations</th></tr></thead><tbody>{settlement_rows}</tbody></table></div></section>
<section class="panel wide"><h2>Observations, model and limits</h2><p>Sentinel-1 before: {escape(facts['source_dates']['before'])}. After: {escape(facts['source_dates']['after'])}. Historical OSM cutoff: {escape(facts['historical_snapshot'])}.</p><p>Before-water classifications: {c['before_water_pixels']:,}; after-water: {c['after_water_pixels']:,}; water on both dates: {c['unchanged_water_pixels']:,}. Both dates use the identical assessed domain.</p><p>{escape(facts['ai_role'])} Checkpoint: {facts['model_bytes']:,} bytes; classifier threshold 0.5; scores are uncalibrated.</p><ul>{limits}</ul><p class="links"><a href="report.json" download>Computed report facts</a><a href="access-proof.json" download>Routes and cut certificates</a><a href="source-and-output-manifest.json" download>Report source bindings</a><a href="map.png" download>Native-grid image</a></p><small>Contains modified Copernicus Sentinel data 2026. Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. © OpenStreetMap contributors. ODbL applies to OSM data. Training: Sen1Floods11, Bonafilia, Tellman, Anderson and Issenberg, CVPR Workshops 2020. AI-assisted implementation.</small></section></main>
<script>const proofs={route_json};const layer=document.getElementById('route-layer');function draw(points,color,width){{const p=document.createElementNS('http://www.w3.org/2000/svg','polyline');p.setAttribute('points',points.map(v=>v.join(',')).join(' '));p.setAttribute('fill','none');p.setAttribute('stroke',color);p.setAttribute('stroke-width',width);layer.append(p)}}document.querySelectorAll('.route').forEach(b=>b.onclick=()=>{{layer.replaceChildren();const v=proofs[b.dataset.id];Object.values(v.paths).flat().forEach(p=>draw(p,'#2638b3',4));v.cuts.forEach(p=>draw(p,'#ec214e',8));document.getElementById('route-note').textContent=b.textContent+': blue = saved route; pink = removed-edge cut. Evidence outside the AOI is clipped from this view. No route means unavailable or disconnected within the modeled graph, not a field finding.';document.getElementById('evidence-map').scrollIntoView({{behavior:'smooth',block:'center'}})}});document.getElementById('clear-route').onclick=()=>{{layer.replaceChildren();document.getElementById('route-note').textContent='Choose a settlement to draw its saved route and cut evidence.'}};document.getElementById('toggle-roads').onclick=function(){{const e=document.getElementById('road-layer');e.style.display=e.style.display==='none'?'':'none';this.textContent=e.style.display==='none'?'Show roads':'Hide roads'}};</script></html>'''


def pdf_report(path, facts, map_path, water_record, exposure_record):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from pypdf import PdfReader
    if path.exists():
        raise FileExistsError('Preserve earlier PDF artifacts')
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = {name: ParagraphStyle(name, fontName='Helvetica-Bold' if name in ('title','heading') else 'Helvetica',
        fontSize=size, leading=leading, textColor=colors.HexColor('#17353e'), spaceAfter=6)
        for name,size,leading in [('title',21,24),('heading',10,13),('body',9,12),('small',7.4,9.5)]}
    p = lambda text, style='body': Paragraph(escape(text), styles[style])
    c=facts['counts']; scenario=facts['access_scenario']; story=[p('AKASH / SOURCE-BOUND RESEARCH CASE','small'),
        p('Satellite change and access evidence' if facts.get('optical') else 'Water-state and access evidence','title'),p('Event '+facts['case']['event_date']+' | '+facts['case']['case_id'],'small'),
        p('Experimental classifications, not confirmed flood, debris, damage or passability. Local hazard accuracy unvalidated.'),
        p('Acquisition and observation coverage','heading'),
        p('Sentinel-1: '+facts['source_dates']['before']+' before / '+facts['source_dates']['after']+' after. OSM cutoff: '+facts['historical_snapshot']+'.','small'),
        p(f"Assessed {c['assessed_pixels']:,} of {c['aoi_pixels']:,} 10 m cells ({facts['assessed_fraction']*100:.2f}%); {c['unknown_pixels']:,} remain unknown. Before-water {c['before_water_pixels']:,}; after-water {c['after_water_pixels']:,}; water on both dates {c['unchanged_water_pixels']:,}."),
        p(f"New-water-state candidates: {c['new_water_candidate_pixels']:,} cells ({facts['new_water_classification_candidate_area_km2']:.4f} km2 of classifications). Lost-water-state: {c['lost_water_candidate_pixels']:,} cells. Permanent and flood water are inseparable; these are not measured flood area."),
        p('Native-grid water-state map','heading')]
    png_width,png_height=struct.unpack('>II',map_path.read_bytes()[16:24])
    thumb_height=130 if facts.get('optical') else 145
    thumb=Image(str(map_path),width=thumb_height*png_width/png_height,height=thumb_height)
    legend_rows=[]
    for key,label in [('unknown','Unknown'),('assessed','Assessed, no water-state candidate'),('unchanged','Water on both dates'),('new','New water-state candidate'),('lost','Lost water-state candidate')]:
        legend_rows.append(['',p(label,'small')])
    legend=Table(legend_rows,colWidths=[13,295])
    legend.setStyle(TableStyle([('BACKGROUND',(0,i),(0,i),colors.HexColor(COLORS[key]))
        for i,key in enumerate(('unknown','assessed','unchanged','new','lost'))]+[('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    context=[legend,Spacer(1,8),p('AOI west/south/east/north: '+', '.join(str(x) for x in facts['case']['bbox'])+'. North approximately up. The 10 m UTM classification image retains native cells. It does not display observed damage or a complete hazard inventory.','small')]
    map_table=Table([[thumb,context]],colWidths=[130,341])
    map_table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
    story.extend([map_table])
    if facts.get('optical'):
        o=facts['optical'];oc=o['counts']
        gate=('original 70% coverage gate fails on a source date' if not all(x['original70percentgate_passes'] for x in o['raw_scl_single_scene_clear'].values()) else 'raw 70% SCL gate passes both dates; joint support is stricter')
        story.append(p(f"Optical partial review: {oc['joint_assessed_pixels']:,}/{oc['aoi_pixels']:,} native 20 m cells assessed ({o['joint_assessed_fraction']*100:.2f}%); {oc['unknown_pixels']:,} unknown; {oc['candidate_pixels']:,} vegetation-loss/exposed-surface candidate cells. Untuned thresholds; no validated debris finding; {gate}.",'small'))
    story.append(p('Mapped-feature intersections','heading'))
    table=[['Feature','Mapped','Some assessed','Some unknown','New overlap','Lost overlap']]
    for kind in ('roads','bridges','buildings'):
        table.append([kind.title(),*[f"{facts['mapped_features'][kind][key]:,}" for key in
            ('mapped_feature_count','features_with_assessed_water_state','features_with_unknown_water_state','features_with_new_water_candidate_overlap','features_with_lost_water_candidate_overlap')]])
    t=Table([[p(str(x),'small') for x in row] for row in table], colWidths=[70,57,95,95,77,77])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5efeb')),('VALIGN',(0,0),(-1,-1),'TOP'),
                          ('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,-1),.25,colors.HexColor('#cbd9d3'))]))
    story.extend([t,Spacer(1,7),p('Roads are physical segments; 10 m half-width corridors are geometric associations, not observed width. Bridge segments are a subset of roads. Building counts are mapped OSM polygons. Assessed/unknown totals may overlap. No damage or closure count is inferred.','small'),
        p('Hypothetical nearest-destination access','heading'),
        p(f"{scenario['n_uncertain_physical_segments']:,} new-water-overlapping road segments are uncertain open/closed; all other baseline edges are assumed available. No observed closures. "+'; '.join(f"{value} {STATUS_LABELS[key].lower()}" for key,value in scenario['status_counts'].items())+'.'),
        p(f"{scenario['certificate_checks']['paths']} paths and {scenario['certificate_checks']['cuts']} cuts checked against source edges. Destinations stay fixed by baseline directed road length plus node associations, not travel time. Facilities, entrances, map completeness and operability are unverified. Unknown radar coverage does not certify passability.",'small'),
        p('Model and remaining limits','heading'),p(f"Frozen Sen1Floods11-trained CPU water model: {facts['model_bytes']:,} byte checkpoint, threshold 0.5. Report numbers are deterministic source clauses. Scores are uncalibrated. No debris layer, local flood validation, absolute geolocation validation or unseen-event raw-to-report proof is established.",'small'),
        p('Source bindings','heading'),p('Water SHA256: '+water_record['sha256'],'small'),p('Exposure binding SHA256: '+exposure_record['sha256'],'small'),
        p('Contains modified Copernicus Sentinel data 2026. Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. © OpenStreetMap contributors. ODbL applies to OSM data. Training: Sen1Floods11, Bonafilia et al., CVPR Workshops 2020. AI-assisted implementation.','small')])
    if facts.get('optical'):
        story.insert(-1,p('Optical SHA256: '+facts['optical']['manifest']['sha256'],'small'))
    def footer(canvas,doc):
        canvas.setFont('Helvetica',7);canvas.drawString(42,24,'Research candidate report - not an operational safety assessment')
        canvas.drawRightString(A4[0]-42,24,str(doc.page))
    SimpleDocTemplate(str(path),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=34,bottomMargin=42,
                      title='Akash fresh case evidence report').build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(path)
    if len(reader.pages)!=1:
        raise ValueError('Repair PDF layout: situation report must be exactly one page')
    text=' '.join(reader.pages[0].extract_text().split())
    for value in (f"{c['assessed_pixels']:,}", f"{c['unknown_pixels']:,}", 'No observed closures', 'No debris layer'):
        if value not in text:
            raise ValueError('PDF is missing a required computed value or limit')


def run(water_path, exposure_path, out, pdf_output=None, optical_path=None):
    start=time.perf_counter();water_path=Path(water_path).resolve();exposure_path=Path(exposure_path).resolve();out=Path(out).resolve()
    if out.exists():
        raise FileExistsError('Use a fresh report output folder')
    sources=[record(water_path),record(exposure_path)]
    water, arrays, grid=read_water(water_path)
    _, summary, proof, files=read_exposure(exposure_path,water_path,water)
    facts=report_facts(water,summary,proof)
    if optical_path:
        facts['optical'],optical_image=read_optical(Path(optical_path),water)
        sources.append(record(optical_path))
    out.mkdir(parents=True,exist_ok=False)
    if optical_path:
        write_png(out/'optical-overview.png',optical_image)
        optical_document=read_json(Path(optical_path))
        (out/'optical-manifest.json').write_bytes(Path(optical_path).read_bytes())
        (out/'optical-protocol.json').write_bytes(bound(optical_document['protocol'],Path(optical_path).parent).read_bytes())
    view=map_image(out,arrays,grid);svg,routes=evidence_svg(out,grid,view,files,proof,arrays['aoi-mask.tif'])
    (out/'report.html').write_text(html_report(facts,svg,routes))
    (out/'report.json').write_text(json.dumps(facts,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    (out/'map.svg').write_text(svg)
    (out/'access-proof.json').write_bytes(files['access-proof.json'].read_bytes())
    if pdf_output:
        pdf_report(Path(pdf_output).resolve(),facts,out/'map.png',sources[0],sources[1])
    for item in sources:
        bound(item,out)
    binding={'schema':'akash-fresh-report-binding-v1','status':'completed','created_at':datetime.now(timezone.utc).isoformat(),
        'case':water['case'],'inputs':sources,'code':[record(Path(__file__))],
        'outputs':{p.name:record(p) for p in sorted(out.iterdir()) if p.is_file()},
        'map_rendering':view,'pdf':record(pdf_output) if pdf_output else None,
        'pdf_visual_review':'pending' if pdf_output else 'not generated',
        'model_invoked_by_reporter':False,'reference_labels_loaded':False,'gpu_used':False,'wall_seconds':time.perf_counter()-start,
        'fresh_unseen_event_validated':False,'full_track_b_requirements_satisfied':False}
    (out/'source-and-output-manifest.json').write_text(json.dumps(binding,indent=2)+'\n')
    print(json.dumps({'status':'completed','report':record(out/'report.html'),'counts':facts['counts'],
                     'scenario_counts':facts['access_scenario']['status_counts'],'wall_seconds':binding['wall_seconds']},indent=2))
    return binding


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--water-manifest',type=Path,required=True)
    p.add_argument('--exposure-manifest',type=Path,required=True)
    p.add_argument('--optical-manifest',type=Path,help='Optional separately bound partial optical candidate evidence')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--pdf-output',type=Path,help='Optional one-page PDF; requires reportlab and pypdf. Preserve existing files.')
    a=p.parse_args();run(a.water_manifest,a.exposure_manifest,a.out,a.pdf_output,a.optical_manifest)


if __name__=='__main__':main()
