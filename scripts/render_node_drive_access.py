"""Render the separate historical node/drive supplement without a building claim.

The complete saved network is retained for analysis. The view draws candidate
segments and selected source-edge witnesses only; it never loads the 1 GB road
presentation export or turns a hypothetical path into safe passage.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path

import numpy as np
from pyproj import Transformer
from rasterio.features import shapes
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from acquire_node_drive_historic_osm import read_core
from akash_flood.access import build_road_graph
from akash_flood.osm import named_points
from build_fresh_report import bound, map_image, read_json, read_water, record
from prove_access_bounds import verify_certificates


def verify_targets(graph, settlements, destinations, proof):
    """Separate vectorized snap and sparse shortest-path input checks.

    This checks the saved graph target policy, never physical entrance or
    passability. Parallel edges use minimum length rather than sparse sums.
    """
    nodes = sorted(graph.nodes, key=str)
    index = {node: i for i, node in enumerate(nodes)}
    lon = np.radians([graph.nodes[node]['x'] for node in nodes])
    lat = np.radians([graph.nodes[node]['y'] for node in nodes])
    def snap(point):
        if not nodes: return None, math.inf
        plon, plat = np.radians([point['lon'], point['lat']])
        h = np.sin((lat - plat) / 2) ** 2 + np.cos(plat) * np.cos(lat) * np.sin((lon - plon) / 2) ** 2
        distances = 2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        i = int(np.argmin(distances)); distance = float(distances[i])
        return (nodes[i] if distance <= 500 else None), distance
    def close(left, right):
        if not math.isfinite(float(left)) or not math.isclose(left, right, abs_tol=1e-6, rel_tol=1e-12):
            raise ValueError('Saved target/association distance differs from the source graph')
    if proof.get('max_snap_m') != 500:
        raise ValueError('Require the declared 500 metre association policy')
    target_snaps = {p['id']: snap(p) for p in destinations}
    lengths = {}
    groups = {}
    for u, v, data in graph.edges(data=True):
        pair = (index[u], index[v])
        lengths[pair] = min(lengths.get(pair, math.inf), data['length_m'])
        group = f'way-{data["way_id"]}-nodes-{min(u, v)}-{max(u, v)}'
        groups.setdefault(group, set()).add(data['edge_id'])
    for group, edges in proof['uncertain_links'].items():
        if group not in groups or len(edges) != len(set(edges)) or set(edges) != groups[group]:
            raise ValueError('Uncertain physical group differs from its source edge identities')
    pairs = list(lengths)
    matrix = coo_matrix(([lengths[p] for p in pairs],
                         ([p[0] for p in pairs], [p[1] for p in pairs])), shape=(len(nodes), len(nodes))).tocsr()
    points = {p['id']: p for p in settlements}
    for row in proof['settlements']:
        point = points[row['id']]
        if row['name'] != point['name']:
            raise ValueError('Settlement name differs from the source POI')
        node, distance = snap(point)
        if row['snap_node_id'] != node:
            raise ValueError('Settlement node differs from the source association')
        if math.isfinite(distance): close(row['snap_distance_m'], distance)
        choices = []
        if node is not None:
            road_lengths = dijkstra(matrix, directed=True, indices=index[node])
            for key, (target_node, target_distance) in target_snaps.items():
                if target_node is not None and math.isfinite(road_lengths[index[target_node]]):
                    road = float(road_lengths[index[target_node]])
                    choices.append((road + distance + target_distance, str(key), key, target_node, road, target_distance))
        selected = row['nearest_baseline_destination']
        if not choices:
            if selected is not None: raise ValueError('Saved target has no baseline source path')
            continue
        score, _, key, target_node, road, target_distance = min(choices)
        if (selected is None or selected['destination_id'] != key
                or selected['destination_snap_node_id'] != target_node):
            raise ValueError('Saved target differs from the fixed nearest source destination')
        close(selected['baseline_road_distance_m'], road)
        close(selected['baseline_distance_including_association_m'], score)
        close(selected['destination_snap_distance_m'], target_distance)
        for name in ('path_witness', 'optimistic_path_witness'):
            witness = row[name]
            if witness and (witness['destination_id'] != key or witness['destination_node_id'] != target_node):
                raise ValueError('Witness destination differs from the actual source POI')
            if witness: close(witness['destination_snap_distance_m'], target_distance)


def read_supplement(water_path, supplement_path):
    water, arrays, grid = read_water(water_path)
    binding = read_json(supplement_path)
    if (binding.get('schema') != 'akash-node-drive-access-binding-v1'
            or binding.get('status') != 'completed'
            or binding.get('case') != water['case']
            or binding.get('building_inventory_available') is not False
            or any(binding.get(k) is not False for k in ('reference_labels_loaded', 'model_inference_run'))
            or type(binding.get('network_requests')) is not int or binding['network_requests'] != 0):
        raise ValueError('Require a completed reference-free node/drive supplement')
    for item in binding['inputs'] + binding['code']:
        bound(item, supplement_path.parent)
    files = {name: bound(item, supplement_path.parent) for name, item in binding['outputs'].items()}
    summary, proof = read_json(files['summary.json']), read_json(files['access-proof.json'])
    source = record(water_path)
    if (summary.get('schema') != 'akash-node-drive-access-v1'
            or summary.get('status') != 'completed' or summary['case'] != water['case']
            or summary['temporal_counts'] != water['counts'] or summary['water_source'] != source
            or proof['case'] != water['case'] or proof['temporal_counts'] != water['counts']
            or proof['water_source'] != source or summary.get('buildings') is not None
            or summary['category_availability'].get('buildings') is not False
            or 'buildings.geojson' in files or proof.get('building_inventory_available') is not False):
        raise ValueError('Water case/category binding is inconsistent')
    scenario = summary['access_scenario']
    if (scenario.get('scenario_is_hypothetical') is not True
            or scenario.get('confirmed_closures') != 0 or proof['certain_closed_edge_ids']
            or proof.get('observed_hazard_or_closures_used') is not False
            or proof.get('destination_switching_after_closure') is not False):
        raise ValueError('Only the saved hypothetical fixed-target scenario is supported')
    statuses = dict(Counter(r['status'] for r in proof['settlements']))
    if (statuses != scenario['status_counts'] or statuses != proof['status_counts']
            or len(proof['settlements']) != scenario['n_settlements']
            or len(proof['uncertain_links']) != scenario['n_uncertain_physical_segments']
            or sum(map(len, proof['uncertain_links'].values())) != scenario['n_uncertain_directed_edges']):
        raise ValueError('Recounted scenario totals disagree')
    historic = summary['historical_osm']
    osm, checked_historic, _ = read_core(bound(historic['source'], supplement_path.parent),
                                      bound(historic['provenance'], supplement_path.parent), water['case'])
    if historic != checked_historic:
        raise ValueError('Historical source identity differs')
    settlements, destinations = named_points(osm)
    if (len(destinations) != scenario['n_destinations']
            or {p['id'] for p in settlements} != {r['id'] for r in proof['settlements']}):
        raise ValueError('Saved POIs differ from the bound source')
    graph = build_road_graph(osm)
    verify_targets(graph, settlements, destinations, proof)
    checks = verify_certificates(graph, proof)
    if checks != scenario['certificate_checks'] or checks != proof['certificate_checks']:
        raise ValueError('Source-edge path/cut checks disagree')
    for row in proof['settlements']:
        selection = row['nearest_baseline_destination']
        for key in ('path_witness', 'optimistic_path_witness'):
            witness = row[key]
            if witness and (selection is None or witness['destination_id'] != selection['destination_id']
                            or witness['destination_node_id'] != selection['destination_snap_node_id']):
                raise ValueError('Path changed away from the fixed selected target')
        if row['status'] == 'robust_connected' and (not row['path_witness'] or row['path_witness']['scenario'] != 'pessimistic'):
            raise ValueError('Robust connectivity requires a pessimistic source path')
    return water, arrays, grid, summary, proof, files, graph, settlements, destinations


def scene(out, arrays, grid, proof, graph, settlements):
    view = map_image(out, arrays, grid)
    tx = Transformer.from_crs(4326, grid[0], always_xy=True)
    inverse = ~grid[1]
    def xy(lon, lat):
        col, row = inverse * tx.transform(lon, lat)
        return [round(col - view['col_start'], 2), round(row - view['row_start'], 2)]
    required = {edge for edges in proof['uncertain_links'].values() for edge in edges}
    for row in proof['settlements']:
        for key in ('path_witness', 'optimistic_path_witness'):
            if row[key]: required.update(row[key]['path_edge_ids'])
    geometry = {data['edge_id']: [xy(*p) for p in data['geometry'].coords]
                for _, _, data in graph.edges(data=True) if data['edge_id'] in required}
    if set(geometry) != required:
        raise ValueError('A saved source edge lacks display geometry')
    uncertain = [geometry[edges[0]] for edges in proof['uncertain_links'].values()]
    def line(points):
        coords = ' '.join(f'{x},{y}' for x, y in points)
        return f'<polyline points="{coords}"/>'
    aoi = arrays['aoi-mask.tif'][view['row_start']:view['row_start'] + view['height'],
                                 view['col_start']:view['col_start'] + view['width']]
    clips = []
    for g, value in shapes(aoi, mask=aoi == 1):
        if value == 1:
            path = ' '.join('M' + ' L'.join(f'{x},{y}' for x, y in ring) + ' Z' for ring in g['coordinates'])
            clips.append(f'<path d="{path}" clip-rule="evenodd"/>')
    png = base64.b64encode((out / 'map.png').read_bytes()).decode()
    svg = (f'<svg id="map" viewBox="0 0 {view["width"]} {view["height"]}" xmlns="http://www.w3.org/2000/svg" '
           'role="img" aria-label="Native water cells, candidate roads and selected hypothetical path">'
           f'<defs><clipPath id="aoi">{"".join(clips)}</clipPath></defs>'
           f'<image href="data:image/png;base64,{png}" width="{view["width"]}" height="{view["height"]}" style="image-rendering:pixelated"/>'
           f'<g clip-path="url(#aoi)" fill="none" stroke="#a33b29" stroke-width="3">{"".join(map(line, uncertain))}</g>'
           '<g id="path" clip-path="url(#aoi)" fill="none" stroke="#2866c5" stroke-width="4"></g></svg>')
    points = {p['id']: p for p in settlements}
    payload = []
    for row in proof['settlements']:
        witness = row['path_witness'] or row['optimistic_path_witness']
        payload.append({'id': row['id'], 'name': row['name'], 'status': row['status'],
                        'selection': row['nearest_baseline_destination'], 'point': points[row['id']],
                        'witness_distance_m': witness['distance_m'] if witness else None,
                        'paths': [geometry[e] for e in witness['path_edge_ids']] if witness else []})
    view.update(displayed_uncertain_physical_segments=len(uncertain), full_baseline_network_drawn=False,
                selected_witness_geometry_derived_from_bound_osm=True, aoi_cell_clip=True)
    return svg, payload, view


def inside(point, bbox):
    return bbox[0] <= point['lon'] <= bbox[2] and bbox[1] <= point['lat'] <= bbox[3]


def run(water_path, supplement_path, out):
    water_path, supplement_path, out = map(lambda p: Path(p).resolve(), (water_path, supplement_path, out))
    if out.exists() or out.is_symlink():
        raise FileExistsError('Preserve previous views; choose a fresh output directory')
    water, arrays, grid, summary, proof, files, graph, settlements, destinations = read_supplement(water_path, supplement_path)
    out.mkdir(parents=True)
    svg, routes, view = scene(out, arrays, grid, proof, graph, settlements)
    counts, scenario = water['counts'], summary['access_scenario']
    snapshot_date = water['case']['osm_snapshot'].split('T')[0]
    after_date = datetime.fromisoformat(water['source_dates']['after'].replace('Z', '+00:00')).date()
    event_date = datetime.fromisoformat(water['case']['event_date']).date()
    observation_lag_days = (after_date - event_date).days
    scope = {'settlements_in_aoi': sum(inside(p, water['case']['bbox']) for p in settlements),
             'settlements_in_context': sum(inside(p, water['case']['context_bbox']) for p in settlements),
             'incidental_settlements_outside_context': [p for p in settlements if not inside(p, water['case']['context_bbox'])],
             'all_source_settlements': len(settlements), 'all_source_destinations': len(destinations)}
    facts = {'schema': 'akash-node-drive-report-facts-v1', 'case': water['case'], 'counts': counts,
             'roads': summary['roads'], 'bridges': summary['bridges'], 'buildings': None,
             'category_availability': summary['category_availability'], 'access_scenario': scenario,
             'point_scope': scope, 'source_dates': water['source_dates'], 'supplement_cpu_seconds': summary['wall_seconds'],
             'after_observation_calendar_days_from_event': observation_lag_days,
             'local_accuracy_established': False, 'confirmed_closures': 0,
             'all_other_baseline_edges_assumed_available': True}
    destination_names = {p['id']: p['name'] for p in destinations}
    options = ''.join(f'<option value="{escape(r["id"], quote=True)}">{escape(r["name"])}</option>' for r in routes)
    data = json.dumps({'routes': routes, 'destinations': destination_names}, ensure_ascii=False, separators=(',', ':'))
    data = data.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    legend = ''.join(f'<span><i style="background:{color}"></i>{label}</span>' for color, label in
                     [('#C1C8CF', 'Unknown'), ('#EDF2EE', 'Assessed, no candidate'), ('#99BBB8', 'Water both dates'),
                      ('#087E82', 'New-water candidate'), ('#D87C43', 'Lost-water candidate'),
                      ('#a33b29', 'Candidate-selected road segment'), ('#2866c5', 'Selected hypothetical path')])
    limitations = ''.join(f'<li>{escape(x)}</li>' for x in summary['limitations'])
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Akash | Kathmandu conditional access</title><style>*{box-sizing:border-box}body{margin:0;background:#eff3f1;color:#17353e;font:16px/1.55 system-ui}main{max-width:1240px;margin:auto;padding:32px}h1{font-size:36px;line-height:1.2}h2{font-size:22px}.kicker{color:#087e82;font-size:13px;letter-spacing:.12em;font-weight:700}.notice{background:#fff3d9;padding:16px 20px;border-radius:8px}.grid{display:grid;grid-template-columns:1.05fr 1fr;gap:22px;margin:24px 0}.panel{background:#fff;border:1px solid #d4dfdc;border-radius:10px;padding:24px}.metric{font-size:36px;font-weight:700;line-height:1.2}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:18px}.metrics p{margin:7px 0}svg#map{width:100%;height:580px}.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px}.legend span{display:flex;align-items:center;gap:6px}i{width:12px;height:12px;display:inline-block}select{width:100%;padding:12px;font:inherit;background:#f4f8f6;border:1px solid #a8bdb5;border-radius:6px}a{color:#096c75}code{overflow-wrap:anywhere;font-size:12px}small{font-size:13px}.links{display:flex;flex-wrap:wrap;gap:18px}.scope{color:#47656d;font-size:14px}@media(max-width:850px){main{padding:18px}.grid{grid-template-columns:1fr}h1{font-size:29px}}</style><main>
<div class="kicker">AKASH / SECOND CASE / HISTORICAL ROAD SCENARIOS</div><h1>Kathmandu: inspect the path behind the conclusion</h1>
<p>Event date ''' + escape(water['case']['event_date']) + ' · Historical OSM ' + escape(snapshot_date) + ''' · Native 10 m water grid</p>
<p class="notice">These are hypothetical graph scenarios from provisional water classifications. All other baseline edges, including unknown and outside-AOI edges, are assumed available. A surviving path does not establish safe passage, physical access or an operating hospital.</p>
<div class="grid"><section class="panel"><h2>Source cells and selected witness</h2>''' + svg + '<div class="legend">' + legend + '''</div><p class="scope">The map draws source centerlines and the selected saved path, clipped to exact AOI cells. Line widths aid display; candidate association uses a separate 20 m corridor with flat ends. The full baseline road network is omitted from this view and retained in the analysis. Browser scaling does not change source cells.</p></section>
<section class="panel"><h2>What survives the declared uncertainty?</h2><div class="metrics"><div><div class="metric">''' + f'{scenario["n_uncertain_physical_segments"]:,}' + '''</div><p>road segments intersect new-water candidates</p></div><div><div class="metric">''' + str(scenario['status_counts'].get('robust_connected', 0)) + '''</div><p>settlements retain a path with all candidate segments closed</p></div></div>
<p>''' + str(scenario['n_settlements']) + ' source settlements · ' + str(scenario['n_destinations']) + ''' node-tagged town/hospital targets · 0 confirmed closures</p><label for="settlement">Inspect a settlement and its fixed target</label><select id="settlement">''' + options + '''</select><div id="detail" aria-live="polite"></div>
<p class="scope">Selected destination: nearest baseline-reachable mapped town or hospital by directed road length plus fixed point-association distances. The saved witness uses shortest hop count; its length is descriptive, not the shortest distance or travel time.</p>
<h2>Category and point coverage</h2><p><strong>Buildings: unavailable / not requested.</strong> Polygon-only POIs are excluded; neither source completeness nor facility operation is validated.</p><p>''' + f'{scope["settlements_in_aoi"]} settlements lie in the AOI, {scope["settlements_in_context"]} in the extraction context. {len(scope["incidental_settlements_outside_context"])} additional settlement arrived as a way-node dependency outside the context; the backend retains it.' + '''</p></section></div>
<section class="panel"><div class="metrics"><div><div class="metric">''' + f'{counts["assessed_pixels"] / counts["aoi_pixels"] * 100:.2f}%' + '''</div><p>assessed coverage, not accuracy</p><p>''' + f'{counts["unknown_pixels"]:,} unknown cells · {counts["new_water_candidate_pixels"]:,} new-water candidates · {counts["lost_water_candidate_pixels"]:,} lost-water candidates' + '''</p></div><div><div class="metric">''' + f'{summary["wall_seconds"]:.2f} s' + '''</div><p>later CPU exposure/access supplement</p><p>0 network requests and 0 new inference in this mapping stage. Its time excludes original water processing, historical acquisition and this rendering.</p></div></div>
<p>Before: ''' + escape(water['source_dates']['before']) + ' · After: ' + escape(water['source_dates']['after']) + f'''. The after observation is {observation_lag_days} calendar days after the chosen event date. No local flood/debris accuracy has been established.</p><ul>''' + limitations + '''</ul>
<p class="links"><a href="report.json" download>Computed facts</a><a href="access-proof.json" download>All saved path/cut evidence</a><a href="supplement-binding.json" download>Original source bindings</a><a href="source-and-output-manifest.json" download>View bindings</a><a href="map.png" download>Native cell image</a></p>
<small>Contains modified Copernicus Sentinel data 2024. Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. © OpenStreetMap contributors. ODbL. Training: Sen1Floods11, Bonafilia, Tellman, Anderson and Issenberg, CVPR Workshops 2020. AI-assisted implementation. Original absolute file paths in copied receipts describe acquisition provenance; this view uses its local saved display files.</small></section></main>
<script id="evidence" type="application/json">''' + data + '''</script><script>
const evidence=JSON.parse(document.getElementById('evidence').textContent);
const statuses={robust_connected:'Path exists in both modeled extremes',ambiguous:'Depends on candidate-road states',baseline_disconnected:'No baseline graph path',robust_newly_disconnected:'No path even in optimistic scenario',unmapped:'Point association unavailable'};
function show(){const row=evidence.routes.find(r=>r.id===document.getElementById('settlement').value);const target=row.selection;const detail=document.getElementById('detail');detail.replaceChildren();
for(const text of [row.name+' · '+row.id, 'Status: '+statuses[row.status], target?'Fixed target: '+(evidence.destinations[target.destination_id]||target.destination_id)+' · '+target.destination_id:'Fixed target unavailable', row.witness_distance_m===null?'No path witness': 'Saved graph-path length: '+(row.witness_distance_m/1000).toFixed(2)+' km; point associations are separate', 'Physical passability and facility operation remain unknown.']){const p=document.createElement('p');p.textContent=text;detail.appendChild(p)}
const layer=document.getElementById('path');layer.replaceChildren();for(const points of row.paths){const line=document.createElementNS('http://www.w3.org/2000/svg','polyline');line.setAttribute('points',points.map(p=>p.join(',')).join(' '));layer.appendChild(line)}}
document.getElementById('settlement').addEventListener('change',show);show();</script></html>'''
    (out / 'index.html').write_text(html)
    (out / 'report.json').write_text(json.dumps(facts, indent=2, ensure_ascii=False) + '\n')
    (out / 'access-proof.json').write_bytes(files['access-proof.json'].read_bytes())
    (out / 'supplement-binding.json').write_bytes(supplement_path.read_bytes())
    binding = {'schema': 'akash-node-drive-access-render-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
               'inputs': [record(water_path), record(supplement_path)], 'rendering': view,
               'code': [record(Path(__file__)), record(Path(__file__).with_name('build_fresh_report.py')),
                        record(Path(__file__).with_name('prove_access_bounds.py'))],
               'outputs': {p.name: record(p) for p in sorted(out.iterdir()) if p.is_file()},
               'source_edge_certificates_rechecked': summary['access_scenario']['certificate_checks'],
               'model_invoked': False, 'network_requests': 0, 'reference_labels_loaded': False,
               'building_inventory_available': False, 'local_accuracy_established': False}
    (out / 'source-and-output-manifest.json').write_text(json.dumps(binding, indent=2) + '\n')
    return {'status': 'completed', 'html': record(out / 'index.html'), 'point_scope': scope,
            'source_edge_certificates_rechecked': binding['source_edge_certificates_rechecked']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('water-manifest', 'supplement-manifest', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.water_manifest, args.supplement_manifest, args.out), indent=2))
