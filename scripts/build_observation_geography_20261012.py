"""Build a new local geographic viewer from immutable saved point/road evidence.

Original code MIT; source geography retains OSM/upstream terms. No network,
model, graph solve, labels or old source/output modification is performed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / 'outputs/observation-access-viewer-2026-10-11'
PROTOCOL = ROOT / 'research/observation-access-2026-10-11/protocol.frozen.json'
POINT_FILES = {
    'rasuwa': 'outputs/fresh-one-command-replay/exposure',
    'kathmandu': 'outputs/kathmandu-node-drive-access-2026-10-08',
}
EXPECTED = {
    'rasuwa': {'ambiguous': 32, 'unmapped': 20, 'baseline_disconnected': 3},
    'kathmandu': {'robust_connected': 28, 'ambiguous': 41},
}

def read(path):
    return json.loads(Path(path).read_text())

def record(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest}

def verify(bound):
    path = Path(bound['path'])
    actual = record(path if path.is_absolute() else ROOT / path)
    if any(actual[k] != bound[k] for k in ('bytes', 'sha256')):
        raise ValueError('Changed frozen source: ' + bound['path'])
    return actual

def point(value):
    if (not isinstance(value, list) or len(value) != 2 or
        not all(isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x) for x in value) or
        not -180 <= value[0] <= 180 or not -90 <= value[1] <= 90):
        raise ValueError('Invalid source longitude/latitude')
    return value

def extent(points):
    return [min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points)]

def counts(rows):
    return dict(Counter(r['status'] for r in rows))

def feature_index(path):
    d = read(path)
    result = {}
    for i, f in enumerate(d['features']):
        if f['id'] in result or f['geometry']['type'] != 'Point':
            raise ValueError('Duplicate/nonpoint historical feature')
        p = point(f['geometry']['coordinates'])
        if f['properties']['id'] != f['id'] or p != [f['properties']['lon'], f['properties']['lat']]:
            raise ValueError('Point ID/coordinate serialization mismatch')
        result[f['id']] = (f, i)
    return result

def verify_route(c, compact, support, witness):
    if support is None:
        if compact is not None:
            raise ValueError('Unexpected compact route')
        return 0
    if compact is None or compact['edge_ids'] != [s['edge_id'] for s in support['segments']]:
        raise ValueError('Saved route edge order mismatch')
    if witness['path_edge_ids'] != compact['edge_ids']:
        raise ValueError('Witness differs from support')
    pairs = [('graph_distance_m', 'graph_distance_m'),
             ('projected_centerline_total_m', 'projected_centerline_total_m'),
             ('lengths_m', 'lengths_m'), ('fractions', 'fractions_of_projected_centerline')]
    if any(compact[a] != support[b] for a, b in pairs):
        raise ValueError('Compact support differs from saved support')
    for j, s in enumerate(support['segments']):
        canonical = c['segments'][s['edge_id']]
        for k in ('coordinates', 'centerline_length_m', 'assessed_length_m', 'unknown_within_aoi_length_m', 'outside_aoi_length_m'):
            if canonical[k] != s[k]:
                raise ValueError('Compact segment differs from source geometry/measure')
        for p in s['coordinates']:
            point(p)
        if s['from_node_id'] != witness['path_node_ids'][j] or s['to_node_id'] != witness['path_node_ids'][j + 1]:
            raise ValueError('Route node chain mismatch')
        if j and support['segments'][j-1]['coordinates'][-1] != s['coordinates'][0]:
            raise ValueError('Saved route geometry discontinuity')
    if not compact['edge_ids']:
        if compact['projected_centerline_total_m'] != 0 or any(v is not None for v in compact['fractions'].values()):
            raise ValueError('Empty route must retain null support')
    return 1

def build(out):
    started = time.monotonic()
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError('Use a new output directory; do not overwrite a prior attempt')
    protocol = read(PROTOCOL)
    frozen_before = [verify(v) for v in protocol['records']]
    old_receipt = read(VIEWER / 'verification.json')
    old_data_record = record(VIEWER / 'data.json')
    if any(old_data_record[k] != old_receipt['outputs']['data.json'][k] for k in ('bytes', 'sha256')):
        raise ValueError('Old compact viewer payload differs from its receipt')
    packaged = ROOT / 'output/submission-hardening-r2-2026-10-11/05-viewers/observation-access/data.json'
    if record(packaged)['sha256'] != old_data_record['sha256']:
        raise ValueError('Packaged compact viewer is different')
    data = read(VIEWER / 'data.json')
    context = {'schema': 'akash-geographic-context-v1', 'cases': {},
        'coordinate_system': 'Supplied longitude/latitude degrees, [longitude,latitude]',
        'display_projection': 'Local longitude scaled by cosine of map midpoint latitude, north up; display only',
        'invented_geography': False, 'inferred_point_to_road_connectors': False,
        'route_vertex_simplification': False, 'source_data_license': 'OSM/upstream terms; no MIT grant over data'}
    sources = [record(PROTOCOL), old_data_record, record(VIEWER / 'verification.json'), record(packaged),
        record(ROOT / 'research/observation-access-2026-10-11/independent-verification.json'), record(ROOT / 'LICENSE')]
    checks = {'frozen_records': len(frozen_before), 'source_point_rows': 0, 'fixed_target_points': 0,
        'route_supports_checked': 0, 'source_geometry_vertices_checked': 0, 'empty_witness_rows': [], 'cases': {}}
    copied_receipts = {'protocol.frozen.json': PROTOCOL,
        'independent-verification.json': ROOT / 'research/observation-access-2026-10-11/independent-verification.json'}
    for key, c in data['cases'].items():
        base = ROOT / f'outputs/observation-access-2026-10-11/{key}-v1'
        receipt = read(base / 'receipt.json')
        sources.append(record(base / 'receipt.json'))
        sources.extend(verify(v) for v in receipt['outputs'])
        expanded = read(base / 'expanded-scenario.json')
        originals = read(base / 'original-witness-support.json')
        raw_rows = {r['id']: r for r in expanded['settlements']}
        raw_originals = {r['id']: r for r in originals['witnesses']}
        settlement_file = ROOT / POINT_FILES[key] / 'settlements.geojson'
        target_file = ROOT / POINT_FILES[key] / 'destinations.geojson'
        source_features, target_features = feature_index(settlement_file), feature_index(target_file)
        sources.extend([record(settlement_file), record(target_file)])
        if counts(c['settlements']) != EXPECTED[key] or counts(expanded['settlements']) != EXPECTED[key]:
            raise ValueError('Preserved scenario counts differ')
        if c['case'] != expanded['case'] or set(raw_rows) != {r['id'] for r in c['settlements']}:
            raise ValueError('Case or row set differs')
        node_coordinates = {}
        node_origins = {}
        for i, r in enumerate(expanded['settlements']):
            for j, seg in enumerate((r.get('path_support') or {}).get('segments', [])):
                for label, coordinate in [('from', seg['coordinates'][0]), ('to', seg['coordinates'][-1])]:
                    node_id = seg[f'{label}_node_id']
                    if node_id in node_coordinates and node_coordinates[node_id] != coordinate:
                        raise ValueError('One saved node has conflicting coordinates')
                    node_coordinates[node_id] = coordinate
                    node_origins.setdefault(node_id, {'source_path': str(base / 'expanded-scenario.json'),
                        'source_pointer': f'/settlements/{i}/path_support/segments/{j}/coordinates/{0 if label == "from" else len(seg["coordinates"])-1}'})
        targets, endpoints, point_origins = {}, {}, {}
        for r in c['settlements']:
            raw = raw_rows[r['id']]
            feature, i = source_features[r['id']]
            p = point(r['point_coordinates'])
            if p != raw['point_coordinates'] or p != feature['geometry']['coordinates']:
                raise ValueError('Historical source point mismatch')
            for k in ('name', 'status', 'previous_status', 'inside_water_aoi', 'inside_extraction_context'):
                if r[k] != raw[k]:
                    raise ValueError('Compact source status/scope differs')
            for b, flag in [(c['case']['bbox'], 'inside_water_aoi'), (c['case']['context_bbox'], 'inside_extraction_context')]:
                if (b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]) != r[flag]:
                    raise ValueError('Source scope does not match supplied point')
            if r['destination'] != raw['nearest_baseline_destination']:
                raise ValueError('Fixed destination mismatch')
            point_origins[r['id']] = {'source': record(settlement_file), 'source_pointer': f'/features/{i}/geometry/coordinates'}
            if raw.get('path_support') is not None:
                checks['route_supports_checked'] += verify_route(c, r['expanded_route'], raw['path_support'], raw['path_witness'])
                if r['expanded_witness_scenario'] != raw['path_witness']['scenario']:
                    raise ValueError('Expanded witness extreme label differs from source')
                if not r['expanded_route']['edge_ids']:
                    checks['empty_witness_rows'].append({'case': key, 'id': r['id'], 'fractions': r['expanded_route']['fractions']})
                for endpoint, node_id in [('source', raw['path_witness']['source_node_id']), ('target', raw['path_witness']['destination_node_id'])]:
                    if node_id not in node_coordinates:
                        raise ValueError('No supplied coordinate for witness endpoint')
                    endpoints.setdefault(r['id'], {})[endpoint] = {'node_id': node_id,
                        'coordinates': node_coordinates[node_id], 'origin': node_origins[node_id],
                        'role': 'Snapped graph node, not actual settlement/facility entrance'}
            elif r['expanded_route'] is not None:
                raise ValueError('Unexpected route on unsupported row')
            original = raw_originals.get(r['id'])
            if original:
                # Original support edge sequence and source/target node chain are supplied directly.
                witness = {'path_edge_ids': [s['edge_id'] for s in original['support']['segments']],
                    'path_node_ids': ([original['support']['segments'][0]['from_node_id']] +
                        [s['to_node_id'] for s in original['support']['segments']]) if original['support']['segments'] else [original['source_node_id']]}
                checks['route_supports_checked'] += verify_route(c, r['original_route'], original['support'], witness)
            elif r['original_route'] is not None:
                raise ValueError('Unexpected original route')
            if r['destination']:
                tid = r['destination']['destination_id']
                target, j = target_features[tid]
                targets[tid] = {'id': tid, 'name': target['properties']['name'], 'kind': target['properties']['kind'],
                    'coordinates': point(target['geometry']['coordinates']), 'source': record(target_file),
                    'source_pointer': f'/features/{j}/geometry/coordinates', 'physical_entrance_or_operation_validated': False}
        points = [r['point_coordinates'] for r in c['settlements']]
        route_points = [p for s in c['segments'].values() for p in s['coordinates']]
        checks['source_geometry_vertices_checked'] += len(route_points)
        case_check = {'all_source_status_counts': counts(c['settlements']),
            'water_aoi_status_counts': counts([r for r in c['settlements'] if r['inside_water_aoi']]),
            'context_status_counts': counts([r for r in c['settlements'] if r['inside_extraction_context']]),
            'source_point_extent_lonlat': extent(points), 'saved_route_extent_lonlat': extent(route_points),
            'whole_case_extent_lonlat': extent(points + route_points + [t['coordinates'] for t in targets.values()]),
            'fixed_target_points': len(targets), 'displayed_segment_count': len(c['segments']),
            'mixed_support_segments': sum(sum(s[k] > 0 for k in ('assessed_length_m', 'unknown_within_aoi_length_m', 'outside_aoi_length_m')) > 1 for s in c['segments'].values())}
        checks['cases'][key] = case_check
        checks['source_point_rows'] += len(points)
        checks['fixed_target_points'] += len(targets)
        context['cases'][key] = {'targets': targets, 'graph_endpoints': endpoints,
            'source_point_origins': point_origins, 'extents': case_check}
        copied_receipts[f'{key}.json'] = base / 'receipt.json'
    out.mkdir(parents=True)
    (out / 'receipts').mkdir()
    shutil.copyfile(VIEWER / 'data.json', out / 'data.json')
    (out / 'context.json').write_text(json.dumps(context, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    new_code = {
        'index.html': ROOT / 'web/observation-geography-20261012.html',
        'core.mjs': ROOT / 'web/observation-geography-core-20261012.mjs',
        'ui.mjs': ROOT / 'web/observation-geography-ui-20261012.mjs',
    }
    for target, source in new_code.items():
        shutil.copyfile(source, out / target)
    for target, source in copied_receipts.items():
        shutil.copyfile(source, out / 'receipts' / target)
    shutil.copyfile(ROOT / 'LICENSE', out / 'CODE-LICENSE.txt')
    guide = '''# Saved geographic context viewer

Run from this folder with Python's standard-library server:

```sh
python3 -m http.server 8806 --bind 127.0.0.1 --directory .
```

Open http://127.0.0.1:8806/ . Serve over localhost; file:// cannot load/verify JSON modules. All HTML, modules, data and receipts are local. There is no basemap, external font, paid API, satellite acquisition, graph solve or model inference.

data.json is an exact byte-copy of the preserved compact observation-access payload. context.json adds only joins to hash-bound actual historical target/source points and snapped graph-node coordinates from saved segments. No inferred point-to-road connector is drawn. Actual source and target points are distinct from graph endpoints; operation and physical entrances remain unvalidated.

verification.json binds each original input/new output and lists exact point/route extents, status/scope counts, geometry assertions and the null empty-route case. The frozen protocol's 70 records are hashed before and after this build. Hashes establish preservation, not hazard truth. Browser and human usability/Nepali review are not performed by the builder.

Original/new viewer code is MIT on the explicit owner assertion; CODE-LICENSE.txt does not grant rights over data. Derived historical OpenStreetMap source/road data remains subject to ODbL and attribution. Contains modified Copernicus Sentinel data 2026 and 2024; Copernicus DEM/WorldDEM and Sen1Floods11 retain their upstream terms. No raw/reference geometry, model weights or vendor runtime is added here. The graph data is conditional evidence and not operational guidance.

This is a new standalone output. Earlier viewers, protocols, rasters, graph results, copilot checks and kits are unchanged. It improves geographic inspection only, not hazard accuracy or the arbitrary-area raw-data requirement.
'''
    (out / 'SOURCES-AND-RUN.md').write_text(guide)
    frozen_after = [verify(v) for v in protocol['records']]
    if frozen_after != frozen_before or record(VIEWER / 'data.json') != old_data_record:
        raise ValueError('Original frozen inputs changed during viewer build')
    unique_sources = {v['path']: v for v in sources}
    outputs = {str(p.relative_to(out)): {k: v for k, v in record(p).items() if k != 'path'}
        for p in sorted(out.rglob('*')) if p.is_file()}
    verification = {'schema': 'akash-geographic-viewer-verification-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'coordinate_status_route_bindings_and_frozen_preservation_passed; browser_review_pending',
        'sources': list(unique_sources.values()), 'new_code': [record(Path(__file__)), *[record(p) for p in new_code.values()]],
        'outputs': outputs, 'checks': checks, 'frozen_sources_before_after': frozen_before,
        'all_original_frozen_inputs_and_scenario_outputs_unchanged': True,
        'copied_data_byte_identical_to_preserved_viewer': True, 'route_geometry_invented': False,
        'point_to_road_connector_inferred': False, 'model_inference_performed': False, 'graph_analysis_performed': False,
        'reference_maps_loaded': False, 'network_requests': 0, 'gpu_used': False,
        'browser_review': 'pending root review', 'human_usability_or_nepali_review': False,
        'claims': 'Usability and exact saved geographic context only; no new flood accuracy, observed closures, physical access or arbitrary raw-data capability',
        'wall_seconds': time.monotonic() - started}
    (out / 'verification.json').write_text(json.dumps(verification, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'status': verification['status'], 'out': str(out), 'checks': checks,
        'original_frozen_records_unchanged': len(frozen_before), 'wall_seconds': verification['wall_seconds']}, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    build(p.parse_args().out)
