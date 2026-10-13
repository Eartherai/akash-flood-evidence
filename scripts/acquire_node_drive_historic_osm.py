"""One bounded historical driving-way/node-POI extract; buildings unavailable.

This new category-aware source is separate from the old all-category attempts.
Default is offline protocol only. No retry, endpoint sweep or current-map fallback.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import time

import requests
from akash_flood.catalog import validate_bbox, utc
from akash_flood.fresh_case import event_boundary, normalized_utc
from akash_flood.access import _DRIVE_HIGHWAYS
from acquire_historic_osm import record, unique_object, validate_elements, wall_deadline_handler

ENDPOINT = 'https://overpass-api.de/api/interpreter'
# Separate third campaign after preserved10MB/30MBfailures. Its selectors
# change to graph-eligible driving ways and nodePOIs; date/context unchanged.
MAX_BYTES = 30_000_000
MAX_OBJECTS = 500_000
DEADLINE_SECONDS = 120
CATEGORIES = {'roads': 'graph_drive_highway_classes_requested', 'places': 'node_tagged_requested',
              'hospitals': 'node_tagged_requested', 'buildings': 'not_requested_unavailable'}


def now(): return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def core_query(bbox, cutoff):
    west, south, east, north = validate_bbox(bbox)
    extent = '(' + ','.join(f'{v:.12g}' for v in (south, west, north, east)) + ')'
    highway_expression = '^(' + '|'.join(sorted(_DRIVE_HIGHWAYS)) + ')$'
    return (f'[out:json][timeout:45][maxsize:134217728][date:"{cutoff}"];('
            f'way["highway"~"{highway_expression}"]{extent};node["place"~"^(city|town|village|hamlet)$"]{extent};'
            f'node["amenity"="hospital"]{extent};);(._;>;);out body;')


def validate_dependencies(data):
    identities = {(item['type'], item['id']) for item in data['elements']}
    for item in data['elements']:
        references = ([('node', node) for node in item['nodes']] if item['type'] == 'way' else
                      [(member['type'], member['ref']) for member in item['members']] if item['type'] == 'relation' else [])
        if any(identity not in identities for identity in references):
            raise ValueError('Core source omits a referenced way node or relation member')
    return {'objects': len(identities), 'all_way_node_and_relation_dependencies_present': True}


def acquire(bbox, snapshot, event_date, out, *, execute=False, session=None):
    bbox = list(validate_bbox(bbox)); _, _, cutoff = event_boundary(event_date, snapshot)
    if (bbox[2]-bbox[0]) * (bbox[3]-bbox[1]) > 1 or cutoff.time() != datetime.min.time():
        raise ValueError('Use a <=1squaredegree extent and a UTC-midnight historical cutoff')
    cutoff_text = normalized_utc(cutoff); query = core_query(bbox, cutoff_text)
    out = Path(out).absolute()
    if out.exists() or out.is_symlink(): raise FileExistsError('Preserve earlier attempts; use a fresh core directory')
    out.mkdir(parents=True)
    protocol = {'schema': 'akash-node-drive-historic-map-protocol-v1', 'recorded_before_network_at': now(),
                'bbox': bbox, 'snapshot': cutoff_text, 'event_date': event_date, 'query': query,
                'endpoint': ENDPOINT, 'categories': CATEGORIES, 'maximum_requests': 1,
                'maximum_body_bytes': MAX_BYTES, 'maximum_objects': MAX_OBJECTS,
                'deadline_seconds': DEADLINE_SECONDS, 'stop_granularity_bytes': 65536,
                'code': [record(Path(__file__)), record(Path(__file__).with_name('acquire_historic_osm.py')),
                         record(Path(__file__).parents[1] / 'src/akash_flood/access.py')],
                'execute_requested': execute, 'tls_verified': True, 'redirects': False, 'automatic_retry': False,
                'scope': 'Server-executed historical body query; driving-way classes exactly match the existing graph, node-only POIs match its destination adapter. Polygon/relation POIs and non-driving highways are not requested; no OSM completeness or independent historical-authenticity claim',
                'reference_labels_loaded': False, 'model_inference_run': False}
    save(out / 'protocol.json', protocol)
    if not execute: return {'status': 'protocol_only', 'network_requests': 0, 'protocol': record(out / 'protocol.json')}
    client = session or requests.Session(); client.trust_env = False
    if getattr(client, 'auth', None) or getattr(client, 'cookies', None) or any(k.casefold() in
            ('authorization', 'proxy-authorization', 'cookie') for k in getattr(client, 'headers', {})):
        raise ValueError('Core source accepts no credentials or cookies')
    if session is None: client.mount('https://', requests.adapters.HTTPAdapter(max_retries=0))
    receipt = {'schema': 'akash-node-drive-historic-map-transport-v1', 'status': 'running', 'started_at': now(),
               'protocol': record(out / 'protocol.json'), 'attempted_requests': 0, 'received_body_bytes': 0,
               'response_status': None, 'endpoint': ENDPOINT, 'query': query,
               'accounting': 'Application-delivered bytes including failed partial transfers; internal unread socket/wire bytes unavailable'}
    save(out / 'network-receipt.json', receipt)
    partial = out / 'received-response.partial.bin'; start = time.monotonic(); response = None; previous = None
    try:
        if signal.getitimer(signal.ITIMER_REAL) != (0., 0.): raise RuntimeError('A free process wall timer is required')
        previous = signal.signal(signal.SIGALRM, wall_deadline_handler); signal.setitimer(signal.ITIMER_REAL, DEADLINE_SECONDS)
        receipt['attempted_requests'] = 1; save(out / 'network-receipt.json', receipt)
        response = client.get(ENDPOINT, params={'data': query}, stream=True, timeout=(15, 105),
                              allow_redirects=False, verify=True, headers={'Accept-Encoding': 'identity', 'User-Agent': 'AkashHistoricalCore/1.0'})
        receipt['response_status'] = response.status_code
        if response.status_code != 200 or response.headers.get('Content-Encoding', 'identity') not in ('', 'identity'):
            raise ValueError('Core source requires direct identity-encoded HTTP200')
        length = response.headers.get('Content-Length')
        if length and (not length.isdecimal() or int(length) > MAX_BYTES): raise ValueError('Excessive/invalid declared core source size')
        digest = hashlib.sha256()
        with partial.open('xb') as destination:
            for chunk in response.iter_content(65536):
                if not chunk: continue
                receipt['received_body_bytes'] += len(chunk); digest.update(chunk)
                if receipt['received_body_bytes'] > MAX_BYTES: raise ValueError('Core source body exceeds cap (64KiBstopgranularity)')
                destination.write(chunk)
        receipt['delivered_body_sha256'] = digest.hexdigest()
        if length and int(length) != receipt['received_body_bytes']: raise ValueError('Core source body is incomplete')
        data = json.loads(partial.read_bytes(), object_pairs_hook=unique_object)
        if (not isinstance(data, dict) or data.get('remark') or not isinstance(data.get('elements'), list)
                or not 0 < len(data['elements']) <= MAX_OBJECTS): raise ValueError('Core source empty, excessive or incomplete')
        validate_elements(data, cutoff); dependencies = validate_dependencies(data)
        if not any(item['type'] == 'way' and 'highway' in item.get('tags', {}) for item in data['elements']):
            raise ValueError('Core extract contains no mapped highway way')
        raw = out / 'response.json'; partial.rename(raw); save(out / 'osm.json', data)
        osm_record = record(out / 'osm.json')
        provenance = {'schema': 'akash-node-drive-historic-map-source-v1', 'snapshot': cutoff_text, 'event_date': event_date,
                      'bbox': bbox, 'query': query, 'endpoint': ENDPOINT, 'categories': CATEGORIES,
                      'element_count': len(data['elements']), 'sha256': osm_record['sha256'], 'raw_response': record(raw),
                      'protocol': record(out / 'protocol.json'), 'history_evidence': protocol['scope'],
                      'dependency_checks': dependencies, 'attribution': '© OpenStreetMap contributors; ODbL',
                      'transport_complete': True, 'map_completeness_validated': False, 'building_inventory_available': False,
                      'retrieved_at': now()}
        save(out / 'provenance.json', provenance)
        receipt.update(status='completed', osm=osm_record, provenance=record(out / 'provenance.json'),
                       raw_response=record(raw), dependency_checks=dependencies)
    except Exception as error:
        receipt.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        if partial.exists(): receipt['retained_partial_response'] = record(partial)
        raise
    finally:
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
        if response is not None: response.close()
        if session is None: client.close()
        receipt.update(finished_at=now(), elapsed_seconds=time.monotonic()-start)
        save(out / 'network-receipt.json', receipt)
    return receipt


def read_core(osm_path, provenance_path, case):
    from map_fresh_exposure import checked_record
    osm_path, provenance_path = Path(osm_path).resolve(), Path(provenance_path).resolve()
    provenance = json.loads(provenance_path.read_text(), object_pairs_hook=unique_object)
    if (provenance.get('schema') != 'akash-node-drive-historic-map-source-v1' or provenance.get('categories') != CATEGORIES
            or provenance.get('endpoint') != ENDPOINT or provenance.get('building_inventory_available') is not False
            or provenance.get('transport_complete') is not True or provenance.get('map_completeness_validated') is not False):
        raise ValueError('Require explicitly category-limited historical core evidence')
    _, _, cutoff = event_boundary(case['event_date'], case['osm_snapshot']); normalized = normalized_utc(cutoff)
    if (provenance['snapshot'] != normalized or provenance['event_date'] != case['event_date']
            or provenance['bbox'] != case['context_bbox'] or provenance['query'] != core_query(case['context_bbox'], normalized)
            or provenance['sha256'] != record(osm_path)['sha256']):
        raise ValueError('Core source cutoff, query, extent or content differs from the frozen case')
    protocol_path = checked_record(provenance['protocol'], provenance_path.parent, 'core protocol')
    protocol = json.loads(protocol_path.read_text(), object_pairs_hook=unique_object)
    controls = {'maximum_requests': 1, 'maximum_body_bytes': MAX_BYTES, 'maximum_objects': MAX_OBJECTS,
                'deadline_seconds': DEADLINE_SECONDS, 'stop_granularity_bytes': 65536,
                'execute_requested': True, 'tls_verified': True, 'redirects': False, 'automatic_retry': False}
    if any(protocol.get(key) != value or type(protocol.get(key)) is not type(value) for key, value in controls.items()):
        raise ValueError('Core source protocol resource/TLS/attempt controls differ')
    expected_code = [record(Path(__file__)), record(Path(__file__).with_name('acquire_historic_osm.py')),
                     record(Path(__file__).parents[1] / 'src/akash_flood/access.py')]
    if protocol.get('code') != expected_code:
        raise ValueError('Core source protocol code bindings differ or are incomplete')
    for key in ('schema', 'snapshot', 'event_date', 'bbox', 'query', 'endpoint', 'categories'):
        expected = 'akash-node-drive-historic-map-protocol-v1' if key == 'schema' else provenance[key]
        if protocol.get(key) != expected: raise ValueError('Core source protocol identity differs')
    receipt_path = protocol_path.with_name('network-receipt.json')
    receipt = json.loads(receipt_path.read_text(), object_pairs_hook=unique_object)
    if (receipt.get('schema') != 'akash-node-drive-historic-map-transport-v1' or receipt.get('status') != 'completed'
            or receipt.get('attempted_requests') != 1 or receipt.get('response_status') != 200
            or receipt.get('protocol') != record(protocol_path) or receipt.get('osm') != record(osm_path)
            or receipt.get('provenance') != record(provenance_path) or receipt.get('endpoint') != ENDPOINT
            or receipt.get('query') != provenance['query'] or not 0 < receipt.get('received_body_bytes', 0) <= protocol['maximum_body_bytes']):
        raise ValueError('Core source has no matching successful bounded transport')
    raw_path = checked_record(provenance['raw_response'], provenance_path.parent, 'core raw body')
    if (receipt['raw_response'] != record(raw_path) or receipt['received_body_bytes'] != raw_path.stat().st_size
            or receipt.get('delivered_body_sha256') != record(raw_path)['sha256']):
        raise ValueError('Core raw response differs from its actual transfer')
    data = json.loads(osm_path.read_text(), object_pairs_hook=unique_object)
    raw = json.loads(raw_path.read_text(), object_pairs_hook=unique_object)
    if (not isinstance(data, dict) or not isinstance(data.get('elements'), list)
            or not 0 < len(data['elements']) <= MAX_OBJECTS or data != raw or data.get('remark')
            or len(data['elements']) != provenance['element_count']
            or not any(item.get('type') == 'way' and 'highway' in item.get('tags', {}) for item in data['elements'])):
        raise ValueError('Core normalized/raw source JSON differs or is incomplete')
    validate_elements(data, cutoff); dependencies = validate_dependencies(data)
    if dependencies != provenance['dependency_checks'] or dependencies != receipt['dependency_checks']:
        raise ValueError('Core dependency checks differ')
    records = [record(path) for path in (osm_path, provenance_path, protocol_path, receipt_path, raw_path)]
    for item in protocol['code']:
        records.append(record(checked_record(item, protocol_path.parent, 'core source code')))
    return data, {'snapshot': normalized, 'source': records[0], 'provenance': records[1],
                 'query': provenance['query'], 'extraction_bbox': provenance['bbox'], 'categories': CATEGORIES,
                 'attribution': provenance['attribution'], 'history_evidence': provenance['history_evidence'],
                 'post_event_osm_used': False, 'building_inventory_available': False,
                 'independent_historical_server_authenticity_validated': False}, records


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--bbox', type=float, nargs=4, required=True)
    p.add_argument('--snapshot', required=True); p.add_argument('--event-date', required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--execute', action='store_true')
    a = p.parse_args(); print(json.dumps(acquire(a.bbox, a.snapshot, a.event_date, a.out, execute=a.execute), indent=2))
