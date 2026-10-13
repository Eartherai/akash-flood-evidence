"""Add category-limited roads/bridges/access to immutable water-only output.

Buildings remain unavailable (null), including incidental POI dependencies.
This is a separate supplement schema, never a complete legacy exposure report.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import map_fresh_exposure as mapper
from acquire_node_drive_historic_osm import read_core
from akash_flood.fresh_case import reject_source_path

ROOT = Path(__file__).resolve().parents[1]


def run(water_manifest, plan_path, osm_path, provenance_path, out):
    water_manifest, plan_path, osm_path, provenance_path, out = map(lambda p: Path(p).resolve(),
                                                               (water_manifest, plan_path, osm_path, provenance_path, out))
    if out.exists() or out.is_symlink(): raise FileExistsError('Preserve earlier supplements; use a fresh output directory')
    reject_source_path(plan_path); start = time.perf_counter(); plan_record = mapper.record(plan_path)
    plan = json.loads(plan_path.read_text())
    arrays, grid, counts, water, water_records = mapper.read_water(water_manifest, plan_path, plan)
    osm, historic, core_records = read_core(osm_path, provenance_path, plan['case'])
    out.mkdir(parents=True)
    mapper.write_json(out / 'source-scope.json', {'schema': 'akash-node-drive-access-source-scope-v1',
        'case': plan['case'], 'original_processing_plan': plan_record, 'water_source': mapper.record(water_manifest),
        'historical_osm': historic, 'building_exposure': None, 'building_category_available': False,
        'prior_water_walltime_includes_these_stages': False, 'created_at': datetime.now(timezone.utc).isoformat()})
    mapper.write_json(out / 'scenario-policy.json', {**mapper.SCENARIO,
        'recorded_before_feature_overlap_and_access_at': datetime.now(timezone.utc).isoformat(),
        'water_source': mapper.record(water_manifest), 'building_inventory_available': False})
    masks, polygons, regions = mapper.mask_regions(arrays, grid)
    # Incidental building tags are not an inventory and do not enter this
    # supplement's geometry mapper. Every graph/POI property remains intact.
    geometry_source = {'elements': [{**item, 'tags': {key: value for key, value in item.get('tags', {}).items()
                                                       if key != 'building'}} for item in osm['elements']]}
    collections, audits, graph = mapper.overlap_features(geometry_source, grid, regions)
    collections.pop('buildings'); audits['buildings'] = {'category_available': False, 'reason': 'Not requested in historical core query'}
    candidate_layers, candidate_summary = mapper.candidate_features(polygons, masks, grid); collections.update(candidate_layers)
    proof, settlements, destinations = mapper.access_scenario(graph, osm, collections['roads'])
    proof.update(case=plan['case'], temporal_counts=counts, water_source=mapper.record(water_manifest),
                 source_plan=plan_record, historical_osm=historic, building_inventory_available=False)
    mapper.write_json(out / 'access-proof.json', proof)
    collections['settlements'] = mapper.point_features(settlements, proof_rows=proof['settlements'], bbox=plan['case']['bbox'])
    collections['destinations'] = mapper.point_features(destinations, bbox=plan['case']['bbox'])
    for name, collection in collections.items(): mapper.write_json(out / (name + '.geojson'), collection)
    selected = Counter(row['nearest_baseline_destination']['destination_id'] for row in proof['settlements']
                       if row['nearest_baseline_destination'])
    scenario = {'schema': proof['schema'], 'scenario_is_hypothetical': True,
                'status_counts': proof['status_counts'], 'n_uncertain_physical_segments': len(proof['uncertain_links']),
                'n_uncertain_directed_edges': sum(map(len, proof['uncertain_links'].values())),
                'n_settlements': len(settlements), 'n_destinations': len(destinations), 'confirmed_closures': 0,
                'fixed_selected_destination_counts': dict(selected), 'destination_rule': proof['destination_rule'],
                'destination_switching_after_closure': False, 'certificate_checks': proof['certificate_checks'],
                'fixed_destination_certificate_checks': proof['fixed_destination_certificate_checks']}
    summary = {'schema': 'akash-node-drive-access-v1', 'status': 'completed', 'case': plan['case'],
               'water_source': mapper.record(water_manifest), 'plan_source': plan_record, 'source_dates': water['source_dates'],
               'historical_osm': historic, 'temporal_counts': counts,
               'grid': {'crs': grid[0].to_string(), 'transform': list(grid[1]), 'shape': [grid[3], grid[2]],
                        'pixel_area_m2': abs(grid[1].determinant), 'resampling_used': False},
               'road_association': {'half_width_m': 10, 'full_width_m': 20, 'end_caps': 'flat', 'join_style': 'mitre',
                                    'observed_road_width': False, 'positive_overlap_area_tolerance_m2': mapper.EPSILON},
               **{name: mapper.summarize_features(collections[name]) for name in ('roads', 'bridges')},
               'buildings': None, 'category_availability': {'roads': True, 'bridges': True, 'buildings': False,
                   'node_tagged_settlements_and_destinations': True},
               'candidates': candidate_summary, 'source_feature_audits': audits, 'access_scenario': scenario,
               'scope': 'Separate core-only historical access supplement; no complete building exposure or observed closures',
               'reference_labels_loaded': False, 'model_inference_run': False, 'network_requests': 0, 'gpu_used': False,
               'prior_water_walltime_includes_these_stages': False,
               'limitations': ['Building inventory/exposure unavailable, not zero.',
                   'Node-tagged POIs only; polygon-only facilities excluded; operation and point associations unvalidated.',
                   'All other baseline edges assumed available, including unknown/outside-AOI; no passability or rescue-safety findings.',
                   'OSM completeness/history authenticity independent of server query remain unvalidated.',
                   'Water candidates are provisional; no local flood/debris accuracy, physical damage or confirmed closure.'],
               'wall_seconds': time.perf_counter()-start}
    mapper.write_json(out / 'summary.json', summary)
    inputs = [plan_record, *water_records, *core_records]
    for item in inputs: mapper.checked_record(item, Path(item['path']).parent, 'core supplement input')
    code = [Path(__file__), ROOT / 'scripts/acquire_node_drive_historic_osm.py', ROOT / 'scripts/map_fresh_exposure.py',
            ROOT / 'scripts/map_radar_exposure.py', ROOT / 'scripts/prove_access_bounds.py',
            *[ROOT / ('src/akash_flood/' + name) for name in
              ('access.py', 'access_bounds.py', 'nearest_access.py', 'osm.py', 'buildings.py', 'snap_products.py')]]
    mapper.write_json(out / 'source-and-output-manifest.json', {'schema': 'akash-node-drive-access-binding-v1',
        'status': 'completed', 'case': plan['case'], 'created_at': datetime.now(timezone.utc).isoformat(),
        'inputs': inputs, 'water_source': mapper.record(water_manifest), 'plan_source': plan_record,
        'outputs': {p.name: mapper.record(p) for p in sorted(out.iterdir()) if p.is_file()}, 'code': [mapper.record(p) for p in code],
        'reference_labels_loaded': False, 'model_inference_run': False, 'network_requests': 0,
        'building_inventory_available': False, 'prior_water_walltime_includes_these_stages': False})
    return summary


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('water-manifest', 'plan', 'osm', 'osm-provenance', 'out'): p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); print(json.dumps(run(a.water_manifest, a.plan, a.osm, a.osm_provenance, a.out), indent=2))
