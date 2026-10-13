"""Add historical-map exposure to completed, immutable water-only processing.

Water is checked against its original source plan. The later OSM binding is
separate; neither processing metadata nor the earlier timing receipt is edited.
All geometry, scenario and certificate calculations reuse the existing mapper.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from akash_flood.fresh_case import event_boundary, osm_plan, reject_source_path
import map_fresh_exposure as mapper

ROOT = Path(__file__).resolve().parents[1]


def checked_supplement(plan_path, plan, osm_path, provenance_path):
    """Bind only the original case's exact pre-event snapshot and map extent."""
    _, _, cutoff = event_boundary(plan['case']['event_date'], plan['case']['osm_snapshot'])
    historic = osm_plan(cutoff, plan['case']['context_bbox'],
                        osm_path=osm_path, provenance_path=provenance_path)
    supplemental = {
        'schema': 'akash-supplemental-historic-map-plan-v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'case': plan['case'], 'osm': historic,
        'original_processing_plan': mapper.record(plan_path),
        'scope': 'Later historical-map/access supplement; original water processing and execution receipt unchanged',
        'new_pixel_acquisition_or_model_inference': False,
        'prior_water_walltime_includes_these_stages': False,
    }
    osm, source, records = mapper.read_historical_osm(plan_path, supplemental)
    return supplemental, osm, source, records


def run(water_manifest, plan_path, osm_path, provenance_path, out):
    water_manifest, plan_path, osm_path, provenance_path, out = [Path(path).resolve()
        for path in (water_manifest, plan_path, osm_path, provenance_path, out)]
    if out.exists() or out.is_symlink():
        raise FileExistsError('Use a new supplement directory; preserve frozen cases')
    reject_source_path(plan_path)
    started = time.perf_counter()
    plan_record = mapper.record(plan_path)
    plan = json.loads(plan_path.read_text())
    arrays, grid, counts, water, water_records = mapper.read_water(water_manifest, plan_path, plan)
    supplement, osm, historical_osm, osm_records = checked_supplement(plan_path, plan, osm_path, provenance_path)
    out.mkdir(parents=True)
    mapper.write_json(out/'historical-map-plan.json', supplement)
    supplement_record = mapper.record(out/'historical-map-plan.json')
    inputs = [plan_record, supplement_record, *water_records, *osm_records]
    mapper.write_json(out/'scenario-policy.json', {**mapper.SCENARIO,
        'recorded_before_feature_overlap_and_access_at': datetime.now(timezone.utc).isoformat(),
        'water_source': mapper.record(water_manifest), 'case_id': plan['case']['case_id'],
        'positive_overlap_area_tolerance_m2': mapper.EPSILON,
        'supplemental_historical_map_plan': supplement_record})
    masks, polygons, regions = mapper.mask_regions(arrays, grid)
    collections, audits, graph = mapper.overlap_features(osm, grid, regions)
    candidates, candidate_summary = mapper.candidate_features(polygons, masks, grid)
    collections.update(candidates)
    proof, settlements, destinations = mapper.access_scenario(graph, osm, collections['roads'])
    proof.update(case=plan['case'], temporal_counts=counts, water_source=mapper.record(water_manifest),
        source_plan=plan_record, historical_osm=historical_osm, source_feature_audits=audits,
        supplemental_historical_map_plan=supplement_record)
    mapper.write_json(out/'access-proof.json', proof)
    collections['settlements'] = mapper.point_features(settlements, proof_rows=proof['settlements'], bbox=plan['case']['bbox'])
    collections['destinations'] = mapper.point_features(destinations, bbox=plan['case']['bbox'])
    for name, collection in collections.items():
        mapper.write_json(out/(name+'.geojson'), collection)
    selection_counts = Counter(row['nearest_baseline_destination']['destination_id']
        for row in proof['settlements'] if row['nearest_baseline_destination'])
    scenario = {
        'schema': proof['schema'], 'scenario_is_hypothetical': True,
        'status_counts': proof['status_counts'], 'n_uncertain_physical_segments': len(proof['uncertain_links']),
        'n_uncertain_directed_edges': sum(map(len, proof['uncertain_links'].values())),
        'n_settlements': len(settlements), 'n_destinations': len(destinations), 'confirmed_closures': 0,
        'fixed_selected_destination_counts': dict(sorted(selection_counts.items())),
        'destination_rule': proof['destination_rule'], 'destination_switching_after_closure': False,
        'certificate_checks': proof['certificate_checks'],
        'fixed_destination_certificate_checks': proof['fixed_destination_certificate_checks'],
        'proof': mapper.record(out/'access-proof.json'), 'scenario_policy': mapper.record(out/'scenario-policy.json'),
        'interpretation': 'Conditional bounds for the declared candidate-selected uncertainty set; unspecified failures excluded. Unknown radar and candidate absence do not establish passability.',
    }
    summary = {
        'schema': 'akash-fresh-exposure-v1', 'status': 'completed', 'case': plan['case'],
        'water_source': mapper.record(water_manifest), 'plan_source': plan_record,
        'supplemental_historical_map_plan': supplement_record, 'source_dates': water['source_dates'],
        'historical_osm': historical_osm, 'temporal_counts': counts,
        'grid': {'crs': grid[0].to_string(), 'transform': list(grid[1]), 'shape': [grid[3],grid[2]],
            'pixel_area_m2': abs(grid[1].determinant), 'resampling_used': False,
            'polygonization': 'Exact four-connected native pixel cells; no smoothing, simplification or geometry repair'},
        'road_association': {'half_width_m': 10, 'full_width_m': 20, 'end_caps': 'flat', 'join_style': 'mitre',
            'observed_road_width': False, 'positive_overlap_area_tolerance_m2': mapper.EPSILON},
        **{name: mapper.summarize_features(collections[name]) for name in ('roads','bridges','buildings')},
        'candidates': candidate_summary, 'source_feature_audits': audits, 'access_scenario': scenario,
        'scope': 'Later historical OSM supplement, including unknown/outside-AOI source features and source audit exclusions; no original source plan changes',
        'reference_labels_loaded': False, 'model_inference_run': False, 'network_requests': 0, 'gpu_used': False,
        'prior_water_walltime_includes_these_stages': False,
        'limitations': [
            'Water-state changes are provisional; permanent and flood water were inseparable in training labels.',
            'No debris, confirmed flood, damage, observed closure or operational safety inference.',
            'The 20 m corridor is a geometric association, not a measured road width.',
            'OSM completeness, node-only destinations, facility operation and road restrictions remain unvalidated.',
            'Access bounds retain baseline edges except the declared uncertainty set; unknown failures are excluded.',
            'Historical map and access were added after the saved water-only execution; its elapsed time excludes this work.',
        ], 'wall_seconds': time.perf_counter()-started,
    }
    mapper.write_json(out/'summary.json', summary)
    for item in inputs:
        mapper.checked_record(item, Path(item['path']).parent, 'supplement input during execution')
    code_paths = [Path(__file__), ROOT/'scripts/map_fresh_exposure.py', ROOT/'scripts/map_radar_exposure.py',
        ROOT/'scripts/prove_access_bounds.py',
        *[ROOT/('src/akash_flood/'+name) for name in ('access.py','access_bounds.py','nearest_access.py','buildings.py','osm.py','snap_products.py','fresh_case.py')]]
    binding = {
        'schema': 'akash-fresh-exposure-binding-v1', 'status': 'completed', 'case': plan['case'],
        'created_at': datetime.now(timezone.utc).isoformat(), 'summary': mapper.record(out/'summary.json'),
        'water_source': mapper.record(water_manifest), 'plan_source': plan_record,
        'supplemental_historical_map_plan': supplement_record, 'inputs': inputs,
        'outputs': {path.name:mapper.record(path) for path in sorted(out.iterdir()) if path.is_file()},
        'code': [mapper.record(path) for path in code_paths],
        'reference_labels_loaded': False, 'model_inference_run': False, 'network_requests': 0,
        'prior_water_walltime_includes_these_stages': False,
    }
    mapper.write_json(out/'source-and-output-manifest.json', binding)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--water-manifest',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--osm',type=Path,required=True)
    parser.add_argument('--osm-provenance',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    result = run(args.water_manifest,args.plan,args.osm,args.osm_provenance,args.out)
    print(json.dumps({key:result[key] for key in ('status','case','access_scenario','wall_seconds')},indent=2))


if __name__ == '__main__':
    main()
