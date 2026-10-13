"""Fresh nearest-baseline-destination proof, preserving the frozen any-target run."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from akash_flood.access import build_road_graph
from akash_flood.nearest_access import analyze_nearest_access
from akash_flood.osm import named_points
from prove_access_bounds import verify_certificates

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT/'outputs/nearest-access-proof'
    if out.exists():
        raise FileExistsError('Preserve earlier proofs; use a fresh output folder')
    source = ROOT/'data/rasuwa-pilot/osm.json'
    graph = build_road_graph(json.loads(source.read_text()))
    settlements, destinations = named_points(json.loads(source.read_text()))
    groups = defaultdict(list)
    for u, v, _, data in graph.edges(keys=True, data=True):
        if data['bridge']:
            groups[f"way-{data['way_id']}-nodes-{min(u,v)}-{max(u,v)}"].append(data['edge_id'])
    start = time.perf_counter()
    result = analyze_nearest_access(graph, settlements, destinations, uncertain_links=dict(groups))
    checks = verify_certificates(graph, result)
    for row in result['settlements']:
        selection = row['nearest_baseline_destination']
        if selection:
            for key in ('path_witness', 'optimistic_path_witness'):
                if row[key] and row[key]['destination_id'] != selection['destination_id']:
                    raise AssertionError('Proof switched away from the fixed baseline destination')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    result['provenance'] = {'created_at': datetime.now(timezone.utc).isoformat(),
        'osm_sha256': sha(source), 'code_sha256': sha(ROOT/'src/akash_flood/nearest_access.py'),
        'source_snapshot': json.loads(source.with_suffix('.provenance.json').read_text())['snapshot'],
        'scenario_is_synthetic': True, 'scenario_rule': 'All bridge-tagged physical road segments uncertain; no observed closures.',
        'n_uncertain_physical_segments': len(groups), 'certificate_checks': checks,
        'analysis_and_verification_seconds': time.perf_counter()-start}
    out.mkdir()
    (out/'proof.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status_counts':result['status_counts'],
        'any_destination_status_counts':result['any_destination_status_counts_for_comparison'],
        'certificate_checks':checks,'wall_seconds':result['provenance']['analysis_and_verification_seconds'],
        'proof_sha256':sha(out/'proof.json')},indent=2))


if __name__ == '__main__':
    main()
