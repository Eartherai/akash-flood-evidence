"""Derive compact public displays from the unmodified, hash-bound saved case.
Run: python3 docs/site/derive_evidence.py
No acquisition, fitting, model inference, graph solve or candidate reclassification.
"""
from pathlib import Path
import hashlib, json
from collections import Counter

SITE = Path(__file__).resolve().parent
SOURCE = SITE / 'case/case-facts.json'
EXPECTED = 'd5a4b09abfbc724a464c701ac0962af6d056e1c158f1ed111166b9211970547c'
raw = SOURCE.read_bytes()
assert hashlib.sha256(raw).hexdigest() == EXPECTED, 'Saved case changed; do not derive under the old binding.'
facts = json.loads(raw)
settlements = []
routes = {}
for i, original in enumerate(facts['settlements']):
    item = {k: original.get(k) for k in ['id', 'name', 'status', 'point_coordinates', 'inside_water_aoi', 'inside_extraction_context', 'nearest_baseline_destination']}
    item['source_pointer'] = f'/settlements/{i}'
    support = original.get('path_support')
    item['support'] = ({k:v for k,v in support.items() if k != 'segments'} if support else None)
    cut = original.get('cut_certificate')
    item['cut_boundary_edge_count'] = len(cut.get('removed_boundary_edge_ids', [])) if cut else None
    item['cut_reachable_node_count'] = len(cut.get('reachable_node_ids', [])) if cut else None
    item['cut_scope'] = cut.get('scope') if cut else None
    witness = original.get('path_witness')
    item['witness'] = ({k:v for k,v in witness.items() if k not in ['path_node_ids', 'path_edge_ids']} if witness else None)
    settlements.append(item)
    if support:
        route = []
        for segment in support['segments']:
            # A segment is mixed if more than one support length is positive.
            lengths = [segment['assessed_length_m'], segment['unknown_within_aoi_length_m'], segment['outside_aoi_length_m']]
            positives = [index for index, value in enumerate(lengths) if value > 1e-9]
            category = ['assessed', 'unknown', 'outside'][positives[0]] if len(positives) == 1 else 'mixed'
            route.append({'coordinates': segment['coordinates'], 'category': category, 'edge_id': segment['edge_id']})
        routes[item['id']] = route
summary = {k:v for k,v in facts.items() if k not in ['settlements', 'inputs']}
summary['settlements'] = settlements
summary['source_binding'] = {'path':'case/case-facts.json', 'bytes':len(raw), 'sha256':EXPECTED, 'derivation':'derive_evidence.py', 'reclassifies_hazards':False, 'solves_graph':False}
summary['public_view_scope'] = 'Saved completed case. Exact copied statistics; rounded percentages for display. Route vertices copied without simplification. No model or graph computation in this website.'
for name, data in [('evidence.json', summary), ('routes.json', routes)]:
    (SITE/name).write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n')
manifest = {'schema':'akash-public-display-derivation-v1','source': summary['source_binding'], 'derived_files':{name:{'bytes':(SITE/name).stat().st_size,'sha256':hashlib.sha256((SITE/name).read_bytes()).hexdigest()} for name in ['evidence.json','routes.json']},'settlement_count':len(settlements),'source_status_counts':dict(Counter(item['status'] for item in settlements)),'route_coordinate_simplification':False,'unknown_is_not_negative':True}
(SITE/'derivation-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
