"""The supplement must never replace a saved water-processing source plan."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import map_supplemental_exposure as supplement
import map_fresh_exposure as mapper
from test_map_fresh_exposure import fixture, write_json


def water_only_fixture(tmp_path):
    manifest, plan_path, water, plan, osm = fixture(tmp_path)
    osm_path = tmp_path/'osm.json'
    provenance = tmp_path/'osm.provenance.json'
    # Set up a synthetic water-only run with the original unmet map query.
    plan['osm'] = {'status':'query_planned_not_retrieved','local_inputs':[]}
    write_json(plan_path,plan)
    processing_path = tmp_path/'processing.json'
    processing = json.loads(processing_path.read_text())
    processing['source_plan'] = mapper.record(plan_path)
    write_json(processing_path,processing)
    water['processing_record'] = mapper.record(processing_path)
    write_json(manifest,water)
    return manifest,plan_path,osm_path,provenance


def test_later_map_reuses_original_water_and_preserves_all_inputs(tmp_path):
    paths = water_only_fixture(tmp_path)
    originals = {path:mapper.record(path) for path in tmp_path.rglob('*') if path.is_file()}
    result = supplement.run(*paths,tmp_path/'supplement')
    assert result['status'] == 'completed'
    assert result['model_inference_run'] is False and result['network_requests'] == 0
    assert result['prior_water_walltime_includes_these_stages'] is False
    assert result['access_scenario']['confirmed_closures'] == 0
    assert {path:mapper.record(path) for path in originals} == originals
    binding = json.loads((tmp_path/'supplement/source-and-output-manifest.json').read_text())
    assert binding['plan_source'] == mapper.record(paths[1])
    assert binding['supplemental_historical_map_plan']['path'] != str(paths[1])
    assert binding['code'][0]['path'].endswith('map_supplemental_exposure.py')
    assert all(mapper.record(item['path']) == item for item in binding['outputs'].values())
    proof = json.loads((tmp_path/'supplement/access-proof.json').read_text())
    assert proof['source_plan'] == mapper.record(paths[1])
    assert proof['historical_osm']['source'] == mapper.record(paths[2])


@pytest.mark.parametrize('field,value', [
    ('snapshot','2026-07-02T00:00:00Z'),
    ('event_date','2026-08-21'),
    ('bbox',[85.0,28.0,85.001,28.001]),
    ('sha256','0'*64),
])
def test_mismatched_map_identity_is_rejected_before_output(tmp_path,field,value):
    paths = water_only_fixture(tmp_path)
    value_json = json.loads(paths[3].read_text())
    value_json[field] = value
    write_json(paths[3],value_json)
    with pytest.raises(ValueError):
        supplement.run(*paths,tmp_path/'supplement')
    assert not (tmp_path/'supplement').exists()


def test_replacement_source_plan_and_changed_raster_fail(tmp_path):
    paths = water_only_fixture(tmp_path)
    replacement = tmp_path/'replacement.json'
    replacement.write_bytes(paths[1].read_bytes())
    with pytest.raises(ValueError,match='same-plan'):
        supplement.run(paths[0],replacement,*paths[2:],tmp_path/'supplement')
    label = tmp_path/'water/new-water-candidates.tif'
    with label.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError,match='Changed checksum-bound'):
        supplement.run(*paths,tmp_path/'supplement')
