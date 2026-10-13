import copy

import pytest

from akash_flood.radar import processing_graph


SCENE = {"id": "test-scene", "properties": {"datetime": "2026-08-16T12:21:41Z",
    "sat:relative_orbit": 85, "sat:orbit_state": "ascending", "sar:instrument_mode": "IW", "sar:polarizations": ["VV", "VH"]}}


def test_recipe_fixes_orbit_dem_and_output_units_without_collapsing_time():
    graph = processing_graph(SCENE, [85.30, 28.12, 85.43, 28.32])
    load = graph['load']['arguments']
    assert load['temporal_extent'] == ['2026-08-16', '2026-08-17']
    assert load['properties']['relativeOrbitNumber']['process_graph']['eq']['arguments']['y'] == 85
    assert graph['calibrate']['arguments']['elevation_model'] == 'COPERNICUS_30'
    assert graph['calibrate']['arguments']['coefficient'] == 'sigma0-ellipsoid'
    assert graph['grid']['arguments']['projection'] == 32645
    assert not any(n['process_id'] in {'reduce_dimension', 'log'} for n in graph.values())


def test_southern_hemisphere_uses_southern_utm():
    graph = processing_graph(SCENE, [20, -30, 20.1, -29.9])
    assert graph['grid']['arguments']['projection'] == 32734


@pytest.mark.parametrize('property,value', [('sat:relative_orbit', None), ('sar:polarizations', ['HH']), ('sat:orbit_state', '')])
def test_missing_required_metadata_rejects_preparation(property, value):
    scene = copy.deepcopy(SCENE)
    scene['properties'][property] = value
    with pytest.raises(ValueError):
        processing_graph(scene, [85.30, 28.12, 85.43, 28.32])
