"""Declared-provider isolation and failure retention; no HTTP request in tests."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import fetch_historic_osm_20261012 as adapter
import pytest


def args(out):
    return ['--bbox','85.26','27.72','85.35','27.81','--event-date','2024-09-28',
        '--osm-snapshot','2024-09-01','--provider','private-coffee','--out',str(out)]


def test_explicit_provider_restores_base_endpoint_and_preserves_failed_attempt(tmp_path):
    endpoint=adapter.base.OSM_ENDPOINT
    seen=[]
    def fail(bbox,snapshot,event,out,**kwargs):
        seen.append((adapter.base.OSM_ENDPOINT,bbox,snapshot,event))
        raise ValueError('fixture historical server failed')
    with patch.object(adapter.base,'fetch_bounded_historic_osm',fail):
        with pytest.raises(ValueError,match='fixture historical'):
            adapter.main(args(tmp_path/'attempt'))
    assert len(seen)==1
    assert seen[0][0]==adapter.ENDPOINTS['private-coffee']
    assert seen[0][2:] == ('2024-09-01','2024-09-28')
    assert adapter.base.OSM_ENDPOINT==endpoint
    failure=json.loads((tmp_path/'attempt/failure.json').read_text())
    assert failure['automatic_retry'] is False
    assert failure['current_data_fallback'] is False


def test_unlisted_provider_is_rejected_before_transport(tmp_path):
    values=args(tmp_path/'attempt');values[values.index('private-coffee')]='https://arbitrary.invalid'
    with patch.object(adapter.base,'fetch_bounded_historic_osm') as transport:
        with pytest.raises(SystemExit):adapter.main(values)
        transport.assert_not_called()
    assert not (tmp_path/'attempt').exists()
