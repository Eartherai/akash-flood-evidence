import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("historic_acquisition", Path(__file__).parents[1]/"scripts/acquire_historic_osm.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class Response:
    def __init__(self, payload, status=200, advertised=None):
        self.payload = payload
        self.status_code = status
        self.headers = {} if advertised is None else {"Content-Length": str(advertised)}
        self.closed = False

    def iter_content(self, size):
        yield self.payload

    def close(self):
        self.closed = True


class Session:
    auth = None
    headers = {}

    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error:
            raise self.error
        return self.response


def run(tmp_path, session=None, **kwargs):
    return module.acquire([85.22, 27.59, 85.38, 27.85], "2024-09-01", "2024-09-28",
        tmp_path/"case", session=session, **kwargs)


def test_offline_default_and_pre_event_cutoff(tmp_path):
    client = Session(error=AssertionError("Network forbidden"))
    result = run(tmp_path, client)
    assert result["status"] == "protocol_only" and not client.calls
    protocol = json.loads((tmp_path/"case/protocol.json").read_text())
    assert '[date:"2024-09-01T00:00:00Z"]' in protocol["query"]
    assert 'nwr["building"]' in protocol["query"]
    with pytest.raises(ValueError, match="strictly before"):
        module.acquire([85.22,27.59,85.38,27.85], "2024-09-28", "2024-09-28", tmp_path/"bad")


def test_success_binds_exact_raw_and_local_source(tmp_path):
    payload = b'{"elements":[{"type":"node","id":1,"lat":27.7,"lon":85.3}]}'
    response = Response(payload, advertised=len(payload))
    client = Session(response)
    result = run(tmp_path, client, execute=True)
    assert result["status"] == "completed" and result["attempted_requests"] == 1
    assert result["received_body_bytes"] == len(payload) and response.closed
    assert result["raw_response"] == module.record(tmp_path/"case/response.json")
    provenance = json.loads((tmp_path/"case/provenance.json").read_text())
    assert provenance["sha256"] == result["osm"]["sha256"]
    assert client.calls[0][1]["allow_redirects"] is False and client.calls[0][1]["verify"] is True


@pytest.mark.parametrize("response,error", [(Response(b"", status=504), None), (None, ConnectionError("offline"))])
def test_http_and_connection_failures_are_counted_without_retry(tmp_path, response, error):
    client = Session(response, error)
    with pytest.raises((ValueError, ConnectionError)):
        run(tmp_path, client, execute=True)
    result = json.loads((tmp_path/"case/network-receipt.json").read_text())
    assert result["status"] == "failed" and result["attempted_requests"] == 1
    assert len(client.calls) == 1 and result["received_body_bytes"] == 0


@pytest.mark.parametrize("payload", [
    b'{"elements":[],"remark":"runtime error"}',
    b'{"elements":[{"timestamp":"2024-09-29T00:00:00Z"}]}',
    b'{"elements":[{}],"elements":[{}]}',
])
def test_incomplete_future_or_ambiguous_json_is_retained_and_rejected(tmp_path, payload):
    with pytest.raises(ValueError):
        run(tmp_path, Session(Response(payload)), execute=True)
    result = json.loads((tmp_path/"case/network-receipt.json").read_text())
    assert result["status"] == "failed" and result["received_body_bytes"] == len(payload)
    assert result["retained_partial_response"]["sha256"]
    assert not (tmp_path/"case/osm.json").exists()


def test_truncated_body_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="incomplete"):
        run(tmp_path, Session(Response(b"{}", advertised=3)), execute=True)
    assert json.loads((tmp_path/"case/network-receipt.json").read_text())["received_body_bytes"] == 2


def test_wall_deadline_interrupts_and_retains_failure(tmp_path,monkeypatch):
    import signal
    import time

    class Slow(Response):
        def iter_content(self,size):
            time.sleep(.1)
            yield self.payload

    monkeypatch.setattr(module,'DEADLINE_SECONDS',.01)
    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError,match='wall deadline'):
        run(tmp_path,Session(Slow(b'{}')),execute=True)
    receipt = json.loads((tmp_path/'case/network-receipt.json').read_text())
    assert receipt['status'] == 'failed' and receipt['attempted_requests'] == 1
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0,0.0)
