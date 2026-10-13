import json
import pytest

from akash_flood.public_discovery import public_search


def item(name):
    return {'type':'Feature','id':name,'collection':'sentinel-1-grd'}


class Response:
    status_code=200
    def __init__(self,features,links=()):
        self.content=json.dumps({'type':'FeatureCollection','features':features,'links':list(links)}).encode()
    def raise_for_status(self):pass


class Session:
    def __init__(self,responses):self.responses=iter(responses);self.calls=[]
    def request(self,method,url,**kwargs):
        self.calls.append((method,url,kwargs));return next(self.responses)


def test_full_pages_and_raw_bytes_bound(tmp_path):
    s=Session([Response([item('one')],[{'rel':'next','href':'https://earth-search.aws.element84.com/v1/search?next=2'}]),Response([item('two')])])
    items,r=public_search('sentinel-1-grd',[85,28,85.1,28.1],'2026-08-26',tmp_path/'out',session=s)
    assert [i['id'] for i in items]==['one','two'] and r['search_complete']
    assert s.calls[0][0]=='POST' and s.calls[1][0]=='GET'
    assert len(r['pages'])==2 and len(r['pages'][0]['sha256'])==64


@pytest.mark.parametrize('url',['http://earth-search.aws.element84.com/v1/search','https://evil.example/v1/search','https://earth-search.aws.element84.com@evil.example/v1/search'])
def test_external_pagination_refused_before_request(tmp_path,url):
    s=Session([Response([item('one')],[{'rel':'next','href':url}])])
    with pytest.raises(ValueError,match='official'):
        public_search('sentinel-1-grd',[85,28,85.1,28.1],'2026-08-26',tmp_path/'out',session=s)
    assert len(s.calls)==1


def test_page_cap_and_duplicate_not_declared_complete(tmp_path):
    link={'rel':'next','href':'https://earth-search.aws.element84.com/v1/search?next=2'}
    with pytest.raises(ValueError,match='Page budget'):
        public_search('sentinel-1-grd',[85,28,85.1,28.1],'2026-08-26',tmp_path/'cap',session=Session([Response([item('one')],[link])]),max_pages=1)
    with pytest.raises(ValueError,match='Repeated item'):
        public_search('sentinel-1-grd',[85,28,85.1,28.1],'2026-08-26',tmp_path/'dup',session=Session([Response([item('one')],[link]),Response([item('one')])]))
