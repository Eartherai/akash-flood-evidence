"""Check physical scaling, nodata, cloud neighborhood and bounded transport."""
import copy
import hashlib

import numpy as np
import pytest

from akash_flood.optical_change import reflectance,average_2x2,screen,StrictRangeHTTP,select_optical_pair,PUBLIC_HOST


def bands(h=15):
    before={k:np.full((h,h),v,dtype='float32') for k,v in [('blue',.1),('red',.1),('nir',.4),('swir16',.1)]}
    after={k:np.full((h,h),v,dtype='float32') for k,v in [('blue',.08),('red',.2),('nir',.22),('swir16',.4)]}
    return before,after


def test_scale_offset_applied_once_and_dn_zero_stays_unknown():
    x,valid=reflectance(np.array([[0,1000,2000,5000,32767]],dtype='uint16'),.0001,-.1,0)
    assert np.isnan(x[0,:2]).all() and np.isnan(x[0,4])
    assert np.allclose(x[0,2:4],[.1,.4])
    assert valid.tolist()==[[False,False,True,True,False]]


def test_one_missing_native_sample_invalidates_entire20m_cell():
    values=np.arange(16,dtype='float32').reshape(4,4)+1;valid=np.ones((4,4),bool);valid[0,0]=False
    values[0,0]=np.nan;x,v=average_2x2(values,valid)
    assert np.isnan(x[0,0]) and not v[0,0]
    assert x[1,1]==13.5
    with pytest.raises(ValueError):average_2x2(values[:3],valid[:3])


def test_cloud_and60m_neighborhood_stay_unknown_not_negative_detection():
    before,after=bands();scl=np.full((15,15),4,dtype='uint8');post=scl.copy();post[7,7]=9
    labels,assessed,_,counts,_=screen(before,after,scl,post,np.ones((15,15),bool))
    assert labels[7,7]==255 and labels[7,10]==255 and labels[7,11]==1
    assert not assessed[7,7] and counts['unknown_pixels']>1
    assert counts['candidate_pixels']==counts['joint_assessed_pixels']


def test_stable_vegetation_is_not_surface_change_and_invalid_reflectance_remains_unknown():
    before,_=bands();scl=np.full((15,15),4,dtype='uint8');same=copy.deepcopy(before)
    labels,_,_,counts,_=screen(before,same,scl,scl,np.ones((15,15),bool))
    assert counts['candidate_pixels']==0 and labels[7,7]==0
    same['nir'][7,7]=np.nan
    labels,_,_,_,_=screen(before,same,scl,scl,np.ones((15,15),bool))
    assert labels[7,7]==255


def test_water_scl_excluded_even_when_spectral_criteria_match():
    before,after=bands();scl=np.full((15,15),6,dtype='uint8')
    labels,_,_,counts,clear=screen(before,after,scl,scl,np.ones((15,15),bool))
    assert np.all(labels==255) and counts['joint_assessed_pixels']==0
    assert clear['after']==225  # Original clear-SCL audit included water; screening is land-only.


def item(identity,when,cloud=0,tile='AA'):
    return {'type':'Feature','id':identity,'collection':'sentinel-2-c1-l2a',
            'geometry':{'type':'Polygon','coordinates':[[[0,0],[1,0],[1,1],[0,1],[0,0]]]},'assets':{},
            'properties':{'datetime':when,'platform':'sentinel-2a','proj:epsg':32631,
                          'mgrs:utm_zone':31,'mgrs:latitude_band':'N','mgrs:grid_square':tile,'eo:cloud_cover':cloud}}


def test_selector_ignores_cloud_ranking_and_excludes_entire_event_day():
    document={'type':'FeatureCollection','features':[
        item('oldclear','2026-08-10T01:00:00Z'),item('latestcloudy','2026-08-25T10:00:00Z',90),
        item('eventday','2026-08-26T20:00:00Z'),item('firstcloudy','2026-08-27T09:00:00Z',80),
        item('laterclear','2026-08-28T09:00:00Z')]}
    pair=select_optical_pair(document,[.1,.1,.9,.9],'2026-08-26')
    assert pair['before']['id']=='latestcloudy' and pair['after']['id']=='firstcloudy'


class Response:
    def __init__(self,data,status=206,etag='fixed',content_range='bytes 0-3/100'):
        self.data=data;self.status_code=status;self.headers={'Content-Range':content_range,'ETag':etag}
    def iter_content(self,n):yield self.data
    def close(self):pass


class Session:
    def __init__(self,responses):self.responses=iter(responses);self.calls=[]
    def get(self,url,**kwargs):self.calls.append((url,kwargs));return next(self.responses)


URL='https://'+PUBLIC_HOST+'/sentinel-2-c1-l2a/31/N/AA/2026/8/a/B04.tif'


def test_ranges_verify_tls_exactbytes_etag_and_bodysha_without_redirect():
    session=Session([Response(b'abcd')]);client=StrictRangeHTTP(session=session)
    payload,total=client.range(URL,0,4)
    assert payload==b'abcd' and total==100 and client.bytes_read==4
    assert client.requests[0]['sha256']==hashlib.sha256(b'abcd').hexdigest()
    assert session.calls[0][1]['verify'] is True and session.calls[0][1]['allow_redirects'] is False
    assert session.trust_env is False


@pytest.mark.parametrize('response',[Response(b'abcd',status=200),Response(b'abcd',content_range='bytes 1-4/100'),Response(b'abc'),Response(b'abcde'),Response(b'abcd',etag='')])
def test_no_full_response_wrong_range_incomplete_or_missingidentity_fallback(response):
    client=StrictRangeHTTP(session=Session([response]))
    with pytest.raises(ValueError):client.range(URL,0,4)


def test_range_budget_refused_before_second_request_and_remote_identity_change_rejected():
    session=Session([Response(b'abcd')]);client=StrictRangeHTTP(5,session=session);client.range(URL,0,4)
    with pytest.raises(ValueError,match='limit'):client.range(URL,4,8)
    assert len(session.calls)==1
    session=Session([Response(b'abcd'),Response(b'efgh',etag='changed',content_range='bytes 4-7/100')]);client=StrictRangeHTTP(session=session);client.range(URL,0,4)
    with pytest.raises(ValueError,match='identity changed'):client.range(URL,4,8)


def test_unsigned_allowlist_rejects_credentials_query_and_other_hosts():
    client=StrictRangeHTTP(session=Session([]))
    for url in (URL+'?token=secret',URL.replace('https://','http://'),URL.replace(PUBLIC_HOST,'untrusted.example')):
        with pytest.raises(ValueError,match='exact public'):client.range(url,0,4)
