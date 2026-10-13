"""Prepare a truthful SNAP-readable derivative wrapper for public GRD COGs.

The mirror converted/renamed the measurements. Original ESA manifest checksums
are preserved separately and are not claimed to verify these COG bytes. Each
derived dataObject receives its actual size/MD5; SHA-256/alias provenance is kept.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import requests
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/s1-snap-native'

def digest(path,algorithm='sha256'):
    with path.open('rb') as stream:return hashlib.file_digest(stream,algorithm).hexdigest()

def fetch_metadata(url):
    """Use the same public bucket/path if the regional DNS endpoint fails."""
    urls=[url]
    regional='.s3.eu-central-1.amazonaws.com/'
    if regional in url:urls.append(url.replace(regional,'.s3.amazonaws.com/'))
    for index,candidate in enumerate(urls):
        try:
            response=requests.get(candidate,timeout=30)
            response.raise_for_status()
            if len(response.content)>10_000_000:raise ValueError('Metadata/preview individual cap exceeded')
            return response.content,candidate
        except requests.ConnectionError:
            if index+1==len(urls):raise

def main():
    collection=[]
    access_proof=json.loads((ROOT.parent/'research/data-access-probes/anonymous-s1-check.json').read_text())
    probes={(p['item_id'],p['asset']):p for p in access_proof['earthsearch']['asset_probes']}
    for scene in ['before','after']:
        item=json.loads((DATA/'metadata'/f'{scene}.stac.json').read_text())
        measurement=item['assets']['vv']['href']
        bucket,key=measurement[5:].split('/',1);base='https://'+bucket+'.s3.eu-central-1.amazonaws.com/'+key.rsplit('/measurement/',1)[0]
        name=key.split('/')[-3]+'.SAFE'
        folder=DATA/'derived-safe'/name
        folder.mkdir(parents=True,exist_ok=True)
        original=DATA/'metadata'/f'{scene}.original-manifest.safe'
        doc=ET.fromstring(original.read_bytes());records=[]
        for obj in doc.iter():
            if obj.tag.split('}')[-1]!='dataObject':continue
            location=next((e for e in obj.iter() if e.tag.split('}')[-1]=='fileLocation'),None)
            if location is None:continue
            relative=location.attrib['href'].removeprefix('./')
            target=(folder/relative).resolve()
            if not target.is_relative_to(folder.resolve()):raise ValueError('Manifest path escapes derivative product')
            target.parent.mkdir(parents=True,exist_ok=True)
            is_vv='-vv-' in relative;pol='vv' if is_vv else 'vh'
            if relative.startswith('measurement/'):
                source=DATA/'raw-source'/scene/f'iw-{pol}.tiff'
                probe=probes[(item['id'],pol)]
                expected_size=int(probe['header_range_probe']['content_range'].split('/')[-1])
                if source.stat().st_size!=expected_size:raise ValueError('Source measurement is incomplete')
                with source.open('rb') as stream:header_sha=hashlib.sha256(stream.read(65536)).hexdigest()
                if header_sha!=probe['header_range_probe']['sha256']:raise ValueError('Source header differs from verified public asset')
                url=base+'/measurement/iw-'+pol+'.tiff'
                if not target.exists():os.link(source,target)
            else:
                if relative.startswith('annotation/rfi/'):
                    public_relative='annotation/rfi/rfi-iw-'+pol+'.xml'
                elif relative.startswith('annotation/calibration/noise-'):
                    public_relative='annotation/calibration/noise-iw-'+pol+'.xml'
                elif relative.startswith('annotation/calibration/calibration-'):
                    public_relative='annotation/calibration/calibration-iw-'+pol+'.xml'
                elif relative.startswith('annotation/'):
                    public_relative='annotation/iw-'+pol+'.xml'
                else:public_relative=relative
                url=base+'/'+public_relative
                transport_url=url
                if not target.exists():
                    content,transport_url=fetch_metadata(url)
                    target.write_bytes(content)
            stream=next(e for e in obj.iter() if e.tag.split('}')[-1]=='byteStream')
            original_size=stream.attrib.get('size');stream.set('size',str(target.stat().st_size))
            checksum=next((e for e in stream.iter() if e.tag.split('}')[-1]=='checksum'),None)
            original_checksum=checksum.text if checksum is not None else None
            actual_md5=digest(target,'md5')
            if checksum is not None:checksum.text=actual_md5;checksum.set('checksumName','MD5')
            records.append({'alias':relative,'public_url':url,'bytes':target.stat().st_size,'sha256':digest(target),
                            'transport_url':transport_url if not relative.startswith('measurement/') else None,
                            'transport_record_scope':'this wrapper run; existing files retain only source URL',
                            'original_manifest_size':original_size,'original_manifest_checksum':original_checksum,
                            'derived_md5':actual_md5,'source_identity':'provider COG/renamed asset; derived wrapper'})
        output=folder/'manifest.safe';ET.ElementTree(doc).write(output,encoding='utf-8',xml_declaration=True)
        report={'source_item_id':item['id'],'scene':scene,'derived_safe':str(folder.relative_to(ROOT)),
                'original_manifest_sha256':digest(original),'derived_manifest_sha256':digest(output),
                'not_original_esa_safe':True,'purpose':'SNAP standard reader wrapper around provider-converted public COG pixels and corresponding source metadata',
                'files':records,'created_at':datetime.now(timezone.utc).isoformat()}
        (folder/'derived-wrapper-provenance.json').write_text(json.dumps(report,indent=2)+'\n');collection.append(report)
        print(scene,'prepared',len(records),'manifest objects',flush=True)
    (DATA/'derived-wrapper-index.json').write_text(json.dumps(collection,indent=2)+'\n')

if __name__=='__main__':main()
