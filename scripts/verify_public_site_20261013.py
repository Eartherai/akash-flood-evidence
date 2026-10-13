"""Check saved display provenance and portable links; no hazard/model evaluation."""
from collections import Counter
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'docs/site'

def require(value, reason):
    if not value:
        raise AssertionError(reason)

def check_record(path, record):
    raw = path.read_bytes()
    require(len(raw) == record['bytes'], f'Length mismatch: {path}')
    require(sha256(raw).hexdigest() == record['sha256'], f'Hash mismatch: {path}')

manifest = json.loads((SITE / 'derivation-manifest.json').read_text())
check_record(SITE / manifest['source']['path'], manifest['source'])
for name, rec in manifest['derived_files'].items():
    check_record(SITE / name, rec)

# Replay only the presentation derivation in an isolated temporary directory.
# This script never refits a model or re-solves the road graph.
with TemporaryDirectory() as tmp:
    replay = Path(tmp)
    (replay / 'case').mkdir()
    shutil.copy2(SITE / 'case/case-facts.json', replay / 'case/case-facts.json')
    shutil.copy2(SITE / 'derive_evidence.py', replay / 'derive_evidence.py')
    subprocess.run([sys.executable, str(replay / 'derive_evidence.py')], check=True, capture_output=True)
    for name in ['evidence.json', 'routes.json', 'derivation-manifest.json']:
        require((SITE / name).read_bytes() == (replay / name).read_bytes(), f'Presentation replay differs: {name}')

facts = json.loads((SITE / 'case/case-facts.json').read_text())
evidence = json.loads((SITE / 'evidence.json').read_text())
routes = json.loads((SITE / 'routes.json').read_text())
require(len(facts['settlements']) == len(evidence['settlements']) == manifest['settlement_count'] == 55, 'Settlement cohort changed')
require(dict(Counter(s['status'] for s in facts['settlements'])) == manifest['source_status_counts'], 'Status counts changed')
for original, display in zip(facts['settlements'], evidence['settlements']):
    require(original['id'] == display['id'], 'Source point order differs')
    require(original['status'] == display['status'], 'Status changed during presentation')
    if original.get('path_support'):
        src = original['path_support']['segments']
        dst = routes[original['id']]
        require(len(src) == len(dst), 'Route segments omitted')
        for a, b in zip(src, dst):
            require(a['coordinates'] == b['coordinates'] and a['edge_id'] == b['edge_id'], 'Witness geometry changed')

case_manifest = json.loads((SITE / 'case/manifest.json').read_text())
for name, record in case_manifest['outputs'].items():
    check_record(SITE / 'case' / name, record)

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {'src', 'href', 'poster'} and value:
                self.targets.append((key, value))

checked = 0
for page in SITE.rglob('*.html'):
    links = Links()
    links.feed(page.read_text())
    for kind, raw in links.targets:
        url = urlsplit(raw)
        if url.scheme or url.netloc or not url.path:
            continue
        require(not url.path.startswith('/'), f'Nonportable root path in {page}: {raw}')
        target = (page.parent / unquote(url.path)).resolve()
        require(target.is_relative_to(SITE.resolve()), f'Link escapes website: {raw}')
        require(target.exists(), f'Missing {kind} in {page}: {raw}')
        checked += 1
for required in ['index.html', 'NOTICES.md', 'case/index.html', 'geography/index.html']:
    require((SITE / required).is_file(), f'Missing public entry: {required}')
print(json.dumps({'status': 'passed', 'settlements': 55, 'route_geometry_unchanged': True,
                  'presentation_derivation_byte_identical': True, 'local_html_links_checked': checked,
                  'model_or_hazard_evaluation': False}, indent=2))
