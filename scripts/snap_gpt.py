"""Run the checksum-verified official SNAP payload with project-local caches."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'data/snap-runtime/portable'

def command(arguments):
    java=RUNTIME/'jre/Contents/Home/bin/java'
    if not java.is_file():raise FileNotFoundError('Verified portable SNAP runtime not prepared')
    cp=':'.join(str(path) for path in sorted(RUNTIME.rglob('*.jar')))
    return [str(java),'-Xmx6G','-Djava.awt.headless=true','-Dsnap.home='+str(RUNTIME),
            '-Dsnap.userdir='+str(RUNTIME.parent/'user-cache'),
            '-DAuxDataPath='+str(RUNTIME.parent/'auxdata'),
            '-DDEM.srtm1HgtDEM_HTTP=https://step.esa.int/auxdata/dem/SRTMGL1/',
            '-cp',cp,
            'org.esa.snap.core.gpf.main.GPT',*arguments]

if __name__=='__main__':raise SystemExit(subprocess.call(command(sys.argv[1:])))
