"""Verify a portable capsule, then run all frozen phases on this device."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capsule',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resource-mode',choices=['shared-calibration','exclusive'],default='shared-calibration')
    p.add_argument('--device',choices=['0','3'],default='0')
    p.add_argument('--correctness-only',action='store_true')
    p.add_argument('--compare-direct-pack',action='store_true')
    p.add_argument('--graph-batch',type=int,default=1)
    p.add_argument('--timing-rounds',type=int,default=5)
    a=p.parse_args();manifest=json.loads((a.capsule/'capsule.json').read_text())
    for name,meta in manifest['files'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe capsule path')
        data=(a.capsule/path).read_bytes()
        if len(data)!=meta['bytes'] or hashlib.sha256(data).hexdigest()!=meta['sha256']:raise ValueError(f'Corrupt capsule: {name}')
    from thunderworld_fp4.cutlass.artifact import verify_library
    verify_library(a.library)
    cmd=[sys.executable,str(Path(__file__).with_name('probe.py')),'--library',str(a.library),
        '--capture',str(a.capsule/'capture'),'--config',str(a.capsule/'config.json'),
        '--assets',str(a.capsule/'assets'),'--output',str(a.output),
        '--resource-mode',a.resource_mode,'--device',a.device,'--phases','0','1','2','3',
        '--graph-batch',str(a.graph_batch),'--timing-rounds',str(a.timing_rounds)]
    if a.correctness_only:cmd.append('--correctness-only')
    if a.compare_direct_pack:cmd.append('--compare-direct-pack')
    subprocess.run(cmd,check=True)


if __name__=='__main__':main()
