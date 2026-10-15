"""Attach successful addressing/sanitizer checks to an immutable variant."""
import argparse
import json
from pathlib import Path
from build import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',type=Path,required=True)
    p.add_argument('--qualification',type=Path,required=True)
    a=p.parse_args();path=a.variant/'variant.json';receipt=json.loads(path.read_text())
    if receipt.get('qualified'):raise ValueError('Already sealed')
    for name,digest in receipt['variant_hashes'].items():
        if sha(a.variant/'thunderworld_fp4'/name)!=digest:raise ValueError('Variant changed: '+name)
    library=sha(a.variant/'cutlass_sm110.so')
    files={}
    for name in ('linear','memcheck','racecheck'):
        source=a.qualification/(name+'.json');d=json.loads(source.read_text())
        if not d['complete'] or d['library_sha256']!=library:raise ValueError('Unqualified artifact: '+name)
        if not all(r['bitwise_equal'] and r['same_mask'] and r['same_codes_scales'] for r in d['rows']):raise ValueError('Numerical mismatch')
        if len(d['rows'])!=(112 if name=='linear' else 16) or len(d['graphs'])!=(14 if name=='linear' else 2):raise ValueError('Missing cases')
        files[name]=dict(path=str(source),sha256=sha(source))
        if name!='linear':
            log=source.with_suffix('.log');s=log.read_text()
            expected='ERROR SUMMARY: 0 errors' if name=='memcheck' else '0 hazards displayed (0 errors, 0 warnings)'
            if expected not in s:raise ValueError('Unclean sanitizer: '+name)
            files[name]['log_sha256']=sha(log)
    receipt.update(qualified=True,memory_qualification=files)
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(qualified=True,variant=str(a.variant),sha256=sha(path))),flush=True)


if __name__=='__main__':main()
