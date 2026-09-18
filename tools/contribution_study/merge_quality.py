"""Merge disjoint method batches evaluated against the exact same BF16 receipt."""
import argparse,json
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    merged=None
    for path in a.inputs:
        value=json.loads(path.read_text())
        if not value['complete']:raise ValueError('Incomplete metric batch')
        if merged is None:
            merged=dict(value,rows=list(value['rows']),means=dict(value['means']),receipts=dict(value['receipts']))
        else:
            if value['seeds']!=merged['seeds'] or value['reference_sha256']!=merged['reference_sha256']:
                raise ValueError('Batches differ in seeds or BF16 reference')
            if set(value['means'])&set(merged['means']):raise ValueError('Duplicate methods')
            merged['rows']+=value['rows'];merged['means'].update(value['means']);merged['receipts'].update(value['receipts'])
    merged['batch_sha256']={str(p):sha(p) for p in a.inputs};write(a.output,merged)


if __name__=='__main__':main()
