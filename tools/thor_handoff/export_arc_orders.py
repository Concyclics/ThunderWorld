"""Export exact Edge ARC permutations from the existing full-M teacher panels."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    import torch
    from baselines import sha,write
    from portable import tensor_hash
    torch.set_num_threads(2)
    manifest = json.loads((a.capture/'manifest.json').read_text())
    if not manifest['complete']:raise ValueError('Incomplete teacher panels')
    names = sorted(set(row['name'] for row in manifest['rows'] if row['kind']=='linear'))
    orders = {}
    for name in names:
        rows = [r for r in manifest['rows'] if r['kind']=='linear' and r['name']==name]
        if len(rows)!=16:raise ValueError('Incomplete ARC calibration '+name)
        peak = None
        for row in rows:
            file = a.capture/row['file']
            if sha(file) != row['sha256']:raise ValueError('Changed teacher panel')
            x = torch.load(file,weights_only=True,map_location='cpu')['x']
            current = x.float().abs().amax(0)
            peak = current if peak is None else torch.maximum(peak,current)
        orders[name] = torch.argsort(peak,stable=True)
        print(json.dumps(dict(completed=len(orders),total=len(names),name=name)),flush=True)
    a.output.mkdir(parents=True,exist_ok=False)
    torch.save(orders,a.output/'orders.pt')
    write(a.output/'manifest.json',dict(schema='arc-channel-order-v1',complete=True,
        model_identity=manifest['model_identity'],source_capture_sha256=sha(a.capture/'manifest.json'),
        panels_per_module=16,file='orders.pt',sha256=sha(a.output/'orders.pt'),
        tensor_hashes={k:tensor_hash(v) for k,v in orders.items()}))


if __name__=='__main__':main()
