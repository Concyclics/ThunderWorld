"""Verify fresh 1.5x outputs against the frozen imported quality cohort."""
import argparse
import hashlib
import json
from pathlib import Path

from baselines import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    manifest_path = a.study/'manifest.json'
    study = json.loads(manifest_path.read_text())
    digest = sha(manifest_path)
    method = 'b1.5_protected_n'
    result = dict(complete=True, manifest_sha256=digest, method=method,
                  cases=len(study['cases']), outputs=0, all_bitwise_equal=True, rows=[])
    for entry in study['cases']:
        case = entry['case']['id']
        roots = [a.study/'fresh_1p5'/case, a.study/'generation'/method/case]
        receipts = [json.loads((root/'result.json').read_text()) for root in roots]
        hashes = []
        for root, receipt in zip(roots, receipts):
            identity = receipt['identity']
            if (not receipt['complete'] or identity['manifest_sha256'] != digest or
                    identity['method'] != method or identity['case'] != entry['case'] or
                    identity['image_sha256'] != entry['image_sha256'] or
                    identity['seeds'] != entry['seeds']):
                raise ValueError('Unmatched fresh/imported quality receipt: '+str(root))
            rows = receipt['rows']
            if len(rows) != len(entry['seeds']) or {r['seed'] for r in rows} != set(entry['seeds']):
                raise ValueError('Missing or duplicate seeds: '+str(root))
            actual = {}
            for row in rows:
                path = root/row['outputs']
                if sha(path) != row['outputs_sha256']:
                    raise ValueError('Changed output file: '+str(path))
                with np.load(path, allow_pickle=False) as arrays:
                    actual[str(row['seed'])] = {
                        key: hashlib.sha256(arrays[key].tobytes()).hexdigest() for key in arrays.files}
                if actual[str(row['seed'])] != row['sha256']:
                    raise ValueError('Output content disagrees with receipt: '+str(path))
            hashes.append(actual)
        if hashes[0] != hashes[1]:
            raise ValueError('Fresh 1.5x output differs from the frozen version: '+case)
        result['outputs'] += len(entry['seeds'])
        result['rows'].append(dict(case=case, seeds=entry['seeds'], output_hashes=hashes[0],
            fresh_receipt_sha256=sha(roots[0]/'result.json'),
            imported_receipt_sha256=sha(roots[1]/'result.json'), bitwise_equal=True))
    write(a.study/'fresh_1p5_verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
