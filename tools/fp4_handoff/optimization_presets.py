"""Export a new materialized study after complete development byte qualification."""
import argparse
import copy
import json
from pathlib import Path

from bundle import ROOT
from baselines import sha, write
from optimize import VARIANTS
from optimization_suite import METHODS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--qualification', type=Path, required=True,
                   help='Directory containing quality/<method>_<variant>/result.json')
    p.add_argument('--variant', choices=['calibrated', 'pointwise', 'rotary', 'codec', 'norm'], default='norm')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    source = a.study/'manifest.json'
    manifest = copy.deepcopy(json.loads(source.read_text()))
    cases = {c['case']['id']: set(c['seeds']) for c in manifest['cases']}
    proofs = {}
    for name in METHODS:
        receipt = a.qualification/'quality'/f'{name}_{a.variant}'/'result.json'
        d = json.loads(receipt.read_text())
        if not d['complete'] or d['identity']['manifest_sha256'] != sha(source):
            raise ValueError('Unmatched complete qualification')
        if set(d['cases']) != set(cases):
            raise ValueError('Qualification must include all frozen development scenes')
        for case, seeds in cases.items():
            rows = d['cases'][case]['rows']
            if len(rows) != len(seeds) or {r['seed'] for r in rows} != seeds:
                raise ValueError('Qualification must include all paired seeds exactly once')
            if not all(r['source_hashes_match'] is True for r in rows):
                raise ValueError('Execution optimization changed output bytes')
        if d['identity']['effective_spec']['runtime_options'] != VARIANTS[a.variant]:
            raise ValueError('Execution option mismatch')
        manifest['methods'][name]['runtime_options'] = VARIANTS[a.variant]
        proofs[name] = dict(receipt=str(receipt.resolve()), sha256=sha(receipt))
    manifest['execution_revision'] = dict(variant=a.variant, parent_manifest_sha256=sha(source),
        qualification=proofs, scope='Same offline assets and repair plans; development output bytes preserved. Device-specific speed is separate.')
    a.output.mkdir(parents=True, exist_ok=False)
    write(a.output/'manifest.json', manifest)
    print(str(a.output/'manifest.json'))


if __name__ == '__main__':
    main()
