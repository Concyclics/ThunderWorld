"""Check public documentation links, result/figure identities and source boundaries."""
import csv
import json
from pathlib import Path
import re
from common import ROOT, sha


def main():
    docs = [ROOT / 'README.md', ROOT / 'CONTRIBUTING.md', ROOT / 'THIRD_PARTY_NOTICES.md']
    docs += [ROOT / 'docs' / (name + '.md') for name in
             ('INSTALL', 'INFERENCE', 'CALIBRATION', 'BASELINES', 'REPRODUCE', 'RESULTS', 'ARCHITECTURE', 'PERFORMANCE', 'JUDGE')]
    checked = 0
    for file in docs:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', file.read_text()):
            target = target.split('#')[0]
            if not target or re.match(r'[a-z]+://', target): continue
            if not (file.parent / target).exists(): raise ValueError(f'Broken link: {file}: {target}')
            checked += 1
    for value in json.loads((ROOT / 'docs/assets/provenance.json').read_text()):
        if sha(ROOT / 'docs/assets' / value['file']) != value['sha256']:
            raise ValueError('Changed attributed paper artwork')
    provenance = json.loads((ROOT / 'docs/results/provenance.json').read_text())
    for file, digest in provenance['files_sha256'].items():
        if sha(ROOT / file) != digest: raise ValueError('Changed published measurement source')
    rows = list(csv.DictReader((ROOT / 'docs/results/pixel_summary.csv').open()))
    keys = [(r['model'], r['benchmark'], r['method'], r['aggregation']) for r in rows]
    if len(keys) != len(set(keys)) or len(rows) != 3 * 2 * 8 * 2:
        raise ValueError('Incomplete/duplicate result table')
    for row in rows:
        n = {'rbench': 650, 'paibench': 1044}[row['benchmark']]
        if int(row['scenes']) != n or int(row['videos']) != 2 * n:
            raise ValueError('Unexpected benchmark cardinality')
    if 'Apache License' not in (ROOT / 'LICENSE').read_text(): raise ValueError('Missing project license')
    print(json.dumps(dict(complete=True, local_document_links=checked, pixel_rows=len(rows),
        figure_hashes=3, performance_eligible=False), indent=2))


if __name__ == '__main__':
    main()
