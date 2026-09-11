"""Check the portable handoff using only the Python standard library."""
from pathlib import Path
import hashlib
import json
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]


def main():
    profiles = ROOT / 'tools/optimization_handoff/profiles'
    manifest = json.loads((profiles / 'manifest.json').read_text())
    for model, receipt in manifest.items():
        path = ROOT / receipt['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['sha256']
        compact = json.loads(path.read_text())
        source = ROOT / compact['source_policy']['path']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == compact['source_policy']['sha256']
        full = json.loads(source.read_text())
        assert compact['complete'] and compact['model'] == model
        for key in ('model_revision', 'phases', 'launches', 'validation_gate'):
            assert compact[key] == full[key], (model, key)
        assert compact['policies'].keys() == full['policies'].keys()
        for key, policy in compact['policies'].items():
            assert policy['assignments'] == full['policies'][key]['assignments']
    result_dir = ROOT / 'results/optimization_handoff_20260911'
    audit = json.loads((result_dir / 'retime.json').read_text())
    assert audit['complete'] and len(audit['rows']) == 86
    assert sum(row['family'] == 'A' for row in audit['rows']) == 56
    for row in audit['rows']:
        assert all(len(samples) == 5 for samples in row['samples_ms'].values())
        if row['family'] == 'A':
            assert all(value['bitwise'] for value in row['errors'].values())
        else:
            assert row['errors']['plane']['bitwise']
    for model in ('edge', 'nano'):
        smoke = json.loads((result_dir / f'smoke_{model}.json').read_text())
        assert smoke['complete'] and smoke['args']['steps'] == 2
        assert len(smoke['rows']) == 4 and smoke['fused_ffn_layers'] > 0
        for row in smoke['rows']:
            assert row['phase_graph']['replays'] > 0
            assert row['counts']['reused_output'] == row['counts']['reused_delta'] == 0
            if row['mode'] != 'bf16':
                assert row['ffn_calls'] > 0 and row['budget_actual']['tier_calls']
    docs = [ROOT / 'docs/OPTIMIZATION_HANDOFF.md', ROOT / 'tools/optimization_handoff/README.md']
    for path in docs:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            assert (path.parent / unquote(target.split('#')[0])).exists(), (path, target)
    print('PASS: two runtime profiles, 86 retimings, Edge/Nano smoke and handoff links')


if __name__ == '__main__':
    main()
