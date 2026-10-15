"""Append two asymmetric FP4 products to a frozen BF16-consistency cohort."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from baselines import sha, write

HERE = Path(__file__).resolve().parent
DEFAULT_PARENT = HERE.parents[2]/'fp4_expansion_demo_20260921'
VARIANTS = {'fp4_two_a': (1, 2, 3), 'fp4_two_w': (2, 1, 17)}


def prepare(root, parent):
    target = root/'edge/final/manifest.json'
    if target.exists():
        raise ValueError('Extension already frozen; resume without preparing again')
    origin = parent/'edge/final/manifest.json'
    base = json.loads(origin.read_text())
    study = copy.deepcopy(base)
    methods = {}
    for name, (wp, ap, mask) in VARIANTS.items():
        spec = copy.deepcopy(base['methods']['fp4_base'])
        spec['expansion'] = dict(weight_planes=wp, activation_planes=ap, mask=mask, terms=2, dual=False)
        methods[name] = spec
    study.update(methods=methods, role='exploratory_two_term_extension',
        extension=dict(parent_root=str(parent), manifest_sha256=sha(origin),
            parent_metrics_sha256=sha(parent/'report/results.json'), source_sha256=sha(__file__),
            reuse='Reuse original BF16 outputs with verified exact input, seed, scope and model identities'),
        expansion_assets=str(parent/'offline_weights'), implementation_lock=str(parent/'implementation.lock.json'))
    for e in study['cases']:
        if sha(e['case']['image']) != e['image_sha256']:
            raise ValueError('Changed original input')
    write(target, study)
    write(root/'cohort.json', dict(frozen=True, manifest=str(target), manifest_sha256=sha(target), cases=study['cases']))
    return dict(complete=True, cases=len(study['cases']), methods=list(methods))


def load(root):
    path = root/'edge/final/manifest.json'
    study = json.loads(path.read_text())
    parent = Path(study['extension']['parent_root'])
    origin = parent/'edge/final/manifest.json'
    base = json.loads(origin.read_text())
    if sha(origin) != study['extension']['manifest_sha256']:
        raise ValueError('Changed parent manifest')
    for key in ('cases', 'steps', 'model', 'protocol', 'statistics'):
        if study[key] != base[key]:
            raise ValueError('Unmatched parent '+key)
    return study, parent, base


def qualify(root):
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    from expansion_runtime import source_identity
    study, parent, _ = load(root)
    lock = json.loads(Path(study['implementation_lock']).read_text())
    if lock['source_sha256'] != source_identity():
        raise ValueError('Changed parent kernels')
    folder = root/'qualification'; folder.mkdir(parents=True, exist_ok=True)
    test = HERE/'test_expansion_two_term.py'
    command = [sys.executable, '-m', 'pytest', str(test), '-x', '-q']
    commands = {'pytest': command}
    for tool in ('memcheck', 'racecheck'):
        commands[tool] = ['/usr/local/cuda/bin/compute-sanitizer', '--tool', tool,
            '--error-exitcode', '99', sys.executable, str(HERE/'expansion_two_term.py'), 'smoke', '--root', str(root)]
    result = dict(complete=False, source_sha256=source_identity(), test_sha256=sha(test))
    with shared_worker():
        result['resource'] = select_shared_device()
        for name, cmd in commands.items():
            log = folder/(name+'.log')
            with log.open('w') as stream:
                code = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT).returncode
            result[name] = dict(exit_code=code, log=str(log), sha256=sha(log), command=cmd)
            write(folder/'results.json', result)
            print(json.dumps({name: result[name]}), flush=True)
            if code:
                raise RuntimeError('Two-term qualification failed: '+name)
    result['complete'] = True
    write(folder/'results.json', result)
    return result


def smoke():
    import torch
    from expansion_kernels import pack_linear, linear, attention
    torch.manual_seed(92223)
    with torch.inference_mode():
        x = torch.randn(129, 256, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16)
        q = torch.randn(1, 129, 4, 128, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(1, 145, 2, 128, device='cuda', dtype=torch.bfloat16)
        v = torch.randn_like(k)
        for wp, ap, mask in VARIANTS.values():
            wc, ws, wg, _ = pack_linear(w, wp)
            y = linear(x, (wc, ws, wg), wp, ap, mask)
            z = attention(q, k, v, wp, ap, mask)
            torch.cuda.synchronize()
            assert torch.isfinite(y).all() and torch.isfinite(z).all()
    return dict(complete=True, masks=[3, 17])


def metrics(root, follow=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    import lpips
    from thunderworld_fp4.cosmos.quality import compare
    torch.set_num_threads(8); torch.hub.set_dir('/tmp/thunderworld-lpips-cache')
    metric = lpips.LPIPS(net='alex', version='0.1').eval().cpu()
    study, parent, base = load(root)
    identity = dict(manifest_sha256=sha(root/'edge/final/manifest.json'),
        reference_manifest_sha256=sha(parent/'edge/final/manifest.json'), source_sha256=sha(__file__),
        quality_sha256=sha(HERE.parents[1]/'packages/fp4-repair/src/thunderworld_fp4/cosmos/quality.py'))
    output = root/'edge/final/metrics'; output.mkdir(parents=True, exist_ok=True)
    def signals(folder, row):
        path = folder/row['outputs']
        if sha(path) != row['outputs_sha256']:
            raise ValueError('Corrupt raw output')
        with np.load(path) as data:
            arrays = dict(data)
        if {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()} != row['sha256']:
            raise ValueError('Changed raw signal')
        return arrays
    while True:
        done = 0
        for entry in study['cases']:
            cid = entry['case']['id']
            bf = parent/'edge/final/generation/bf16'/cid
            reference = json.loads((bf/'result.json').read_text())
            if not reference['complete'] or reference['identity']['manifest_sha256'] != identity['reference_manifest_sha256']:
                raise ValueError('Wrong BF16 reference')
            for name in study['methods']:
                folder = root/'edge/final/generation'/name/cid
                if not (folder/'result.json').exists():
                    continue
                receipt = json.loads((folder/'result.json').read_text())
                if not receipt['complete']:
                    continue
                if receipt['identity']['manifest_sha256'] != identity['manifest_sha256']:
                    raise ValueError('Wrong generated cohort')
                for key in ('case', 'image_sha256', 'seeds'):
                    if receipt['identity'][key] != reference['identity'][key]:
                        raise ValueError('Unmatched '+key)
                if receipt['model_identity'] != reference['model_identity'] or receipt['targets'] != reference['targets']:
                    raise ValueError('Unmatched model/projection scope')
                if [r['seed'] for r in receipt['rows']] != entry['seeds'] or [r['seed'] for r in reference['rows']] != entry['seeds']:
                    raise ValueError('Missing paired seeds')
                provenance = dict(identity=identity, reference_sha256=sha(bf/'result.json'), candidate_sha256=sha(folder/'result.json'))
                target = output/name/(cid+'.json')
                if target.exists():
                    previous = json.loads(target.read_text())
                    if not previous['complete'] or previous['provenance'] != provenance:
                        raise ValueError('Changed metric inputs')
                else:
                    rows = []
                    for row, gold in zip(receipt['rows'], reference['rows']):
                        for item in (row, gold):
                            if item['steps'] != 35 or item['calls'] != 70 or item['debug'] != 'off' or item['execution'] != 'graph':
                                raise ValueError('Changed generation settings')
                        rows.append(dict(seed=row['seed'], metrics=compare(signals(folder, row), signals(bf, gold), metric)))
                    write(target, dict(complete=True, name=name, case_id=cid, benchmark=entry['benchmark'],
                        category=entry['category'], rows=rows, provenance=provenance))
                    print(json.dumps(dict(method=name, case=cid)), flush=True)
                done += 1
        state = dict(identity=identity, complete=done == 20, completed=done, expected=20)
        write(output/'progress.json', state)
        if state['complete'] or not follow:
            return state
        suite = root/'edge/final/generation/suite.json'
        if suite.exists() and json.loads(suite.read_text()).get('paused'):
            raise RuntimeError('Generation paused; metric progress preserved')
        time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'qualify', 'smoke', 'run', 'metrics'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--gpus', nargs='+', default=['0', '1'])
    args = parser.parse_args(); root = args.root.resolve()
    if args.operation == 'prepare':
        result = prepare(root, args.parent.resolve())
    elif args.operation == 'qualify':
        result = qualify(root)
    elif args.operation == 'smoke':
        result = smoke()
    elif args.operation == 'metrics':
        result = metrics(root, args.follow)
    else:
        from expansion_experiment import run
        q = json.loads((root/'qualification/results.json').read_text())
        if not q['complete']:
            raise ValueError('Run two-term qualification first')
        for key in ('pytest', 'memcheck', 'racecheck'):
            if q[key]['exit_code'] != 0 or sha(q[key]['log']) != q[key]['sha256']:
                raise ValueError('Changed qualification')
        result = run(root, list(VARIANTS), args.gpus)
    print(json.dumps(result))
