"""Portable, exact-byte recipes for the frozen Edge repair/baseline comparison."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'packages/fp4-repair/src'),
               str(ROOT / 'tools/contribution_study'),
               str(ROOT / 'experiments/cutlass_fused_repair')]
from baselines import sha, write
from portable import PACKED, tensor_hash, rebuild

REPAIRS = ('b1.5_protected_n', 'b2_joint', 'b2.5_ffn_guarded')
METHODS = (*REPAIRS, 'arcquant', 'arcquant100', 'sharq_perf', 'bf16')
MODEL = {'repo': 'nvidia/Cosmos3-Edge', 'revision': 'a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba'}


def relative(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Package path escapes its root: ' + str(name))
    return path


def check_package(root):
    value = json.loads((root / 'package.json').read_text())
    if value.get('schema') != 'fp4-baselines-handoff-v1' or not value.get('complete'):
        raise ValueError('Incomplete or incompatible package')
    for name, digest in value['files'].items():
        if sha(relative(root, name)) != digest:
            raise ValueError('Changed package file: ' + name)
    return value


def _hashes(asset):
    import torch
    return {k: tensor_hash(v) for k, v in asset.items() if isinstance(v, torch.Tensor)}


def corrections(restored, source):
    import torch
    if set(restored) != set(source):
        raise ValueError('Asset metadata schema changed')
    patch = {}
    for key, value in restored.items():
        if not isinstance(value, torch.Tensor):
            if value != source[key]:
                raise ValueError('Non-tensor metadata mismatch: ' + key)
            continue
        target = source[key]
        if value.dtype != target.dtype or value.shape != target.shape:
            raise ValueError('Tensor schema mismatch: ' + key)
        x = value.contiguous().reshape(-1).view(torch.uint8)
        y = target.contiguous().reshape(-1).view(torch.uint8)
        indices = (x != y).nonzero().flatten()
        if indices.numel():
            patch[key] = (indices, y[indices].clone())
    return patch


def apply_corrections(asset, patch):
    import torch
    result = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in asset.items()}
    for key, (indices, values) in patch.items():
        data = result[key].reshape(-1).view(torch.uint8)
        if indices.dtype != torch.int64 or values.dtype != torch.uint8 or indices.shape != values.shape:
            raise ValueError('Invalid byte correction')
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= data.numel()):
            raise ValueError('Byte correction outside tensor')
        data[indices] = values
    return result


def export(a):
    import torch
    from safetensors import safe_open
    from thunderworld_fp4.cosmos.runner import model_identity
    from thunderworld_fp4.cosmos.calibration import quantize_weight
    torch.set_num_threads(a.threads)
    study_path = a.study / 'manifest.json'
    original = json.loads(study_path.read_text())
    model = Path(original['model'])
    identity = model_identity(model)
    a.output.mkdir(parents=True, exist_ok=False)
    template = {k: copy.deepcopy(original[k]) for k in
                ('schema', 'steps', 'debug', 'resource_mode', 'cases', 'batch_protocol')}
    template.update(frozen=True, methods={n: copy.deepcopy(original['methods'][n]) for n in METHODS},
                    adapters={n: copy.deepcopy(original['adapters'][n]) for n in METHODS if n in original['adapters']},
                    performance_eligible=False, selection_eligible=False,
                    role='Frozen development cohort; relocated reproduction, not new selection',
                    source_manifest_sha256=sha(study_path), model=MODEL)
    template['batch_protocol']['methods'] = list(METHODS)
    for entry in template['cases']:
        path = Path(entry['case']['image'])
        if sha(path) != entry['image_sha256']:
            raise ValueError('Changed source image')
        name = 'inputs/' + entry['case']['id'] + path.suffix
        dest = a.output / name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, dest)
        entry['case']['image'] = name
    banks = {}
    value = dict(schema='fp4-baselines-handoff-v1', complete=False, model=MODEL,
                 model_identity=identity, source_manifest_sha256=sha(study_path), repairs={},
                 scope='No model weights or teacher panels. Rebuild exact packed tensor bytes from fixed HF weights.')
    for name, spec in template['methods'].items():
        if name not in REPAIRS:
            continue
        source_spec = original['methods'][name]
        bank = Path(source_spec['assets'])
        if sha(bank/'manifest.json') != source_spec['assets_sha256']:
            raise ValueError('Changed source asset manifest')
        manifest = json.loads((bank/'manifest.json').read_text())
        if not manifest['complete'] or manifest['model_identity'] != identity:
            raise ValueError('Wrong source model')
        folder = a.output / name
        (folder/'recipes').mkdir(parents=True)
        plan = Path(spec['config']['budget_plan'])
        if sha(plan) != spec['plan_sha256']:
            raise ValueError('Changed frozen plan')
        shutil.copyfile(plan, folder/'plan.json')
        spec['config']['budget_plan'] = name + '/plan.json'
        spec['assets'] = name + '/assets'
        banks[name] = (bank, manifest)
        value['repairs'][name] = dict(source_manifest=manifest, modules={})
    index = json.loads((model/'transformer/diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    names = sorted({n for _, m in banks.values() for n in m['modules']})
    for count, module in enumerate(names):
        key = module + '.weight'
        with safe_open(model/'transformer'/index[key], framework='pt', device='cpu') as handle:
            weight = handle.get_tensor(key).bfloat16()
        cache = {}
        for name, (bank, manifest) in banks.items():
            entry = manifest['modules'][module]
            source_path = bank/entry['file']
            if sha(source_path) != entry['sha256']:
                raise ValueError('Changed source asset: ' + str(source_path))
            source = torch.load(source_path, weights_only=True, map_location='cpu')
            if source.get('mandatory_w_blocks'):
                raise ValueError('This recipe version requires no outlier removal mask')
            metadata = {k: v for k, v in source.items() if k not in PACKED}
            recipe_key = (tensor_hash(metadata['prescale']), tuple(metadata['weight_scale']))
            if recipe_key not in cache:
                cache[recipe_key] = quantize_weight(weight, metadata['prescale'],
                    base=recipe_key[1][0], residual=recipe_key[1][1])
            rebuilt = rebuild(weight, metadata, cache[recipe_key])
            patch = corrections(rebuilt, source)
            if _hashes(apply_corrections(rebuilt, patch)) != _hashes(source):
                raise ValueError('Export roundtrip mismatch')
            filename = name + '/recipes/' + Path(entry['file']).name
            torch.save(dict(metadata=metadata, corrections=patch), a.output/filename)
            value['repairs'][name]['modules'][module] = dict(file=filename,
                tensor_hashes=_hashes(source), source_file_sha256=entry['sha256'],
                correction_bytes=sum(indices.numel()*8 + values.numel() for indices, values in patch.values()))
        print(json.dumps(dict(exported=count+1, total=len(names), module=module)), flush=True)
    # ARC only needs the channel order, not the 32 GB of teacher activations.
    capture = Path(original['external']['capture'])
    if sha(capture/'manifest.json') != original['external']['capture_sha256']:
        raise ValueError('Changed ARC capture')
    panels = json.loads((capture/'manifest.json').read_text())
    orders = {}
    for module in names:
        rows = [r for r in panels['rows'] if r['kind']=='linear' and r['name']==module]
        if len(rows) != 16:
            raise ValueError('Incomplete ARC panels: ' + module)
        peak = None
        for row in rows:
            path = capture/row['file']
            if sha(path) != row['sha256']:
                raise ValueError('Changed ARC panel')
            x = torch.load(path, weights_only=True, map_location='cpu')['x']
            current = x.float().abs().amax(0)
            peak = current if peak is None else torch.maximum(peak, current)
        orders[module] = torch.argsort(peak, stable=True)
    (a.output/'arc_orders').mkdir()
    torch.save(orders, a.output/'arc_orders/orders.pt')
    write(a.output/'arc_orders/manifest.json', dict(schema='arc-channel-order-v1', complete=True,
        model_identity=identity, source_capture_sha256=sha(capture/'manifest.json'), panels_per_module=16,
        file='orders.pt', sha256=sha(a.output/'arc_orders/orders.pt'),
        tensor_hashes={k:tensor_hash(v) for k,v in orders.items()}))
    lock = json.loads(Path(original['external']['sources']).read_text())
    lock['sources'] = {n:{k:v for k,v in lock['sources'][n].items() if k!='path'} for n in ('arcquant','sharq')}
    write(a.output/'sources.lock.json', lock)
    oracle = {}
    for name in METHODS:
        oracle[name] = {}
        for entry in template['cases']:
            case_id = entry['case']['id']
            path = a.study/'generation'/name/case_id/'result.json'
            receipt = json.loads(path.read_text())
            if not receipt['complete'] or receipt['identity']['manifest_sha256'] != sha(study_path):
                raise ValueError('Incomplete/mismatched frozen output receipt')
            oracle[name][case_id] = {str(row['seed']):row['sha256'] for row in receipt['rows']}
    write(a.output/'source_output_hashes.json', oracle)
    write(a.output/'study.json', template)
    value['files'] = {str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()}
    value['complete'] = True
    write(a.output/'package.json', value)


def restore(a):
    import torch
    from safetensors import safe_open
    from thunderworld_fp4.cosmos.runner import model_identity
    from thunderworld_fp4.cosmos.calibration import quantize_weight
    torch.set_num_threads(a.threads)
    value = check_package(a.package)
    if model_identity(a.model) != value['model_identity']:
        raise ValueError('Wrong model revision')
    a.output.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for name, entry in value['repairs'].items():
        bank = a.output/name/'assets'
        bank.mkdir(parents=True)
        manifests[name] = {**entry['source_manifest'], 'complete':False, 'modules':{}}
        write(bank/'manifest.json', manifests[name])
    index = json.loads((a.model/'transformer/diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    names = sorted({n for r in value['repairs'].values() for n in r['modules']})
    for count, module in enumerate(names):
        key = module+'.weight'
        with safe_open(a.model/'transformer'/index[key], framework='pt', device='cpu') as handle:
            weight = handle.get_tensor(key).bfloat16()
        cache = {}
        for name, entry in value['repairs'].items():
            spec = entry['modules'][module]
            recipe = torch.load(relative(a.package, spec['file']), weights_only=True, map_location='cpu')
            metadata = recipe['metadata']
            recipe_key = (tensor_hash(metadata['prescale']), tuple(metadata['weight_scale']))
            if recipe_key not in cache:
                cache[recipe_key] = quantize_weight(weight, metadata['prescale'],
                    base=recipe_key[1][0], residual=recipe_key[1][1])
            asset = apply_corrections(rebuild(weight, metadata, cache[recipe_key]), recipe['corrections'])
            if _hashes(asset) != spec['tensor_hashes']:
                raise ValueError('Reconstructed tensor mismatch: '+name+'/'+module)
            filename = Path(spec['file']).name
            path = a.output/name/'assets'/filename
            torch.save(asset, path)
            manifests[name]['modules'][module] = dict(file=filename, sha256=sha(path))
        print(json.dumps(dict(restored=count+1, total=len(names), module=module)), flush=True)
    for name, manifest in manifests.items():
        manifest['complete'] = True
        write(a.output/name/'assets/manifest.json', manifest)
    write(a.output/'restored.json', dict(complete=True, package_sha256=sha(a.package/'package.json'),
        model_identity=value['model_identity'], modules_per_method={n:len(m['modules']) for n,m in manifests.items()},
        all_tensor_hashes_match=True, performance_eligible=False))


def materialize(a):
    value = check_package(a.package)
    receipt = json.loads((a.restored/'restored.json').read_text())
    if not receipt['complete'] or receipt['package_sha256'] != sha(a.package/'package.json'):
        raise ValueError('Assets were restored from another package')
    from thunderworld_fp4.cosmos.runner import model_identity
    if model_identity(a.model) != value['model_identity']:
        raise ValueError('Wrong model revision')
    study = json.loads((a.package/'study.json').read_text())
    study['model'] = str(a.model.resolve())
    study['handoff_package_sha256'] = sha(a.package/'package.json')
    study['source_output_hashes'] = str((a.package/'source_output_hashes.json').resolve())
    study['methods'] = {n:study['methods'][n] for n in a.methods}
    study['adapters'] = {n:v for n,v in study['adapters'].items() if n in a.methods}
    study['batch_protocol']['methods'] = list(a.methods)
    for name, spec in study['methods'].items():
        if name in REPAIRS:
            spec['config']['budget_plan'] = str(relative(a.package, spec['config']['budget_plan']))
            spec['assets'] = str(relative(a.restored, spec['assets']))
            spec['assets_sha256'] = sha(Path(spec['assets'])/'manifest.json')
    study['external'] = {}
    if any(n in study['adapters'] for n in a.methods):
        if a.sources is None:
            raise ValueError('Baseline methods need --sources LOCAL/sources.lock.json')
        local_sources = json.loads(a.sources.read_text())['sources']
        pinned = json.loads((a.package/'sources.lock.json').read_text())['sources']
        needed = {study['adapters'][n]['method'] for n in a.methods if n in study['adapters']}
        for name in needed:
            if any(local_sources[name][key] != pinned[name][key] for key in ('url','commit')):
                raise ValueError('Wrong baseline source version: '+name)
        study['external'].update(sources=str(a.sources.resolve()), sources_sha256=sha(a.sources))
        if 'sharq' in needed:
            build_root = Path(local_sources['sharq']['path'])/'kernels/build_cmake_sm120a'
            build = json.loads((build_root/'contribution-build.json').read_text())
            if (not build['complete'] or build['source_commit'] != pinned['sharq']['commit']
                    or sha(build_root/'sharq_ops.so') != build['library_sha256']):
                raise ValueError('Unqualified SharQ build')
            study['external'].update(sharq_build=str(build_root.resolve()),
                sharq_build_sha256=sha(build_root/'contribution-build.json'))
        if 'arcquant' in needed:
            if a.arc_build is None:
                raise ValueError('ARC methods need --arc-build')
            build = json.loads((a.arc_build/'build.json').read_text())
            if not build['complete'] or build['source_commit'] != pinned['arcquant']['commit']:
                raise ValueError('Wrong ARC build')
            study['external'].update(arc_build=str(a.arc_build.resolve()), arc_build_sha256=sha(a.arc_build/'build.json'),
                capture=str((a.package/'arc_orders').resolve()), capture_sha256=sha(a.package/'arc_orders/manifest.json'))
    for entry in study['cases']:
        entry['case']['image'] = str(relative(a.package, entry['case']['image']))
    a.output.mkdir(parents=True, exist_ok=False)
    write(a.output/'manifest.json', study)
    print(str(a.output/'manifest.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    e = commands.add_parser('export'); e.add_argument('--study', type=Path, required=True)
    r = commands.add_parser('restore'); r.add_argument('--model', type=Path, required=True)
    m = commands.add_parser('materialize'); m.add_argument('--model', type=Path, required=True)
    m.add_argument('--restored', type=Path, required=True)
    m.add_argument('--sources', type=Path); m.add_argument('--arc-build', type=Path)
    m.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    v = commands.add_parser('verify')
    for p in (r,m,v):p.add_argument('--package', type=Path, required=True)
    for p in (e,r,m):p.add_argument('--output', type=Path, required=True)
    for p in (e,r):p.add_argument('--threads', type=int, default=4)
    a = parser.parse_args()
    if a.operation == 'verify':
        result = check_package(a.package)
        print(json.dumps(dict(complete=True, files=len(result['files']), package_sha256=sha(a.package/'package.json'))))
    else:globals()[a.operation](a)


if __name__ == '__main__':main()
