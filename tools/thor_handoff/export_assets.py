"""Export compact, exact-tensor recipes for Edge and Nano without teacher panels."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'tools/fp4_handoff'), str(ROOT/'tools/contribution_study'),
               str(ROOT/'packages/fp4-repair/src')]
from bundle import PACKED, _hashes, corrections, apply_corrections, rebuild
from baselines import sha, write

REPAIRS = ('b1.5_protected_n', 'b2_joint', 'b2.5_ffn_guarded')
METHODS = ('bf16', 'fp8', 'fp4_base', *REPAIRS, 'fp4_three', 'fp4_four')
MODELS = {
    'edge': ('nvidia/Cosmos3-Edge', 'a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba'),
    'nano': ('nvidia/Cosmos3-Nano', '411f42a8fdfb8c5b2583cb8786e0938f49796eaa'),
}


def export(study, output, model_name, threads):
    import torch
    from safetensors import safe_open
    from thunderworld_fp4.cosmos.runner import model_identity
    from thunderworld_fp4.cosmos.calibration import quantize_weight
    torch.set_num_threads(threads)
    original = json.loads((study/'manifest.json').read_text())
    model = Path(original['model'])
    identity = model_identity(model)
    output.mkdir(parents=True, exist_ok=False)
    repo, revision = MODELS[model_name]
    template = dict(schema=1, model_name=model_name, steps=original['steps'],
                    cases=copy.deepcopy(original['cases']),
                    methods={n: copy.deepcopy(original['methods'][n]) for n in METHODS},
                    source_manifest_sha256=sha(study/'manifest.json'),
                    performance_eligible=False, selection_eligible=False)
    for entry in template['cases']:
        src = Path(entry['case']['image'])
        if sha(src) != entry['image_sha256']:
            raise ValueError('Changed input')
        dst = output/'inputs'/(entry['case']['id']+src.suffix)
        dst.parent.mkdir(exist_ok=True)
        shutil.copyfile(src, dst)
        entry['case']['image'] = str(dst.relative_to(output))
    package = dict(schema='fp4-baselines-handoff-v1', complete=False,
                   model=dict(repo=repo, revision=revision), model_identity=identity,
                   source_manifest_sha256=sha(study/'manifest.json'), repairs={})
    banks = {}
    for name in REPAIRS:
        spec = template['methods'][name]
        bank = Path(spec['assets'])
        manifest = json.loads((bank/'manifest.json').read_text())
        if (sha(bank/'manifest.json') != spec['assets_sha256'] or not manifest['complete']
                or manifest['model_identity'] != identity):
            raise ValueError('Changed/wrong source assets')
        plan = Path(spec['config']['budget_plan'])
        if sha(plan) != spec['plan_sha256']:
            raise ValueError('Changed plan')
        (output/name/'recipes').mkdir(parents=True)
        shutil.copyfile(plan, output/name/'plan.json')
        spec['config']['budget_plan'] = name+'/plan.json'
        spec['assets'] = name+'/assets'
        banks[name] = bank, manifest
        package['repairs'][name] = dict(source_manifest=manifest, modules={})
    index = json.loads((model/'transformer/diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    names = sorted(set.union(*(set(m['modules']) for _, m in banks.values())))
    for count, module in enumerate(names, 1):
        key = module+'.weight'
        with safe_open(model/'transformer'/index[key], framework='pt', device='cpu') as handle:
            weight = handle.get_tensor(key).bfloat16()
        cache = {}
        for name, (bank, manifest) in banks.items():
            entry = manifest['modules'][module]
            path = bank/entry['file']
            if sha(path) != entry['sha256']:
                raise ValueError('Changed source: '+str(path))
            source = torch.load(path, weights_only=True, map_location='cpu')
            if source.get('mandatory_w_blocks'):
                raise ValueError('Outlier asset requires an explicit removal-mask recipe')
            metadata = {k:v for k,v in source.items() if k not in PACKED}
            from portable import tensor_hash
            ck = tensor_hash(metadata['prescale']), tuple(metadata['weight_scale'])
            if ck not in cache:
                cache[ck] = quantize_weight(weight, metadata['prescale'], base=ck[1][0], residual=ck[1][1])
            rebuilt = rebuild(weight, metadata, cache[ck])
            patch = corrections(rebuilt, source)
            if _hashes(apply_corrections(rebuilt, patch)) != _hashes(source):
                raise ValueError('Exact-byte recipe roundtrip failed')
            file = name+'/recipes/'+Path(entry['file']).name
            torch.save(dict(metadata=metadata, corrections=patch), output/file)
            package['repairs'][name]['modules'][module] = dict(file=file,
                tensor_hashes=_hashes(source), source_file_sha256=entry['sha256'],
                correction_bytes=sum(i.numel()*8+v.numel() for i,v in patch.values()))
        print(json.dumps(dict(model=model_name, exported=count, total=len(names))), flush=True)
    write(output/'study.json', template)
    package['files'] = {str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    package['complete'] = True
    write(output/'package.json', package)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    export(a.study, a.output, a.model, a.threads)
