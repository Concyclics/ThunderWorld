"""Reconstruct frozen calibration cohorts from pinned public prompt/image sources."""
import argparse
import hashlib
from pathlib import Path
import urllib.request
import subprocess
from common import ROOT, immutable, read, sha

COSMOS_REVISION = '5b50bba0fb97f4aa07b03d299bc1a22bafa10ade'
PROMPTS = 'evaluation/cosmos3/generator'
RBENCH = ('common_manipulation', 'dual_arm', 'humanoid', 'long-horizon_planning',
          'multi-entity_collaboration', 'quad', 'single_arm', 'spatial_relationship',
          'visual_reasoning')
NEGATIVE = 'distorted objects, flickering, blurry, low quality'


def prompt_catalog(cache, cosmos_root=None):
    cache = Path(cache)
    if cosmos_root:
        actual = subprocess.check_output(['git', '-C', str(cosmos_root), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != COSMOS_REVISION:
            raise ValueError('Use the pinned NVIDIA/cosmos prompt revision')
    rows, files = {}, {}
    paths = [('rbench', f'{PROMPTS}/rbench/assets/prompts/{c}_prompts.json') for c in RBENCH]
    paths.append(('paibench', f'{PROMPTS}/paibench_g/assets/i2v_prompts.json'))
    for benchmark, relative in paths:
        path = Path(cosmos_root) / relative if cosmos_root else cache / relative
        if not path.exists():
            if cosmos_root:
                raise FileNotFoundError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f'https://raw.githubusercontent.com/NVIDIA/cosmos/{COSMOS_REVISION}/{relative}'
            with urllib.request.urlopen(url, timeout=120) as response:
                contents = response.read()
            temp = path.with_suffix('.tmp'); temp.write_bytes(contents); temp.replace(path)
        files[relative] = sha(path)
        for row in read(path):
            if benchmark == 'rbench':
                ident, prompt, image = row['name'], row['prompt'], 'imgs/' + row['image_path']
                category = relative.rsplit('/', 1)[1].removesuffix('_prompts.json')
            else:
                ident, prompt, image = row['video_id'], row['prompt_en'], 'condition_image/' + row['image_name']
                category = ident.split('_')[0]
            key = (benchmark, ident)
            if key in rows:
                raise ValueError(f'Duplicate public prompt: {key}')
            rows[key] = dict(prompt=prompt, image=image, category=category)
    return rows, files


def image_entry(meta, row, dataset, *, local_files_only=False, dataset_root=None):
    from PIL import Image
    if dataset_root:
        path = Path(dataset_root) / meta['benchmark'] / row['image']
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(dataset['repo_id'], row['image'], repo_type='dataset',
            revision=dataset['revision'], local_files_only=local_files_only))
    image_hash = sha(path)
    with Image.open(path) as img:
        img = img.convert('RGB')
        pixel_hash = hashlib.sha256(str(img.size).encode() + img.tobytes()).hexdigest()
    prompt_hash = hashlib.sha256(row['prompt'].encode()).hexdigest()
    for key, value in [('image_sha256', image_hash), ('pixel_sha256', pixel_hash), ('prompt_sha256', prompt_hash)]:
        if key in meta and meta[key] != value:
            raise ValueError(f'Changed {key}: {meta["source_id"]}')
    case = dict(id=meta['case_id'], task='i2v', prompt=row['prompt'], image=str(path.resolve()),
                height=192, width=320, frames=61, negative_prompt=NEGATIVE)
    return dict(benchmark=meta['benchmark'], category=meta['category'], source_id=meta['source_id'],
                case=case, seeds=meta['seeds'], image_sha256=image_hash, pixel_sha256=pixel_hash)


def prepare(model, output, base_model, *, cosmos_root=None, dataset_root=None,
            local_files_only=False, full=False):
    output = Path(output).resolve()
    metadata = ROOT / f'configs/public/{model}-calibration.json'
    frozen = read(metadata)
    catalog, sources = prompt_catalog(output / 'prompts', cosmos_root)
    roles = dict(frozen['splits'])
    if full:
        # Full-corpus evaluation includes earlier development scenes; no held-out claim.
        roles['full'] = [dict(benchmark=b, category=r['category'], source_id=i,
            case_id=f'{b}__{i}', seeds=[501001 + 2*j, 501002 + 2*j])
            for j, ((b, i), r) in enumerate(sorted(catalog.items()))]
    cases = {}
    for role, entries in roles.items():
        cases[role] = [image_entry(e, catalog[e['benchmark'], e['source_id']],
            frozen['datasets'][e['benchmark']], local_files_only=local_files_only,
            dataset_root=dataset_root) for e in entries]
    result = dict(schema_version=1, frozen=True, model=str(Path(base_model).resolve()),
        model_name=model, steps=35, cases=cases, source_calibration_sha256=sha(metadata),
        datasets=frozen['datasets'], prompt_source=dict(repository='NVIDIA/cosmos',
            revision=COSMOS_REVISION, files_sha256=sources),
        scope='Frozen calibration/development/validation/final roles. Optional full corpus overlaps development; no cross-scene generalization claim.')
    immutable(output / 'cohort.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=('edge', 'nano', 'super'), default='edge')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-model', type=Path, required=True)
    p.add_argument('--cosmos-root', type=Path, help='Optional pinned NVIDIA/cosmos checkout')
    p.add_argument('--dataset-root', type=Path, help='Optional directory with rbench/ and paibench/ Hub file layouts')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--full', action='store_true', help='Also include all 650 RBench and 1044 PAI-Bench-G prompts')
    a = p.parse_args()
    result = prepare(a.model, a.output, a.base_model, cosmos_root=a.cosmos_root,
        dataset_root=a.dataset_root, local_files_only=a.local_files_only, full=a.full)
    print({role: len(entries) for role, entries in result['cases'].items()})


if __name__ == '__main__':
    main()
