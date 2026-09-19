"""Freeze every RBench / PAI-Bench I2V input under the existing 35-step recipe."""
import argparse
import copy
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re

from PIL import Image
from baselines import sha
from dataset_subset import SOURCES
from quality_benchmark import immutable

EXPECTED = {'rbench': 650, 'paibench': 1044}
NEGATIVE = 'distorted objects, flickering, blurry, low quality'


def prompt_rows(cosmos):
    root = cosmos/'evaluation/cosmos3/generator'
    rows = []; hashes = {}
    for path in sorted((root/'rbench/assets/prompts').glob('*.json')):
        hashes[str(path)] = sha(path)
        for row in json.loads(path.read_text()):
            rows.append(dict(benchmark='rbench', category=path.stem.removesuffix('_prompts'),
                source_id=row['name'], image_path='imgs/'+row['image_path'], prompt=row['prompt']))
    path = root/'paibench_g/assets/i2v_prompts.json'; hashes[str(path)] = sha(path)
    for row in json.loads(path.read_text()):
        rows.append(dict(benchmark='paibench', category=row['image_name'].split('_')[0],
            source_id=row['video_id'], image_path='condition_image/'+row['image_name'], prompt=row['prompt_en']))
    if dict(Counter(row['benchmark'] for row in rows)) != EXPECTED:
        raise ValueError('Unexpected full I2V corpus sizes')
    if len({(row['benchmark'], row['source_id']) for row in rows}) != len(rows):
        raise ValueError('Duplicate benchmark sample ID')
    return sorted(rows, key=lambda row: (row['benchmark'], row['category'], row['source_id'])), hashes


def historical_seeds(workspace):
    """Audit local saved outputs without descending model/code dependency caches."""
    found = set(); pattern = re.compile(r'seed(\d+)\.npz$')
    skip = {'.git', '.venvs', '.cache', 'node_modules', 'artifacts', 'data', 'models',
            'checkpoints', 'third_party', 'assets', '__pycache__', '.pytest_cache'}
    for folder, dirs, files in os.walk(workspace, followlinks=False):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith('.venv')]
        for name in files:
            match = pattern.fullmatch(name)
            if match: found.add(int(match[1]))
    return found


def freeze(a):
    rows, hashes = prompt_rows(a.cosmos)
    prior = {}
    for model in ('edge', 'nano'):
        m = json.loads((a.previous/model/'final/manifest.json').read_text())
        prior[model] = {e['case']['id']: e for e in m['cases']}
    if prior['edge'] != prior['nano']:
        raise ValueError('Previous models did not use the same paired cases')
    roles = defaultdict(dict); reserved = set()
    for model, path in [('edge', a.workspace/'repair_redesign_20260917/cohort.json'),
                        ('nano', a.workspace/'full_quality_ready_20260919/nano/cohort.json')]:
        cohort = json.loads(path.read_text())
        for role, cases in cohort['cases'].items():
            for entry in cases:
                roles[entry['case']['id']].setdefault(model, []).append(role)
                reserved.update(entry['seeds'])
    used = historical_seeds(a.workspace) | reserved
    needed = (len(rows)-len(prior['edge']))*2
    start = a.seed_start
    while any(seed in used for seed in range(start, start+needed)): start += 10000
    seed = start; cases = []; reused = []; image_groups = defaultdict(list)
    for row in rows:
        cid = row['benchmark']+'__'+row['source_id']
        image = a.root/'data'/row['benchmark']/row['image_path']
        digest = sha(image)
        with Image.open(image) as im:
            rgb = im.convert('RGB'); pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
        case = dict(id=cid, task='i2v', prompt=row['prompt'], image=str(image.resolve()),
                    height=192, width=320, frames=61, negative_prompt=NEGATIVE)
        previous = prior['edge'].get(cid)
        if previous:
            old_case = dict(previous['case']); old_case['image'] = case['image']
            if old_case != case or previous['image_sha256'] != digest:
                raise ValueError('Previous subset differs from full corpus input: '+cid)
            entry = copy.deepcopy(previous); reused.append(cid)
        else:
            entry = dict(benchmark=row['benchmark'], category=row['category'], source_id=row['source_id'],
                case=case, image_sha256=digest, pixel_sha256=pixel_hash, seeds=[seed, seed+1])
            seed += 2
        entry['prior_scene_roles'] = roles.get(cid, {})
        image_groups[digest].append(cid); cases.append(entry)
    if set(reused) != set(prior['edge']): raise ValueError('Missing original subset case')
    value = dict(schema=1, frozen=True, complete=True, coverage='all_I2V_cases', counts=EXPECTED,
        sources=SOURCES, prompt_hashes=hashes, cases=cases, reused_case_ids=reused,
        reuse_root=str(a.previous.resolve()),
        reuse_manifest_sha256={model: sha(a.previous/model/'final/manifest.json') for model in prior},
        protocol=dict(width=320, height=192, frames=61, fps=24, steps=35, guidance_scale=6.,
            flow_shift=3., scheduler='HostScheduleUniPC', seeds_per_scene=2,
            positive_prompt='original prompt / prompt_en', negative_prompt=NEGATIVE,
            add_resolution_template=False, add_duration_template=False, enable_sound=False),
        seed_audit=dict(requested_start=a.seed_start, assigned_range=[start, seed-1],
            existing_local_output_or_cohort_seeds=len(used), new_seeds=needed,
            scope='Local seed*.npz files plus both frozen model cohorts; no remote-history claim'),
        duplicate_image_groups=[ids for ids in image_groups.values() if len(ids)>1],
        prior_scene_role_counts={model:dict(Counter(role for e in cases for role in
            e['prior_scene_roles'].get(model, []))) for model in prior},
        scope='Complete RBench 650 / PAI-Bench 1044 I2V inputs under the previous reduced-resolution '
              'generation protocol. Not the Cosmos 720p/50-step reproduction recipe. No T2V. '
              'All scenes retained, including declared previous calibration/development overlap.')
    immutable(a.root/'cohort.json', value)
    print(json.dumps(dict(frozen=str(a.root/'cohort.json'), counts=EXPECTED, reused_scenes=len(reused),
                         new_seeds=[start, seed-1], videos_all_models_methods=len(cases)*20)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--cosmos', type=Path, default=Path('/home/yufeng/thunderworld/cosmos'))
    p.add_argument('--workspace', type=Path, default=Path('/home/yufeng/thunderworld'))
    p.add_argument('--previous', type=Path, default=Path('/home/yufeng/thunderworld/judge_generation_20260919'))
    p.add_argument('--seed-start', type=int, default=401001)
    freeze(p.parse_args())


if __name__ == '__main__': main()
