"""Reproduce pipeline optimizations: GPU tests, exact trajectories, isolated timing.

Requires the previously frozen joint_e2e profiles/assets. No toolkit updates.
Never runs GPU jobs concurrently. Performance is gated on exact output equality.
"""
import argparse
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASES = ('bf16', 'joint125', 'cutlass150')
VARIANTS = ('original', 'codec', 'codec_norm')


def read(p): return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--previous', type=Path, default=ROOT/'runs5090/joint_e2e_20260914')
    p.add_argument('--stage', choices=('trajectory', 'performance'), required=True)
    a = p.parse_args(); a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    if a.stage == 'trajectory':
        subprocess.run([sys.executable, '-m', 'pytest', '-q',
            'experiments/cutlass_fused_repair/test_pipeline_optimizations.py'], cwd=ROOT, check=True)
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('GPU tests must run on CUDA, not skip')
    else:
        gate = read(a.output/'trajectory_summary.json')
        if not gate['complete'] or len(gate['equivalence']) != 12 or not all(r['equal'] for r in gate['equivalence']):
            raise RuntimeError('All three controls and both candidates must pass both trajectory seeds first')
    jobs = []; rng = random.Random(214)
    for repeat in range(5 if a.stage == 'performance' else 1):
        names = [(base, variant) for base in BASES for variant in VARIANTS]
        if a.stage == 'performance': rng.shuffle(names)
        for base, variant in names:
            name = base+'_'+variant; dest = a.output/f'{a.stage}_{repeat:02d}_{name}'
            command = [sys.executable, '-m', 'thunderworld_fp4.cosmos',
                'benchmark' if a.stage=='performance' else 'infer', '--model', str(ROOT/'models/Cosmos3-Edge-a9d944e'),
                '--config', str(a.previous/f'configs/{base}.json'), '--tuning', str(a.previous/f'profiles/{base}.json'),
                '--case', str(ROOT/'tools/joint_inference/cases/robot_observe.json'), '--output', str(dest),
                '--steps', '35', '--seeds', *(['7103','7104'] if a.stage=='trajectory' else ['93001','93002','93003'])]
            if base != 'bf16': command += ['--assets', str(ROOT/f'data_local/cutlass5090_rebuilt_20260914/i2v_b{base[-3:]}')]
            if variant != 'original': command += ['--codec-graph']
            if variant == 'codec_norm': command += ['--fuse-norm']
            jobs.append(dict(base=base,variant=variant,name=name,round=repeat,output=str(dest),command=command))
    manifest = dict(complete=False, stage=a.stage, random_order_seed=214, jobs=jobs)
    mp = a.output/f'{a.stage}_matrix.json'
    def save(): mp.write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    for job in jobs:
        dest = Path(job['output']); result = dest/'result.json'
        if result.exists():
            if not read(result).get('complete'): raise RuntimeError(f'Inspect incomplete job: {dest}')
        else:
            print('Running',dest,flush=True)
            with dest.with_suffix('.log').open('w') as log:
                job['exit_code'] = subprocess.run(job['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
            save()
            if job['exit_code']: raise RuntimeError(f'Failed job: {dest}')
    # Audit all receipts before declaring completion, including resumed jobs.
    common = None; by_key = {}; equivalence = []; summaries = []; sources = None; configs = {}
    for job in jobs:
        r = read(Path(job['output'])/'result.json')
        if not r['complete'] or r['tuning_missing']: raise RuntimeError('Incomplete run / missing tuning')
        identity = {k:r[k] for k in ('model_identity','case','image_sha256','environment','source_sha256')}
        if common is not None and identity != common: raise RuntimeError('Mixed source/environment/input')
        common = identity; sources = r['source_sha256']
        if r['codec_graph']['enabled'] != (job['variant']!='original') or r['norm_fusion']['enabled'] != (job['variant']=='codec_norm'):
            raise RuntimeError('Candidate flags do not match receipt')
        fixed = {k:r[k] for k in ('config','asset_identity','cutlass_artifact','observed_shapes')}
        if job['base'] in configs and configs[job['base']] != fixed: raise RuntimeError('Candidate changed model/repair/launch configuration')
        configs[job['base']] = fixed
        if not r['config']['strict_tuning'] or r['config']['debug']!='off': raise RuntimeError('Unfrozen / instrumented run')
        if len(r['rows']) != (2 if a.stage=='trajectory' else 3): raise RuntimeError('Missing seeds')
        for row in r['rows']:
            if not row['performance_eligible'] or row['timing_scope']!='complete_request_v2_including_host_setup':
                raise RuntimeError('Invalid timing scope')
            key = (job['base'],job['variant'],job['round'],row['seed']); by_key[key] = row
        if a.stage=='performance' and sources != read(a.output/'trajectory_summary.json')['source_sha256']:
            raise RuntimeError('Source changed after trajectory gate')
    seeds = (7103,7104) if a.stage=='trajectory' else (93001,93002,93003)
    for base in BASES:
        for seed in seeds:
            reference = by_key[base,'original',0,seed]['sha256']
            for variant in VARIANTS:
                for repeat in range(5 if a.stage=='performance' else 1):
                    if by_key[base,variant,repeat,seed]['sha256'] != reference:
                        raise RuntimeError(f'Output changed: {base}/{variant}/{repeat}/{seed}')
                if variant!='original':equivalence.append(dict(base=base,variant=variant,seed=seed,equal=True))
        for variant in VARIANTS:
            distinct = {json.dumps(by_key[base,variant,0,s]['sha256'],sort_keys=True) for s in seeds}
            if len(distinct) != len(seeds): raise RuntimeError('Stale / repeated seeded outputs')
            rows = [v for (b,c,_,_),v in by_key.items() if (b,c)==(base,variant)]
            summaries.append(dict(base=base,variant=variant,n=len(rows),median_seconds=statistics.median(r['wall_seconds'] for r in rows),
                peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
                peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows)))
    if a.stage=='trajectory':
        for base in BASES:
            previous = read(a.previous/f'trajectory_00_{base}/result.json')
            for row in previous['rows']:
                if row['seed'] in seeds and by_key[base,'original',0,row['seed']]['sha256'] != row['sha256']:
                    raise RuntimeError('New control differs from previous implementation')
    summary = dict(complete=True,source_sha256=sources,equivalence=equivalence,rows=summaries,
        timing_eligible=a.stage=='performance',scope='Same-budget exact outputs. Fair comparison applies identical pipeline flags to BF16 and FP4.',
        samples=[dict(row,base=b,variant=v,round=r,seed=s) for (b,v,r,s),row in by_key.items()])
    (a.output/f'{a.stage}_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    manifest['complete'] = True; save()


if __name__=='__main__':main()
