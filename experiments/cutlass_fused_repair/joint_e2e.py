"""Frozen-budget Edge I2V: tune, verify trajectories, then isolated E2E timing.

Run through tools/kernel_handoff/python5090.sh with the pinned Diffusers checkout.
All timing jobs use strict profiles, debug=off, full request warmup, and idle guards.
The CLI benchmark timer includes conditioning and decoding, but excludes model
loading, offline quantization, compilation, graph capture, and NPZ serialization.
"""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
NAMES = ['bf16', 'cutlass125', 'joint125', 'cutlass150', 'joint150']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['tune', 'trajectory', 'performance', 'diagnostic'], required=True)
    p.add_argument('--model', type=Path, default=ROOT/'models/Cosmos3-Edge-a9d944e')
    p.add_argument('--policies', type=Path, default=ROOT/'data_local/cutlass_5090_download/ThunderWorld-CUTLASS-5090/policies')
    p.add_argument('--assets', type=Path, default=ROOT/'data_local/cutlass5090_rebuilt_20260914')
    p.add_argument('--library', type=Path, default=ROOT/'build/cutlass5090_20260914/repair_n128.so')
    p.add_argument('--small-library', type=Path, default=ROOT/'build/cutlass5090_20260914/repair_n64.so')
    p.add_argument('--case', type=Path, default=ROOT/'tools/joint_inference/cases/robot_observe.json')
    a = p.parse_args()
    a.output = a.output.resolve()
    if a.stage == 'performance':
        quality = json.loads((a.output/'trajectory_summary.json').read_text())
        if not quality['complete'] or len(quality['equivalence']) != 4 or not all(r['bitwise_equal'] for r in quality['equivalence']):
            raise ValueError('Verify both budgets on both development trajectories before performance')
    configs = a.output/'configs'; configs.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        if name == 'bf16':
            config = json.loads((ROOT/'tools/joint_inference/configs/bf16.json').read_text())
        else:
            policy = a.policies/f'i2v_b{name[-3:]}'
            config = json.loads((policy/'config.json').read_text())
            config.update(resource_mode='exclusive', budget_plan=str((policy/'plan.json').resolve()),
                linear_backend='cutlass-fused', cutlass_library=str(a.library.resolve()),
                cutlass_small_library=str(a.small_library.resolve()),
                joint_tuning=name.startswith('joint'), strict_tuning=True)
        path = configs/f'{name}.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError(f'Refusing to change existing configuration: {path}')
        path.write_text(json.dumps(config, indent=2)+'\n')
    profiles = a.output/'profiles'; profiles.mkdir(exist_ok=True)
    jobs = []; rng = random.Random(120)
    # Joint first; reuse identical attention keys while filling other profiles.
    tune_names = ['joint125', 'cutlass125', 'joint150', 'cutlass150', 'bf16']
    for repeat in range(5 if a.stage == 'performance' else 1):
        names = tune_names.copy() if a.stage == 'tune' else NAMES.copy()
        if a.stage == 'performance': rng.shuffle(names)
        for name in names:
            dest = a.output/(f'tune_{name}' if a.stage == 'tune' else f'{a.stage}_{repeat:02d}_{name}')
            command = [sys.executable, '-m', 'thunderworld_fp4.cosmos',
                {'tune':'warmup', 'trajectory':'infer', 'performance':'benchmark', 'diagnostic':'infer'}[a.stage],
                '--model', str(a.model.resolve()), '--config', str(configs/f'{name}.json'),
                '--case', str(a.case.resolve()), '--output', str(dest), '--steps', '35']
            if name != 'bf16':command += ['--assets', str((a.assets/f'i2v_b{name[-3:]}').resolve())]
            if a.stage == 'tune':command += ['--tune']
            else:
                command += ['--tuning', str(profiles/f'{name}.json'), '--seeds',
                    *({'trajectory':['7103','7104'], 'performance':['93001','93002','93003'], 'diagnostic':['6099','7103']}[a.stage])]
            if a.stage == 'diagnostic':command += ['--debug','timing']
            jobs.append(dict(config=name, round=repeat, output=str(dest), command=command))
    manifest = dict(complete=False, stage=a.stage, random_order_seed=120, jobs=jobs)
    manifest_path = a.output/f'{a.stage}_matrix.json'
    def save():manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    save(); previous = None
    for job in jobs:
        dest = Path(job['output']); result = dest/'result.json'
        if result.exists():
            if not json.loads(result.read_text()).get('complete'):
                raise RuntimeError(f'Inspect unfinished job before rerunning: {dest}')
            job['reused'] = True
        else:
            if a.stage == 'tune' and previous is not None:
                job['command'] += ['--tuning', str(previous)]
            print('Running', dest, flush=True)
            save()
            with dest.with_suffix('.log').open('w') as log:
                job['exit_code'] = subprocess.run(job['command'], cwd=ROOT,
                    stdout=log, stderr=subprocess.STDOUT).returncode
            save()
            if job['exit_code']:raise RuntimeError(f'Failed job: {dest}')
        if a.stage == 'tune':
            previous = profiles/f"{job['config']}.json"
            previous.write_bytes((dest/'tuning.json').read_bytes())
        save()
    manifest['complete'] = True; save()


if __name__ == '__main__':main()
