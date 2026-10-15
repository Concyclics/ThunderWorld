"""Evaluate previously selected 1.5x/2x linear strategies on the frozen 72 scenes."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from baselines import sha, write

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
CONTROL = WORK / 'fp4_attention_base_validation_20260922/expanded72/edge/final'
SELECTED = WORK / 'judge_full_i2v_20260919/edge/final/manifest.json'
NEW = {'selected_1p5': 'b1.5_protected_n', 'selected_2p0': 'b2_joint',
       'rtn_a_full': 'b1.5_protected_n', 'rtn_w_full': 'b1.5_protected_n',
       'joint_a_full': 'b2_joint', 'joint_w_full': 'b2_joint'}
FAMILIES = {n: ('activation' if '_a_' in n else 'weight' if '_w_' in n else 'bilateral') for n in NEW}
CAPTURE = WORK/'activation_importance_20260917/capture/manifest.json'


def side_plan(original, bank, family):
    if family not in ('activation', 'weight'):
        raise ValueError('Expected one-sided repair')
    plan = copy.deepcopy(original)
    plan['budget'] = 2.
    for key in plan['linears']:
        name = key.split('|', 2)[-1]
        n, k = bank['modules'][name]['shape']
        if n%128 or k%128:
            raise ValueError('Unsupported side-control tile')
        plan['linears'][key] = dict(a_blocks=k//128 if family=='activation' else 0,
            w_blocks=list(range((n//128)*(k//128))) if family=='weight' else [], cross=False)
    return plan


def prepare(root):
    from selected_linear_runtime import source_identity
    from thunderworld_fp4.cosmos import RunConfig
    old = json.loads((CONTROL/'manifest.json').read_text())
    selected = json.loads(SELECTED.read_text())
    if old['model'] != selected['model']:
        raise ValueError('Different model revisions')
    if (root/'study.json').exists():
        raise ValueError('Study already frozen; use run/resume')
    root.mkdir(parents=True, exist_ok=True)
    methods = {'bf16': copy.deepcopy(old['methods']['bf16'])}
    audits = {}
    for name, origin in NEW.items():
        spec = copy.deepcopy(selected['methods'][origin])
        source = Path(spec['config']['budget_plan'])
        plan = json.loads(source.read_text())
        source_hash = sha(source)
        if source_hash != spec['plan_sha256']:
            raise ValueError('Selected plan was changed')
        if sha(Path(spec['assets'])/'manifest.json') != spec['assets_sha256']:
            raise ValueError('Selected assets were changed')
        family = FAMILIES[name]
        if family != 'bilateral':
            bank = root/'full_control_assets'/origin
            full = json.loads((bank/'manifest.json').read_text())
            if not full['complete'] or full['selected_bank_sha256'] != spec['assets_sha256']:
                raise ValueError('Matching full W1 control bank not prepared')
            plan = side_plan(plan, full, family)
            spec.update(assets=str(bank), assets_sha256=sha(bank/'manifest.json'), budget=2.)
        plan['attention'] = {k: dict(qk_counts=[0]*16, pv_counts=[0]*16,
            qk_terms=2, pv_terms=2) for k in plan['attention']}
        plan['design'] = dict(original=plan['design'], original_plan=str(source),
            original_plan_sha256=source_hash, attention='All corrections off; same base attention kernel as controls',
            linear_policy='Frozen before this cohort; no refitting or redistribution of freed attention budget')
        path = root/'policies'/name/'plan.json'
        write(path, plan)
        spec.update(origin=origin, repair_family=family, runtime_options={}, attention_repair=False,
            selected_source_plan=str(source), selected_source_plan_sha256=source_hash,
            plan_sha256=sha(path),
            scope='Previously selected offline assets and exact linear supports/quotas, common uncorrected FP4 attention; other execution fusions disabled to match controls')
        spec['config'].update(budget_plan=str(path), debug='off', execution='graph',
            compute_budget=spec['budget'],
            resource_mode='shared-calibration', fuse_ffn=False, share_pack=False,
            linear_backend='triton', kernel_variant='optimized', joint_tuning=False,
            strict_tuning=False)
        spec['config']['attention'] = copy.deepcopy(old['methods']['linear_a']['config']['attention'])
        spec['config']['attention']['default'].update(coverage=0., terms=2, sparse_pack=False)
        spec['config']['attention']['rules'] = []
        RunConfig.from_dict(spec['config'])
        methods[name] = spec
        audits[name] = dict(origin=origin, assets_sha256=spec['assets_sha256'],
            original_plan_sha256=source_hash, derived_plan_sha256=sha(path),
            linear_entries=len(plan['linears']), cross=False,
            original_attention_removed=True, linear_policy_retained=True)
    study = copy.deepcopy(old)
    study.update(methods=methods, selected_implementation_lock=str(root/'implementation.lock.json'),
        role='fixed_selected_policy_followup_on_existing_cohort', selection_eligible=False,
        source_output_hashes=None, controls_manifest=str(CONTROL/'manifest.json'),
        controls_manifest_sha256=sha(CONTROL/'manifest.json'),
        selected_source_manifest=str(SELECTED), selected_source_manifest_sha256=sha(SELECTED),
        statistics=dict(unit='scene; average the two paired seeds first', bootstrap_seed=923001,
            replicates=10000, strata=['benchmark', 'category'],
            primary_comparisons=[[n,r] for n in ('selected_1p5','selected_2p0')
                for r in ('rtn_a_full','rtn_w_full','joint_a_full','joint_w_full')],
            intervals='95% paired scene intervals; Bonferroni familywise 95% over eight LPIPS comparisons',
            no_policy_fitting=True, followup_after_observing_fixed_quota_controls=True),
        interpretation='Empirical attainable quality over previously frozen representations and selected repair policies. Each bilateral method has exact same-W0 full-A/full-W controls. The cohort has already been inspected; not a new independent final test.')
    study['protocol']['cost'] = '1.5x/2x linear MMA caps; actual lower whole-request ratio also checked and reported'
    study['protocol']['scope'] = 'Selected production linear assets/selection, identical one-plane FP4 QK/PV and common BF16 AR/nonlinear execution'
    for key in ('attention_base_lock', 'sparse_support_directory', 'attention_ablation'):
        study.pop(key, None)
    write(root/'edge/final/manifest.json', study)
    write(root/'implementation.lock.json', dict(complete=True, sources=source_identity(),
        qualification='No new GPU kernels. Existing production linear kernels plus the qualified frozen one-plane expansion attention.'))
    write(root/'study.json', dict(frozen=True, methods=list(NEW), scenes=len(study['cases']),
        seeds=sum(len(e['seeds']) for e in study['cases']), new_videos=144*len(NEW),
        manifest_sha256=sha(root/'edge/final/manifest.json'), audits=audits,
        performance_eligible=False, no_test_set_fitting=True))
    # Reuse exactly identical reference signals, retaining the original receipt
    # and its hash in the new receipt; the old experiment is never rewritten.
    from importance_workflow import import_method
    import_method(CONTROL, root/'edge/final', 'bf16', root/'edge/final/manifest.json')
    return json.loads((root/'study.json').read_text())


def run(root, gpu):
    frozen = json.loads((root/'study.json').read_text())
    if frozen['manifest_sha256'] != sha(root/'edge/final/manifest.json'):
        raise ValueError('Changed frozen study')
    logs = root/'logs'; logs.mkdir(exist_ok=True)
    status = dict(complete=False, completed=['bf16'], active=None,
        performance_eligible=False, paused=False)
    try:
        for name in NEW:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                TW_SHARED_CALIBRATION_GPUS='0,1', TW_SHARED_CALIBRATION_LOCK_SCOPE='device',
                PYTHONUNBUFFERED='1')
            command = [sys.executable, str(Path(__file__).resolve()), 'worker',
                       '--root', str(root), '--method', name]
            with (logs/(name+'.log')).open('a') as stream:
                stream.write(json.dumps(dict(command=command, gpu=gpu, started=time.time()))+'\n')
                stream.flush()
                proc = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                status['active'] = dict(method=name, pid=proc.pid)
                write(root/'workflow.json', status)
                print(json.dumps(status['active']), flush=True)
                code = proc.wait()
            if code:
                raise RuntimeError(f'{name} exited {code}; results preserved for resume')
            for folder in (root/'edge/final/generation'/name).iterdir():
                if folder.is_dir() and not (folder/'generation.log').exists():
                    (folder/'generation.log').symlink_to(logs/(name+'.log'))
            status['completed'].append(name)
            write(root/'workflow.json', status)
            print(json.dumps(dict(finished=name)), flush=True)
        status.update(complete=True, active=None)
    except Exception as error:
        status.update(paused=True, error=repr(error))
        raise
    finally:
        write(root/'workflow.json', status)
        write(root/'edge/final/generation/suite.json', status)
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('assets','prepare','run','worker'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--gpu', type=int, choices=(0,1), default=1)
    p.add_argument('--method', choices=tuple(NEW))
    a = p.parse_args(); root = a.root.resolve()
    if a.operation == 'assets':
        from selected_linear_assets import restore
        result = restore(root, SELECTED, CAPTURE)
    elif a.operation == 'prepare':
        result = prepare(root)
    elif a.operation == 'run':
        result = run(root, a.gpu)
    else:
        if not a.method:
            p.error('worker needs --method')
        from selected_linear_runtime import install
        install()
        from dataset_infer import run as worker
        result = worker(root/'edge/final/manifest.json', a.method,
                        root/'edge/final/generation'/a.method)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
