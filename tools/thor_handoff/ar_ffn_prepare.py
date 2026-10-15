"""Freeze the FFN-only expansion study without touching previous experiments."""
import argparse
import copy
import json
from pathlib import Path
import shutil
from expand_build import build
from expand_worker import sha, write


NATIVE = ('bf16', 'fp8', 'fp4_base', 'b1.5_protected_n', 'b2_joint')
EXTENDED = [('bf16', 'ar_ffn4'), ('b1.5_protected_n', 'ar_ffn4'),
            ('b2_joint', 'ar_ffn4'), ('arcquant', 'trim'), ('sharq_perf', 'trim'),
            ('arcquant', 'ar_ffn4'), ('sharq_perf', 'ar_ffn4'),
            ('fp8', 'ar_ffn8'), ('fp4_base', 'ar_ffn1')]


def prepare_direct(root, prior):
    """Reuse the frozen controls and runtime; measure only the changed AR policy."""
    protocol = json.loads((prior/'protocol.json').read_text())
    campaign = json.loads((prior/'body-campaign.json').read_text())
    if not campaign.get('complete'):
        raise ValueError('The parent study must be complete before reuse')
    variant = json.loads((prior/'variant/variant.json').read_text())
    for name, digest in variant['variant_hashes'].items():
        if sha(prior/'variant/thunderworld_fp4'/name) != digest:
            raise ValueError('Changed parent runtime: '+name)
    root.mkdir(parents=True, exist_ok=False)
    for name in ('body-runs', 'manifests', 'assets'):
        (root/name).mkdir()
    # LaunchTable mixes resolved package roots with module __file__ paths;
    # a physical copy preserves source hashes without an ambiguous symlink root.
    shutil.copytree(prior/'variant', root/'variant',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    reused=[]
    for folder in sorted((prior/'body-runs').iterdir()):
        receipt=folder/'result.json'
        r=json.loads(receipt.read_text())
        if not r.get('complete') or not r.get('vae_fully_released') or 'RTX PRO 6000' not in r['device']:
            raise ValueError('Unqualified parent memory result: '+str(folder))
        (root/'body-runs'/folder.name).symlink_to(folder.resolve(), target_is_directory=True)
        reused.append(dict(name=folder.name, receipt=str(receipt), sha256=sha(receipt)))
    jobs=[]
    for model in ('edge','nano'):
        shutil.copy2(prior/'manifests'/(model+'.json'), root/'manifests'/(model+'.json'))
        # Assets are content-addressed; use a new namespace so the old study is immutable.
        (root/'assets'/model).mkdir()
        for asset in (prior/'assets'/model).glob('*.pt'):
            (root/'assets'/model/asset.name).symlink_to(asset.resolve())
        jobs.extend([model,method,'ar_ffn1'] for method in
                    ('b1.5_protected_n','b2_joint','arcquant','sharq_perf'))
    for model,method,profile in jobs:
        if (root/'body-runs'/f'{model}-{method}-{profile}').exists():
            raise ValueError('The requested direct-AR run is already present in the parent')
    write(root/'jobs.json', jobs)
    write(root/'protocol.json',dict(protocol,
        comparison_ar_fp4_profile='ar_ffn1', devices=['PRO6000'],
        quality_requested=False,quality_evaluation=False,performance_eligible=False,
        scope='PRO 6000 body memory only; direct W4A4 AR FFN, no AR repair',
        ar_policy=dict(ours='FFN-only W4A4, no repair',
            external='Frozen native DM baselines plus shared direct W4A4 AR FFN adaptation',
            fp8='FFN-only native W8A8',fp4_base='FFN-only W4A4',bf16='Unchanged BF16 control'),
        reused_prior=str(prior), parent_protocol_sha256=sha(prior/'protocol.json'),
        reused=reused,new_jobs=jobs))
    print(json.dumps(dict(root=str(root),jobs=len(jobs),reused=len(reused),ar_fp4_profile='ar_ffn1')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--workspace', type=Path)
    p.add_argument('--prior-root', type=Path)
    p.add_argument('--parent', type=Path)
    p.add_argument('--device', choices=('PRO6000', 'Thor'))
    p.add_argument('--direct-from', type=Path,
                   help='Reuse a completed PRO6000 study and measure direct W4A4 AR FFN for the four repaired DM methods')
    a = p.parse_args()
    if a.direct_from:
        prepare_direct(a.root, a.direct_from)
        return
    if not all((a.workspace,a.prior_root,a.parent,a.device)):
        p.error('Fresh preparation requires --workspace, --prior-root, --parent and --device')
    a.root.mkdir(parents=True, exist_ok=True)
    if (a.root / 'manifests').exists():
        raise FileExistsError('Use a fresh study root; frozen manifests are never overwritten')
    for folder in ('manifests', 'body-runs', 'runs'):
        (a.root / folder).mkdir()
    build(a.parent, a.root / 'variant')
    jobs, quality_jobs, reuse = [], [], []
    for model in ('edge', 'nano'):
        prior = a.prior_root / 'manifests' / (model + '.json')
        fixed = json.loads(prior.read_text())
        if a.device == 'PRO6000':
            source = a.workspace / 'full_quality_ready_20260919/edge/smoke/manifest.json'
            ext = json.loads(source.read_text())
            fixed['external'] = copy.deepcopy(ext['external'])
            fixed['adapters'] = {m: ext['adapters'][m] for m in ('arcquant', 'sharq_perf')}
            if model == 'nano':
                bank = a.workspace / 'full_quality_ready_20260919/nano'
                fixed['external'].update(arc_build=str(bank/'arc_build'),
                    arc_build_sha256=sha(bank/'arc_build/build.json'), capture=str(bank/'arc_orders'),
                    capture_sha256=sha(bank/'arc_orders/manifest.json'))
        else:
            source = a.workspace / 'comparisons/section54_fp4_retest_20260923' / model / 'fixed-model.json'
            ext = json.loads(source.read_text())
            fixed.update(baseline_platform='Thor', external_root=str(a.workspace),
                external_assets=ext['external_assets'], arc_build=ext['arc_build'],
                adapters=dict(arcquant=dict(method='arcquant', fraction=.5),
                              sharq_perf=dict(method='sharq', extra_fusion=True)))
        for method in ('arcquant', 'sharq_perf'):
            fixed['methods'][method] = copy.deepcopy(ext['methods'][method])
        for spec in fixed['methods'].values():
            if spec['config'].get('cutlass_library'):
                spec['config']['cutlass_library'] = str(a.root/'variant/cutlass_sm110.so')
                spec['config']['cutlass_small_library'] = None
        fixed.update(ar_ffn_parent_manifest_sha256=sha(prior),
            baseline_manifest_sha256=sha(source), study_scope='AR FFN only; common four-term AR extension for repair and external FP4 rows')
        write(a.root/'manifests'/(model+'.json'), fixed)
        for method in NATIVE:
            name = f'{model}-{method}-trim'
            folder = a.prior_root/'body-runs'/name
            receipt = json.loads((folder/'result.json').read_text())
            if not receipt['complete'] or not receipt['vae_fully_released']:
                raise ValueError('Unqualified reused run: '+str(folder))
            (a.root/'body-runs'/name).symlink_to(folder.resolve(), target_is_directory=True)
            reuse.append(dict(name=name, receipt=str(folder/'result.json'), sha256=sha(folder/'result.json')))
            quality_jobs.append([model, method, 'trim'])
        if a.device == 'PRO6000':
            (a.root/'runs'/f'{model}-bf16-original').symlink_to(
                (a.prior_root/'runs'/f'{model}-bf16-original').resolve(), target_is_directory=True)
        jobs.extend([model, method, profile] for method, profile in EXTENDED)
        quality_jobs.extend([model, method, profile] for method, profile in EXTENDED)
    write(a.root/'jobs.json', jobs)
    write(a.root/'quality-jobs.json', quality_jobs)
    write(a.root/'protocol.json', dict(device=a.device, performance_eligible=False,
        quality_evaluation=False, quality_requested=False, resource_mode='shared-calibration',
        cases_per_model=2, seeds=[301017,301018,301031], width=320,height=192,frames=61,steps=35,
        scope='BF16 VAE excluded; AR QKVO, embedding and auxiliary projections unchanged',
        budget_scope='1.5x/2x labels refer to frozen DM budgets; AR FFN work is additional',
        baseline_extension='ARC/SharQ original rows plus explicit common AR FP4-four-term adaptation; not upstream default',
        reused=reuse, jobs=jobs))
    print(json.dumps(dict(root=str(a.root), jobs=len(jobs), reused=len(reuse))))


if __name__ == '__main__':
    main()
