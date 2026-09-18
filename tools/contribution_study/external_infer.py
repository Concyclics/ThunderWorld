"""Full current-input baseline generation through the existing Cosmos runner."""
import argparse,json,hashlib,os
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('model','case','sources','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--method',choices=('bf16','fp8','sharq','sharq_sim','arcquant','attention_uniform','attention_marginal','awq','ar3','ar4',
        'joint_budget','fp4_base','fp4_three','fp4_four'),required=True)
    p.add_argument('--runtime-config',type=Path)
    p.add_argument('--assets',type=Path)
    p.add_argument('--label')
    p.add_argument('--arc-build',type=Path)
    p.add_argument('--capture',type=Path)
    p.add_argument('--arc-fraction',type=float,default=.5)
    p.add_argument('--attention-study',type=Path)
    p.add_argument('--awq-assets',type=Path)
    p.add_argument('--seeds',nargs='+',type=int,default=[7103,7104])
    p.add_argument('--cohort',type=Path)
    p.add_argument('--split',choices=('trajectory','validation','final'),default='trajectory')
    p.add_argument('--frozen-selection',type=Path)
    p.add_argument('--execution',choices=('graph','eager'),default='eager')
    a=p.parse_args()
    from protocol import admit_run
    identity=dict(method=a.method,model=str(a.model.resolve()),case_sha256=sha(a.case),
        execution=a.execution,arc_fraction=a.arc_fraction,source_lock_sha256=sha(a.sources),
        attention_study_sha256=sha(a.attention_study) if a.attention_study else None,
        arc_build_sha256=sha(a.arc_build/'build.json') if a.arc_build else None,
        awq_assets_sha256=sha(a.awq_assets/'manifest.json') if a.awq_assets else None,
        capture_sha256=sha(a.capture/'manifest.json') if a.capture else None)
    if a.runtime_config:
        runtime=json.loads(a.runtime_config.read_text())
        identity.update(runtime_config_sha256=sha(a.runtime_config),budget_plan_sha256=sha(runtime['budget_plan']),
            assets_sha256=sha(a.assets/'manifest.json'))
    if a.method=='sharq':
        source=json.loads(a.sources.read_text())['sources']['sharq']
        library=Path(os.environ.get('SHARQ_KERNEL_BUILD',str(Path(source['path'])/'kernels/build_cmake_sm120a')))/'sharq_ops.so'
        identity['sharq_library_sha256']=sha(library)
    identity_sha256=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    admission=admit_run(a.cohort,a.split,a.seeds,a.label or a.method,a.frozen_selection,identity_sha256)
    a.output.mkdir(parents=True,exist_ok=False)
    case=json.loads(a.case.read_text())
    if case.get('image'):case['image']=str((a.case.parent/case['image']).resolve())
    lock=json.loads(a.sources.read_text())
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    with shared_worker():
        resource=select_shared_device()
        import torch,numpy as np
        from thunderworld_fp4.cosmos import CosmosRunner,RunConfig
        from external import install_sharq,install_arc
        config=RunConfig(precision='fp8' if a.method=='fp8' else 'bf16',attention_backend='bf16',
            execution=a.execution,debug='off',resource_mode='shared-calibration')
        if a.method=='joint_budget':
            from dataclasses import replace
            if a.runtime_config is None or a.assets is None:raise ValueError('Joint candidate needs config and assets')
            config=replace(RunConfig.from_dict(runtime),resource_mode='shared-calibration',execution=a.execution,debug='off')
            if config.precision!='fp4' or config.attention_backend!='fp4' or not config.budget_plan:
                raise ValueError('Joint budget must execute native FP4 projections and attention')
        elif a.method.startswith('fp4_'):
            from thunderworld_fp4 import RepairConfig,RepairPolicy
            from thunderworld_fp4.cosmos.settings import LinearSpec
            coverage=0. if a.method=='fp4_base' else 1.;terms=4 if a.method=='fp4_four' else 3
            config=RunConfig(precision='fp4',attention_backend='fp4',execution=a.execution,debug='off',resource_mode='shared-calibration',
                ffn=LinearSpec(coverage,coverage,terms==4),projection=LinearSpec(coverage,coverage,terms==4),
                attention=RepairPolicy(RepairConfig(coverage=coverage,terms=terms,query_tile=32,transpose_v=True,sparse_pack=True)))
        if a.method.startswith('attention_'):
            from dataclasses import replace
            from thunderworld_fp4 import RepairConfig,RepairPolicy,RepairRule
            from head_allocation import allocate_head_quotas
            if a.attention_study is None:raise ValueError('Attention study required')
            study=json.loads(a.attention_study.read_text())
            if not study['complete'] or len(study['layers'])!=28:raise ValueError('All-layer attention study required')
            common=dict(query_tile=32,transpose_v=True,sparse_pack=True,terms=3)
            default=RepairConfig(coverage=.5,**common);rules=[];phases=(0,7,18,28,35)
            if a.method=='attention_marginal':
                for layer,data in study['layers'].items():
                    for row in data['interventions']:
                        if row['seed']!=7101:continue
                        losses=row['measured_quota_losses'];nt=len(losses[0])-1
                        quotas=allocate_head_quotas(losses,len(losses)*((nt+1)//2))
                        cfg=RepairConfig(coverage=0.,head_quotas=quotas,**common)
                        phase=row['phase']
                        rules.append(RepairRule(config=cfg,layers=(int(layer),),
                            steps=(phases[phase],phases[phase+1]),branches=(str(row['branch']),)))
            config=replace(config,attention_backend='fp4',attention=RepairPolicy(default,tuple(rules)),attention_only_attribution=True)
        runner=CosmosRunner(a.model,config,assets=a.assets,ar_cache=True,first_frame=True).prepare()
        adaptation=None
        if a.method.startswith('sharq'):
            entry=lock['sources']['sharq']
            adaptation=install_sharq(runner,entry['path'],entry['commit'],
                'SHARQ_SIM' if a.method=='sharq_sim' else 'SHARQ')
            if a.method=='sharq' and adaptation['library_sha256']!=identity['sharq_library_sha256']:
                raise ValueError('Loaded a different SharQ binary than the frozen configuration')
        elif a.method=='arcquant':
            if a.execution!='eager' or a.arc_build is None or a.capture is None:
                raise ValueError('ARC requires eager, --arc-build and --capture')
            adaptation=install_arc(runner,a.arc_build,a.capture,a.arc_fraction)
        elif a.method.startswith('attention_'):
            adaptation=dict(method=a.method,scope='Attention-only attribution: projections explicitly BF16; not a joint FP4 candidate',
                attention_study_sha256=sha(a.attention_study),selection_seed=7101,
                policy=config.to_dict()['attention'],performance_eligible=False)
        elif a.method=='awq':
            from external import install_awq
            if a.awq_assets is None:raise ValueError('--awq-assets required')
            adaptation=install_awq(runner,a.awq_assets)
        elif a.method in ('ar3','ar4'):
            from external import install_ar_repair
            adaptation=install_ar_repair(runner,int(a.method[-1]))
        receipt=dict(complete=False,method=a.method,actual_generation_precision=a.method,
            adapter=adaptation,resource=resource,performance_eligible=False,rows=[],
            case=case,model_identity=runner.identity,targets=runner.engine.targets,
            image_sha256=sha(case['image']) if case.get('image') else None,
            source_lock_sha256=sha(a.sources),
            admission=admission,implementation_sha256={f:sha(Path(__file__).parent/f) for f in
                ('external_infer.py','external.py','head_allocation.py','protocol.py')},
            configuration=identity,configuration_sha256=identity_sha256,
            label=a.label or a.method,
            scope='Baseline adaptation; unchanged AR cache. Attention policy is explicitly recorded. '
                  'Host engine precision is a dispatch placeholder when external linears are installed; actual precision is recorded separately.')
        target=a.output/'result.json';write(target,receipt)
        try:
            runner.warmup(case,35)
            if hasattr(runner,'_contribution_ar_controller'):runner._contribution_ar_controller.warming=False
            for seed in a.seeds:
                signals,row=runner.infer(case,seed,35)
                filename=f'seed{seed}.npz';np.savez_compressed(a.output/filename,**signals)
                row.update(outputs=filename,outputs_sha256=sha(a.output/filename))
                receipt['rows'].append(row);write(target,receipt)
            receipt.update(complete=True,runtime=runner.receipt());write(target,receipt)
        except Exception as error:
            receipt.update(error=repr(error),paused=True);write(target,receipt);raise


if __name__=='__main__':main()
