"""One resident model, serial independent scenes, bounded request caches."""
import argparse,gc,json
from pathlib import Path
from baselines import sha,write


def clear_case(runner):
    """Drop captured input-dependent state before a different prompt/image shape."""
    import torch
    runner.engine.clear()
    for key in ('attention_prepared','attention_shapes','shapes','trace_masks','pair_masks'):
        getattr(runner.engine,key).clear()
    runner.engine.mask_history.clear();runner.engine.budget_ledger=None;runner._warmed.clear()
    if runner.ar_cache:
        runner.ar_cache.active=None;runner.ar_cache.graphs.clear();runner.ar_cache.slots.clear();runner.ar_cache.seen.clear()
    gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);p.add_argument('--method',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();study=json.loads(a.manifest.read_text());spec=study['methods'][a.method]
    if not study['frozen']:raise ValueError('Freeze data and policies first')
    if spec.get('assets') and sha(Path(spec['assets'])/'manifest.json')!=spec['assets_sha256']:raise ValueError('Changed assets')
    if spec['config'].get('budget_plan') and sha(spec['config']['budget_plan'])!=spec['plan_sha256']:raise ValueError('Changed plan')
    adapter=spec.get('baseline_adapter')
    if adapter:
        implementations={'awq':'external.py','marlin_awq':'awq_marlin.py'}
        if adapter['method'] not in implementations:raise ValueError('Unsupported multi-scene baseline adapter')
        if spec.get('assets') or spec['config']['precision']!='bf16' or spec['config']['attention_backend']!='bf16':
            raise ValueError('AWQ reference requires BF16 dispatch/attention and separate AWQ assets')
        if sha(Path(adapter['assets'])/'manifest.json')!=adapter['assets_sha256'] or sha(Path(__file__).with_name(implementations[adapter['method']]))!=adapter['implementation_sha256']:
            raise ValueError('Changed AWQ assets or adapter implementation')
    a.output.mkdir(parents=True,exist_ok=True);identity=dict(manifest_sha256=sha(a.manifest),method=a.method,source_sha256=sha(__file__),smoke=a.smoke)
    progress=a.output/'progress.json'
    if progress.exists() and json.loads(progress.read_text())['identity']!=identity:raise ValueError('Changed resume identity')
    state=dict(identity=identity,complete=False,completed=[],performance_eligible=False);write(progress,state)
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    with shared_worker():
        resource=select_shared_device()
        import torch,numpy as np
        from thunderworld_fp4.cosmos import CosmosRunner,RunConfig
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        config=RunConfig.from_dict(spec['config'])
        if config.debug!='off' or config.resource_mode!='shared-calibration' or config.execution!='graph':raise ValueError('Changed execution conditions')
        runner=CosmosRunner(study['model'],config,assets=spec.get('assets'),ar_cache=True,first_frame=True).prepare()
        adaptation=None
        if adapter:
            if adapter['method']=='awq':from external import install_awq
            else:from awq_marlin import install as install_awq
            adaptation=install_awq(runner,Path(adapter['assets']))
        actual_precision=('NVFP4-AWQ-lite-W4A16' if adapter and adapter['method']=='awq' else adaptation['method']) if adaptation else None
        prepared=dict(identity=identity,resource=resource,runtime=runner.receipt())
        if adaptation:prepared.update(adapter=adaptation,actual_generation_precision=actual_precision,storage_comparison_eligible=adaptation.get('storage_comparison_eligible',False))
        write(a.output/'prepared.json',prepared)
        try:
            cases=study['cases']
            if a.smoke:
                first=study['cases'][0]
                second=next((entry for entry in study['cases'] if entry.get('benchmark')!=first.get('benchmark')),None)
                if second is None:raise ValueError('Smoke roundtrip needs two different dataset scenes')
                cases=[first,second]
            for entry in cases:
                case=entry['case'];folder=a.output/case['id'];folder.mkdir(exist_ok=True)
                path=folder/'result.json';case_identity=dict(**identity,case=case,image_sha256=entry['image_sha256'],seeds=entry['seeds'])
                if sha(case['image'])!=entry['image_sha256']:raise ValueError('Changed image')
                result=dict(identity=case_identity,complete=False,rows=[],performance_eligible=False)
                if adaptation:result.update(adapter=adaptation,actual_generation_precision=actual_precision,storage_comparison_eligible=adaptation.get('storage_comparison_eligible',False))
                if path.exists():
                    result=json.loads(path.read_text())
                    if result['identity']!=case_identity:raise ValueError('Changed case during resume')
                    if result['complete']:
                        state['completed'].append(case['id']);continue
                clear_case(runner);runner.warmup(case,study['steps'])
                if runner.engine.budget_ledger:
                    if not runner.engine.budget_ledger['feasible']:raise ValueError('Current scene violates compute cap')
                    write(folder/'compute.json',runner.engine.budget_ledger)
                for seed in entry['seeds']:
                    old=next((r for r in result['rows'] if r['seed']==seed),None)
                    if old:
                        if sha(folder/old['outputs'])!=old['outputs_sha256']:raise ValueError('Corrupt resumed output')
                        continue
                    signals,row=runner.infer(case,seed,study['steps']);filename=f'seed{seed}.npz'
                    np.savez_compressed(folder/filename,**signals)
                    ledger=row.pop('compute_ledger')
                    row.update(outputs=filename,outputs_sha256=sha(folder/filename),
                        compute_sha256=sha(folder/'compute.json') if ledger else None,
                        actual_mma_ratio=ledger['ratio'] if ledger else None)
                    result['rows'].append(row);write(path,result)
                result.update(complete=True,ar_cache=runner.ar_cache.receipt(),model_identity=runner.identity,targets=runner.engine.targets)
                write(path,result);state['completed'].append(case['id']);write(progress,state)
                print(json.dumps(dict(method=a.method,completed=len(state['completed']),total=len(study['cases']),case=case['id'])),flush=True)
            if a.smoke:
                import hashlib
                first=cases[0];clear_case(runner);runner.warmup(first['case'],study['steps'])
                signals,_=runner.infer(first['case'],first['seeds'][0],study['steps'])
                original=json.loads((a.output/first['case']['id']/'result.json').read_text())['rows'][0]['sha256']
                if {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in signals.items()}!=original:
                    raise AssertionError('A -> different prompt/image/shape -> A changed output')
                state['roundtrip_bitwise_equal']=True
            state['complete']=True;write(progress,state)
        except Exception as error:
            state.update(paused=True,error=repr(error));write(progress,state);raise
        finally:
            del runner;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
