"""Resume the existing Edge algorithm's independent Nano calibration pipeline.

Uses only calibration panels. No architecture/asset borrowing or final-set fitting.
The sequence preserves the original protected-N -> joint -> whole-FFN guard recipe.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from baselines import sha, write
from quality_benchmark import immutable

HERE = Path(__file__).resolve().parent


def legacy_panels(root):
    """Read-only manifest view of the exact full-M capture for official adapters."""
    capture = root/'capture'
    original = json.loads((capture/'manifest.json').read_text())
    if not original['complete'] or len(original['cases']) != 16:
        raise ValueError('Full sixteen-scene capture required')
    view = root/'baseline_capture'
    view.mkdir(exist_ok=True)
    value = copy.deepcopy(original)
    value.update(split='calibration', source_capture_sha256=sha(capture/'manifest.json'),
                 scope='Adapter schema view of the exact calibration-only full-M panels')
    value['weights'] = {n:e['file'] for n,e in original['weights'].items()}
    for row in value['rows']:
        row['kind'] = 'linear'
    for item in list(original['weights'].values())+original['rows']:
        src = capture/item['file']; dst = view/item['file']
        if sha(src) != item['sha256']:
            raise ValueError('Changed full-M panel/weight')
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(src)
        if dst.resolve() != src.resolve():
            raise ValueError('Changed adapter view')
    immutable(view/'manifest.json', value)
    return view


def arc_orders(root):
    import torch
    torch.set_num_threads(8)
    from portable import tensor_hash
    capture = root/'capture'
    manifest = json.loads((capture/'manifest.json').read_text())
    orders = {}
    for name in manifest['weights']:
        rows = sorted((r for r in manifest['rows'] if r['name'] == name), key=lambda r:r['scene_index'])
        if len(rows) != 16 or [r['scene_index'] for r in rows] != list(range(16)):
            raise ValueError('Incomplete ARC calibration')
        peak = None
        for row in rows:
            file = capture/row['file']
            if sha(file) != row['sha256']:
                raise ValueError('Changed ARC teacher input')
            x = torch.load(file, weights_only=True, map_location='cpu')['x']
            p = x.float().abs().amax(0)
            peak = p if peak is None else torch.maximum(peak, p)
        orders[name] = torch.argsort(peak, stable=True)
    out = root/'arc_orders'; out.mkdir(exist_ok=True)
    path = out/'orders.pt'
    if path.exists():
        old = torch.load(path, weights_only=True)
        if set(old) != set(orders) or any(not torch.equal(old[n],orders[n]) for n in orders):
            raise ValueError('Changed ARC channel ordering')
    else:
        torch.save(orders, path)
    immutable(out/'manifest.json',dict(schema='arc-channel-order-v1',complete=True,
        model_identity=manifest['model_identity'],panels_per_module=16,file=path.name,sha256=sha(path),
        tensor_hashes={n:tensor_hash(v) for n,v in orders.items()},capture_sha256=sha(capture/'manifest.json')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True,help='Nano preparation directory')
    p.add_argument('--workspace',type=Path,default=Path('/home/yufeng/thunderworld'))
    p.add_argument('--phase',choices=('repair','baselines','all'),default='all')
    p.add_argument('--awq-mode',choices=('reference','marlin'),default='reference')
    a = p.parse_args(); root=a.root.resolve(); root.mkdir(parents=True,exist_ok=True)
    statepath=root/('prepare_'+a.phase+'_'+a.awq_mode+'.json')
    state=dict(complete=False,phase=a.phase,source_sha256=sha(__file__),stages={},performance_eligible=False)
    if statepath.exists():
        state=json.loads(statepath.read_text())
        if state['source_sha256']!=sha(__file__):raise ValueError('Changed preparation driver')
    def stage(name, receipt, tool, *args):
        if receipt.exists() and json.loads(receipt.read_text()).get('complete'):
            if name in state['stages'] and state['stages'][name]!=sha(receipt):
                raise ValueError('Changed completed stage')
        else:
            with (root/(name+'.log')).open('a') as log:
                subprocess.run([sys.executable,str(HERE/tool),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)
        if not json.loads(receipt.read_text())['complete']:
            raise ValueError('Incomplete stage '+name)
        state['stages'][name]=sha(receipt);write(statepath,state)
        print(json.dumps(dict(stage=name,complete=True)),flush=True)
    try:
        stage('capture',root/'capture/manifest.json','importance_capture.py','--cohort',root/'cohort.json','--output',root/'capture')
        if a.phase in ('repair','all'):
            stage('calibration',root/'calibration/manifest.json','redesign_calibrate.py',root/'cohort.json','--output',root/'calibration')
            stage('ffn_calibration',root/'ffn_calibration/manifest.json','redesign_ffn_calibrate.py',
                  '--cohort',root/'cohort.json','--assets',root/'calibration/rtn_assets','--output',root/'ffn_calibration')
            template=root/'plan_source.json';immutable(template,dict(methods={},scope='No imported model-specific quotas'))
            stage('protected_plan',root/'protected_candidates/candidates.json','redesign_plan.py',
                  '--cohort',root/'cohort.json','--calibration',root/'calibration','--previous',template,
                  '--output',root/'protected_candidates','--skip-legacy-control','--budgets',1.5,2.)
            stage('ffn_plan',root/'ffn_candidates/candidates.json','redesign_plan.py',
                  '--cohort',root/'cohort.json','--calibration',root/'calibration','--previous',template,
                  '--output',root/'ffn_candidates','--ffn-calibration',root/'ffn_calibration',
                  '--skip-legacy-control','--budgets',2.5)
            # Adapt only catalog labels required by the existing coordinate-search ABI.
            protected=json.loads((root/'protected_candidates/candidates.json').read_text())
            if not (root/'joint_calibration/progress.json').exists():
                immutable(root/'candidates/candidates.json',protected)
            ffn=json.loads((root/'ffn_candidates/candidates.json').read_text())
            for candidate in ffn['candidates']:
                if candidate['name']=='b2.5_ffn_aware_n':candidate['name']='b2.5_ffn_aware'
            immutable(root/'ffn_followup/candidates/candidates.json',ffn)
            stage('priors',root/'priors/manifest.json','importance_priors.py','--previous',root,'--output',root/'priors')
            stage('joint',root/'joint_calibration/progress.json','importance_joint_calibrate.py','--study',root,'--previous',root)
            stage('ffn_guard',root/'ffn_guarded/progress.json','importance_ffn_guard.py',
                  '--study',root,'--previous',root,'--after',root/'joint_calibration/progress.json')
        if a.phase in ('baselines','all'):
            view=legacy_panels(root)
            if not (root/'arc_orders/manifest.json').exists():arc_orders(root)
            state['stages']['arc_orders']=sha(root/'arc_orders/manifest.json');write(statepath,state)
            sources=a.workspace/'contribution_baselines_20260917/sources.lock.json'
            stage('arc_probe',root/'arc_probe.json','arc_probe.py','--model','nano',
                  '--build',root/'arc_build','--output',root/'arc_probe.json')
            stage('sharq_probe',root/'sharq_probe.json','external_probe.py','--all-shapes',
                  '--capture',view,'--sources',sources,'--output',root/'sharq_probe.json')
            stage('awq_reference',root/'awq_reference/manifest.json','awq_prepare.py',
                  '--capture',view,'--sources',sources,'--output',root/'awq_reference','--resume')
            if a.awq_mode=='marlin':
                stage('awq_export',root/'awq_native/manifest.json','awq_marlin.py','export',
                      '--capture',view,'--reference',root/'awq_reference','--output',root/'awq_native')
                stage('awq_probe',root/'awq_probe/probe.json','awq_marlin.py','probe',
                      '--assets',root/'awq_native','--output',root/'awq_probe/probe.json')
            else:
                state['awq_execution']='Exact official decoded quality reference; native Marlin is a separate qualification'
        state.update(complete=True);state.pop('paused',None);state.pop('error',None)
    except Exception as error:
        state.update(paused=True,error=repr(error));raise
    finally:
        write(statepath,state)


if __name__=='__main__':main()
