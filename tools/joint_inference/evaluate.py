"""Compute every requested modality and apply the paired FP8 gate per task."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from thunderworld_fp4.cosmos.quality import compare,gate


def main():
    p=argparse.ArgumentParser();p.add_argument('matrix',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--lpips-cache',type=Path)
    a=p.parse_args();m=json.loads(a.matrix.read_text())
    if not m['complete'] or m['suite'] not in ('quality','validation'):raise ValueError('Complete quality/validation matrix required')
    if a.output.exists():raise FileExistsError(a.output)
    if a.lpips_cache:torch.hub.set_dir(str(a.lpips_cache))
    import lpips
    metric=lpips.LPIPS(net='alex',version='0.1').eval()  # CPU; separate from all inference timings.
    torch.set_num_threads(8)
    jobs={(j['case'],j['config']):j for j in m['jobs']}
    result=dict(complete=False,model=m['model'],suite=m['suite'],rows=[],decisions=[],provenance={},
        matrix_sha256=hashlib.sha256(a.matrix.read_bytes()).hexdigest(),
        metric_definition='LPIPS alex v0.1; all-frame SSIM; waveform log-STFT 512/hop128; paired bootstrap 10000')
    result['lpips_state_sha256']={k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in metric.state_dict().items()}
    def save():a.output.write_text(json.dumps(result,indent=2)+'\n')
    for case in sorted(set(k[0] for k in jobs)):
        base=Path(jobs[case,'bf16']['output'])
        for (c,config),job in jobs.items():
            if c!=case or config=='bf16':continue
            folder=Path(job['output']);run=json.loads((folder/'result.json').read_text())
            bf=json.loads((base/'result.json').read_text())
            if bf['config']['precision']!='bf16' or bf['config']['attention_backend']!='bf16':
                raise ValueError('BF16 control is not native BF16')
            if config=='fp8' and (run['config']['precision']!='fp8' or run['config']['attention_backend']!='bf16'):
                raise ValueError('FP8 control must be W8A8 projections plus BF16 attention')
            if run['config']['execution']!=bf['config']['execution']:raise ValueError('Execution mode mismatch')
            result['provenance'][case+'|'+config]={k:run[k] for k in ('config','model_identity','targets')}
            result['provenance'][case+'|'+config].update({k:run.get(k) for k in ('asset_identity','ffn_policy_identity')})
            for key in ('model_identity','targets','case'):
                if run[key]!=bf[key]:raise ValueError(f'Unmatched baseline {key}')
            for row in run['rows']:
                if m['suite']=='quality' and row['steps']!=35:raise ValueError('Final quality requires 35 steps')
                with np.load(folder/row['outputs']) as z:signals=dict(z)
                reference_row=next(x for x in bf['rows'] if x['seed']==row['seed'])
                if row['steps']!=reference_row['steps']:raise ValueError('Step mismatch')
                with np.load(base/reference_row['outputs']) as z:reference=dict(z)
                result['rows'].append(dict(case=case,task=row['task'],config=config,seed=row['seed'],
                    metrics=compare(signals,reference,metric)))
                save()
        fp8=[r for r in result['rows'] if r['case']==case and r['config']=='fp8']
        for config in sorted(set(r['config'] for r in result['rows'] if r['case']==case)-{'fp8'}):
            candidate=[r for r in result['rows'] if r['case']==case and r['config']==config]
            result['decisions'].append(dict(case=case,config=config,**gate(candidate,fp8,candidate[0]['task'],final=m['suite']=='quality')))
    result['complete']=True;save()


if __name__=='__main__':main()
