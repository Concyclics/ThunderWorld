"""Strict full-request precision comparison, paired speedups and CPU quality metrics."""
import argparse,hashlib,json,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
from baseline_compare import ROOT,OUT,NAMES,read

LABELS={'bf16_original':'Original BF16','fp8_original':'Original FP8','fp4_original':'Original FP4 (no repair)',
        'bf16_tuned':'Tuned BF16','fp8_tuned':'Tuned FP8','fp4_tuned':'Tuned FP4 (no repair)',
        'thunder125':'ThunderWorld 1.25','thunder150':'ThunderWorld 1.50'}

def performance():
    matrix=read(OUT/'performance_matrix.json');assert matrix['complete'];values=defaultdict(list);samples={};hashes={};fixed={};common=None;meta={}
    for job in matrix['jobs']:
        name=job['name'];r=read(Path(job['output'])/'result.json');assert r['complete'] and not r['tuning_missing']
        identity={k:r[k] for k in ('model_identity','case','image_sha256','environment','source_sha256')}
        if common is not None:assert identity==common
        common=identity;assert identity==matrix['common']
        config={k:r[k] for k in ('config','asset_identity','cutlass_artifact','observed_shapes')}
        if name in fixed:assert fixed[name]==config
        fixed[name]=config
        assert r['config']['strict_tuning'] and r['config']['execution']=='graph' and r['config']['debug']=='off'
        tuned=name.endswith('_tuned') or name.startswith('thunder')
        for flag in ('ar_cache','first_frame','codec_graph','norm_fusion','pointwise_fusion','static_rotary','calibrated_fusion'):assert r[flag]['enabled']==tuned,(name,flag)
        if tuned:assert r['ar_cache']['last_request_prefills']==2
        if name.startswith(('bf16','fp8')):
            assert r['config']['precision']==name.split('_')[0] and r['config']['attention_backend']=='bf16'
        elif name.startswith('fp4'):
            assert r['config']['precision']=='fp4' and r['config']['attention_backend']=='fp4'
            for part in ('ffn','projection'):
                assert r['config'][part]['activation']==0 and r['config'][part]['weight']==0 and not r['config'][part]['cross']
            assert r['config']['attention']['default']['coverage']==0 and not r['config']['attention']['rules']
            assert r['config']['budget_plan'] is None and r['config']['compute_budget'] is None
            assert all(x['actual_a']==0 and x['actual_w']==0 for x in r['observed_shapes'])
        assert len(r['rows'])==3
        meta[name]=dict(resident_model_state_bytes=r['resident_model_state_bytes'],linear_backend=('torch_bf16' if r['config']['precision']=='bf16' else 'vllm_native_cutlass_w8a8' if r['config']['precision']=='fp8' else r['config']['linear_backend']),asset_identity=r['asset_identity'])
        for row in r['rows']:
            assert row['performance_eligible'] and row['debug']=='off' and row['steps']==35 and row['calls']==70
            assert row['timing_scope']=='complete_request_v2_including_host_setup'
            family=name.split('_')[0] if not name.startswith('thunder') else name
            hk=family,row['seed']
            if hk in hashes:assert hashes[hk]==row['sha256'],'Non-determinism / tuning changed output'
            hashes[hk]=row['sha256'];key=name,job['round'],row['seed'];assert key not in samples
            samples[key]=row['wall_seconds'];values[name].append(row)
    assert len(samples)==120 and set(values)==set(NAMES)
    for family in ('bf16','fp8','fp4'):assert meta[family+'_original']['asset_identity']==meta[family+'_tuned']['asset_identity']
    for name in NAMES:
        assert {(r,s) for n,r,s in samples if n==name}=={(r,s) for r in range(5) for s in (93001,93002,93003)}
        family=name.split('_')[0] if not name.startswith('thunder') else name
        assert len({json.dumps(hashes[family,s],sort_keys=True) for s in (93001,93002,93003)})==3
    for name,oldname in [('bf16_tuned','bf16_best'),('thunder125','joint125_best'),('thunder150','cutlass150_best')]:
        previous=read(ROOT/f'runs5090/next_tuning_20260914/performance/performance_00_{oldname}/result.json')
        family=name.split('_')[0] if not name.startswith('thunder') else name
        for row in previous['rows']:assert hashes[family,row['seed']]==row['sha256'],'Changed previously validated output'
    medians={name:statistics.median(r['wall_seconds'] for r in values[name]) for name in NAMES};rng=np.random.default_rng(91421)
    rows=[]
    for name in NAMES:
        ratios={}
        for control in NAMES[:6]:
            v=np.array([statistics.median(samples[control,r,s]/samples[name,r,s] for s in (93001,93002,93003)) for r in range(5)])
            boots=np.median(v[rng.integers(0,5,(10000,5))],axis=1)
            ratios[control]=dict(ratio_of_medians=medians[control]/medians[name],paired_median=float(np.median(v)),ci95=np.quantile(boots,[.025,.975]).tolist())
        row=dict(name=name,label=LABELS[name],n=15,median_seconds=medians[name],p90_seconds=float(np.quantile([r['wall_seconds'] for r in values[name]],.9)),
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in values[name]),peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in values[name]),speedups=ratios,**meta[name]);rows.append(row)
    result=dict(complete=True,common=common,fp4_backend=matrix['fp4_backend'],rows=rows,samples=[dict(name=n,round=r,seed=s,seconds=t) for (n,r,s),t in samples.items()],
      scope='120 full resident requests: eight configurations, five randomized isolated process rounds, three seeds. Original baselines already use transformer graphs and host scheduler; recent pipeline flags disabled. Native FP8 DM projections/BF16 attention; plain FP4 has no repair. Same-family tuned outputs exact; cross-precision quality is separate.')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def quality():
    import torch,lpips
    from thunderworld_fp4.cosmos.quality import compare
    torch.set_num_threads(8);torch.hub.set_dir(str(ROOT/'data_local/perceptual_cache'))
    model=lpips.LPIPS(net='alex',version='0.1').eval();rows=[]
    for seed in (7103,7104):
        def load(name):
            with np.load(OUT/f'qualification_{name}/seed{seed}.npz') as f:return dict(f)
        ref=load('bf16_original')
        for name in ('fp8_original','fp4_original','thunder125','thunder150'):
            row=dict(name=name,seed=seed,metrics=compare(load(name),ref,model));rows.append(row);print(name,seed,row['metrics']['video'],flush=True)
    summary={name:{m:statistics.mean(r['metrics']['video'][m] for r in rows if r['name']==name) for m in ('lpips','ssim_mean','relative_l2','rmse')} for name in ('fp8_original','fp4_original','thunder125','thunder150')}
    result=dict(complete=True,rows=rows,means=summary,lpips_state_sha256={k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()},
        scope='Two development seeds on one I2V prompt/image, all output frames, CPU metrics outside timing. Paired BF16 consistency, not held-out quality or task success. Tuned baseline outputs equal originals.')
    (OUT/'quality.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['performance','quality'],required=True);a=p.parse_args()
    performance() if a.stage=='performance' else quality()
if __name__=='__main__':main()
