"""Qualification and randomized original-vs-tuned precision baseline comparison."""
import argparse,hashlib,json,random,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs5090/baseline_compare_20260914'
SHARED=['--ar-cache','--first-frame','--codec-graph','--fuse-norm','--fuse-pointwise','--static-rotary','--fuse-calibrated']
NAMES=['bf16_original','fp8_original','fp4_original','bf16_tuned','fp8_tuned','fp4_tuned','thunder125','thunder150']

def read(p):return json.loads(p.read_text())
def command(name,dest,stage,*,fp4_backend='fp4'):
    family=name.split('_')[0];tuned=name.endswith('_tuned') or name.startswith('thunder')
    previous=ROOT/'runs5090/joint_e2e_20260914'
    if name.startswith('thunder'):
        old='joint125' if name=='thunder125' else 'cutlass150'
        config=previous/f'configs/{old}.json';profile=previous/f'profiles/{old}.json';assets=ROOT/f'data_local/cutlass5090_rebuilt_20260914/i2v_b{name[-3:]}'
    else:
        backend=fp4_backend if family=='fp4' and tuned else family
        config=OUT/f'configs/{backend}.json';profile=OUT/f'profiles/{backend}.json'
        assets=None if family=='bf16' else OUT/f'prepare_{family}/assets'
    cmd=[sys.executable,'-m','thunderworld_fp4.cosmos','benchmark' if stage=='performance' else 'infer','--model',str(ROOT/'models/Cosmos3-Edge-a9d944e'),
        '--config',str(config),'--tuning',str(profile),'--case',str(ROOT/'tools/joint_inference/cases/robot_observe.json'),'--output',str(dest),'--steps','35','--seeds',*(['93001','93002','93003'] if stage=='performance' else ['7103','7104'])]
    if assets:cmd+=['--assets',str(assets)]
    if tuned:cmd+=SHARED
    return cmd

def run(name,cmd,dest):
    if (dest/'result.json').exists():assert read(dest/'result.json')['complete'];return
    print('Running',name,flush=True)
    with dest.with_suffix('.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['qualification','performance'],required=True);a=p.parse_args()
    assert read(OUT/'setup_complete.json')['complete']
    if a.stage=='qualification':
        import importlib.metadata,importlib.util
        vroot=Path(importlib.util.find_spec('vllm').origin).parent
        metadata=dict(vllm_version=importlib.metadata.version('vllm'),vllm_native_binaries={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in vroot.glob('_C*.so')},
            custom_ops_sha256=hashlib.sha256((vroot/'_custom_ops.py').read_bytes()).hexdigest(),
            diffusers_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'data_local/cutlass_diffusers_20260914',text=True).strip())
        (OUT/'runtime_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        # Probe both native and CUTLASS no-repair paths before choosing an exact counterpart.
        jobs=[]
        for name in NAMES:
            dest=OUT/f'qualification_{name}';cmd=command(name,dest,a.stage);run(name,cmd,dest);jobs.append(dict(name=name,output=str(dest),command=cmd))
        dest=OUT/'qualification_fp4_cutlass';cmd=command('fp4_tuned',dest,a.stage,fp4_backend='fp4_cutlass');run('fp4_cutlass',cmd,dest)
        original=read(OUT/'qualification_fp4_original/result.json');candidate=read(dest/'result.json')
        equal=[a['sha256']==b['sha256'] for a,b in zip(original['rows'],candidate['rows'])]
        native=read(OUT/'qualification_fp4_tuned/result.json')
        # Independent pilot selects backend; formal data never selects its own winner.
        import statistics
        cutlass_ms=statistics.median(r['wall_seconds'] for r in candidate['rows']);native_ms=statistics.median(r['wall_seconds'] for r in native['rows'])
        chosen='fp4_cutlass' if all(equal) and cutlass_ms < native_ms*.97 else 'fp4'
        common=None
        for job in jobs:
            r=read(Path(job['output'])/'result.json');assert r['complete'] and not r['tuning_missing']
            signature={k:r[k] for k in ('model_identity','case','image_sha256','environment','source_sha256')}
            if common is not None:assert common==signature
            common=signature
        for family in ('bf16','fp8','fp4'):
            old=read(OUT/f'qualification_{family}_original/result.json');new=read(OUT/f'qualification_{family}_tuned/result.json')
            assert [r['sha256'] for r in old['rows']]==[r['sha256'] for r in new['rows']],family
            assert old['asset_identity']==new['asset_identity']
        for name,oldname in [('bf16_original','bf16'),('thunder125','joint125'),('thunder150','cutlass150')]:
            now=read(OUT/f'qualification_{name}/result.json');old=read(ROOT/f'runs5090/joint_e2e_20260914/trajectory_00_{oldname}/result.json')
            assert [r['sha256'] for r in now['rows']]==[r['sha256'] for r in old['rows']]
        (OUT/'qualification_summary.json').write_text(json.dumps(dict(complete=True,common=common,jobs=jobs,fp4_backend=chosen,cutlass_exact=equal,
            native_pilot_median_seconds=native_ms,cutlass_pilot_median_seconds=cutlass_ms,cutlass_probe=str(dest),scope='Two development seeds; exactness and independent backend pilot, not final timing'),indent=2)+'\n')
    else:
        gate=read(OUT/'qualification_summary.json');assert gate['complete'];rng=random.Random(91421);jobs=[]
        for repeat in range(5):
            names=NAMES.copy();rng.shuffle(names)
            for name in names:
                dest=OUT/f'performance_{repeat:02d}_{name}';cmd=command(name,dest,a.stage,fp4_backend=gate['fp4_backend'])
                jobs.append(dict(name=name,round=repeat,output=str(dest),command=cmd))
        matrix=dict(complete=False,random_order_seed=91421,jobs=jobs,common=gate['common'],fp4_backend=gate['fp4_backend'])
        mp=OUT/'performance_matrix.json';mp.write_text(json.dumps(matrix,indent=2)+'\n')
        for job in jobs:
            dest=Path(job['output']);run(job['name'],job['command'],dest)
            assert read(dest/'result.json')['source_sha256']==gate['common']['source_sha256']
        matrix['complete']=True;mp.write_text(json.dumps(matrix,indent=2)+'\n')
if __name__=='__main__':main()
