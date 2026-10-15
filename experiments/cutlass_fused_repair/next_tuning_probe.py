"""Cumulative 1-4 tuning pilots; fresh runners, two seeds, exact-output gates."""
import argparse, gc, json
from pathlib import Path
import torch
from thunderworld_fp4.cosmos.runner import CosmosRunner
from thunderworld_fp4.cosmos.settings import RunConfig
from thunderworld_fp4.cosmos.cli import idle_guard
ROOT=Path(__file__).resolve().parents[2]
VARIANTS={
 'ar':{},
 'first':dict(first_frame=True),
 'codec':dict(first_frame=True,codec_graph=True),
 'norm':dict(first_frame=True,codec_graph=True,fuse_norm=True),
 'pointwise':dict(first_frame=True,codec_graph=True,fuse_norm=True,fuse_pointwise=True),
 'best':dict(first_frame=True,codec_graph=True,fuse_norm=True,fuse_pointwise=True,static_rotary=True,fuse_calibrated=True),
 'rotary':dict(first_frame=True,codec_graph=True,fuse_norm=True,fuse_pointwise=True,static_rotary=True),
}

def make(name,variant):
    base=ROOT/'runs5090/joint_e2e_20260914'
    return CosmosRunner(ROOT/'models/Cosmos3-Edge-a9d944e',RunConfig.from_dict(json.loads((base/f'configs/{name}.json').read_text())),
      tuning=base/f'profiles/{name}.json',ar_cache=True,**VARIANTS[variant],
      assets=None if name=='bf16' else ROOT/f'data_local/cutlass5090_rebuilt_20260914/i2v_b{name[-3:]}')

def case_input():
    cp=ROOT/'tools/joint_inference/cases/robot_observe.json';c=json.loads(cp.read_text());c['image']=str((cp.parent/c['image']).resolve());return c

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='joint125');p.add_argument('--changed-inputs',action='store_true');p.add_argument('--variants',nargs='+',default=list(VARIANTS));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    old=json.loads((ROOT/f'runs5090/joint_e2e_20260914/trajectory_00_{a.name}/result.json').read_text())
    refs={('original',r['seed']):r['sha256'] for r in old['rows']};result=dict(complete=False,name=a.name,variants={})
    def save():(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    for variant in a.variants:
        idle_guard();run=make(a.name,variant);run.prepare();case=case_input();run.warmup(case,35);run.warm_request(case,35)
        record=dict(flags=VARIANTS[variant],rows=[]);result['variants'][variant]=record
        cases=[('original',case,7103),('original',case,7104)]
        if a.changed_inputs:
            from PIL import Image,ImageOps
            changed=dict(case);changed['prompt']=case['prompt'].replace('robot','human')
            assert len(run.pipe.text_tokenizer(changed['prompt']).input_ids)==len(run.pipe.text_tokenizer(case['prompt']).input_ids)
            imagepath=a.output/'flipped_input.png'
            ImageOps.mirror(Image.open(case['image'])).save(imagepath)
            imagecase=dict(case,image=str(imagepath.resolve()))
            cases += [('changed_prompt',changed,7103),('changed_image',imagecase,7103),('original',case,7103)]
        for label,c,seed in cases:
            run.warmup(c,35)
            _,row=run.infer(c,seed,35);row['probe_case']=label;record['rows'].append(row)
            if variant=='ar':
                if (label,seed) in refs:assert row['sha256']==refs[label,seed]
                refs[label,seed]=row['sha256']
            record['exact']=row['sha256']==refs[label,seed];save()
            print(variant,seed,'seconds',row['wall_seconds'],'exact',record['exact'],flush=True)
            if not record['exact']:raise RuntimeError(f'{variant} changed the full trajectory at seed {seed}')
        record['receipt']=run.receipt();save();del run;gc.collect();torch.cuda.empty_cache()
    result['complete']=True;save()
if __name__=='__main__':main()
