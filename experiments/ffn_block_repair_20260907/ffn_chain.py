"""Full SiLU(gate)*up -> down replay with each candidate's own activations."""
import json,os,math
import torch
from safetensors import safe_open
from study import ROOT,D,load_state,quant,amask,wmask,metric
from native import Native

@torch.inference_mode()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','3');torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    state=load_state();cal=torch.load(ROOT/'results/f4/cosmos3_nano_dev_activations_true_steps_cond_gpu0.pt',map_location='cpu',mmap=True,weights_only=True)['panels'];eva=torch.load(ROOT/'results/f4/cosmos3_nano_pai_robot026_dev_activations_true_steps_cond_gpu0.pt',map_location='cpu',mmap=True,weights_only=True)['panels']
    index=ROOT/'artifacts/huggingface/models/Cosmos3-Nano-411f42a/transformer/diffusion_pytorch_model.safetensors.index.json';idx=json.loads(index.read_text())['weight_map']
    plans={'base':(128,0,0),'full_a':(128,1,0),'k128_a50':(128,.5,0),'k128_a75':(128,.75,0),'k128_a75_w6':(128,.75,.0625),'k128_a93_w6':(128,.9375,.0625),'k256_a50':(256,.5,0),'k256_a75':(256,.75,0),'k256_a75_w6':(256,.75,.0625),'k256_a93_w6':(256,.9375,.0625)}
    report={'complete':False,'scope':'Full local FFN chain on layers 0/24, robot steps 0/17/34,128 rows; frozen AV channel support; candidate down input recomputed from candidate gate/up; no full diffusion rollout','plans':plans,'rows':[]}
    for layer in (0,24):
        raw={};ops={key:{} for key in plans}
        for kind in ('gate','up','down'):
            name=f'layers.{layer}.mlp_moe_gen.{kind}_proj'
            with safe_open(index.parent/idx[name+'.weight'],framework='pt',device='cpu') as sf:w=sf.get_tensor(name+'.weight').cuda()
            raw[kind]=w;d=state[name+'.input_quantizer._pre_quant_scale'].cuda().float();wt=(w.float()/d).bfloat16().contiguous();wp,wc,ws,wg=quant(wt,torch.ones_like(d));w0,w1=wp
            x=torch.cat([p[:128] for p in cal[name]]).cuda();ap,*_=quant(x,d);actual=(x.float()*d).bfloat16().float();score=(actual-ap[0]).square().mean(0)*w0.square().sum(0);wscore=(wt.float()-w0).square()*actual.square().mean(0)
            for key,(b,pa,pw) in plans.items():
                ops[key][kind]=Native(wc,ws,wg,d,amask(score[None],b,pa),wmask(wscore,b,pw),b,False,32 if b==128 else 64,True)
            del wp,wc,ws,w0,w1,x,ap,actual,wscore
        def run(x,bank):
            gate=bank['gate'](x);up=bank['up'](x)
            return bank['down'](torch.nn.functional.silu(gate)*up)
        for i,step in enumerate((0,17,34)):
            x=eva[f'layers.{layer}.mlp_moe_gen.gate_proj'][i][:128].cuda()
            assert torch.equal(x.cpu(),eva[f'layers.{layer}.mlp_moe_gen.up_proj'][i][:128])
            z=torch.nn.functional.silu(torch.nn.functional.linear(x,raw['gate']))*torch.nn.functional.linear(x,raw['up'])
            ref=torch.nn.functional.linear(z,raw['down']).float()
            cap=eva[f'layers.{layer}.mlp_moe_gen.down_proj'][i][:128].cuda()
            row={'layer':layer,'step':step,'bf16_internal_vs_captured_relative_l2':float((z.float()-cap.float()).norm()/cap.float().norm()),'metrics':{key:metric(run(x,bank).float(),ref) for key,bank in ops.items()}}
            report['rows'].append(row)
        print(layer,'complete',flush=True);del raw,ops;torch.cuda.empty_cache()
    report['summary']={key:math.sqrt(sum(r['metrics'][key]['sse'] for r in report['rows'])/sum(r['metrics'][key]['energy'] for r in report['rows'])) for key in plans}
    report['complete']=True;(D/'ffn_chain.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['summary'],indent=2),flush=True)
if __name__=='__main__':main()
