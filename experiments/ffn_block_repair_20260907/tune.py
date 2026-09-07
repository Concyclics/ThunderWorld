"""Bounded tuning: block size, producer reuse and M tile; no accuracy fitting."""
import json,os,statistics,random
import torch,triton,triton.testing
from safetensors import safe_open
from study import ROOT,D,load_state,quant,amask,wmask
from native import Native,gemm

@torch.inference_mode()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','3');torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    state=load_state();caps=torch.load(ROOT/'results/f4/cosmos3_nano_dev_activations_true_steps_cond_gpu0.pt',map_location='cpu',mmap=True,weights_only=True)['panels'];eva=torch.load(ROOT/'results/f4/cosmos3_nano_pai_robot026_dev_activations_true_steps_cond_gpu0.pt',map_location='cpu',mmap=True,weights_only=True)['panels']
    index=ROOT/'artifacts/huggingface/models/Cosmos3-Nano-411f42a/transformer/diffusion_pytorch_model.safetensors.index.json';idx=json.loads(index.read_text())['weight_map']
    report={'complete':False,'scope':'Shared GPU; tuning observed two local FFN shapes; no end-to-end speedup. Static AV-selected support reused for current robot activations.','rows':[]}
    for kind in ('gate','down'):
        name=f'layers.0.mlp_moe_gen.{kind}_proj'
        with safe_open(index.parent/idx[name+'.weight'],framework='pt',device='cpu') as sf:w=sf.get_tensor(name+'.weight').cuda()
        d=state[name+'.input_quantizer._pre_quant_scale'].cuda().float();wt=(w.float()/d).bfloat16().contiguous();wp,wc,ws,wg=quant(wt,torch.ones_like(d));w0,w1=wp
        cx=torch.cat([p[:128] for p in caps[name]]).cuda();ca,*_=quant(cx,d);act=(cx.float()*d).bfloat16().float();score=(act-ca[0]).square().mean(0)*w0.square().sum(0);wscore=(wt.float()-w0).square()*act.square().mean(0)
        for m in (128,512):
            x=eva[name][0][:128].cuda().repeat(m//128,1).contiguous();ap,*_=quant(x,d);ops={};checks={}
            plans=[(b,bm,inter,.5,0) for b in (128,256) for bm in (32,64,128) for inter in (False,True)]
            # Include matched full ACT2 baseline and small W repairs at fixed or extra budgets.
            plans += [(256,bm,True,pa,pw) for bm in (64,128) for pa,pw in [(0,0),(1,0),(.75,0),(.5,.0625),(.75,.0625),(.9375,.0625),(.6875,.0625)]]
            for b,bm,inter,pa,pw in plans:
                key=f'k{b}_m{bm}_'+('fused' if inter else 'indices')+f'_a{pa:g}_w{pw:g}'
                am=amask(score[None],b,pa).repeat(m//128,1);wm=wmask(wscore,b,pw)
                try:
                    op=Native(wc,ws,wg,d,am,wm,b,False,bm,inter);y=op(x).float()
                    ref=ap[0]@w0.T+(ap[1]*am.repeat_interleave(128,0))@w0.T+ap[0]@(w1*wm).T
                    error=float((y-ref).norm()/ref.norm());assert error<.004,(key,error)
                    ops[key]=op;checks[key]={'pass':True,'relative_l2':error,'weight_residual_bytes':op.residual_weight_bytes,'actual_a_fraction':float(am.float().mean()),'actual_w_fraction':float(wm.float().mean())}
                except Exception as exc:checks[key]={'pass':False,'error':str(exc)[:600]};print('failed',key,str(exc)[:150],flush=True)
            fns={key:lambda op=op:op(x) for key,op in ops.items()};fns['bf16']=lambda:torch.nn.functional.linear(x,w)
            ms={key:[] for key in fns};rng=random.Random(981)
            for _ in range(3):
                order=list(fns);rng.shuffle(order)
                for key in order:ms[key].append(triton.testing.do_bench_cudagraph(fns[key],rep=20))
            row={'module':name,'M':m,'checks':checks,'ms':{key:{'median':statistics.median(v),'runs':v} for key,v in ms.items()}}
            report['rows'].append(row);(D/'tuning.json').write_text(json.dumps(report,indent=2)+'\n')
            print(name,m,sorted([(k,round(v['median'],4)) for k,v in row['ms'].items()],key=lambda t:t[1])[:8],flush=True)
            del ops,fns,ap,x
        del w,wt,wp,wc,ws,w0,w1,cx,ca,act,wscore;torch.cuda.empty_cache()
    report['complete']=True;(D/'tuning.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
