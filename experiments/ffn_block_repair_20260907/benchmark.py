"""Real FFN dimensions, sparse native correctness and graph timing."""
import json,os,random,statistics
from pathlib import Path
import torch,triton,triton.testing
from safetensors import safe_open
from study import ROOT,D,load_state,quant,amask,wmask,decode
from native import Native,gemm
from thunderworld_fp4 import FP4Linear,LinearRepairConfig

@torch.inference_mode()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','3')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    state=load_state();cal=torch.load(ROOT/'results/f4/cosmos3_nano_dev_activations_true_steps_cond_gpu0.pt',map_location='cpu',mmap=True,weights_only=True)['panels'];eva=torch.load(ROOT/'results/f4/cosmos3_nano_pai_robot026_dev_activations_true_steps_cond_gpu0.pt',map_location='cpu',mmap=True,weights_only=True)['panels']
    index=ROOT/'artifacts/huggingface/models/Cosmos3-Nano-411f42a/transformer/diffusion_pytorch_model.safetensors.index.json';idx=json.loads(index.read_text())['weight_map']
    report={'complete':False,'device':torch.cuda.get_device_name(),'scope':'shared GPU graph operator latency; online pack included, offline W prep excluded; M512 repeats captured rows for shape timing only','rows':[],'checks':[]}
    for kind in ('gate','down'):
        name=f'layers.0.mlp_moe_gen.{kind}_proj'
        with safe_open(index.parent/idx[name+'.weight'],framework='pt',device='cpu') as sf:w=sf.get_tensor(name+'.weight').cuda()
        d=state[name+'.input_quantizer._pre_quant_scale'].cuda().float();wt=(w.float()/d).bfloat16().contiguous()
        wp,wc,ws,wg=quant(wt,torch.ones_like(d));w0,w1=wp
        cx=torch.cat([x[:128] for x in cal[name]]).cuda();ca,*_=quant(cx,d);actual=(cx.float()*d).bfloat16().float()
        score=(actual-ca[0]).square().mean(0)*w0.square().sum(0);wscore=(wt.float()-w0).square()*actual.square().mean(0)
        for m in (128,512):
            x=eva[name][0][:128].cuda().repeat(m//128,1).contiguous();ap,*_=quant(x,d)
            methods={};ops={}
            plans=[(128,0,0,False),(128,.5,0,False),(128,.75,0,False),(128,1,0,False),(128,.5,.0625,False),(128,.75,.0625,False),(128,.5,.0625,True)]
            if m==128:plans.extend([(64,.5,0,False),(256,.5,0,False)])
            for b,pa,pw,cross in plans:
                am=amask(score[None],b,pa).repeat(m//128,1);wm=wmask(wscore,b,pw)
                key=f'k{b}_a{pa:g}_w{pw:g}'+('_cross' if cross else '')
                try:
                    op=Native(wc,ws,wg,d,am,wm,b,cross);y=op(x).float()
                    ma=am.repeat_interleave(128,0)
                    ref=ap[0]@w0.T+(ap[1]*ma)@w0.T+ap[0]@(w1*wm).T
                    if cross:ref+=(ap[1]*ma)@(w1*wm).T
                    err=float((y-ref).norm()/ref.norm());assert err<.004,(key,err)
                    report['checks'].append({'module':name,'M':m,'mode':key,'relative_l2_decoded_reference':err,'pass':True})
                    ops[key]=op;methods[key]=lambda op=op:op(x)
                except Exception as exc:
                    report['checks'].append({'module':name,'M':m,'mode':key,'pass':False,'error':str(exc)[:400]});print('unsupported',key,str(exc)[:200],flush=True)
            base=FP4Linear(wc[0],ws[0],wg,prescale=d,config=LinearRepairConfig(coverage=0))
            dynamic=FP4Linear(wc[0],ws[0],wg,prescale=d,config=LinearRepairConfig(coverage=.5))
            methods['bf16']=lambda:torch.nn.functional.linear(x,w)
            methods['package_base']=lambda:base(x)
            methods['package_dynamic_energy50']=lambda:dynamic(x)
            timings={key:[] for key in methods};rng=random.Random(711)
            for repeat in range(3):
                order=list(methods);rng.shuffle(order)
                for key in order:timings[key].append(triton.testing.do_bench_cudagraph(methods[key],rep=40))
            row={'module':name,'M':m,'N':w.shape[0],'K':w.shape[1],'ms':{key:{'median':statistics.median(v),'runs':v} for key,v in timings.items()},'residual_weight_bytes':{key:op.residual_weight_bytes for key,op in ops.items()}}
            report['rows'].append(row);(D/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n');print(name,m,{key:round(v['median'],4) for key,v in row['ms'].items()},flush=True)
            del methods,ops,base,dynamic,ap,x
        del w,wt,wp,wc,ws,w0,w1,cx,ca,actual,wscore;torch.cuda.empty_cache()
    compiled=list(gemm.device_caches[torch.cuda.current_device()][0].values())
    report['native_e2m1_mma']=all('mma.sync.aligned' in k.asm['ptx'] and 'e2m1' in k.asm['ptx'] for k in compiled)
    report['complete']=True;(D/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
