"""Portable FFN/attention benchmark: no model loader or source-server paths.

Correctness runs before timing. A frozen baseline copy plus this entrypoint can
be profiled and changed by the receiving Codex. Data/model preparation is outside
the timed region. Use --inner-only to isolate packed FFN GEMM, never to claim
end-to-end acceleration. Larger row/key counts repeat captured data.
"""
import argparse,hashlib,importlib.util,json,math,os,random,statistics,sys
from pathlib import Path
import torch,triton,triton.testing
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'packages/fp4-repair/src'))
from thunderworld_fp4 import RepairConfig,fp4_attention,preset,policy_from_head_atlas

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def decode(c,s):
    ids=torch.stack((c&15,c>>4),-1).flatten(-2).long();lut=torch.tensor([0,.5,1,1.5,2,3,4,6],device=c.device)
    return lut[ids&7]*torch.where(ids>=8,-1,1)*s.float().repeat_interleave(16,-1)

def top_mask(score,ratio):
    mask=torch.zeros_like(score.flatten(),dtype=torch.bool);count=math.ceil(mask.numel()*ratio)
    if count:mask[score.flatten().argsort(descending=True,stable=True)[:count]]=True
    return mask.reshape(score.shape)

def repeat_rows(x,n,dim):
    repeats=[1]*x.ndim;repeats[dim]=math.ceil(n/x.shape[dim]);return x.repeat(*repeats).narrow(dim,0,n).contiguous()

FFN={'base':(0,0),'a50':(.5,0),'a75':(.75,0),'full_act2':(1,0),'a75_w6':(.75,.0625),'a93_w6':(.9375,.0625)}

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--asset',required=True);p.add_argument('--weights',type=Path);p.add_argument('--kind',choices=['ffn','attention'],required=True);p.add_argument('--preset',required=True);p.add_argument('--m',type=int,default=128);p.add_argument('--step-index',type=int,default=0);p.add_argument('--block-k',type=int,choices=[64,128,256],default=128);p.add_argument('--tile-m',type=int,choices=[16,32,64,128],default=32);p.add_argument('--splits',type=int,choices=[1,2,4,8],default=2);p.add_argument('--key-tokens',type=int);p.add_argument('--rounds',type=int,default=5);p.add_argument('--rep-ms',type=int,default=40);p.add_argument('--inner-only',action='store_true');p.add_argument('--profile',action='store_true');p.add_argument('--skip-verify',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count()!=1:raise RuntimeError('Select one authorized GPU explicitly')
    if torch.cuda.get_device_capability()!=(12,0):raise RuntimeError('SM120 required')
    if min(a.m,a.rounds,a.rep_ms)<=0 or (a.key_tokens is not None and a.key_tokens<=0):raise ValueError('Sizes and timing counts must be positive')
    if a.output.exists():raise FileExistsError('Use a fresh output path to preserve baselines')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    f=(a.data/a.asset).resolve()
    if not f.is_relative_to(a.data.resolve()):raise ValueError('Asset must be inside the bundle')
    payload=torch.load(f,weights_only=True,map_location='cpu')
    out={'config':vars(a).copy(),'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'triton':triton.__version__,'source_sha256':{},'scope':'shared captured values repeated for larger shapes; one operator, not full model','verification':None}
    out['config']={k:str(v) if isinstance(v,Path) else v for k,v in out['config'].items()}
    source=[Path(__file__),ROOT/'experiments/ffn_block_repair_20260907/native.py',*sorted((ROOT/'packages/fp4-repair/src/thunderworld_fp4').glob('*.py'))]
    out['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source}
    if a.kind=='ffn':
        if a.preset not in FFN or a.tile_m not in (32,64,128):raise ValueError('FFN preset/tile invalid')
        if not 0<=a.step_index<len(payload['eval_inputs']):raise ValueError('Invalid step index')
        if a.weights is None:raise ValueError('FFN requires --weights pointing to receiver-prepared weights')
        ready=torch.load(a.weights/a.asset,weights_only=True,map_location='cpu')
        if ready['module']!=payload['module'] or ready['input_sha256']!=hashlib.sha256(f.read_bytes()).hexdigest():raise ValueError('Prepared weights do not match this input/calibration asset')
        for key in ('weight_bf16','codes','scales','global_scale'):payload[key]=ready[key]
        del ready
        native=load_module('handoff_native',ROOT/'experiments/ffn_block_repair_20260907/native.py')
        x=repeat_rows(payload['eval_inputs'][a.step_index].cuda(),a.m,0);w=payload['weight_bf16'].cuda();c=payload['codes'].cuda();s=payload['scales'].cuda();g=payload['global_scale'].cuda();d=payload['prescale'].cuda();n,k=w.shape;pa,pw=FFN[a.preset]
        small_a=top_mask(payload['activation_score'].reshape(-1,a.block_k).sum(-1),pa)
        am=small_a.repeat_interleave(a.block_k)[None].repeat(math.ceil(a.m/128),1).cuda()
        small_w=top_mask(payload['weight_block_scores'][str(a.block_k)],pw)
        wm=small_w.repeat_interleave(128,0).repeat_interleave(a.block_k,1).cuda()
        op=native.Native(c,s,g,d,am,wm,a.block_k,False,a.tile_m,True)
        fn=lambda:op(x)
        if a.inner_only:
            packed=op.pack(x);fn=lambda:op.run_packed(x,packed)
        bf16=lambda:torch.nn.functional.linear(x,w)
        if not a.skip_verify:
            from thunderworld_fp4.linear import _pack
            ac,sc,ag,_=_pack(x,d,True,True,x);ap=decode(ac,sc)[:,:a.m]*ag;wp=decode(c,s)*g
            ma=am.repeat_interleave(128,0)[:a.m]
            ref=ap[0]@wp[0].T+(ap[1]*ma)@wp[0].T+ap[0]@(wp[1]*wm).T
            y=fn().float();err=float((y-ref).norm()/ref.norm());assert torch.isfinite(y).all() and err<.004,err
            baseline=bf16().float();out['verification']={'decoded_reference_relative_l2':err,'bf16_relative_l2':float((y-baseline).norm()/baseline.norm())}
            del ac,sc,ap,wp,ma,ref,y,baseline
        frac_a=float(small_a.float().mean());frac_w=float(small_w.float().mean())
        out.update(shape={'M':a.m,'N':n,'K':k},actual_activation_coverage=frac_a,actual_weight_coverage=frac_w,residual_weight_bytes=op.residual_weight_bytes)
        base_flops=2*a.m*n*k;multiplier=1+frac_a+frac_w
        # Constructor owns W0/compact W1. Do not retain full W1 preparation planes.
        del c,s,am,wm
    else:
        if a.inner_only:raise ValueError('--inner-only applies to FFN; attention stage isolation requires a kernel-level probe')
        q=repeat_rows(payload['q'].cuda(),a.m,1);k=payload['k'].cuda();v=payload['v'].cuda()
        if a.key_tokens:k=repeat_rows(k,a.key_tokens,1);v=repeat_rows(v,a.key_tokens,1)
        from dataclasses import replace
        if a.preset=='head_guided':
            atlas=json.loads((a.data/'head_atlas.json').read_text());cfg=policy_from_head_atlas(atlas,coverage=.25,protected_coverage=.75).resolve(layer=payload['layer'])
        else:cfg=preset(a.preset)
        cfg=replace(cfg,query_tile=a.tile_m,splits=a.splits)
        # Quotas are immutable configuration metadata, prepared outside capture.
        # Selection and all current Q/K/V quantization stay inside the timed call.
        from thunderworld_fp4.runtime import _forward
        quotas=torch.tensor(cfg.head_counts(q.shape[2],k.shape[1]),device=q.device,dtype=torch.int32)
        fn=lambda:_forward(q,k,v,cfg,counts_tensor=quotas)[0]
        ratio=q.shape[2]//k.shape[2];kk=k.repeat_interleave(ratio,dim=2).transpose(1,2).contiguous();vv=v.repeat_interleave(ratio,dim=2).transpose(1,2).contiguous();qq=q.transpose(1,2)
        def bf16():
            from torch.nn.attention import sdpa_kernel,SDPBackend
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):return torch.nn.functional.scaled_dot_product_attention(qq,kk,vv).transpose(1,2)
        if not a.skip_verify:
            oracle=load_module('handoff_oracle',ROOT/'packages/fp4-repair/tests/oracle.py')
            y,mask=fp4_attention(q,k,v,cfg,return_mask=True)
            repair=any(cfg.head_counts(q.shape[2],k.shape[1]))
            ref=oracle.reference(q,k,v,mask.int()*2+1,repair,cfg.terms,cfg.splits)
            err=float((y.float()-ref).norm()/ref.norm());assert torch.isfinite(y).all() and err<.012,err
            baseline=bf16().float();out['verification']={'decoded_reference_relative_l2':err,'bf16_relative_l2':float((y.float()-baseline).norm()/baseline.norm())};del ref,y,mask,baseline
        out.update(shape={'M':a.m,'N':k.shape[1],'Hq':q.shape[2],'Hkv':k.shape[2],'D':128},repair_config=cfg.to_dict(),cost=cfg.cost(q.shape[2],k.shape[1]))
        base_flops=4*a.m*k.shape[1]*q.shape[2]*128;multiplier=cfg.cost(q.shape[2],k.shape[1])['matrix_product_multiplier']
    # Precompile outside the profiler range. Keep graph measurements independent
    # of CUDA allocator history created by decoded correctness references.
    for _ in range(3):fn();bf16()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    if a.profile:
        torch.cuda.nvtx.range_push('steady');fn();torch.cuda.synchronize();torch.cuda.nvtx.range_pop();out['profile']='one current-input call inside NVTX steady'
    else:
        samples={'fp4':[],'bf16':[]};rng=random.Random(5090)
        for _ in range(a.rounds):
            names=['fp4','bf16'];rng.shuffle(names)
            for name in names:samples[name].append(triton.testing.do_bench_cudagraph(fn if name=='fp4' else bf16,rep=a.rep_ms))
        out['latency_ms']={key:{'median':statistics.median(vals),'min':min(vals),'max':max(vals),'samples':vals} for key,vals in samples.items()}
        ms=out['latency_ms']['fp4']['median'];out['useful_base_tflops']=base_flops/ms/1e9;out['nominal_repaired_math_tflops']=base_flops*multiplier/ms/1e9
        out['flop_scope']='logical dense/repair matmuls only; excludes tail padding, quantization/softmax; not vendor sparse AI TOPS'
        out['bf16_over_fp4_latency_ratio']=out['latency_ms']['bf16']['median']/ms
    out['process_peak_allocated_bytes']=torch.cuda.max_memory_allocated();out['memory_scope']='entire harness, including BF16 reference weights/GQA expansion; not deployable model footprint'
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
