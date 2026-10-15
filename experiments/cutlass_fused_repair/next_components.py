"""Independent codec-layout and timestep-broadcast experiments; never promoted implicitly."""
import argparse,json,time,statistics
from pathlib import Path
import torch
from next_tuning_probe import ROOT,make,case_input
from thunderworld_fp4.cosmos.cli import idle_guard

def measure(fn,repeats=5):
    for _ in range(2):fn()
    times=[]
    for _ in range(repeats):
        torch.cuda.synchronize();t=time.perf_counter();fn();torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
    return dict(samples_ms=times,median_ms=statistics.median(times))

def compare(y,ref):
    delta=y.float()-ref.float();mse=delta.square().mean().item()
    return dict(exact=torch.equal(y,ref),mae=delta.abs().mean().item(),max_abs=delta.abs().max().item(),rmse=mse**.5,normalized_rmse=(mse/ref.float().square().mean().item())**.5)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;idle_guard()
    run=make('joint125','ar');run.prepare();case=case_input();run.warmup(case,35);run.warm_request(case,35)
    captured={};time_inputs=[];originals={}
    for name in ('encode','decode'):
        orig=getattr(run.pipe.vae,'_'+name);originals[name]=orig
        def capture(x,*args,_orig=orig,_name=name,**kw):captured[_name]=x.clone();return _orig(x,*args,**kw)
        setattr(run.pipe.vae,'_'+name,capture)
    orig_forward=run.pipe.transformer.forward
    def capture_time(*args,**kw):
        if run.engine.call%2==0:time_inputs.append(kw['vision_timesteps'].clone())
        return orig_forward(*args,**kw)
    run.pipe.transformer.forward=capture_time
    run.infer(case,7103,35)
    for name,orig in originals.items():setattr(run.pipe.vae,"_"+name,orig)
    run.pipe.transformer.forward=orig_forward
    result=dict(complete=False,scope='Component experiments on real seed 7103 inputs; timings are diagnostic, not end-to-end speedups',codec=[],timestep={})
    def save():(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    save();vae=run.pipe.vae;x=captured['encode'];latent=captured['decode'];captured.clear()
    enc=vae._encode;dec=lambda x:vae._decode(x,return_dict=False)[0]
    full=enc(x);first=enc(x[:,:,:1].contiguous())
    result['first_frame']=dict(input_shape=list(x.shape),first_shape=list(first.shape),comparison=compare(first,full[:,:,:1]),full=measure(lambda:enc(x)),first=measure(lambda:enc(x[:,:,:1].contiguous())))
    save();print('first_frame',result['first_frame'],flush=True)
    inputs=(latent,latent*.875+.03125);refs=[dec(v).clone() for v in inputs]
    for label,tiled,cl in [('original',True,False),('channels_last_weights',True,True),('untiled_channels_last_weights',False,True),('untiled',False,False)]:
        vae.use_tiling=tiled
        torch.nn.utils.convert_conv3d_weight_memory_format(vae,torch.channels_last_3d if cl else torch.contiguous_format)
        torch.cuda.reset_peak_memory_stats()
        rows=[compare(dec(v),ref) for v,ref in zip(inputs,refs)]
        timing=measure(lambda:dec(latent));peak=torch.cuda.max_memory_allocated()
        record=dict(variant=label,tiled=tiled,channels_last_conv3d_weights=cl,comparisons=rows,timing=timing,peak_allocated_bytes=peak)
        result['codec'].append(record);save();print('codec',record,flush=True)
    model=run.pipe.transformer;checks=[]
    def original(t):return model.time_embedder(model.time_proj(t*model.config.timestep_scale))
    def broadcast(t):return model.time_embedder(model.time_proj(t[:1]*model.config.timestep_scale)).expand(t.shape[0],-1)
    for i,t in enumerate(time_inputs):
        uniform=bool((t==t[0]).all());ref=original(t);out=broadcast(t)
        checks.append(dict(step=i,shape=list(t.shape),uniform=uniform,fp32=compare(out,ref),bf16=compare(out.bfloat16(),ref.bfloat16())))
    result['timestep']=dict(checks=checks,original=measure(lambda:original(time_inputs[0]),20),broadcast=measure(lambda:broadcast(time_inputs[0]),20))
    result['complete']=True;save();print('timestep exact',all(r['bf16']['exact'] for r in checks),flush=True)
if __name__=='__main__':main()
