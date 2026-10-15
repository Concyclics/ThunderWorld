"""Official ModelOpt AWQ-lite, explicitly W4A16, offline quality-reference assets.

Decoded weights are cached for quality only. Their storage/latency must not be
reported as a packed W4A16 implementation or compared as native FP4 performance.
"""
import argparse,copy,json
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('capture','sources','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--modules',nargs='+');p.add_argument('--resume',action='store_true')
    p.add_argument('--require-pinned-source',action='store_true',help='Require installed ModelOpt to come from the collected source checkout')
    a=p.parse_args();capture=json.loads((a.capture/'manifest.json').read_text())
    sources=json.loads(a.sources.read_text())
    names=a.modules or list(capture['weights'])
    if not capture['complete'] or capture['split'] not in ('development','calibration'):raise ValueError('Complete non-test calibration capture required')
    identity=dict(capture_sha256=sha(a.capture/'manifest.json'),sources_sha256=sha(a.sources),modules=names,source_sha256=sha(__file__))
    a.output.mkdir(parents=True,exist_ok=a.resume)
    path=a.output/'manifest.json'
    result=dict(complete=False,provenance=identity,model_identity=capture['model_identity'],modules={},
        performance_eligible=False,execution_type='official_calibration_decoded_quality_reference',
        scope='NVFP4_AWQ_LITE_CFG with input/output quantizers disabled before calibration. '
              'Weights decoded once offline; activation is BF16 with calibrated channel prescale. '
              'Decoded BF16 storage is not a packed W4 memory claim.')
    if a.resume and path.exists():
        result=json.loads(path.read_text())
        if result['provenance']!=identity:raise ValueError('Resume provenance mismatch')
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    with shared_worker():
        result['resource']=select_shared_device()
        import torch
        import importlib.metadata
        import modelopt
        import modelopt.torch.quantization as mtq
        source_root=Path(sources['sources']['modelopt']['path']).resolve()
        installed=Path(mtq.__file__).resolve()
        if a.require_pinned_source:
            from baselines import git
            if not installed.is_relative_to(source_root) or git(source_root,'rev-parse','HEAD')!=sources['sources']['modelopt']['commit'] or git(source_root,'status','--porcelain','--untracked-files=no'):
                raise ValueError('Installed ModelOpt must be the clean pinned checkout; install it editable')
        result['modelopt_installed']=dict(module=str(installed),version=getattr(modelopt,'__version__','unknown'),
            pinned_source_required=a.require_pinned_source)
        torch.set_num_threads(8)
        config=copy.deepcopy(mtq.NVFP4_AWQ_LITE_CFG)
        config['quant_cfg'] += [{'quantizer_name':'*input_quantizer','enable':False},
                                {'quantizer_name':'*output_quantizer','enable':False}]
        result['modelopt_configuration']=config
        write(path,result)
        try:
            with torch.inference_mode():
                for index,name in enumerate(names):
                    if name in result['modules']:continue
                    w=torch.load(a.capture/capture['weights'][name],weights_only=True).cuda()
                    # AWQ may fold channel scales into weight in place.
                    original_weight=w.clone()
                    linear=torch.nn.Linear(w.shape[1],w.shape[0],bias=False,device='meta',dtype=torch.bfloat16)
                    linear.weight=torch.nn.Parameter(w,requires_grad=False)
                    model=torch.nn.Sequential(linear)
                    rows=[r for r in capture['rows'] if r['kind']=='linear' and r['name']==name]
                    if len(rows)!=16:raise ValueError('All full-M phase/CFG panels required')
                    panels=[torch.load(a.capture/r['file'],weights_only=True)['x'].cuda() for r in rows]
                    def forward_loop(current):
                        for x in panels:current(x)
                    mtq.quantize(model,config,forward_loop=forward_loop)
                    module=model[0]
                    if module.input_quantizer.is_enabled or not module.weight_quantizer.is_enabled:
                        raise ValueError('Expected W4A16 configuration')
                    decoded=module.weight_quantizer(module.weight).detach().contiguous()
                    scale=module.input_quantizer.pre_quant_scale
                    if scale is None:scale=torch.ones(w.shape[1],device=w.device,dtype=w.dtype)
                    errors=[]
                    for x in panels:
                        original=model(x)
                        restored=torch.nn.functional.linear(x*scale,decoded)
                        if not torch.equal(original,restored):raise ValueError('Offline AWQ export is not bitwise equal to official forward')
                        gold=torch.nn.functional.linear(x,original_weight).float()
                        errors.append(float((restored.float()-gold).square().sum()/gold.square().sum().clamp_min(1e-20)))
                    filename=f'{index:04d}.pt'
                    torch.save(dict(weight=decoded.cpu(),prescale=scale.cpu()),a.output/filename)
                    result['modules'][name]=dict(file=filename,sha256=sha(a.output/filename),panel_nmse=errors,
                        bitwise_export=True,weight_bits=[2,1],activation_bits=16,
                        represented_weight_bytes=decoded.numel()*2)
                    write(path,result);print(json.dumps(dict(module=name,mean_nmse=sum(errors)/len(errors))),flush=True)
                    del panels,model,module,decoded,w,linear,original_weight
                    torch.cuda.empty_cache()
            result['complete']=True;write(path,result)
        except Exception as error:
            result.update(error=repr(error),paused=True);write(path,result);raise


if __name__=='__main__':main()
