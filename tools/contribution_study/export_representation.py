"""Export fixed all-layer offline assets; no step-dependent W0 requantization."""
import argparse
import json
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('capture','assets','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--method',choices=('rtn','four_six'),required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    source=json.loads((a.assets/'manifest.json').read_text())
    capture=json.loads((a.capture/'manifest.json').read_text())
    if not source['complete'] or not capture['complete'] or source['model_identity']!=capture['model_identity']:
        raise ValueError('Complete matching sources required')
    identity=dict(source=sha(a.assets/'manifest.json'),capture=sha(a.capture/'manifest.json'),
                  method=a.method,encoder=sha(Path(__file__).with_name('representation.py')))
    a.output.mkdir(parents=True,exist_ok=a.resume)
    manifest={**source,'complete':False,'modules':{},'contribution_representation':identity}
    target=a.output/'manifest.json'
    if a.resume and target.exists():
        manifest=json.loads(target.read_text())
        if manifest.get('contribution_representation')!=identity:raise ValueError('Resume inputs changed')
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    with shared_worker():
        manifest['resource']=select_shared_device()
        import torch
        from thunderworld_fp4.cosmos.assets import expand,compact
        from representation import planes
        torch.set_num_threads(8)
        try:
            with torch.inference_mode():
                for name,entry in source['modules'].items():
                    if name in manifest['modules']:
                        if sha(a.output/entry['file'])!=manifest['modules'][name]['sha256']:
                            raise ValueError('Changed completed asset')
                        continue
                    if sha(a.assets/entry['file'])!=entry['sha256']:raise ValueError('Changed source asset')
                    old=expand(torch.load(a.assets/entry['file'],weights_only=True),1.)
                    if old.get('mandatory_w_blocks'):raise ValueError('Cannot silently undo a mandatory support asset')
                    weight=torch.load(a.capture/capture['weights'][name],weights_only=True).cuda()
                    d=old['prescale'].cuda()
                    rep=planes((weight.float()/d).bfloat16(),a.method)
                    n,k=weight.shape
                    out={**old,**{key:rep[key].cpu() for key in ('codes','scales','factor')},
                         'offline_encoder':a.method,'mandatory_w_blocks':()}
                    out['score']=rep['scales'][1].float().reshape(n//128,128,k//128,8).square().sum((1,3)).cpu()
                    file=a.output/entry['file']
                    torch.save(compact(out,1.),file)
                    manifest['modules'][name]=dict(file=entry['file'],sha256=sha(file))
                    write(target,manifest)
                    print(name,flush=True)
            manifest['complete']=True;manifest.pop('error',None);write(target,manifest)
        except Exception as error:
            manifest.update(error=repr(error),paused=True);write(target,manifest);raise


if __name__=='__main__':main()
