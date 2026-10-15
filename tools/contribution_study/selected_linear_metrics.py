"""Qualified all-frame metrics for the frozen selected-policy follow-up."""
import argparse
import json
from pathlib import Path

from baselines import sha, write
from selected_linear_study import NEW


def read(path):
    return json.loads(path.read_text())


def run(root):
    import numpy as np
    import torch
    from full_video_metrics import VideoMetrics, load_video, environment
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    manifest = root/'edge/final/manifest.json'
    study = read(manifest)
    if read(root/'study.json')['manifest_sha256'] != sha(manifest):
        raise ValueError('Changed study')
    if not read(root/'workflow.json')['complete']:
        raise ValueError('Finish generation before starting GPU metrics')
    output = root/'edge/final/metrics'
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(manifest_sha256=sha(manifest), source_sha256=sha(__file__),
        backend_sha256=sha(Path(__file__).with_name('full_video_metrics.py')),
        protocol='All 61 decoded float32 frames, AlexNet LPIPS 0.1, RGB SSIM, range-1 PSNR; identical definitions to controls')

    def receipt(method, entry):
        folder = root/'edge/final/generation'/method/entry['case']['id']
        path = folder/'result.json'; value = read(path)
        if (not value['complete'] or value['identity']['manifest_sha256'] != sha(manifest)
            or value['identity']['case'] != entry['case']
            or value['identity']['image_sha256'] != entry['image_sha256']
            or value['identity']['seeds'] != entry['seeds']
            or sorted(r['seed'] for r in value['rows']) != sorted(entry['seeds'])):
            raise ValueError('Mismatched or incomplete generation')
        return folder, path, value

    def load(folder, row):
        if (row['steps'],row['debug'],row['execution'],row['calls']) != (35,'off','graph',70):
            raise ValueError('Changed request protocol')
        return load_video(dict(path=str(folder/row['outputs']), npz_sha256=row['outputs_sha256'],
                               video_sha256=row['sha256']['video']))

    with shared_worker(), torch.inference_mode():
        resource = select_shared_device()
        evaluator = VideoMetrics()
        env = environment()
        qpath = root/'metric_qualification.json'
        qid = dict(identity=identity, environment=env)
        if qpath.exists():
            qualified = read(qpath)
            if not qualified['complete'] or qualified['identity'] != qid:
                raise ValueError('Changed metric qualification')
        else:
            # Independent skimage SSIM and ordinary CPU LPIPS/PSNR on two real
            # benchmark pairs, before using the reusable GPU metric backend.
            import lpips
            from thunderworld_fp4.cosmos.quality import compare
            cpu = lpips.LPIPS(net='alex',version='0.1').eval().cpu()
            checks = []
            for benchmark in ('rbench','paibench'):
                entry = next(e for e in study['cases'] if e['benchmark']==benchmark)
                bf, _, gold = receipt('bf16',entry)
                cf, _, candidate = receipt('selected_1p5',entry)
                ref = load(bf,gold['rows'][0]); video = load(cf,candidate['rows'][0])
                evaluator.set_reference(ref)
                measured = evaluator.compare(video)['video']
                independent = compare({'video':video},{'video':ref},cpu)['video']
                errors = {k: abs(measured[k]-independent[k]) for k in
                          ('lpips','ssim_mean','psnr_db','relative_l2')}
                tolerances = dict(lpips=2e-6,ssim_mean=2e-6,psnr_db=1e-9,relative_l2=1e-9)
                if any(errors[k]>tolerances[k] for k in errors):
                    raise ValueError('GPU metric definition drift: '+str(errors))
                checks.append(dict(benchmark=benchmark,case=entry['case']['id'],errors=errors,tolerances=tolerances))
            del cpu
            write(qpath,dict(complete=True,identity=qid,checks=checks,resource=resource))
        completed = 0
        for entry in study['cases']:
            cid = entry['case']['id']; bf, bp, gold = receipt('bf16',entry)
            pending = {}
            for name in NEW:
                cf, cp, value = receipt(name,entry)
                if value['model_identity']!=gold['model_identity'] or value['targets']!=gold['targets']:
                    raise ValueError('Changed model/scope')
                provenance = dict(identity=identity,reference_sha256=sha(bp),candidate_sha256=sha(cp),
                    qualification_sha256=sha(qpath))
                target = output/name/(cid+'.json')
                if target.exists():
                    saved = read(target)
                    if saved['provenance'] != provenance or not saved['complete']:
                        raise ValueError('Changed metric inputs')
                    completed += 1
                else:
                    pending[name] = dict(folder=cf,value=value,target=target,provenance=provenance,rows=[])
            for seed in entry['seeds']:
                if not pending:
                    break
                evaluator.set_reference(load(bf,next(r for r in gold['rows'] if r['seed']==seed)))
                for name, p in pending.items():
                    row = next(r for r in p['value']['rows'] if r['seed']==seed)
                    metrics = evaluator.compare(load(p['folder'],row))
                    if not all(np.isfinite(metrics['video'][k]) for k in ('lpips','ssim_mean','psnr_db','relative_l2')):
                        raise ValueError('Nonfinite metrics')
                    p['rows'].append(dict(seed=seed,metrics=metrics))
            for name, p in pending.items():
                write(p['target'],dict(complete=True,name=name,case_id=cid,benchmark=entry['benchmark'],
                    category=entry['category'],rows=p['rows'],provenance=p['provenance']))
                completed += 1
            write(output/'progress.json',dict(identity=identity,complete=False,completed=completed,
                expected=len(NEW)*len(study['cases']),performance_eligible=False))
            print(json.dumps(dict(case=cid,completed=completed,expected=len(NEW)*len(study['cases']))),flush=True)
        result = dict(identity=identity,complete=True,completed=completed,
            expected=len(NEW)*len(study['cases']),performance_eligible=False)
        write(output/'progress.json',result)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();print(json.dumps(run(args.root.resolve())))
