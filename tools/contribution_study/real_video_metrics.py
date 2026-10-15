"""Paired real-video metrics, with conditioned frame excluded from the primary score."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from baselines import sha,write


def combine_full(future,first,video,reference):
    from thunderworld_fp4.cosmos.quality import signal_metrics,video_psnr
    result={**signal_metrics(video,reference),**video_psnr(video,reference)}
    for key in ('ssim_mean','lpips'):
        result[key]=(future[key]*(len(video)-1)+first[key])/len(video)
    return result


def run(root,follow):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import torch
    import lpips
    from thunderworld_fp4.cosmos.quality import compare
    torch.set_num_threads(8);torch.hub.set_dir('/tmp/thunderworld-lpips-cache')
    metric=lpips.LPIPS(net='alex',version='0.1').eval().cpu()
    manifest=root/'edge/final/manifest.json';study=json.loads(manifest.read_text())
    rp=Path(study['reference_manifest'])
    if sha(rp)!=study['reference_manifest_sha256']:raise ValueError('Changed real-video reference manifest')
    refs=json.loads(rp.read_text())
    if not study['frozen'] or not refs['complete']:raise ValueError('Unfrozen inputs')
    output=root/'edge/final/metrics_real';output.mkdir(parents=True,exist_ok=True)
    identity=dict(manifest_sha256=sha(manifest),reference_manifest_sha256=sha(rp),
        source_sha256=sha(__file__),quality_sha256=sha(Path(__file__).parents[2]/'packages/fp4-repair/src/thunderworld_fp4/cosmos/quality.py'),
        primary='future frames 1..60',secondary='all frames 0..60')
    progress=output/'progress.json'
    if progress.exists() and json.loads(progress.read_text())['identity']!=identity:
        raise ValueError('Metric resume identity changed')
    expected=len(study['methods'])*len(study['cases'])
    while True:
        completed=0;new=0
        for entry in study['cases']:
            cid=entry['case']['id'];ref=refs['cases'][cid];path=Path(ref['aligned_file'])
            if sha(path)!=ref['aligned_sha256']:raise ValueError('Changed aligned reference')
            with np.load(path) as data:gold=data['video']
            if hashlib.sha256(gold.tobytes()).hexdigest()!=ref['signal_sha256']:raise ValueError('Reference signal changed')
            for name in study['methods']:
                folder=root/'edge/final/generation'/name/cid;receipt=folder/'result.json'
                if not receipt.exists():continue
                generated=json.loads(receipt.read_text())
                if not generated['complete']:continue
                if generated['identity']['manifest_sha256']!=identity['manifest_sha256']:
                    raise ValueError('Wrong generation cohort')
                if generated['identity']['case']!=entry['case'] or generated['identity']['seeds']!=entry['seeds']:
                    raise ValueError('Unmatched generation request')
                if generated['identity']['image_sha256']!=ref['input_image_sha256']:
                    raise ValueError('Reference did not produce the conditioning image')
                target=output/name/(cid+'.json')
                provenance=dict(identity=identity,generation_receipt_sha256=sha(receipt),aligned_sha256=ref['aligned_sha256'])
                if target.exists():
                    old=json.loads(target.read_text())
                    if not old['complete'] or old['provenance']!=provenance:raise ValueError('Changed metric input')
                    completed+=1;continue
                rows=[]
                for row in generated['rows']:
                    source=folder/row['outputs']
                    if sha(source)!=row['outputs_sha256']:raise ValueError('Changed generated video')
                    with np.load(source) as data:video=data['video']
                    if hashlib.sha256(video.tobytes()).hexdigest()!=row['sha256']['video']:
                        raise ValueError('Generated signal hash mismatch')
                    if video.shape!=gold.shape or row['steps']!=35:raise ValueError('Changed frame shape or steps')
                    future=compare({'video':video[1:]},{'video':gold[1:]},metric)['video']
                    first=compare({'video':video[:1]},{'video':gold[:1]},metric)['video']
                    rows.append(dict(seed=row['seed'],future=future,first_frame=first,
                        all_frames=combine_full(future,first,video,gold)))
                if {r['seed'] for r in rows}!=set(entry['seeds']):raise ValueError('Missing paired seeds')
                write(target,dict(complete=True,method=name,case_id=cid,category=entry['category'],
                    rows=rows,provenance=provenance))
                completed+=1;new+=1
                print(json.dumps(dict(method=name,case=cid,reference='real_video')),flush=True)
                write(progress,dict(identity=identity,complete=False,completed=completed,expected=expected))
        state=dict(identity=identity,complete=completed==expected,completed=completed,expected=expected)
        write(progress,state)
        if state['complete'] or not follow:return state
        suite=root/'edge/final/generation/suite.json'
        if suite.exists():
            status=json.loads(suite.read_text())
            if status.get('paused'):raise RuntimeError('Generation paused: '+str(status.get('failures')))
            if status.get('complete'):raise RuntimeError('Generation completed with missing real-reference metrics')
        if not new:time.sleep(5)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--follow',action='store_true')
    args=parser.parse_args();print(json.dumps(run(args.root.resolve(),args.follow)))
