"""Decode measured body latents with the unchanged BF16 VAE and evaluate quality.

Golden BF16 videos must reconstruct exactly before any candidate is evaluated.
This separate operation cannot affect the measured body memory/timing results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from expand_worker import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--jobs',type=Path,help='Explicit [model, method, profile] rows; preserves historical default if omitted')
    p.add_argument('--after',type=Path,help='Wait for a serial measurement campaign before creating a GPU context')
    a=p.parse_args()
    while a.after:
        prior=json.loads(a.after.read_text())
        if prior.get('failed'):raise RuntimeError('Measurement campaign failed: '+str(prior['failed']))
        if prior.get('complete'):break
        time.sleep(15)
    import torch
    from diffusers import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor
    torch.set_num_threads(4)
    destination=a.root/'body-quality';destination.mkdir(parents=True,exist_ok=True)
    records=[];checks=[]
    with torch.inference_mode():
        for model in ('edge','nano'):
            manifest=json.loads((a.root/'manifests'/(model+'.json')).read_text())
            vae=AutoencoderKLWan.from_pretrained(str(Path(manifest['model'])/'vae'),
                torch_dtype=torch.bfloat16,local_files_only=True).cuda().eval()
            vae.enable_tiling()
            processor=VideoProcessor(vae_scale_factor=vae.config.scale_factor_spatial,resample='bilinear')
            mean=torch.tensor(vae.config.latents_mean,dtype=vae.dtype).cuda()
            inv=(1.0/torch.tensor(vae.config.latents_std,dtype=vae.dtype)).cuda()
            teacher_dir=a.root/'runs'/(model+'-bf16-original')
            teacher=json.loads((teacher_dir/'result.json').read_text())
            lookup={(r['case'],r['seed']):r for r in teacher['rows']}
            chosen_dtype=None
            methods=[('bf16','trim'),('fp8','trim'),('b1.5_protected_n','transformer2'),('b2_joint','transformer2')]
            if a.jobs:
                methods=[(method,profile) for key,method,profile in json.loads(a.jobs.read_text()) if key==model]
                if not methods or methods[0] != ('bf16','trim'):
                    raise ValueError('Each model must start with the unchanged BF16 control')
            for method,profile in methods:
                folder=a.root/'body-runs'/'-'.join((model,method,profile))
                run=json.loads((folder/'result.json').read_text())
                if not run.get('complete') or not run.get('latent_isolation_bitwise_equal'):raise ValueError('Unqualified body run')
                target=destination/'-'.join((model,method,profile));target.mkdir(exist_ok=True)
                for row in run['rows']:
                    if sha(folder/row['output'])!=row['npz_sha256']:raise ValueError('Changed body latent')
                    ref=lookup[row['case'],row['seed']]
                    array=np.load(folder/row['output'])['latent']
                    latent=torch.from_numpy(array).cuda().to(vae.dtype)
                    raw=latent/inv.view(1,-1,1,1,1)+mean.view(1,-1,1,1,1)
                    decoded=vae.decode(raw).sample
                    # The old body NPZ schema used float32 lossless storage for
                    # either latent dtype. Resolve postprocessing from a golden
                    # BF16 control, then hold the choice fixed for every method.
                    if chosen_dtype is None:
                        matched=[]
                        for dtype in (torch.bfloat16,torch.float32):
                            out=np.asarray(processor.postprocess_video(decoded.to(dtype),output_type='np')[0])
                            if hashlib.sha256(out.tobytes()).hexdigest()==ref['sha256']['video']:matched.append(dtype)
                        if not matched:raise AssertionError('Standalone BF16 codec does not reproduce full pipeline')
                        chosen_dtype=matched[0]
                    video=np.asarray(processor.postprocess_video(decoded.to(chosen_dtype),output_type='np')[0])
                    digest=hashlib.sha256(video.tobytes()).hexdigest()
                    if method=='bf16' and profile=='trim':
                        if digest!=ref['sha256']['video']:raise AssertionError('Codec golden output mismatch')
                        checks.append(dict(model=model,case=row['case'],seed=row['seed'],bitwise_equal=True,
                                           postprocess_dtype=str(chosen_dtype)))
                    name=Path(row['output']).stem+'.video.npz';np.savez(target/name,video=video)
                    records.append(dict(model=model,method=method,profile=profile,case=row['case'],seed=row['seed'],
                        path=str(target/name),npz_sha256=sha(target/name),video_sha256=digest,
                        reference=str(teacher_dir/ref['output']),reference_npz_sha256=ref['npz_sha256'],
                        source_body_receipt_sha256=sha(folder/'result.json')))
                    del decoded,latent,raw,video
                write(destination/'decoded.partial.json',dict(golden_checks=checks,outputs=records))
            del vae
            torch.cuda.empty_cache()
    write(destination/'decoded.json',dict(golden_checks=checks,outputs=records))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'contribution_study'))
    from full_video_metrics import VideoMetrics
    metric=VideoMetrics(device='cuda:0')
    for row in records:
        if sha(row['reference'])!=row['reference_npz_sha256']:raise ValueError('Changed BF16 reference')
        reference=np.load(row['reference'])['video'];candidate=np.load(row['path'])['video']
        metric.set_reference(reference);row['metrics']=metric.compare(candidate)['video']
        write(destination/'metrics.partial.json',dict(golden_checks=checks,rows=records))
    groups={}
    for row in records:groups.setdefault((row['model'],row['method'],row['profile']),[]).append(row)
    summary=[]
    for (model,method,profile),values in groups.items():
        mean={k:float(np.mean([v['metrics'][k] for v in values])) for k in ('lpips','ssim_mean','relative_l2','rmse')}
        psnr=[v['metrics']['psnr_db'] for v in values]
        mean['psnr_db']=float(np.mean(psnr)) if all(v is not None for v in psnr) else None
        summary.append(dict(model=model,method=method,profile=profile,n=len(values),mean_metrics=mean,
                            exact_psnr_videos=sum(v is None for v in psnr)))
    write(destination/'report.json',dict(golden_checks=checks,rows=records,summary=summary,
        scope='Same 2 reused scenes x 3 seeds; all candidates decoded with BF16 VAE; no generalization or quality-gate claim'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
