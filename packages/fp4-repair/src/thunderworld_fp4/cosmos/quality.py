"""Paired per-task FP8 quality gates. Missing modalities/metrics fail closed."""
import numpy as np


def interval(values):
    values=np.asarray(values,dtype=np.float64)
    if len(values)<8 or not np.isfinite(values).all():raise ValueError('Eight finite paired seeds required')
    rng=np.random.default_rng(91000)
    means=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
    return np.quantile(means,[.025,.975]).tolist()


def gate(candidate,fp8,task,*,final=True,cohort=None):
    """Rows contain seed and metrics relative to the identical BF16 reference."""
    def index(rows):
        result={r['seed']:r for r in rows}
        if len(result)!=len(rows):raise ValueError('Repeated seeds are not independent quality samples')
        return result
    a,b=index(candidate),index(fp8)
    if set(a)!=set(b):raise ValueError('Mismatched quality seeds')
    if cohort is not None:
        cohort.validate('final' if final else 'validation',list(a),complete=True)
    elif final and set(a)!=set(range(91001,91009)):raise ValueError('Final quality requires reserved seeds 91001..91008')
    required={'video':('lpips','ssim_mean','relative_l2')}
    primary=[('video','lpips')]
    if task=='policy':
        required['action']=('relative_l2','rmse');primary.append(('action','relative_l2'))
    if task=='av':
        required['waveform']=('log_spectral_l1','relative_l2')
        required['sound_latent']=('relative_l2',);primary.append(('waveform','log_spectral_l1'))
    details={};failures=[]
    for modality,metrics in required.items():
        for metric in metrics:
            key=modality+'.'+metric
            try:
                diff=np.array([a[s]['metrics'][modality][metric]-b[s]['metrics'][modality][metric] for s in sorted(a)])
            except KeyError:
                failures.append('missing:'+key);continue
            if not np.isfinite(diff).all():failures.append('nonfinite:'+key);continue
            mean=float(diff.mean());record={'mean_paired_difference':mean,'worst_seed_difference':float(diff.min() if metric=='ssim_mean' else diff.max())}
            if (modality,metric) in primary:
                try:record['ci95']=interval(diff)
                except ValueError:failures.append('insufficient_seeds:'+key);continue
                if record['ci95'][1]>=0:failures.append(key)
            elif (mean<0 if metric=='ssim_mean' else mean>0):failures.append(key)
            details[key]=record
    return dict(passed=not failures,failures=failures,metrics=details,seeds=sorted(a),
                scope='Per-task BF16 consistency versus FP8, not task success')


def signal_metrics(x,reference):
    a,b=np.asarray(x,np.float64),np.asarray(reference,np.float64)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid signal pair')
    return dict(relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12)),
                rmse=float(np.sqrt(np.mean((a-b)**2))))


def compare(signals,reference,lpips_model):
    import torch
    from skimage.metrics import structural_similarity
    if set(signals)!=set(reference):raise ValueError('Output modalities differ from BF16')
    result={k:signal_metrics(v,reference[k]) for k,v in signals.items()}
    a,b=signals['video'],reference['video']
    result['video']['ssim_mean']=float(np.mean([structural_similarity(x,y,data_range=1.,channel_axis=-1) for x,y in zip(a,b)]))
    distances=[]
    device=next(lpips_model.parameters()).device
    with torch.inference_mode():
        for start in range(0,len(a),8):
            x=torch.from_numpy(a[start:start+8]).permute(0,3,1,2).float().to(device)*2-1
            y=torch.from_numpy(b[start:start+8]).permute(0,3,1,2).float().to(device)*2-1
            distances.extend(lpips_model(x,y).flatten().cpu().tolist())
    result['video']['lpips']=float(np.mean(distances))
    if 'waveform' in signals:
        a=torch.as_tensor(signals['waveform']).float().reshape(-1,signals['waveform'].shape[-1])
        b=torch.as_tensor(reference['waveform']).float().reshape_as(a)
        window=torch.hann_window(512)
        sa=torch.stft(a,n_fft=512,hop_length=128,window=window,return_complex=True).abs()
        sb=torch.stft(b,n_fft=512,hop_length=128,window=window,return_complex=True).abs()
        result['waveform']['log_spectral_l1']=float((sa.clamp_min(1e-5).log()-sb.clamp_min(1e-5).log()).abs().mean())
    return result
