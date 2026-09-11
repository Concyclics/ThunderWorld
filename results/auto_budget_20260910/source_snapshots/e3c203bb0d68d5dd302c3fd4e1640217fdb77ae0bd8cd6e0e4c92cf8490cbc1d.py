"""Decode persisted AV latents with the main Transformer released from VRAM."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import sys

import numpy as np
import torch
import scipy.io.wavfile
import diffusers.utils.import_utils as imports
imports._xformers_available=False
from diffusers.models.autoencoders.autoencoder_cosmos3_audio import Cosmos3AVAEAudioTokenizer


def metrics(a,b):
    diff=a-b
    aa=torch.from_numpy(a);bb=torch.from_numpy(b)
    window=torch.hann_window(512)
    sa=torch.stft(aa,n_fft=512,hop_length=128,window=window,return_complex=True).abs()
    sb=torch.stft(bb,n_fft=512,hop_length=128,window=window,return_complex=True).abs()
    return dict(relative_l2=float(np.linalg.norm(diff)/max(np.linalg.norm(b),1e-12)),
                rmse=float(np.sqrt(np.mean(diff**2))),
                log_spectral_l1=float((sa.clamp_min(1e-5).log()-sb.clamp_min(1e-5).log()).abs().mean()))


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);a=p.parse_args()
    torch.set_num_threads(8)
    root=Path(__file__).resolve().parents[2]
    report=json.loads((a.input/'result.json').read_text())
    decoder=Cosmos3AVAEAudioTokenizer.from_pretrained(str(root/'models/Cosmos3-Nano-411f42a/sound_tokenizer'),torch_dtype=torch.bfloat16,local_files_only=True).cuda()
    rate=decoder.config.sampling_rate
    references={};anchors={};repairs={};rows=[]
    # Warm the first observed shape outside per-item decode timing.
    warmed=set()
    for row in report['rows']:
        stem=f'{row["mode"]}_{row["case"]}_seed{row["seed"]}_r{row["repeat"]}'
        data=np.load(a.input/(stem+'_outputs.npz'))
        if 'sound_latent' not in data:continue
        latent=torch.from_numpy(data['sound_latent']).cuda().to(torch.bfloat16)
        if tuple(latent.shape) not in warmed:
            decoder.decode(latent.unsqueeze(0));warmed.add(tuple(latent.shape))
        torch.cuda.synchronize();start=time.perf_counter()
        audio=decoder.decode(latent.unsqueeze(0)).squeeze(0).float().cpu().numpy()
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        assert np.isfinite(audio).all()
        key=row['case'],row['seed']
        if row['mode']=='bf16':references[key]=audio
        if row['mode']=='ffn_act2':anchors[key]=audio
        if row['mode']=='attn_dual50':repairs[key]=audio
        rec=dict(mode=row['mode'],case=row['case'],seed=row['seed'],repeat=row['repeat'],
                 decode_seconds=seconds,shape=list(audio.shape),sampling_rate=rate,
                 sha256=hashlib.sha256(audio.tobytes()).hexdigest())
        if row['mode']!='bf16':rec['vs_bf16']=metrics(audio,references[key])
        if key in anchors and row['mode']!='ffn_act2':rec['vs_ffn_act2']=metrics(audio,anchors[key])
        if key in repairs and row['mode'].startswith(('mask_reuse','delta_reuse')):rec['vs_current_repair']=metrics(audio,repairs[key])
        scipy.io.wavfile.write(a.input/(stem+'.wav'),rate,audio.T)
        rows.append(rec)
        (a.input/'audio_decode.json').write_text(json.dumps(dict(complete=False,rows=rows),indent=2))
        print(stem,round(seconds,3),rec.get('vs_bf16'),flush=True)
    (a.input/'audio_decode.json').write_text(json.dumps(dict(complete=True,rows=rows,
        scope='GPU audio decode in separate process. Decode time excludes loading and warmup. Can be listed alongside Generator time but is not a measured integrated-request latency.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()
