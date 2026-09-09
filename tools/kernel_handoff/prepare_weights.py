"""Receiver-only offline packing from HF-downloaded BF16 shards.

Never called in a timed inference path. No complete pipeline or AWQ checkpoint
is needed: the data bundle already contains the small calibrated prescale.
"""
import argparse,hashlib,json,os,sys
from pathlib import Path
import torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'packages/fp4-repair/src'))
from thunderworld_fp4.linear import _pack
from verify_data import verify

def thash(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count()!=1:raise RuntimeError('Explicitly select one authorized GPU')
    if torch.cuda.get_device_capability()!=(12,0):raise RuntimeError('SM120 required')
    m=verify(a.data);a.output.mkdir(parents=True,exist_ok=True);receipt={'model_repo':m['model_repo'],'revision':m['bf16_revision'],'torch':torch.__version__,'files':{}}
    for entry in m['ffn']:
        src=a.data/entry['file'];out=a.output/entry['file']
        if out.exists():raise FileExistsError(f'{out}: use a new output directory')
        d=torch.load(src,weights_only=True,map_location='cpu');path=a.model/d['weight_shard']
        with safe_open(path,framework='pt',device='cpu') as sf:w=sf.get_tensor(d['module']+'.weight').contiguous()
        if thash(w)!=d['expected_tensor_sha256']['weight_bf16']:raise ValueError('BF16 tensor differs from pinned source')
        prescale=d['prescale'].cuda();wt=(w.cuda().float()/prescale).bfloat16().contiguous();ones=torch.ones_like(prescale)
        codes,scales,g,_=_pack(wt,ones,True,True,wt)
        tensors={'weight_bf16':w,'codes':codes.cpu(),'scales':scales.cpu(),'global_scale':g.cpu()}
        for key,value in tensors.items():
            if thash(value)!=d['expected_tensor_sha256'][key]:raise ValueError(f'{key} differs from source packing; check torch/triton versions before benchmarking')
        torch.save(dict(module=d['module'],input_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),**tensors),out)
        receipt['files'][out.name]={'bytes':out.stat().st_size,'verified_tensor_sha256':d['expected_tensor_sha256']};print('Prepared and verified',out,flush=True)
        del tensors,w,wt,codes,scales;torch.cuda.empty_cache()
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
