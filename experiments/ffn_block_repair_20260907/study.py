"""Fixed-weight FFN sparse repair study; AV calibration, robot replay only.

W0 uses local BF16 weights, community AWQ prescale, and this package's RTN
packer. It is NOT the community AWQ checkpoint's packed W. No model rollouts.
"""
import json,os,sys,math,time,hashlib
from pathlib import Path
import torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'packages/fp4-repair/src'))
from thunderworld_fp4.linear import _pack
D=ROOT/'results/f4/ffn_block_repair_20260907'

class Metadata:pass

def load_state():
    path=ROOT/'artifacts/external_fp4/Reza2kn_Cosmos3_Nano_NVFP4_AWQ/transformer/modelopt_quantized.pt'
    with torch.serialization.safe_globals([(Metadata,'modelopt.torch.quantization.qtensor.nvfp4_tensor.NVFP4QTensor')]):
        return torch.load(path,weights_only=True,map_location='cpu',mmap=True)['model_state_dict']

def decode(c,s):
    ids=torch.stack((c&15,c>>4),-1).flatten(-2).long()
    lut=torch.tensor([0,.5,1,1.5,2,3,4,6],device=c.device)
    return lut[ids&7]*torch.where(ids>=8,-1,1)*s.float().repeat_interleave(16,-1)

def quant(x,d):
    c,s,g,mp=_pack(x,d,True,True,x)
    a=decode(c,s)[:,:x.shape[0]]*g
    return a,c,s,g

def top_mask(score,ratio):
    flat=score.flatten();count=math.ceil(flat.numel()*ratio)
    mask=torch.zeros_like(flat,dtype=torch.bool)
    if count:mask[torch.argsort(flat,descending=True,stable=True)[:count]]=True
    return mask.reshape(score.shape)

def amask(score,b,ratio):
    # One K support shared by all N consumers per 128 activation rows.
    grouped=score.reshape(score.shape[0],-1,b).sum(-1)
    return torch.stack([top_mask(s,ratio) for s in grouped]).repeat_interleave(b,-1)

def wmask(score,b,ratio):
    n,k=score.shape
    return top_mask(score.reshape(n//128,128,k//b,b).sum((1,3)),ratio).repeat_interleave(128,0).repeat_interleave(b,1)

def metric(y,ref):
    return {'sse':float((y-ref).square().sum()),'energy':float(ref.square().sum()),'relative_l2':float((y-ref).norm()/ref.norm())}

@torch.inference_mode()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','3')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    state=load_state();caps={label:torch.load(ROOT/'results/f4'/file,map_location='cpu',mmap=True,weights_only=True)['panels'] for label,file in [('cal','cosmos3_nano_dev_activations_true_steps_cond_gpu0.pt'),('eval','cosmos3_nano_pai_robot026_dev_activations_true_steps_cond_gpu0.pt')]}
    index=ROOT/'artifacts/huggingface/models/Cosmos3-Nano-411f42a/transformer/diffusion_pytorch_model.safetensors.index.json';idx=json.loads(index.read_text())['weight_map']
    names=[f'layers.{layer}.mlp_moe_gen.{kind}_proj' for layer in (0,24) for kind in ('gate','up','down')]
    report={'complete':False,'protocol':{'modules':names,'calibration':'AV 3 captured steps,128 rows per step','evaluation':'robot026 3 captured steps,128 rows per step; previously observed development inputs','weight_format':'RTN W4 of BF16-rounded local W / community AWQ prescale; fixed W0 across candidates','reference':'FP32 matmul of original BF16 X and W; no bias','selection':'A shared across all N per M128; W selected globally across N128/K blocks offline','no_training':True,'no_online_weight_quantization':True,'limitations':['Local FFN outputs, not rollout quality','Shared GPU timings measured separately','K16 precision upper bound needs gather/packing for Tensor Core execution']} ,'rows':[],'assets':{},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for name in names:
        with safe_open(index.parent/idx[name+'.weight'],framework='pt',device='cpu') as sf:w=sf.get_tensor(name+'.weight').cuda()
        d=state[name+'.input_quantizer._pre_quant_scale'].cuda().float();wt=(w.float()/d).bfloat16().contiguous();ones=torch.ones_like(d)
        wp,wc,ws,wg=quant(wt,ones);w0,w1=wp;wn=w0.square().sum(0)
        cal=torch.cat([x[:128] for x in caps['cal'][name]]).cuda();ap,*_=quant(cal,d)
        actual=(cal.float()*d).bfloat16().float()
        avar=actual.square().mean(0);ares=(actual-ap[0]).square().mean(0)*wn
        wr=(wt.float()-w0)
        wscore=wr.square()*avar
        report['assets'][name]={'N':w.shape[0],'K':w.shape[1],'base_bytes':w.numel()*9//16+4,'weight_repair_bytes_at_6_25pct':math.ceil((w.shape[0]//128)*(w.shape[1]//128)*.0625)*128*128*9//16+(w.shape[0]//128)*(w.shape[1]//128)*4}
        # Freeze W selections and static A supports before observing robot inputs.
        frozen={(b,p):wmask(wscore,b,p) for b in (16,64,128,256) for p in (.03125,.0625,.125)}
        static={(b,p):amask(ares[None],b,p) for b in (16,64,128,256) for p in (.25,.5,.75)}
        for step,panel in zip((0,17,34),caps['eval'][name]):
            x=panel[:128].cuda();ap,*_=quant(x,d);a0,a1=ap;actual=(x.float()*d).bfloat16().float();ref=x.float()@w.float().T;base=a0@w0.T
            row={'module':name,'step':step,'metrics':{'base':metric(base,ref),'full_a':metric(base+a1@w0.T,ref),'full_dual':metric(base+a1@w0.T+a0@w1.T,ref),'full_four':metric((a0+a1)@(w0+w1).T,ref)},'selected_fractions':{}}
            scores={'static':ares[None],'residual':((actual-a0).square().mean(0)*wn)[None],'energy':actual.square().mean(0)[None]}
            for b in (16,64,128,256):
                for kind,score in scores.items():
                    for pa in (.25,.5,.75):
                        ma=static[b,pa] if kind=='static' else amask(score,b,pa)
                        ya=base+(a1*ma)@w0.T
                        key=f'k{b}_{kind}_a{pa:g}'
                        row['metrics'][key]=metric(ya,ref);row['selected_fractions'][key]=float(ma.float().mean())
                        if kind=='residual' and pa in (.5,.75):
                            for pw in (.03125,.0625,.125):
                                mw=frozen[b,pw];rw=w1*mw
                                y=ya+a0@rw.T
                                j=key+f'_w{pw:g}'
                                row['metrics'][j]=metric(y,ref)
                                row['metrics'][j+'_cross']=metric(y+(a1*ma)@rw.T,ref)
                                row['selected_fractions'][j]=float(ma.float().mean())+float(mw.float().mean())
            for budget in (.5,.75,1.):
                for pw in (0.,.0625,.125,.25,.5):
                    pa=budget-pw
                    if pa<0:continue
                    ma=amask(scores['residual'],128,pa)
                    mw=wmask(wscore,128,pw)
                    y=base+(a1*ma)@w0.T+a0@(w1*mw).T
                    key=f'equal{budget:g}_a{pa:g}_w{pw:g}'
                    row['metrics'][key]=metric(y,ref)
            report['rows'].append(row)
        D.mkdir(exist_ok=True,parents=True);(D/'quality_equal.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name, 'done',flush=True)
        del w,wt,wp,w0,w1,wr,wscore,frozen,cal,ap,actual,ref,base,ya,y,rw
        torch.cuda.empty_cache()
    keys=report['rows'][0]['metrics'];report['summary']={key:math.sqrt(sum(r['metrics'][key]['sse'] for r in report['rows'])/sum(r['metrics'][key]['energy'] for r in report['rows'])) for key in keys}
    report['by_kind']={kind:{key:math.sqrt(sum(r['metrics'][key]['sse'] for r in report['rows'] if kind in r['module'])/sum(r['metrics'][key]['energy'] for r in report['rows'] if kind in r['module'])) for key in keys} for kind in ('gate_proj','up_proj','down_proj')}
    report['complete']=True;(D/'quality_equal.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['summary'],indent=2),flush=True)
if __name__=='__main__':main()
