"""Independent decoded-FP4 oracle, tail/mask tests and mutable graph replay."""
import os,json,math
import torch
import torch.nn.functional as F


def quant_ref(x):
    r,c=x.shape
    factor=x.abs().amax(-1).div(1536).clamp_min(1e-12)
    z=(x/factor[:,None]).reshape(r,c//16,16)
    scale=(z.abs().amax(-1)/6).clamp_min(1/512).to(torch.float8_e4m3fn).float()
    levels=torch.tensor([0,.5,1,1.5,2,3,4,6],device=x.device)
    edges=torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5],device=x.device)
    def round_fp4(a):
        az=a.abs().contiguous();ids=torch.bucketize(az,edges)
        # Native round-to-nearest-even at exact halfway values.
        tie=(ids<7)&(az==edges[ids.clamp_max(6)])&((ids%2)==1)
        ids=ids+tie.int()
        return levels[ids]*a.sign()
    base=round_fp4(z/scale[:,:,None])*scale[:,:,None]
    residual=z-base
    rs=(residual.abs().amax(-1)/6).clamp_min(1/512).to(torch.float8_e4m3fn).float()
    residual=round_fp4(residual/rs[:,:,None])*rs[:,:,None]
    return base.reshape(r,c)*factor[:,None],residual.reshape(r,c)*factor[:,None]

def decoded_inputs(q,k,v,repair):
    from thunderworld_fp4._kernels import pack_inputs
    packed,mp,np=pack_inputs(q,k,v,repair,False,3)[:3]
    levels=torch.tensor([0,.5,1,1.5,2,3,4,6],device=q.device)
    outputs=[]
    for i in range(3):
        codes,scales,f=packed[3*i:3*i+3]
        ids=torch.stack([codes&15,codes>>4],-1).reshape(codes.shape[0],-1,128).long()
        val=levels[ids&7]*torch.where(ids>=8,-1,1)
        deq=(val.reshape(codes.shape[0],-1,8,16)*scales.float()[:,:,:,None]).reshape(codes.shape[0],-1,128)*f[None,:,None]
        if not repair:deq=torch.cat([deq,torch.zeros_like(deq)])
        if i<2:deq=deq.reshape(2,q.shape[2] if i==0 else k.shape[2],mp if i==0 else np,128)
        else:deq=deq.reshape(2,k.shape[2],np//128,128,128).transpose(-1,-2).reshape(2,k.shape[2],np,128)
        outputs.append(deq)
    return outputs

def reference(q,k,v,flags,repair,terms=3,splits=2,bn=128):
    m,n,h,hk=q.shape[1],k.shape[1],q.shape[2],k.shape[2]
    nt=math.ceil(n/128);np=nt*128;outputs=[]
    splits=min(splits,math.ceil(n/bn));chunk=math.ceil(math.ceil(n/bn)/splits)
    splits=math.ceil(nt/chunk)
    qp,kp,vp=decoded_inputs(q,k,v,repair)
    for head in range(h):
        kh=head//(h//hk)
        q0,q1=qp[:,head,:m];k0,k1=kp[:,kh];vb,vr=vp[:,kh]
        parts=[]
        for sp in range(splits):
            acc=torch.zeros((m,128),device=q.device);mx=torch.full((m,),-float('inf'),device=q.device);den=torch.zeros(m,device=q.device)
            for start in range(sp*chunk,min((sp+1)*chunk,math.ceil(n/bn))):
                sl=slice(start*bn,(start+1)*bn)
                selected=flags.reshape(h,math.ceil(m/128),nt)[head,torch.arange(m,device=q.device)//128,start*bn//128]>1
                score=q0@k0[sl].T
                if repair:
                    extra=q1@k0[sl].T
                    if terms>=3:extra=extra+q0@k1[sl].T
                    if terms==4:extra=extra+q1@k1[sl].T
                    score=score+extra*selected[:,None]
                score=score/math.sqrt(128)
                score[:,torch.arange(start*bn,(start+1)*bn,device=q.device)>=n]=-float('inf')
                nm=torch.maximum(mx,score.amax(-1));alpha=(mx-nm).exp();p=(score-nm[:,None]).exp()
                den=den*alpha+p.sum(-1);p0,p1=quant_ref(p)
                tile=p0@vb[sl]
                if repair:
                    extra=p1@vb[sl]
                    if terms>=3:extra=extra+p0@vr[sl]
                    if terms==4:extra=extra+p1@vr[sl]
                    tile=tile+extra*selected[:,None]
                acc=acc*alpha[:,None]+tile;mx=nm
            parts.append((acc,mx,den))
        top=torch.stack([p[1] for p in parts]).amax(0)
        alphas=[(p[1]-top).exp() for p in parts]
        outputs.append(sum(p[0]*a[:,None] for p,a in zip(parts,alphas))/sum(p[2]*a for p,a in zip(parts,alphas))[:,None])
    return torch.stack(outputs,1).unsqueeze(0)
