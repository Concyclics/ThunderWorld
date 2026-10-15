"""Training-free offline scaling, represented quantization, and outlier proposals."""
import torch


def prescale_candidates(rms, weight):
    """X/s @ (W*s). Runtime prescale d=1/s, applied before BF16 rounding."""
    rms=rms.float().clamp_min(1e-6)
    wmax=weight.float().abs().amax(0).clamp_min(1e-6)
    yield 'identity',torch.ones_like(rms)
    for alpha in (0.,.25,.5,.75,1.):
        s=rms.pow(alpha)/wmax.pow(1-alpha)
        s=s/(s.min()*s.max()).sqrt()
        yield f'awq_alpha_{alpha:g}',s.reciprocal().clamp(1/16,16)
    # The earlier orientation is an explicit control, not silently relabeled AWQ.
    old=(rms.sqrt()/wmax.sqrt()).clamp(1/16,16)
    yield 'legacy_orientation',(old/(old.min()*old.max()).sqrt()).clamp(1/16,16)


def panel_objective(errors):
    if not errors:raise ValueError('No calibration panels')
    return sum(errors)/len(errors)+.25*max(errors)


def decode(codes,scales,factor):
    ids=torch.stack((codes&15,codes>>4),-1).reshape(*codes.shape[:-1],codes.shape[-1]*2).long()
    levels=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.],device=codes.device)
    value=levels[ids&7]*torch.where(ids>=8,-1.,1.)
    return (value.reshape(*scales.shape,16)*scales.float()[...,None]).reshape(value.shape)*factor


def _encode(z, scale_multiplier):
    grouped=z.reshape(z.shape[0],z.shape[1]//16,16)
    scales=(grouped.abs().amax(-1)*scale_multiplier/6).clamp(1/512,448).to(torch.float8_e4m3fn)
    v=(grouped/scales.float()[...,None]).reshape(z.shape)
    edges=torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5.],device=z.device)
    mag=v.abs().contiguous()
    ids=torch.bucketize(mag,edges)
    ids+=((ids<7)&(mag==edges[ids.clamp_max(6)])&((ids%2)==1)).long()
    ids=(ids+(torch.signbit(v).long()*8)).to(torch.uint8)
    codes=ids[:,0::2]|(ids[:,1::2]<<4)
    return codes,scales


def quantize_weight(weight,prescale,*,base=1.,residual=1.,remove=None):
    """Independent offline encoder. W1 references original scaled W, including removed values."""
    w=(weight.float()/prescale).bfloat16().float()
    peak=w.abs().amax()
    factor=torch.where(peak>0,peak/2688.,torch.ones_like(peak))
    z=w/factor
    base_input=z if remove is None else z.masked_fill(remove,0.)
    c0,s0=_encode(base_input,base)
    residual_z=z-decode(c0,s0,torch.ones_like(factor))
    c1,s1=_encode(residual_z,residual)
    return torch.stack((c0,c1)),torch.stack((s0,s1)),factor


def outlier_mask(weight,prescale,activation_rms,reserved_blocks,threshold):
    """At most one contribution outlier per K16 group, only inside reserved W1 tiles."""
    n,k=weight.shape
    if n%128 or k%128 or threshold not in (None,2.,3.):raise ValueError('Invalid outlier proposal')
    mask=torch.zeros_like(weight,dtype=torch.bool)
    if threshold is None:return mask,()
    contribution=weight.float().abs()*activation_rms[None,:]
    groups=contribution.reshape(n,k//16,16)
    peak,index=groups.max(-1)
    active=peak>threshold*groups.square().mean(-1).sqrt()
    allowed=torch.zeros((n//128,k//128),device=weight.device,dtype=torch.bool)
    ids=torch.as_tensor(reserved_blocks,device=weight.device,dtype=torch.long)
    if ids.numel():allowed.flatten()[ids]=True
    active &= allowed.repeat_interleave(128,0).repeat_interleave(8,1)
    mask.reshape(n,k//16,16).scatter_(2,index[...,None],active[...,None])
    mandatory=mask.reshape(n//128,128,k//128,128).any((1,3)).flatten().nonzero().flatten().tolist()
    return mask,tuple(mandatory)
