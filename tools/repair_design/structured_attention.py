"""Research-only role/head/phase prior mapped to real mixed-role token tiles.

The prior predicts attention density, NOT calibrated FP4 repair benefit.
A fixed total padded-tile quota is selected offline; no base edges are removed.
"""
import math
import numpy as np
import torch

ROLES=('ar','vision_clean','vision_noisy','sound','action','special')

def compile_mask(density,query_roles,key_roles,coverage):
    density=np.asarray(density,dtype=np.float64)
    qr=np.asarray(query_roles);kr=np.asarray(key_roles)
    if density.ndim!=3 or density.shape[1:]!=(6,6) or not np.isfinite(density).all() or np.any(density<0):
        raise ValueError('Finite nonnegative [heads,6,6] density required')
    if not 0<=coverage<=1 or not len(qr) or not len(kr) or qr.ndim!=1 or kr.ndim!=1:
        raise ValueError('Nonempty token layouts and coverage in [0,1] required')
    if not np.issubdtype(qr.dtype,np.integer) or not np.issubdtype(kr.dtype,np.integer) or np.any(qr<0) or np.any(qr>5) or np.any(kr<0) or np.any(kr>5):raise ValueError('Invalid role IDs')
    def counts(labels):
        out=np.zeros((math.ceil(len(labels)/128),6))
        for i in range(len(out)):out[i]=np.bincount(labels[i*128:(i+1)*128],minlength=6)
        return out
    qc=counts(qr);kc=counts(kr)
    # Mean query distribution times total key mass; mixed boundary tiles keep
    # every role. Query-head scores compete under one explicit global tile cap.
    score=np.einsum('qr,hrs,ks->hqk',qc/qc.sum(1,keepdims=True),density,kc)
    keep=math.floor(score.size*coverage);mask=np.zeros(score.size,dtype=bool)
    mask[np.argsort(-score.ravel(),kind='stable')[:keep]]=True
    return mask.reshape(score.shape)

def roles_from_kwargs(kwargs):
    n=int(kwargs['sequence_length']);ar=int(kwargs['und_len'])
    labels=np.full(n,5,dtype=np.int64);labels[:ar]=0
    for name,role in [('vision',1),('sound',3),('action',4)]:
        idx=kwargs.get(name+'_sequence_indexes')
        if idx is not None:labels[idx.detach().cpu().numpy()]=role
    labels[kwargs['vision_mse_loss_indexes'].detach().cpu().numpy()]=2
    return labels[ar:],labels
