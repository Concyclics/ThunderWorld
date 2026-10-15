"""Paired quantizer/order endpoints, decoded sparse products, Graph and budget."""
import math
import pytest
import torch
import expansion_sparse_kernels as sparse
from expansion_kernels import pack_linear,pack_attention,decode,linear,attention
from expansion_sparse_budget import linear_cost,attention_cost,compile_request
from test_expansion_two_term import probability_planes


@pytest.mark.parametrize('m',[129,416])
@torch.inference_mode()
def test_linear_endpoints_decoded_partial_and_graph(m):
    torch.manual_seed(92600+m);torch.backends.cuda.matmul.allow_tf32=False
    x=torch.randn(m,512,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(256,512,device='cuda',dtype=torch.bfloat16)
    wc,ws,wg,_=pack_linear(w,2);ac,asc,ag,mp=pack_linear(x,2)
    a,b=decode(ac,asc,ag)[:,:m],decode(wc,ws,wg)
    shape=((m+127)//128,4)
    for aa,ww,mask in [(False,False,1),(True,False,3),(False,True,17),(True,True,19)]:
        am=torch.full(shape,aa,device='cuda',dtype=torch.bool)
        wm=torch.full((2,4),ww,device='cuda',dtype=torch.bool)
        y=sparse.linear(x,(wc,ws,wg),wm,torch.ones(4,device='cuda'),0,am=am)
        expected=linear(x,(wc,ws,wg),2,2,mask)
        assert torch.equal(y,expected),'Full/empty support changed original arithmetic'
    for num in (1,2):
        wm,sens=sparse.weight_support((wc,ws,wg),num)
        e=sparse.energies(ac,asc,ag)[1,0]
        am=sparse.choose(e*sens,num)
        assert torch.all(am.sum(-1)==num) and torch.all(wm.sum(-1)==num)
        ma=am.repeat_interleave(128,0).repeat_interleave(128,1)[:m]
        mw=wm.repeat_interleave(128,0).repeat_interleave(128,1)
        ref=a[0]@b[0].T+(a[1]*ma)@b[0].T+a[0]@(b[1]*mw).T
        y=sparse.linear(x,(wc,ws,wg),wm,sens,num)
        torch.testing.assert_close(y.float(),ref,rtol=.004,atol=.005)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=sparse.linear(x,(wc,ws,wg),wm,sens,num)
        x.mul_(.5).add_(.125);graph.replay()
        assert torch.equal(out,sparse.linear(x,(wc,ws,wg),wm,sens,num))
        x.zero_();graph.replay();assert torch.count_nonzero(out)==0
        x.normal_()
        # Refresh decoded current input for the next fraction.
        ac,asc,ag,mp=pack_linear(x,2);a=decode(ac,asc,ag)[:,:m]


def reference_attention(q,k,v,masks):
    packed,mp,npad=pack_attention(q,k,v,2,2)
    qc,qs,qf,kc,ks,kf,vc,vs,vf=packed
    m,n,h,hk=q.shape[1],k.shape[1],q.shape[2],k.shape[2];nt=npad//128
    qd=decode(qc,qs,qf[None,:,None]).reshape(2,h,mp,128)
    kd=decode(kc,ks,kf[None,:,None]).reshape(2,hk,npad,128)
    vc=vc.reshape(2,hk,nt,64,128).transpose(-1,-2).contiguous()
    vs=vs.reshape(2,hk,nt,8,128).transpose(-1,-2).contiguous()
    vd=decode(vc,vs,vf.reshape(1,hk,nt,128,1))
    qm,pm=masks;out=torch.empty_like(q,dtype=torch.float32)
    for head in range(h):
        kh=head//(h//hk)
        mask=qm[:,head].repeat_interleave(32,1).repeat_interleave(128,2)[:,:m,:n]
        score=(qd[0,head,:m]@kd[0,kh,:n].T+
            (qd[1,head,:m]@kd[0,kh,:n].T)*mask[0]+
            (qd[0,head,:m]@kd[1,kh,:n].T)*mask[1])/math.sqrt(128)
        p=torch.exp(score-score.amax(-1,keepdim=True));total=torch.zeros((m,128),device=q.device)
        for block in range(nt):
            pt=torch.nn.functional.pad(p[:,block*128:min((block+1)*128,n)],(0,max(0,(block+1)*128-n)))
            pd=probability_planes(pt,2)
            mask=pm[:,head,:,block].repeat_interleave(32,1)[:,:m,None]
            total+=pd[0]@vd[0,kh,block].T+(pd[1]@vd[0,kh,block].T)*mask[0]+(pd[0]@vd[1,kh,block].T)*mask[1]
        out[0,:,head]=total/p.sum(-1,keepdim=True)
    return out


@pytest.mark.parametrize('m,n,h,hk',[(129,513,4,2),(416,257,2,2)])
@torch.inference_mode()
def test_attention_endpoints_partial_reference_gqa_tails_graph(m,n,h,hk):
    torch.manual_seed(92700+m);torch.backends.cuda.matmul.allow_tf32=False
    q=torch.randn(1,m,h,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,n,hk,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    shape=(2,h,(m+31)//32,(n+127)//128)
    for aa,ww,mask in [(False,False,1),(True,False,3),(False,True,17),(True,True,19)]:
        support=torch.empty(shape,device='cuda',dtype=torch.bool);support[0]=aa;support[1]=ww
        y=sparse.attention(q,k,v,0,masks=(support,support))
        assert torch.equal(y,attention(q,k,v,2,2,mask)),'Attention endpoint arithmetic changed'
    for num in (1,2):
        packed,mp,npad=pack_attention(q,k,v,2,2)
        masks=sparse.attention_support(packed,mp,npad,m,h,hk,num)
        for mask in masks:assert torch.all(mask.sum(-1)==shape[-1]*num//4)
        y=sparse.attention(q,k,v,num)
        ref=reference_attention(q,k,v,masks)
        err=torch.linalg.vector_norm(y.float()-ref)/torch.linalg.vector_norm(ref)
        assert err<.004,float(err)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=sparse.attention(q,k,v,num)
        q.mul_(.7);k.add_(.1);v.mul_(.5);graph.replay()
        assert torch.equal(out,sparse.attention(q,k,v,num))
        q.zero_();k.zero_();v.zero_();graph.replay();assert torch.count_nonzero(out)==0
        q.normal_();k.normal_();v.normal_()


@torch.inference_mode()
def test_energy_decode_and_stable_zero_selection():
    x=torch.randn(129,512,device='cuda',dtype=torch.bfloat16)
    c,s,f,mp=pack_linear(x,2);e=sparse.energies(c,s,f)[:,0]
    d=decode(c,s,f)
    d=torch.nn.functional.pad(d,(0,0,0,256-mp))
    ref=d.reshape(2,2,128,4,128).square().sum((2,4))
    torch.testing.assert_close(e,ref,rtol=2e-6,atol=1e-6)
    for count in (0,1,2,4):
        mask=sparse.choose(torch.zeros((3,4),device='cuda'),count)
        expected=torch.arange(4,device='cuda')[None,:].expand(3,4)<count
        assert torch.equal(mask,expected)


def test_padded_budget_all_shapes_cfg_and_phases():
    for num,budget in [(1,1.5),(2,2.)]:
        rows=[]
        for branch in (0,1):
            for m in (129,416,960):
                for k in (128,256,512,1024):
                    r=linear_cost(m,256,k,num);assert r['ratio']<=budget
                    assert r['total_mma']==r['base_mma']+r['a_mma']+r['w_mma']
                    rows.append(dict(r,branch=branch))
                for n in (129,416,960):
                    r=attention_cost(m,n,4,2,num);assert r['ratio']<=budget
                    rows.append(dict(r,branch=branch))
        receipt=compile_request(rows,budget)
        assert receipt['total_mma']==35*sum(r['total_mma'] for r in rows)
        assert receipt['ratio']<=budget and receipt['selector_mma']==receipt['cross_mma']==0
