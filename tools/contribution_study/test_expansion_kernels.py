"""Native kernel qualification before any full-trajectory quality run."""
import pytest
import torch

from expansion_kernels import pack_linear, linear, term_mask, pack_attention, attention, decode


@pytest.mark.parametrize('m',[129,416])
@torch.inference_mode()
def test_nested_linear_matches_existing_two_planes_and_decoded_reference(m):
    from thunderworld_fp4.linear import _pack
    from thunderworld_fp4._native import Native
    torch.manual_seed(72101+m)
    torch.backends.cuda.matmul.allow_tf32=False
    n,k=256,256
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
    d=torch.ones(k,device='cuda')
    ac,asc,ag,mp=pack_linear(x,4)
    wc,ws,wg,_=pack_linear(w,4)
    olda=_pack(x,d,True,True,x);oldw=_pack(w,d,True,True,w)
    assert torch.equal(ac[:2],olda[0]) and torch.equal(asc[:2].view(torch.uint8),olda[1].view(torch.uint8))
    assert torch.equal(wc[:2],oldw[0]) and torch.equal(ws[:2].view(torch.uint8),oldw[1].view(torch.uint8))
    assert torch.equal(ag,olda[2]) and torch.equal(wg,oldw[2])
    a=decode(ac,asc,ag)[:,:m];b=decode(wc,ws,wg)
    for wp,ap,dual in [(1,1,False),(2,2,True),(2,2,False),(2,3,False),
                       (3,2,False),(3,3,False),(3,4,False),(4,3,False),(4,4,False)]:
        mask=term_mask(wp,ap,dual)
        result=linear(x,(wc[:wp],ws[:wp],wg),wp,ap,mask)
        reference=sum(a[j]@b[i].T for i in range(wp) for j in range(ap) if mask&(1<<(i*4+j)))
        torch.testing.assert_close(result.float(),reference,rtol=.004,atol=.004)
        if wp==ap==2:
            am=torch.ones(((m+127)//128,k),device='cuda',dtype=torch.bool)
            wm=torch.ones((n,k),device='cuda',dtype=torch.bool)
            native=Native(*oldw[:3],d,am,wm,cross=not dual,bm=128,interleave=True,num_stages=2)
            assert torch.equal(result,native(x)), 'Two-plane arithmetic changed from Native'


@torch.inference_mode()
def test_zero_planes_and_current_input_graph():
    x=torch.zeros((129,128),device='cuda',dtype=torch.bfloat16)
    w=torch.zeros((128,128),device='cuda',dtype=torch.bfloat16)
    wc,ws,wg,_=pack_linear(w,4)
    assert torch.count_nonzero(wc)==0 and torch.isfinite(ws.float()).all()
    assert torch.count_nonzero(linear(x,(wc,ws,wg),4,4,term_mask(4,4)))==0
    w.normal_();wc,ws,wg,_=pack_linear(w,4)
    for _ in range(2):linear(x,(wc,ws,wg),4,4,term_mask(4,4))
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):output=linear(x,(wc,ws,wg),4,4,term_mask(4,4))
    x.normal_();graph.replay();first=output.clone()
    assert torch.equal(first,linear(x,(wc,ws,wg),4,4,term_mask(4,4)))
    x.mul_(.5).add_(.25);graph.replay()
    assert torch.equal(output,linear(x,(wc,ws,wg),4,4,term_mask(4,4)))
    assert not torch.equal(first,output)


@pytest.mark.parametrize('m,n,h,hk',[(129,145,4,2),(416,257,2,2)])
@torch.inference_mode()
def test_attention_gqa_tails_matches_legacy_and_high_order_reference(m,n,h,hk):
    from thunderworld_fp4._kernels import pack_inputs
    from thunderworld_fp4 import RepairConfig
    from thunderworld_fp4.runtime import _forward
    torch.manual_seed(72103+m)
    q=torch.randn((1,m,h,128),device='cuda',dtype=torch.bfloat16)
    k=torch.randn((1,n,hk,128),device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    packed,mp,np=pack_attention(q,k,v,4,4)
    old,omp,onp=pack_inputs(q,k,v,True,False,4,True)
    assert (mp,np)==(omp,onp)
    for index,(new,previous) in enumerate(zip(packed,old)):
        if index%3==2:assert torch.equal(new,previous)
        else:assert torch.equal(new[:2].view(torch.uint8),previous.view(torch.uint8))
    for wp,ap,dual,terms in [(1,1,False,1),(2,2,True,3),(2,2,False,4)]:
        y=attention(q,k,v,wp,ap,term_mask(wp,ap,dual))
        cfg=RepairConfig(coverage=0. if terms==1 else 1.,terms=max(terms,2),query_tile=32,
            splits=2,transpose_v=True,fast_p=True)
        oldy=_forward(q,k,v,cfg)[0]
        assert torch.equal(y,oldy), 'Two-plane attention arithmetic changed'
    reference=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2).float(),
        k.transpose(1,2).float().repeat_interleave(h//hk,1),
        v.transpose(1,2).float().repeat_interleave(h//hk,1)).transpose(1,2)
    for wp,ap in [(2,3),(3,2),(3,3),(3,4),(4,3),(4,4)]:
        y=attention(q,k,v,wp,ap,term_mask(wp,ap))
        assert torch.isfinite(y).all()
        error=torch.linalg.vector_norm(y.float()-reference)/torch.linalg.vector_norm(reference)
        assert error < (.025 if min(wp,ap)==2 else .006),float(error)
    assert torch.count_nonzero(attention(torch.zeros_like(q),torch.zeros_like(k),torch.zeros_like(v),
        4,4,term_mask(4,4)))==0
