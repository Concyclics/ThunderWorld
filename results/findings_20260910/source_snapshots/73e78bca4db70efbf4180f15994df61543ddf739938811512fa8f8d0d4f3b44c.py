"""Verify that reuse actually skips work and cannot leak across branches/requests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
import pytest
import runtime


def controller(mode):
    c = object.__new__(runtime.AttentionStudy)
    c.reset(6, 2)
    c.mode, c.observer = mode, None
    c.layer, c.branch, c.step = 0, 0, 0
    c.original = lambda q,k,v,**kw: q + v.mean()
    return c


def test_output_reuse_branch_request_and_refresh():
    c = controller('output_reuse2')
    q = torch.ones(1, 4, 2, 3); k = q.clone(); v = q.clone()
    first = c.dispatch(q,k,v)
    c.step = 1
    assert torch.equal(c.dispatch(q*7,k,v),first)
    c.branch = 1
    assert torch.equal(c.dispatch(q*7,k,v), q*7+1)
    c.branch = 0; c.step = 2
    assert torch.equal(c.dispatch(q*3,k,v),q*3+1)
    assert c.counts['full'] == 3 and c.counts['reused_output'] == 1
    c.reset(6,2); c.step,c.branch = 0,0
    assert torch.equal(c.dispatch(q*9,k,v),q*9+1)
    assert c.counts['reused_output'] == 0


def test_delta_reuse_recomputes_current_base(monkeypatch):
    calls = []
    def fake(q,k,v,cfg,*args,**kwargs):
        calls.append(cfg.coverage)
        return q*2+(q*.25 if cfg.coverage else 0), None, None
    monkeypatch.setattr(runtime,'fp4',fake)
    c = controller('delta_reuse2')
    q=torch.ones(1,4,2,3)
    assert torch.equal(c.dispatch(q,q,q),q*2.25)
    c.step=1
    assert torch.equal(c.dispatch(q*3,q,q),q*6.25)
    assert calls == [0,.5,0]
    assert c.counts['base']==2 and c.counts['full']==1 and c.counts['reused_delta']==1


def test_mask_reuse_does_not_reuse_values(monkeypatch):
    calls=[]
    mask=torch.tensor([[[True,False]]])
    def fake(q,k,v,cfg,repair_mask=None,return_mask=False):
        calls.append((repair_mask,return_mask))
        return q*2,mask,None
    monkeypatch.setattr(runtime,'fp4',fake)
    c=controller('mask_reuse2');q=torch.ones(1,4,2,3)
    c.dispatch(q,q,q);c.step=1
    assert torch.equal(c.dispatch(q*5,q,q),q*10)
    assert calls[0] == (None,True)
    assert torch.equal(calls[1][0],mask) and not calls[1][1]


def test_late_reuse_and_causal_bypass():
    c=controller('output_reuse2_late');q=torch.ones(1,4,2,3)
    c.dispatch(q,q,q);c.step=1
    assert torch.equal(c.dispatch(q*2,q,q),q*3)
    c.step=3
    assert torch.equal(c.dispatch(q*7,q,q),q*3)
    assert torch.equal(c.dispatch(q*8,q,q,is_causal=True),q*9)
    assert c.counts['reused_output']==1


def test_stable_group_reuse_only_skips_selected_heads():
    c=controller('output_reuse2_stable')
    c.group_policy={'group_ids_per_layer':{'0':[1]},'active_fraction':[0,1]}
    c.group_indices={}
    q=torch.ones(1,4,4,3);k=torch.ones(1,4,2,3)
    c.original=lambda q,k,v,**kwargs:q+1
    c.dispatch(q,k,k);c.step=1
    out=c.dispatch(q*5,k,k)
    assert torch.equal(out[:,:,:2],q[:,:,:2]*6)
    assert torch.equal(out[:,:,2:],q[:,:,2:]*2)
    assert c.counts['full']==2 and c.counts['reused_output']==1


@pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU sampling oracle')
def test_observer_uniform_attention_and_current_output():
    with torch.inference_mode():
        obs=runtime.Observer(1,4,2,1,samples_per_role=16)
        q=torch.zeros(1,128,4,128,device='cuda',dtype=torch.bfloat16)
        k=torch.zeros(1,160,2,128,device='cuda',dtype=torch.bfloat16)
        v=torch.ones_like(k);out=torch.ones_like(q)
        kw=dict(sequence_length=160,und_len=32,
                vision_sequence_indexes=torch.arange(32,160,device='cuda'),
                vision_mse_loss_indexes=torch.arange(64,160,device='cuda'))
        obs.observe(q,k,v,out,kw,0,0,0,1000.)
        obs.observe(q,k,v*2,out*2,kw,1,0,0,500.)
        import numpy as np
        assert np.allclose(obs.mass[:,0,0,:,1:3,0],32/160)
        assert np.allclose(obs.mass[:,0,0,:,1:3,1],32/160)
        assert np.allclose(obs.mass[:,0,0,:,1:3,2],96/160)
        assert np.allclose(obs.features[1,0,0,:,1:3,6],.5)
        assert np.allclose(obs.features[1,0,0,:,1:3,10],1)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Native FFN decoded oracle')
def test_ffn_terms_with_multiple_mask_rows_and_tail():
    from ffn import TermLinear
    def decode(c,s,g):
        codes=torch.stack((c&15,c>>4),dim=-1).flatten(-2).long()
        values=torch.tensor([0,.5,1,1.5,2,3,4,6],device=c.device)
        return values[codes&7]*torch.where(codes>=8,-1.,1.)*s.float().repeat_interleave(16,-1)*g
    with torch.inference_mode():
        torch.manual_seed(941)
        source=torch.nn.Linear(256,256,bias=False,device='cuda',dtype=torch.bfloat16)
        layer=TermLinear(source)
        x=torch.randn(257,256,device='cuda',dtype=torch.bfloat16)
        for mode in ['ffn_base','ffn_act2','ffn_wgt2','ffn_dual','ffn_four']:
            layer.configure(mode)
            actual=layer(x).float()
            ac,asc,ag,mp=layer.op.pack(x)
            av=decode(ac,asc,ag)[:,:len(x)]
            wv=decode(layer.codes.cuda(),layer.scales.cuda(),layer.factor.cuda())
            ref=av[0]@wv[0].T
            if mode in ['ffn_act2','ffn_dual','ffn_four']:ref+=av[1]@wv[0].T
            if mode in ['ffn_wgt2','ffn_dual','ffn_four']:ref+=av[0]@wv[1].T
            if mode=='ffn_four':ref+=av[1]@wv[1].T
            torch.cuda.synchronize()
            assert (actual-ref).norm()/ref.norm()<.004
            assert layer.op.am.is_contiguous() and layer.op.am.shape[0]==3
