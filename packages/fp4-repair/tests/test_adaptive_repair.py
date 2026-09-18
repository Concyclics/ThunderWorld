"""Adaptive A selection stays below the static cap and consumes current inputs."""
import math
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from thunderworld_fp4.cosmos.settings import RunConfig
from thunderworld_fp4.cosmos.budget import linear_cost

@pytest.mark.parametrize('value',[0,-1,1.1,float('nan'),float('inf'),True])
def test_bad_energy(value):
    with pytest.raises(ValueError):RunConfig(activation_repair_energy=value)

def test_full_quota_cannot_claim_mask_savings():
    # Both existing CUDA backends bypass the A mask for a full quota.
    with pytest.raises(ValueError,match='unconditional'):
        linear_cost(129,256,512,a_blocks=4,a_support=[(),(0,1,2,3)])

@pytest.mark.parametrize('m',[129,416])
@pytest.mark.parametrize('bm',[32,64,128])
def test_variable_support_cost(m,bm):
    mt=math.ceil(m/128);supports=[(),(0,1)]+[ (2,) for _ in range(mt-2)]
    bound=linear_cost(m,256,512,bm=bm,a_blocks=2,w_blocks=(0,1,4),cross=True)
    exact=linear_cost(m,256,512,bm=bm,a_blocks=2,w_blocks=(0,1,4),cross=True,a_support=supports)
    unit=2*bm*128*128
    expected=sum(len(supports[i*bm//128]) for i in range(math.ceil(m/bm)))*2*unit
    assert exact['mma_by_dtype']['fp4']['a']==expected
    assert exact['total_mma']<=bound['total_mma']
    assert exact['packing']['selected_residual_write_bytes']<=bound['packing']['selected_residual_write_bytes']

@pytest.mark.parametrize('m',[129,416])
@pytest.mark.parametrize('energy',[.5,.75,1.])
@pytest.mark.parametrize('tile',[64,128])
def test_cuda_current_input_graph_and_dense_encoding(m,energy,tile):
    import torch,triton
    if not torch.cuda.is_available():pytest.skip('GPU required')
    library=os.environ.get('TW_CUTLASS_LIBRARY' if tile==128 else 'TW_CUTLASS_OTHER_LIBRARY',f'/tmp/tw_cutlass_fused_n{tile}.so')
    if not Path(library).exists():pytest.skip('Build the production CUTLASS library first')
    from thunderworld_fp4._sparse_pack import pack_sparse,choose
    from thunderworld_fp4._calibrated import pack
    from thunderworld_fp4.cosmos.calibration import decode,quantize_weight
    from thunderworld_fp4.cosmos.linears import RepairLinear
    from thunderworld_fp4.cosmos.settings import LinearSpec
    from thunderworld_fp4.cosmos.tuning import LaunchTable
    k=512;torch.manual_seed(49)
    with torch.inference_mode():
        x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
        # Uneven residual contribution, including the final partial M128 tile.
        x[:,:128]*=8
        d=torch.linspace(.5,2,k,device='cuda');sens=torch.linspace(.1,4,k,device='cuda')
        mask=torch.empty((math.ceil(m/128),4),device='cuda',dtype=torch.bool)
        def check():
            sparse=pack_sparse(x,d,sens,mask,2,base=.875,res=1.125,energy=energy)
            dense=pack(x,d,True,True,x,base=.875,res=1.125)
            residual=(x.float()*d).bfloat16().float()-decode(dense[0][0,:m],dense[1][0,:m],dense[2])
            expected=torch.zeros_like(mask)
            for mb in range(len(mask)):
                score=(residual[mb*128:(mb+1)*128].square()*sens).reshape(-1,4,128).sum((0,2))
                order=score.argsort(descending=True,stable=True);running=0.
                for index in order[:2]:
                    if energy==1 or running<float(score.sum())*energy:expected[mb,index]=True
                    running+=float(score[index])
            assert torch.equal(mask,expected)
            for a,b in zip(sparse[:2],dense[:2]):
                assert torch.equal(a[0].view(torch.uint8),b[0].view(torch.uint8))
                channels=a.shape[-1]//4
                used=mask.repeat_interleave(128,0)[:m].repeat_interleave(channels,1)
                assert torch.equal(a[1,:m].view(torch.uint8)[used],b[1,:m].view(torch.uint8)[used])
            return sparse
        check()
        w=torch.randn((256,k),device='cuda',dtype=torch.bfloat16)
        c,s,g=quantize_weight(w,d)
        asset=dict(codes=c.cpu(),scales=s.cpu(),factor=g.cpu(),prescale=d.cpu(),bias=None,
                   score=torch.ones(2,4),selector='residual_sensitivity',activation_sensitivity=sens.cpu(),activation_scale=(.875,1.125))
        source=torch.nn.Linear(k,256,bias=False,device='cuda',dtype=torch.bfloat16);source.weight.copy_(w)
        spec=LinearSpec(.5,1,True)
        cfg=RunConfig(linear_backend='cutlass-fused',cutlass_library=library,
                      cutlass_direct_pack=True,activation_repair_energy=energy)
        ctl=SimpleNamespace(config=cfg,warming=True,table=LaunchTable(),linear_spec=lambda n,d:spec,record_linear=lambda *a:None)
        layer=RepairLinear(source,ctl,'test',spec,asset);layer(x);ctl.warming=False
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):y=layer(x)
        for zero in (False,True,False):
            if zero:x.zero_()
            else:x.normal_();x[:,:128]*=8
            check();graph.replay();torch.cuda.synchronize()
            expected=layer(x)
            assert torch.equal(y,expected)
            assert torch.equal(layer.bound.am,mask)
            saved=y.clone()
            # Exact dense encoder plus the chosen mask is an independent packing path.
            full=pack(x,layer.bound.d,True,True,x,base=.875,res=1.125)
            assert torch.equal(saved,layer.bound.run_packed(x,full))
            assert (mask.sum(1)<=2).all()
            if zero and energy<1:assert not mask.any()

@pytest.mark.parametrize('k',[2048,9216])
@pytest.mark.parametrize('energy',[.5,.75,1.])
def test_cuda_selector_real_channel_width(k,energy):
    import torch,triton
    if not torch.cuda.is_available():pytest.skip('GPU required')
    from thunderworld_fp4._sparse_pack import choose
    m=416;kb=k//128;cap=kb//2
    # Includes non-power-of-two K tiles, ties, zero summaries, and a tail M tile.
    summary=torch.arange(kb,device='cuda',dtype=torch.float32).remainder(7).repeat(math.ceil(m/16),1)
    summary[:8]=0
    mask=torch.empty((math.ceil(m/128),kb),device='cuda',dtype=torch.bool)
    choose[(len(mask),)](summary,mask,m,k,triton.next_power_of_2(kb),cap,energy,num_warps=4)
    expected=torch.zeros_like(mask)
    for mb in range(len(mask)):
        scores=summary[mb*8:(mb+1)*8].sum(0)
        order=scores.argsort(descending=True,stable=True);running=0.
        for index in order[:cap]:
            if energy==1 or running<float(scores.sum())*energy:expected[mb,index]=True
            running+=float(scores[index])
    assert torch.equal(mask,expected)
