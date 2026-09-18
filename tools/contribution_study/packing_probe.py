"""Same-support selected/full packing equivalence and actual residual store coverage."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from baselines import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    with shared_worker():
        resource=select_shared_device()
        import torch
        from thunderworld_fp4.cosmos.linears import RepairLinear
        from thunderworld_fp4.cosmos.settings import LinearSpec
        from thunderworld_fp4.cosmos.tuning import LaunchTable
        from thunderworld_fp4._sparse_pack import pack_sparse
        from thunderworld_fp4._calibrated import pack
        torch.manual_seed(170917)
        source=torch.nn.Linear(2048,128,bias=False,device='cuda',dtype=torch.bfloat16)
        result=dict(complete=False,performance_eligible=False,resource=resource,rows=[],
            scope='Identical selected support and native accumulation. Selected writes are counted from the actual mask and code store predicates; '
                  'scratch still reserves a full residual plane. Selector has no additional MMA; summary, sorting and base re-encoding remain non-MMA work.')
        with torch.inference_mode():
            for m in (129,416,960):
                x=torch.randn(m,2048,device='cuda',dtype=torch.bfloat16)
                for quota in (1,4,8):
                    for cross in (False,True):
                        spec=LinearSpec(0,1.,cross,activation_blocks=quota)
                        controller=SimpleNamespace(warming=True,table=LaunchTable(),linear_spec=lambda *args:spec,record_linear=lambda *args:None)
                        layer=RepairLinear(source,controller,'probe',spec)
                        layer(x);op=layer.bound
                        sparse=pack_sparse(x,op.d,layer.sensitivity,op.am,quota)
                        dense=pack(x,op.d,True,True,op.am)
                        left=op.run_packed(x,sparse);right=op.run_packed(x,dense)
                        torch.testing.assert_close(left,right,rtol=0,atol=0)
                        mp=sparse[3];rowmask=op.am.repeat_interleave(128,0)[:mp]
                        for selected,full in zip(sparse[:2],dense[:2]):
                            width=selected.shape[-1]//16
                            torch.testing.assert_close(selected[0].view(torch.uint8),full[0].view(torch.uint8),rtol=0,atol=0)
                            torch.testing.assert_close(selected[1].view(torch.uint8).reshape(mp,16,width)[rowmask],
                                full[1].view(torch.uint8).reshape(mp,16,width)[rowmask],rtol=0,atol=0)
                        written=int(rowmask.sum())*128*9//16
                        full_bytes=mp*2048*9//16
                        result['rows'].append(dict(m=m,k=2048,n=128,a_quota=quota,cross=cross,
                            output_bitwise_equal=True,consumed_codes_bitwise_equal=True,
                            residual_store_bytes=written,full_residual_store_bytes=full_bytes,
                            residual_scratch_bytes=full_bytes,summary_bytes=((m+15)//16)*16*4,
                            selected_base_reencoded_elements=int(rowmask.sum())*128,selector_mma=0))
                        write(a.output,result)
            result['complete']=True;write(a.output,result)


if __name__=='__main__':main()
