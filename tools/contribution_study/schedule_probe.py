"""Fractional QK/PV quota equivalence, GQA/tails and current-input graph replay."""
import argparse
from dataclasses import replace
from pathlib import Path
from baselines import write,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    with shared_worker():
        resource=select_shared_device()
        import torch
        from thunderworld_fp4 import RepairConfig,FP4Attention,fp4_attention
        torch.manual_seed(171917);rows=[]
        with torch.inference_mode():
            for m,n in ((129,417),(416,1025)):
                q=torch.randn(1,m,4,128,device='cuda',dtype=torch.bfloat16)
                k=torch.randn(1,n,2,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
                cfg=RepairConfig(coverage=0.,head_budgets=((0,1.),(1,.5),(2,0.),(3,.75)),
                    pv_coverage=0.,pv_head_budgets=((1,1.),(2,.25),(3,.5)),query_tile=32,transpose_v=True,sparse_pack=True)
                explicit=replace(cfg,head_budgets=(),pv_head_budgets=(),pv_coverage=None,
                    head_quotas=cfg.head_counts(4,n),pv_head_quotas=cfg.pv_counts(4,n))
                runner=FP4Attention(cfg)
                first=runner(q,k,v);saved=first.clone()
                for x,y,z in ((q,k,v),(q*.75,k+.125,v-.25),(torch.zeros_like(q),torch.zeros_like(k),torch.zeros_like(v))):
                    expected=fp4_attention(x,y,z,explicit)
                    actual=runner(x,y,z)
                    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
                torch.testing.assert_close(first,saved,atol=0,rtol=0)
                runner.clear()
                rows.append(dict(m=m,n=n,h=4,hk=2,bitwise_equal=True,current_input_updated=True,output_lifetime=True,
                                 qk=cfg.head_counts(4,n),pv=cfg.pv_counts(4,n)))
        write(a.output,dict(complete=True,source_sha256=sha(__file__),resource=resource,rows=rows,performance_eligible=False))


if __name__=='__main__':main()
