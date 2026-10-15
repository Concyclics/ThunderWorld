"""ARC packed-byte independent GEMM reference, including Cosmos K9216 and tails."""
import argparse,json
from pathlib import Path
from baselines import write,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=('edge','nano'),default='edge')
    p.add_argument('--shape',nargs=3,type=int,action='append',metavar=('M','N','K'))
    a=p.parse_args()
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    with shared_worker():
        resource=select_shared_device()
        import torch
        from external import load_arc,ARCLinear
        from thunderworld_fp4.cosmos.calibration import decode
        torch.backends.cuda.matmul.allow_tf32=False
        library,build=load_arc(a.build)
        def unpack(codes,raw):
            rows= codes.shape[0];cols=codes.shape[1]//8
            i=torch.arange(((rows+127)//128)*128*cols,device=codes.device)
            r=i//(128*cols)*128+(i//4%4)*32+i//16%32
            c=i//512%(cols//4)*4+i%4
            out=torch.empty((rows,cols),device=codes.device,dtype=torch.uint8)
            out[r[r<rows],c[r<rows]]=raw.flatten()[i[r<rows]]
            return decode(codes,out.view(torch.float8_e4m3fn),1.)
        result=dict(complete=False,performance_eligible=False,resource=resource,build_sha256=sha(a.build/'build.json'),rows=[])
        with torch.inference_mode():
            shapes=a.shape or (((129,1024,4096),(416,12288,4096),(960,4096,12288),(1024,4096,4096)) if a.model=='nano' else ((129,128,2048),(416,1024,2048),(960,2048,9216)))
            for m,n,k in shapes:
                for fraction in (0.,.5,1.):
                    torch.manual_seed(917)
                    source=torch.nn.Linear(k,n,bias=False,device='cuda',dtype=torch.bfloat16)
                    op=ARCLinear(source,library,torch.arange(k),fraction)
                    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
                    c,s,g=op.pack_input(x)
                    ref=((unpack(c,s)@unpack(op.weight_codes,op.weight_scales).T)*(g*op.factor)).bfloat16()
                    pred=op(x)
                    error=float((pred.float()-ref.float()).norm()/ref.float().norm())
                    torch.testing.assert_close(pred.float(),ref.float(),atol=.002,rtol=.02)
                    result['rows'].append(dict(shape=[m,n,k],fraction=fraction,relative_l2=error,passed=True))
                    write(a.output,result)
            result['complete']=True;write(a.output,result)


if __name__=='__main__':main()
