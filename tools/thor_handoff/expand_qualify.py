"""Independent codec/lookup reconstruction and eager/Graph input tests."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sanitizer',action='store_true')
    a=p.parse_args();sys.path.insert(0,str(a.variant))
    import torch
    from thunderworld_fp4._expanded_runtime import PackedRows,PackedEmbedding,PackedCausalConv3d,Expansion,packed_conv
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d
    from diffusers.models.autoencoders.autoencoder_kl_cosmos import CosmosCausalConv3d
    torch.set_num_threads(4);torch.manual_seed(4517)
    rows=[]
    def check(ok,label):
        if not ok:raise AssertionError(label)
        rows.append(label)
    def reference(op):
        lut=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],device='cuda')
        values=[]
        for i in range(op.planes):
            c=getattr(op,'c'+str(i)).long()
            codes=torch.stack((c&15,c>>4),-1).reshape(op.n,op.kp)
            values.append(lut[codes]*getattr(op,'s'+str(i)).float().repeat_interleave(16,1))
        value=values[0] if len(values)==1 else values[0]+values[1]
        return (value[:,:op.k]*op.factor[:,None]).to(op.output_dtype)
    with torch.inference_mode():
        for k in (1,16,129,416,960,2048):
            for planes in (1,2):
                w=torch.randn(17,k,dtype=torch.bfloat16)
                w[0]=0
                op=PackedRows(w,planes).cuda();expected=reference(op)
                check(torch.equal(op.decode(),expected),f'decode_exact_k{k}_p{planes}')
                check(torch.equal(op.decode()[0],w[0].cuda()),f'zero_k{k}_p{planes}')
                ids=torch.tensor([16,0,5,5,2],device='cuda')
                check(torch.equal(op.decode(ids),expected[ids]),f'gather_k{k}_p{planes}')
                loaded=PackedRows(w,planes,op.export()).cuda()
                check(torch.equal(loaded.decode(),expected),f'reload_k{k}_p{planes}')
                if k==416:
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(2):op.decode(ids)
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):out=op.decode(ids)
                    torch.cuda.current_stream().wait_stream(stream)
                    saved=out.clone();ids.copy_(torch.tensor([1,3,2,0,16],device='cuda'))
                    graph.replay()
                    check(torch.equal(out,expected[ids]),f'current_graph_p{planes}')
                    check(not torch.equal(saved,out),f'output_changed_p{planes}')
                    check(op.decode(torch.empty(0,dtype=torch.long,device='cuda')).shape==(0,k),f'empty_p{planes}')
        for planes in (1,2):
            for kernel,stride,padding in (((1,3,3),1,1),((3,1,1),1,0),((3,3,3),(2,2,2),1),(1,1,0)):
                conv=CosmosCausalConv3d(3,5,kernel,stride=stride,padding=padding).to('cuda',torch.bfloat16)
                packed=PackedCausalConv3d(conv,planes).cuda()
                conv.weight.copy_(reference(packed.rows).reshape_as(conv.weight))
                x=torch.randn(1,3,5,17,19,device='cuda',dtype=torch.bfloat16)
                check(torch.equal(packed(x),conv(x)),f'causal_conv_k{kernel}_s{stride}_p{planes}')
            for cache_frames in (0,1,2):
                conv=WanCausalConv3d(3,5,3,padding=1).to('cuda',torch.bfloat16)
                packed=packed_conv(conv,planes).cuda()
                conv.weight.copy_(reference(packed.rows).reshape_as(conv.weight))
                x=torch.randn(1,3,3,17,19,device='cuda',dtype=torch.bfloat16)
                cache=torch.randn(1,3,cache_frames,17,19,device='cuda',dtype=torch.bfloat16) if cache_frames else None
                check(torch.equal(packed(x,cache),conv(x,cache)),f'wan_cache{cache_frames}_p{planes}')
                check(isinstance(packed,WanCausalConv3d),f'wan_cache_count{cache_frames}_p{planes}')
            conv=torch.nn.Conv2d(3,5,3,padding=1).to('cuda',torch.bfloat16)
            packed=packed_conv(conv,planes).cuda()
            conv.weight.copy_(reference(packed.rows).reshape_as(conv.weight))
            x=torch.randn(2,3,17,19,device='cuda',dtype=torch.bfloat16)
            check(torch.equal(packed(x),conv(x)),f'conv2d_p{planes}')
        for bad in (-1,3,True):
            try:Expansion(vae_planes=bad)
            except ValueError:rows.append('invalid_planes_'+str(bad))
            else:raise AssertionError('Invalid profile accepted')
    result=dict(passed=True,checks=rows,count=len(rows),device=torch.cuda.get_device_name(),
                scope='Packed row and causal-convolution numerical correctness, not full-generation quality',
                sanitizer=a.sanitizer)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
