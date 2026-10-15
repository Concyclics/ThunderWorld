"""Independent packed-value kernel checks for the Thor speed-only study."""
import argparse
import fcntl
import json
from pathlib import Path
import time

from build_baselines import write, sha


def unpack(codes, raw):
    import torch
    from thunderworld_fp4.cosmos.calibration import decode
    rows = codes.shape[0]
    cols = codes.shape[1]//8
    i = torch.arange(((rows+127)//128)*128*cols, device=codes.device)
    r = i//(128*cols)*128+(i//4%4)*32+i//16%32
    c = i//512%(cols//4)*4+i%4
    out = torch.empty((rows,cols),device=codes.device,dtype=torch.uint8)
    out[r[r<rows],c[r<rows]] = raw.flatten()[i[r<rows]]
    return decode(codes,out.view(torch.float8_e4m3fn),1.)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/home/chenhan/thunderworld'))
    p.add_argument('--method',choices=('arcquant','sharq','awq_marlin'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--manifest',type=Path,help='Frozen model-specific projection/asset inventory')
    p.add_argument('--only-k',type=int,help='Sanitizer subset for a newly instantiated K')
    a = p.parse_args()
    lock = (a.root/'logs/gpu-test.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    from compare_speed import idle_guard
    idle_guard()
    import torch
    from baseline_adapter import build_record, sharq_class
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(180922)
    assert torch.cuda.get_device_capability() == (11,0)
    result = dict(complete=False,method=a.method,scope='Operator correctness only; no generation quality evaluation',
                  performance_eligible=False,started=time.time(),rows=[])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    write(a.output,result)
    shapes = [(129,128,2048),(416,1024,2048),(960,2048,9216)]
    frozen=json.loads(a.manifest.read_text()) if a.manifest else None
    if frozen:
        shapes=[(m,n,k) for n,k in sorted(set(tuple(v) for v in frozen['dimensions'].values())) for m in (129,416,960)]
        result.update(model=frozen['model_key'],manifest_sha256=sha(a.manifest),dimensions=frozen['dimensions'])
    if a.only_k:shapes=[v for v in shapes if v[2]==a.only_k]
    if a.quick: shapes = shapes[:1]
    result['shapes']=shapes
    def check(pred, ref, **labels):
        error = float((pred.float()-ref.float()).square().sum()/ref.float().square().sum().clamp_min(1e-20))
        if not torch.isfinite(pred).all() or error > 1e-4:
            raise ValueError(f'Independent decoded-reference NMSE exceeds 1e-4: {labels}: {error}')
        result['rows'].append(dict(**labels,nmse=error,passed=True))
        write(a.output,result)
    try:
        with torch.inference_mode():
            if a.method == 'arcquant':
                from external import load_arc, ARCLinear
                folder, record = build_record(a.root,'arcquant',frozen.get('arc_build') if frozen else None)
                library, _ = load_arc(folder)
                result['build'] = record
                for m,n,k in shapes:
                    for fraction in (.5,1.):
                        source = torch.nn.Linear(k,n,bias=False,device='cuda',dtype=torch.bfloat16)
                        op = ARCLinear(source,library,torch.arange(k),fraction)
                        for zero in (False,True):
                            x = torch.zeros(m,k,device='cuda',dtype=torch.bfloat16) if zero else torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
                            codes,scales,factor = op.pack_input(x)
                            ref = ((unpack(codes,scales)@unpack(op.weight_codes,op.weight_scales).T)*(factor*op.factor)).bfloat16()
                            check(op(x),ref,shape=[m,n,k],fraction=fraction,zero=zero,current_input=True)
            elif a.method == 'sharq':
                from external import SharQLinear
                official,record = sharq_class(a.root)
                from quantize import load_sharq_ops
                lib = load_sharq_ops()
                result['build'] = record
                for m,n,k in shapes:
                    source = torch.nn.Linear(k,n,bias=False,device='cuda',dtype=torch.bfloat16)
                    op = SharQLinear(source,official)
                    op.official.extra_fusion = True
                    for zero in (False,True):
                        x = torch.zeros(m,k,device='cuda',dtype=torch.bfloat16) if zero else torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
                        y = op(x)
                        assert torch.equal(y,op.official(x)) and torch.equal(y,op(x[None])[0])
                        prepared = op.official.prepare_input(x)
                        _,compressed,meta,sf,qres,sfr,global_scale = prepared
                        debug = lib.fused_sparse_residual_quantize_x_debug((x/global_scale).bfloat16(),n)
                        # The official debug API exposes the uncompressed sparse
                        # codes. Use independent FP32 GEMMs, not another FP4 GEMM.
                        qsparse,sf2,qres2,sfr2 = debug
                        assert torch.equal(sf,sf2) and torch.equal(qres,qres2) and torch.equal(sfr,sfr2)
                        sfdense = lib.duplicate_sfa32_to_sfa16(sf,m,n,k)
                        weight = unpack(op.official.weight_q,op.official.scale_w_dense)
                        scale = global_scale*op.official.weight_scale
                        first = ((unpack(qsparse,sfdense)@weight.T)*scale).bfloat16()
                        ref = ((unpack(qres,sfr)@weight.T)*scale+first.float()).bfloat16()
                        check(y,ref,shape=[m,n,k],zero=zero,adapter_bitwise_equal=True)
            else:
                from awq_marlin import make_linear
                from thunderworld_fp4.cosmos.calibration import decode
                assets = Path(frozen['external_assets']['awq_marlin']) if frozen else a.root/'assets/awq_marlin'
                manifest = json.loads((assets/'manifest.json').read_text())
                seen = set()
                for name,entry in manifest['modules'].items():
                    file = assets/entry['file']
                    assert sha(file) == entry['sha256']
                    asset = torch.load(file,weights_only=True)
                    shape = tuple(asset['shape'])
                    if shape in seen:continue
                    seen.add(shape)
                    op = make_linear(asset)
                    weight = decode(asset['codes'].cuda(),asset['scales'].cuda().float()*asset['factor'].cuda(),1.).bfloat16()
                    for m in ([129] if a.quick else [129,416,960]):
                        fixed = torch.randn(m,op.in_features,device='cuda',dtype=torch.bfloat16)
                        for _ in range(3):op(fixed)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):out = op(fixed)
                        for zero in (False,True):
                            current = torch.zeros_like(fixed) if zero else torch.randn_like(fixed)
                            fixed.copy_(current);graph.replay()
                            assert torch.equal(out,op(current))
                            ref = torch.nn.functional.linear(current*op.prescale,weight)
                            check(out,ref,shape=[m,*shape],zero=zero,graph_eager_bitwise_equal=True)
                        del graph,out
                    if a.quick:break
                result['assets_sha256'] = sha(assets/'manifest.json')
        torch.cuda.synchronize()
        idle_guard()
        result.update(complete=True,finished=time.time())
    except Exception as error:
        result['error'] = repr(error)
        raise
    finally:
        write(a.output,result)
    print(json.dumps(dict(method=a.method,passed=len(result['rows']))),flush=True)


if __name__ == '__main__':main()
