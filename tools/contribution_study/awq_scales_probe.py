"""Official-fake equality and tail/zero coverage for offline AWQ scale export."""
import argparse
from pathlib import Path
from baselines import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    with shared_worker():
        resource = select_shared_device()
        import torch
        from awq_scales import export_scales
        from awq_marlin import exact_codes_from_codebook
        from modelopt.torch.kernels.quantization.gemm.fp4_kernel_hopper import fp4_fake_quant_block
        from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
        rows = []
        torch.set_num_threads(8)
        with torch.inference_mode():
            for n, k in ((129, 416), (416, 960), (960, 1024), (1024, 2048)):
                generator = torch.Generator(device='cuda').manual_seed(180919+n+k)
                original = torch.randn((n, k), device='cuda', dtype=torch.bfloat16, generator=generator)
                original[:3, :16] = 0
                for label, w in (('random', original), ('small', original*1e-7), ('zero', torch.zeros_like(original))):
                    amax = w.abs().amax().float()
                    factor = amax/2688.
                    reference = fp4_fake_quant_block(w, amax)
                    scales = export_scales(w, factor)
                    codes = exact_codes_from_codebook(reference, scales, factor)
                    q = NVFP4QTensor(w.shape, torch.bfloat16, codes)
                    decoded = q.dequantize(dtype=torch.bfloat16, scale=scales, double_scale=factor, block_sizes={-1: 16})
                    if not torch.equal(decoded, reference):
                        raise ValueError('Offline scale export failed official fake-quant equality')
                    rows.append(dict(n=n, k=k, input=label, official_fake_bitwise_equal=True))
        write(a.output, dict(complete=True, rows=rows, resource=resource, performance_eligible=False,
            source_sha256=sha(__file__), scale_export_sha256=sha(Path(__file__).with_name('awq_scales.py')),
            scope='Offline export only. Same original global scale and official fake-quant helper; no refitting from decoded values. '
                  'Current native Marlin accumulation and full quality require their separate probes.'))


if __name__ == '__main__':
    main()
