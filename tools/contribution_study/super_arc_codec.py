"""Independent byte-level check of official ARC K16 base/residual quantization."""
import argparse
import json
import os
from pathlib import Path

from baselines import sha, write


def reference(x, order, count, *, activation):
    import torch
    rows, k = x.shape
    groups = x[:, order.long()].float().reshape(rows, k//16, 16)
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=x.device)
    indexes = torch.arange(8, device=x.device)

    def quantize(values):
        scale = (values.abs().amax(-1)/6).clamp(2**-9, 448).to(torch.float8_e4m3fn)
        normalized = (values * scale.float().reciprocal().unsqueeze(-1)).clamp(-6, 6)
        distances = (normalized.abs().unsqueeze(-1)-levels).abs()
        nearest = distances == distances.amin(-1, keepdim=True)
        even = nearest & ((indexes % 2) == 0)
        tie_even = torch.where(even.any(-1, keepdim=True), even, nearest)
        index = torch.where(tie_even, indexes, 8).amin(-1)
        # NumericConverter preserves the sign of underflowed negatives (E2M1 -0).
        codes = index | (torch.signbit(normalized).to(torch.int64) << 3)
        decoded = levels[index] * torch.where(torch.signbit(normalized), -1., 1.)
        return codes, scale, decoded

    c0, s0, value0 = quantize(groups)
    residual = (groups - value0*s0.float().unsqueeze(-1)).bfloat16().float()
    c1, s1, _ = quantize(residual) if activation else (c0, s0, value0)
    cut = (k-count)//16
    codes = torch.cat((c0[:, :cut], torch.stack((c0[:, cut:], c1[:, cut:]), dim=2).flatten(1,2)), dim=1).flatten(1)
    # E4M3 tensors do not have every eager arithmetic/cat operation on all builds.
    scales = torch.cat((s0.view(torch.uint8)[:, :cut],
        torch.stack((s0.view(torch.uint8)[:, cut:], s1.view(torch.uint8)[:, cut:]), dim=2).flatten(1)), dim=1)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return packed, scales


def unpack_scales(raw, rows, columns):
    import torch
    i = torch.arange(((rows+127)//128)*128*columns, device=raw.device)
    r = i//(128*columns)*128 + (i//4%4)*32 + i//16%32
    c = i//512 % (columns//4)*4 + i%4
    out = torch.empty((rows, columns), device=raw.device, dtype=torch.uint8)
    out[r[r<rows], c[r<rows]] = raw.view(torch.uint8).flatten()[i[r<rows]]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--widths', nargs='+', type=int, default=[5120, 8192, 25600])
    a = p.parse_args()
    from super_runtime import pair_worker, admit
    if len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) == 1:
        from thunderworld_fp4.cosmos.resources import shared_worker as pair_worker, select_shared_device as admit
    import torch
    from external import load_arc
    with pair_worker(), torch.inference_mode():
        resource = admit(minimum_mib=12288)
        library, _ = load_arc(a.build)
        result = dict(complete=False, build_sha256=sha(a.build/'build.json'),
            source_sha256=sha(__file__), resource=resource, checks=[], performance_eligible=False)
        for k in a.widths:
            torch.manual_seed(928)
            order = torch.randperm(k, device='cuda').to(torch.int16)
            for data in ('random', 'zeros'):
                x = torch.randn(129, k, device='cuda', dtype=torch.bfloat16) if data == 'random' else torch.zeros(129, k, device='cuda', dtype=torch.bfloat16)
                for fraction in (0., .5, 1.):
                    count = int(k*fraction)//64*64
                    for side, fn in [('activation', library.reorder_quantize_x), ('weight', library.reorder_quantize_w)]:
                        c, s = fn(x, order, count)
                        expected_c, expected_s = reference(x, order, count, activation=side=='activation')
                        actual_s = unpack_scales(s, len(x), (k+count)//16)
                        if not torch.equal(c, expected_c):
                            mismatch = int((c != expected_c).sum())
                            raise ValueError(f'ARC codes differ from independent reference: {k}/{fraction}/{data}/{side}: {mismatch}')
                        if not torch.equal(actual_s, expected_s):
                            raise ValueError(f'ARC scales differ: {k}/{fraction}/{data}/{side}')
                        result['checks'].append(dict(k=k, rows=len(x), fraction=fraction, data=data, side=side,
                            codes_bitwise_equal=True, scales_bitwise_equal=True))
                        write(a.output, result)
        result['complete'] = True
        write(a.output, result)
        print(json.dumps(dict(complete=True, checks=len(result['checks']))))


if __name__ == '__main__':
    main()
