"""Diagnose native Marlin scale loss against the unchanged official AWQ asset.

The clipped reference explains an error; it never replaces the exact AWQ target
or relaxes the native numerical qualification threshold.
"""
import argparse
from pathlib import Path
import json

from baselines import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--module', default='layers.0.mlp_moe_gen.gate_proj')
    p.add_argument('--rows', type=int, default=129)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Keep the original diagnostic; choose a new output')
    manifest = json.loads((a.assets/'manifest.json').read_text())
    entry = manifest['modules'][a.module]
    file = a.assets/entry['file']
    if not manifest['complete'] or sha(file) != entry['sha256']:
        raise ValueError('Complete unchanged native export required')
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    with shared_worker():
        resource = select_shared_device()
        import torch
        import vllm
        from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4
        from thunderworld_fp4.cosmos.calibration import decode
        from awq_marlin import make_linear
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.inference_mode():
            asset = torch.load(file, weights_only=True)
            op = make_linear(asset)
            scales = asset['scales'].cuda().float()
            factor = marlin_utils_fp4._nvfp4_compute_scale_factor(scales, torch.bfloat16)
            dropped = (scales > 0) & (scales*factor*128 < 2)
            corrected = scales.masked_fill(dropped, 0)
            seed = 180918+a.rows
            x = torch.randn(a.rows, op.in_features, device='cuda', dtype=torch.bfloat16,
                generator=torch.Generator(device='cuda').manual_seed(seed))
            predicted = op(x).float()
            errors = {}
            for label, s in (('exact', scales), ('clipped', corrected)):
                weight = decode(asset['codes'].cuda(), s*asset['factor'].cuda(), 1.).bfloat16()
                gold = torch.nn.functional.linear(x*op.prescale, weight).float()
                errors[label] = float((predicted-gold).square().sum()/gold.square().sum().clamp_min(1e-20))
            finite = bool(predicted.isfinite().all())
            if not finite:
                raise ValueError('Nonfinite native output')
            write(a.output, dict(complete=True, module=a.module, shape=[a.rows,op.out_features,op.in_features],
                seed=seed, resource=resource, asset_sha256=sha(file), source_sha256=sha(__file__),
                vllm_version=vllm.__version__, marlin_source_sha256=sha(marlin_utils_fp4.__file__),
                nmse_exact_reference=errors['exact'], nmse_marlin_clipped_scale_reference=errors['clipped'],
                clipped_scale_fraction=float(dropped.float().mean()), marlin_scale_factor=factor,
                min_nonzero_scale=float(scales[scales>0].min()), max_scale=float(scales.max()),
                scale_cutoff=2/(128*factor), native_threshold=1e-4,
                native_qualified=errors['exact']<=1e-4, performance_eligible=False,
                scope='One-operator causal diagnostic, not an all-layer native qualification. Exact reference remains the target.'))


if __name__ == '__main__':
    main()
