"""Representative frozen repair GEMMs for Nsight; never an E2E speed estimate."""
import argparse
import json
from pathlib import Path

from bundle import ROOT
from baselines import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--method', default='b2_joint')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    from thunderworld_fp4.cosmos.budget import RepairPlan
    from sm_batch_probe import make_method
    import torch
    manifest = json.loads((a.study/'manifest.json').read_text())
    capture = json.loads((a.capture/'manifest.json').read_text())
    spec = manifest['methods'][a.method]
    assets = Path(spec['assets'])
    am = json.loads((assets/'manifest.json').read_text())
    plan = RepairPlan.load(spec['config']['budget_plan'])
    if not (capture['complete'] and am['complete'] and
            capture['model_identity'] == am['model_identity'] == plan.value['model_identity']):
        raise ValueError('Frozen models differ')
    if sha(assets/'manifest.json') != spec['assets_sha256'] or sha(spec['config']['budget_plan']) != spec['plan_sha256']:
        raise ValueError('Frozen policy or assets changed')
    result = dict(complete=False, method=a.method, rows=[], source_sha256=sha(__file__),
        performance_eligible=False, scope='Layer 0, phase 0, CFG branch 0, real full-M BF16 teacher inputs. '
            'Same frozen native FP4 repair GEMM and launch; representative counters, not full-request saturation or speed.')
    with shared_worker():
        result['resource'] = select_shared_device()
        torch.set_num_threads(8)
        with torch.inference_mode():
            for suffix in ('self_attn.add_q_proj', 'self_attn.add_k_proj',
                           'mlp_moe_gen.up_proj', 'mlp_moe_gen.down_proj'):
                name = 'layers.0.'+suffix
                entry = next(r for r in capture['rows'] if r['kind'] == 'linear' and
                             r['name'] == name and r['phase'] == 0 and r['branch'] == 0)
                input_path = a.capture/entry['file']
                weight_path = a.capture/capture['weights'][name]
                asset_path = assets/am['modules'][name]['file']
                if sha(asset_path) != am['modules'][name]['sha256']:
                    raise ValueError('Changed asset tensor')
                x = torch.load(input_path, weights_only=True, map_location='cpu')['x'].cuda()
                weight = torch.load(weight_path, weights_only=True, map_location='cpu')
                asset = torch.load(asset_path, weights_only=True, map_location='cpu')
                layer = make_method('repair', weight, asset, None, plan, name, len(x), 0, 0)
                for _ in range(5):
                    expected = layer._compute(x)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push('steady')
                actual = layer._compute(x)
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                if not torch.equal(actual, expected):
                    raise ValueError('Counter probe changed the operator output')
                op = layer.bound
                result['rows'].append(dict(module=name, m=len(x), n=op.n, k=op.k,
                    bm=op.bm, stages=op.num_stages, a_blocks=op.na, w_blocks=len(op.weight_ids),
                    input_sha256=sha(input_path), weight_sha256=sha(weight_path), asset_sha256=sha(asset_path),
                    repeated_output_bitwise=True))
                write(a.output, result)
                print(json.dumps(result['rows'][-1]), flush=True)
                del layer, x, weight, asset, actual, expected
                torch.cuda.empty_cache()
    result['complete'] = True
    write(a.output, result)


if __name__ == '__main__':
    main()
