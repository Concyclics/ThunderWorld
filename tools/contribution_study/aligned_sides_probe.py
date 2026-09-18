"""Qualify unchanged single-side arithmetic and full ACT2 graph/packing endpoints."""
import argparse
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from baselines import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from thunderworld_fp4.cosmos.resources import select_shared_device, shared_worker
    with shared_worker():
        resource = select_shared_device()
        import torch
        from thunderworld_fp4 import FP4Attention, RepairConfig, fp4_attention
        from thunderworld_fp4.cosmos.linears import RepairLinear
        from thunderworld_fp4.cosmos.settings import LinearSpec
        from thunderworld_fp4.cosmos.tuning import LaunchTable
        torch.set_num_threads(8)
        torch.manual_seed(271917)
        rows = []
        with torch.inference_mode():
            source = torch.nn.Linear(1536, 256, bias=False, device='cuda', dtype=torch.bfloat16)
            for side in ('a', 'w'):
                old = LinearSpec(1. if side == 'a' else 0., 1. if side == 'w' else 0.)
                new = LinearSpec(0., 0., activation_blocks=12 if side == 'a' else 0,
                                 weight_support=tuple(range(24)) if side == 'w' else ())
                plan = SimpleNamespace(linears={'0|0|probe': dict(a_blocks=new.activation_blocks,
                    w_blocks=list(new.weight_support), cross=False)})
                def controller(spec, budget_plan):
                    return SimpleNamespace(warming=True, table=LaunchTable(), budget_plan=budget_plan,
                        linear_spec=lambda *unused: spec, record_linear=lambda *unused: None)
                left_ctl, right_ctl = controller(old, None), controller(new, plan)
                left = RepairLinear(source, left_ctl, 'probe', old)
                right = RepairLinear(source, right_ctl, 'probe', new)
                if 'codes' in right.asset:
                    raise AssertionError('Planned endpoint retained a hidden dense residual plane')
                if right.asset['residual_indices'].numel() != (24 if side == 'w' else 0):
                    raise AssertionError('Wrong resident W1 support')
                for m in (129, 416, 960):
                    x = torch.randn(m, 1536, device='cuda', dtype=torch.bfloat16)
                    for value in (x, x * .75 + .125, torch.zeros_like(x)):
                        torch.testing.assert_close(left(value), right(value), atol=0, rtol=0)
                    rows.append(dict(kind='linear', side=side, m=m, n=256, k=1536,
                                     planned_vs_original_bitwise=True, input_variants=3))
            for m, n in ((129, 417), (416, 1025)):
                q = torch.randn(1, m, 4, 128, device='cuda', dtype=torch.bfloat16)
                k = torch.randn(1, n, 2, 128, device='cuda', dtype=torch.bfloat16)
                v = torch.randn_like(k)
                config = RepairConfig(coverage=1., terms=2, query_tile=32,
                                      transpose_v=True, sparse_pack=True)
                legacy = replace(config, sparse_pack=False)
                runner = FP4Attention(config)
                first = runner(q, k, v)
                saved = first.clone()
                for x, y, z in ((q, k, v), (q * .75, k + .125, v - .25),
                                (torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v))):
                    torch.testing.assert_close(runner(x, y, z), fp4_attention(x, y, z, legacy), atol=0, rtol=0)
                torch.testing.assert_close(first, saved, atol=0, rtol=0)
                runner.clear()
                rows.append(dict(kind='attention', m=m, n=n, h=4, hk=2, terms=2,
                    full_vs_selective_pack_bitwise=True, graph_current_input=True, output_lifetime=True))
        write(args.output, dict(complete=True, source_sha256=sha(__file__), resource=resource,
                               rows=rows, performance_eligible=False))


if __name__ == '__main__':
    main()
