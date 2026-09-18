"""Qualify mixed QK/PV cross configurations; this is not a performance test."""
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
        from thunderworld_fp4 import RepairConfig, FP4Attention, fp4_attention
        torch.manual_seed(180918)
        torch.set_num_threads(8)
        rows = []
        with torch.inference_mode():
            for m, n in ((129, 417), (416, 1025), (960, 1001)):
                q = torch.randn(1, m, 4, 128, device='cuda', dtype=torch.bfloat16)
                k = torch.randn(1, n, 2, 128, device='cuda', dtype=torch.bfloat16)
                v = torch.randn_like(k)
                for qt, pt in ((4, 4), (3, 4), (4, 3), (3, 3)):
                    cfg = RepairConfig(coverage=1., terms=qt, pv_terms=pt, query_tile=32,
                                       transpose_v=True, sparse_pack=True)
                    graph = FP4Attention(cfg)
                    first = graph(q, k, v)
                    saved = first.clone()
                    for x, y, z in ((q, k, v), (q*.75, k+.125, v-.25),
                                    (torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v))):
                        expected = fp4_attention(x, y, z, cfg)
                        actual = graph(x, y, z)
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    torch.testing.assert_close(first, saved, atol=0, rtol=0)
                    graph.clear()
                    rows.append(dict(m=m, n=n, h=4, hk=2, qk_terms=qt, pv_terms=pt,
                        graph_eager_bitwise_equal=True, current_input_updated=True, output_lifetime=True))
        write(a.output, dict(complete=True, resource=resource, rows=rows, source_sha256=sha(__file__),
            performance_eligible=False, scope='Configuration/eager-Graph qualification using existing kernels; not a new independent numerical reference or sanitizer run.'))


if __name__ == '__main__':
    main()
