"""Sequential, current-input attention search with fixed repair budgets.

Each candidate is checked against the decoded oracle for its split schedule.
Changing split count changes online-P rounding; BF16 error and output drift are
recorded. No candidate may worsen BF16 relative L2 by more than 0.002 absolute.
Final recommendations require an independent five-round bench.py rerun.
"""
import argparse
from dataclasses import replace
import hashlib
import itertools
import json
import random
import statistics
from pathlib import Path

import torch
import triton.testing
from bench import ROOT, load_module, repeat_rows
from thunderworld_fp4 import preset, policy_from_head_atlas
from thunderworld_fp4.runtime import _forward


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--rep-ms', type=int, default=20)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    oracle = load_module('attention_oracle', ROOT/'packages/fp4-repair/tests/oracle.py')
    report = {'complete': False, 'gpu': torch.cuda.get_device_name(), 'rows': [],
              'source_sha256': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                                for f in (ROOT/'packages/fp4-repair/src/thunderworld_fp4').glob('*.py')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    modes = ['fp4_base', 'full_act2', 'balanced', 'quality', 'full_dual', 'head_guided']
    for asset, m, mode in itertools.product(
            ['attention_av_step17.pt', 'attention_robot_step17.pt'], [128, 560], modes):
        payload = torch.load(args.data/asset, weights_only=True)
        q = repeat_rows(payload['q'].cuda(), m, 1)
        k, v = payload['k'].cuda(), payload['v'].cuda()
        cfg = (policy_from_head_atlas(json.loads((args.data/'head_atlas.json').read_text()),
                                     coverage=.25, protected_coverage=.75).resolve(layer=payload['layer'])
               if mode == 'head_guided' else preset(mode))
        cfg = replace(cfg, query_tile=32, splits=2, num_warps=4)
        quotas = torch.tensor(cfg.head_counts(q.shape[2], k.shape[1]), device='cuda', dtype=torch.int32)
        baseline, mask, _ = _forward(q, k, v, cfg, counts_tensor=quotas, return_mask=True)
        ratio = q.shape[2]//k.shape[2]
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            bf = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1,2), k.repeat_interleave(ratio,2).transpose(1,2).contiguous(),
                v.repeat_interleave(ratio,2).transpose(1,2).contiguous()).transpose(1,2).float()
        baseline_error = float((baseline.float()-bf).norm()/bf.norm())
        row = {'asset': asset, 'm': m, 'preset': mode,
               'baseline_bf16_relative_l2': baseline_error, 'candidates': []}
        fns = {}
        for splits in [1, 2, 4, 8]:
            ref = oracle.reference(q,k,v,mask.int()*2+1,any(cfg.head_counts(q.shape[2],k.shape[1])),cfg.terms,splits)
            for bm, warps in itertools.product([16, 32, 64], [4, 8]):
                candidate = replace(cfg, query_tile=bm, splits=splits, num_warps=warps)
                key = f'm{bm}_s{splits}_w{warps}'
                entry = {'key': key, 'tile_m': bm, 'splits': splits, 'num_warps': warps}
                try:
                    fn = lambda c=candidate: _forward(q,k,v,c,counts_tensor=quotas)[0]
                    y = fn().float()
                    error = float((y-ref).norm()/ref.norm())
                    bf_error = float((y-bf).norm()/bf.norm())
                    entry.update(decoded_reference_relative_l2=error, bf16_relative_l2=bf_error,
                                 baseline_output_relative_l2=float((y-baseline.float()).norm()/baseline.float().norm()))
                    entry['passed'] = bool(torch.isfinite(y).all()) and error < .012 and bf_error <= baseline_error + .002
                    if entry['passed']:
                        fns[key] = fn
                except Exception as exc:
                    entry.update(passed=False, error=str(exc))
                row['candidates'].append(entry)
            del ref
        samples = {key: [] for key in fns}
        rng = random.Random(5090)
        for _ in range(args.rounds):
            order = list(fns)
            rng.shuffle(order)
            for key in order:
                samples[key].append(triton.testing.do_bench_cudagraph(fns[key], rep=args.rep_ms))
        for entry in row['candidates']:
            if entry['key'] in samples:
                values = samples[entry['key']]
                entry['latency_ms'] = dict(median=statistics.median(values), min=min(values), max=max(values), samples=values)
        good = [e for e in row['candidates'] if e['passed']]
        row['winner'] = min(good, key=lambda e:e['latency_ms']['median'])['key']
        report['rows'].append(row)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(asset, m, mode, row['winner'], round(min(e['latency_ms']['median'] for e in good)*1000,2), 'us', flush=True)
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
