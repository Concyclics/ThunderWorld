"""Audit Nano performance evidence without importing Torch or running the GPU."""
import argparse
import json
from pathlib import Path

from build_baselines import sha,write
from standard_comparison import checked_run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args();root=a.campaign
    manifest=json.loads((root/'manifest.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    if manifest.get('model_key')!='nano' or not summary['complete']:raise ValueError('Incomplete Nano comparison')
    if not summary['common_auxiliary_runtime']:raise ValueError('Auxiliary runtime differs between methods')
    qual=root/'qualification'
    linear=json.loads((qual/'linear.json').read_text());attention=json.loads((qual/'attention.json').read_text())
    if not linear['complete'] or len(linear['rows'])!=42 or len(linear['sparse_packing'])!=18:
        raise ValueError('Incomplete Nano linear qualification')
    if linear.get('model_shapes')!='nano' or not all(r['previous_bitwise_equal'] for r in linear['rows']):
        raise ValueError('Wrong shapes or changed mainloop arithmetic')
    if not attention['complete'] or not attention.get('nano_heads') or len(attention['tests'])!=24:
        raise ValueError('Incomplete Nano attention qualification')
    if not all(r['native_v3_bitwise_equal'] and r['native_transposed_v_bitwise_equal'] for r in attention['tests']):
        raise ValueError('Attention layout/group changed output')
    for name in ('memcheck-linear','racecheck-linear','memcheck-attention','racecheck-attention'):
        d=json.loads((qual/(name+'.json')).read_text());log=(qual/(name+'.log')).read_text()
        key,ref=('library_sha256',linear) if name.endswith('linear') else ('kernel_sha256',attention)
        if not d['complete'] or d[key]!=ref[key]:raise ValueError('Sanitizer artifact mismatch: '+name)
        clean='ERROR SUMMARY: 0 errors' if name.startswith('memcheck') else '0 hazards displayed (0 errors, 0 warnings)'
        if clean not in log:raise ValueError('Unclean sanitizer: '+name)
    requests=0;ratios={};dispatch={};sources={};capture_counts={}
    for rnd in manifest['rounds']:
        for name in manifest['methods']:
            path=root/'runs'/f"round-{rnd['index']:02d}"/name/'result.json'
            d=checked_run(path,manifest,name);sources[str(path)]=sha(path)
            if manifest['methods'][name]['variant']:
                if d['effective_backend_override']['library_sha256']!=linear['library_sha256']:
                    raise ValueError('Unqualified measured library')
                if d['variant']['kernel_sha256']!=attention['kernel_sha256']:
                    raise ValueError('Unqualified measured attention')
            for row in d['rows']:
                requests+=1
                if not row['sha256'].get('video'):raise ValueError('No video output')
                ledger=row.get('compute_ledger')
                if ledger:
                    ratios.setdefault(name,set()).add(ledger['ratio'])
                    projections=[r for r in ledger['rows'] if r['kind']=='linear']
                    if sum(r['calls'] for r in projections)!=36*7*35*2:raise ValueError('Missing generated projections')
                    if any(r.get('execution_backend')!='cutlass-mainloop-fused' or
                           r.get('gemm_kernel_launches')!=1 or r.get('cutlass_fp32_gmem_scratch_bytes')!=0 for r in projections):
                        raise ValueError('Unexpected fallback or split repair execution')
                    dispatch[name]=dict(generation_projection_calls=17640,backend='cutlass-mainloop-fused',
                        source='Prepared execution ledger, not hardware instruction counters')
            for case in manifest['cases']:
                case_rows=[r for r in d['rows'] if r['case_index']==case]
                # CosmosRunner reports cumulative len(engine.cache), not the
                # number captured during this request. Do not require zero.
                counts={r['captures'] for r in case_rows}
                if len(counts)!=1:raise ValueError('Graph cache grew between timed repeats')
                capture_counts[f"{rnd['index']}:{name}:{case}"]=next(iter(counts))
                hashes=[r['sha256']['video'] for r in case_rows]
                if len(set(hashes))!=len(manifest['seeds']):raise ValueError('Identical outputs across seeds; inspect current-input execution')
    receipt=dict(complete=True,quality_evaluation=False,performance_eligible=False,
        requests=requests,rounds=len(manifest['rounds']),methods=len(manifest['methods']),
        source_sha256=sha(Path(__file__)),summary_sha256=sha(root/'summary.json'),
        linear_cases=42,sparse_packing_cases=18,attention_cases=24,memcheck=True,racecheck=True,
        all_measured_fp4_artifacts_qualified=True,stable_graph_count_between_measured_repeats=True,
        cumulative_graph_counts=capture_counts,
        common_auxiliary_runtime=True,distinct_output_hashes_per_seed=True,
        budget_ratios={k:sorted(v) for k,v in ratios.items()},dispatch=dispatch,sources=sources,
        limitation='Speed/memory pilot with frozen assets; correctness checks do not establish video quality. '
            'A full request is warmed before each case; the archived rows record cumulative graph counts, not a pre/post counter for the first timed request.')
    write(root/'verification.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('sources',)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
