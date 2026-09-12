"""Audit frozen final cohorts and report the observed quality/compute/storage frontier."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from capture_budget import atomic_json
from thunderworld_fp4.cosmos.cohorts import Cohort
from thunderworld_fp4.cosmos.quality import gate


def frontier(rows,metric='lpips'):
    """Finite three-objective frontier; no claim about unsearched configurations."""
    keys=(metric,'mma','prepared_weight_bytes')
    return [r['name'] for r in rows if not any(
        all(q[k]<=r[k] for k in keys) and any(q[k]<r[k] for k in keys)
        for q in rows if q is not r)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('quality','matrix','selection','cohort','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();cohort=Cohort.load(a.cohort)
    quality=json.loads(a.quality.read_text());matrix=json.loads(a.matrix.read_text());selection=json.loads(a.selection.read_text())
    if not quality['complete'] or quality['suite']!='quality' or not matrix['complete']:
        raise ValueError('Complete frozen final quality required')
    if quality['matrix_sha256']!=hashlib.sha256(a.matrix.read_bytes()).hexdigest():
        raise ValueError('Quality/matrix mismatch')
    if matrix['cohort_sha256']!=cohort.fingerprint or selection['cohort_sha256']!=cohort.fingerprint:
        raise ValueError('Cohort mismatch')
    if a.output.exists():raise FileExistsError(a.output)
    jobs={j['config']:j for j in matrix['jobs']}
    def samples(name):
        rows=[r for r in quality['rows'] if r['config']==name]
        cohort.validate('final',[r['seed'] for r in rows],complete=True)
        return rows
    summaries={};runs={}
    for name,job in jobs.items():
        run=json.loads((Path(job['output'])/'result.json').read_text());runs[name]=run
        if not run['complete'] or any(r['debug']!='off' or r.get('performance_eligible',True) for r in run['rows']):
            raise ValueError('Incomplete/debug/non-shared final run')
        cohort.validate('final',[r['seed'] for r in run['rows']],complete=True)
        metrics={} if name=='bf16' else {modality:{metric:float(np.mean([r['metrics'][modality][metric] for r in samples(name)]))
            for metric in samples(name)[0]['metrics'][modality]} for modality in samples(name)[0]['metrics']}
        ledger=run.get('compute_ledger')
        if ledger:
            digest=hashlib.sha256(json.dumps(ledger,sort_keys=True).encode()).hexdigest()
            if not ledger['feasible'] or any(r.get('compute_ledger_sha256')!=digest for r in run['rows']):
                raise ValueError('Unaudited request budget')
        summaries[name]=dict(metrics=metrics,compute=None if ledger is None else {k:v for k,v in ledger.items() if k not in ('rows','w1_support_union_blocks')},
            prepared_weight_bytes=run['prepared_weight_bytes'],resident_model_state_bytes=run.get('resident_model_state_bytes'),
            support_buffer_bytes=run.get('support_buffer_bytes'),memory=run['memory'],
            request_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in run['rows']),
            request_peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in run['rows']),
            live_non_model_allocated_after_warmup_bytes=run['memory']['after_warmup']['allocated_bytes']-run.get('resident_model_state_bytes',0),
            non_model_memory_scope='Combined live activation/scratch/Graph allocations, not an isolated scratch peak',
            gqa_warmup_samples=run.get('warmup_gqa_union_density'),gqa_scope=run.get('packing_density_scope'),
            precision=run['config']['precision'],attention=run['config']['attention_backend'])
    if 'full_dual' in runs and 'fp4' in runs:
        anchor=runs['full_dual'];base=runs['fp4'];cfg=anchor['config']['attention']['default']
        def shapes(run):
            return {(r['module'],r['branch'],r['m'],r['n'],r['k'],r['launch']['bm']) for r in run['observed_shapes']}
        full_linear=all(r['actual_a']==1 and r['actual_w']==1 and not json.loads(r['repair'])['cross'] for r in anchor['observed_shapes'])
        full_attention=cfg['coverage']==1 and cfg['terms']==3 and cfg.get('pv_terms') in (None,3) and not cfg.get('head_budgets') and cfg.get('head_quotas') is None and cfg.get('pv_head_quotas') is None and not anchor['config']['attention'].get('rules')
        if full_linear and full_attention and shapes(anchor)==shapes(base) and cfg['query_tile']==base['config']['attention']['default']['query_tile']:
            c0=base['compute_ledger']['base_mma']
            summaries['full_dual']['analytical_anchor_mma']=dict(base_mma=c0,total_mma=3*c0,ratio=3.,mma_by_dtype=dict(fp4=3*c0,bf16=0),
                scope='Derived full dual cost: matched observed FP4 tiles, base plus both first-order terms, no selector/cross; outside the four budgets')
    comparisons=[];points=[]
    for decision in selection['decisions']:
        name=decision['selected'];control=decision['control'];rows=samples(name);task=rows[0]['task']
        same=gate(rows,samples(control),task,cohort=cohort)
        same['scope']='Paired consistency against the validation-selected uncalibrated control at the same compute cap'
        fp8=gate(rows,samples('fp8'),task,cohort=cohort)
        comparisons.append(dict(budget=decision['budget'],selected=name,control=control,same_budget_gate=same,fp8_gate=fp8,
            actual_mma_ratios=dict(selected=summaries[name]['compute']['ratio'],control=summaries[control]['compute']['ratio']),
            comparison_scope='Same hard cap; actual issued MMA can differ due to allocation and integer quotas'))
    modality,metric={'policy':('action','relative_l2'),'av':('waveform','log_spectral_l1')}.get(cohort.value['task'],('video','lpips'))
    for name,summary in summaries.items():
        if summary['compute']:
            points.append(dict(name=name,lpips=summary['metrics']['video']['lpips'],mma=summary['compute']['total_mma'],
                primary_error=summary['metrics'][modality][metric],
                prepared_weight_bytes=summary['prepared_weight_bytes']))
    title=f"{cohort.value['model']} {cohort.value['task']}"
    result=dict(complete=True,scope=title+' final new seeds on existing scenes; observed compute frontier only',
        performance_eligible=False,resource_mode='shared-calibration',cohort_sha256=cohort.fingerprint,
        quality_sha256=hashlib.sha256(a.quality.read_bytes()).hexdigest(),selection_sha256=hashlib.sha256(a.selection.read_bytes()).hexdigest(),
        summaries=summaries,comparisons=comparisons,frontier=frontier(points,'primary_error'),frontier_objectives=[f'mean {modality}.{metric}','issued MMA FLOPs','prepared quantized weight bytes'],
        expand_to_next_task=any(c['same_budget_gate']['passed'] for c in comparisons),
        limitations=['Shared GPU timings cannot establish acceleration or latency Pareto optimality.',
            'Packing traffic is an implementation-derived bound; fields ending in upper are not measured savings.',
            'Residual address space is still allocated in full; sparse writes do not imply proportional scratch reduction.',
            'Prepared weight bytes cover replaced projections; resident model state and request peaks are reported separately.',
            'New seeds in existing scenes do not establish cross-scene generalization.'])
    a.output.mkdir(parents=True);atomic_json(a.output/'report.json',result)
    lines=['# '+title+' compute-constrained FP4 final results','',
        'All runs used shared PRO 6000 GPUs, 35 steps, identical inputs and Graph execution with debug off. No latency speedup is claimed.','',
        '| Budget | Selected | LPIPS ↓ | Control LPIPS ↓ | SSIM ↑ | Relative L2 ↓ | Same-budget gate | FP8 gate | MMA/C0 (selected/control) | W1 MiB |',
        '|---|---|---:|---:|---:|---:|---|---|---:|---:|']
    for c in comparisons:
        s=summaries[c['selected']];m=s['metrics']['video'];ledger=s['compute']
        control=summaries[c['control']]
        lines.append(f"| {c['budget']:g} | {c['selected']} | {m['lpips']:.6f} | {control['metrics']['video']['lpips']:.6f} | {m['ssim_mean']:.6f} | {m['relative_l2']:.6f} | {c['same_budget_gate']['passed']} | {c['fp8_gate']['passed']} | {ledger['ratio']:.6f}/{control['compute']['ratio']:.6f} | {ledger['w1_support_union_bytes']/2**20:.2f} |")
    if modality!='video':
        lines+=['',f'| Budget | Selected {modality}.{metric} ↓ | Control ↓ |','|---|---:|---:|']
        for c in comparisons:
            lines.append(f"| {c['budget']:g} | {summaries[c['selected']]['metrics'][modality][metric]:.8f} | {summaries[c['control']]['metrics'][modality][metric]:.8f} |")
    lines+=['','Paired 95% bootstrap intervals, worst-seed degradation, complete cost and memory breakdowns are in report.json.',
        '', 'Observed nondominated configurations: '+', '.join(result['frontier']), '',*result['limitations']]
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
