"""Model- and method-driven paired quality report with explicit inference families."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
from awq_multiscene_report import paired_contrast
from baselines import sha, write
from dataset_infer import validate_rows
from dataset_report import METRICS, average


def report(root):
    manifest=root/'manifest.json';study=json.loads(manifest.read_text());digest=sha(manifest)
    progress=json.loads((root/'metrics/progress.json').read_text())
    expected=(len(study['methods'])-1)*len(study['cases'])
    if (not progress['complete'] or progress['completed']!=expected or progress['expected']!=expected or
            progress['identity']['manifest_sha256']!=digest):
        raise ValueError('Complete matching paired metric coverage required')
    inventory=json.loads((Path(__file__).resolve().parents[2]/'results/full_benchmark_inventory_20260919/inventory.json').read_text())
    targets=set(inventory['models'][study['model_name']]['generation_targets'])
    values=collections.defaultdict(dict);scenes=[];raw=[];costs=[];indices=[]
    for entry in study['cases']:
        cid=entry['case']['id'];reference_path=root/'generation/bf16'/cid/'result.json'
        reference=json.loads(reference_path.read_text())
        for name,spec in study['methods'].items():
            folder=root/'generation'/name/cid;path=folder/'result.json';receipt=json.loads(path.read_text())
            if not receipt['complete'] or receipt['identity']['manifest_sha256']!=digest:
                raise ValueError('Incomplete or mismatched generation')
            for field in ('case','image_sha256','seeds'):
                if receipt['identity'][field]!=entry[field] or reference['identity'][field]!=entry[field]:
                    raise ValueError('Unpaired inputs')
            if receipt['model_identity']!=reference['model_identity'] or set(receipt['targets'])!=targets:
                raise ValueError('Model or generated projection scope changed')
            adapter=receipt.get('adapter') or {}
            if spec.get('baseline_adapter',{}).get('method')=='awq':
                if (adapter.get('execution_type')!='offline_decoded_quality_reference' or
                    adapter.get('assets_sha256')!=spec['baseline_adapter']['assets_sha256'] or
                    adapter.get('storage_comparison_eligible') is not False):
                    raise ValueError('Unqualified exact AWQ reference')
            external=study.get('adapters',{}).get(name,{})
            if external.get('method')=='arcquant':
                if (adapter.get('execution_type')!='official_native_cosmos_adapter' or
                    adapter.get('channel_fraction')!=external['fraction']):
                    raise ValueError('Changed ARCQuant native adapter/fraction')
            if external.get('method')=='sharq':
                if (adapter.get('execution_type')!='official_native_adapter' or
                    adapter.get('extra_fusion')!=external['extra_fusion']):
                    raise ValueError('Changed SharQ native adapter')
            validate_rows(receipt,folder,receipt['identity'])
            ratios=[]
            for row in receipt['rows']:
                if (row['steps']!=35 or row['debug']!='off' or row['execution']!=spec['config']['execution'] or
                        row['performance_eligible']):
                    raise ValueError('Changed quality execution')
                ratio=row['actual_mma_ratio'];ratios.append(ratio)
                cap=spec['config'].get('compute_budget')
                if cap is not None and (ratio is None or ratio>cap):raise ValueError('Missing or exceeded budget')
            indices.append(dict(method=name,case=cid,receipt_sha256=sha(path),
                                outputs=[dict(seed=r['seed'],file=str(folder/r['outputs']),sha256=r['outputs_sha256']) for r in receipt['rows']]))
            costs.append(dict(method=name,case=cid,mma_ratios=ratios,
                peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in receipt['rows']),
                peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in receipt['rows']),performance_eligible=False,
                storage_comparison_eligible=spec.get('storage_comparison_eligible',True),
                execution_type=adapter.get('execution_type','native_runtime')))
            if name=='bf16':continue
            item=json.loads((root/'metrics'/name/(cid+'.json')).read_text())
            wanted=dict(identity=progress['identity'],candidate_sha256=sha(path),reference_sha256=sha(reference_path))
            if (not item['complete'] or item['provenance']!=wanted or len(item['rows'])!=len(entry['seeds']) or
                    {r['seed'] for r in item['rows']}!=set(entry['seeds'])):
                raise ValueError('Incomplete or changed paired metrics')
            for row in item['rows']:
                v=row['metrics']['video']
                if any(not np.isfinite(v[k]) for k in METRICS if k!='psnr_db'):
                    raise ValueError('Nonfinite metrics')
                if v['psnr_db'] is None and not v['psnr_exact']:raise ValueError('Missing PSNR')
                if v['psnr_db'] is not None and not np.isfinite(v['psnr_db']):raise ValueError('Nonfinite PSNR')
                values[entry['benchmark'],name][cid,row['seed']]=v
            raw.append(item)
            scenes.append(dict(method=name,benchmark=entry['benchmark'],category=entry['category'],case=cid,
                metrics={k:average([r['metrics']['video'][k] for r in item['rows']]) for k in METRICS}))
    summaries={};contrasts=[]
    for benchmark in sorted({e['benchmark'] for e in study['cases']}):
        entries=[e for e in study['cases'] if e['benchmark']==benchmark]
        categories={e['case']['id']:e['category'] for e in entries}
        sizes=collections.Counter(categories.values())
        inferential=study['role']=='final' and min(sizes.values())>=2
        summaries[benchmark]={}
        for name in study['methods']:
            if name=='bf16':continue
            own=[r for r in scenes if r['method']==name and r['benchmark']==benchmark]
            cat={c:{k:average([r['metrics'][k] for r in own if r['category']==c]) for k in METRICS}
                 for c in sorted(sizes)}
            summaries[benchmark][name]=dict(n_scenes=len(own),n_videos=sum(len(e['seeds']) for e in entries),categories=cat,
                category_macro={k:average([v[k] for v in cat.values()]) for k in METRICS})
        for pair in study['primary_comparisons']:
            comparison=paired_contrast(values[benchmark,pair['candidate']],values[benchmark,pair['reference']],categories,study['statistics'])
            eligible=inferential and pair['inferential']
            contrasts.append(dict(benchmark=benchmark,**pair,**comparison,
                inference_eligible=eligible,quality_gate_passed=bool(eligible and comparison['candidate_improves'])))
    result=dict(complete=True,model=study['model_name'],role=study['role'],manifest_sha256=digest,
        summaries=summaries,contrasts=contrasts,statistics=study['statistics'],seed_metrics=raw,scenes=scenes,costs=costs,
        performance_eligible=False,scope=study['scope'])
    out=root/'report';out.mkdir(exist_ok=True)
    write(out/'quality.json',result);write(out/'generation_index.json',dict(complete=True,rows=indices))
    count=sum(len(e['seeds']) for e in study['cases'])*len(study['methods'])
    lines=[f'# {study["model_name"]}: {study["role"]} quality', '',
        f'{len(study["cases"])} scenes, {len(study["methods"])} methods, {count} generated outputs. 35 steps, debug off.',
        'Metrics measure consistency with this model’s same-seed BF16 reference. Shared-GPU timings are ineligible.',
        'Means average seeds within scenes, then scenes within categories, then categories equally. Exact PSNR is +infinity (JSON null + exact marker).']
    if study['role']!='final':lines+=['','Development/smoke execution qualification only; no final quality superiority claim.']
    for benchmark, methods in summaries.items():
        lines+=['',f'## {benchmark}','','| Method | LPIPS ↓ | SSIM ↑ | PSNR dB ↑ | Relative L2 ↓ |','|---|---:|---:|---:|---:|',
                '| bf16 reference | 0 | 1 | +∞ | 0 |']
        for name,item in methods.items():
            v=item['category_macro'];psnr='+∞' if v['psnr_db'] is None else f'{v["psnr_db"]:.4f}'
            lines.append(f'| {name} | {v["lpips"]:.6f} | {v["ssim_mean"]:.6f} | {psnr} | {v["relative_l2"]:.6f} |')
    lines+=['','All paired deltas, ordinary 95% intervals, Bonferroni intervals, worst seeds and quality guards are in quality.json.',
            'Only the predeclared ARCQuant100 and FP8 comparison families can support corrected superiority claims; final results never feed calibration.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    report(p.parse_args().root)
