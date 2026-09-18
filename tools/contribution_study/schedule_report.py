"""Paired schedule quality and shape-safe compute/storage accounting."""
import argparse
import collections
import gzip
import json
from pathlib import Path
import numpy as np
from baselines import sha,write
from dataset_report import stratified_interval
from schedule_common import CONTROLS

FIELDS=('lpips','ssim_mean','relative_l2','psnr_db')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--split',choices=['development','validation'],default='development');a=p.parse_args()
    root=a.study/a.split;manifest=json.loads((root/'manifest.json').read_text());progress=json.loads((root/'metrics/progress.json').read_text())
    if not progress['complete'] or progress['identity']['manifest_sha256']!=sha(root/'manifest.json'):raise ValueError('Complete matched metrics required')
    summaries={};raw=[];costs={};by_seed=collections.defaultdict(dict);scene_groups={}
    for method,spec in manifest['methods'].items():
        records=[];bounds=[];allocated=[];reserved=[];captures=[];ledger_summaries=[]
        for case in manifest['cases']:
            name=case['case']['id'];folder=root/'generation'/method/name;receipt=json.loads((folder/'result.json').read_text())
            if not receipt['complete']:raise ValueError('Incomplete generation')
            for row in receipt['rows']:
                if row['performance_eligible'] or row['debug']!='off' or row['steps']!=35:raise ValueError('Unexpected execution conditions')
                allocated.append(row['peak_allocated_bytes']);reserved.append(row['peak_reserved_bytes'])
                captures.append(row['captures'])
                if row['actual_mma_ratio'] is not None:
                    if row['actual_mma_ratio']>spec['budget']:raise ValueError('Request budget violated')
                    bounds.append(row['actual_mma_ratio'])
            if (folder/'compute.json').exists():
                if any(row['compute_sha256']!=sha(folder/'compute.json') for row in receipt['rows']):raise ValueError('Changed request cost')
                ledger=json.loads((folder/'compute.json').read_text());ledger_summaries.append({k:v for k,v in ledger.items() if k not in ('rows','w1_support_union_blocks')})
            if method=='bf16':continue
            path=root/'metrics'/method/(name+'.json');data=json.loads(path.read_text())
            if not data['complete'] or data['provenance']['identity']!=progress['identity']:raise ValueError('Changed quality implementation')
            if data['provenance']['candidate_sha256']!=sha(folder/'result.json') or data['provenance']['reference_sha256']!=sha(root/'generation/bf16'/name/'result.json'):raise ValueError('Changed generated outputs')
            if {r['seed'] for r in data['rows']}!=set(case['seeds']) or len(data['rows'])!=len(case['seeds']):raise ValueError('Unmatched seeds')
            raw.append(data)
            for row in data['rows']:by_seed[case['benchmark'],method][name,row['seed']]=row['metrics']['video']
            values={field:float(np.mean([r['metrics']['video'][field] for r in data['rows']])) for field in FIELDS}
            if not all(np.isfinite(value) for value in values.values()):raise ValueError('Finite paired quality metrics required')
            records.append(dict(benchmark=case['benchmark'],category=case['category'],case=name,metrics=values))
        for benchmark in ('rbench','paibench'):
            subset=[r for r in records if r['benchmark']==benchmark]
            if not subset:continue
            categories=sorted({r['category'] for r in subset})
            summaries.setdefault(benchmark,{})[method]={field:float(np.mean([np.mean([r['metrics'][field] for r in subset if r['category']==category]) for category in categories])) for field in FIELDS}
            scene_groups[benchmark,method]=subset
        costs[method]=dict(request_bound_range=[min(bounds),max(bounds)] if bounds else None,
                           peak_allocated_bytes=[min(allocated),max(allocated)],peak_reserved_bytes=[min(reserved),max(reserved)],
                           graph_captures_range=[min(captures),max(captures)],
                           offline=spec.get('cost'),request_ledgers=ledger_summaries,
                           scope='Cross MMA is an upper bound for arbitrary current A selection. Non-MMA and packing modeled separately; no shared-load speed claim.')
    pairs={}
    for (benchmark,method),rows in scene_groups.items():
        refs=['fp8']
        if method.startswith('b'):
            budget=manifest['methods'][method]['budget'];refs.append(CONTROLS[budget])
        for reference in refs:
            if reference==method or (benchmark,reference) not in scene_groups:continue
            old={r['case']:r for r in scene_groups[benchmark,reference]};detail={}
            for field in FIELDS:
                groups=collections.defaultdict(list)
                for row in rows:groups[row['category']].append(row['metrics'][field]-old[row['case']]['metrics'][field])
                stats=dict(mean_delta=float(np.mean([np.mean(v) for v in groups.values()])))
                if a.split=='validation':
                    if any(len(v)<2 for v in groups.values()):raise ValueError('At least two independent scenes/category required')
                    stats.update(stratified_interval(groups,replicates=10000,seed=171917,family=12))
                seed_values=by_seed[benchmark,method];reference_values=by_seed[benchmark,reference]
                if seed_values.keys()!=reference_values.keys():raise ValueError('Unmatched paired videos')
                diffs=[(k,v[field]-reference_values[k][field]) for k,v in seed_values.items()]
                lower=field in ('lpips','relative_l2');key,worst=(max if lower else min)(diffs,key=lambda x:x[1])
                stats.update(worst_seed=key[1],worst_case=key[0],worst_delta=worst,
                             degraded_videos=sum(d>0 if lower else d<0 for _,d in diffs),videos=len(diffs))
                detail[field]=stats
            pairs[f'{benchmark}|{method}|{reference}']=detail
    choices={}
    for cap,control in CONTROLS.items():
        if control not in manifest['methods']:continue
        candidates=[name for name,spec in manifest['methods'].items() if spec.get('budget')==cap and name!=control]
        eligible=[];status={}
        for name in candidates:
            failures={}
            for b in summaries:
                new,old=summaries[b][name],summaries[b][control]
                failures[b]=[metric for metric,failed in (
                    ('lpips',new['lpips']>=old['lpips']),('ssim_mean',new['ssim_mean']<old['ssim_mean']),
                    ('relative_l2',new['relative_l2']>old['relative_l2'])) if failed]
            passed=not any(failures.values())
            statistical=passed and a.split=='validation' and all(pairs[f'{b}|{name}|{control}']['lpips']['familywise_ci'][1]<0 for b in summaries)
            beats_fp8=passed and a.split=='validation' and all(pairs[f'{b}|{name}|fp8']['lpips']['familywise_ci'][1]<0 and
                summaries[b][name]['ssim_mean']>=summaries[b]['fp8']['ssim_mean'] and summaries[b][name]['relative_l2']<=summaries[b]['fp8']['relative_l2'] for b in summaries)
            status[name]=dict(mean_guard=passed,mean_guard_failures=failures,
                              statistical_guard=statistical,exceeds_fp8=beats_fp8)
            if passed:eligible.append(name)
        selected=min(eligible,key=lambda name:sum(summaries[b][name]['lpips'] for b in summaries)) if eligible else control
        choices[str(cap)]=dict(control=control,selected=selected,status=status)
    result=dict(complete=True,split=a.split,summaries=summaries,pairs=pairs,costs=costs,raw=raw,budgets=choices,
                aliases=manifest.get('aliases',{}),manifest_sha256=sha(root/'manifest.json'),performance_eligible=False,
                statistics=dict(bootstrap='scene cluster, category stratified; seeds averaged within scene',family=12,replicates=10000) if a.split=='validation' else None,
                scope='BF16 output consistency, not official benchmark/task success. Development selects; validation is untouched by fitting; final remains reserved.')
    report=root/'report';report.mkdir(exist_ok=True);write(report/'quality.json',result)
    write(report/'selection.json',dict(complete=True,split=a.split,budgets=choices,quality_sha256=sha(report/'quality.json')))
    lines=[f'# 联合修复调度：{a.split}', '',f'{len(manifest["cases"])} 场景，每场景 {len(manifest["cases"][0]["seeds"])} 个固定种子；35 steps，CFG 6，Edge I2V，Graph/debug off。相对 BF16 的输出一致性；共享 GPU 不报告加速。','',
           '开发集只报告均值。验证集按场景聚类、类别分层做配对 bootstrap，同时给出 95% 区间和 12 项比较的保守 familywise 区间；完整原始记录见 JSON。']
    for b,methods in summaries.items():
        lines+=['',f'## {b}','','| 配置 | LPIPS ↓ | SSIM ↑ | L2 ↓ | PSNR dB ↑ |','|---|---:|---:|---:|---:|']
        for name,values in methods.items():lines.append(f'| {name} | '+ ' | '.join(f'{values[field]:.6f}' for field in FIELDS)+' |')
    lines+=['','## 开发选择或验证判断','','```json',json.dumps(choices,ensure_ascii=False,indent=2),'```',
            '','## 计算上界与显存','','| 配置 | 请求 MMA/C0 上界范围 | W1 MiB | allocated 峰值 GiB | Graph 数 |','|---|---:|---:|---:|---:|']
    for name,cost in costs.items():
        bounds=cost['request_bound_range'];mma='—' if not bounds else '–'.join(f'{v:.6f}' for v in bounds)
        w1='—' if not cost['offline'] else f'{cost["offline"]["w1_bytes"]/2**20:.3f}'
        lines.append(f'| {name} | {mma} | {w1} | '+ '–'.join(f'{v/2**30:.3f}' for v in cost['peak_allocated_bytes'])+' | '+
                     '–'.join(str(v) for v in cost['graph_captures_range'])+' |')
    lines+=['','相同 W1 存储上限，不等于相同 Graph/scratch 占用；跨阶段组合会增加 Graph 数量。跨项 MMA 采用可证明上界，不能称为已测得的实际指令数。',
            '', '去重的相同执行策略：`'+json.dumps(manifest.get('aliases',{}),ensure_ascii=False)+'`。']
    (report/'REPORT.md').write_text('\n'.join(lines)+'\n');a.archive.mkdir(exist_ok=True,parents=True)
    (a.archive/(a.split.upper()+'.md')).write_text('\n'.join(lines)+'\n')
    for label,path in [('quality',report/'quality.json'),('manifest',root/'manifest.json'),('selection',report/'selection.json')]:
        (a.archive/f'{a.split}_{label}.json.gz').write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    print(json.dumps(choices))


if __name__=='__main__':main()
