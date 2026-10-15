"""Same-representation endpoints and attainable quality, without per-scene selection."""
import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np
from baselines import sha, write
from attention_base_report import bootstrap_indices
from selected_linear_study import NEW

KEYS = ('lpips','ssim_mean','psnr_db','relative_l2')
LABELS = dict(selected_1p5='Selected bilateral 1.5x', selected_2p0='Selected bilateral 2x',
    rtn_a_full='RTN full A (2x)', rtn_w_full='RTN full W (2x)',
    joint_a_full='Calibrated full A (2x)', joint_w_full='Calibrated full W (2x)')


def read(path):
    return json.loads(path.read_text())


def report(root):
    from selected_linear_runtime import source_identity
    manifest=root/'edge/final/manifest.json';study=read(manifest)
    if read(root/'study.json')['manifest_sha256']!=sha(manifest):
        raise ValueError('Changed frozen study')
    if read(root/'implementation.lock.json')['sources']!=source_identity():
        raise ValueError('Changed numerical sources')
    if not read(root/'edge/final/metrics/progress.json')['complete']:
        raise ValueError('Incomplete metrics')
    oldpath=Path(study['controls_manifest'])
    if sha(oldpath)!=study['controls_manifest_sha256']:
        raise ValueError('Changed controls')
    old=read(oldpath)
    if study['cases']!=old['cases'] or study['model']!=old['model']:
        raise ValueError('Unmatched controls')
    raw=[];scenes=[];costs=collections.defaultdict(list);audits=[]
    for e in study['cases']:
        cid=e['case']['id']
        gold=root/'edge/final/generation/bf16'/cid/'result.json'
        reference=read(gold)
        oldc0=read(oldpath.parent/'generation/linear_a'/cid/'compute.json')['base_mma']
        for name in NEW:
            folder=root/'edge/final/generation'/name/cid
            rp=folder/'result.json';r=read(rp);cp=folder/'compute.json';c=read(cp)
            if not r['complete'] or r['identity']['manifest_sha256']!=sha(manifest):
                raise ValueError('Different/incomplete generation')
            if any(r['identity'][k]!=reference['identity'][k] for k in ('case','seeds','image_sha256')):
                raise ValueError('Unpaired inputs')
            if r['model_identity']!=reference['model_identity'] or r['targets']!=reference['targets']:
                raise ValueError('Different model/scope')
            if r['adapter']['attention_repair'] is not False or r['adapter']['source_sha256']!=source_identity():
                raise ValueError('Wrong numerical adapter')
            if not c['feasible'] or c['linear_ratio']>c['linear_budget'] or c['attention_ratio']!=1 or c['base_mma']!=oldc0:
                raise ValueError('Unmatched/exceeded MMA budget')
            for row in r['rows']:
                if (row['steps'],row['calls'],row['debug'],row['execution'])!=(35,70,'off','graph'):
                    raise ValueError('Changed request protocol')
                if sha(folder/row['outputs'])!=row['outputs_sha256'] or row['compute_sha256']!=sha(cp):
                    raise ValueError('Corrupt generation or budget')
            mp=root/'edge/final/metrics'/name/(cid+'.json');m=read(mp)
            if not m['complete'] or m['provenance']['candidate_sha256']!=sha(rp) or m['provenance']['reference_sha256']!=sha(gold):
                raise ValueError('Changed metric inputs')
            if m['provenance']['identity']['manifest_sha256']!=sha(manifest) or sorted(x['seed'] for x in m['rows'])!=sorted(e['seeds']):
                raise ValueError('Wrong metric pairing')
            values=[]
            for row in m['rows']:
                value={k:float(row['metrics']['video'][k]) for k in KEYS}
                if not all(np.isfinite(x) for x in value.values()):
                    raise ValueError('Invalid metric')
                values.append(value)
                raw.append(dict(method=name,case_id=cid,benchmark=e['benchmark'],category=e['category'],seed=row['seed'],**value))
            scenes.append(dict(method=name,case_id=cid,benchmark=e['benchmark'],category=e['category'],
                **{k:float(np.mean([v[k] for v in values])) for k in KEYS}))
            lr=[x for x in c['rows'] if x['kind']=='linear']
            parts={term:sum(x['mma_by_dtype']['fp4'][term]*x['calls'] for x in lr)
                   for term in ('base','a','w','cross')}
            costs[name].append(dict(case_id=cid,ratio=c['ratio'],linear_ratio=c['linear_ratio'],
                linear_components=parts,w1_payload_bytes=c['w1_support_union_bytes'],
                packing_traffic=c['packing_traffic'],non_mma_work=c['non_mma_work']))
            audits.append(dict(method=name,case_id=cid,generation_sha256=sha(rp),metric_sha256=sha(mp),compute_sha256=sha(cp)))
    lookup={(r['method'],r['case_id']):r for r in scenes}
    ids=bootstrap_indices(study['cases'],10000,study['statistics']['bootstrap_seed'])
    alpha=.05/len(study['statistics']['primary_comparisons'])
    comparisons=[]
    for candidate,reference in study['statistics']['primary_comparisons']:
        metrics={}
        for key in KEYS:
            v=np.array([lookup[candidate,e['case']['id']][key]-lookup[reference,e['case']['id']][key] for e in study['cases']])
            draws=v[ids].mean(1);higher=key in ('ssim_mean','psnr_db');regression=-v if higher else v
            metrics[key]=dict(mean_difference=float(v.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),
                ci_familywise=np.quantile(draws,[alpha/2,1-alpha/2]).tolist(),
                improved_scenes=int((regression<0).sum()),worst_scene=study['cases'][int(regression.argmax())]['case']['id'],
                worst_scene_regression=float(regression.max()))
        same=(study['methods'][candidate]['origin']==study['methods'][reference]['origin'])
        guard=metrics['ssim_mean']['mean_difference']>=0 and metrics['relative_l2']['mean_difference']<=0
        comparisons.append(dict(candidate=candidate,reference=reference,same_w0_prescale=same,metrics=metrics,
            lpips_familywise_improvement=metrics['lpips']['ci_familywise'][1]<0,
            ssim_l2_guard=guard,all_four_mean_improvement=guard and metrics['psnr_db']['mean_difference']>=0 and metrics['lpips']['mean_difference']<0))
    methods={}
    for name in NEW:
        vals=[lookup[name,e['case']['id']] for e in study['cases']]
        metrics={}
        for k in KEYS:
            v=np.array([r[k] for r in vals])
            metrics[k]=dict(mean=float(v.mean()),ci95=np.quantile(v[ids].mean(1),[.025,.975]).tolist())
        methods[name]=dict(metrics=metrics,
            mma_ratio_range=[min(r['ratio'] for r in costs[name]),max(r['ratio'] for r in costs[name])],
            linear_ratio_range=[min(r['linear_ratio'] for r in costs[name]),max(r['linear_ratio'] for r in costs[name])],
            w1_payload_bytes=costs[name][0]['w1_payload_bytes'],
            by_benchmark={b:{k:float(np.mean([r[k] for r in vals if r['benchmark']==b])) for k in KEYS}
                          for b in ('rbench','paibench')})
    best_a=min(('rtn_a_full','joint_a_full'),key=lambda n:methods[n]['metrics']['lpips']['mean'])
    best_w=min(('rtn_w_full','joint_w_full'),key=lambda n:methods[n]['metrics']['lpips']['mean'])
    result=dict(complete=True,scenes=72,seeds=144,methods=methods,comparisons=comparisons,
        best_measured_single_sides=dict(activation=best_a,weight=best_w),costs=dict(costs),audit=audits,
        manifest_sha256=sha(manifest),performance_eligible=False,
        claim_scope='Empirical attainable quality over two frozen representation banks and selected linear policies; no per-scene best-of, no theoretical upper bound, no test-set fitting. Follow-up on an inspected cohort.')
    out=root/'report';out.mkdir(exist_ok=True)
    write(out/'results.json',result)
    for name,rows in [('per_seed',raw),('per_scene',scenes)]:
        with (out/(name+'.csv')).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Selected 双侧与单侧可达质量：共同无修复 FP4 attention','',
        'Edge，RBench 36 + PAI-Bench 36 场景，每场景两个固定种子，35 steps / CFG 6 / Graph / debug off。所有指标为相对同种子 BF16 的一致性；不是绝对任务质量。',
        '', '1.5×、2× 是线性 MMA 上限。保留已冻结 selected 线性策略；释放的 attention 预算没有重新搜索或分配。两套资产分别补齐 A-only/W-only：336 个模块的 W0、outer scale 和已有 W1 均逐字节一致。',
        '', '这里比较已搜索范围内的经验可达质量。该 72 场景已用于观察前一轮固定配额结果，属于后续验证，不宣称新的独立最终测试；未按场景选择配置，未用这些输出改动策略。',
        '', '| 配置 | LPIPS ↓ | SSIM ↑ | PSNR ↑ | L2 ↓ | 线性 MMA | 整请求控制 MMA | W1 MiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,v in methods.items():
        mm=v['metrics'];lo,hi=v['mma_ratio_range']
        lines.append(f"| {LABELS[name]} | {mm['lpips']['mean']:.6f} | {mm['ssim_mean']['mean']:.6f} | {mm['psnr_db']['mean']:.3f} | {mm['relative_l2']['mean']:.6f} | {v['linear_ratio_range'][0]:.6f} | {lo:.6f}–{hi:.6f} | {v['w1_payload_bytes']/2**20:.3f} |")
    lines+=['','最佳已测 A-only：`'+best_a+'`；最佳已测 W-only：`'+best_w+'`。这是全体场景均值选择，不是逐场景拼接。','',
        '| 双侧 | 单侧对照 | 同 W0 / prescale | ΔLPIPS (双侧−单侧) | 配对 95% CI | 八比较校正 CI | LPIPS 校正后改善且 SSIM/L2 不退化 |',
        '|---|---|---|---:|---|---|---|']
    for c in comparisons:
        v=c['metrics']['lpips'];ok=c['lpips_familywise_improvement'] and c['ssim_l2_guard']
        lines.append(f"| {c['candidate']} | {c['reference']} | {'是' if c['same_w0_prescale'] else '否'} | {v['mean_difference']:+.6f} | [{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}] | [{v['ci_familywise'][0]:+.6f}, {v['ci_familywise'][1]:+.6f}] | {'通过' if ok else '未通过'} |")
    lines+=['','统计先对同场景两个种子取均值，再按 benchmark × category 分层 bootstrap 10,000 次。八个预指定 LPIPS 比较同时给出 Bonferroni 家族置信区间；完整 JSON 包括 SSIM、PSNR、L2、最差场景和逐请求计算量。',
        '', '同 W0 对照用于检验修复覆盖分配；跨资产结果用于比较冻结配置中的可达效果。单侧只覆盖两套已选资产的全量端点，没有完成针对单侧的所有量化搜索，不能称为单侧方法的理论上限。',
        '', '所有 base attention blocks 保留，Q/P/K/V 修复全部关闭。只报告计算量与 W1 payload；共享 GPU 用时不作为速度证据，W1 payload 也不等于整个模型占用。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    plot(out,methods)
    return result


def plot(out,methods):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(9.4,3.3),sharey=True)
    for ax,names,title in zip(axs,[['rtn_a_full','rtn_w_full','selected_1p5'],
                                  ['joint_a_full','joint_w_full','selected_2p0']],
                              ['RTN representation','Calibrated representation']):
        v=[methods[n]['metrics']['lpips'] for n in names]
        means=np.array([x['mean'] for x in v]);ci=np.array([x['ci95'] for x in v])
        ax.bar(range(3),means,color=['#5592b8','#a9a6a1','#d3863d'],width=.64)
        ax.errorbar(range(3),means,yerr=np.stack((means-ci[:,0],ci[:,1]-means)),fmt='none',color='#242424',capsize=3)
        for i,y in enumerate(means):ax.text(i,ci[i,1]+.006,f'{y:.4f}',ha='center',fontsize=9)
        ax.set_xticks(range(3),['Full A\n2x cap','Full W\n2x cap','Selected A/W\n'+('1.5x cap' if names[-1]=='selected_1p5' else '2x cap')])
        ax.set_title(title,fontsize=11);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axs[0].set_ylabel('LPIPS to paired BF16 ↓')
    ymax=max(x['metrics']['lpips']['ci95'][1] for x in methods.values())
    axs[0].set_ylim(0,ymax*1.2)
    fig.suptitle('Fixed base FP4 attention · 72 scenes × 2 seeds · scene bootstrap 95% CI',fontsize=10)
    fig.tight_layout()
    for suffix in ('pdf','svg','png'):fig.savefig(out/f'selected_quality.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();r=report(args.root.resolve());print(json.dumps(dict(complete=r['complete'],methods=r['methods'])))
