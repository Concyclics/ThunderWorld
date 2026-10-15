"""Audit and report the frozen low-budget extension, retaining negative outcomes."""
import argparse
import csv
import html
import json
import os
from pathlib import Path
import numpy as np
from baselines import sha,write
from expansion_two_term import load
from expansion_two_term_report import combined
from expansion_report import LABELS,METRICS,collect,interval

NEW={'fp4_both_1p5':'Both sides (1.5x cap)','fp4_both_2p0':'Both sides (2x cap)'}
ORDER=['fp4_base','fp4_two_a','fp4_two_w','fp4_both_1p5','fp4_both_2p0','fp4_three','fp8','awq_reference']


def report(root):
    LABELS.update(NEW)
    study,parent,base=load(root)
    side=Path(study['preregistration']['controls_root'])
    side_study,_,old_rows,old=combined(side)
    for key in ('cases','model','steps','protocol','statistics'):
        if side_study[key]!=study[key]:raise ValueError('Control mismatch: '+key)
    _,new_rows,new=collect(root)
    rows=[r for r in old_rows+new_rows if r['method'] in ORDER]
    lookup={(r['method'],r['case_id']):r for r in rows}
    if len(rows)!=80 or len(lookup)!=80:raise ValueError('Incomplete paired matrix')
    costs={n:[] for n in NEW};audit=[];base_by_case={}
    for entry in study['cases']:
        cid=entry['case']['id']
        gold_path=parent/'edge/final/generation/bf16'/cid/'result.json'
        gold=json.loads(gold_path.read_text())
        for name in NEW:
            folder=root/'edge/final/generation'/name/cid
            receipt=json.loads((folder/'result.json').read_text())
            metric_path=root/'edge/final/metrics'/name/(cid+'.json')
            metric=json.loads(metric_path.read_text())
            ledger=json.loads((folder/'compute.json').read_text())
            if not receipt['complete'] or not ledger['feasible']:raise ValueError('Incomplete request')
            for key in ('case','image_sha256','seeds'):
                if receipt['identity'][key]!=gold['identity'][key]:raise ValueError('Unpaired '+key)
            for key in ('model_identity','targets'):
                if receipt[key]!=gold[key]:raise ValueError('Mismatched model scope')
            if receipt['adapter']['weight_manifest_sha256']!=sha(parent/'offline_weights/manifest.json'):
                raise ValueError('Different W0 bank')
            if metric['provenance']['candidate_sha256']!=sha(folder/'result.json') or metric['provenance']['reference_sha256']!=sha(gold_path):
                raise ValueError('Metric receipts changed')
            for row in receipt['rows']:
                if (row['steps'],row['calls'],row['execution'],row['debug'])!=(35,70,'graph','off'):
                    raise ValueError('Changed execution protocol')
                if row['outputs_sha256']!=sha(folder/row['outputs']) or row['compute_sha256']!=sha(folder/'compute.json'):
                    raise ValueError('Changed raw output or budget')
            if ledger['total_mma']>ledger['base_mma']*ledger['budget'] or ledger['ratio']!=ledger['total_mma']/ledger['base_mma']:
                raise ValueError('Budget exceeded or misreported')
            if cid in base_by_case and base_by_case[cid]!=ledger['base_mma']:raise ValueError('Different C0')
            base_by_case[cid]=ledger['base_mma']
            costs[name].append(dict(case_id=cid,base_mma=ledger['base_mma'],total_mma=ledger['total_mma'],ratio=ledger['ratio']))
            audit.append(dict(method=name,case_id=cid,receipt_sha256=sha(folder/'result.json'),
                metric_sha256=sha(metric_path),compute_sha256=sha(folder/'compute.json')))
            log=folder/'generation.log'
            if not log.exists():log.symlink_to(root/'logs'/(name+'.log'))
    rng=np.random.default_rng(study['statistics']['bootstrap_seed'])
    indices=rng.integers(0,10,(study['statistics']['replicates'],10))
    comparisons=[]
    for candidate in NEW:
        for reference in ('fp4_two_a','fp4_two_w','fp4_three','fp4_base','fp8'):
            metrics={}
            for key,_ in METRICS:
                d=np.array([lookup[candidate,e['case']['id']][key]-lookup[reference,e['case']['id']][key] for e in study['cases']])
                regression=-d if key in ('ssim_mean','psnr_db') else d
                metrics[key]=dict(mean_difference=float(d.mean()),ci95=interval(d,indices),
                    improved_scenes=int((regression<0).sum()),worst_regression=float(regression.max()),
                    worst_case=study['cases'][int(regression.argmax())]['case']['id'])
            comparisons.append(dict(candidate=candidate,reference=reference,metrics=metrics,
                lpips_mean_improves=metrics['lpips']['mean_difference']<0,
                all_four_means_improve=all(v['mean_difference']>0 if k in ('ssim_mean','psnr_db') else v['mean_difference']<0 for k,v in metrics.items()),
                inferential=False))
    methods={**old['methods'],**new['methods']};methods={n:methods[n] for n in ORDER}
    evidence=dict(complete=True,methods=methods,comparisons=comparisons,costs=costs,audit=audit,
        source_sha256=sha(__file__),manifest_sha256=sha(root/'edge/final/manifest.json'),
        parent_report_sha256=sha(parent/'report/results.json'),side_report_sha256=sha(side/'report/results.json'),
        statistics=study['statistics'],performance_eligible=False,
        scope='Strict shared-bank/support ablation on ten exploratory scenes; no output-driven tuning. '
              'New methods use floor 25%/50% A and W supports per operator, not the separately calibrated production policies. '
              'MMA matched/capped; full two-plane packing and storage retained; no latency/memory claim.')
    out=root/'report';out.mkdir(exist_ok=True)
    write(out/'results.json',evidence)
    with (out/'per_demo.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Strict matched-budget bilateral FP4 repair','',
        'Same ten scenes, seeds, offline W0/W1 bank, FP4 quantizer, parent accumulation order, '
        'projection/FFN/QK/PV scope, 320×192, 61 frames, 35 steps, CFG 6, Graph and debug off. '
        'Only additive correction supports change. The original outputs remain immutable.','',
        'A-side means Q/P and W-side means current K/V inside attention. '
        'Both variants omit the cross term and preserve every base attention block. '
        'Weights and weight-support scores are prepared offline; activation supports read the current input.','',
        'The two fractions were fixed before generation: A/W each 25% or 50%, independently rounded down '
        'to integer tiles. W1 ranks represented block energy; A1 ranks represented residual energy times '
        'fixed W0 channel-block sensitivity. QK uses current operand norm-product proxies; PV uses '
        'V0/V1 norms. No scene-output fitting or selector MMA is used. These simple proxies do not '
        'guarantee a repair benefit and are not the separately calibrated production schedule.','',
        '## BF16 consistency','',
        '| Method | MMA / C0 | LPIPS ↓ | SSIM ↑ | PSNR ↑, dB | Relative L2 ↓ |',
        '|---|---:|---:|---:|---:|---:|']
    for n in ORDER:
        v=methods[n]['metrics']
        budget=(f"{min(r['ratio'] for r in costs[n]):.6f}–{max(r['ratio'] for r in costs[n]):.6f}" if n in NEW else
            {'fp4_base':'1','fp4_two_a':'2','fp4_two_w':'2','fp4_three':'3'}.get(n,'different dtype/path'))
        lines.append(f"| {LABELS[n]} | {budget} | {v['lpips']['mean']:.6f} | {v['ssim_mean']['mean']:.6f} | {v['psnr_db']['mean']:.4f} | {v['relative_l2']['mean']:.6f} |")
    lines+=['','## Paired results','',
        'Candidate minus reference. Exploratory scene-level percentile bootstrap (10,000 resamples); '
        'unadjusted intervals do not establish a broad or confirmatory dominance claim.','',
        '| Candidate | Reference | ΔLPIPS [95% interval] | Improved scenes | ΔSSIM | ΔPSNR | All four means improve |',
        '|---|---|---:|---:|---:|---:|---|']
    for c in comparisons:
        v=c['metrics'];lo,hi=v['lpips']['ci95']
        lines.append(f"| {LABELS[c['candidate']]} | {LABELS[c['reference']]} | {v['lpips']['mean_difference']:+.6f} [{lo:+.6f}, {hi:+.6f}] | {v['lpips']['improved_scenes']}/10 | {v['ssim_mean']['mean_difference']:+.6f} | {v['psnr_db']['mean_difference']:+.4f} | {c['all_four_means_improve']} |")
    lines+=['','## Interpretation','']
    for n in NEW:
        cc=[c for c in comparisons if c['candidate']==n and c['reference'] in ('fp4_two_a','fp4_two_w')]
        outcome='lower mean LPIPS than both single-side controls' if all(c['lpips_mean_improves'] for c in cc) else 'does not beat both single-side controls in mean LPIPS'
        lines.append(f'- {LABELS[n]} {outcome}. All-four-metric mean guard: {all(c["all_four_means_improve"] for c in cc)}.')
    lines+=['','MMA includes padded tiles and both CFG branches over all 35 steps. QK/PV corrections '
        'are each counted, selectors add no matrix products, and cross terms are disabled. '
        'AR prefill and nonlinear operators retain the common BF16 path outside C0. '
        'All new requests verify the cap before Graph capture.','',
        'Full W0/W1 buffers and all activation residual codes are deliberately retained for this '
        'mechanism ablation. Lower MMA is not a claim of lower packing, memory or latency. '
        'PSNR uses clip-global raw RGB MSE at range 1; SSIM/LPIPS are frame averages across all '
        '61 frames including the conditioning frame, matching the original report.','',
        '- [Raw paired metrics](per_demo.csv)','- [Full evidence and cost receipts](results.json)',
        '- [Frozen protocol](../edge/final/manifest.json)','- [Qualification](../qualification/results.json)',
        '- [Generation logs](../logs/)','- [Video gallery](gallery.html)','']
    (out/'report.md').write_text('\n'.join(lines))
    gallery(root,parent,side,study,out)
    return dict(complete=True,report=str(out/'report.md'),new_videos=20)


def gallery(root,parent,side,study,out):
    sources={}
    for bank in (parent,side,root):
        index=bank/'edge/judge_index.jsonl'
        if not index.exists():continue
        for line in index.read_text().splitlines():
            row=json.loads(line);sources[row['method'],row['case_id']]=row
    names=['bf16','fp4_two_a','fp4_two_w',*NEW,'fp4_three']
    if any((n,e['case']['id']) not in sources for n in names for e in study['cases']):
        raise ValueError('Export videos before building the report')
    page=['<!doctype html><meta charset="utf-8"><title>Matched bilateral repair</title>',
        '<style>body{font:16px system-ui;margin:30px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}video{width:100%}figure{margin:0}h2{margin-top:32px}</style>',
        '<h1>Strict matched-budget bilateral FP4 repair</h1><p>Ten original paired scenes. '
        'Same W0 and quantizer; actual padded MMA caps of 1.5×/2×. These are exploratory fidelity results.</p>']
    for e in study['cases']:
        cid=e['case']['id'];page.append('<h2>'+html.escape(cid)+'</h2><div class="grid">')
        for n in names:
            row=sources[n,cid];path=os.path.relpath(row['video_path'],out)
            page.append('<figure><figcaption>'+html.escape(LABELS[n])+'</figcaption><video controls muted loop preload="none" src="'+html.escape(path,quote=True)+'"></video></figure>')
        page.append('</div>')
    (out/'gallery.html').write_text('\n'.join(page))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    print(json.dumps(report(p.parse_args().root.resolve())))
