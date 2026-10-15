"""CPU-only paired memory accounting and figures from archived raw receipts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

LABELS={'bf16':'BF16','fp8':'FP8','fp4_base':'W4A4','b1.5_protected_n':'Repair 1.5x','b2_joint':'Repair 2x'}


def enrich(campaign, rows):
    """Infer resets the peak, so reconstruct the maximum across all intervals."""
    raw={}
    for row in rows:
        path=campaign/'runs'/row['id']/'result.json'
        data=json.loads(path.read_text());raw[row['id']]=data
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['source_sha256']:
            raise ValueError('Raw result no longer matches summary: '+row['id'])
        intervals=list(data['memory'].values())+data['rows']
        for kind in ('allocated','reserved'):
            row['deployment_peak_'+kind+'_gib']=max(v['peak_'+kind+'_bytes'] for v in intervals)/2**30
        row['prepare_seconds']=data['prepare_seconds']
        row['graph_and_shape_warmup_seconds']=data['timings'].get('warmup_seconds')
        row['prepared_weight_gib']=data['runner']['prepared_weight_bytes']/2**30
        row['resident_model_state_gib']=data['runner']['resident_model_state_bytes']/2**30
        times=[r['wall_seconds'] for r in data['rows']]
        row['minimum_seconds']=min(times);row['maximum_seconds']=max(times)
        row['timing_cv_percent']=100*statistics.pstdev(times)/statistics.mean(times)
        row['other_allocated_at_request_peak_gib']=row['peak_allocated_gib']-row['resident_model_state_gib']
        if row['other_allocated_at_request_peak_gib']<0:
            raise ValueError('Resident model storage exceeds request peak')
    return raw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plot',action='store_true')
    p.add_argument('--previous-campaign',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((a.campaign/'summary.json').read_text())
    source_sha=hashlib.sha256((a.campaign/'summary.json').read_bytes()).hexdigest()
    rows=summary['rows'];controls=[];scaling=[]
    raw=enrich(a.campaign,rows)
    index={(r['model'],r['mode'],r['method'],tuple(r['shape'])):r for r in rows}
    for row in rows:
        original=index.get((row['model'],row['mode'],row['method'],(320,192,61)))
        if original:
            before=raw[original['id']]['runner'];after=raw[row['id']]['runner']
            for key in ('asset_identity','prepared_weight_bytes','resident_model_state_bytes'):
                if before[key]!=after[key]:raise ValueError('Model state changed across shapes: '+row['id']+' '+key)
    for r in rows:
        if r['mode']!='compact':continue
        base=index.get((r['model'],'compact','bf16',tuple(r['shape'])))
        fp8=index.get((r['model'],'compact','fp8',tuple(r['shape'])))
        out={k:r[k] for k in ('model','method','shape','requests','seconds','fps','peak_allocated_gib','peak_reserved_gib',
              'prepare_peak_allocated_gib','graph_warmup_peak_allocated_gib','ready_process_rss_gib','ready_system_available_gib','budget_ratio',
              'deployment_peak_allocated_gib','deployment_peak_reserved_gib','prepare_seconds','graph_and_shape_warmup_seconds',
              'prepared_weight_gib','resident_model_state_gib','other_allocated_at_request_peak_gib',
              'minimum_seconds','maximum_seconds','timing_cv_percent')}
        if base:
            out['saving_vs_bf16_gib']=base['peak_allocated_gib']-r['peak_allocated_gib']
            out['saving_vs_bf16_percent']=100*out['saving_vs_bf16_gib']/base['peak_allocated_gib']
            out['deployment_saving_vs_bf16_gib']=base['deployment_peak_allocated_gib']-r['deployment_peak_allocated_gib']
            out['deployment_saving_vs_bf16_percent']=100*out['deployment_saving_vs_bf16_gib']/base['deployment_peak_allocated_gib']
        if fp8:out['saving_vs_fp8_gib']=fp8['peak_allocated_gib']-r['peak_allocated_gib']
        scaling.append(out)
    for c in summary['controls']:
        if not c['bitwise_equal']:raise ValueError('Old/new full trajectory changed: '+str(c))
        old=index[(c['model'],'legacy',c['method'],(320,192,61))]
        new=index[(c['model'],'compact',c['method'],(320,192,61))]
        oldraw=json.loads((a.campaign/'runs'/old['id']/'result.json').read_text())
        newraw=json.loads((a.campaign/'runs'/new['id']/'result.json').read_text())
        for key in ('asset_identity','prepared_weight_bytes','resident_model_state_bytes'):
            if oldraw['runner'][key]!=newraw['runner'][key]:raise ValueError('Weight state changed: '+key)
        ledgerold=oldraw['rows'][0].get('compute_ledger');ledgernew=newraw['rows'][0].get('compute_ledger')
        if ledgerold and any(ledgerold[k]!=ledgernew[k] for k in ('total_mma','base_mma','w1_support_union_bytes')):
            raise ValueError('Frozen work/support changed')
        middle=index.get((c['model'],'release',c['method'],(320,192,61)))
        extra=dict(old_rss_gib=old['ready_process_rss_gib'],new_rss_gib=new['ready_process_rss_gib'],
            old_allocated_gib=old['peak_allocated_gib'],new_allocated_gib=new['peak_allocated_gib'],
            released_tensor_gib=new['released_host_tensor_gib'],weight_storage_identical=True,mma_identical=True)
        extra.update(old_deployment_peak_gib=old['deployment_peak_allocated_gib'],
                     new_deployment_peak_gib=new['deployment_peak_allocated_gib'],
                     deployment_peak_saved_gib=old['deployment_peak_allocated_gib']-new['deployment_peak_allocated_gib'])
        if middle:
            extra.update(release_rss_gib=middle['ready_process_rss_gib'],
                         compact_incremental_peak_saving_mib=1024*(middle['peak_allocated_gib']-new['peak_allocated_gib']))
        footprints=newraw.get('packing_footprints',[])
        compact=[v for v in footprints if v['enabled']]
        extra['compact_prepared_operators']=len(compact)
        extra['largest_single_residual_capacity_saving_mib']=max(
            ((v['dense_residual_bytes']-v['residual_bytes']-v['address_map_bytes'])/2**20 for v in compact),default=0)
        controls.append({**c,**extra})
    loader=[]
    if a.previous_campaign:
        # The old campaign was deliberately stopped after a request. Its last
        # completed raw receipt is valid even if its parent summary lags behind.
        for row in rows:
            if row['mode']=='legacy':continue
            previous=a.previous_campaign/'runs'/row['id']/'result.json'
            if not previous.exists():continue
            before=json.loads(previous.read_text())
            if not before['complete']:continue
            after=raw[row['id']]
            if [r['sha256'] for r in before['rows']]!=[r['sha256'] for r in after['rows']]:
                raise ValueError('Loader full-request output differs: '+row['id'])
            for key in ('asset_identity','prepared_weight_bytes','resident_model_state_bytes'):
                if before['runner'][key]!=after['runner'][key]:raise ValueError('Loader weight state differs: '+key)
            old_peak=max(v['peak_allocated_bytes'] for v in list(before['memory'].values())+before['rows'])/2**30
            loader.append(dict(id=row['id'],model=row['model'],method=row['method'],mode=row['mode'],shape=row['shape'],
                old_prepare_peak_gib=before['memory']['after_prepare']['peak_allocated_bytes']/2**30,
                new_prepare_peak_gib=row['prepare_peak_allocated_gib'],old_deployment_peak_gib=old_peak,
                new_deployment_peak_gib=row['deployment_peak_allocated_gib'],
                deployment_peak_saved_gib=old_peak-row['deployment_peak_allocated_gib'],bitwise_equal=True,
                previous_sha256=hashlib.sha256(previous.read_bytes()).hexdigest(),current_sha256=row['source_sha256']))
    result=dict(complete=summary['complete'],source_sha256=source_sha,controls=controls,scaling=scaling,loader_controls=loader,
                failures=summary['failures'],quality_evaluation=False,performance_eligible=False,
                caveats=['Torch allocated/reserved and process/system RAM overlap; never add them.',
                         'RSS is a ready-state sample; host monitor extrema are sampled and include result-recording gaps.',
                         'Prepared-operator capacity savings cannot be summed as resident memory; Graph pools reuse buffers.',
                         'Whole-process startup peak is separate from warmed-request allocation.',
                         'Resident model state includes packed weights and quantization metadata; other allocated bytes are the remainder, not a separately timed activation peak.',
                         'All baseline auxiliary flags match; these are diagnostic memory runs on unlocked-clock Thor.'])
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    for name,data in [('paired-controls',controls),('memory-scaling',scaling),('loader-controls',loader)]:
        if not data:continue
        fields=list(dict.fromkeys(k for r in data for k in r))
        with (a.output/(name+'.csv')).open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(data)
    lines=['# Thor 内存优化：匹配控制与尺寸扩展','',
           ('全部配置点已处理。' if summary['complete'] else '**阶段结果，测试仍在进行。**'),
           '相同 RBench 场景、每点三个种子、35 步。新旧版本输出 hash、权重存储和 MMA 清单独立核对。','',
           '| 模型 | 方法 | 原就绪 RSS GiB | 新就绪 RSS GiB | 原请求 allocated GiB | 新请求 allocated GiB | 仅紧凑缓冲的峰值变化 MiB | 输出一致 |',
           '|---|---|---:|---:|---:|---:|---:|---|']
    for c in controls:
        change=c.get('compact_incremental_peak_saving_mib')
        delta='—' if change is None else f'{change:.2f}'
        lines.append(f"| {c['model']} | {LABELS[c['method']]} | {c['old_rss_gib']:.3f} | {c['new_rss_gib']:.3f} | {c['old_allocated_gib']:.3f} | {c['new_allocated_gib']:.3f} | {delta} | 是 |")
    lines+=['','“峰值变化”正值表示节省；若为负值，保留实测回归，不将局部缓冲缩小宣称为端到端峰值下降。',
            '权重和修复预算未变。CPU RSS 变化包含释放资产与清理空闲堆的共同作用，不能全部归因于权重张量字节数。','',
            '| 模型 | 尺寸（宽×高×帧） | 方法 | 请求 allocated GiB | 全过程 allocated GiB | 比 BF16 少 GiB（请求） | 比 BF16 少 %（请求） |',
            '|---|---|---|---:|---:|---:|---:|']
    for r in scaling:
        if 'saving_vs_bf16_gib' not in r:continue
        lines.append(f"| {r['model']} | {'×'.join(map(str,r['shape']))} | {LABELS[r['method']]} | {r['peak_allocated_gib']:.3f} | {r['deployment_peak_allocated_gib']:.3f} | {r['saving_vs_bf16_gib']:.3f} | {r['saving_vs_bf16_percent']:.1f} |")
    if loader:
        lines+=['','加载顺序单独消融：CPU 上替换投影，再搬迁剩余模型。下表保持 packing 模式不变。','',
                '| 模型 | 模式 | 方法 | 原加载峰值 GiB | 新加载峰值 GiB | 原全过程峰值 GiB | 新全过程峰值 GiB |',
                '|---|---|---|---:|---:|---:|---:|']
        for r in loader:
            if r['shape']!=[320,192,61]:continue
            lines.append(f"| {r['model']} | {r['mode']} | {LABELS[r['method']]} | {r['old_prepare_peak_gib']:.3f} | {r['new_prepare_peak_gib']:.3f} | {r['old_deployment_peak_gib']:.3f} | {r['new_deployment_peak_gib']:.3f} |")
        lines+=['','上述加载对照的三个完整 35 步请求输出逐位一致，常驻权重字节数一致。','']
    lines+=['','加载、建图、就绪 RSS、系统可用内存等完整分项见 [原始汇总](../report.md)。',
            '新尺寸的预算不成立时，该点不产生性能数字。没有静默降低修复、改变模型精度或复用旧轨迹。',
            '全过程 Torch 峰值取加载、建图、完整预热与各测量请求区间的最大值；不是整机物理 RAM 峰值。',
            'allocated 与 reserved 定义参考 [PyTorch 2.11 内存管理文档](https://docs.pytorch.org/docs/2.11/notes/cuda.html#memory-management)。','']
    lines+=['1.5× 保留现有每 head 的整数 attention tile 配额，长序列中的覆盖率会降低；2× 保留全量三项 attention 修复。',
            '本轮未验证大尺寸视频质量，预算档位是计算上界，不代表每个尺寸都恰好花满预算。','']
    for f in summary['failures']:lines.append(f"- {f['job']['id']}: {f['status']}；{f['error']}")
    lines+=['','请求延迟是外部内存采样下的诊断数据，每点三个种子的中位数，含编码、35 步、解码和 CPU 输出；不含结果落盘。',
            '没有锁定频率，也没有重复五轮交错测试，不能据此宣称微小加速或更新正式速度榜。','',
            '| 模型 | 宽×高×帧 | 方法 | 请求秒（中位数） | 最小—最大秒 | 生成 FPS | MMA 比例 |',
            '|---|---|---|---:|---|---:|---:|']
    for r in scaling:
        budget='—' if r['budget_ratio'] is None else f"{r['budget_ratio']:.4f}"
        lines.append(f"| {r['model']} | {'×'.join(map(str,r['shape']))} | {LABELS[r['method']]} | {r['seconds']:.3f} | {r['minimum_seconds']:.3f}—{r['maximum_seconds']:.3f} | {r['fps']:.3f} | {budget} |")
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                             'axes.grid':True,'grid.alpha':.18,'savefig.dpi':170})
        colors={'bf16':'#333333','fp8':'#b37412','fp4_base':'#9866bb','b1.5_protected_n':'#188977','b2_joint':'#2569a6'}
        styles={'bf16':'-','fp8':(0,(7,3)),'fp4_base':(0,(2,2)),'b1.5_protected_n':'-','b2_joint':'-'}
        fig,axes=plt.subplots(2,2,figsize=(11,7),sharey='row',layout='constrained')
        groups=[[(320,192,61),(320,192,125),(320,192,189)],[(320,192,61),(480,288,61),(640,384,61)]]
        for i,model in enumerate(('edge','nano')):
            for j,shapes in enumerate(groups):
                ax=axes[i,j]
                for method,label in LABELS.items():
                    ys=[index.get((model,'compact',method,shape),{}).get('peak_allocated_gib',np.nan) for shape in shapes]
                    ax.plot(range(3),ys,label=label,color=colors[method],linestyle=styles[method],marker='o',markersize=5)
                labels=[str(s[2]) for s in shapes] if j==0 else [f'{s[0]}x{s[1]}' for s in shapes]
                ax.set_xticks(range(3),labels);ax.set_ylim(bottom=0)
                ax.set_title(f"{model.title()}: {'more frames, 320x192' if j==0 else 'larger frame, 61 frames'}")
                ax.set_xlabel('Frames' if j==0 else 'Resolution');ax.set_ylabel('Request peak allocated (GiB)' if j==0 else '')
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5)
        fig.suptitle('Thor: warmed-request memory, identical auxiliary runtime flags\nMissing points are unmeasured or rejected by the frozen compute budget',fontsize=12)
        for suffix in ('png','pdf'):fig.savefig(a.output/('memory-scaling.'+suffix))
        plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
        keys=[(m,k) for m in ('edge','nano') for k in ('b1.5_protected_n','b2_joint')]
        for ax,field,title in [(axes[0],'ready_process_rss_gib','Ready process RSS'),(axes[1],'peak_allocated_gib','Request peak allocated')]:
            for j,(mode,label,color) in enumerate([('legacy','v6','#9b9b9b'),('release','CPU asset / loading fixes','#7dafcf'),('compact','+ Compact A1','#188977')]):
                ys=[index.get((model,mode,method,(320,192,61)),{}).get(field,np.nan) for model,method in keys]
                ax.bar(np.arange(len(keys))+(j-1)*.25,ys,width=.23,label=label,color=color)
            ax.set_xticks(range(len(keys)),[f"{m.title()}\n{LABELS[k]}" for m,k in keys]);ax.set_ylabel('GiB');ax.set_title(title)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3)
        fig.suptitle('320x192, 61 frames: isolate host release and compact activation storage\nRSS and Torch allocation overlap; do not add them',fontsize=12)
        for suffix in ('png','pdf'):fig.savefig(a.output/('memory-ablation.'+suffix))
        plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
        for ax,model in zip(axes,('edge','nano')):
            values=[index.get((model,'compact',method,(320,192,61))) for method in LABELS]
            state=[v['resident_model_state_gib'] if v else np.nan for v in values]
            other=[v['other_allocated_at_request_peak_gib'] if v else np.nan for v in values]
            ax.bar(range(5),state,color='#527eaa',label='Resident model state')
            ax.bar(range(5),other,bottom=state,color='#aec9a7',label='Other at request peak')
            ax.set_xticks(range(5),['BF16','FP8','W4A4','Repair\n1.5x','Repair\n2x'])
            ax.set_ylabel('Allocated GiB');ax.set_title(model.title())
        fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2)
        fig.suptitle('320x192, 61 frames: persistent model state and remaining allocation\nPacked weight metadata included; this is not a separate activation peak',fontsize=12)
        for suffix in ('png','pdf'):fig.savefig(a.output/('memory-components.'+suffix))
        plt.close(fig)
        lines+=['![内存扩展曲线](memory-scaling.png)','', '![内存消融](memory-ablation.png)','',
                '![模型状态与其余分配](memory-components.png)','']
    (a.output/'analysis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=summary['complete'],controls=len(controls),scaling=len(scaling),failures=len(summary['failures']))),flush=True)


if __name__=='__main__':main()
