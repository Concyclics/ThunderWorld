"""Frozen, randomized same-condition Thor comparison using the existing runner.

No baseline tuning or numerical changes. The BF16-attention FP4 arm is an
explicit attribution ablation, never a replacement for joint FP4 results.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from build_baselines import sha,write
from compare_speed import idle_guard
from attribution_report import percentile
from section54_protocol import EXTERNAL, model_dimensions, freeze_assets, write_csv, statistics_rows

METHODS={
    'awq_marlin':dict(method='awq_marlin',label='NVFP4-AWQ Marlin W4A16',variant=False,external=True,execution='graph',attention_backend='bf16'),
    'arcquant50':dict(method='arcquant',label='ARCQuant 50%',variant=False,external=True,execution='eager',attention_backend='bf16'),
    'arcquant100':dict(method='arcquant100',label='ARCQuant 100%',variant=False,external=True,execution='eager',attention_backend='bf16'),
    'sharq':dict(method='sharq_perf',label='SharQ',variant=False,external=True,execution='eager',attention_backend='bf16'),
    'bf16':dict(method='bf16',label='BF16',variant=False),
    'fp8':dict(method='fp8',label='FP8 W8A8 + BF16 attention',variant=False),
    'fp4':dict(method='fp4_base',label='W4A4 + FP4 attention',variant=True),
    'repair15':dict(method='b1.5_protected_n',label='修复 1.5× + FP4 attention',variant=True),
    'repair20':dict(method='b2_joint',label='修复 2× + FP4 attention',variant=True),
    'fp4_bf16_attention':dict(method='fp4_base',label='W4A4 + BF16 attention（归因）',variant=True,attention_backend='bf16'),
}


def snapshot():
    result={}
    for name,cmd in [('power',['nvpmodel','-q']),('gpu',['nvidia-smi']),
        ('clocks',['nvidia-smi','--query-gpu=clocks.current.graphics,clocks.current.memory,temperature.gpu,power.draw','--format=csv'])]:
        r=subprocess.run(cmd,text=True,capture_output=True)
        result[name]=dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    return result


def freeze_model(a):
    """Resolve the transferred recipe without changing weights or calibration."""
    package=a.root/'data'/a.model
    spec=json.loads((package/'package.json').read_text())
    study=json.loads((package/'study.json').read_text())
    if not spec['complete'] or study['steps']!=35:raise ValueError('Unqualified transfer package')
    inventory=json.loads((a.root/'data/model_files.json').read_text())
    matches=[(name,meta) for name,meta in inventory.items()
        if meta['repo']==spec['model']['repo'] and meta['revision']==spec['model']['revision']]
    if len(matches)!=1:raise ValueError('Exact model revision not found')
    folder,files=matches[0];model=a.root/'models'/folder
    verified=json.loads((model/'thunderworld_verified.json').read_text())
    if not verified['complete'] or verified['files']!=files['files']:raise ValueError('Model receipt mismatch')
    for name,info in files['files'].items():
        if (model/name).stat().st_size!=info['bytes']:raise ValueError('Model file missing/truncated: '+name)
    restored=a.root/'assets'/a.model
    receipt=json.loads((restored/'restored.json').read_text())
    if not receipt['complete'] or not receipt['all_tensor_hashes_match'] or receipt['package_sha256']!=sha(package/'package.json'):
        raise ValueError('Exact repair asset reconstruction incomplete')
    methods={}
    for name in {METHODS[key]['method'] for key in a.methods}:
        item=copy.deepcopy(study['methods']['bf16' if name in EXTERNAL else name]);cfg=item['config']
        if name in EXTERNAL:
            item['name']=name
            cfg.update(precision='bf16',attention_backend='bf16',execution='graph' if name=='awq_marlin' else 'eager')
        if cfg['debug']!='off':raise ValueError('Expected debug off')
        cfg['resource_mode']='exclusive';cfg['strict_tuning']=False
        if item.get('assets'):
            item['assets']=str(restored/name/'assets')
            item['assets_sha256']=sha(Path(item['assets'])/'manifest.json')
            cfg['budget_plan']=str(package/name/'plan.json');item['plan_sha256']=sha(cfg['budget_plan'])
        methods[name]=item
    cases=copy.deepcopy(study['cases'])
    for row in cases:
        row['case']['image']=str(package/row['case']['image'])
        if sha(row['case']['image'])!=row['image_sha256']:raise ValueError('Frozen input changed')
        if any(row['case'][key]!=value for key,value in [('width',320),('height',192),('frames',61)]):
            raise ValueError('Generation shape changed')
    requirements=json.loads(Path(__file__).with_name('section54_fp4_retest_spec.json').read_text())
    if spec['model']['revision']!=requirements['models'][a.model]['revision'] or sha(package/'study.json')!=requirements['models'][a.model]['study_sha256']:
        raise ValueError('Model revision or study changed')
    dimensions=model_dimensions(model)
    if len(dimensions)!=requirements['models'][a.model]['generation_projections']:raise ValueError('Wrong target count')
    external_assets,asset_receipts=freeze_assets(a.root,a.model,model,methods,dimensions,a.output)
    for row,expected in zip(cases,requirements['cases']):
        if row['case']['id']!=expected['scene_id'] or row['image_sha256']!=expected['image_sha256']:raise ValueError('Wrong study scenes')
        row['prompt_sha256']=__import__('hashlib').sha256(row['case']['prompt'].encode()).hexdigest()
    frozen=a.output/'fixed-model.json'
    write(frozen,dict(schema='thor-frozen-model-v1',model_key=a.model,model=str(model),package=str(package),
        package_sha256=sha(package/'package.json'),study_sha256=sha(package/'study.json'),
        model_verified_sha256=sha(model/'thunderworld_verified.json'),asset_receipt_sha256=sha(restored/'restored.json'),
        methods=methods,cases=cases,steps=35,quality_evaluation=False,targets=sorted(dimensions),dimensions=dimensions,
        external_assets=external_assets,asset_receipts=asset_receipts,arc_build=str(a.arc_build),
        arc_build_sha256=sha(a.arc_build/'build.json'),
        generation=dict(width=320,height=192,frames=61,steps=35,cfg=6,flow_shift=3,fps=24),
        timing_scope='Full request: host input setup + encode + 35 denoise steps + decode + CPU output; serialization excluded'))
    return frozen


def prepare(a):
    a.output.mkdir(parents=True,exist_ok=False)
    fixed=freeze_model(a)
    base=json.loads(fixed.read_text())
    version=json.loads((a.variant/'variant.json').read_text())
    if version['backend']!='cutlass-mainloop-fused':raise ValueError('Expected qualified fused backend')
    requirements=json.loads(Path(__file__).with_name('section54_fp4_retest_spec.json').read_text())
    if sha(a.variant/'variant.json')!=requirements['variant_receipt_sha256']:raise ValueError('Expected frozen v6, not v8')
    rng=random.Random(230923);rounds=[]
    for i in range(a.rounds):
        order=list(a.methods);rng.shuffle(order);rounds.append(dict(index=i,order=order))
    write(a.output/'manifest.json',dict(schema='thor-standard-comparison-v1',created=time.time(),
        fixed_manifest=str(fixed),fixed_manifest_sha256=sha(fixed),variant=str(a.variant),
        variant_receipt_sha256=sha(a.variant/'variant.json'),methods={name:METHODS[name] for name in a.methods},rounds=rounds,
        model_key=a.model,
        seeds=[301017,301018,301031],cases=[0,1],generation=base['generation'],
        concurrency=1,batch=1,execution='per_method',debug='off',quality_evaluation=False,
        worker_sha256=sha(Path(__file__).with_name('tune_speed.py')),
        source_hashes={n:sha(Path(__file__).with_name(n)) for n in ('standard_comparison.py','tune_speed.py','baseline_adapter.py','section54_protocol.py','probe_baselines.py','section54_resources.py','section54_profiles.py','compare_speed.py','build_baselines.py','attribution_report.py')},
        clock_policy='MAXN, DVFS; no root permission for jetson_clocks, no clock override',
        timing_scope=base['timing_scope'],environment=snapshot(),
        limitations=['Controlled in-project comparison; not MLPerf or official peak throughput.',
            'All selected arms use the existing production implementations; no FP8/BF16 retuning.',
            f'{a.rounds} randomized rounds; each arm has a fresh process, prepare and full request warmup.',
            'Two representative scenes x three seeds per round; not a dataset-wide quality evaluation.',
            'Frozen attention paths differ; external baselines use BF16 attention, joint FP4 methods use FP4 attention.',
            'GPU compute exclusivity checked at request boundaries; desktop graphics remain active.']))


def load_manifest(a):
    p=a.output/'manifest.json';m=json.loads(p.read_text())
    if sha(m['fixed_manifest'])!=m['fixed_manifest_sha256']:raise ValueError('Frozen source manifest changed')
    if sha(Path(m['variant'])/'variant.json')!=m['variant_receipt_sha256']:raise ValueError('Variant changed')
    if sha(Path(__file__).with_name('tune_speed.py'))!=m['worker_sha256']:raise ValueError('Measurement worker changed')
    for name,digest in m.get('source_hashes',{}).items():
        if sha(Path(__file__).with_name(name))!=digest:raise ValueError('Frozen source changed: '+name)
    return m


def command(a,m,name,out,operation='measure',case_index=0,seed=None):
    spec=m['methods'][name]
    cmd=[sys.executable,'-u',str(Path(__file__).with_name('tune_speed.py')),operation,
        '--root',str(a.root),'--manifest',m['fixed_manifest'],'--method',spec['method'],'--output',str(out),
        '--cases',*[str(i) for i in (m['cases'] if operation=='measure' else [case_index])],
        '--repeats',str(len(m['seeds']) if operation=='measure' else 1),
        '--seeds',*[str(i) for i in (m['seeds'] if operation=='measure' else [seed if seed is not None else m['seeds'][0]])]]
    if spec['variant']:cmd+=['--variant',m['variant']]
    if spec.get('attention_backend') and not spec.get('external'):cmd+=['--attention-backend',spec['attention_backend']]
    return cmd


def checked_run(path,m,name):
    d=json.loads(path.read_text())
    if not d['complete'] or d['operation']!='measure' or len(d['rows'])!=len(m['cases'])*len(m['seeds']):
        raise ValueError(f'Incomplete measurement: {path}')
    if d['fixed_manifest_sha256']!=m['fixed_manifest_sha256']:raise ValueError('Unmatched manifest')
    spec=m['methods'][name]
    if d['method']!=spec['method']:raise ValueError('Method mismatch')
    if spec['variant']:
        if d['variant']['receipt_sha256']!=m['variant_receipt_sha256']:raise ValueError('Wrong measured variant')
    elif d['variant'] is not None:raise ValueError('Unexpected baseline variant')
    cfg=d['effective_config']
    expected_precision='bf16' if name=='bf16' or spec.get('external') else 'fp8' if name=='fp8' else 'fp4'
    if cfg['precision']!=expected_precision:raise ValueError('Precision changed')
    expected_attention=spec.get('attention_backend','bf16' if name in ('bf16','fp8','fp4_bf16_attention') else 'fp4')
    if spec.get('external'):
        adapter=d.get('adapter')
        expected_dtype='W4A16_BF16_Marlin' if name=='awq_marlin' else 'FP4_tensor_core'
        if not adapter or not adapter['no_fallback'] or adapter['gemm_dtype']!=expected_dtype or adapter['execution']!=spec['execution']:raise ValueError('Actual adapter path mismatch')
        fixed=json.loads(Path(m['fixed_manifest']).read_text())
        if set(adapter['targets'])!=set(fixed['targets']) or adapter['dimensions']!=fixed['dimensions']:raise ValueError('Adapter scope mismatch')
    if cfg['attention_backend']!=expected_attention:raise ValueError('Attention scope changed')
    if sorted((r['case_index'],r['seed']) for r in d['rows'])!=sorted((i,s) for i in m['cases'] for s in m['seeds']):
        raise ValueError('Unpaired cases or seeds')
    for r in d['rows']:
        if r['debug']!='off' or r['execution']!=spec.get('execution','graph') or r.get('profiled') or r['steps']!=35 or r['calls']!=70:
            raise ValueError('Instrumented or changed request')
        if r.get('compute_ledger') and not r['compute_ledger']['feasible']:raise ValueError('Budget violation')
    return d


def run(a):
    m=load_manifest(a);status=dict(complete=False,completed_runs=0,total_runs=len(m['rounds'])*len(m['methods']),started=time.time())
    write(a.output/'status.json',status)
    for rnd in m['rounds']:
        for name in rnd['order']:
            failure=a.output/'qualification'/(name+'.unavailable.json')
            if failure.exists():
                status.setdefault('unavailable',{})[name]=str(failure);continue
            if m['methods'][name].get('external'):
                qualification=a.output/'qualification'/(name+'.qualified.json')
                q=json.loads(qualification.read_text())
                if not q.get('complete') or q['manifest_sha256']!=m['fixed_manifest_sha256'] or sha(q['probe'])!=q['probe_sha256']:raise ValueError('External qualification missing or changed: '+name)
            out=a.output/'runs'/f"round-{rnd['index']:02d}"/name
            if (out/'result.json').exists():
                checked_run(out/'result.json',m,name);status['completed_runs']+=1;continue
            if out.exists():raise RuntimeError('Incomplete run retained: '+str(out))
            idle_guard();status.update(method=name,round=rnd['index'],updated=time.time())
            write(a.output/'status.json',status)
            logs=a.output/'logs';logs.mkdir(exist_ok=True)
            log=logs/f"round-{rnd['index']:02d}-{name}.log"
            environment=dict(before=snapshot());started=time.time()
            from section54_resources import monitored_run
            with log.open('w') as stream:r=monitored_run(command(a,m,name,out),stream,log)
            environment.update(after=snapshot(),started=started,finished=time.time(),exit_code=r.returncode)
            write(log.with_suffix('.environment.json'),environment)
            if r.returncode:
                status.update(error='Worker failed; inspect '+str(log),exit_code=r.returncode)
                write(a.output/'status.json',status);raise RuntimeError(status['error'])
            checked_run(out/'result.json',m,name);status['completed_runs']+=1
            write(a.output/'status.json',status);report(a)
    status.update(complete=True,finished=time.time());write(a.output/'status.json',status);report(a)


def report(a):
    m=json.loads((a.output/'manifest.json').read_text());rows=[];expected_keys=None;identity=None
    sources={};numerical_scope={};projection_scope=None
    for name,spec in m['methods'].items():
        records=[]
        for rnd in m['rounds']:
            path=a.output/'runs'/f"round-{rnd['index']:02d}"/name/'result.json'
            if not path.exists() or not json.loads(path.read_text())['complete']:continue
            d=checked_run(path,m,name);records.append(d);sources[str(path)]=sha(path)
            model=d['runner']['model_identity']
            if identity is None:identity=model
            if identity!=model:raise ValueError('Model identity differs')
            keys=sorted((r['case'],r['seed'],r['steps']) for r in d['rows'])
            if expected_keys is None:expected_keys=keys
            if keys!=expected_keys:raise ValueError('Input keys differ')
            scope={k:d['runner'][k] for k in ('asset_identity','targets')}
            for k in ('ar_cache','first_frame','codec_graph'):
                scope[k]={field:value for field,value in d['runner'][k].items()
                    if field in ('enabled','qkv_streams','max_graphs','scope')}
            signature=dict(scope=scope,config=d['effective_config'],runtime_options=d['runtime_options'],adapter=d.get('adapter'),
                source_sha256=d['runner']['source_sha256'])
            if name in numerical_scope and numerical_scope[name]!=signature:raise ValueError('Method changed across rounds')
            numerical_scope[name]=signature
            if name!='bf16':
                if projection_scope is None:projection_scope=scope['targets']
                if projection_scope!=scope['targets']:raise ValueError('Different quantized projection scope')
        if not records:continue
        data=[r for d in records for r in d['rows']];times=[r['wall_seconds'] for r in data]
        row=dict(name=name,label=spec['label'],rounds=len(records),requests=len(data),
            median_seconds=statistics.median(times),p90_seconds=percentile(times,.9),fps=60/statistics.median(times),new_frame_fps_at_median=60/statistics.median(times),
            allocated_gib=max(r['peak_allocated_bytes'] for r in data)/2**30,
            reserved_gib=max(r['peak_reserved_bytes'] for r in data)/2**30,
            per_round=[dict(median_seconds=statistics.median(r['wall_seconds'] for r in d['rows']),
                prepare_seconds=d['prepare_seconds'],timings=d['timings']) for d in records])
        inventories=[d['postwarm_inventory'] for d in records]
        resident={v['resident_model_state_bytes'] for v in inventories}
        if len(resident)!=1:raise ValueError('Resident storage changed across rounds: '+name)
        row['resident_model_state_bytes']=next(iter(resident))
        row['resident_model_state_gib']=row['resident_model_state_bytes']/2**30
        row['scratch_bytes_reported']=max(v.get('support_buffer_bytes',0) for v in inventories)
        row['actual_execution']=spec.get('execution','graph')
        row['attention_backend']=d['effective_config']['attention_backend']
        samples=[sample for d in records for sample in d.get('memory',{}).values()]
        samples.extend(r['memory_after'] for r in data if 'memory_after' in r)
        if samples:
            row['unified_memory']=dict(process_rss_max_gib=max(s['process_rss_bytes'] for s in samples)/2**30,
                process_hwm_gib=max(s['VmHWM_bytes'] for s in samples)/2**30,
                system_available_min_gib=min(s['system_available_bytes'] for s in samples)/2**30,
                scope='Process/host boundary samples, overlapping with Torch allocated/reserved; do not add.')
        ledgers=[r['compute_ledger'] for r in data if r.get('compute_ledger')]
        if ledgers:
            row['compute_ratios']=sorted(set(l['ratio'] for l in ledgers))
            row['cost_by_case']={}
            for r in data:
                if str(r['case_index']) in row['cost_by_case']:continue
                ledger=r.get('compute_ledger')
                if not ledger:continue
                by={}
                for v in ledger['rows']:
                    sums=by.setdefault(v['kind'],dict(base_flops=0,issued_flops=0,padded_base_flops=0))
                    for dest,key in [('base_flops','base_mma'),('issued_flops','total_mma'),('padded_base_flops','padded_base_mma')]:
                        sums[dest]+=v.get(key,0)*v['calls']
                for v in by.values():v['ratio']=v['issued_flops']/v['base_flops']
                row['cost_by_case'][str(r['case_index'])]=dict(components=by,
                    w1_union_bytes=ledger['w1_support_union_bytes'],packing_traffic=ledger['packing_traffic'],
                    non_mma_work=ledger['non_mma_work'],scope='Issued-operation cost ledger, not measured hardware counters')
        rows.append(row)
    runtime_flags={name:dict(options=scope['runtime_options'],codec_graph=scope['scope']['codec_graph'])
        for name,scope in numerical_scope.items()}
    common_runtime=bool(runtime_flags) and all(v==next(iter(runtime_flags.values())) for v in runtime_flags.values())
    runtime_caveat=('Auxiliary runtime options and codec Graph settings match across measured methods.' if common_runtime else
        'Input settings match, but execution modes and auxiliary norm/rotary/calibrated pointwise/codec Graph flags differ. '
        'Baseline implementations retain original flags. This is an implementation comparison, not an arithmetic-only repair ablation.')
    round_count=len(m['rounds']);model_label=m.get('model_key','edge').capitalize()
    result=dict(complete=len(rows)==len(m['methods']) and all(r['rounds']==round_count for r in rows),rows=rows,
        manifest_sha256=sha(a.output/'manifest.json'),sources=sources,quality_evaluation=False,
        model_key=m.get('model_key','edge'),round_count=round_count,
        common_auxiliary_runtime=common_runtime,auxiliary_runtime_by_method=runtime_flags,
        performance_eligible=False,measurement_class=f'{round_count} randomized rounds, exclusive-checked DVFS',
        numerical_scope=numerical_scope,limitations=m['limitations']+[runtime_caveat])
    index={r['name']:r for r in rows}
    for r in rows:
        for ref in ('fp4','arcquant50','arcquant100','sharq'):
            if ref in index:r['speedup_vs_'+ref]=index[ref]['median_seconds']/r['median_seconds']
    request_rows=[]
    missing={}
    for name,spec in m['methods'].items():
        unavailable=a.output/'qualification'/(name+'.unavailable.json')
        if unavailable.exists():missing[name]=json.loads(unavailable.read_text())
        for rnd in m['rounds']:
            path=a.output/'runs'/f"round-{rnd['index']:02d}"/name/'result.json'
            if not path.exists() or not json.loads(path.read_text()).get('complete'):continue
            d=checked_run(path,m,name)
            for r in d['rows']:
                keep={k:r.get(k) for k in ('case','case_index','seed','steps','frames','new_frames','wall_seconds','new_frame_fps','debug','execution','profiled','peak_allocated_bytes','peak_reserved_bytes','output_shapes','sha256')}
                request_rows.append(dict(cohort=str(a.output),device='Thor',model=m['model_key'],method_id=spec['method'],
                    method=name,round=rnd['index'],source=str(path),source_sha256=sha(path),
                    manifest_sha256=sha(a.output/'manifest.json'),performance_eligible=False,**keep))
    summaries=[]
    for name,spec in m['methods'].items():
        data=[r for r in request_rows if r['method']==name]
        if data:summaries.extend(statistics_rows(data,spec['method'],m['model_key'],sha(a.output/'manifest.json')))
    comparisons={(r['case'],r['method_id']):r for r in summaries}
    for row in summaries:
        for field,reference in [('relative_to_plain_fp4','fp4_base'),('relative_to_arc50','arcquant'),('relative_to_arc100','arcquant100'),('relative_to_sharq','sharq_perf')]:
            ref=comparisons.get((row['case'],reference))
            row[field]=None if ref is None else ref['median_seconds']/row['median_seconds']
        config_row=next(r for r in rows if m['methods'][r['name']]['method']==row['method_id'])
        row['resident_model_state_bytes']=config_row['resident_model_state_bytes']
    round_rows=[]
    for name,spec in m['methods'].items():
        for rnd in m['rounds']:
            data=[r for r in request_rows if r['method']==name and r['round']==rnd['index']]
            if data:round_rows.extend(dict(round=rnd['index'],**r) for r in statistics_rows(data,spec['method'],m['model_key'],sha(a.output/'manifest.json')))
    write_csv(a.output/'round_summary.csv',round_rows)
    result['unavailable']=missing
    result['complete']=all(name in missing or any(r['name']==name and r['rounds']==round_count for r in rows) for name in m['methods'])
    result['successful_requests']=len(request_rows)
    write_csv(a.output/'requests.csv',request_rows)
    write_csv(a.output/'summary.csv',summaries)
    write(a.output/'summary.json',result)
    lines=[f'# Thor {model_label} 同设置 {round_count} 轮速度复测','',
        f'{model_label}，320×192、61 帧、35 步、CFG 6、flow shift 3；batch=1、单请求、按冻结方法采用 Graph/eager、debug off。',
        f'每种配置独立进程加载与预热，{round_count} 轮固定随机顺序，每轮两个输入各三个种子。包含编码、生成、解码与 CPU 输出；不含加载、建图和视频文件保存。',
        'MAXN、DVFS 未锁频。标准化项目内对照，不等同于 MLPerf 或 NVIDIA 峰值测试。',
        '原生联合 FP4 使用冻结 CUTLASS v6；外部适配器保持原算法与执行模式，未另行调优。',
        ('各方法保留相同的外围融合与 codec Graph 开关，具体值见 summary.json。' if common_runtime else
         '各方法的外围融合与 codec Graph 开关存在差异，具体值见 summary.json；不将全部收益归于精度格式。'),'',
        '| 方法 | 轮/请求 | 中位秒 | P90 秒 | 生成 FPS | 相对普通 FP4 加速 | 相对 SharQ 加速 | allocated GiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        plain='—' if 'speedup_vs_fp4' not in r else f"{r['speedup_vs_fp4']:.3f}×"
        sharq='—' if 'speedup_vs_sharq' not in r else f"{r['speedup_vs_sharq']:.3f}×"
        lines.append(f"| {r['label']} | {r['rounds']}/{r['requests']} | {r['median_seconds']:.3f} | {r['p90_seconds']:.3f} | {r['fps']:.3f} | {plain} | {sharq} | {r['allocated_gib']:.3f} |")
    if any('unified_memory' in r for r in rows):
        lines+=['','Thor 使用统一内存；下列 Torch、进程与系统指标互有重叠，不能相加。系统可用内存仅在请求边界采样。','',
            '| 方法 | Torch reserved 峰值 GiB | 进程 RSS 最大采样 GiB | 进程 VmHWM GiB | 系统可用最小采样 GiB |',
            '|---|---:|---:|---:|---:|']
        for r in rows:
            if 'unified_memory' not in r:continue
            v=r['unified_memory']
            lines.append(f"| {r['label']} | {r['reserved_gib']:.3f} | {v['process_rss_max_gib']:.3f} | {v['process_hwm_gib']:.3f} | {v['system_available_min_gib']:.3f} |")
    if 'fp4_bf16_attention' in m['methods']:
        lines+=['','W4A4 + BF16 attention 仅用于隔离 attention 的开销；不能作为联合 FP4 方法的速度或质量结果。']
    if missing:lines+=['','不可用方法（不按零时延处理）：'+json.dumps(missing,ensure_ascii=False)]
    lines+=['','1.5×／2× 是相对普通 W4A4 的整请求 MMA 预算，不表示请求延迟倍数。',
        'FPS 为 60 / 完整请求秒数，不包含条件帧，与视频播放帧率不同。质量未重测。','']
    (a.output/'report.md').write_text('\n'.join(lines))


def profile(a):
    m=load_manifest(a)
    if not json.loads((a.output/'status.json').read_text())['complete']:
        raise ValueError('Finish the uninstrumented campaign before profiling')
    if set(m['methods'])=={'fp4','awq_marlin','arcquant50','arcquant100','sharq','repair15','repair20'}:
        for model in ('edge','nano'):
            status=a.output.parent/model/'status.json'
            if not status.exists() or not json.loads(status.read_text())['complete']:raise ValueError('Finish both model timings before profiling')
    dest=a.output/'profiles';dest.mkdir(exist_ok=True)
    tmp=dest/'tmp';tmp.mkdir(exist_ok=True)
    environment=dict(os.environ,TMPDIR=str(tmp))
    for name in (a.profile_methods or list(m['methods'])):
        if name not in m['methods']:raise ValueError('Unfrozen profile method')
        failure=a.output/'qualification'/(name+'.unavailable.json')
        if failure.exists():continue
        for case in a.profile_cases:
            if case not in m['cases']:raise ValueError('Unfrozen profile case')
            idle_guard()
            tag=f'{name}-case{case}-seed{a.profile_seed}'
            receipt=dest/(tag+'.complete.json')
            if receipt.exists():
                if not json.loads(receipt.read_text())['complete']:raise ValueError('Incomplete profile retained')
                continue
            output=dest/(tag+'.nsys-rep')
            cmd=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
                '--capture-range=cudaProfilerApi','--capture-range-end=stop','--cuda-graph-trace=node',
                '--output',str(dest/tag)]+command(a,m,name,dest/(tag+'-request'),'profile',case,a.profile_seed)
            with (dest/(tag+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=environment)
            if r.returncode:raise RuntimeError('Nsight failed: '+tag+'\n'+(dest/(tag+'.log')).read_text(errors='replace')[-3500:])
            cmd=['nsys','stats','--report','cuda_gpu_kern_sum','--format','csv',
                '--output',str(dest/(tag+'-kernels')),str(output)]
            with (dest/(tag+'-stats.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=environment)
            if r.returncode:raise RuntimeError('Nsight export failed: '+tag)
            csv=dest/(tag+'-kernels_cuda_gpu_kern_sum.csv')
            request=dest/(tag+'-request/result.json');d=json.loads(request.read_text())
            if not d['complete'] or len(d['rows'])!=1 or not d['rows'][0]['profiled']:raise ValueError('Incomplete profile')
            write(receipt,dict(complete=True,method=name,method_id=m['methods'][name]['method'],case_index=case,
                seed=a.profile_seed,profile_path=str(output),profile_sha256=sha(output),csv_sha256=sha(csv),
                source_manifest_sha256=sha(a.output/'manifest.json'),request_sha256=sha(request),
                sqlite=[dict(path=str(v),sha256=sha(v)) for v in dest.glob(tag+'*.sqlite')]))
    from section54_profiles import summarize
    summarize(a.output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('prepare','run','report','profile'))
    p.add_argument('--root',type=Path,default=Path('/home/chenhan/thunderworld'))
    p.add_argument('--variant',type=Path,default=Path('/home/chenhan/thunderworld/variants/cutlass-mainloop-v6'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arc-build',type=Path,default=Path('/home/chenhan/thunderworld/external/arcquant-sm110-section54'))
    p.add_argument('--profile-methods',nargs='+',choices=tuple(METHODS))
    p.add_argument('--profile-cases',nargs='+',type=int,default=[0,1])
    p.add_argument('--profile-seed',type=int,default=301017)
    p.add_argument('--model',choices=('edge','nano'),default='edge',help='Prepare only; run/report read the frozen manifest')
    p.add_argument('--rounds',type=int,choices=range(1,6),default=5,help='Prepare only; one round is a pilot, not a five-round ranking')
    p.add_argument('--methods',nargs='+',choices=tuple(METHODS),default=list(METHODS),help='Prepare only')
    a=p.parse_args();globals()[a.operation](a)


if __name__=='__main__':main()
