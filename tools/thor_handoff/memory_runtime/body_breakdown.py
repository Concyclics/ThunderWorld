"""Reconcile the matched, VAE-excluded body receipts without running inference.

Rebuild every resident byte from executed matrix shapes, actual W1 support
unions, checkpoint headers and the hash-matched storage implementation. Keep
unattributed runtime memory explicitly grouped rather than calling it activation.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import struct

GIB, MIB = 2**30, 2**20
METHODS = ('bf16', 'fp8', 'fp4_base', 'b1.5_protected_n', 'b2_joint')
LABELS = dict(bf16='BF16', fp8='FP8', fp4_base='W4A4',
              b1_5='修复 1.5×', b2_joint='修复 2×')
LABELS['b1.5_protected_n'] = LABELS.pop('b1_5')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def read_receipt(root, device, model, method, sources):
    raw = root / ('thor/raw' if device == 'Thor' else 'raw')
    name = f'body-runs/{model}-{method}-trim/result.json.gz'
    index = {r['path']: r for r in json.loads((raw/'index.json').read_text())}
    blob = (raw/name).read_bytes(); decoded = gzip.decompress(blob)
    if digest(blob) != index[name]['compressed_sha256'] or digest(decoded) != index[name]['uncompressed_sha256']:
        raise ValueError('Changed compressed evidence: '+name)
    r = json.loads(decoded)
    if (r['model'], r['method']) != (model, method):
        raise ValueError('Receipt identity differs from its path')
    if not (r['complete'] and r['vae_fully_released'] and r['latent_isolation_bitwise_equal']):
        raise ValueError('Unqualified body measurement')
    if (r['profile_name'] != 'trim' or r['profile'].get('ar','bf16') != 'bf16'
            or any(r['profile'].get(k,0) for k in ('vae_planes','embedding_planes','auxiliary_planes'))):
        raise ValueError('Different quantization coverage')
    if len(r['rows']) != 6 or any(x['calls'] != 70 or x['ar_prefills'] != 2 for x in r['rows']):
        raise ValueError('Incomplete request cohort')
    sources[str(raw/name)] = dict(sha256=digest(blob), uncompressed_sha256=digest(decoded))
    r['_audit_source'] = str(raw/name)
    return r


def dimensions(receipt):
    dims = {}; unions = {}; supports = {}; ops = {}
    for r in receipt['runner']['observed_shapes']:
        name, n, k = r['module'], r['n'], r['k']
        if n % 128 or k % 128:
            raise ValueError('Weight padding must be explicitly extended for this shape')
        if name in dims and dims[name] != (n, k):
            raise ValueError('Changing weight shape')
        dims[name] = (n, k)
        ids = tuple(r['w_blocks'])
        if len(set(ids)) != len(ids) or any(i < 0 or i >= n*k//128**2 for i in ids):
            raise ValueError('Invalid W1 support')
        unions.setdefault(name, set()).update(ids)
        supports.setdefault(name, set()).add(ids)
        key = (name, r['m'], r['repair'])
        if key in ops and ops[key] != ids:
            raise ValueError('Support changes without a prepared-key change')
        ops[key] = ids
    if set(dims) != set(receipt['runner']['targets']):
        raise ValueError('Not all generation targets executed')
    return dims, unions, supports, ops


def check_source(receipt, backend, names, sources):
    for name in names:
        path = backend/name
        data = path.read_bytes()
        if digest(data) != receipt['runner']['source_sha256'][name]:
            raise ValueError('Backend does not match the measurement: '+str(path))
        sources[str(path)] = dict(sha256=digest(data))


def checkpoint_headers(record, sources, model_root=None):
    """Verify bytes/dtypes/shapes using small safetensors headers only."""
    root = Path(record['model_directory'])
    if model_root is not None: root = model_root/root.name
    tensors = {}
    for rel, expected in record['safetensors_header_sha256'].items():
        path = root/rel
        with path.open('rb') as f:
            length = struct.unpack('<Q', f.read(8))[0]
            data = f.read(length)
        if digest(data) != expected:
            raise ValueError('Changed checkpoint header: '+str(path))
        sources[str(path)] = dict(header_sha256=digest(data), header_only=True)
        if rel.split('/')[0] == 'transformer':
            for name, value in json.loads(data).items():
                if name == '__metadata__': continue
                if name in tensors: raise ValueError('Duplicate checkpoint tensor')
                tensors[name] = value
    return tensors


def build(args):
    sources = {}; resident = []; runtime = []; scopes = []; modules = []
    checkpoints = json.loads(args.checkpoints.read_text())
    sources[str(args.checkpoints)] = dict(sha256=digest(args.checkpoints.read_bytes()))
    for model in ('edge', 'nano'):
        ck = next(v for k, v in checkpoints.items() if model in k.lower())
        headers = checkpoint_headers(ck, sources, args.model_root)
        p = {k:v['bytes'] for k,v in ck['transformer_partition'].items()}
        target_bf16 = p['generation_ffn'] + p['generation_attention_projections']
        common = dict(ar_ffn=p['understanding_ffn'], ar_qkvo=p['understanding_attention_projections'],
                      embedding=p['text_embedding'], video_io=p['video_input_output_projections'],
                      timestep_fp32=2*p['timestep_embedding'], norms=p['norms'], rotary_buffer=256)
        common_bytes = sum(common.values())
        ar_k = headers['layers.0.self_attn.to_k.weight']['shape'][0]
        ar_v = headers['layers.0.self_attn.to_v.weight']['shape'][0]
        hidden = headers['layers.0.self_attn.to_q.weight']['shape'][0]
        for suffix, size in (('to_q',hidden),('to_k',ar_k),('to_v',ar_v)):
            if {v['shape'][0] for name,v in headers.items() if name.endswith('.self_attn.'+suffix+'.weight')} != {size}:
                raise ValueError('AR cache split requires uniform head dimensions')
        removed = p['language_output_head'] + p.get('action_heads',0) + p.get('audio_heads',0)
        for device, backend in (('PRO6000',args.pro_backend), ('Thor',args.thor_backend)):
            records = {method:read_receipt(args.evidence,device,model,method,sources) for method in METHODS}
            base = records['bf16']; baseline = base['body_resident_model_bytes']
            baseline_peak = max(x['body_peak_allocated_bytes'] for x in base['rows'])
            if baseline != common_bytes+target_bf16:
                raise ValueError('BF16 checkpoint partition does not reconcile with measured body')
            cohort = {(x['case'],x['seed']) for x in base['rows']}
            dims, _, _, _ = dimensions(records['fp4_base'])
            if 2*sum(n*k for n,k in dims.values()) != target_bf16:
                raise ValueError('Executed shape total disagrees with checkpoint partition')
            for name,(n,k) in dims.items():
                h = headers[name+'.weight']
                if h['dtype'] != 'BF16' or h['shape'] != [n,k] or h['data_offsets'][1]-h['data_offsets'][0] != 2*n*k:
                    raise ValueError('Per-matrix checkpoint identity mismatch')
                if device == 'Thor':
                    modules.append(dict(model=model,module=name,category='generation_ffn' if '.mlp_moe_gen.' in name else 'generation_qkvo',
                        n=n,k=k,bf16_weight_bytes=2*n*k,fp8_code_bytes=n*k,fp4_w0_code_bytes=n*k//2,
                        fp4_one_scale_layout_bytes=n*k//16,fp4_code_ratio=.25))
            share = target_bf16/baseline
            scopes.append(dict(device=device,model=model,target_bf16_bytes=target_bf16,body_bf16_bytes=baseline,
                target_fraction=share,unchanged_body_bytes=common_bytes,unchanged_components=common,
                fp8_ideal_body_ratio=1-share/2,fp4_codes_only_ideal_body_ratio=1-.75*share,
                fp4_codes_only_body_compression_limit=1/(1-.75*share),
                excluded_vae_bytes=ck['components']['vae']['bytes'],removed_i2v_bytes=removed,
                note='Ideal bounds at fixed coverage omit all scales and repair; not new measurements.'))
            for method,r in records.items():
                runner = r['runner']
                if {(x['case'],x['seed']) for x in r['rows']} != cohort or r['manifest_sha256'] != base['manifest_sha256']:
                    raise ValueError('Unmatched inputs/model manifest')
                if sum(x['original_bytes'] for x in runner['expansion']['removed']) != removed:
                    raise ValueError('Unexpected I2V trim')
                if r['excluded_vae_state_bytes'] != ck['components']['vae']['bytes']:
                    raise ValueError('VAE boundary changed')
                source_names = ['cosmos/linears.py','_native.py','cosmos/ar_cache.py','cosmos/engine.py']
                if device == 'Thor': source_names += ['_sm110_cutlass.py','cutlass/layout.py']
                check_source(r,backend,source_names,sources)
                q = dict(base_codes=0,fp8_scale=0,w0_scales=0,extra_w0_scale_layout=0,prescale=0,sensitivity=0,
                         outer_w0_factor=0,base_csr_offsets=0,w1_codes=0,w1_scales=0,w1_columns=0,w1_locations=0,w1_extra_offsets=0)
                count = len(dims); total_params = target_bf16//2; blocks=0
                if method == 'bf16':
                    q['base_codes'] = target_bf16
                    if runner['prepared_weight_bytes'] != 0: raise ValueError('Unexpected BF16 prepared weights')
                elif method == 'fp8':
                    q.update(base_codes=total_params, fp8_scale=4*count)
                    if sum(q.values()) != runner['prepared_weight_bytes']: raise ValueError('FP8 storage mismatch')
                else:
                    current,unions,supports,ops = dimensions(r)
                    if current != dims: raise ValueError('Changed target scope')
                    blocks = sum(map(len,unions.values()))
                    q.update(base_codes=total_params//2,w0_scales=total_params//16,
                        extra_w0_scale_layout=total_params//16 if device=='Thor' else 0,
                        prescale=4*sum(k for n,k in dims.values()), sensitivity=4*sum(k for n,k in dims.values()),
                        outer_w0_factor=4*count,base_csr_offsets=4*sum(n//128+1 for n,k in dims.values()),
                        w1_codes=blocks*128*64,w1_scales=blocks*128*8,
                        w1_columns=4*sum(len(s) for sets in supports.values() for s in sets),
                        w1_locations=4*sum(map(len,ops.values())) if runner['config']['budget_plan'] else 0,
                        w1_extra_offsets=4*sum((dims[name][0]//128+1)*(len(s)-1) for name,s in supports.items()))
                    # Reconstruct from shapes/indices, not prepared bytes minus an
                    # unexplained residual. Sensitivity is a registered buffer.
                    if sum(q.values())-q['sensitivity'] != runner['prepared_weight_bytes']:
                        raise ValueError('FP4 prepared storage does not reconcile: '+str((device,model,method)))
                quantized = sum(q.values())
                if common_bytes+quantized != r['body_resident_model_bytes'] or runner['resident_model_state_bytes'] != r['body_resident_model_bytes']:
                    raise ValueError('Resident byte reconciliation failed')
                resident.append(dict(device=device,model=model,method=method,**common,**q,
                    target_bf16_bytes=target_bf16,target_resident_bytes=quantized,body_resident_bytes=common_bytes+quantized,
                    target_ratio=quantized/target_bf16,target_compression_factor=target_bf16/quantized,
                    body_ratio=(common_bytes+quantized)/baseline,
                    w1_union_blocks=blocks,w1_union_density=blocks/(total_params//128**2),
                    model_only_weight_scope='Generation FFN and Q/K/V/O; shared body components unchanged',
                    evidence=r['_audit_source'],asset_identity=runner['asset_identity']))
                ar = runner['ar_cache']; ar_total=ar['buffer_bytes']; channel_total=ar_k+ar_v+hidden
                if ar_total % channel_total: raise ValueError('AR cache split not integral')
                payload_per_channel=ar_total//channel_total
                named = dict(ar_k_cache=payload_per_channel*ar_k,ar_v_cache=payload_per_channel*ar_v,
                    ar_layer_outputs=payload_per_channel*hidden,prepared_selector_and_workspace=runner['support_buffer_bytes'],
                    cached_bf16_conditions=r['condition_tensor_bytes'])
                ready = r['body_ready_allocated_bytes']; peakrow=max(r['rows'],key=lambda x:x['body_peak_allocated_bytes'])
                peak=peakrow['body_peak_allocated_bytes']; state=r['body_resident_model_bytes']
                other = ready-state-sum(named.values()); extra=peak-ready
                if other<0 or extra<0: raise ValueError('Named runtime buckets overlap or cannot reconcile')
                runtime.append(dict(device=device,model=model,method=method,body_resident_bytes=state,**named,
                    other_ready_allocations=other,ready_allocated_bytes=ready,
                    request_highwater_above_ready=extra,request_peak_allocated_bytes=peak,
                    request_non_model_peak=peak-state,max_request_reserved_bytes=max(x['body_peak_reserved_bytes'] for x in r['rows']),
                    peak_ratio_vs_bf16=peak/baseline_peak,ar_slots=ar['slots'],n_requests=6,
                    peak_case=peakrow['case'],peak_seed=peakrow['seed'],evidence=r['_audit_source']))
    sources[str(Path(__file__).resolve())]=dict(sha256=digest(Path(__file__).read_bytes()))
    return dict(scope='Matched trimmed I2V body; BF16 VAE physically removed; 320x192, 61 frames, 35 steps; 2 scenes x 3 seeds per row.',
        resident=resident,runtime=runtime,scopes=scopes,modules=modules,sources=sources,
        verification=dict(resident_rows_exact=len(resident),runtime_rows_exact=len(runtime),checkpoint_headers_verified=True,
            all_weight_metadata_reconstructed=True,weight_tail_padding=False,all_fp4_w0_code_ratios=.25),
        limits=['CUDA tensor storage and allocator bytes; GiB=2**30, MiB=2**20. Not physical Thor system RAM.',
            'Other ready allocations jointly include graph inputs/outputs/intermediates, attention/packing scratch and allocator rounding; no separate pure activation peak.',
            'AR cache payload split follows retained k/v/out shapes and the matched AR-cache implementation; three warmed slots, including both scene templates.',
            'Request highwater above ready is a difference of live-memory levels, not total allocation traffic.',
            'W1 union is stored once across phases; A1 is online scratch, not a second resident model.',
            'FP4 Q/K/V/P operands do not imply that hidden states, residuals, KV, softmax and accumulators are all stored in FP4.',
            'Extended transformer2 quantization is a separate quality-changing configuration and is not used in these primary rows.',
            'No new GPU run, quality result, or performance claim.'])


def markdown(report):
    rows=report['resident'];rt=report['runtime'];sc=report['scopes']
    get=lambda model,method:next(r for r in rows if r['device']=='Thor' and r['model']==model and r['method']==method)
    text=['# 为什么低精度主体内存没有缩至 BF16 的 1/2 或 1/4','',
        '本表绑定最近完成的 VAE 隔离、相同 I2V 裁剪实验。FP8/W4A4 是生成分支 GEMM 的精度定义；AR 和其他公共组件仍保留原精度。20 行常驻和请求峰值均逐字节对账。没有重新运行 GPU 或改变推理配置。','',
        '## 1. 哪些矩阵压到了 1/4','',
        '| W0 权重编码 | Edge BF16 → FP4 GiB | Nano BF16 → FP4 GiB | 比例 |','|---|---:|---:|---:|']
    for group,label in [('generation_ffn','生成分支 FFN'),('generation_qkvo','生成分支 Q/K/V/O 投影')]:
        vals=[]
        for model in ('edge','nano'):
            subset=[r for r in report['modules'] if r['model']==model and r['category']==group]
            vals.append(f"{sum(r['bf16_weight_bytes'] for r in subset)/GIB:.3f} → {sum(r['fp4_w0_code_bytes'] for r in subset)/GIB:.3f}")
        text.append(f'| {label} | {vals[0]} | {vals[1]} | 25.00% |')
    text += ['','这里的 1/4 指 W0 的四位编码载荷，未包含量化尺度或 W1。attention 中 QK/softmax/PV 的临时张量不属于这张常驻权重表。','',
        '## 2. 主体中超过一半仍保持原存储','',
        '| 主体 BF16 分项 | Edge GiB | Nano GiB | 主比较是否压缩 |','|---|---:|---:|---|']
    groups=[('ar_ffn','AR FFN'),('ar_qkvo','AR Q/K/V/O'),('embedding','文本 embedding'),('video_io','视频输入/输出头'),('timestep_fp32','timestep FP32'),('norms','norm'),('rotary_buffer','rotary buffer')]
    for key,label in groups:
        digits=9 if key=='rotary_buffer' else 6
        text.append(f"| {label} | {get('edge','bf16')[key]/GIB:.{digits}f} | {get('nano','bf16')[key]/GIB:.{digits}f} | 否 |")
    for model in ('edge','nano'):
        s=next(x for x in sc if x['device']=='Thor' and x['model']==model)
        text+=['',f"{model.title()}：量化目标 {s['target_bf16_bytes']/GIB:.3f} GiB，占主体 {s['target_fraction']:.2%}；未压缩主体 {s['unchanged_body_bytes']/GIB:.3f} GiB。仅保留理想 4-bit codes 时，主体大小的理论下界仍是 BF16 的 **{s['fp4_codes_only_ideal_body_ratio']:.2%}**（最多 {s['fp4_codes_only_body_compression_limit']:.3f}× 压缩）。"]
    text+=['','令 f 为 BF16 主体中的量化目标占比，r 为目标压缩后的字节比例，则主体比例为 `1 − f + f·r`。FP8 的 r≈1/2；FP4 仅 codes 的 r=1/4。该公式适用于当前固定覆盖范围，不能当作全模型量化的极限。','',
        '## 3. 从四位 codes 到真实常驻：Thor 字节账本','',
        '| 项目 | Edge MiB | Nano MiB |','|---|---:|---:|']
    base_groups=[('base_codes','W0 FP4 codes'),('w0_scales','K16 E4M3 scales'),('extra_w0_scale_layout','CUTLASS scale 布局副本'),('prescale','FP32 prescale'),('sensitivity','FP32 通道敏感度'),('outer_w0_factor','FP32 外尺度'),('base_csr_offsets','CSR 行偏移')]
    for key,label in base_groups:
        text.append(f"| {label} | {get('edge','fp4_base')[key]/MIB:.4f} | {get('nano','fp4_base')[key]/MIB:.4f} |")
    text+=['','每 16 个 FP4 元素有一个一字节尺度：单布局载荷为 4.5 bit/weight，是 BF16 的 28.125%。Thor 为主 GEMM 和修复保留两种尺度布局，基础载荷达到 5 bit/weight，即 31.25%，另有上表的小量元数据。PRO 6000 当前路径只有一份尺度布局，分项也在 CSV/JSON 中。','',
        '| 模型 | 修复档位 | W1 支持并集密度 | W1 codes+scales MiB | W1 索引 MiB |','|---|---|---:|---:|---:|']
    for model in ('edge','nano'):
        for method in METHODS[3:]:
            r=get(model,method)
            text.append(f"| {model} | {LABELS[method]} | {r['w1_union_density']:.3%} | {(r['w1_codes']+r['w1_scales'])/MIB:.4f} | {(r['w1_columns']+r['w1_locations']+r['w1_extra_offsets'])/MIB:.4f} |")
    text+=['','1.5×/2× 是计算预算，不是权重存储倍率。W1 仅保存跨阶段支持并集；激活 A1 是在线工作区。主体常驻中没有因此加入第二份完整 BF16 生成权重。','',
        '## 4. 三种分母分别报告','',
        '| 模型 | 方法 | 目标真实常驻/BF16 | 目标压缩倍数 | 主体常驻/BF16 | 主体峰值/BF16 |','|---|---|---:|---:|---:|---:|']
    for model in ('edge','nano'):
        for method in METHODS:
            r=get(model,method);v=next(x for x in rt if x['device']=='Thor' and x['model']==model and x['method']==method)
            text.append(f"| {model} | {LABELS[method]} | {r['target_ratio']:.2%} | {r['target_compression_factor']:.3f}× | {r['body_ratio']:.2%} | {v['peak_ratio_vs_bf16']:.2%} |")
    text+=['','## 5. 请求内存：可辨认常驻缓冲与其余分配','',
        '以下为 Thor，单位 MiB。这里的 AR 输出指每层保留的条件 hidden states，与 K/V 分列。三种 AR cache 都在三个已预热形状槽中保留，所有方法相同。','',
        '| 模型/方法 | AR K/V | AR 层输出 | selector/已登记 workspace | BF16 条件 | 其余 ready 分配 | 请求高水位增量 | 非模型峰值合计 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in rt:
        if v['device']!='Thor':continue
        vals=[v['ar_k_cache']+v['ar_v_cache'],v['ar_layer_outputs'],v['prepared_selector_and_workspace'],v['cached_bf16_conditions'],v['other_ready_allocations'],v['request_highwater_above_ready'],v['request_non_model_peak']]
        text.append(f"| {v['model']}/{LABELS[v['method']]} | "+' | '.join(f'{x/MIB:.3f}' for x in vals)+' |')
    text+=['','`峰值 = 主体常驻 + AR K/V + AR 层输出 + 已登记 selector/workspace + 固定条件 + 其余 ready 分配 + 相对 ready 的高水位差`。',
        '“其余 ready”包含 Graph 输入/输出及中间存储、attention/packing scratch 和 allocator 对齐，现有 receipt 没有将其逐项分离，不能把余量改名为纯激活或纯 Graph 内存。“高水位增量”是 live memory 水位差，不是一次请求分配过的所有字节之和。',
        'W4A4 仍保留 BF16 hidden states/残差与 AR cache，并产生 FP4 packing/repair 缓冲；softmax 与累加也不统一存为 FP4。因此它不承诺激活峰值降至 1/4。当前峰值收益主要来自权重，非模型余量可以略增。','',
        '## 6. 可用于论文的表述','',
        '> We quantize the generation-branch FFN and Q/K/V/O projections. Their base FP4 codes occupy exactly 25% of the corresponding BF16 weight bytes. This scope accounts for 45.5% (Edge) and 47.7% (Nano) of the trimmed Cosmos3 body; AR weights, embeddings, and auxiliary parameters remain at their original precision. Including group scales, retained backend layouts, sparse residual weights, and indices, the 1.5× configuration stores the targeted weights at approximately 32.7% of BF16 on Thor (3.06× compression). The resulting body request-peak reductions are 28.5% and 31.4%, respectively. Peak memory also includes graph-resident buffers, AR caches, and packing/repair workspaces. All methods use the same I2V trimming, and the BF16 VAE is excluded from this body-only measurement.',
        '', '不要将 “FP4 codes 为 1/4”写成“整个模型或请求峰值为 1/4”；不要把当前保留 AR 的工程范围解释为 AR 永远不能量化。扩展 AR/embedding 的结果与质量代价另见上一轮主体实验。','',
        '## 证据与复测','',
        '逐矩阵载荷见 `quantized_modules.csv`，常驻各项见 `resident_components.csv`，请求分项见 `request_components.csv`。`breakdown.json` 记录输入文件 hash、匹配源码与验证结果。所有 20 行常驻由张量维度和实际支持重建，与实测逐字节相等。','',
        '本轮仅作 CPU 证据核对；沿用原始实验的两个场景、每场景三个种子。CUDA allocated 与 reserved 分开保存；不与 Thor RSS 相加，不声称系统物理 RAM 或冷启动峰值同等缩减。','']
    return '\n'.join(text)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,required=True)
    p.add_argument('--model-root',type=Path,help='Local parent of the unchanged checkpoint directories')
    p.add_argument('--pro-backend',type=Path,required=True)
    p.add_argument('--thor-backend',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    report=build(args)
    write(args.output/'breakdown.json',report)
    csv_write(args.output/'resident_components.csv',report['resident'])
    csv_write(args.output/'request_components.csv',report['runtime'])
    csv_write(args.output/'quantized_modules.csv',report['modules'])
    (args.output/'README.zh.md').write_text(markdown(report))
    print(json.dumps(report['verification']))


if __name__=='__main__':main()
