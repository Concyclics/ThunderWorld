"""Reconcile FFN-only expansion receipts and compare native ARC50/SharQ assets."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

GIB = 2**30
METHODS = ('bf16', 'fp8', 'fp4_base', 'arcquant', 'sharq_perf', 'b1.5_protected_n', 'b2_joint')
LABELS = dict(bf16='BF16', fp8='FP8', fp4_base='W4A4', arcquant='ARCQuant 50%',
              sharq_perf='SharQ', **{'b1.5_protected_n': 'Repair 1.5×', 'b2_joint': 'Repair 2×'})
PROFILES = dict(bf16='trim', fp8='ar_ffn8', fp4_base='ar_ffn1', arcquant='ar_ffn4',
               sharq_perf='ar_ffn4', **{'b1.5_protected_n': 'ar_ffn4', 'b2_joint': 'ar_ffn4'})


def comparison_profiles(ar_fp4_profile):
    if ar_fp4_profile not in ('ar_ffn1','ar_ffn4'):
        raise ValueError('Unknown AR FP4 comparison profile')
    return {method:(ar_fp4_profile if method in
            ('arcquant','sharq_perf','b1.5_protected_n','b2_joint') else profile)
            for method,profile in PROFILES.items()}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def historical_groups(row):
    """Previously independently reconstructed bytes, only for unchanged trim rows."""
    parts = dict(base_payload=row['base_codes'],
        metadata=sum(row[k] for k in ('fp8_scale','w0_scales','prescale','sensitivity','outer_w0_factor','base_csr_offsets')),
        extra_scale_layout=row['extra_w0_scale_layout'],
        repair_payload=row['w1_codes'],
        repair_metadata=sum(row[k] for k in ('w1_scales','w1_columns','w1_locations','w1_extra_offsets')))
    result = [dict(component='generation', part=k, bytes=v) for k,v in parts.items()]
    result.extend(dict(component=k, part='base_payload' if k=='ar_ffn' else 'unquantized', bytes=row[k])
                  for k in ('ar_ffn','ar_qkvo','embedding'))
    result.append(dict(component='auxiliary', part='unquantized',
                      bytes=sum(row[k] for k in ('video_io','timestep_fp32','norms','rotary_buffer'))))
    return result


def groups_bytes(groups, components, part=None):
    return sum(g['bytes'] for g in groups if g['component'] in components and
               (part is None or g['part'] == part))


def collect(root, device, prior, output):
    residents, components, requests, samples = [], [], [], []
    controls={model:json.loads((root/'body-runs'/f'{model}-bf16-trim/result.json').read_text())
              for model in ('edge','nano')}
    for folder in sorted((root/'body-runs').iterdir()):
        path = folder/'result.json'
        if not path.exists(): continue
        r = json.loads(path.read_text())
        if not r.get('complete'): continue
        if ('Thor' if device=='Thor' else 'RTX PRO 6000') not in r['device']:
            raise AssertionError('Measured device does not match report label')
        model, method, profile = r['model'], r['method'], r['profile_name']
        if method not in METHODS: continue
        control=controls[model]
        if (r['runner']['model_identity'] != control['runner']['model_identity'] or
                r['condition_sha256'] != control['condition_sha256'] or
                r['excluded_vae_state_bytes'] != control['excluded_vae_state_bytes'] or
                r['runner']['targets'] != control['runner']['targets']):
            raise AssertionError('Changed model, conditioning, VAE or projection scope')
        if {(v['case'],v['seed']) for v in r['rows']} != {(v['case'],v['seed']) for v in control['rows']}:
            raise AssertionError('Unmatched request identities')
        if not r['vae_fully_released'] or not r['latent_isolation_bitwise_equal']:
            raise AssertionError('Invalid VAE measurement boundary')
        if len(r['rows']) != 6 or any(v['calls']!=70 or v['ar_prefills']!=2 for v in r['rows']):
            raise AssertionError('Changed six-request cohort')
        if r['profile'].get('embedding_planes',0) or r['profile'].get('auxiliary_planes',0) or r['profile'].get('vae_planes',0):
            raise AssertionError('Not an AR-FFN-only comparison')
        if profile!='trim' and r['profile'].get('ar_scope')!='ffn':
            raise AssertionError('AR attention coverage changed')
        groups = r.get('resident_inventory',{}).get('groups')
        if groups is None:
            if profile!='trim': raise AssertionError('Missing expanded live inventory')
            groups=historical_groups(prior[device,model,method])
        if sum(g['bytes'] for g in groups) != r['body_resident_model_bytes']:
            raise AssertionError('Resident bytes do not reconcile')
        if profile=='ar_ffn1':
            if groups_bytes(groups,{'ar_ffn'},'repair_payload') or groups_bytes(groups,{'ar_ffn'},'repair_metadata'):
                raise AssertionError('Direct FP4 AR FFN must not retain W1 storage')
            expansion=r['runner']['expansion']
            if not expansion['ar_prepared_shapes'] or any(v['terms']!=1 for v in expansion['ar_prepared_shapes']):
                raise AssertionError('Direct FP4 AR FFN must execute one term')
            if any(v['kind']!='fp4_base' or '.mlp.' not in v['module'] for v in expansion['quantized']):
                raise AssertionError('Direct FP4 AR FFN coverage changed')
        gen={'generation','generation_ffn','generation_qkvo'}
        identity=dict(device=device,model=model,method=method,profile=profile)
        resident=dict(**identity,
            generation_bytes=groups_bytes(groups,gen),
            ar_ffn_bytes=groups_bytes(groups,{'ar_ffn'}),
            ar_qkvo_bytes=groups_bytes(groups,{'ar_qkvo'}),
            embedding_bytes=groups_bytes(groups,{'embedding'}),
            auxiliary_bytes=groups_bytes(groups,{'auxiliary'}),
            body_resident_bytes=r['body_resident_model_bytes'],
            ready_allocated_bytes=r['body_ready_allocated_bytes'],
            request_peak_allocated_bytes=max(v['body_peak_allocated_bytes'] for v in r['rows']),
            request_peak_reserved_bytes=max(v['body_peak_reserved_bytes'] for v in r['rows']),
            execution=r['runner']['config']['execution'],
            ar_arithmetic=r['profile'].get('ar','bf16'),
            receipt=str(path),receipt_sha256=sha(path),performance_eligible=False)
        resident['request_non_model_bytes']=resident['request_peak_allocated_bytes']-resident['body_resident_bytes']
        if min(resident['request_non_model_bytes'],resident['ready_allocated_bytes']-resident['body_resident_bytes'])<0:
            raise AssertionError('Negative non-model memory')
        residents.append(resident)
        components.extend(dict(**identity,**g) for g in groups)
        for value in r['rows']:
            requests.append(dict(**identity,case=value['case'],seed=value['seed'],
                peak_allocated_bytes=value['body_peak_allocated_bytes'],peak_reserved_bytes=value['body_peak_reserved_bytes'],
                performance_eligible=False))
            samples.append(dict(**identity,case=value['case'],seed=value['seed'],
                latent=str(folder/value['output']),npz_sha256=value['npz_sha256']))
        archive=output/'raw'/device/folder.name/'result.json.gz'
        archive.parent.mkdir(parents=True,exist_ok=True)
        archive.write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    lookup={(r['model'],r['method'],r['profile']):r for r in residents}
    for r in residents:
        ref=lookup[r['model'],'bf16','trim']
        if (r['ar_qkvo_bytes'],r['embedding_bytes'],r['auxiliary_bytes']) != (
                ref['ar_qkvo_bytes'],ref['embedding_bytes'],ref['auxiliary_bytes']):
            raise AssertionError('Unchanged components unexpectedly differ')
        base=lookup.get((r['model'],r['method'],'trim'))
        if base:
            if r['generation_bytes']!=base['generation_bytes']:
                raise AssertionError('AR expansion changed generation resident assets')
            r['ar_ffn_saved_bytes']=base['ar_ffn_bytes']-r['ar_ffn_bytes']
            r['body_saved_vs_same_dm_bytes']=base['body_resident_bytes']-r['body_resident_bytes']
            if r['body_saved_vs_same_dm_bytes']!=r['ar_ffn_saved_bytes']:
                raise AssertionError('Extra saving attributed to AR FFN')
        r['body_reduction_vs_bf16']=1-r['body_resident_bytes']/ref['body_resident_bytes']
    return residents,components,requests,samples


def chart_rows(residents,components,device,extended,ar_fp4_profile='ar_ffn4'):
    result=[]
    profiles=comparison_profiles(ar_fp4_profile)
    for model in ('edge','nano'):
        for method in METHODS:
            profile=profiles[method] if extended else 'trim'
            match=[r for r in residents if (r['device'],r['model'],r['method'],r['profile'])==(device,model,method,profile)]
            if len(match)!=1: return []
            groups=[r for r in components if (r['device'],r['model'],r['method'],r['profile'])==(device,model,method,profile)]
            selected={'generation','generation_ffn','generation_qkvo'} | ({'ar_ffn'} if extended else set())
            result.append(dict(device=device,model=model,method=method,profile=profile,
                base_payload_bytes=groups_bytes(groups,selected,'base_payload'),
                base_metadata_bytes=groups_bytes(groups,selected,'metadata'),
                extra_scale_layout_bytes=groups_bytes(groups,selected,'extra_scale_layout'),
                w1_bytes=groups_bytes(groups,selected,'repair_payload')+groups_bytes(groups,selected,'repair_metadata'),
                target_total_bytes=groups_bytes(groups,selected),
                ar_fp4_comparison_profile=ar_fp4_profile,
                scope='DM + AR FFN' if extended else 'DM',display_unit='GiB'))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--thor-root',type=Path)
    p.add_argument('--prior-breakdown',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    old=json.loads(a.prior_breakdown.read_text())
    prior={(r['device'],r['model'],r['method']):r for r in old['resident']}
    residents=[];components=[];requests=[];samples=[]
    for device,root in [('PRO6000',a.root)]+([('Thor',a.thor_root)] if a.thor_root else []):
        data=collect(root,device,prior,a.output)
        for target,values in zip((residents,components,requests,samples),data):target.extend(values)
    residents.sort(key=lambda r:(r['device'],r['model'],METHODS.index(r['method']),r['profile']!='trim'))
    quality_path=a.root/'body-quality/report.json'
    protocol=json.loads((a.root/'protocol.json').read_text())
    quality_requested=protocol.get('quality_requested',True)
    quality=json.loads(quality_path.read_text()) if quality_requested and quality_path.exists() else None
    ar_fp4_profile=protocol.get('comparison_ar_fp4_profile','ar_ffn4')
    profiles=comparison_profiles(ar_fp4_profile)
    comparison=[r for r in residents if r['profile']==profiles[r['method']]]
    direct_deltas=[]
    lookup={(r['device'],r['model'],r['method'],r['profile']):r for r in residents}
    for r in residents:
        before=lookup.get((r['device'],r['model'],r['method'],'ar_ffn4'))
        if r['profile']=='ar_ffn1' and before:
            direct_deltas.append(dict(device=r['device'],model=r['model'],method=r['method'],
                from_profile='ar_ffn4',to_profile='ar_ffn1',
                ar_ffn_saved_bytes=before['ar_ffn_bytes']-r['ar_ffn_bytes'],
                body_resident_saved_bytes=before['body_resident_bytes']-r['body_resident_bytes'],
                request_peak_saved_bytes=before['request_peak_allocated_bytes']-r['request_peak_allocated_bytes']))
    write(a.output/'breakdown.json',dict(resident=residents,components=components,
        requests=requests,samples=samples,quality=quality,
        comparison=comparison,comparison_profiles=profiles,ar_direct_deltas=direct_deltas,
        prior_breakdown_sha256=sha(a.prior_breakdown),quality_complete=quality is not None,
        quality_requested=quality_requested,
        quality_status='not_requested' if not quality_requested else ('complete' if quality else 'pending'),
        scope='Body-only, BF16 VAE excluded; isolated AR FFN extension',performance_eligible=False))
    csv_write(a.output/'resident_components.csv',residents)
    csv_write(a.output/'comparison.csv',comparison)
    if direct_deltas:csv_write(a.output/'ar_direct_vs_four.csv',direct_deltas)
    csv_write(a.output/'storage_parts.csv',components)
    csv_write(a.output/'request_components.csv',requests)
    write(a.output/'samples.json',samples)
    if quality:
        if len(quality['golden_checks']) != 12 or not all(v['bitwise_equal'] for v in quality['golden_checks']):
            raise AssertionError('Missing golden BF16 decode qualification')
        expected={(r['model'],r['method'],r['profile'],r['case'],r['seed'])
                  for r in samples if r['device']=='PRO6000'}
        observed={(r['model'],r['method'],r['profile'],r['case'],r['seed'])
                  for r in quality['rows']}
        if expected!=observed or len(observed)!=len(quality['rows']):
            raise AssertionError('Missing or duplicated quality samples')
        csv_write(a.output/'quality.csv',[dict(model=r['model'],method=r['method'],profile=r['profile'],n=r['n'],**r['mean_metrics']) for r in quality['summary']])
        lookup={(r['model'],r['method'],r['profile']):r for r in quality['summary']}
        samples_by_key={(r['model'],r['method'],r['profile'],r['case'],r['seed']):r['metrics'] for r in quality['rows']}
        deltas=[]
        for r in quality['summary']:
            if r['profile']=='trim':continue
            ref=lookup[r['model'],r['method'],'trim']
            delta=dict(model=r['model'],method=r['method'],profile=r['profile'],n=r['n'])
            for key in ('lpips','ssim_mean','psnr_db','relative_l2'):
                left,right=r['mean_metrics'][key],ref['mean_metrics'][key]
                delta['delta_'+key]=left-right if left is not None and right is not None else None
            pairs=[v['metrics']['lpips']-samples_by_key[r['model'],r['method'],'trim',v['case'],v['seed']]['lpips']
                   for v in quality['rows'] if (v['model'],v['method'],v['profile'])==(r['model'],r['method'],r['profile'])]
            if len(pairs)!=r['n']:raise AssertionError('Unmatched quality pairs')
            delta['worst_seed_delta_lpips']=max(pairs)
            deltas.append(delta)
        csv_write(a.output/'ar_ffn_quality_deltas.csv',deltas)
    for device in sorted({r['device'] for r in residents}):
        for extended in (False,True):
            rows=chart_rows(residents,components,device,extended,ar_fp4_profile)
            if rows:write(a.output/f"memory_breakdown_{device.lower()}_{'ar_ffn' if extended else 'dm_only'}.chart-data.json",rows)
    lines=['# AR FFN 单独量化与 baseline 内存分项','',
        '同一模型、两场景各三个种子，320×192、61 帧、35 步。VAE 在测量前真正释放；AR Q/K/V/O、embedding 和辅助投影保持原精度。',
        '扩展范围对比中的 FP8 行使用 DM FP8 + AR FFN 原生 W8A8 FP8（ar_ffn8），权重离线量化、激活在线 per-token 量化并调用 CUTLASS FP8 GEMM。表中 AR=bf16 的 FP8 行仅保留为原范围对照。',
        ('当前主对比的 FP4 方法统一采用直接 W4A4 AR FFN（ar_ffn1），仅执行基础项，不保存 AR W1。四项 AR 行仅为历史对照；comparison.csv 和图固定使用直接量化行。' if ar_fp4_profile=='ar_ffn1' else '当前主对比的修复 FP4 方法使用 AR FFN 四项修复（ar_ffn4）；普通 W4A4 对照仍用单项 AR。'),
        'ARCQuant 50% / SharQ 保留原生 eager 生成分支与 BF16 attention；AR FFN 为公共适配，不能称为官方默认全模型方案。1.5×/2× 仍为 DM 预算，AR 工作单独记录。',
        '本轮共享设备耗时不用于速度比较；下面是 CUDA 去重常驻及 peak allocated，不是纯激活峰值。','',
        '| 设备 | 模型 | 方法 | AR FFN | DM 资产 GiB | AR FFN GiB | 其他主体 GiB | 主体常驻 GiB | 主体峰值 GiB |',
        '|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in residents:
        other=r['ar_qkvo_bytes']+r['embedding_bytes']+r['auxiliary_bytes']
        lines.append(f"| {r['device']} | {r['model']} | {LABELS[r['method']]} | {r['ar_arithmetic']} | {r['generation_bytes']/GIB:.3f} | {r['ar_ffn_bytes']/GIB:.3f} | {other/GIB:.3f} | {r['body_resident_bytes']/GIB:.3f} | {r['request_peak_allocated_bytes']/GIB:.3f} |")
    lines+=['','图分别给出原 DM 范围和 DM+AR FFN 范围，仅画目标权重，不画未量化部分或峰值余量。柱顶标实际 GiB；四项图例为基础权重、尺度与元数据、额外尺度布局、W1 与索引。','',
            '单平面 W4 的编码 payload 是同范围 BF16 权重的 1/4；尺度、对齐与索引另计。直接量化 AR 行没有 W1；历史 AR 四项行保存两个权重平面，不能混用这两种配置的质量或内存结论。',
            '本次 ARCQuant 50% 原生适配的 weight_codes 宽度为原 K 的 1.5 倍，权重编码占 BF16 的 37.5%；SharQ 同时常驻 dense/sparse 两份尺度布局，额外布局单独着色。这些是当前实现的实际存储，而不是所有同类实现的必然下限。','',
            '完整存储账本为 storage_parts.csv；每个常驻值与 live CUDA storage 去重核对。原始结果无损压缩保存在 raw，latent 文件索引及 hash 保存在 samples.json。','']
    if direct_deltas:
        lines+=['## AR 从四项修复切换为直接量化的内存变化','',
                '| 模型 | DM 方法 | 常驻减少 GiB | 请求峰值减少 GiB |',
                '|---|---|---:|---:|']
        for r in direct_deltas:
            if r['ar_ffn_saved_bytes']!=r['body_resident_saved_bytes']:
                raise AssertionError('Direct-AR saving includes an unrelated resident change')
            lines.append(f"| {r['model']} | {LABELS[r['method']]} | {r['body_resident_saved_bytes']/GIB:.3f} | {r['request_peak_saved_bytes']/GIB:.3f} |")
        lines+=['','请求峰值来自各配置独立运行的六个真实请求，不是把旧峰值减去理论权重大小。']
    if quality:
        lines+=['## 相对同条件 BF16 的视频一致性','',
            '| 模型 | 方法 | AR 配置 | n | LPIPS ↓ | SSIM ↑ | PSNR dB ↑ | 相对 L2 ↓ |',
            '|---|---|---|---:|---:|---:|---:|---:|']
        for r in quality['summary']:
            m=r['mean_metrics'];psnr='∞' if m['psnr_db'] is None else f"{m['psnr_db']:.3f}"
            lines.append(f"| {r['model']} | {LABELS[r['method']]} | {r['profile']} | {r['n']} | {m['lpips']:.5f} | {m['ssim_mean']:.5f} | {psnr} | {m['relative_l2']:.5f} |")
        lines+=['','## 单独加入 AR FFN 量化的变化','',
            '以下差值为扩展后减去相同 DM 方法的原范围结果。LPIPS/L2 越负越好，SSIM/PSNR 越正越好；保留退化值，不把平均值的小幅变化称为质量等价。','',
            '| 模型 | 方法 | AR 配置 | ΔLPIPS | ΔSSIM | ΔPSNR dB | Δ相对 L2 | 最差种子 ΔLPIPS |',
            '|---|---|---|---:|---:|---:|---:|---:|']
        for r in deltas:
            psnr='n/a' if r['delta_psnr_db'] is None else f"{r['delta_psnr_db']:+.3f}"
            lines.append(f"| {r['model']} | {LABELS[r['method']]} | {r['profile']} | {r['delta_lpips']:+.5f} | {r['delta_ssim_mean']:+.5f} | {psnr} | {r['delta_relative_l2']:+.5f} | {r['worst_seed_delta_lpips']:+.5f} |")
        lines+=['','这是两个已使用场景的六请求机制验证，未作为大 benchmark 质量门槛或跨场景泛化结论。BF16 对自身的 PSNR 为无穷大；涉及无穷大的 PSNR 差值不列有限数值。']
    elif not quality_requested:
        lines+=['按本轮要求只补内存压缩效果，未重测质量。新增 AR FFN 配置不能据此宣称质量等价；历史原范围质量结果仍保留。']
    else:lines+=['质量解码与评分尚未完成，不能从内存收益推断质量等价。']
    (a.output/'README.zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(resident_rows=len(residents),requests=len(requests),quality_complete=quality is not None)))


if __name__=='__main__':main()
