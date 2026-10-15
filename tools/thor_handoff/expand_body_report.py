"""Report directly measured body-only storage/peaks, with explicit scope."""
import argparse
import json
from pathlib import Path
import statistics
from expand_worker import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=[]
    for path in sorted((a.root/'body-runs').glob('*/result.json')):
        r=json.loads(path.read_text())
        if not r.get('complete'):continue
        if not (r.get('vae_fully_released') and r.get('latent_isolation_bitwise_equal')):
            raise ValueError('Unqualified codec isolation')
        if any(s['calls']!=70 or s['ar_prefills']!=2 for s in r['rows']):raise ValueError('Incomplete 35-step/CFG execution')
        rows.append(dict(model=r['model'],method=r['method'],profile=r['profile_name'],device=r['device'],
            n=len(r['rows']),resident_bytes=r['body_resident_model_bytes'],
            peak_bytes=max(s['body_peak_allocated_bytes'] for s in r['rows']),
            peak_reserved_bytes=max(s['body_peak_reserved_bytes'] for s in r['rows']),
            ready_allocated_bytes=r['body_ready_allocated_bytes'],
            non_model_peak_bytes=max(s['body_non_model_peak_bytes'] for s in r['rows']),
            ready_non_model_bytes=r['body_ready_non_model_bytes'],
            excluded_vae_state_bytes=r['excluded_vae_state_bytes'],
            condition_tensor_bytes=r['condition_tensor_bytes'],
            median_body_seconds=statistics.median(s['body_seconds'] for s in r['rows']),
            vae_fully_released=True,latent_isolation_bitwise_equal=True,evidence=str(path),
            evidence_sha256=sha(path),
            performance_eligible=False,scope=r['scope']))
    if len({r['device'] for r in rows})>1:
        raise ValueError('Report each device independently; do not pool device-specific storage layouts')
    order={name:i for i,name in enumerate(('bf16','fp8','fp4_base','b1.5_protected_n','b2_joint'))}
    rows.sort(key=lambda r:(r['model'],r['profile']!='trim',order.get(r['method'],99)))
    bases={r['model']:r for r in rows if r['method']=='bf16' and r['profile']=='trim'}
    for r in rows:
        if r['model'] in bases:
            b=bases[r['model']]
            r.update(resident_reduction_vs_trimmed_bf16=1-r['resident_bytes']/b['resident_bytes'],
                peak_reduction_vs_trimmed_bf16=1-r['peak_bytes']/b['peak_bytes'])
    report=dict(rows=rows,performance_eligible=False,
        scope='Cosmos3 AR+DM body. BF16 image encoding is fixed outside measurement; codec is physically released.',
        exclusions='VAE encoder/decoder, video postprocess and CPU output conversion; not full video generation time',
        memory_scope='PyTorch CUDA storage/allocator bytes, GiB = 2**30 bytes; not process RSS or physical system RAM',
        caveat='Peak includes weights, Graph pools, AR/KV, inputs, outputs, activations and workspaces. Non-model remainder is not pure activation memory.',
        metadata_note='Raw expansion.protected was scanned before codec isolation and can list the removed VAE; early runs may also predate DM replacement. Use body_resident_model_bytes for the final measured state.')
    write(a.output/'body-report.json',report)
    lines=['# Cosmos3 主体内存（BF16 VAE 边界）','',
        '相同 I2V 裁剪、320×192、61 帧、35 步、两个 CFG prefill。编码前置且所有方法使用 BF16；编码后删除 VAE，检查对象已释放。隔离前后输出 latent 逐位一致。','',
        '设备：'+', '.join(sorted({r['device'] for r in rows}))+f'；已完成 {len(rows)} 个配置，每个配置两个场景、三个种子。单位 GiB = 2³⁰ bytes。','',
        '| 模型 | 生成分支方法 | 主体配置 | 常驻 GiB | 主体峰值 GiB | 常驻较 BF16 降低 | 峰值较 BF16 降低 |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        a1=r.get('resident_reduction_vs_trimmed_bf16');a2=r.get('peak_reduction_vs_trimmed_bf16')
        s1='—' if a1 is None else f'{a1:.1%}';s2='—' if a2 is None else f'{a2:.1%}'
        lines.append(f"| {r['model']} | {r['method']} | {r['profile']} | {r['resident_bytes']/2**30:.3f} | {r['peak_bytes']/2**30:.3f} | {s1} | {s2} |")
    lines+=['','`trim` 保留相同历史量化投影范围；`transformer2` 额外量化 AR/embedding/辅助矩阵，属于新覆盖范围消融，质量另行验收。',
        '主体峰值由删除 VAE 后的运行直接测得，不是从整请求峰值减去 VAE 权重。差额包含 Graph、AR/KV、activation 和 workspace，不应标为纯激活峰值。',
        '常驻和峰值采用 PyTorch CUDA 存储/分配口径；reserved 另存 JSON。Thor 的统一内存不另加进程 RSS，也不将这些值称为物理系统总内存。',
        'JSON 中的主体耗时是诊断结果，不含 VAE 与视频转换，不能称为端到端视频延迟/FPS。','']
    (a.output/'BODY_MEMORY.zh.md').write_text('\n'.join(lines))
    print(json.dumps(dict(rows=len(rows),output=str(a.output))))


if __name__=='__main__':main()
