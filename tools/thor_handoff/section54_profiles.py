"""Mutually exclusive kernel accounting for the ten independent Edge profiles."""
import argparse
import csv
import json
from pathlib import Path
from build_baselines import sha,write
from section54_protocol import write_csv


def category(name):
    lower=name.lower()
    if ('reorder_x_kernel' in name or 'fused_sparse_residual_x_kernel' in name or name in
        ('scaled_amax','finish_amax','pack_activation','pointwise_amax','pack_kernel','base_summary',
         'choose','residual_only','calibrated_relu_amax','pack_qk','pack_v','residual_qk','residual_v',
         'select_tiles_head','summarize','select_blocks','support_union','pack_qk_sparse','pack_v_sparse')):
        return 'preparation_selection'
    if name=='combine' or 'splitkv_combine_kernel' in name:return 'combine'
    if name=='streaming_attention' or 'flash_fwd' in name or 'fmha_' in name:return 'attention'
    if 'cutlass::device_kernel<' in name and ('GemmUniversal' in name or 'GemmUniversalAdapter' in name):
        return 'projection_ffn'
    return 'other'


def summarize(root):
    groups=('preparation_selection','projection_ffn','attention','combine','other')
    summary=[];kernels=[]
    for file in sorted((root/'profiles').glob('*.complete.json')):
        receipt=json.loads(file.read_text())
        tag=file.name.removesuffix('.complete.json')
        csv_path=file.parent/(tag+'-kernels_cuda_gpu_kern_sum.csv')
        request=file.parent/(tag+'-request/result.json')
        if sha(csv_path)!=receipt['csv_sha256'] or sha(request)!=receipt['request_sha256']:raise ValueError('Changed profile')
        rows=list(csv.DictReader(csv_path.open())); totals={k:0 for k in groups}
        for row in rows:
            kind=category(row['Name']);ns=int(row['Total Time (ns)']);totals[kind]+=ns
            kernels.append(dict(profile=tag,method_id=receipt['method_id'],case_index=receipt['case_index'],
                kernel=row['Name'],category=kind,calls=int(row['Instances']),nanoseconds=ns))
        total=sum(totals.values())
        if total!=sum(int(r['Total Time (ns)']) for r in rows):raise ValueError('Nonexclusive buckets')
        raw=json.loads(request.read_text());r=raw['rows'][0]
        summary.append(dict(profile=tag,method_id=receipt['method_id'],case=r['case'],seed=r['seed'],
            **{k+'_ms':v/1e6 for k,v in totals.items()},total_kernel_ms=total/1e6,
            profiled_wall_seconds=r['wall_seconds'],debug_off_wall_seconds=None,identifiable_gap_ms=None,
            source_csv=str(csv_path),source_sha256=sha(csv_path),receipt=receipt,
            classification_scope='Named dedicated preparation only; generic casts/reductions stay other; fused kernels indivisible',
            performance_eligible=False))
    write(root/'profiles/classification.json',dict(rows=summary,performance_eligible=False,
        scope='GPU duration sums only; no CPU API sum or inferred timeline gaps'))
    write_csv(root/'profiles/classification.csv',summary)
    write_csv(root/'profiles/kernel_details.csv',kernels)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    summarize(p.parse_args().root)
