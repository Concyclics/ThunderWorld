"""Archive bounded optimization evidence without models, videos, or full traces."""
import argparse
import gzip
import json
from pathlib import Path
import statistics

# Initialize the study import paths before resolving its baselines helper.
from bundle import ROOT
from baselines import sha, write


def control_summary(root, source):
    """Check the untouched controls against the previous generation receipts."""
    import numpy as np
    if not json.loads((root/'controls/benchmark_progress.json').read_text())['complete']:
        raise ValueError('Incomplete control sweep')
    summary = {}
    for method in ('bf16', 'fp8'):
        cases = {}
        repeat_hashes = {}
        for repeat in range(5):
            record = json.loads((root/'controls/benchmark'/f'{method}_frozen_r{repeat}'/'result.json').read_text())
            if not record['complete'] or record['identity']['effective_spec']['runtime_options']:
                raise ValueError('Incomplete or modified control')
            for case, detail in record['cases'].items():
                if len(detail['rows']) != 3 or not detail['complete']:
                    raise ValueError('Control requires three seeds in each round')
                reference = json.loads((source/'generation'/method/case/'result.json').read_text())
                if reference['model_identity'] != record['prepared']['model_identity']:
                    raise ValueError('Control model changed')
                gold = {r['seed']: r['sha256'] for r in reference['rows']}
                for row in detail['rows']:
                    key = (case, row['seed'])
                    if row['steps'] != 35:
                        raise ValueError('Control step count changed')
                    if row['seed'] in gold and row['sha256'] != gold[row['seed']]:
                        raise ValueError('Control no longer matches the frozen output')
                    if key in repeat_hashes and row['sha256'] != repeat_hashes[key]:
                        raise ValueError('Control output changed between rounds')
                    repeat_hashes[key] = row['sha256']
                cases.setdefault(case, dict(benchmark=detail['benchmark'], rows=[]))['rows'].extend(detail['rows'])
        summary[method] = {}
        for case, detail in cases.items():
            rows = detail['rows']
            seconds = [r['wall_seconds'] for r in rows]
            median = statistics.median(seconds)
            summary[method][case] = dict(benchmark=detail['benchmark'], requests=len(rows),
                median_seconds=median, p90_seconds=float(np.quantile(seconds, .9)), fps=61/median,
                allocated_peak_gib=max(r['peak_allocated_bytes'] for r in rows)/2**30,
                reserved_peak_gib=max(r['peak_reserved_bytes'] for r in rows)/2**30,
                existing_output_hashes_verified=True)
    return dict(methods=summary, performance_eligible=False,
        scope='Unchanged baseline implementations. Five randomized independent-load rounds, run after the FP4 sweep; cross-method ratios are diagnostic, not jointly interleaved estimates.')


def profile_summary(root):
    def group(name):
        if name == 'gemm':
            return 'fp4_projection_gemm'
        if name in {'base_summary', 'residual_only', 'scaled_amax', 'finish_amax',
                    'choose', 'calibrated_relu_amax'}:
            return 'linear_packing_and_selector'
        if name in {'streaming_attention', 'pack_qk', 'pack_v', 'residual_qk',
                    'residual_v', 'combine', 'support_union'}:
            return 'fp4_attention_including_packing'
        if 'cudnn' in name.lower() or ('xmma' in name and 'fprop' in name):
            return 'convolution_and_layout'
        return 'other'

    result = {}
    for variant in ('frozen', 'norm'):
        receipt = json.loads((root/'profile'/variant/'result.json').read_text())
        if not receipt['complete']:
            raise ValueError('Incomplete profile')
        entries = []
        for file in sorted((root/'profile'/variant).glob('*.trace.json')):
            trace = json.loads(file.read_text())
            kernels = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel' and e.get('ph') == 'X']
            if not kernels:
                raise ValueError('No kernel events in '+str(file))
            grouped = {}
            for event in kernels:
                item = grouped.setdefault(event['name'], dict(count=0, duration_us=0.))
                item['count'] += 1
                item['duration_us'] += event['dur']
            total = sum(e['dur'] for e in kernels)
            groups = {}
            for name, values in grouped.items():
                item = groups.setdefault(group(name), dict(count=0, duration_us=0.))
                item['count'] += values['count']
                item['duration_us'] += values['duration_us']
            for values in groups.values():
                values['fraction'] = values['duration_us']/total
            entries.append(dict(trace=file.name, trace_sha256=sha(file), kernel_count=len(kernels),
                summed_kernel_us=total, groups=groups,
                kernels=[dict(name=name, **values, fraction=values['duration_us']/total)
                         for name, values in sorted(grouped.items(), key=lambda item: -item[1]['duration_us'])]))
        if len(entries) != 2:
            raise ValueError('Expected two independent seed traces')
        result[variant] = entries
    return dict(variants=result, performance_eligible=False,
        scope='Separate instrumented complete requests. Summed kernel duration is neither request critical-path time nor Tensor Core utilization; CPU time is not added.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--reference-study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = a.study
    if not json.loads((root/'followup_progress.json').read_text())['complete']:
        raise ValueError('Finish the bounded follow-up first')
    a.output.mkdir(parents=True, exist_ok=True)
    write(a.output/'controls.json', control_summary(root, a.reference_study))
    write(a.output/'profile.json', profile_summary(root))
    paths = set()
    for pattern in ('*_protocol.json', '*_progress.json', 'qualify/*/result.json',
                    'benchmark/*/result.json', 'quality/*/result.json',
                    'selector/*_metrics.json', 'selector/*_protocol.json', 'selector/*_progress.json',
                    'selector/quality/*/result.json', 'controls/*_protocol.json', 'controls/*_progress.json',
                    'controls/benchmark/*/result.json', 'profile/*/result.json',
                    'profile/*/*.profile.json', 'report/speed.json', 'optimized-study/manifest.json',
                    'benchmark/*_r0/*.compute.json', 'selector/quality/*/*.compute.json'):
        paths.update(root.glob(pattern))
    records = {str(path.relative_to(root)): dict(sha256=sha(path), data=json.loads(path.read_text()))
               for path in sorted(paths)}
    logs = {str(path.relative_to(root)): dict(sha256=sha(path), text=path.read_text())
            for path in sorted((root/'logs').glob('followup_*check.log'))}
    payload = dict(schema_version=1, records=records, logs=logs,
        performance_eligible=False, source_root=str(root.resolve()),
        scope='Raw request receipts, protocols, quality metrics, compute ledgers, and sanitizer logs. Models, NPZ samples, and full Chrome traces remain outside Git; their hashes and local indices are retained.')
    archive = a.output/'measurements.json.gz'
    archive.write_bytes(gzip.compress(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode(), mtime=0))
    for name in ('speed.json',):
        (a.output/name).write_bytes((root/'report'/name).read_bytes())
    for method in ('b1.5_protected_n', 'b2_joint'):
        (a.output/(method+'_selector.json')).write_bytes((root/'selector'/(method+'_metrics.json')).read_bytes())
    write(a.output/'archive-index.json', dict(archive=archive.name, sha256=sha(archive), bytes=archive.stat().st_size,
        records=len(records), source_sha256=sha(__file__), raw_results_root=str(root.resolve())))
    print(json.dumps(dict(records=len(records), bytes=archive.stat().st_size)))


if __name__ == '__main__':
    main()
