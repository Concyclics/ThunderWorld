"""Audit unique GPU kernels; never add CPU parent attribution to GPU time.

This classifier is deliberately specific to the frozen Edge I2V study. It fails
if its 28-layer / 35-step structural checks no longer hold. Profiling times and
unprofiled request timings are separate observations, never an additive model.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

from bundle import ROOT  # Establish the existing contribution-study imports.
from baselines import sha, write
from baseline_breakdown import METHODS


ATTENTION = {'streaming_attention', 'combine', 'pack_qk', 'pack_v',
             'residual_qk', 'residual_v', 'select_tiles_head', 'support_union'}
LINEAR_PACK = {'pack_activation', 'base_summary', 'residual_only', 'choose',
               'scaled_amax', 'finish_amax', 'pointwise_amax', 'calibrated_relu_amax'}


def bf16_targets(events):
    """Locate exactly the 6 generation linears per layer, excluding AR/codecs.

    In each frozen DM graph, four helper GEMMs precede 28*(Q,K,V,O,up,down),
    followed by the output head. The two AR graphs have a different signature.
    Assert both the complete sequence and kernel tile types before classifying.
    """
    graphs = sorted((e for e in events if e.get('cat') == 'cuda_runtime' and
                     e.get('ph') == 'X' and e['name'] == 'cudaGraphLaunch'), key=lambda e: e['ts'])
    assert len(graphs) == 72, 'Expected two AR prefills and 70 DM graphs'
    kernels = defaultdict(list)
    for i, e in enumerate(events):
        if e.get('cat') == 'kernel' and e.get('ph') == 'X':
            kernels[e['args']['correlation']].append((i, e))
    targets, graphs_dm = {}, 0
    pattern = ('128x128_32x4', '128x64_64x4', '128x64_64x4',
               '128x128_32x4', '256x128_32x3', '256x128_32x3')
    modules = ('q', 'k', 'v', 'o', 'up', 'down')
    for index, graph in enumerate(graphs):
        sequence = sorted(kernels[graph['args']['correlation']], key=lambda p: p[1]['ts'])
        mm = [(i, e) for i, e in sequence if 'gemm' in e['name'].lower()]
        if index in (0, 2):
            assert len(mm) == 169, 'AR signature changed'
            continue
        assert len(mm) == 173, 'DM signature changed'
        assert 'simt_sgemm' in mm[3][1]['name'] and '64x64_32x10' in mm[-1][1]['name']
        graphs_dm += 1
        for offset, (i, e) in enumerate(mm[4:-1]):
            assert 'bf16_s16816gemm' in e['name'] and pattern[offset % 6] in e['name']
            targets[i] = modules[offset % 6]
    assert graphs_dm == 70 and len(targets) == 11760
    return targets


def category(method, index, event, targets):
    name = event['name']
    if index in targets or name == 'gemm' or 'void marlin::Marlin<' in name:
        return 'generation_gemm'
    if name.startswith('_ZN7cutlass13device_kernel'):
        return 'generation_gemm'
    if name in ATTENTION or 'pytorch_flash::flash_' in name:
        return 'attention_including_named_packing'
    if name in LINEAR_PACK or 'reorder_x_kernel<' in name or 'fused_sparse_residual_x_kernel<' in name or 'dynamic_per_token_scaled_fp8_quant_kernel' in name:
        return 'dedicated_linear_preparation'
    if 'cudnn' in name.lower():
        return 'cudnn_convolution_and_layout'
    if 'gemm' in name.lower():
        return 'other_gemm'
    return 'other_pointwise_reduction_copy'


def profile(path, method):
    data = json.loads(path.read_text())
    events = data['traceEvents']
    targets = bf16_targets(events) if method == 'bf16' else {}
    groups, by_name, grids, module_ms = {}, {}, {}, defaultdict(float)
    total, count = 0., 0
    for i, e in enumerate(events):
        if e.get('cat') != 'kernel' or e.get('ph') != 'X':
            continue
        ms, name = e['dur'] / 1000, e['name']
        group = category(method, i, e, targets)
        row = groups.setdefault(group, dict(calls=0, ms=0.))
        row['calls'] += 1; row['ms'] += ms
        row = by_name.setdefault(name, dict(category=group, calls=0, ms=0.))
        # BF16 has identical kernel names in target and non-target GEMMs.
        if row['category'] != group:
            row['category'] = 'mixed_generation_and_other_gemm'
        row['calls'] += 1; row['ms'] += ms
        if i in targets:
            module_ms[targets[i]] += ms
        if group == 'generation_gemm':
            a = e['args']
            key = json.dumps([name, a.get('grid'), a.get('block'), a.get('registers per thread'), a.get('shared memory')])
            row = grids.setdefault(key, dict(name=name, grid=a.get('grid'), block=a.get('block'),
                registers=a.get('registers per thread'), shared_bytes=a.get('shared memory'), calls=0, ms=0.))
            row['calls'] += 1; row['ms'] += ms
        total += ms; count += 1
    assert math.isclose(sum(r['ms'] for r in groups.values()), total, abs_tol=1e-6)
    expected = 23520 if method == 'sharq_perf' else 11760
    assert groups['generation_gemm']['calls'] == expected, (method, groups['generation_gemm'])
    runtime = Counter(e['name'] for e in events if e.get('cat') in ('cuda_runtime', 'cuda_driver') and e.get('ph') == 'X')
    overhead = Counter(e['name'] for e in events if e.get('cat') == 'overhead')
    return dict(file=str(path), sha256=sha(path), kernel_count=count, kernel_sum_ms=total,
                groups=groups, bf16_module_ms=dict(module_ms), runtime_counts=dict(runtime),
                profiler_overhead_counts=dict(overhead),
                kernels=[dict(name=k, **v) for k, v in sorted(by_name.items(), key=lambda p: -p[1]['ms'])],
                gemm_launches=list(grids.values()))


def percentile(values, q):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def timings(roots):
    groups = defaultdict(list)
    for cohort, root in roots.items():
        for path in sorted((root/'benchmark').glob('*/result.json')):
            d = json.loads(path.read_text())
            assert d['complete'] and not d['performance_eligible']
            identity = d['identity']; method, variant = identity['method'], identity['variant']
            for case, result in d['cases'].items():
                for row in result['rows']:
                    assert row['steps'] == 35 and row['debug'] == 'off' and not row['profiled']
                    assert row['timing_scope'] == 'complete_request_v2_including_host_setup'
                    # Only seeded reference outputs have an oracle hash.
                    assert row.get('source_hashes_match') is not False
                    groups[(cohort, method, variant, case)].append(dict(file=str(path), file_sha256=sha(path), **row))
    result = []
    for (cohort, method, variant, case), rows in groups.items():
        assert len(rows) == 15, (cohort, method, variant, case, len(rows))
        times = [r['wall_seconds'] for r in rows]
        result.append(dict(cohort=cohort, method=method, variant=variant, case=case, n=len(rows),
            median_seconds=statistics.median(times), p90_seconds=percentile(times, .9),
            fps_at_median=61/statistics.median(times), frames_per_request=61,
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
            peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows), rows=rows))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--previous-runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plot', type=Path, help='Optional static figure prefix (PNG and SVG)')
    a = p.parse_args()
    assert json.loads((a.study/'progress.json').read_text())['complete']
    output = dict(schema='fp4-breakdown-v1', performance_eligible=False, profiles={}, timings=[],
        caveats=[
            'Unique cat=kernel, ph=X events only. Kernel sums are diagnostic work, not a wall-time partition.',
            'Dedicated preparation is a lower bound for ARC/SharQ: generic div/amax/cast remain in other operations. AWQ dequantization is inside Marlin, not separately timed.',
            'Attention includes named packing, residual selection, streaming softmax and combine; generic attention amax/copies can remain in other operations.',
            'Do not subtract profiled kernel sums from unprofiled wall time. CPU events and profiler overhead are not additive GPU components.',
            'Three timing cohorts use the same runtime/inputs/protocol but were not jointly interleaved; no formal cross-method ranking.',
            'Profiles: one RBench scene, two seeds. Timings: one scene per dataset, five independent loads by three seeds. Not a dataset-wide speed estimate.'])
    for method in METHODS:
        files = sorted((a.study/'profile'/method).glob('*.trace.json'))
        assert len(files) == 2
        rows = [profile(f, method) for f in files]
        categories = set().union(*(r['groups'] for r in rows))
        means = {k: statistics.mean(r['groups'].get(k, {'ms': 0})['ms'] for r in rows) for k in sorted(categories)}
        output['profiles'][method] = dict(mean_kernel_sum_ms=statistics.mean(r['kernel_sum_ms'] for r in rows),
            mean_ms=means, rows=rows)
        print(method, {k: round(v, 2) for k, v in means.items()}, flush=True)
    base = output['profiles']['bf16']
    fraction = base['mean_ms']['generation_gemm']/base['mean_kernel_sum_ms']
    output['amdahl_illustration'] = dict(eligible_kernel_fraction=fraction,
        ideal_gemm_speedup=4, kernel_work_only_speedup=1/(1-fraction+fraction/4),
        scope='Hypothetical unchanged non-target kernel work and zero added packing/repair overhead; NOT an E2E or hardware ceiling.')
    output['timings'] = timings({'external_baselines': a.study/'timing',
        'repair_ablation': a.previous_runtime, 'bf16_fp8_controls': a.previous_runtime/'controls'})
    csv_path = a.study/'counters.csv'
    if csv_path.exists():
        inputs = json.loads((a.study/'counter_inputs.json').read_text())
        raw = list(csv.DictReader(csv_path.open()))
        metrics = ['gpu__time_duration.sum', 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
                   'sm__warps_active.avg.pct_of_peak_sustained_active', 'dram__throughput.avg.pct_of_peak_sustained_elapsed',
                   'launch__registers_per_thread', 'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum',
                   'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']
        rows = [r for r in raw if r['ID']]
        assert inputs['complete'] and len(rows) == len(inputs['rows']) == 4
        output['representative_counters'] = dict(scope=inputs['scope'], cache_control='none', clock_control='none',
            metrics_units={k: raw[0][k] for k in metrics}, input_receipt_sha256=sha(a.study/'counter_inputs.json'),
            csv_sha256=sha(csv_path), rows=[dict(input=panel, grid=r['Grid Size'],
                metrics={k: float(r[k]) for k in metrics}) for panel, r in zip(inputs['rows'], rows)])
    write(a.output, output)
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        labels = ['BF16', 'FP8', 'Plain FP4', 'AWQ / Marlin', 'ARC 50%', 'ARC 100%',
                  'SharQ', 'Ours 1.5x', 'Ours 2x', 'Ours 2.5x']
        categories = [('generation_gemm', 'Generation GEMM'),
                      ('dedicated_linear_preparation', 'Named packing / selection'),
                      ('attention_including_named_packing', 'Attention + named packing'),
                      ('cudnn_convolution_and_layout', 'cuDNN convolution / layout'),
                      ('other_gemm', 'Other GEMM'),
                      ('other_pointwise_reduction_copy', 'Other vector / copy work')]
        fig, ax = plt.subplots(figsize=(11, 6.5), layout='constrained')
        left = [0.] * len(METHODS)
        colors = ['#245D91', '#39A6A3', '#9276B5', '#D09C46', '#8A9EAE', '#C9CDD1']
        for (key, label), color in zip(categories, colors):
            widths = [output['profiles'][m]['mean_ms'].get(key, 0.) for m in METHODS]
            ax.barh(labels, widths, left=left, label=label, color=color)
            left = [x+y for x, y in zip(left, widths)]
        ax.invert_yaxis()
        ax.set_xlabel('Sum of GPU kernel durations (ms), mean of two profiled seeds')
        ax.set_title('Frozen Edge I2V on PRO 6000: where GPU work is spent\n'
                     'Diagnostic profiles, not a partition of unprofiled request latency', fontsize=12)
        ax.legend(loc='lower center', bbox_to_anchor=(.5, -0.29), ncol=3, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        a.plot.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ('.png', '.svg'):
            fig.savefig(a.plot.with_suffix(suffix), dpi=180)
        svg = a.plot.with_suffix('.svg')
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
        plt.close(fig)


if __name__ == '__main__':
    main()
