"""Summarize paired independent-load diagnostic rounds without pooling scenes."""
import argparse
import json
from pathlib import Path
import statistics

from bundle import ROOT
from baselines import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--variant', default='norm')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    protocol = json.loads((a.study/'benchmark_protocol.json').read_text())
    progress = json.loads((a.study/'benchmark_progress.json').read_text())
    if not progress['complete']:
        raise ValueError('Finish all frozen benchmark jobs before reporting')
    records, rows = {}, []
    sources = None
    for method in protocol['methods']:
        timings = {}
        for variant in ('frozen', a.variant):
            for repeat in range(5):
                file = a.study/'benchmark'/f'{method}_{variant}_r{repeat}'/'result.json'
                d = json.loads(file.read_text())
                if not d['complete']:
                    raise ValueError('Incomplete round')
                if sources is None:
                    sources = d['identity']['source']
                elif d['identity']['source'] != sources:
                    raise ValueError('Runtime source changed between benchmark workers')
                records[str(file)] = sha(file)
                for case, detail in d['cases'].items():
                    if len(detail['rows']) != 3 or not detail['complete']:
                        raise ValueError('Each round must include three seeds')
                    if any(r['source_hashes_match'] is False for r in detail['rows']):
                        raise ValueError('A timed execution changed a qualified output')
                    timings[variant, repeat, case] = detail
        cases = sorted({key[2] for key in timings})
        for case in cases:
            for repeat in range(5):
                old_hashes = {r['seed']: r['sha256'] for r in timings['frozen', repeat, case]['rows']}
                new_hashes = {r['seed']: r['sha256'] for r in timings[a.variant, repeat, case]['rows']}
                if old_hashes != new_hashes:
                    raise ValueError('Paired benchmark outputs differ, including the third timing seed')
            per_variant = {}
            round_medians = {}
            for variant in ('frozen', a.variant):
                samples = [r for repeat in range(5) for r in timings[variant, repeat, case]['rows']]
                seconds = [r['wall_seconds'] for r in samples]
                per_variant[variant] = dict(median_seconds=statistics.median(seconds),
                    p90_seconds=float(np.quantile(seconds, .9)), fps=61/statistics.median(seconds),
                    allocated_peak_gib=max(r['peak_allocated_bytes'] for r in samples)/2**30,
                    reserved_peak_gib=max(r['peak_reserved_bytes'] for r in samples)/2**30)
                round_medians[variant] = np.array([statistics.median(r['wall_seconds'] for r in
                     timings[variant, repeat, case]['rows']) for repeat in range(5)])
            ratios = round_medians['frozen']/round_medians[a.variant]
            rng = np.random.default_rng(181919)
            boot = ratios[rng.integers(0, 5, (20000, 5))].mean(1)
            rows.append(dict(method=method, case=case,
                benchmark=timings['frozen', 0, case]['benchmark'], measurements=per_variant,
                speedup=per_variant['frozen']['median_seconds']/per_variant[a.variant]['median_seconds'],
                round_speedup_mean=float(ratios.mean()), round_bootstrap_ci95=np.quantile(boot, [.025,.975]).tolist(),
                paired_round_speedups=ratios.tolist()))
    result = dict(complete=True, rows=rows, records=records,
                  protocol_sha256=sha(a.study/'benchmark_protocol.json'), performance_eligible=False,
                  scope='Idle GPU, fixed implementation, five independently loaded randomized rounds x three seeds x two fixed inputs. No tuning-table qualification, no formal speed ranking or hardware ceiling.')
    a.output.mkdir(parents=True, exist_ok=True)
    write(a.output/'speed.json', result)
    lines = ['# Frozen-asset execution optimization', '', result['scope'], '',
             '| Budget | Input | Old ms | New ms | Speedup | Old FPS | New FPS | Allocated GiB old/new |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        old = row['measurements']['frozen']; new = row['measurements'][a.variant]
        lines.append(f"| {row['method']} | {row['benchmark']} | {old['median_seconds']*1000:.2f} | "
                     f"{new['median_seconds']*1000:.2f} | {row['speedup']:.4f} | {old['fps']:.2f} | {new['fps']:.2f} | "
                     f"{old['allocated_peak_gib']:.3f}/{new['allocated_peak_gib']:.3f} |")
    (a.output/'SPEED.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    main()
