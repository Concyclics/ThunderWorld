"""Supplement numerically unqualified Nano AWQ timings without changing frozen runs.

Explicit speed-only authorization is required. Results remain separate from the
randomly interleaved cohort and retain the original failed numerical receipt.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from build_baselines import sha, write
from compare_speed import compute_pids
from section54_protocol import statistics_rows, write_csv
from section54_resources import monitored_run
from standard_comparison import checked_run, command, load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('/home/chenhan/thunderworld'))
    parser.add_argument('--allow-numerically-unqualified-timing', action='store_true', required=True)
    parser.add_argument('--quick', action='store_true', help='One round, first scene, first two frozen seeds')
    args = parser.parse_args()
    output = args.campaign / 'nano-awq-speed-only'
    output.mkdir(exist_ok=True)
    lock = (output / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(complete=False, pid=os.getpid(), started=time.time(),
                 quality_evaluation=False, numerical_qualification_passed=False,
                 performance_eligible=False,
                 timing_only_authorization='User: speed testing need not gate on generation quality; retain numerical failure label',
                 measurement_class='Supplemental non-interleaved speed-only runs; not the randomized seven-arm cohort')

    def status(**values):
        state.update(values, updated=time.time())
        write(output / 'status.json', state)

    try:
        # Do not interrupt or compete with the frozen campaign, including profiles.
        status(stage='waiting_for_main_campaign')
        while not json.loads((args.campaign / 'campaign_status.json').read_text()).get('complete'):
            main_state = json.loads((args.campaign / 'campaign_status.json').read_text())
            if main_state.get('state') == 'stopped_requires_review':
                raise RuntimeError('Main campaign stopped; retain supplement pending review')
            time.sleep(30)
        a = SimpleNamespace(output=args.campaign / 'nano', root=args.root)
        m = load_manifest(a)
        if args.quick:
            m['rounds'] = m['rounds'][:1]
            m['cases'] = m['cases'][:1]
            m['seeds'] = m['seeds'][:2]
        write(output / 'timing_selection.json', dict(quick=args.quick, rounds=m['rounds'],
              cases=m['cases'], seeds=m['seeds'], source_manifest_sha256=sha(a.output / 'manifest.json'),
              limitation='Two requests are a speed spot-check, not a stable tail-latency estimate' if args.quick else state['measurement_class']))
        planned = len(m['rounds']) * len(m['cases']) * len(m['seeds'])
        status(planned_requests=planned)
        receipt = a.output / 'qualification/awq_marlin.unavailable.json'
        failure = json.loads(receipt.read_text())
        if 'Independent decoded-reference NMSE exceeds' not in failure.get('reason', ''):
            raise ValueError('Only decoded-reference numerical mismatch may be bypassed')
        if failure['manifest_sha256'] != m['fixed_manifest_sha256'] or sha(failure['probe']) != failure['probe_sha256']:
            raise ValueError('Failure receipt identity changed')
        write(output / 'numerical_failure.json', failure)
        rows = []
        for rnd in m['rounds']:
            out = output / 'runs' / f"round-{rnd['index']:02d}"
            result = out / 'result.json'
            if not result.exists():
                stable = 0
                while stable < 3:
                    import psutil
                    pids = compute_pids()
                    enough = psutil.virtual_memory().available >= 12 * 2**30
                    stable = stable + 1 if not pids and enough else 0
                    status(stage='waiting_for_resources', other_compute_pids=pids)
                    if stable < 3:
                        time.sleep(20)
                log = output / f"round-{rnd['index']:02d}.log"
                status(stage='timing', round=rnd['index'])
                with log.open('w') as stream:
                    run = monitored_run(command(a, m, 'awq_marlin', out), stream, log)
                if run.returncode:
                    raise RuntimeError(f'Worker failed or resource overlap; retained attempt: {log}')
            # Retain all worker finite-output, no-fallback, shape and source checks.
            data = checked_run(result, m, 'awq_marlin')
            rows.extend(dict(r, round=rnd['index'], numerical_qualification_passed=False,
                             performance_eligible=False, source=str(result), source_sha256=sha(result))
                        for r in data['rows'])
            write_csv(output / 'requests.csv', rows)
            summaries = statistics_rows(rows, 'awq_marlin', 'nano', m['fixed_manifest_sha256'])
            for row in summaries:
                row.update(numerical_qualification_passed=False, measurement_class=state['measurement_class'])
            write_csv(output / 'summary.csv', summaries)
            write(output / 'summary.json', dict(state, rows=summaries, successful_requests=len(rows),
                  complete=len(rows) == planned, numerical_failure= failure))
            status(completed_requests=len(rows))
        status(complete=True, stage='complete', finished=time.time())
    except Exception as error:
        status(stage='stopped_requires_review', error=repr(error))
        raise


if __name__ == '__main__':
    main()
