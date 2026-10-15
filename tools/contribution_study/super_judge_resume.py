"""Resume only the prepared Super judge, preserving the frozen scoring code."""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time

from baselines import sha, write
from super_quality_campaign import check_campaign
from super_task_judge import launch


def now():
    return datetime.now(timezone.utc).isoformat()


def validate(root):
    check_campaign(root)
    generation_hash = sha(root / 'super/final/manifest.json')
    exported = json.loads((root / 'super/export_progress.json').read_text())
    metrics = json.loads((root / 'metrics/progress.json').read_text())
    qualification = json.loads((root / 'metrics/qualification.json').read_text())
    if (not exported['complete'] or exported['completed'] != 30492
            or exported['identity']['manifest_sha256'] != generation_hash):
        raise ValueError('Complete matching video exports required')
    if (not metrics['complete'] or metrics['completed_pairs'] != 27104
            or metrics['expected_pairs'] != 27104
            or metrics['identity']['manifest_sha256'] != generation_hash
            or not qualification['complete']
            or qualification['identity'] != metrics['identity']):
        raise ValueError('Complete qualified pixel metrics required')
    judge = root / 'judge'
    sys.path.insert(0, str(judge / 'runtime_code'))
    from qwen_task_judge import verify_manifest
    manifest, digest = verify_manifest(judge)
    for package, expected in manifest['dependencies'].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError('Judge dependency changed: ' + package)
    prepared = json.loads((judge / 'prepared.json').read_text())
    if (not prepared['complete'] or prepared['manifest_sha256'] != digest
            or manifest['counts']['videos'] != 30492):
        raise ValueError('Complete matching judge preparation required')
    return dict(judge_manifest_sha256=digest, counts=manifest['counts'],
                exports_sha256=sha(root / 'super/export_progress.json'),
                pixel_metrics_sha256=sha(root / 'metrics/progress.json'))


def run(root, verified):
    judge = root / 'judge'
    with (root / 'campaign.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (judge / 'PAUSE_JUDGE.json').exists():
            raise RuntimeError('Judge has an explicit pause marker')
        history = judge / 'history' / ('prepared_resume_' + str(time.time_ns()))
        history.mkdir(parents=True)
        for name in ('progress.json', 'resource_launch.json', 'super/workflow.json'):
            target = history / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if (root / name).exists():
                shutil.copy2(root / name, target)
        # The frozen supervisor polls this file before the pool creates it on
        # its first launch. Initialize an explicitly unready state, never a
        # ready service or synthetic result. It still waits for the real pool
        # PID and the original per-replica health/ingestion qualification.
        with (judge / 'resource_session.lock').open('a+') as session_lock:
            fcntl.flock(session_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not (judge / 'servers.json').exists():
                write(judge / 'servers.json', dict(status='not_started', pid=None,
                    replicas=[], control_root=str(judge), initialized_utc=now()))
        state = json.loads((root / 'progress.json').read_text())
        state.pop('error', None)
        state.pop('paused', None)
        state.update(complete=False, phase='full_task_judge', updated_utc=now())
        receipt = dict(**verified, started_utc=now(), source_sha256=sha(__file__),
            source=str(Path(__file__).resolve()), history=str(history),
            frozen_scoring_code_unchanged=True, pixel_metrics_reused=True,
            performance_eligible=False)
        write(judge / 'prepared_resume.json', receipt)
        write(root / 'progress.json', state)
        write(root / 'super/workflow.json', dict(paused=False, generation_complete=True))
        try:
            complete = judge / 'report/complete.json'
            if not complete.exists():
                launch(judge)
            result = json.loads(complete.read_text())
            if (not result['complete'] or result['missing']
                    or result['manifest_sha256'] != verified['judge_manifest_sha256']):
                raise ValueError('Incomplete or changed judge report')
            state.setdefault('stages', {})['full_task_judge'] = sha(complete)
            state.update(complete=True, task_judge_complete=True,
                phase='generation_metrics_and_task_judge_complete',
                completed_utc=now(), updated_utc=now())
            receipt.update(complete=True, finished_utc=now(), report_sha256=sha(complete))
        except BaseException as error:
            state.update(complete=False, paused=True, error=repr(error), updated_utc=now())
            receipt.update(complete=False, error=repr(error), ended_utc=now())
            write(root / 'super/workflow.json', dict(paused=True,
                generation_complete=True, error=repr(error)))
            raise
        finally:
            write(root / 'progress.json', state)
            write(judge / 'prepared_resume.json', receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    verified = validate(root)
    print(json.dumps(verified), flush=True)
    if not args.check:
        run(root, verified)


if __name__ == '__main__':
    main()
