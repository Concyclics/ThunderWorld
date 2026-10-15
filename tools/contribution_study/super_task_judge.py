"""Extend the completed frozen nine-method Qwen protocol to Super, without score reuse."""
import argparse
import ast
import copy
import difflib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from baselines import sha, write

DEFAULT_PREVIOUS = Path('/home/yufeng/thunderworld/qwen_task_judge_all_methods_20260924')
FILES = ('qwen_task_protocol.py', 'qwen_task_judge.py', 'qwen_task_report.py',
         'qwen_task_resume.py', 'qwen_judge_pool.py', 'qwen_resource_session.py')


def patch_protocol(source):
    anchor = "MODELS = ('edge', 'nano')"
    if source.count(anchor) != 1:
        raise ValueError('Expected frozen two-model protocol')
    revised = source.replace(anchor, "MODELS = ('super',)")
    ast.parse(revised)
    return revised


def patch_report(source):
    revised = source
    for old, new in [('family144', 'family72'), ('family-144', 'family-72'),
                     ('.05 / (2 * 144)', '.05 / (2 * 72)'),
                     ('for four model/benchmark groups.', 'for two Super/benchmark groups.')]:
        if old not in revised:
            raise ValueError('Missing statistical patch anchor: ' + old)
        revised = revised.replace(old, new)
    revised = revised.replace(
        'The historical four-method repeated-judgement subset (not including fp4_base or the four additional baselines) is summarized in `judge_repeatability.csv`. ',
        'Repeated judgements have not been measured independently for Super; Edge/Nano repeatability is not reused. ')
    ast.parse(revised)
    return revised


def freeze(root, previous):
    old = json.loads((previous / 'manifest.json').read_text())
    done = json.loads((previous / 'report/complete.json').read_text())
    if not done['complete'] or done['manifest_sha256'] != sha(previous / 'manifest.json'):
        raise ValueError('Completed nine-method source judge required')
    runtime = root / 'runtime_code'
    runtime.mkdir(parents=True, exist_ok=True)
    frozen = {Path(p).name: (Path(p), digest) for p, digest in old['code_sha256'].items()}
    changes, receipts = [], {}
    for name in (*FILES, Path(__file__).name):
        if name == Path(__file__).name:
            path, digest = Path(__file__), sha(__file__)
        else:
            path, digest = frozen[name]
        if sha(path) != digest:
            raise ValueError('Changed historical judge source: ' + str(path))
        source = path.read_text()
        revised = patch_protocol(source) if name == 'qwen_task_protocol.py' else (
            patch_report(source) if name == 'qwen_task_report.py' else source)
        target = runtime / name
        if target.exists() and target.read_text() != revised:
            raise ValueError('Changed existing Super judge snapshot')
        target.write_text(revised)
        receipts[name] = dict(source=str(path), source_sha256=digest, runtime_sha256=sha(target))
        changes.extend(difflib.unified_diff(source.splitlines(True), revised.splitlines(True),
            fromfile=str(path), tofile=str(target)))
    write(root / 'runtime_source_receipt.json', receipts)
    (root / 'runtime_patch.diff').write_text(''.join(changes))
    return runtime


def prepare(campaign, root, previous):
    runtime = root / 'runtime_code'
    if Path(__file__).resolve().parent != runtime.resolve():
        runtime = freeze(root, previous)
        subprocess.run([sys.executable, str(runtime / Path(__file__).name), 'prepare',
            '--campaign', str(campaign), '--root', str(root), '--previous', str(previous)], check=True)
        return
    from qwen_task_protocol import build_inputs, protocol, now
    from qwen_task_judge import verify_manifest
    if protocol()['models'] != ['super'] or len(protocol()['methods']) != 9:
        raise ValueError('Wrong isolated Super judge protocol')
    old = json.loads((previous / 'manifest.json').read_text())
    for package, expected in old['dependencies'].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError('Judge dependency changed: ' + package)
    for name in ('sources', 'deps', 'chains'):
        target = root / name
        source = (previous / name).resolve()
        if not target.exists():
            target.symlink_to(source, target_is_directory=True)
        if target.resolve() != source:
            raise ValueError('Changed source or question-chain link')
    full = json.loads((campaign / 'super/final/manifest.json').read_text())
    exported = json.loads((campaign / 'super/export_progress.json').read_text())
    if not exported['complete'] or exported['completed'] != 30492:
        raise ValueError('All 30,492 Super videos must be exported before judging')
    metric_root = root / 'source_inputs'
    definitions = {}
    for entry in full['cases']:
        records = {}
        for method in protocol()['methods']:
            records[method] = {}
            for seed in entry['seeds']:
                metadata = campaign / 'super/videos' / entry['benchmark'] / method / entry['case']['id'] / f'seed{seed}.json'
                sample = json.loads(metadata.read_text())
                receipt_path = Path(sample['generation_receipt'])
                receipt = json.loads(receipt_path.read_text())
                if not receipt['complete'] or receipt['identity']['manifest_sha256'] != sha(campaign / 'super/final/manifest.json'):
                    raise ValueError('Unmatched Super generation receipt')
                adapter = receipt.get('adapter')
                if method in ('arcquant', 'arcquant100'):
                    if (adapter['execution_type'] != 'official_native_cosmos_adapter' or
                        adapter['channel_fraction'] != (.5 if method == 'arcquant' else 1.)):
                        raise ValueError('Unqualified ARC definition')
                if method == 'sharq_perf' and (adapter['execution_type'] != 'official_native_adapter' or not adapter['extra_fusion']):
                    raise ValueError('Unqualified SharQ definition')
                if method == 'awq_reference' and adapter['execution_type'] != 'offline_decoded_quality_reference':
                    raise ValueError('Changed AWQ definition')
                records[method][str(seed)] = dict(metadata=str(metadata), metadata_sha256=sha(metadata),
                    path=sample['raw_outputs'], npz_sha256=sample['provenance']['npz_sha256'],
                    sample_id=sample['sample_id'], video_sha256=sample['request']['sha256']['video'])
                definitions.setdefault(method, dict(spec=sample['method_spec'], adapter=adapter))
        write(metric_root / 'tasks/super' / (entry['case']['id']+'.json'), dict(model='super',
            case_id=entry['case']['id'], case=entry['case'], source_id=entry['source_id'],
            benchmark=entry['benchmark'], category=entry['category'], seeds=entry['seeds'],
            image_sha256=entry['image_sha256'], prior_scene_roles=entry.get('prior_scene_roles', {}), records=records))
    write(metric_root / 'manifest.json', dict(complete=True, model='super', videos=30492,
        generation_manifest_sha256=sha(campaign / 'super/final/manifest.json'),
        scope='Judge source index; numerical pixel metrics live in ../metrics'))
    inputs, _ = build_inputs(root, metric_root, campaign.parent / 'cosmos')
    scenes = json.loads((previous / 'scenes.json').read_text())
    for cid, value in inputs['scenes'].items():
        if value != {k: v for k, v in scenes[cid].items() if k != 'queries'}:
            raise ValueError('Scene/seed/annotation differs from frozen Edge/Nano judge')
    for cid, expected in old['chain_sha256'].items():
        if sha(root / 'chains' / (cid+'.json')) != expected:
            raise ValueError('Changed frozen question chain')
    jobs = sorted(inputs['jobs'], key=lambda x: x['id'])
    write(root / 'scenes.json', scenes)
    (root / 'jobs.jsonl').write_text(''.join(json.dumps(j, ensure_ascii=False)+'\n' for j in jobs))
    manifest = copy.deepcopy(old)
    manifest.update(created_utc=now(), protocol=protocol(),
        inputs=dict(scenes_sha256=sha(root/'scenes.json'), jobs_sha256=sha(root/'jobs.jsonl'),
            upstream=inputs['source_hashes'], metric_manifest_sha256=inputs['metric_manifest_sha256']),
        code_sha256={str(p): sha(p) for p in sorted(runtime.glob('*.py'))},
        counts=dict(videos=len(jobs), scenes=len(scenes),
            judge_requests=sum(len(scenes[j['case_id']]['queries']) for j in jobs),
            binary_requests=sum(q['kind']=='binary' for j in jobs for q in scenes[j['case_id']]['queries']),
            empty_annotation_scenes=len(inputs['empty_official_vqa'])),
        server_configuration=dict(template=old['server_configuration'], gpus=[0,2],
            cpu_memory_limit_gib=64, actual_service_receipt=str(root/'servers.json')),
        extension=dict(source_root=str(previous), source_manifest_sha256=sha(previous/'manifest.json'),
            purpose='New Super outputs, identical frozen task questions; no scores imported', reused_scores=0))
    if (root / 'manifest.json').exists():
        saved = json.loads((root/'manifest.json').read_text())
        manifest['created_utc'] = saved['created_utc']
        if saved != manifest:
            raise ValueError('Refusing to overwrite frozen Super judge manifest')
    else:
        write(root / 'manifest.json', manifest)
    plan = json.loads((previous/'report_plan.json').read_text())
    plan.update(frozen_utc=now(), report_code_sha256=sha(runtime/'qwen_task_report.py'), family_size=72,
        primary_aggregation='category macro, Super x 2 benchmarks x 36 method pairs',
        reused_scores=0)
    write(root/'report_plan.json', plan)
    write(root/'baseline_definition_audit.json', dict(complete=True, methods=definitions, videos=len(jobs)))
    verify_manifest(root)
    write(root/'prepared.json', dict(complete=True, counts=manifest['counts'], manifest_sha256=sha(root/'manifest.json')))


def launch(root):
    model = root.parent.parent / 'models/Qwen3-VL-8B-Instruct-FP8-9cdc631'
    # root is <campaign>/judge; workspace is its grandparent.
    unit = 'thunderworld-super-qwen-' + str(time.time_ns())
    command = ['systemd-run', '--user', '--wait', '--unit='+unit,
        '--property=MemoryHigh=48G', '--property=MemoryMax=64G', '--property=MemorySwapMax=0',
        '--property=TasksMax=1024', '--property=LimitCORE=0',
        '--property=StandardOutput=append:'+str(root/'resource_session.log'),
        '--property=StandardError=append:'+str(root/'resource_session.log'),
        '/usr/bin/env', 'VLLM_PLUGINS=', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1',
        'PYTHONPATH='+str(root/'deps')+os.pathsep+os.environ.get('PYTHONPATH',''),
        sys.executable, str(root/'runtime_code/qwen_resource_session.py'), '--root', str(root),
        '--model', str(model), '--gpus', '0', '2', '--video-workers', '12']
    write(root/'resource_launch.json', dict(unit=unit, command=command, memory_max_gib=64))
    subprocess.run(command, check=True)
    complete = json.loads((root/'report/complete.json').read_text())
    if not complete['complete'] or complete['missing']:
        raise RuntimeError('Super judge ended with incomplete coverage')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('freeze', 'prepare', 'run'))
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--root', type=Path)
    p.add_argument('--previous', type=Path, default=DEFAULT_PREVIOUS)
    a = p.parse_args()
    root = (a.root or a.campaign/'judge').resolve()
    if a.operation == 'freeze':
        freeze(root, a.previous)
    else:
        prepare(a.campaign.resolve(), root, a.previous)
        if a.operation == 'run':
            launch(root)


if __name__ == '__main__':
    main()
