"""Complete the nine-method judge comparison without changing historical runs.

Only AWQ NVFP4, ARCQuant 50/100% and SharQ need new answers. The previous
five-method runtime supplies the exact judge, questions, parser and serving
configuration. A fresh snapshot expands the method list and comparison report.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import copy
import difflib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import shutil
import sys

from qwen_task_extend_w4a4 import ProvenanceValidator, frozen_source, read, sha


PREVIOUS_METHODS = ('bf16', 'fp8', 'fp4_base', 'b1.5_protected_n', 'b2_joint')
ADDED_METHODS = ('awq_reference', 'arcquant', 'arcquant100', 'sharq_perf')
METHODS = (*PREVIOUS_METHODS[:3], *ADDED_METHODS, *PREVIOUS_METHODS[3:])
PREVIOUS_PAIRS = [
    ('b2_joint', 'bf16'), ('b2_joint', 'fp8'), ('fp8', 'bf16'),
    ('b1.5_protected_n', 'bf16'), ('b1.5_protected_n', 'fp8'),
    ('b1.5_protected_n', 'b2_joint'), ('fp4_base', 'bf16'), ('fp4_base', 'fp8'),
    ('b1.5_protected_n', 'fp4_base'), ('b2_joint', 'fp4_base'),
]
PAIRS = list(PREVIOUS_PAIRS)
for _reference, _candidate in itertools.combinations(METHODS, 2):
    if frozenset((_candidate, _reference)) not in {frozenset(p) for p in PAIRS}:
        PAIRS.append((_candidate, _reference))
FAMILY_SIZE = len(PAIRS) * 4
BOOTSTRAP_TRIALS = 100000
RUNTIME_FILES = (
    'qwen_task_protocol.py', 'qwen_task_judge.py', 'qwen_task_report.py',
    'qwen_task_resume.py', 'qwen_judge_pool.py', 'qwen_resource_session.py',
    'qwen_task_extend_w4a4.py',
)


def patch_protocol(source):
    lines = source.splitlines(keepends=True)
    matches = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
               and any(getattr(t, 'id', None) == 'METHODS' for t in n.targets)]
    if len(matches) != 1 or tuple(ast.literal_eval(matches[0].value)) != PREVIOUS_METHODS:
        raise ValueError('Expected the frozen five-method protocol')
    node = matches[0]
    lines[node.lineno - 1:node.end_lineno] = ['METHODS = ' + repr(METHODS) + '\n']
    return ''.join(lines)


def patch_report(source):
    pairs = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Tuple)
             and [getattr(t, 'id', None) for t in n.target.elts] == ['candidate', 'reference']]
    if len(pairs) != 1 or ast.literal_eval(pairs[0].iter) != PREVIOUS_PAIRS:
        raise ValueError('Expected the frozen ten-pair comparison plan')
    node = pairs[0].iter
    lines = source.splitlines(keepends=True)
    prefix = lines[node.lineno - 1][:node.col_offset]
    suffix = lines[node.end_lineno - 1][node.end_col_offset:]
    lines[node.lineno - 1:node.end_lineno] = [prefix + repr(PAIRS) + suffix]
    result = ''.join(lines)
    replacements = {
        'family40': f'family{FAMILY_SIZE}',
        'family-40': f'family-{FAMILY_SIZE}',
        '.05 / (2 * 40)': f'.05 / (2 * {FAMILY_SIZE})',
        'trials=20000': f'trials={BOOTSTRAP_TRIALS}',
        'Ten pairs among five methods': 'Thirty-six pairs among nine methods',
        'BF16 + FP8 + fp4_base + b1.5_protected_n + b2_joint':
            'BF16 + FP8 + fp4_base + AWQ NVFP4 + ARCQuant 50% + ARCQuant 100% + SharQ + b1.5_protected_n + b2_joint',
        'not including fp4_base': 'not including fp4_base or the four additional baselines',
    }
    for before, after in replacements.items():
        if before not in result:
            raise ValueError('Report patch anchor missing: ' + before)
        result = result.replace(before, after)
    anchor = "'These are not official leaderboard scores.', '',"
    if anchor not in result:
        raise ValueError('Report scope anchor missing')
    result = result.replace(anchor, anchor + "\n        " + repr(
        'AWQ is the official ModelOpt NVFP4 AWQ-lite decoded W4A16 quality reference. '
        'ARCQuant 50%/100% and SharQ use official native Cosmos adapters with BF16 attention; '
        'our repaired methods and ordinary W4A4 use FP4 attention. These are end-to-end '
        'method comparisons, not matched attention or compute-budget ablations. '
        'No generation speed or packed-weight storage conclusions come from this judge run.') + ", '',")
    ast.parse(result)
    return result


def freeze_runtime(args):
    previous = read(args.from_root / 'manifest.json')
    done = read(args.from_root / 'report/complete.json')
    if (not done['complete'] or done['manifest_sha256'] != sha(args.from_root / 'manifest.json')
            or tuple(previous['protocol']['methods']) != PREVIOUS_METHODS):
        raise ValueError('A complete frozen five-method run is required')
    for name, expected in read(args.from_root / 'report/file_hashes.json').items():
        if sha(args.from_root / 'report' / name) != expected:
            raise ValueError('Historical report changed: ' + name)
    locked = {Path(path).name: frozen_source(args.from_root, path, expected)
              for path, expected in previous['code_sha256'].items()}
    if sha(locked['qwen_task_report.py']) != read(args.from_root / 'report_plan.json')['report_code_sha256']:
        raise ValueError('Historical statistical implementation changed')
    runtime = args.root / 'runtime_code'
    runtime.mkdir(parents=True, exist_ok=True)
    changes, receipts = [], {}
    for name in (*RUNTIME_FILES, Path(__file__).name):
        src = locked[name] if name in locked else Path(__file__)
        source = src.read_text()
        revised = patch_protocol(source) if name == 'qwen_task_protocol.py' else (
            patch_report(source) if name == 'qwen_task_report.py' else source)
        target = runtime / name
        if target.exists() and target.read_text() != revised:
            raise ValueError('Refusing to change an existing snapshot: ' + name)
        target.write_text(revised)
        receipts[name] = dict(source=str(src), source_sha256=sha(src), runtime_sha256=sha(target))
        changes.extend(difflib.unified_diff(source.splitlines(True), revised.splitlines(True),
                                          fromfile=str(src), tofile=str(target)))
    (args.root / 'runtime_patch.diff').write_text(''.join(changes))
    (args.root / 'runtime_source_receipt.json').write_text(json.dumps(receipts, indent=2) + '\n')
    return runtime


def check_method_definition(method, meta, receipt=None):
    if not meta['complete'] or meta['method'] != method or meta['request']['calls'] != 70:
        raise ValueError('Incomplete or mismatched baseline generation')
    spec = meta['method_spec']
    if spec['config']['attention_backend'] != 'bf16':
        raise ValueError('Unexpected external baseline attention definition')
    if method == 'awq_reference':
        if (spec['baseline_adapter']['method'] != 'awq'
                or 'Official ModelOpt NVFP4 AWQ-lite' not in spec['scope']
                or spec['performance_eligible'] or spec['storage_comparison_eligible']):
            raise ValueError('Expected the decoded ModelOpt NVFP4 AWQ quality reference')
    elif 'actual_generation_precision' in spec:
        expected = dict(arcquant='arcquant', arcquant100='ARCQuant-100%',
                        sharq_perf='SharQ-native-existing-epilogue-fusion')
        if (spec['actual_generation_precision'] != expected[method]
                or spec['execution_type'] != 'official_native_cosmos_adapter'):
            raise ValueError('Incorrect native baseline adapter')
        if method == 'arcquant' and not any('50%' in x for x in spec['limitations']):
            raise ValueError('ARCQuant 50% policy is not recorded')
    elif receipt is None:
        raise ValueError('Missing native adapter provenance in legacy export')
    if receipt is not None:
        if (not receipt['complete'] or receipt['identity']['method'] != method
                or meta['request'] not in receipt['rows']):
            raise ValueError('Generation receipt differs from exported request')
        adapter = receipt['adapter']
        if method.startswith('arcquant'):
            fraction = .5 if method == 'arcquant' else 1.0
            if (adapter['method'] != 'ARCQuant' or adapter['channel_fraction'] != fraction
                    or adapter['execution_type'] != 'official_native_cosmos_adapter'):
                raise ValueError('Incorrect ARCQuant repair fraction in generation receipt')
        elif method == 'sharq_perf':
            if (adapter['method'] != 'SHARQ' or not adapter['extra_fusion']
                    or adapter['execution_type'] != 'official_native_adapter'):
                raise ValueError('Incorrect SharQ generation adapter')
        elif (adapter['method'] != 'NVFP4-AWQ-W4A16'
                or adapter['execution_type'] != 'offline_decoded_quality_reference'
                or adapter['performance_eligible']):
            raise ValueError('Incorrect decoded AWQ generation adapter')
    return dict(exported_method_spec=spec, execution_adapter=receipt['adapter'] if receipt else None)


def audit_definitions(root, jobs):
    from qwen_task_protocol import digest, now, write
    task_cache, receipt_cache, counts, definitions = {}, {}, defaultdict(Counter), {}
    by_group = Counter()
    for job in jobs:
        if job['method'] not in ADDED_METHODS:
            continue
        task = task_cache.setdefault(job['source_task'], None)
        if task is None:
            task = task_cache[job['source_task']] = read(job['source_task'])
        rec = task['records'][job['method']][str(job['seed'])]
        meta = read(rec['metadata'])
        if rec['receipt'] not in receipt_cache:
            receipt_cache[rec['receipt']] = (sha(rec['receipt']), read(rec['receipt']))
        receipt_hash, receipt = receipt_cache[rec['receipt']]
        if receipt_hash != rec['receipt_sha256']:
            raise ValueError('Generation receipt changed since the pixel-metric manifest')
        spec = check_method_definition(job['method'], meta, receipt)
        if not Path(job['source_video']).is_file():
            raise ValueError('Missing baseline video: ' + job['source_video'])
        key = job['model'] + '/' + job['method']
        fingerprint = digest(spec)
        counts[key][fingerprint] += 1
        definitions[fingerprint] = spec
        by_group[key + '/' + job['benchmark']] += 1
    write(root / 'baseline_definition_audit.json', dict(complete=True, utc=now(),
        videos=sum(by_group.values()), by_group=dict(by_group),
        configuration_counts={k: dict(v) for k, v in counts.items()}, definitions=definitions,
        generation_receipt_sha256={k: v[0] for k, v in receipt_cache.items()},
        source_video_hashes_verified_by_judge_before_each_new_video=True,
        note='External baselines retain BF16 attention and their original generation execution modes.'))


def prepare_frozen(args):
    from qwen_task_protocol import build_inputs, now, protocol, write
    from qwen_task_judge import load_jobs, verify_manifest
    from qwen_task_resume import validate_saved
    root, previous = args.root, args.from_root
    if tuple(protocol()['methods']) != METHODS:
        raise ValueError('Use the isolated nine-method runtime')
    for name in ('sources', 'deps', 'chains'):
        target = root / name
        if not target.exists() and not target.is_symlink():
            target.symlink_to((previous / name).resolve(), target_is_directory=True)
    # GPU cache directories belong to this run; never redirect GPU3 into a
    # different device's cache. The two replicas compile/cache independently.
    old = read(previous / 'manifest.json')
    for name, expected in old['dependencies'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('Judge dependency changed: ' + name)
    scenes = read(previous / 'scenes.json')
    validator = ProvenanceValidator(protocol(), scenes)
    old_hash = validator.root(previous)
    inputs, _ = build_inputs(root, args.metrics_root, args.cosmos_root)
    for cid, scene in inputs['scenes'].items():
        if scene != {k: v for k, v in scenes[cid].items() if k != 'queries'}:
            raise ValueError('Source scene differs from frozen questions')
    for cid, expected in old['chain_sha256'].items():
        if sha(root / 'chains' / (cid + '.json')) != expected:
            raise ValueError('Frozen reasoning question chain changed')
    jobs = sorted(inputs['jobs'], key=lambda j: j['id'])
    new_jobs, old_jobs = {j['id']: j for j in jobs}, load_jobs(previous)
    if any(new_jobs.get(j['id']) != j for j in old_jobs):
        raise ValueError('Previously scored generation job changed')
    audit_definitions(root, jobs)
    if not (root / 'manifest.json').exists():
        shutil.copy2(previous / 'scenes.json', root / 'scenes.json')
        with (root / 'jobs.jsonl').open('w') as stream:
            for job in jobs:
                stream.write(json.dumps(job, ensure_ascii=False) + '\n')
        runtime = root / 'runtime_code'
        manifest = copy.deepcopy(old)
        manifest.update(created_utc=now(), protocol=protocol(),
            inputs=dict(scenes_sha256=sha(root / 'scenes.json'), jobs_sha256=sha(root / 'jobs.jsonl'),
                upstream=inputs['source_hashes'], metric_manifest_sha256=inputs['metric_manifest_sha256']),
            code_sha256={str(p): sha(p) for p in sorted(runtime.glob('*.py'))},
            counts=dict(videos=len(jobs), scenes=len(scenes),
                judge_requests=sum(len(scenes[j['case_id']]['queries']) for j in jobs),
                binary_requests=sum(q['kind'] == 'binary' for j in jobs for q in scenes[j['case_id']]['queries']),
                empty_annotation_scenes=len(inputs['empty_official_vqa'])),
            server_configuration=dict(template=old['server_configuration'], gpus=args.gpus,
                actual_service_receipt=str(root / 'servers.json'), cpu_memory_limit_gib=64,
                note='Same model and serving settings; new replica/device identities recorded separately'),
            extension=dict(source_root=str(previous), source_manifest_sha256=old_hash,
                added_methods=list(ADDED_METHODS), purpose='Complete all existing baseline task judgements'))
        write(root / 'manifest.json', manifest)
        write(root / 'report_plan.json', dict(frozen_utc=now(),
            report_code_sha256=sha(runtime / 'qwen_task_report.py'), bootstrap_trials=BOOTSTRAP_TRIALS,
            seed=9232026, family_size=FAMILY_SIZE, comparisons=PAIRS,
            new_comparisons_frozen_before_new_judgements=True,
            primary_aggregation='category macro of paired scene means; two seeds retained',
            equivalence_margin=None, historical_report=str(previous / 'report/REPORT.md'),
            note='100k resamples resolve the expanded Bonferroni tails; historical 20k reports are retained unchanged.'))
        shutil.copy2(previous / 'model_weights_verified.json', root / 'model_weights_verified.json')
    _, mh = verify_manifest(root)
    receipts = []
    for index, job in enumerate(old_jobs, 1):
        source = previous / 'results' / (job['id'] + '.json')
        value = validator.result(source, job)
        proof = dict(source=str(source), source_sha256=sha(source), source_manifest_sha256=old_hash,
                     identical_queries=True, identical_media=True, imported_utc=now())
        target = root / 'results' / source.name
        if target.exists():
            saved = read(target)
            validate_saved(saved, job, scenes[job['case_id']], mh)
            if saved['answers'] != value['answers'] or saved['media'] != value['media']:
                raise ValueError('Imported answers changed')
        else:
            value.update(manifest_sha256=mh, reused_from=proof)
            write(target, value)
        receipts.append(proof)
        if index % 2000 == 0:
            print(json.dumps(dict(imported=index, expected=len(old_jobs))), flush=True)
    write(root / 'reuse_audit.json', dict(complete=True, imported=len(old_jobs),
        source_manifest_sha256=old_hash, manifest_sha256=mh, proofs=receipts,
        historical_source_manifests={str(k): v for k, v in validator.roots.items()}))
    if (previous / 'judge_repeatability.json').exists():
        source = previous / 'judge_repeatability.json'
        value = read(source)
        if value['manifest_sha256'] != old_hash or not value['complete']:
            raise ValueError('Historical repeatability identity mismatch')
        value.update(manifest_sha256=mh, reused_from=dict(source=str(source), source_sha256=sha(source),
            source_manifest_sha256=old_hash), scope='historical four methods only; no new-method repeatability claim')
        write(root / 'judge_repeatability.json', value)
    added = [j for j in jobs if j['method'] in ADDED_METHODS]
    status = dict(complete=True, utc=now(), manifest_sha256=mh, imported_videos=len(old_jobs),
        added_methods=list(ADDED_METHODS), added_videos=len(added),
        added_requests=sum(len(scenes[j['case_id']]['queries']) for j in added),
        by_group=dict(Counter(j['model'] + '/' + j['benchmark'] + '/' + j['method'] for j in added)),
        total_videos=len(jobs), gpus=args.gpus, runtime=str(root / 'runtime_code'))
    if not (root / 'servers.json').exists():
        write(root / 'servers.json', dict(status='prepared', pid=0, replicas=[], control_root=str(root)))
    write(root / 'extension_prepared.json', status)
    write(root / 'campaign.json', dict(status, stage='prepared', complete=False))
    print(json.dumps(status, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--from-root', type=Path, required=True)
    p.add_argument('--metrics-root', type=Path, default=Path('/home/yufeng/thunderworld/judge_full_metrics_20260923'))
    p.add_argument('--cosmos-root', type=Path, default=Path('/home/yufeng/thunderworld/cosmos'))
    p.add_argument('--gpus', type=int, nargs='+', default=[2, 3])
    p.add_argument('--frozen-runtime', action='store_true')
    args = p.parse_args()
    args.root, args.from_root = args.root.resolve(), args.from_root.resolve()
    if args.root == args.from_root or len(args.gpus) != len(set(args.gpus)):
        p.error('Use a fresh root and unique GPU IDs')
    if args.frozen_runtime:
        prepare_frozen(args)
    else:
        runtime = freeze_runtime(args)
        os.execv(sys.executable, [sys.executable, str(runtime / Path(__file__).name),
                                 *sys.argv[1:], '--frozen-runtime'])


if __name__ == '__main__':
    main()
