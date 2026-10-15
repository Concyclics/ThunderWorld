"""Freeze a five-method judge extension while preserving the four-method run.

Only fp4_base needs new judgements. A private runtime snapshot adds the method
and four pairwise contrasts; numerical parsing and all original prompts remain
unchanged. Reused answers retain and validate their complete provenance chain.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import difflib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

METHODS = ('bf16', 'fp8', 'fp4_base', 'b1.5_protected_n', 'b2_joint')
OLD_PAIRS = [('b2_joint', 'bf16'), ('b2_joint', 'fp8'), ('fp8', 'bf16'),
             ('b1.5_protected_n', 'bf16'), ('b1.5_protected_n', 'fp8'),
             ('b1.5_protected_n', 'b2_joint')]
NEW_PAIRS = [('fp4_base', 'bf16'), ('fp4_base', 'fp8'),
             ('b1.5_protected_n', 'fp4_base'), ('b2_joint', 'fp4_base')]
RUNTIME_FILES = ('qwen_task_protocol.py', 'qwen_task_judge.py', 'qwen_task_report.py',
                 'qwen_task_resume.py', 'qwen_judge_pool.py', 'qwen_resource_session.py')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def patch_protocol(source):
    lines = source.splitlines(keepends=True)
    matches = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'METHODS' for t in node.targets)]
    if len(matches) != 1 or tuple(ast.literal_eval(matches[0].value)) != tuple(m for m in METHODS if m != 'fp4_base'):
        raise ValueError('Expected the frozen four-method protocol')
    node = matches[0]
    lines[node.lineno-1:node.end_lineno] = ['METHODS = ' + repr(METHODS) + '\n']
    return ''.join(lines)


def patch_report(source):
    pairs = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.For)
             and isinstance(node.target, ast.Tuple)
             and [getattr(t, 'id', None) for t in node.target.elts] == ['candidate', 'reference']]
    if len(pairs) != 1 or ast.literal_eval(pairs[0].iter) != OLD_PAIRS:
        raise ValueError('Unexpected original pairwise comparison plan')
    node = pairs[0].iter
    lines = source.splitlines(keepends=True)
    prefix = lines[node.lineno-1][:node.col_offset]
    suffix = lines[node.end_lineno-1][node.end_col_offset:]
    lines[node.lineno-1:node.end_lineno] = [prefix + repr(OLD_PAIRS + NEW_PAIRS) + suffix]
    result = ''.join(lines)
    required = {
        'family24': 'family40',
        'family-24': 'family-40',
        '.05 / (2 * 24)': '.05 / (2 * 40)',
        'Six pairs among four methods': 'Ten pairs among five methods',
        'BF16 + FP8 + b1.5_protected_n + b2_joint': 'BF16 + FP8 + fp4_base + b1.5_protected_n + b2_joint',
        'The fixed repeated-judgement subset is summarized':
            'The historical four-method repeated-judgement subset (not including fp4_base) is summarized',
    }
    for before, after in required.items():
        if before not in result:
            raise ValueError('Report patch anchor missing: ' + before)
        result = result.replace(before, after)
    ast.parse(result)
    return result


def frozen_source(root, path, expected):
    path = Path(path)
    if path.exists() and sha(path) == expected:
        return path
    receipt = read(root / 'frozen_code_receipt.json')
    item = receipt[str(path)]
    if item['sha256'] != expected or sha(item['copy']) != expected:
        raise ValueError('Historical scoring implementation is unavailable: ' + str(path))
    return Path(item['copy'])


def freeze_runtime(args):
    original = read(args.from_root / 'manifest.json')
    if not read(args.from_root / 'report/complete.json')['complete']:
        raise ValueError('Base run must be complete')
    args.root.mkdir(parents=True, exist_ok=True)
    runtime = args.root / 'runtime_code'
    runtime.mkdir(exist_ok=True)
    scripts = Path(__file__).resolve().parent
    locked = {Path(path).name: frozen_source(args.from_root, path, expected)
              for path, expected in original['code_sha256'].items()}
    plan = read(args.from_root / 'report_plan.json')
    if sha(scripts/'qwen_task_report.py') != plan['report_code_sha256']:
        raise ValueError('Base statistical implementation changed')
    changes, receipts = [], {}
    for name in (*RUNTIME_FILES, Path(__file__).name):
        src = locked.get(name, scripts/name)
        text = src.read_text()
        revised = patch_protocol(text) if name == 'qwen_task_protocol.py' else (
            patch_report(text) if name == 'qwen_task_report.py' else text)
        target = runtime/name
        if target.exists() and target.read_text() != revised:
            raise ValueError('Refusing to alter an existing runtime snapshot: ' + name)
        target.write_text(revised)
        receipts[name] = dict(source=str(src), source_sha256=sha(src), runtime_sha256=sha(target))
        changes.extend(difflib.unified_diff(text.splitlines(True), revised.splitlines(True),
                                          fromfile=str(src), tofile=str(target)))
    (args.root/'runtime_patch.diff').write_text(''.join(changes))
    (args.root/'runtime_source_receipt.json').write_text(json.dumps(receipts, indent=2)+'\n')
    return runtime


class ProvenanceValidator:
    def __init__(self, expected_protocol, scenes):
        self.protocol = {k: v for k, v in expected_protocol.items() if k != 'methods'}
        self.scenes = scenes
        self.roots = {}
        self.media = {}

    def root(self, root):
        from qwen_task_protocol import digest
        root = root.resolve()
        if root not in self.roots:
            m = read(root/'manifest.json')
            if {k: v for k, v in m['protocol'].items() if k != 'methods'} != self.protocol:
                raise ValueError('Changed historical judge protocol')
            for name in ('scenes', 'jobs'):
                path = root/(name + ('.json' if name == 'scenes' else '.jsonl'))
                if sha(path) != m['inputs'][name+'_sha256']:
                    raise ValueError('Changed historical cohort')
            if read(root/'scenes.json') != self.scenes:
                raise ValueError('Historical question/annotation mismatch')
            for path, expected in m['code_sha256'].items():
                frozen_source(root, path, expected)
            self.roots[root] = sha(root/'manifest.json')
        return self.roots[root]

    def result(self, path, job, *, seen=()):
        from qwen_task_protocol import digest
        from qwen_task_resume import validate_saved
        path = Path(path).resolve()
        if path in seen or len(seen) > 8:
            raise ValueError('Cyclic or excessive reuse provenance')
        mh = self.root(path.parent.parent)
        value = read(path)
        scene = self.scenes[job['case_id']]
        validate_saved(value, job, scene, mh)
        media = value['media']
        if media['path'] not in self.media:
            self.media[media['path']] = sha(media['path'])
        if self.media[media['path']] != media['sha256']:
            raise ValueError('Changed historical judge media')
        prior = value.get('reused_from')
        if prior:
            if sha(prior['source']) != prior['source_sha256']:
                raise ValueError('Changed reused source result')
            source = self.result(prior['source'], job, seen=(*seen, path))
            if (source['manifest_sha256'] != prior['source_manifest_sha256'] or
                    source['answers'] != value['answers'] or source['media'] != media):
                raise ValueError('Reused answers or media were altered')
        else:
            for answer, query in zip(value['answers'], scene['queries'], strict=True):
                if answer['identity'] != digest([mh, job, query['sha256'], media, '']):
                    raise ValueError('Original response identity mismatch')
        return value


def prepare_frozen(args):
    from qwen_task_protocol import build_inputs, digest, now, protocol, write
    from qwen_task_judge import load_jobs, verify_manifest
    from qwen_task_resume import validate_saved
    root, previous = args.root, args.from_root
    if tuple(protocol()['methods']) != METHODS:
        raise ValueError('Prepare must run in the isolated five-method snapshot')
    for name in ('sources', 'deps', 'chains', 'vllm_cache_gpu0', 'vllm_cache_gpu2',
                 'inductor_gpu0', 'inductor_gpu2'):
        target = root/name
        if not target.exists() and not target.is_symlink():
            target.symlink_to((previous/name).resolve(), target_is_directory=True)
    old = read(previous/'manifest.json')
    for name, expected in old['dependencies'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('Judge dependency changed: ' + name)
    scenes = read(previous/'scenes.json')
    validator = ProvenanceValidator(protocol(), scenes)
    old_hash = validator.root(previous)
    inputs, _ = build_inputs(root, args.metrics_root, args.cosmos_root)
    for cid, scene in inputs['scenes'].items():
        if scene != {k: v for k, v in scenes[cid].items() if k != 'queries'}:
            raise ValueError('New source scene differs from the frozen questions')
    for cid, expected in old['chain_sha256'].items():
        if sha(root/'chains'/(cid+'.json')) != expected:
            raise ValueError('Frozen reasoning question chain changed')
    jobs = sorted(inputs['jobs'], key=lambda j: j['id'])
    new_jobs = {j['id']: j for j in jobs}
    old_jobs = load_jobs(previous)
    for job in old_jobs:
        if new_jobs.get(job['id']) != job:
            raise ValueError('Previously scored generation job changed')
    if not (root/'manifest.json').exists():
        shutil.copy2(previous/'scenes.json', root/'scenes.json')
        with (root/'jobs.jsonl').open('w') as stream:
            for job in jobs:
                stream.write(json.dumps(job, ensure_ascii=False)+'\n')
        runtime = root/'runtime_code'
        m = copy.deepcopy(old)
        m.update(created_utc=now(), protocol=protocol(),
            inputs=dict(scenes_sha256=sha(root/'scenes.json'), jobs_sha256=sha(root/'jobs.jsonl'),
                upstream=inputs['source_hashes'], metric_manifest_sha256=inputs['metric_manifest_sha256']),
            code_sha256={str(p): sha(p) for p in sorted(runtime.glob('*.py'))},
            counts=dict(videos=len(jobs), scenes=len(scenes),
                judge_requests=sum(len(scenes[j['case_id']]['queries']) for j in jobs),
                binary_requests=sum(q['kind']=='binary' for j in jobs for q in scenes[j['case_id']]['queries']),
                empty_annotation_scenes=len(inputs['empty_official_vqa'])),
            server_configuration=dict(template=old['server_configuration'],
                actual_service_receipt=str(root/'servers.json'), cpu_memory_limit_gib=64,
                note='Same model and serving settings; runtime process identities recorded separately'),
            extension=dict(source_root=str(previous), source_manifest_sha256=old_hash,
                added_method='fp4_base', purpose='Measure ordinary W4A4 task fidelity without presuming the outcome'))
        write(root/'manifest.json', m)
        write(root/'report_plan.json', dict(frozen_utc=now(),
            report_code_sha256=sha(runtime/'qwen_task_report.py'), bootstrap_trials=20000,
            seed=9232026, family_size=40, comparisons=OLD_PAIRS+NEW_PAIRS,
            new_comparisons_frozen_before_new_judgements=True,
            primary_aggregation='category macro of paired scene means; two seeds retained',
            equivalence_margin=None, historical_report=str(previous/'report/REPORT.md')))
        shutil.copy2(previous/'model_weights_verified.json', root/'model_weights_verified.json')
    manifest, mh = verify_manifest(root)
    imported = 0
    receipts = []
    for job in old_jobs:
        source = previous/'results'/(job['id']+'.json')
        value = validator.result(source, job)
        proof = dict(source=str(source), source_sha256=sha(source), source_manifest_sha256=old_hash,
                     identical_queries=True, identical_media=True, imported_utc=now())
        target = root/'results'/source.name
        if target.exists():
            validate_saved(read(target), job, scenes[job['case_id']], mh)
        else:
            value.update(manifest_sha256=mh, reused_from=proof)
            write(target, value)
        receipts.append(proof)
        imported += 1
        if imported % 2000 == 0:
            print(json.dumps(dict(imported=imported, expected=len(old_jobs))), flush=True)
    write(root/'reuse_audit.json', dict(complete=True, imported=imported,
        source_manifest_sha256=old_hash, manifest_sha256=mh, proofs=receipts,
        historical_source_manifests={str(k): v for k, v in validator.roots.items()}))
    # These remain historical controls; they do not claim W4A4 repeatability.
    old_repeat = previous/'judge_repeatability.json'
    if old_repeat.exists():
        value = read(old_repeat)
        if value['manifest_sha256'] != old_hash or not value['complete']:
            raise ValueError('Historical repeatability identity mismatch')
        value.update(manifest_sha256=mh, reused_from=dict(source=str(old_repeat), source_sha256=sha(old_repeat),
            source_manifest_sha256=old_hash), scope='historical four methods; fp4_base not included')
        write(root/'judge_repeatability.json', value)
    added = [j for j in jobs if j['method']=='fp4_base']
    for job in added:
        if not Path(job['source_video']).is_file():
            raise ValueError('Missing new W4A4 video')
    status = dict(complete=True, utc=now(), manifest_sha256=mh, imported_videos=imported,
        added_videos=len(added), added_requests=sum(len(scenes[j['case_id']]['queries']) for j in added),
        by_group=dict(Counter(j['model']+'/'+j['benchmark'] for j in added)),
        total_videos=len(jobs), runtime=str(root/'runtime_code'))
    if not (root/'servers.json').exists():
        write(root/'servers.json', dict(status='prepared', pid=0, replicas=[], control_root=str(root)))
    write(root/'extension_prepared.json', status)
    write(root/'campaign.json', dict(status, stage='prepared', complete=False))
    print(json.dumps(status, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--from-root', type=Path, required=True)
    p.add_argument('--metrics-root', type=Path, default=Path('/home/yufeng/thunderworld/judge_full_metrics_20260923'))
    p.add_argument('--cosmos-root', type=Path, default=Path('/home/yufeng/thunderworld/cosmos'))
    p.add_argument('--frozen-runtime', action='store_true')
    args = p.parse_args()
    args.root, args.from_root = args.root.resolve(), args.from_root.resolve()
    if args.root == args.from_root:
        p.error('The extension must use a fresh run root')
    if args.frozen_runtime:
        prepare_frozen(args)
    else:
        runtime = freeze_runtime(args)
        os.execv(sys.executable, [sys.executable, str(runtime/Path(__file__).name),
            *sys.argv[1:], '--frozen-runtime'])


if __name__ == '__main__':
    main()
