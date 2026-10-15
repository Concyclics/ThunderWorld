"""Frozen benchmark-task VQA protocol, independent of the generated precision.

This is a Qwen3-VL adaptation, not either benchmark's official leaderboard.
Only prompt/scoring functions are extracted from pinned upstream evaluators;
their heavyweight model-loading and permissive failure handling are not used.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from qwen_judge_pool import write

MODEL_ID = 'Qwen/Qwen3-VL-8B-Instruct-FP8'
MODEL_REVISION = '9cdc6310a8cb770ce18efaf4e9935334512aee45'
RBENCH_REVISION = 'b03df27f0376faa148dcd8cd620a1989a32ca979'
PAI_REVISION = '2f3b687410029b98397fbc51fa4de36bfd45627d'
PAI_DATA_REVISION = 'c36b7beb69bd7c4adcedc1431407fce98f9dfe44'
METHODS = ('bf16', 'fp8', 'b1.5_protected_n', 'b2_joint')
MODELS = ('edge', 'nano')
SAMPLE_SEED = 9232026
TASK_ASPECTS = {
    'common_manipulation': ('action_execution', 'task_completion'),
    'long-horizon_planning': ('action_execution', 'event_completion_ratio'),
    'multi-entity_collaboration': ('action_coordination', 'task_completion'),
    'spatial_relationship': ('spatial_relation_accuracy', 'manipulation_feasibility'),
    'visual_reasoning': ('action_execution', 'visual_reasoning_accuracy'),
}
EMBODIMENTS = ('dual_arm', 'humanoid', 'quad', 'single_arm')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def upstream_functions(path, names):
    tree = ast.parse(Path(path).read_text())
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in definitions} != set(names):
        raise ValueError(f'Missing upstream functions: {path}')
    namespace = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


def upstream(root):
    r = root / 'sources/ReVidgen'
    p = root / 'sources/physical-ai-bench'
    for directory, expected in [(r, RBENCH_REVISION), (p, PAI_REVISION)]:
        actual = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != expected:
            raise ValueError(f'Unexpected upstream revision: {directory}: {actual}')
    funcs, hashes = {}, {}
    for category in TASK_ASPECTS:
        path = r / 'eval/5_tasks' / (category + '.py')
        names = ['create_prompt', 'compute_final_score']
        if category == 'visual_reasoning':
            names.append('create_question_chain')
        funcs[category] = upstream_functions(path, names)
        hashes[str(path)] = sha(path)
    path = r / 'eval/4_embodiments/3_task_adherence_consistency.py'
    funcs['adherence'] = upstream_functions(path, ['create_prompt'])
    hashes[str(path)] = sha(path)
    path = p / 'generation/pbench/prompts/evaluation_prompt.py'
    # Load only literal prompt strings and the pure formatting function.
    tree = ast.parse(path.read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('system_template_binary_v0', 'begin_user_template_binary_v0'):
                    constants[target.id] = ast.literal_eval(node.value)
    constants.update(upstream_functions(path, ['user_template_binary_v0']))
    funcs['pai'] = constants
    hashes[str(path)] = sha(path)
    return funcs, hashes


def score_schema(category):
    detail = {'type': 'object', 'properties': {'reason': {'type': 'string'},
        'score': {'type': 'number', 'minimum': 0, 'maximum': 5}},
        'required': ['reason', 'score'], 'additionalProperties': False}
    if category not in TASK_ASPECTS:
        return {**detail, 'properties': {**detail['properties'],
            'score': {'type': 'integer', 'minimum': 1, 'maximum': 5}}}
    aspects = list(TASK_ASPECTS[category])
    aspects += (['entity_consistency', 'object_consistency', 'anomaly_check']
                if category == 'multi-entity_collaboration' else
                ['object_consistency', 'manipulator_consistency', 'anomaly_check'])
    properties = {key: detail for key in [*aspects, 'total']}
    properties['missing_events'] = {'type': 'array', 'items': {'type': 'string'}}
    return {'type': 'object', 'properties': properties, 'required': [*aspects, 'total'],
            'additionalProperties': False}


def rubric_query(scene, funcs, chain=None):
    meta = scene['annotation']
    cat = scene['category']
    if cat in TASK_ASPECTS:
        kwargs = dict(view=meta['view'], description=meta['prompt'])
        if cat == 'multi-entity_collaboration':
            kwargs.update(manipulated_object=meta['manipulated object'],
                          entity1=meta['entity1'], entity2=meta['entity2'])
        else:
            kwargs.update(robotic_manipulator=meta['robotic manipulator'],
                          manipulated_object=meta['manipulated object'])
        if cat == 'long-horizon_planning':
            kwargs['event_list'] = meta['event_list']
        elif cat == 'spatial_relationship':
            kwargs['spatial_info'] = meta['Spatial Relationship']
        elif cat == 'visual_reasoning':
            if chain is None:
                raise ValueError('Visual reasoning questions must be frozen before scoring')
            kwargs['questions'] = chain
        prompt = funcs[cat]['create_prompt'](**kwargs)
    else:
        if cat not in EMBODIMENTS:
            raise ValueError(f'Unknown RBench category: {cat}')
        prompt = funcs['adherence']['create_prompt'](meta['view'], meta['prompt'])
    return dict(id='task_rubric', kind='rubric', category=cat, system=None,
                prompt=prompt, schema=score_schema(cat), max_tokens=1024)


def scene_queries(scene, funcs, chain=None):
    if scene['benchmark'] == 'rbench':
        return [rubric_query(scene, funcs, chain)]
    questions = []
    for qa in scene['annotation']:
        expected = qa['index2ans'][qa['answer']].upper()
        if expected not in ('YES', 'NO'):
            raise ValueError('Non-binary PAI question')
        if any(q['id'] == qa['uid'] for q in questions):
            raise ValueError('Duplicate PAI question UID')
        questions.append(dict(id=qa['uid'], kind='binary', expected=expected, question=qa['question'],
            system=funcs['pai']['system_template_binary_v0'],
            prompt=funcs['pai']['begin_user_template_binary_v0'] + '\n\n' +
                   funcs['pai']['user_template_binary_v0'](qa, is_reasoning=False),
            max_tokens=16))
    if not questions:
        # A separate, explicitly supplemental metric; NEVER counted as Domain VQA.
        questions.append(dict(id='supplemental_adherence', kind='supplemental',
            category='supplemental', system=None, max_tokens=512,
            prompt=('Evaluate only the visible evidence in this generated video. '
                f'Does it perform the described task: "{scene["prompt"]}"? '
                'Rate prompt adherence from 1 to 5: 1=no relevance, 2=major deviations, '
                '3=partially matches, 4=nearly matches, 5=perfect match. '
                'Use 4 or 5 only when clearly correct. Return only a JSON object '
                'with an integer "score" and a brief "reason".'),
            schema=score_schema('supplemental')))
    return questions


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON response key: ' + key)
        result[key] = value
    return result


def parse_response(query, content, finish_reason):
    if finish_reason != 'stop':
        raise ValueError(f'Unfinished judge response: {finish_reason}')
    content = content.strip()
    if query['kind'] == 'binary':
        answer = content.upper().rstrip('.')
        if answer not in ('YES', 'NO'):
            raise ValueError('Expected an unambiguous YES/NO response')
        return dict(answer=answer, correct=answer == query['expected'])
    if content.startswith('```'):
        content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content)
    obj = json.loads(content, object_pairs_hook=no_duplicate_keys)
    cat = query['category']

    def score(detail, allow_zero=False, integer=False):
        value = detail['score']
        if (type(value) not in (int, float) or not math.isfinite(value) or
                not (0 if allow_zero else 1) <= value <= 5 or
                (integer and int(value) != value) or not isinstance(detail['reason'], str)):
            raise ValueError(f'Invalid score/reason: {detail!r}')
        return float(value)

    if cat not in TASK_ASPECTS:
        value = score(obj, integer=True)
        return dict(score=value, normalized_score=(value - 1) / 4, aspects={'task_adherence': value})
    if 'action_effectiveness' in obj and 'action_execution' not in obj:
        obj['action_execution'] = obj['action_effectiveness']
    aspects = {key: score(obj[key], allow_zero=key in ('visual_reasoning_accuracy', 'event_completion_ratio'))
               for key in query['schema']['required'] if key != 'total'}
    reported_total = score(obj['total'], allow_zero=True)
    a = [aspects[key] for key in TASK_ASPECTS[cat]]
    total = 1.0 if any(v == 1 for v in a) else sum(aspects.values()) / len(aspects)
    # Same final clamp as upstream; recompute the advertised arithmetic instead of
    # allowing a judge's addition error to affect the comparison.
    final = max(min(sum(a) / len(a), total), 1)
    upstream_final = max(min(sum(a) / len(a), reported_total), 1)
    return dict(score=final, normalized_score=(final - 1) / 4, aspects=aspects,
        recomputed_total=total, reported_total=reported_total, upstream_final_score=upstream_final,
        arithmetic_discrepancy=abs(reported_total - total),
        missing_events=obj.get('missing_events'))


def frame_indices(total, count):
    # Exactly numpy.linspace(0, total - 1, count, dtype=int), without numpy import.
    if total < 1 or count < 2:
        raise ValueError('Invalid frame counts')
    if total <= count:
        return list(range(total))
    return [(i * (total - 1)) // (count - 1) for i in range(count)]


def rbench_indices(total=61):
    first = frame_indices(total, 16)
    return [first[i] for i in frame_indices(len(first), 6)]


def build_inputs(root, metrics_root, cosmos_root):
    funcs, source_hashes = upstream(root)
    annotations = {}
    for p in sorted((cosmos_root / 'evaluation/cosmos3/generator/rbench/assets/prompts').glob('*.json')):
        source_hashes[str(p)] = sha(p)
        for row in read(p):
            if row['name'] in annotations:
                raise ValueError('Duplicate RBench annotation')
            annotations[row['name']] = row
    scenes, jobs = {}, []
    empty = []
    for model in MODELS:
        for path in sorted((metrics_root / 'tasks' / model).glob('*.json')):
            task = read(path)
            cid = task['case_id']
            if task['model'] != model:
                raise ValueError('Wrong source model')
            ann = annotations[task['source_id']] if task['benchmark'] == 'rbench' else None
            if ann is None:
                p = root / 'sources/paibench-data/vqa' / (task['source_id'] + '.json')
                ann = read(p)
                source_hashes[str(p)] = sha(p)
            if task['benchmark'] == 'rbench' and ann['prompt'] != task['case']['prompt']:
                raise ValueError('Generated prompt differs from the task rubric')
            scene = dict(case_id=cid, benchmark=task['benchmark'], category=task['category'],
                source_id=task['source_id'], prompt=task['case']['prompt'], annotation=ann,
                seeds=task['seeds'], input_image_sha256=task['image_sha256'],
                prior_scene_roles=task['prior_scene_roles'])
            if cid in scenes and scenes[cid] != scene:
                raise ValueError('Models do not share identical scene/seed annotations')
            scenes[cid] = scene
            for method in METHODS:
                for seed in task['seeds']:
                    rec = task['records'][method][str(seed)]
                    meta = read(rec['metadata'])
                    if ((meta['model'], meta['method'], meta['case_id'], meta['seed']) !=
                            (model, method, cid, seed)):
                        raise ValueError('Source export identity mismatch')
                    if (meta['npz_sha256'] if 'npz_sha256' in meta else meta['provenance']['npz_sha256']) != rec['npz_sha256']:
                        raise ValueError('Source export/raw NPZ digest mismatch')
                    if (meta['frames'], meta['height'], meta['width'], meta['fps'], meta['request']['steps']) != (61, 192, 320, 24, 35):
                        raise ValueError('Inconsistent source generation settings')
                    video = meta['videos']['lossless_rgb']
                    if not video['rgb_roundtrip_bitwise_equal']:
                        raise ValueError('Lossless source RGB verification missing')
                    jobs.append(dict(id=digest([model, method, cid, seed])[:24], model=model,
                        method=method, case_id=cid, seed=seed, benchmark=task['benchmark'],
                        category=task['category'], sample_id=rec['sample_id'],
                        source_task=str(path), source_task_sha256=sha(path),
                        source_video=video['file'], source_video_sha256=video['sha256'],
                        source_rgb_sha256=meta['rgb_uint8_sha256'], source_npz=rec['path'],
                        source_npz_sha256=rec['npz_sha256']))
    expected_jobs = 1694 * len(MODELS) * len(METHODS) * 2
    if len(scenes) != 1694 or len(jobs) != expected_jobs or len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError(f'Incomplete or duplicated cohort: {len(scenes)}, {len(jobs)}')
    for cid, scene in scenes.items():
        if scene['benchmark'] == 'paibench' and not scene['annotation']:
            empty.append(cid)
    inputs = dict(scenes=scenes, jobs=jobs, source_hashes=source_hashes,
                  empty_official_vqa=sorted(empty), metric_manifest_sha256=sha(metrics_root / 'manifest.json'))
    write(root / 'input_audit.json', dict(total_videos=len(jobs), unique_scenes=len(scenes),
        empty_official_vqa=empty, passed=True, sources=source_hashes,
        prior_pixel_manifest_sha256=inputs['metric_manifest_sha256']))
    write(root / 'prepared_inputs.json', inputs)
    return inputs, funcs


def protocol():
    return dict(schema='thunderworld.qwen_task_judge.v1', judge=MODEL_ID, judge_revision=MODEL_REVISION,
        methods=list(METHODS), models=list(MODELS), temperature=0, seed=SAMPLE_SEED,
        precision='official fine-grained FP8 checkpoint; default BF16 KV cache',
        rbench_revision=RBENCH_REVISION, pai_revision=PAI_REVISION, pai_data_revision=PAI_DATA_REVISION,
        generation=dict(width=320, height=192, frames=61, fps=24, steps=35),
        rbench=dict(input='lossless PNG 3x2 chronological grid; upstream two-stage frame sampling',
            frame_indices=rbench_indices(), dimensions='5 task-category rubrics; 4 embodiment task-adherence rubrics',
            excluded='motion amplitude/smoothness, identity stability and composite leaderboard score',
            scoring='upstream final clamp with deterministic arithmetic recomputation; raw upstream score also retained',
            visual_reasoning='one text-only chain per scene, frozen across all methods/models/seeds'),
        paibench=dict(input='16 uniform native video frames from lossless FFV1 source; true timestamps',
            frame_indices=frame_indices(61, 16), questions='all official binary VQA, one question per request',
            sampling_deviation='fixed 16 frames instead of upstream 2 fps, applied identically to every method',
            missing='8 empty official annotations excluded from Domain accuracy; supplemental adherence reported separately',
            decoding='constrained YES/NO; never supplies reference answer to the judge'),
        blinding='opaque media IDs; no precision/model/method labels in judge prompts',
        aggregation='questions -> seed -> scene; category macro and scene micro both reported',
        comparison='paired same-scene, same-seed; bootstrap scene clusters, stratified by category',
        interpretation='VLM task/semantic fidelity; not robot execution success, not official leaderboard',
        performance_eligible=False)
