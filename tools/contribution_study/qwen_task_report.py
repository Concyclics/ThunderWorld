"""Task scores and paired decision consistency; BF16 is a comparator, not truth."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from qwen_task_protocol import METHODS, MODELS, TASK_ASPECTS, digest, now, read, sha, write


def mean(values):
    return float(np.mean(values)) if values else None


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def category_mean(rows, value):
    cats = defaultdict(list)
    for r in rows:
        if r[value] is not None:
            cats[r['category']].append(r[value])
    return mean([mean(v) for v in cats.values()])


def bootstrap_scene_delta(rows, value, *, trials=20000, seed=9232026):
    """Paired rows have already averaged all seeds within each scene."""
    groups = defaultdict(list)
    for r in rows:
        groups[r['category']].append(r[value])
    if not groups:
        return {}
    rng = np.random.default_rng(seed)
    result = np.zeros(trials, dtype=np.float64)
    for values in groups.values():
        v = np.array(values, dtype=np.float64)
        for start in range(0, trials, 500):
            n = min(500, trials - start)
            draws = rng.integers(0, len(v), size=(n, len(v)))
            result[start:start+n] += v[draws].mean(axis=1) / len(groups)
    lo, hi = np.quantile(result, [.025, .975])
    # Six pairs among four methods, for four model/benchmark groups.
    flo, fhi = np.quantile(result, [.05 / (2 * 24), 1 - .05 / (2 * 24)])
    return dict(ci95_low=float(lo), ci95_high=float(hi),
                family24_ci_low=float(flo), family24_ci_high=float(fhi))


def consistency(candidate, reference, scene):
    if len(candidate['answers']) != len(reference['answers']):
        raise ValueError('Unpaired question counts')
    agree, distance, decisions = [], [], []
    reference_correct = both_correct = 0
    if scene['benchmark'] == 'paibench':
        for c, r, q in zip(candidate['answers'], reference['answers'], scene['queries']):
            if c['query_id'] != r['query_id'] or c['query_id'] != q['id']:
                raise ValueError('Unpaired question identities')
            if q['kind'] != 'binary':
                continue
            cp, rp = c['parsed'], r['parsed']
            agree.append(float(cp['answer'] == rp['answer']))
            reference_correct += int(rp['correct'])
            both_correct += int(cp['correct'] and rp['correct'])
        return dict(answer_agreement=mean(agree), within_one_agreement=None,
                    rating_ge4_agreement=None, reference_correct=reference_correct,
                    both_correct=both_correct, units=len(agree))
    cat = scene['category']
    names = TASK_ASPECTS.get(cat, ('task_adherence',))
    ca, ra = candidate['answers'][0]['parsed']['aspects'], reference['answers'][0]['parsed']['aspects']
    for name in names:
        agree.append(float(abs(ca[name] - ra[name]) < 1e-8))
        distance.append(float(abs(ca[name] - ra[name]) <= 1))
        decisions.append(float((ca[name] >= 4) == (ra[name] >= 4)))
    return dict(answer_agreement=mean(agree), within_one_agreement=mean(distance),
                rating_ge4_agreement=mean(decisions), reference_correct=None,
                both_correct=None, units=len(agree))


def report(root):
    from qwen_task_judge import load_jobs, verify_manifest
    if (root / 'report_plan.json').exists():
        plan = read(root / 'report_plan.json')
        if plan['report_code_sha256'] != sha(Path(__file__)):
            raise ValueError('Frozen statistical/report implementation changed')
    manifest, mh = verify_manifest(root)
    scenes = read(root / 'scenes.json')
    jobs = load_jobs(root)
    data, missing = {}, []
    video_rows, question_rows, aspect_rows = [], [], []
    for job in jobs:
        path = root / 'results' / (job['id'] + '.json')
        if not path.exists():
            missing.append(job['id'])
            continue
        result = read(path)
        if result['manifest_sha256'] != mh or result['job_sha256'] != digest(job) or not result['complete']:
            raise ValueError('Unqualified result in report')
        key = (job['model'], job['method'], job['case_id'], job['seed'])
        if key in data:
            raise ValueError('Duplicate video in report')
        data[key] = result
        score = result['metrics'].get('task_score') if job['benchmark'] == 'rbench' else result['metrics']['official_domain_accuracy']
        row = dict(model=job['model'], method=job['method'], benchmark=job['benchmark'],
            category=job['category'], case_id=job['case_id'], seed=job['seed'], id=job['id'],
            score=score, metric='task_vqa_1_to_5' if job['benchmark'] == 'rbench' else 'domain_vqa_accuracy',
            supplemental_adherence=result['metrics'].get('supplemental_adherence'))
        video_rows.append(row)
        scene = scenes[job['case_id']]
        for answer, query in zip(result['answers'], scene['queries']):
            if answer['query_id'] != query['id']:
                raise ValueError('Question alignment changed')
            parsed = answer['parsed']
            base = {k: row[k] for k in ('model', 'method', 'benchmark', 'category', 'case_id', 'seed', 'id')}
            question_rows.append(dict(**base, query_id=query['id'], kind=query['kind'],
                expected=query.get('expected'), answer=parsed.get('answer'),
                correct=parsed.get('correct'), score=parsed.get('score'),
                raw_result=str(path), query_sha256=query['sha256']))
            for aspect, value in parsed.get('aspects', {}).items():
                aspect_rows.append(dict(**base, aspect=aspect, score=value))
    out = root / 'report'
    out.mkdir(exist_ok=True)
    write_csv(out / 'per_video.csv', video_rows)
    write_csv(out / 'per_question.csv', question_rows)
    write_csv(out / 'per_aspect.csv', aspect_rows)
    groups = defaultdict(list)
    for row in video_rows:
        groups[(row['model'], row['benchmark'], row['method'], row['case_id'])].append(row)
    scene_rows = []
    for (model, benchmark, method, cid), rows in groups.items():
        if {r['seed'] for r in rows} != set(scenes[cid]['seeds']):
            continue
        scores = [r['score'] for r in rows if r['score'] is not None]
        scene_rows.append(dict(model=model, benchmark=benchmark, method=method, case_id=cid,
            category=scenes[cid]['category'], score=mean(scores), seeds=len(rows),
            supplemental_adherence=mean([r['supplemental_adherence'] for r in rows if r['supplemental_adherence'] is not None])))
    write_csv(out / 'per_scene.csv', scene_rows)
    summary, categories = [], []
    for model in MODELS:
        for benchmark in ('rbench', 'paibench'):
            for method in METHODS:
                rows = [r for r in scene_rows if (r['model'], r['benchmark'], r['method']) == (model, benchmark, method)]
                if not rows:
                    continue
                summary.append(dict(model=model, benchmark=benchmark, method=method,
                    scenes_with_both_seeds=len(rows), scored_scenes=sum(r['score'] is not None for r in rows),
                    score_scene_micro=mean([r['score'] for r in rows if r['score'] is not None]),
                    score_category_macro=category_mean(rows, 'score'),
                    supplemental_adherence=mean([r['supplemental_adherence'] for r in rows if r['supplemental_adherence'] is not None])))
                for category in sorted({r['category'] for r in rows}):
                    part = [r for r in rows if r['category'] == category]
                    categories.append(dict(model=model, benchmark=benchmark, method=method, category=category,
                        scenes=len(part), scored_scenes=sum(r['score'] is not None for r in part),
                        score=mean([r['score'] for r in part if r['score'] is not None])))
    pairs, paired_videos = [], []
    for model in MODELS:
        for benchmark in ('rbench', 'paibench'):
            for candidate, reference in [('b2_joint', 'bf16'), ('b2_joint', 'fp8'), ('fp8', 'bf16'),
                                         ('b1.5_protected_n', 'bf16'), ('b1.5_protected_n', 'fp8'),
                                         ('b1.5_protected_n', 'b2_joint')]:
                per_scene = []
                count_ref_correct = count_both_correct = 0
                for cid, scene in scenes.items():
                    if scene['benchmark'] != benchmark:
                        continue
                    records = []
                    for seed in scene['seeds']:
                        c = data.get((model, candidate, cid, seed))
                        r = data.get((model, reference, cid, seed))
                        if c is None or r is None:
                            continue
                        metric = 'task_score' if benchmark == 'rbench' else 'official_domain_accuracy'
                        cv, rv = c['metrics'][metric], r['metrics'][metric]
                        if cv is None or rv is None:
                            continue
                        con = consistency(c, r, scene)
                        row = dict(model=model, benchmark=benchmark, candidate=candidate,
                            reference=reference, case_id=cid, category=scene['category'], seed=seed,
                            candidate_score=cv, reference_score=rv, delta=cv - rv, absolute_difference=abs(cv-rv), **con)
                        paired_videos.append(row)
                        records.append(row)
                    if len(records) != len(scene['seeds']):
                        continue
                    per_scene.append(dict(case_id=cid, category=scene['category'],
                        **{key: mean([r[key] for r in records if r[key] is not None]) for key in
                           ['delta', 'absolute_difference', 'answer_agreement', 'within_one_agreement', 'rating_ge4_agreement']}))
                    count_ref_correct += sum(r['reference_correct'] or 0 for r in records)
                    count_both_correct += sum(r['both_correct'] or 0 for r in records)
                if not per_scene:
                    continue
                pairs.append(dict(model=model, benchmark=benchmark, candidate=candidate, reference=reference,
                    paired_scenes=len(per_scene), delta_macro=category_mean(per_scene, 'delta'),
                    delta_micro=mean([r['delta'] for r in per_scene]),
                    absolute_difference_macro=category_mean(per_scene, 'absolute_difference'),
                    answer_agreement_macro=category_mean(per_scene, 'answer_agreement'),
                    within_one_agreement_macro=category_mean(per_scene, 'within_one_agreement'),
                    rating_ge4_agreement_macro=category_mean(per_scene, 'rating_ge4_agreement'),
                    reference_correct_pooled=count_ref_correct,
                    both_correct_pooled=count_both_correct,
                    reference_correct_retention_pooled=count_both_correct/count_ref_correct if count_ref_correct else None,
                    worst_scene_delta=min(r['delta'] for r in per_scene),
                    **bootstrap_scene_delta(per_scene, 'delta')))
    write_csv(out / 'summary.csv', summary)
    write_csv(out / 'per_category.csv', categories)
    write_csv(out / 'paired_comparisons.csv', pairs)
    write_csv(out / 'paired_per_video.csv', paired_videos)
    bias = []
    for model in MODELS:
        for method in METHODS:
            qs = [q for q in question_rows if q['model'] == model and q['method'] == method and q['kind'] == 'binary']
            if not qs:
                continue
            yes = [q for q in qs if q['expected'] == 'YES']
            no = [q for q in qs if q['expected'] == 'NO']
            ya, na = mean([int(q['correct']) for q in yes]), mean([int(q['correct']) for q in no])
            bias.append(dict(model=model, method=method, questions=len(qs),
                expected_yes_fraction=len(yes)/len(qs), observed_yes_fraction=mean([int(q['answer'] == 'YES') for q in qs]),
                accuracy_expected_yes=ya, accuracy_expected_no=na,
                balanced_label_accuracy=(ya+na)/2 if ya is not None and na is not None else None,
                always_yes_question_micro_accuracy=len(yes)/len(qs)))
    write_csv(out / 'pai_answer_bias.csv', bias)
    repeatability = []
    if (root / 'judge_repeatability.json').exists():
        repeated = read(root / 'judge_repeatability.json')
        if repeated['manifest_sha256'] != mh or not repeated['complete']:
            raise ValueError('Judge repeatability provenance mismatch')
        for model in MODELS:
            for benchmark in ('rbench', 'paibench'):
                part = [r for r in repeated['results'] if
                        (r['job']['model'], r['job']['benchmark']) == (model, benchmark)]
                repeatability.append(dict(model=model, benchmark=benchmark, videos=len(part),
                    mean_absolute_score_change=mean([r['absolute_score_change'] for r in part if r['absolute_score_change'] is not None]),
                    mean_answer_agreement=mean([r['consistency']['answer_agreement'] for r in part if r['consistency']['answer_agreement'] is not None])))
        write_csv(out / 'judge_repeatability.csv', repeatability)
    status = dict(complete=not missing, utc=now(), manifest_sha256=mh,
                  total=len(jobs), completed=len(data), missing=len(missing),
                  performance_eligible=False, report_code_sha256=sha(Path(__file__)))
    write(out / 'summary.json', dict(status=status, summary=summary, paired=pairs,
                                   pai_answer_bias=bias, judge_repeatability=repeatability))
    lines = [f'# Qwen3-VL-8B-Instruct-FP8 task judge ({"complete" if not missing else "partial"})', '',
        f'{len(data):,}/{len(jobs):,} videos; Edge + Nano, BF16 + FP8 + b1.5_protected_n + b2_joint.', '',
        'This is a Qwen judge adaptation. BF16 and FP8 are paired comparators, not ground truth. '
        'Scores measure generated-video task/semantic fidelity, not executed robot-task success. '
        'These are not official leaderboard scores.', '',
        'RBench: five task rubrics plus task adherence for four embodiment categories, on the upstream six-frame grid. '
        'The reported task VQA score uses deterministic recomputation of the rubric arithmetic (1–5, higher is better). '
        'Motion, identity-stability and the full official composite are not measured here.', '',
        'PAI-Bench: all 5,636 official binary questions for 1,036 annotated scenes; '
        'eight scenes with empty official question files have separate supplemental adherence scores. '
        'Domain accuracy averages questions within each seed, then seeds within each scene. '
        '16 uniform frames with original timestamps replace the upstream 2 fps sampling.', '',
        'All precision variants use identical prompts, frame indices and judging parameters. '
        'Raw answers, official reported totals and question-chain provenance are retained. '
        'Category-macro is primary; scene-micro is also supplied. Prior generated scenes include earlier development scenes; '
        'this is not a new held-out generalization claim.', '',
        '| Model | Benchmark | Method | Scored scenes | Category macro | Scene micro |',
        '|---|---|---|---:|---:|---:|']
    for r in summary:
        if r['score_category_macro'] is not None:
            lines.append(f'| {r["model"]} | {r["benchmark"]} | {r["method"]} | {r["scored_scenes"]} | '
                         f'{r["score_category_macro"]:.4f} | {r["score_scene_micro"]:.4f} |')
    lines += ['', 'Paired differences are candidate minus reference (positive favors candidate). '
        'Bootstrap resamples whole scenes, retaining both seeds and question sets, stratified by category. '
        'The CSV also gives Bonferroni family-24 intervals. A confidence interval crossing zero does not establish equivalence.', '',
        '| Model | Benchmark | Comparison | Paired scenes | Macro delta [95% CI] | Task-answer agreement |',
        '|---|---|---|---:|---:|---:|']
    for r in pairs:
        lines.append(f'| {r["model"]} | {r["benchmark"]} | {r["candidate"]} − {r["reference"]} | '
            f'{r["paired_scenes"]} | {r["delta_macro"]:+.4f} [{r["ci95_low"]:+.4f}, {r["ci95_high"]:+.4f}] | '
            f'{100*r["answer_agreement_macro"]:.2f}% |')
    lines += ['', 'Task-answer agreement means identical YES/NO answers for PAI and identical task-aspect ratings for RBench. '
        'For RBench, within-one-point and rating≥4 agreement are supplementary; ≥4 is a prespecified rating threshold, '
        'not a measured task success rate. PAI reference-correct retention is pooled over question instances and separately labelled.', '',
        'The PAI annotation set is YES-heavy. `pai_answer_bias.csv` supplies label-balanced accuracy, '
        'YES/NO-specific accuracy and the always-YES baseline so high raw accuracy is not mistaken for a reliable judge.', '',
        'The fixed repeated-judgement subset is summarized in `judge_repeatability.csv`. '
        'Its raw outputs are separate from the primary scores; batch-order variability is not removed by choosing better answers.', '',
        'Inputs remain the existing 320×192, 61-frame, 24 fps, 35-step generations. '
        'Shared-GPU judge throughput is operational data and is not a model-inference speed result.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    paths = sorted(p for p in out.iterdir() if p.is_file() and p.name not in ('file_hashes.json', 'complete.json'))
    write(out / 'file_hashes.json', {p.name: sha(p) for p in paths})
    if not missing:
        write(out / 'complete.json', status)
    return status
