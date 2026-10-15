"""Scientific guardrails for the task judge, requiring no GPU or model downloads."""
import copy
import unittest

from qwen_task_judge import messages, summarize_video
from qwen_task_protocol import parse_response, rbench_indices, frame_indices, score_schema
from qwen_task_report import bootstrap_scene_delta, consistency


class JudgeProtocolTests(unittest.TestCase):
    def binary(self, expected='NO'):
        return dict(id='q1', kind='binary', expected=expected, prompt='Is a robot visible?', system='YES or NO')

    def rubric(self):
        return dict(kind='rubric', category='common_manipulation', schema=score_schema('common_manipulation'))

    def test_reference_answer_is_not_sent(self):
        q = self.binary()
        a = messages(q, dict(kind='video', path='/tmp/opaque.mkv'))
        q['expected'] = 'YES'
        self.assertEqual(a, messages(q, dict(kind='video', path='/tmp/opaque.mkv')))
        self.assertEqual(a[-1]['content'][0]['type'], 'video_url')

    def test_ambiguous_or_truncated_binary_cannot_be_scored(self):
        for text, finish in [('YES and NO', 'stop'), ('NO', 'length'), ('No robot is visible', 'stop')]:
            with self.assertRaises(ValueError):
                parse_response(self.binary(), text, finish)
        self.assertEqual(parse_response(self.binary(), 'NO', 'stop'), {'answer': 'NO', 'correct': True})

    def test_arithmetic_is_recomputed_and_original_preserved(self):
        import json
        obj = {k: {'reason': 'visible', 'score': 4} for k in self.rubric()['schema']['required']}
        obj['total']['score'] = 2  # wrong arithmetic must not become a spurious precision effect
        r = parse_response(self.rubric(), json.dumps(obj), 'stop')
        self.assertEqual(r['score'], 4)
        self.assertEqual(r['upstream_final_score'], 2)
        self.assertEqual(r['arithmetic_discrepancy'], 2)
        obj['task_completion']['score'] = 1
        self.assertEqual(parse_response(self.rubric(), json.dumps(obj), 'stop')['score'], 1)

    def test_invalid_ratings_and_duplicate_keys_rejected(self):
        q = dict(kind='supplemental', category='supplemental')
        for text in ['{"score": 6, "reason": "x"}', '{"score": true, "reason": "x"}',
                     '{"score": 2, "score": 5, "reason": "x"}', '{"score": NaN, "reason": "x"}']:
            with self.assertRaises(ValueError):
                parse_response(q, text, 'stop')

    def test_frame_indices_include_tail_and_match_official_grid(self):
        self.assertEqual(rbench_indices(61), [0, 12, 24, 36, 48, 60])
        self.assertEqual(frame_indices(61, 16), list(range(0, 61, 4)))
        self.assertEqual(frame_indices(3, 16), [0, 1, 2])
        self.assertEqual(frame_indices(62, 16)[-1], 61)

    def test_empty_official_annotation_never_becomes_accuracy(self):
        s = {'benchmark': 'paibench', 'queries': [{'kind': 'supplemental'}]}
        r = summarize_video(s, [{'parsed': {'score': 5}}])
        self.assertIsNone(r['official_domain_accuracy'])
        self.assertEqual(r['binary_questions'], 0)
        self.assertEqual(r['supplemental_adherence'], 5)

    def test_agreement_is_distinct_from_accuracy(self):
        scene = {'benchmark': 'paibench', 'queries': [{'kind': 'binary', 'id': 'q1'}]}
        c = {'answers': [{'query_id': 'q1', 'parsed': {'answer': 'YES', 'correct': False}}]}
        r = consistency(c, copy.deepcopy(c), scene)
        self.assertEqual(r['answer_agreement'], 1)
        self.assertEqual(r['both_correct'], 0)
        self.assertEqual(r['reference_correct'], 0)
        wrong = copy.deepcopy(c)
        wrong['answers'][0]['query_id'] = 'q2'
        with self.assertRaises(ValueError):
            consistency(c, wrong, scene)

    def test_paired_bootstrap_and_category_weighting(self):
        rows = [dict(category='small', delta=1)] + [dict(category='large', delta=-1) for _ in range(20)]
        r = bootstrap_scene_delta(rows, 'delta', trials=2000)
        self.assertEqual(r['ci95_low'], 0)
        self.assertEqual(r['ci95_high'], 0)

    def test_report_preserves_pairing_and_empty_annotation_denominators(self):
        import json
        from pathlib import Path
        import tempfile
        from qwen_task_protocol import protocol, digest, sha, write
        from qwen_task_report import report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenes = {
                'scored': dict(benchmark='paibench', category='robot', seeds=[1, 2],
                    queries=[dict(kind='binary', id='q1', sha256='q1', expected='YES')]),
                'empty': dict(benchmark='paibench', category='robot', seeds=[1, 2],
                    queries=[dict(kind='supplemental', id='supplemental', sha256='s')]),
            }
            write(root/'scenes.json', scenes)
            jobs = []
            for model in ['edge', 'nano']:
                for method in ['bf16', 'fp8', 'b2_joint']:
                    for cid in scenes:
                        for seed in [1, 2]:
                            job = dict(model=model, method=method, case_id=cid, seed=seed,
                                       benchmark='paibench', category='robot')
                            job['id'] = digest(job)[:24]
                            jobs.append(job)
            (root/'jobs.jsonl').write_text('\n'.join(json.dumps(j) for j in jobs)+'\n')
            write(root/'manifest.json', dict(frozen=True, protocol=protocol(), code_sha256={},
                inputs=dict(scenes_sha256=sha(root/'scenes.json'), jobs_sha256=sha(root/'jobs.jsonl'))))
            mh = sha(root/'manifest.json')
            for job in jobs:
                ok = job['method'] != 'b2_joint'
                if job['case_id'] == 'scored':
                    ans = [dict(query_id='q1', parsed=dict(answer='YES' if ok else 'NO', correct=ok))]
                    metrics = dict(official_domain_accuracy=float(ok))
                else:
                    ans = [dict(query_id='supplemental', parsed=dict(score=5))]
                    metrics = dict(official_domain_accuracy=None, supplemental_adherence=5)
                write(root/'results'/(job['id']+'.json'), dict(complete=True,
                    manifest_sha256=mh, job_sha256=digest(job), job=job, answers=ans, metrics=metrics))
            self.assertTrue(report(root)['complete'])
            result = json.loads((root/'report/summary.json').read_text())
            self.assertEqual(len(result['summary']), 6)
            for row in result['summary']:
                self.assertEqual(row['scored_scenes'], 1)
                self.assertEqual(row['scenes_with_both_seeds'], 2)
                self.assertEqual(row['score_scene_micro'], 0 if row['method']=='b2_joint' else 1)
            for row in result['paired']:
                self.assertEqual(row['paired_scenes'], 1)
                self.assertEqual(row['delta_macro'], -1 if row['candidate']=='b2_joint' else 0)
                self.assertEqual(row['answer_agreement_macro'], 0 if row['candidate']=='b2_joint' else 1)


if __name__ == '__main__':
    unittest.main()
