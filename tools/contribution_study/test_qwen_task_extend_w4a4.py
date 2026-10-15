"""Guard extension semantics and transitive reuse of previously scored videos."""
import ast
from collections import defaultdict
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from qwen_task_extend_w4a4 import ProvenanceValidator, patch_protocol, patch_report, sha
from qwen_task_judge import summarize_video
from qwen_task_protocol import digest

HERE = Path(__file__).parent


class ExtensionTest(unittest.TestCase):
    def test_protocol_changes_only_methods(self):
        source = (HERE/'qwen_task_protocol.py').read_text()
        before, after = ast.parse(source), ast.parse(patch_protocol(source))
        for tree in (before, after):
            tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign)
                and any(getattr(t, 'id', None)=='METHODS' for t in n.targets))]
        self.assertEqual(ast.dump(before), ast.dump(after))

    def test_more_comparisons_widen_adjusted_interval_without_changing_estimator(self):
        source = (HERE/'qwen_task_report.py').read_text()
        funcs = []
        for text in (source, patch_report(source)):
            fn = next(n for n in ast.parse(text).body
                      if isinstance(n, ast.FunctionDef) and n.name=='bootstrap_scene_delta')
            env = dict(defaultdict=defaultdict, np=np)
            exec(compile(ast.Module(body=[fn], type_ignores=[]), '<test>', 'exec'), env)
            funcs.append(env['bootstrap_scene_delta'])
        rows = [dict(category=str(i%3), delta=(i%7-3)/5) for i in range(51)]
        old, new = [f(rows, 'delta', trials=2000) for f in funcs]
        self.assertEqual(old['ci95_low'], new['ci95_low'])
        self.assertEqual(old['ci95_high'], new['ci95_high'])
        self.assertLessEqual(new['family40_ci_low'], old['family24_ci_low'])
        self.assertGreaterEqual(new['family40_ci_high'], old['family24_ci_high'])

    def fixture(self, directory):
        root = Path(directory)
        job = dict(id='sample', case_id='scene', model='edge', method='bf16', seed=1)
        query = dict(id='q1', kind='binary', expected='YES')
        query['sha256'] = digest(query)
        scene = dict(benchmark='paibench', queries=[query])
        image = root/'media'; image.write_bytes(b'fixed image')
        media = dict(path=str(image), sha256=sha(image))
        answer = dict(query_id='q1', query_sha256=query['sha256'],
            identity=digest(['original_manifest', job, query['sha256'], media, '']),
            parsed=dict(answer='YES', correct=True),
            response=dict(raw=dict(choices=[dict(finish_reason='stop', message=dict(content='YES'))])))
        original = dict(complete=True, manifest_sha256='original_manifest', job=job,
            job_sha256=digest(job), media=media, answers=[answer],
            metrics=summarize_video(scene, [answer]))
        a, b = root/'a/results/sample.json', root/'b/results/sample.json'
        for p in (a,b):p.parent.mkdir(parents=True)
        a.write_text(json.dumps(original))
        inherited = copy.deepcopy(original)
        inherited.update(manifest_sha256='extended_manifest', reused_from=dict(source=str(a),
            source_sha256=sha(a), source_manifest_sha256='original_manifest'))
        b.write_text(json.dumps(inherited))
        validator = ProvenanceValidator(dict(judge='test'), {'scene':scene})
        hashes = {(root/'a').resolve():'original_manifest', (root/'b').resolve():'extended_manifest'}
        validator.root = lambda r: hashes[r.resolve()]
        return validator, job, scene, a, b, inherited

    def test_transitive_original_identity_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            validator, job, _, _, path, value = self.fixture(directory)
            self.assertEqual(validator.result(path, job)['answers'], value['answers'])

    def test_reused_answer_change_is_rejected_even_if_locally_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            validator, job, scene, _, path, value = self.fixture(directory)
            answer = value['answers'][0]
            answer['response']['raw']['choices'][0]['message']['content']='NO'
            answer['parsed']=dict(answer='NO', correct=False)
            value['metrics']=summarize_video(scene, value['answers'])
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'altered'):
                validator.result(path, job)

    def test_changed_source_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            validator, job, _, source, path, _ = self.fixture(directory)
            source.write_text(source.read_text()+'\n')
            with self.assertRaisesRegex(ValueError, 'source result'):
                validator.result(path, job)


if __name__=='__main__':
    unittest.main()
