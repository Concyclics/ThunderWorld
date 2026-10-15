"""Protect protocol identity, paired comparisons and baseline provenance."""
import ast
from collections import defaultdict
import copy
import itertools
from pathlib import Path
import unittest

import numpy as np

import qwen_task_extend_baselines as extension
import qwen_task_extend_w4a4 as previous

HERE = Path(__file__).parent


class BaselineExtensionTest(unittest.TestCase):
    def test_only_method_list_changes_in_protocol(self):
        source = previous.patch_protocol((HERE / 'qwen_task_protocol.py').read_text())
        trees = [ast.parse(t) for t in (source, extension.patch_protocol(source))]
        for tree in trees:
            tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign)
                and any(getattr(t, 'id', None) == 'METHODS' for t in n.targets))]
        self.assertEqual(ast.dump(trees[0]), ast.dump(trees[1]))

    def test_every_pair_exactly_once_and_historical_direction_retained(self):
        expected = {frozenset(p) for p in itertools.combinations(extension.METHODS, 2)}
        self.assertEqual(len(extension.PAIRS), 36)
        self.assertEqual({frozenset(p) for p in extension.PAIRS}, expected)
        self.assertEqual(extension.PAIRS[:10], extension.PREVIOUS_PAIRS)
        self.assertEqual(extension.FAMILY_SIZE, 144)
        for ours in ('b1.5_protected_n', 'b2_joint'):
            for baseline in extension.ADDED_METHODS:
                self.assertIn((ours, baseline), extension.PAIRS)

    def test_bootstrap_estimator_unchanged_at_equal_draw_count(self):
        source = previous.patch_report((HERE / 'qwen_task_report.py').read_text())
        funcs = []
        for code in (source, extension.patch_report(source)):
            fn = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)
                      and n.name == 'bootstrap_scene_delta')
            env = dict(defaultdict=defaultdict, np=np)
            exec(compile(ast.Module(body=[fn], type_ignores=[]), '<test>', 'exec'), env)
            funcs.append(env['bootstrap_scene_delta'])
        rows = [dict(category=str(i % 4), delta=(i % 9 - 4) / 7) for i in range(55)]
        old, new = [f(rows, 'delta', trials=2000) for f in funcs]
        self.assertEqual(old['ci95_low'], new['ci95_low'])
        self.assertEqual(old['ci95_high'], new['ci95_high'])
        self.assertLessEqual(new['family144_ci_low'], old['family40_ci_low'])
        self.assertGreaterEqual(new['family144_ci_high'], old['family40_ci_high'])
        self.assertEqual(funcs[1].__kwdefaults__['trials'], 100000)

    def test_rejects_wrong_prior_scope(self):
        with self.assertRaisesRegex(ValueError, 'five-method'):
            extension.patch_protocol((HERE / 'qwen_task_protocol.py').read_text())
        with self.assertRaisesRegex(ValueError, 'ten-pair'):
            extension.patch_report((HERE / 'qwen_task_report.py').read_text())

    def test_adapter_precision_overrides_placeholder_dispatch(self):
        meta = dict(complete=True, method='arcquant', request=dict(calls=70), method_spec=dict(
            config=dict(precision='bf16', attention_backend='bf16'),
            actual_generation_precision='arcquant', execution_type='official_native_cosmos_adapter',
            limitations=['ARC fraction fixed at 50%']))
        extension.check_method_definition('arcquant', meta)
        changed = copy.deepcopy(meta)
        changed['method_spec']['actual_generation_precision'] = 'ARCQuant-100%'
        with self.assertRaisesRegex(ValueError, 'adapter'):
            extension.check_method_definition('arcquant', changed)
        changed = copy.deepcopy(meta)
        changed['method_spec']['config']['attention_backend'] = 'fp4'
        with self.assertRaisesRegex(ValueError, 'attention'):
            extension.check_method_definition('arcquant', changed)

    def test_awq_quality_reference_cannot_claim_packed_performance(self):
        meta = dict(complete=True, method='awq_reference', request=dict(calls=70), method_spec=dict(
            config=dict(attention_backend='bf16'), baseline_adapter=dict(method='awq'),
            scope='Official ModelOpt NVFP4 AWQ-lite decoded W4A16 reference',
            performance_eligible=False, storage_comparison_eligible=False))
        extension.check_method_definition('awq_reference', meta)
        meta['method_spec']['performance_eligible'] = True
        with self.assertRaisesRegex(ValueError, 'AWQ quality reference'):
            extension.check_method_definition('awq_reference', meta)

    def test_legacy_nano_export_requires_actual_generation_adapter(self):
        meta = dict(complete=True, method='arcquant', request=dict(calls=70),
                    method_spec=dict(config=dict(precision='bf16', attention_backend='bf16')))
        with self.assertRaisesRegex(ValueError, 'Missing native adapter'):
            extension.check_method_definition('arcquant', meta)
        receipt = dict(complete=True, identity=dict(method='arcquant'), rows=[meta['request']],
            adapter=dict(method='ARCQuant', channel_fraction=.5,
                         execution_type='official_native_cosmos_adapter'))
        extension.check_method_definition('arcquant', meta, receipt)
        receipt['adapter']['channel_fraction'] = 1.0
        with self.assertRaisesRegex(ValueError, 'repair fraction'):
            extension.check_method_definition('arcquant', meta, receipt)


if __name__ == '__main__':
    unittest.main()
