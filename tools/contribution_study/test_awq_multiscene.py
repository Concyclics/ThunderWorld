import json
from pathlib import Path
import tempfile
import unittest

from awq_multiscene import HERE, REPO, import_metrics
from awq_multiscene_report import paired_contrast
from baselines import sha, write


def metrics(lpips=.5, ssim=.8, l2=.3, psnr=20.):
    return dict(lpips=lpips, ssim_mean=ssim, relative_l2=l2, psnr_db=psnr, rmse=.1)


class AWQComparisonTests(unittest.TestCase):
    statistics = dict(replicates=1000, bootstrap_seed=9, family_size=16)

    def test_scene_clusters_and_equal_category_weights(self):
        categories = {'small': 'small', **{f'large{i}': 'large' for i in range(4)}}
        candidate, reference = {}, {}
        for case, category in categories.items():
            for seed in (1, 2):
                reference[case, seed] = metrics()
                candidate[case, seed] = metrics(lpips=.4 if category == 'small' else .2, ssim=.81, l2=.29)
        result = paired_contrast(candidate, reference, categories, self.statistics)
        self.assertAlmostEqual(result['metrics']['lpips']['mean_delta'], -.2)
        self.assertEqual(result['metrics']['lpips']['n_scenes'], 5)
        self.assertTrue(result['candidate_improves'])
        # Duplicating paired seeds within each scene must not change the sampling unit.
        for case in categories:
            candidate[case, 3] = candidate[case, 1]
            candidate[case, 4] = candidate[case, 2]
            reference[case, 3] = reference[case, 1]
            reference[case, 4] = reference[case, 2]
        repeated = paired_contrast(candidate, reference, categories, self.statistics)
        self.assertEqual(result['metrics']['lpips']['ci95'], repeated['metrics']['lpips']['ci95'])
        self.assertEqual(repeated['metrics']['lpips']['n_scenes'], 5)

    def test_lpips_alone_does_not_pass_quality_guard(self):
        result = paired_contrast({('a', 1): metrics(lpips=.1, ssim=.7)},
                                 {('a', 1): metrics()}, {'a': 'one'}, self.statistics)
        self.assertLess(result['metrics']['lpips']['familywise_ci'][1], 0)
        self.assertFalse(result['candidate_improves'])
        self.assertFalse(result['reference_improves'])

    def test_missing_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'paired'):
            paired_contrast({('a', 1): metrics()}, {('a', 2): metrics()}, {'a': 'one'}, self.statistics)

    def test_exact_psnr_does_not_drop_other_metrics(self):
        result = paired_contrast({('a', 1): metrics(lpips=.1, psnr=None)},
                                 {('a', 1): metrics()}, {'a': 'one'}, self.statistics)
        self.assertIsNone(result['metrics']['psnr_db']['ci95'])
        self.assertTrue(result['candidate_improves'])

    def test_metric_reuse_retains_provenance_and_rejects_changed_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / 'old', Path(folder) / 'new'
            case = dict(case=dict(id='case'))
            write(source / 'manifest.json', dict(cases=[case], original=True))
            write(target / 'manifest.json', dict(cases=[case], original=False))
            identity = dict(manifest_sha256=sha(source / 'manifest.json'),
                source_sha256=sha(HERE / 'dataset_metrics.py'),
                quality_sha256=sha(REPO / 'packages/fp4-repair/src/thunderworld_fp4/cosmos/quality.py'))
            write(source / 'metrics/progress.json', dict(complete=True, identity=identity))
            for root in (source, target):
                for name in ('bf16', 'fp8'):
                    write(root / 'generation' / name / 'case/result.json', dict(method=name, source=str(root)))
            original = dict(complete=True, rows=[dict(seed=1, metrics=metrics())], provenance=dict(identity=identity,
                reference_sha256=sha(source / 'generation/bf16/case/result.json'),
                candidate_sha256=sha(source / 'generation/fp8/case/result.json')))
            write(source / 'metrics/fp8/case.json', original)
            import_metrics(source, target, 'fp8')
            imported = json.loads((target / 'metrics/fp8/case.json').read_text())
            self.assertEqual(imported['rows'], original['rows'])
            self.assertEqual(imported['imported_from']['provenance'], original['provenance'])
            self.assertEqual(imported['provenance']['identity']['manifest_sha256'], sha(target / 'manifest.json'))
            path = source / 'generation/bf16/case/result.json'
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'Historical metrics'):
                import_metrics(source, target, 'fp8')


if __name__ == '__main__':
    unittest.main()
