import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from thunderworld_fp4.cosmos.quality import video_psnr
from dataset_subset import stratified
from dataset_report import stratified_interval,average
from dataset_report import main as report_main
from baselines import sha,write


class DatasetTests(unittest.TestCase):
    def test_known_psnr_and_exact(self):
        x=np.full((2,8,8,3),.1);ref=np.zeros_like(x)
        self.assertAlmostEqual(video_psnr(x,ref)['psnr_db'],20.)
        exact=video_psnr(ref,ref)
        self.assertIsNone(exact['psnr_db']);self.assertTrue(exact['psnr_exact'])
        self.assertIsNone(average([20.,None]))

    def test_clip_mse_is_not_mean_frame_psnr(self):
        ref=np.zeros((2,8,8,3));x=ref.copy();x[0]=.1;x[1]=.2
        self.assertAlmostEqual(video_psnr(x,ref)['psnr_db'],-10*np.log10(.025))
        self.assertNotAlmostEqual(video_psnr(x,ref)['psnr_db'],(20-20*np.log10(.2))/2)

    def test_invalid_psnr(self):
        for x in (np.array([1.1]),np.array([np.nan]),np.array([])):
            with self.assertRaises(ValueError):video_psnr(x,np.zeros_like(x))
        with self.assertRaises(ValueError):video_psnr(np.zeros(2),np.zeros(3))

    def test_sampling_is_stratified_reproducible(self):
        groups={str(k):[dict(source_id=str(i)) for i in range(20+k)] for k in range(7)}
        selected=stratified(groups,36)
        self.assertEqual(selected,stratified(dict(reversed(list(groups.items()))),36))
        counts=[sum(c==key for c,_ in selected) for key in groups]
        self.assertEqual(sorted(counts),[5,5,5,5,5,5,6])

    def test_bootstrap_has_scene_units_and_category_weights(self):
        r=stratified_interval({'small':[-.1]*4,'large':[-.3]*20},replicates=100)
        self.assertAlmostEqual(r['mean_delta'],-.2)
        self.assertEqual(r['n_scenes'],24)
        np.testing.assert_allclose(r['ci95'],[-.2,-.2])
        r=stratified_interval({'a':[-1.,0.,1.],'b':[-2.,0.]})
        self.assertLessEqual(r['familywise_ci'][0],r['ci95'][0])
        self.assertGreaterEqual(r['familywise_ci'][1],r['ci95'][1])

    def test_report_retains_seed_metrics_and_rejects_changed_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);generation=root/'generation';metrics=root/'metrics';output=root/'report'
            cases=[dict(benchmark=b,category='category',case=dict(id=b),seeds=[1,2]) for b in ('rbench','paibench')]
            manifest=root/'manifest.json'
            write(manifest,dict(cases=cases,methods={n:dict(config={}) for n in ('bf16','fp8','fp4')},
                statistics=dict(replicates=100,bootstrap_seed=1,family_size=14),scope='test'))
            identity=dict(manifest_sha256=sha(manifest),source_sha256='metric-source',quality_sha256='quality-source')
            write(metrics/'progress.json',dict(complete=True,identity=identity))
            for case in cases:
                name=case['case']['id']
                for method in ('bf16','fp8','fp4'):
                    receipt=generation/method/name/'result.json'
                    write(receipt,dict(complete=True,rows=[dict(peak_allocated_bytes=100,peak_reserved_bytes=200,actual_mma_ratio=None)]))
                    if method=='bf16':continue
                    video=dict(lpips=.1 if method=='fp8' else .05,ssim_mean=.9,relative_l2=.1,psnr_db=20.,rmse=.1,psnr_exact=False)
                    write(metrics/method/(name+'.json'),dict(complete=True,rows=[dict(seed=s,metrics=dict(video=video)) for s in (1,2)],
                        provenance=dict(identity=identity,candidate_sha256=sha(receipt),reference_sha256=sha(generation/'bf16'/name/'result.json'))))
            argv=['dataset_report',str(manifest),'--metrics',str(metrics),'--generation',str(generation),'--output',str(output)]
            with patch('sys.argv',argv):report_main()
            result=json.loads((output/'quality.json').read_text())
            self.assertEqual(sum(len(x['rows']) for x in result['seed_metrics']),8)
            self.assertTrue(all(x['exceeds_fp8'] for x in result['contrasts']))
            changed=generation/'bf16'/'rbench'/'result.json'
            changed.write_text(changed.read_text()+'\n')
            with patch('sys.argv',argv),self.assertRaisesRegex(ValueError,'Changed BF16 reference'):report_main()


if __name__=='__main__':unittest.main()
