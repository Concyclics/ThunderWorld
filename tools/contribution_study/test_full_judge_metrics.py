"""Regression tests for pairing, infinity handling, and clustered weighting."""
import io
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from full_judge_metrics import CANDIDATES, completed_result, sources, worker
from full_judge_metrics_report import bootstrap, interval, report
from full_video_metrics import load_video, validate_video
from baselines import write, sha


class MetricTests(unittest.TestCase):
    def test_hash_checks_file_and_signal(self):
        with tempfile.TemporaryDirectory() as temp:
            video = np.zeros((2,8,9,3),np.float32)
            p=Path(temp)/'v.npz'; np.savez_compressed(p,video=video)
            r=dict(path=str(p), npz_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                video_sha256=hashlib.sha256(video.tobytes()).hexdigest())
            np.testing.assert_array_equal(load_video(r,video.shape),video)
            with self.assertRaisesRegex(ValueError,'Decoded video'):
                load_video(dict(r,video_sha256='wrong'),video.shape)
            p.write_bytes(p.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'NPZ hash'):
                load_video(r,video.shape)

    def test_range_rejects_silent_normalization(self):
        for value in (-.1,1.1,np.nan,np.inf):
            with self.assertRaises(ValueError): validate_video(np.full((1,8,8,3),value,np.float32))

    def test_cluster_weighting_and_pairing(self):
        # Category A has one scene, B three. Candidate 1 improves each scene by 2.
        a=np.array([[[1.],[3.]]]); b=np.array([[[5.],[7.]]]*3)
        macro,micro=bootstrap({'A':a,'B':b},1000,72)
        np.testing.assert_array_equal(macro[:,0,0],np.full(1000,3.))
        np.testing.assert_array_equal(micro[:,0,0],np.full(1000,4.))
        np.testing.assert_array_equal(macro[:,1]-macro[:,0],np.full((1000,1),2.))

    def test_scene_resampling_is_joint_not_independent(self):
        x=np.array([[[1.],[11.]],[[8.],[18.]],[[20.],[30.]]])
        macro,_=bootstrap({'A':x},1000,66)
        np.testing.assert_allclose(macro[:,1]-macro[:,0],10,atol=1e-12)
        self.assertGreater(macro[:,0].std(),0)

    def test_infinite_psnr_no_silent_dropping(self):
        x=np.array([[[np.inf]],[[20.]]])
        macro,_=bootstrap({'A':x},1000,22)
        self.assertFalse(np.isnan(macro).any())
        self.assertGreater(np.isposinf(macro).sum(),0)
        self.assertEqual(interval(macro[:,0,0])[1],None)

    def test_partial_resume_rejects_identity_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); item=dict(result='r.json',sha256='task')
            p=root/'r.json'; p.write_text(json.dumps(dict(identity={'manifest':'x'},task_sha256='task',complete=False,rows=[])))
            self.assertIsNone(completed_result(root,item,{'manifest':'x'}))
            with self.assertRaises(ValueError): completed_result(root,item,{'manifest':'changed'})

    def test_complete_report_and_partial_coverage_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); tasks=[]; pending=[]
            for model in ('edge','nano'):
                for benchmark in ('rbench','paibench'):
                    for ci in range(2):
                        cid=f'{benchmark}_{ci}'; seeds=[10,20]
                        records={m:{str(s):dict(path='synthetic-fixture.npz',video_path='synthetic-fixture.mp4',
                            npz_sha256=m+str(s)) for s in seeds} for m in ['bf16',*CANDIDATES]}
                        t=dict(model=model,benchmark=benchmark,category='cat'+str(ci),case_id=cid,source_id=cid,
                            seeds=seeds,case=dict(frames=61),records=records)
                        p=root/f'tasks/{model}/{cid}.json';write(p,t)
                        item=dict(path=str(p.relative_to(root)),result=f'results/{model}/{cid}.json',sha256=sha(p))
                        tasks.append(item);rows=[]
                        for mi,m in enumerate(CANDIDATES):
                            for s in seeds:
                                value=.01*(mi+1)+.001*ci
                                metrics=dict(lpips=value,psnr_db=30-mi,ssim_mean=1-value,relative_l2=value,
                                    rmse=value,mse=value**2,psnr_exact=False)
                                rows.append(dict(method=m,seed=s,sample_id=m+cid+str(s),reference_sample_id='bf16'+cid+str(s),
                                    metrics={'video':metrics},per_frame=dict(lpips=[value]*61,ssim=[1-value]*61),
                                    candidate_npz_sha256=m+str(s),reference_npz_sha256='bf16'+str(s),decoded_hashes_verified=True))
                        pending.append((item,dict(complete=True,rows=rows,task_sha256=item['sha256'])))
            manifest=dict(tasks=tasks,source_sha256=sources(),expected_pairs=128,models={m:dict(statistics=dict(
                replicates=1000,bootstrap_seed=92,family_size=12)) for m in ('edge','nano')})
            write(root/'manifest.json',manifest);identity={'manifest_sha256':sha(root/'manifest.json')}
            for item,value in pending: write(root/item['result'],dict(identity=identity,**value))
            report(root)
            result=json.loads((root/'report/summary.json').read_text())
            self.assertEqual(result['paired_videos'],128)
            self.assertEqual(len(result['groups']),4)
            self.assertEqual(len(result['paired_comparisons']),48)
            table=result['groups']['edge/rbench']['summaries']['category_macro']['fp8']
            self.assertAlmostEqual(table['lpips'],.0105)
            (root/tasks[-1]['result']).unlink()
            with self.assertRaisesRegex(ValueError,'partial coverage'): report(root)

    def test_worker_resumes_partial_without_recomputing_saved_pairs(self):
        class StubMetrics:
            calls=0
            def set_reference(self,reference): pass
            def compare(self,candidate):
                self.calls+=1
                return dict(video={'lpips':0.},per_frame={'lpips':[0.]*61,'ssim':[1.]*61})
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            item=dict(path='task.json',result='results/c.json',sha256='task',pairs=16)
            manifest=dict(tasks=[item],environment={})
            write(root/'manifest.json',manifest);identity={'manifest_sha256':sha(root/'manifest.json')}
            records={m:{str(s):dict(npz_sha256=m+str(s),sample_id=m+str(s)) for s in [10,20]}
                for m in ['bf16',*CANDIDATES]}
            task=dict(model='edge',benchmark='rbench',category='cat',case_id='c',source_id='c',
                prior_scene_roles={},seeds=[10,20],records=records)
            saved=[dict(method=m,seed=10,checkpoint_preserved=True) for m in CANDIDATES[:5]]
            write(root/item['result'],dict(identity=identity,task_sha256='task',complete=False,rows=saved.copy()))
            evaluator=StubMetrics();args=SimpleNamespace(root=root,gpu=0,global_fd=1,device_fd=2,slot=0,slots=1)
            with patch('full_judge_metrics.verify',return_value=manifest), \
                 patch('full_judge_metrics.load_task',return_value=task), \
                 patch('full_video_metrics.environment',return_value={}), \
                 patch('full_video_metrics.VideoMetrics',return_value=evaluator), \
                 patch('full_video_metrics.load_video',return_value=None), \
                 patch('judge_concurrent_infer.verify_descriptors'):
                worker(args)
                self.assertEqual(evaluator.calls,11)
                result=json.loads((root/item['result']).read_text())
                self.assertEqual(result['rows'][:5],saved)
                self.assertTrue(result['complete'])
                state=json.loads((root/'workers/slot0.json').read_text())
                self.assertEqual(state['completed_pairs'],16)
                self.assertEqual(state['resumed_pairs'],5)
                worker(args)
                self.assertEqual(evaluator.calls,11)


if __name__=='__main__': unittest.main()
