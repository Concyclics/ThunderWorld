"""CPU invariants for numerical and storage mechanisms."""
import unittest
import torch
from representation import planes,removal_masks,storage_union
from paired_stats import paired
from head_allocation import allocate_head_quotas
from thunderworld_fp4.cosmos.calibration import decode
from thunderworld_fp4.cosmos.assets import expand


class StudyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_residual_is_of_represented_base(self):
        w=torch.randn(128,128).bfloat16()
        p=planes(w)
        decoded=decode(p['codes'][0],p['scales'][0],p['factor'])
        self.assertTrue(torch.equal(decoded,p['base']))
        self.assertLess((w.float()-p['base']-p['residual']).square().mean(),
                        (w.float()-p['base']).square().mean())

    def test_zero_and_tail_rows(self):
        for m in (129,416,960):
            for rule in ('rtn','four_six'):
                p=planes(torch.zeros(m,128),rule)
                self.assertEqual(float(p['base'].abs().sum()+p['residual'].abs().sum()),0.)
                self.assertTrue(torch.isfinite(p['factor']))

    def test_scale_compensation(self):
        x=torch.randn(129,128,dtype=torch.float64)
        w=torch.randn(128,128,dtype=torch.float64)
        d=torch.rand(128,dtype=torch.float64)+.25
        torch.testing.assert_close((x*d)@(w/d).T,x@w.T)

    def test_outlier_reservation(self):
        w=torch.randn(128,256)
        masks=removal_masks(w,torch.ones(256),torch.tensor([[True,False]]))
        for mask in masks.values():
            self.assertFalse(mask[:,128:].any())
            self.assertLessEqual(int(mask.reshape(128,16,16).sum(-1).max()),1)

    def test_storage_union_and_mandatory(self):
        p=planes(torch.randn(128,256))
        asset={k:p[k] for k in ('codes','scales','factor')}
        asset.update(score=torch.ones(1,2),prescale=torch.ones(256),mandatory_w_blocks=(0,))
        packed,cost=storage_union(asset,[[0],[0]])
        self.assertEqual(cost['union_density'],.5)
        self.assertEqual(packed['codes_residual'].shape[0],1)
        restored=expand(packed,[0])
        torch.testing.assert_close(restored['codes'][1,:,:64],asset['codes'][1,:,:64])
        self.assertEqual(int(restored['codes'][1,:,64:].sum()),0)
        with self.assertRaises(ValueError):storage_union(asset,[[0],[]])
        with self.assertRaises(ValueError):expand(packed,[0,1])

    def test_paired_family_intervals(self):
        out=paired({i:float(i)/2 for i in range(16)},{i:float(i) for i in range(16)},family_size=4)
        self.assertLessEqual(out['familywise_ci'][0],out['ci95'][0])
        self.assertGreaterEqual(out['familywise_ci'][1],out['ci95'][1])
        with self.assertRaises(ValueError):paired({1:0},{2:0})

    def test_head_quota_global_combination(self):
        import itertools
        losses=[[8.,7.,0.],[8.,3.,2.],[4.,3.,2.]]
        answer=allocate_head_quotas(losses,3)
        oracle=min((sum(row[q] for row,q in zip(losses,qs)),qs)
                   for qs in itertools.product(range(3),repeat=3) if sum(qs)==3)
        self.assertEqual(answer,oracle[1])
        with self.assertRaises(ValueError):allocate_head_quotas(losses,7)

    def test_explicit_attribution_cannot_impersonate_baseline(self):
        from thunderworld_fp4.cosmos.settings import RunConfig
        with self.assertRaises(ValueError):RunConfig(precision='bf16',attention_backend='fp4')
        with self.assertRaises(ValueError):RunConfig(precision='fp8',attention_backend='fp4',attention_only_attribution=True)
        config=RunConfig(precision='bf16',attention_backend='fp4',attention_only_attribution=True,resource_mode='shared-calibration')
        self.assertTrue(config.attention_only_attribution)
        for budget in (1.25,1.5,2.,2.5):
            self.assertEqual(RunConfig(compute_budget=budget).compute_budget,budget)

    def test_final_admission_freezes_configuration(self):
        import json,tempfile
        from pathlib import Path
        from protocol import admit_run
        from baselines import sha
        with tempfile.TemporaryDirectory() as tmp:
            cohort=Path(tmp)/'cohort.json';selection=Path(tmp)/'selection.json'
            cohort.write_text(json.dumps(dict(seed_status='frozen',seeds=dict(validation=[1,2],final=[3,4]))))
            selection.write_text(json.dumps(dict(cohort_sha256=sha(cohort),methods=['candidate'],configuration_sha256={'candidate':'fixed'})))
            self.assertFalse(admit_run(cohort,'final',[3,4],'candidate',selection,'fixed')['selection_eligible'])
            with self.assertRaises(ValueError):admit_run(cohort,'final',[3,4],'candidate',selection,'changed')
            with self.assertRaises(ValueError):admit_run(cohort,'validation',[1],'candidate')
            with self.assertRaises(ValueError):admit_run(None,'trajectory',[181001],'candidate')

    def test_quality_gate_has_guardrails(self):
        from quality_gate import contrast
        rows=[dict(name=n,seed=s,metrics=dict(video=dict(lpips=.1 if n=='a' else .2,
            ssim_mean=.8 if n=='a' else .9,relative_l2=.1))) for n in ('a','b') for s in range(8)]
        result=contrast(dict(seeds=list(range(8)),rows=rows),'a','b',family_size=4)
        self.assertFalse(result['paired_gate_passed'])

    def test_failed_validation_cannot_be_rescued_by_final(self):
        from final_report import qualify
        selection=dict(decisions=[dict(selected='candidate',control='control',validation_guardrails=False)])
        gates=dict(contrasts=[dict(candidate='candidate',control=name,paired_gate_passed=True) for name in ('control','fp8')])
        result=qualify(selection,gates)[0]
        self.assertFalse(result['mechanism_accepted'])
        self.assertFalse(result['exceeds_fp8'])

    def test_portable_rebuild_and_tensor_hash(self):
        from portable import rebuild,tensor_hash
        from thunderworld_fp4.cosmos.calibration import quantize_weight
        from thunderworld_fp4.cosmos.assets import compact
        weight=torch.randn(128,256).bfloat16()
        prescale=torch.linspace(.5,2.,256)
        codes,scales,factor=quantize_weight(weight,prescale)
        source=compact(dict(codes=codes,scales=scales,factor=factor,prescale=prescale,
            score=torch.ones(1,2),weight_scale=(1.,1.),bias=None),[1])
        meta={k:v for k,v in source.items() if k not in ('codes_base','codes_residual','scales_base','scales_residual')}
        restored=rebuild(weight,meta)
        self.assertEqual({k:tensor_hash(v) for k,v in source.items() if isinstance(v,torch.Tensor)},
                         {k:tensor_hash(v) for k,v in restored.items() if isinstance(v,torch.Tensor)})
        self.assertNotEqual(tensor_hash(torch.zeros(0,dtype=torch.uint8)),tensor_hash(torch.zeros(0,dtype=torch.float32)))

    def test_device_retest_rejects_changed_outputs(self):
        from retest import verify_outputs
        result=dict(complete=True,rows=[dict(seed=1,sha256={'video':'fixed'})])
        verify_outputs(result,{'1':{'video':'fixed'}})
        with self.assertRaises(RuntimeError):verify_outputs(result,{'1':{'video':'changed'}})
        result['rows'].append(result['rows'][0])
        with self.assertRaises(RuntimeError):verify_outputs(result,{'1':{'video':'fixed'}})


if __name__=='__main__':unittest.main()
