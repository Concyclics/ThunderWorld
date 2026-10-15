"""CPU protocol tests: reject mislabeled paths, mismatched shapes and probes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from standard_comparison import checked_run, command, METHODS
from section54_protocol import statistics_rows
from section54_profiles import category


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);fixed=self.root/'fixed.json'
        fixed.write_text(json.dumps(dict(targets=['x'],dimensions={'x':[1024,4096]})))
        self.m=dict(fixed_manifest=str(fixed),fixed_manifest_sha256='fixed',methods=METHODS,
                    cases=[0,1],seeds=[1,2,3],variant='/v6',variant_receipt_sha256='v6')
        self.d=dict(complete=True,operation='measure',method='arcquant',fixed_manifest_sha256='fixed',
                    variant=None,effective_config=dict(precision='bf16',attention_backend='bf16'),
                    adapter=dict(no_fallback=True,gemm_dtype='FP4_tensor_core',execution='eager',
                                 targets=['x'],dimensions={'x':[1024,4096]}),
                    rows=[dict(case_index=c,seed=s,debug='off',execution='eager',profiled=False,
                               steps=35,calls=70) for c in [0,1] for s in [1,2,3]])

    def check(self):
        p=self.root/'result.json';p.write_text(json.dumps(self.d))
        return checked_run(p,self.m,'arcquant50')

    def test_native_eager_adapter_with_bf16_placeholder_is_accepted(self):self.check()
    def test_hidden_bf16_gemm_is_rejected(self):
        self.d['adapter']['gemm_dtype']='bf16'
        with self.assertRaises(ValueError):self.check()
    def test_target_shape_mismatch_is_rejected(self):
        self.d['adapter']['dimensions']['x']=[1024,2048]
        with self.assertRaises(ValueError):self.check()
    def test_instrumented_timing_rejected(self):
        self.d['rows'][0]['profiled']=True
        with self.assertRaises(ValueError):self.check()
    def test_external_worker_does_not_get_fp4_attention_override(self):
        cmd=command(SimpleNamespace(root=self.root),self.m,'arcquant50',self.root/'out','profile',1,301031)
        self.assertNotIn('--attention-backend',cmd)
        self.assertEqual(cmd[cmd.index('--cases')+1],'1')
        self.assertEqual(cmd[-1],'301031')
    def test_new_frames_and_linear_p90(self):
        rows=[dict(case='scene',round=i,wall_seconds=t,peak_allocated_bytes=100,peak_reserved_bytes=200)
              for i,t in enumerate([1.,2.,3.])]
        result=statistics_rows(rows,'fp4_base','edge','sha')[0]
        self.assertEqual(result['new_frame_fps_at_median'],30)
        self.assertAlmostEqual(result['p90_seconds'],2.8)
    def test_fused_kernels_not_split(self):
        self.assertEqual(category('streaming_attention'),'attention')
        self.assertEqual(category('calibrated_relu_amax'),'preparation_selection')
        self.assertEqual(category('combine'),'combine')


if __name__=='__main__':unittest.main()
