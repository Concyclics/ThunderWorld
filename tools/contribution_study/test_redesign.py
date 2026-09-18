import unittest
from thunderworld_fp4.cosmos.budget import RepairPlan,PHASES
from thunderworld_fp4 import RepairConfig
from redesign_plan import rank_bundles,select_bundles


class RedesignTests(unittest.TestCase):
    def test_full_attention_tracks_current_tail(self):
        plan=RepairPlan(dict(schema_version=1,task='i2v',steps=35,phases=list(PHASES),budget=2.5,linears={},
            attention={'0|0|0':dict(full_attention=True,qk_terms=3,pv_terms=3)}))
        cfg=plan.attention_config('i2v',0,0,0,RepairConfig(coverage=.5,head_quotas=(1,2),pv_head_quotas=(2,1)))
        for n,nt in ((129,2),(416,4),(960,8),(1025,9)):
            self.assertEqual(cfg.head_counts(2,n),(nt,nt));self.assertEqual(cfg.pv_counts(2,n),(nt,nt))
        plan.attention['0|0|0']['qk_counts']=[1,1]
        with self.assertRaises(ValueError):plan.attention_config('i2v',0,0,0,RepairConfig())

    def test_weight_budget_and_storage_caps_are_both_enforced(self):
        shapes={'x':(129,256,128),'y':(416,128,256)}
        choices=rank_bundles({'x':[1.,.5],'y':[2.]},shapes)
        total=sum(x['mma'] for x in choices);first=choices[0]
        selected,mma,size=select_bundles(choices,total,first['bytes'])
        self.assertLessEqual(size,first['bytes']);self.assertLessEqual(mma,total)
        self.assertEqual(sum(map(len,selected.values())),1)
        selected,mma,size=select_bundles(choices,0,100000000)
        self.assertEqual((selected,mma,size),({},0,0))

    def test_k_bundles_charge_all_output_tiles(self):
        choice=rank_bundles({'x':[1.,2.]},{'x':(416,1024,256)},axis='k')[0]
        self.assertEqual(choice['bytes'],1024*128*9//16)
        self.assertEqual(choice['mma'],2*512*1024*128*70)
        with self.assertRaises(ValueError):rank_bundles({'x':[1.]},{'x':(416,1024,256)},axis='k')


if __name__=='__main__':unittest.main()
