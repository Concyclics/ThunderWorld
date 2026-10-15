"""Match actual work, including mixed/tail tiles and head/step ablations."""
import numpy as np
import pytest

from structure_repair import candidates, mask_cost, random_mask, ranked_mask, role_counts, tile_scores, window_mask


@pytest.mark.parametrize('m,n,h,hk', [(129,257,4,2),(416,547,16,8),(960,1002,16,8)])
@pytest.mark.parametrize('coverage', [0,.25,.5,.75,1])
def test_actual_mma_matched(m,n,h,hk,coverage):
    qr=np.full(m,2);qr[-16:]=4;kr=np.concatenate([np.zeros(n-m,dtype=int),qr])
    prior=np.random.default_rng(1).random((4,2,h,6,6))
    masks=candidates(prior,2,1,qr,kr,coverage,4)
    costs=[mask_cost(x,m,n,hk) for x in masks.values()]
    assert len({c['total_mma'] for c in costs})==1
    assert all(c['mma_by_dtype']['bf16']['selector']==0 for c in costs)
    assert all(np.array_equal(x,masks[k]) for k,x in candidates(prior,2,1,qr,kr,coverage,4).items())
    if coverage in (0,1):assert all(np.all(x==bool(coverage)) for x in masks.values())


def test_mixed_tiles_preserved_and_directional():
    qr=np.array([2]*120+[4]*9);kr=np.array([0]*7+[2]*120+[4]*9)
    p=np.zeros((2,6,6));p[:,2,0]=1.;p[:,4,2]=2.
    counts=role_counts(qr)
    assert counts[0].tolist()==[0,0,120,0,8,0]
    score=tile_scores(p,qr,kr)
    assert score.shape==(2,2,2)
    assert not np.array_equal(score,tile_scores(p.transpose(0,2,1),qr,kr))


def test_window_respects_ar_prefix():
    mask=window_mask((2,2,4),.25,129,385)
    assert np.where(mask[0,0])[0].tolist()==[2]
    assert np.where(mask[0,1])[0].tolist()==[3]


def test_random_ensemble_and_global_rng():
    np.random.seed(17);expected=np.random.random(4)
    np.random.seed(17)
    a=random_mask((4,4,8),.5,31);b=random_mask((4,4,8),.5,32)
    assert not np.array_equal(a,b)
    assert np.array_equal(np.random.random(4),expected)
    assert np.array_equal(a,ranked_mask(np.zeros((4,4,8)),.5,31))


def test_paired_intervals_preserve_request_pairs():
    from structure_repair_study import paired
    base=np.array([[1.,2.,3.,4.],[10.,20.,30.,40.]])
    result=paired(base*.8,base,[1,2,3,4],replicates=200)
    assert result['request_wins']==8
    assert result['requests']==8
    np.testing.assert_allclose(result['reduction_ci95_pct'],[20.,20.],atol=1e-10)
    assert result['difference_ci_familywise95'][1]<0
