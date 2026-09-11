"""Deterministic benefit/cost greedy allocation; no claim of global optimality."""
import math


def allocate(units,fraction):
    if not 0<=fraction<=1:raise ValueError('fraction must be in [0,1]')
    cap=sum(u['weight']*2 for u in units)*fraction
    chosen={u['key']:'base' for u in units};used=0.;time_used=0.
    # Keep the same FLOP ceiling as the uniform comparator. Rank improvements
    # by measured incremental latency, with a small positive floor for noise.
    while True:
        best=None
        for u in units:
            old=u['options'][chosen[u['key']]]
            for tier,new in u['options'].items():
                delta=(new['extra']-old['extra'])*u['weight']
                gain=(old['loss']-new['loss'])*u['importance']
                if delta<0 or gain<=1e-12 or used+delta>cap+1e-6:continue
                dt=(new['ms']-old['ms'])*u['importance']
                score=gain/max(dt,.02*old['ms']*u['importance'],1e-9)
                candidate=(score,gain,-delta,u['key'],tier,delta,dt)
                if best is None or candidate[:3]>best[:3]:best=candidate
        if best is None:break
        _,_,_,key,tier,delta,dt=best;chosen[key]=tier;used+=delta;time_used+=dt
    assert used<=cap+1e-5
    return dict(assignments=chosen,fraction=fraction,extra_flops=used,extra_flop_cap=cap,
                realized_fraction=used/sum(u['weight']*2 for u in units),
                predicted_loss=sum(u['options'][chosen[u['key']]]['loss']*u['importance'] for u in units)/sum(u['importance'] for u in units),
                predicted_ms=sum(u['options'][chosen[u['key']]]['ms']*u['importance'] for u in units))
