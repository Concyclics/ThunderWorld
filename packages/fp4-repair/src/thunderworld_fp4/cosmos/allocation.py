"""Compute-constrained discrete allocation over measured local combinations.

Local scores are proposals for full trajectories. No sum of isolated repair-block
gains is used as an operator's combination quality. Both growth and pruning use
fresh differences between measured combination objectives.
"""
from fractions import Fraction


def nondominated(choices):
    return [c for c in choices if not any(
        d['mma']<=c['mma'] and d['loss']<=c['loss'] and (d['mma']<c['mma'] or d['loss']<c['loss'])
        for d in choices)]


def allocate(groups,budget,*,direction='grow',preference=None,rounds=3):
    if direction not in ('grow','prune') or not 1<=rounds<=3:raise ValueError('Invalid search controls')
    def eligible(c):
        if preference=='a':return not c.get('w_blocks') and not any(c.get('qk_counts',())) and not any(c.get('pv_counts',()))
        if preference=='w':return not c.get('a_blocks',0) and not any(c.get('qk_counts',())) and not any(c.get('pv_counts',()))
        return True
    groups={k:nondominated([c for c in v if eligible(c)]) for k,v in groups.items()}
    if any(not v for v in groups.values()):raise ValueError('Empty candidate group')
    base=sum(min(c['base_mma'] for c in v) for v in groups.values())
    cap=Fraction(str(budget))*base
    chosen={k:min(v,key=(lambda c:(c['mma'],c['loss'])) if direction=='grow' else (lambda c:(c['loss'],c['mma']))) for k,v in groups.items()}
    def total():return sum(c['mma'] for c in chosen.values())
    if direction=='prune':
        while total()>cap:
            moves=[]
            for k,old in chosen.items():
                for c in groups[k]:
                    saved=old['mma']-c['mma']
                    if saved>0:moves.append(((c['loss']-old['loss'])/saved,k,c))
            if not moves:raise ValueError('Mandatory supports exceed budget')
            _,key,new=min(moves,key=lambda x:(x[0],x[1],x[2]['id']));chosen[key]=new
    for _ in range(rounds):
        changed=False
        while True:
            cost=total();moves=[]
            for k,old in chosen.items():
                for c in groups[k]:
                    if not eligible(c):continue
                    extra=c['mma']-old['mma'];gain=old['loss']-c['loss']
                    if gain>0 and cost+extra<=cap:
                        moves.append((gain/max(extra,1),gain,k,c))
            if not moves:break
            _,_,key,new=max(moves,key=lambda x:(x[0],x[1],x[2],-x[3]['id']));chosen[key]=new;changed=True
        # Coordinate withdrawal/reallocation, bounded to 3 passes. Evaluated scores
        # belong to whole local A/W or QK/PV combinations, not isolated gains.
        if not changed:break
    result=dict(base_mma=base,total_mma=total(),ratio=total()/base,budget=budget,
                proxy_loss=sum(c['loss'] for c in chosen.values()),assignments=chosen,
                direction=direction,preference=preference,objective_scope='Measured operator combinations; full trajectory gate required')
    if total()>cap:raise ValueError('Allocator violated budget')
    return result
