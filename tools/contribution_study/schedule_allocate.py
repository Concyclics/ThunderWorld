"""Bounded deterministic coordinate exchange under a full request-shape envelope."""
import numpy as np


def allocate(groups, caps, *, change_limit, attention_floor, rounds=3, initial=None):
    if not 1 <= rounds <= 3:raise ValueError('One to three coordinate rounds')
    keys=sorted(groups);choices=[];owners=[];baseline=[];attention=[]
    for i,key in enumerate(keys):
        options=groups[key]
        anchors=[j for j,c in enumerate(options) if c['label']=='base']
        if len(anchors)!=1:raise ValueError('Exactly one unchanged control per group')
        baseline.append(len(choices)+anchors[0])
        for row in options:
            if not np.isfinite(row['loss']):raise ValueError('Nonfinite candidate loss')
            owners.append(i);choices.append(row)
        attention.append(options[0]['kind']=='attention')
    owner=np.array(owners);mask=np.asarray(attention)[owner,None]
    costs=np.asarray([c['costs'] for c in choices],dtype=np.float64)
    losses=np.asarray([c['loss'] for c in choices],dtype=np.float64)
    baseline=np.asarray(baseline);selected=baseline.copy()
    oldcost=costs[baseline];motion=np.abs(costs-oldcost[owner]);acost=costs*mask
    caps=np.asarray(caps);change_limit=np.asarray(change_limit);attention_floor=np.asarray(attention_floor)
    if np.any(costs[baseline].sum(0)>caps):raise ValueError('Control exceeds prompt envelope')
    initial_accepted=False
    if initial:
        proposed=baseline.copy()
        for i,key in enumerate(keys):
            label=initial.get(key,'base')
            matches=[j for j,c in enumerate(choices) if owner[j]==i and c['label']==label]
            if len(matches)!=1:raise ValueError('Unknown initial choice')
            proposed[i]=matches[0]
        if (np.all(costs[proposed].sum(0)<=caps) and np.all(acost[proposed].sum(0)>=attention_floor)
                and np.all(motion[proposed].sum(0)<=change_limit)):
            selected=proposed;initial_accepted=True
    trace=[]

    def feasible_moves(selection):
        prior=selection[owner];total=costs[selection].sum(0);atotal=acost[selection].sum(0);mtotal=motion[selection].sum(0)
        delta=costs-costs[prior];adelta=acost-acost[prior];mdelta=motion-motion[prior]
        ok=(np.all(total+delta<=caps,axis=1)&np.all(atotal+adelta>=attention_floor,axis=1)
            &np.all(mtotal+mdelta<=change_limit,axis=1))
        return ok,losses[prior]-losses,delta

    def fill(selection):
        moves=0
        while True:
            ok,gain,delta=feasible_moves(selection);ok &= gain>1e-12
            if not ok.any():break
            normalized=np.maximum(delta,0).max(1)
            scores=np.where(ok,gain/np.maximum(normalized,1.),-np.inf)
            j=int(scores.argmax());selection[owner[j]]=j;moves+=1
        return moves

    for round_id in range(rounds):
        direct=fill(selected);start_loss=losses[selected].sum();best=None
        ok,gain,delta=feasible_moves(selected)
        savings=-delta.max(1)
        candidates=np.where(ok & (savings>0) & (gain<=1e-12))[0]
        # Bound expensive exchange trials: 24 least harmful withdrawals, not an
        # unbounded search through development scores. Every refill recomputes
        # whole measured group losses, never sums isolated tile gains.
        candidates=sorted(candidates,key=lambda j:(-gain[j]/savings[j],j))[:24]
        for j in candidates:
            trial=selected.copy();trial[owner[j]]=j;fill(trial)
            loss=losses[trial].sum()
            if loss < start_loss-1e-12 and (best is None or loss<best[0]):best=(loss,trial,j)
        trace.append(dict(round=round_id,direct_moves=direct,proxy_loss=float(start_loss),accepted_exchange=best is not None))
        if best is None:break
        selected=best[1]
    total=costs[selected].sum(0)
    if np.any(total>caps) or np.any(acost[selected].sum(0)<attention_floor) or np.any(motion[selected].sum(0)>change_limit):
        raise AssertionError('Allocation constraint failed')
    return dict(assignments={key:choices[int(selected[i])] for i,key in enumerate(keys)},trace=trace,
                initial_accepted=initial_accepted,
                proxy_loss=float(losses[selected].sum()),control_proxy_loss=float(losses[baseline].sum()),
                totals=[int(x) for x in total],attention_totals=[int(x) for x in acost[selected].sum(0)],
                cost_motion=[int(x) for x in motion[selected].sum(0)])
