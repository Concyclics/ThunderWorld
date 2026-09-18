"""Offline exact integer quota allocation over measured per-head combinations."""
import math


def allocate_head_quotas(losses,total):
    if not losses or any(len(r)!=len(losses[0]) for r in losses):raise ValueError('Rectangular head/quota table required')
    if any(not math.isfinite(float(v)) for r in losses for v in r):raise ValueError('Nonfinite loss')
    maximum=len(losses[0])-1
    if type(total) is not int or not 0<=total<=len(losses)*maximum:raise ValueError('Infeasible integer quota')
    previous={0:(0.,())}
    for row in losses:
        current={}
        for used,(loss,quotas) in previous.items():
            for quota,value in enumerate(row):
                if used+quota>total:continue
                proposal=(loss+float(value),quotas+(quota,))
                if used+quota not in current or proposal<current[used+quota]:current[used+quota]=proposal
        previous=current
    return previous[total][1]
