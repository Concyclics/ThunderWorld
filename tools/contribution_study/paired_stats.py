"""Paired inference and conservative family-wise intervals, no post-test selection."""
import numpy as np


def paired(candidate,control,*,family_size=1,replicates=10000,seed=170917,lower_is_better=True):
    if type(family_size) is not int or family_size<1:raise ValueError('Invalid hypothesis family')
    if set(candidate)!=set(control) or not candidate:raise ValueError('Exactly paired nonempty seeds required')
    seeds=sorted(candidate)
    delta=np.array([candidate[s]-control[s] for s in seeds],dtype=float)
    if not np.isfinite(delta).all():raise ValueError('Nonfinite metric')
    samples=np.random.default_rng(seed).choice(delta,size=(replicates,len(delta))).mean(1)
    alpha=.05/family_size
    worst=int(delta.argmax() if lower_is_better else delta.argmin())
    return dict(seeds=seeds,n=len(seeds),mean_delta=float(delta.mean()),
                ci95=np.quantile(samples,[.025,.975]).tolist(),
                familywise_ci=np.quantile(samples,[alpha/2,1-alpha/2]).tolist(),
                family_size=family_size,correction='Bonferroni bootstrap intervals',
                worst_seed=seeds[worst],worst_delta=float(delta[worst]),lower_is_better=lower_is_better)
