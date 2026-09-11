"""Reject per-task schedules that regress against a same-budget reference.

This is validation-based selection, not an independent final quality test.
Tolerances are provisional engineering settings, not task-success criteria.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

PARENTS={'auto50':('ffn_act2','act2'),'auto75':('ffn_uniform75','a50w')}


def regressions(candidate,reference,lpips_candidate,lpips_reference,wave_candidate,wave_reference,
                ssim_drop=.02,lpips_increase=.01,error_ratio=1.05,error_slack=.001):
    values=[candidate['video']['ssim_mean'],reference['video']['ssim_mean'],lpips_candidate,lpips_reference]
    for modality in ['action','sound_latent']:
        if modality in reference:values.extend([candidate[modality]['relative_l2'],reference[modality]['relative_l2']])
    if wave_reference:
        for metric in ['relative_l2','log_spectral_l1']:values.extend([wave_candidate[metric],wave_reference[metric]])
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite quality metric')
    reasons=[]
    if candidate['video']['ssim_mean']<reference['video']['ssim_mean']-ssim_drop:reasons.append('video_ssim')
    if lpips_candidate>lpips_reference+lpips_increase:reasons.append('video_lpips')
    for modality in ['action','sound_latent']:
        if modality in reference:
            if candidate[modality]['relative_l2']>reference[modality]['relative_l2']*error_ratio+error_slack:
                reasons.append(modality)
    if wave_reference:
        for metric in ['relative_l2','log_spectral_l1']:
            if wave_candidate[metric]>wave_reference[metric]*error_ratio+error_slack:reasons.append('waveform_'+metric)
    return reasons


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ssim-drop',type=float,default=.02);p.add_argument('--lpips-increase',type=float,default=.01)
    p.add_argument('--error-ratio',type=float,default=1.05);p.add_argument('--error-slack',type=float,default=.001)
    a=p.parse_args();r=json.loads((a.run/'result.json').read_text());assert r['complete']
    perc=json.loads((a.run/'perceptual.json').read_text());assert perc['complete']
    def key(x):return x['mode'],x['case'],x['seed'],x['repeat']
    rows={key(x):x for x in r['rows']};lpips={key(x):x['vs_bf16']['mean'] for x in perc['rows']}
    audio_path=a.run/'audio_decode.json';audio=json.loads(audio_path.read_text()) if audio_path.exists() else {'complete':True,'rows':[]}
    assert audio['complete'];waves={key(x):x.get('vs_bf16',{}) for x in audio['rows']}
    policy=r['budget_policy'];audit=[];case_tasks={c['id']:c['task'] for c in r['cases']}
    for mode,(parent,tier) in PARENTS.items():
        pol=policy['policies'][mode]
        for task in pol['by_task']:
            candidates=[x for x in r['rows'] if x['mode']=='ffn_'+mode and case_tasks[x['case']]==task]
            assert len({x['seed'] for x in candidates})>=2,'Need at least two validation seeds per task'
            failures=[]
            for candidate in candidates:
                ck=key(candidate);pk=(parent,*ck[1:]);ref=rows[pk]
                if 'sound_latent' in ref['vs_bf16']:assert ck in waves and pk in waves,'Decode audio before gating AV'
                reasons=regressions(candidate['vs_bf16'],ref['vs_bf16'],lpips[ck],lpips[pk],waves.get(ck),waves.get(pk),
                    a.ssim_drop,a.lpips_increase,a.error_ratio,a.error_slack)
                if reasons:failures.append(dict(case=ck[1],seed=ck[2],repeat=ck[3],metrics=reasons))
            audit.append(dict(mode=mode,task=task,validation_pairs=len(candidates),accepted=not failures,
                              fallback=None if not failures else parent,failures=failures))
            if failures:
                units=[u for u in policy['units'] if u['task']==task];current=pol['by_task'][task]
                assignment={u['key']:tier for u in units};current['assignments']=assignment
                pol['assignments'].update(assignment)
                current['extra_flops']=sum(u['options'][tier]['extra']*u['weight'] for u in units)
                current['realized_fraction']=current['extra_flops']/sum(2*u['weight'] for u in units)
                current['predicted_loss']=sum(u['options'][tier]['loss']*u['importance'] for u in units)/sum(u['importance'] for u in units)
                current['predicted_ms']=sum(u['options'][tier]['ms']*u['importance'] for u in units)
                assert current['extra_flops']<=current['extra_flop_cap']+1e-5
    policy['validation_gate']=dict(scope='Selected using these validation seeds; requires independent final evaluation. Passing means within provisional tolerances of the uniform reference, not absolute quality or task success.',
        tolerances=dict(ssim_drop=a.ssim_drop,lpips_increase=a.lpips_increase,error_ratio=a.error_ratio,error_slack=a.error_slack),
        source_hashes={name:hashlib.sha256((a.run/name).read_bytes()).hexdigest() for name in ['result.json','perceptual.json','audio_decode.json'] if (a.run/name).exists()},decisions=audit)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(policy,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
