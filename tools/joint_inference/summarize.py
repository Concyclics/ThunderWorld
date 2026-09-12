"""Summarize new measurements only; never multiply gains across cohorts."""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np
from provenance import provenance as run_provenance


def main():
    p=argparse.ArgumentParser();p.add_argument('matrix',type=Path);p.add_argument('--quality',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=json.loads(a.matrix.read_text());q=json.loads(a.quality.read_text()) if a.quality else None
    if not m['complete'] or m['suite']!='performance':raise ValueError('Completed performance suite required')
    rows={};environment=None;provenance={};rounds={};hashes={}
    for job in m['jobs']:
        r=json.loads((Path(job['output'])/'result.json').read_text())
        if not r['complete'] or r['tuning_missing']:raise ValueError('Incomplete/untuned result')
        if environment is not None and environment!=r['environment']:raise ValueError('Mixed device/kernel environments')
        environment=r['environment']
        key=(job['case'],job['config'],job['audio_output'])
        rounds.setdefault(key,set()).add(job['round'])
        cfg=dict(r['config']);cfg.pop('strict_tuning',None)
        provenance[job['case']+'|'+job['config']]=run_provenance(r)
        provenance[job['case']+'|'+job['config']]['config']=cfg
        for v in r['rows']:
            if v['debug']!='off' or not v['performance_eligible']:raise ValueError('Diagnostic timing cannot enter leaderboard')
            if v.get('timing_scope')!='complete_request_v2_including_host_setup':
                raise ValueError('Formal report requires the complete request boundary including host setup')
            hash_key=(job['case'],job['config'],job['audio_output'],v['seed'])
            if hash_key in hashes and hashes[hash_key]!=v['sha256']:raise ValueError('Nondeterministic repeated output')
            hashes[hash_key]=v['sha256']
            rows.setdefault((job['case'],job['config'],job['audio_output']),[]).append(v)
    for (case,config,audio,seed),value in hashes.items():
        if config.endswith('_legacy'):
            if hashes.get((case,config.removesuffix('_legacy'),audio,seed))!=value:
                raise ValueError('Legacy/optimized kernels changed complete output')
    summary=[]
    for (case,config,audio),values in sorted(rows.items()):
        samples=[v['wall_seconds'] for v in values]
        if len(samples)!=15 or rounds[case,config,audio]!=set(range(5)):
            raise ValueError('Formal report requires five rounds and three seeds per round')
        if any(v['steps']!=35 for v in values):raise ValueError('Formal report requires 35 steps')
        summary.append(dict(case=case,config=config,audio_output=audio,n=len(samples),
            median_seconds=statistics.median(samples),p90_seconds=float(np.quantile(samples,.9)),
            peak_allocated_bytes=max(v['peak_allocated_bytes'] for v in values),
            peak_reserved_bytes=max(v['peak_reserved_bytes'] for v in values)))
    recommendations={}
    if q:
        if not q['complete'] or q['suite']!='quality':raise ValueError('Final independent quality gate required')
        if q['model']!=m['model']:raise ValueError('Quality/performance model mismatch')
        for key,value in q['provenance'].items():
            if key not in provenance:continue
            other=dict(value);other['config']=dict(value['config']);other['config'].pop('strict_tuning',None)
            for field in ('asset_identity','ffn_policy_identity'):other.setdefault(field,None)
            if other!=provenance[key]:raise ValueError('Quality/performance configuration mismatch: '+key)
        passed={(d['case'],d['config']) for d in q['decisions'] if d['passed']}
        for case in sorted(set(x['case'] for x in summary)):
            choices=[x for x in summary if (case,x['config']) in passed and x['audio_output']=='waveform']
            recommendations[case]=min(choices,key=lambda x:x['median_seconds'])['config'] if choices else None
    a.output.write_text(json.dumps(dict(complete=True,rows=summary,recommendations=recommendations,environment=environment,
        scope='Same-device warm requests. Null recommendation means no demonstrated FP8-quality winner.'),indent=2)+'\n')


if __name__=='__main__':main()
