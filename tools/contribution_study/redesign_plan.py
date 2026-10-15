"""Compile storage-capped, activation-protected RTN repair plans on calibration only.

W1 uses static N128/full-K or K128/full-N bundles across all phases/CFG branches.
Same per-operator N bundle counts give exact MMA/storage-matched position controls.
Full-attention plans account for every tile of the current prompt at preparation.
"""
import argparse,collections,copy,json,math
from pathlib import Path
import numpy as np
from baselines import sha,write
from thunderworld_fp4.cosmos.budget import PHASES,RepairPlan,linear_cost,attention_cost,compile_request


def rank_bundles(gains,shapes,axis='n'):
    """Calibration proposal only; full-trajectory validation decides acceptance."""
    if axis not in ('n','k'):raise ValueError('Expected N or K weight bundles')
    choices=[]
    for name,values in gains.items():
        m,n,k=shapes[name];width=k if axis=='n' else n;unit=2*math.ceil(m/128)*128*128*width*70
        if len(values)!=(n if axis=='n' else k)//128:raise ValueError('Gain table does not match the support axis')
        for block,value in enumerate(values):
            if not math.isfinite(value):raise ValueError('Nonfinite repair benefit')
            if value>0:choices.append(dict(name=name,nblock=block,gain=value,mma=unit,bytes=width*128*9//16))
    return sorted(choices,key=lambda r:(-r['gain']/r['mma'],r['name'],r['nblock']))


def select_bundles(choices,remaining_mma,remaining_bytes):
    selected=collections.defaultdict(list);mma=size=0
    for row in choices:
        if mma+row['mma']<=remaining_mma and size+row['bytes']<=remaining_bytes:
            selected[row['name']].append(row['nblock']);mma+=row['mma'];size+=row['bytes']
    return dict(selected),mma,size


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cohort','calibration','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--previous',type=Path,help='Historical candidate manifest; unnecessary with --skip-legacy-control')
    p.add_argument('--ffn-calibration',type=Path)
    p.add_argument('--weight-axis',choices=('n','k'),default='n')
    p.add_argument('--profile',choices=('protected','balanced'),default='protected')
    p.add_argument('--budgets',nargs='+',type=float)
    p.add_argument('--skip-legacy-control',action='store_true',help='New model without the old Edge-only candidate')
    a=p.parse_args()
    if not a.skip_legacy_control and a.previous is None:p.error('--previous is required unless --skip-legacy-control is set')
    cohort=json.loads(a.cohort.read_text());cm=json.loads((a.calibration/'manifest.json').read_text());old=json.loads(a.previous.read_text()) if a.previous else {}
    if not cm['complete'] or cm['identity']['cohort_sha256']!=sha(a.cohort):raise ValueError('Complete matching calibration required')
    if a.output.exists():raise FileExistsError('Plans are immutable')
    a.output.mkdir();am=json.loads((a.calibration/'rtn_assets/manifest.json').read_text())
    if sha(a.calibration/'rtn_assets/manifest.json')!=cm['assets_sha256']:raise ValueError('Changed RTN assets')
    modules={};attention={};samples=collections.defaultdict(list);head_samples=collections.defaultdict(list);activation_energy={}
    for entry in cohort['cases']['calibration']:
        case_id=entry['case']['id'];path=a.calibration/(case_id+'.json')
        if sha(path)!=cm['cases'][case_id]:raise ValueError('Changed calibration case')
        data=json.loads(path.read_text());dataset_weight=.5/(9 if entry['benchmark']=='rbench' else 7)
        for name,rows in data['linears'].items():
            shapes={tuple(row['shape']) for row in rows}
            if len(shapes)!=1:raise ValueError('One full-M shape per linear required')
            shape=next(iter(shapes))
            if name in modules and modules[name]!=shape:raise ValueError('Mixed linear calibration shapes')
            modules[name]=shape;gain=np.zeros(shape[1]//128)
            for row in rows:
                e=row['errors_by_n128'];calls=PHASES[row['phase']+1]-PHASES[row['phase']]
                gain+=(np.asarray(e['act2'])-np.asarray(e['three']))*calls/70
                if a.weight_axis=='k':
                    activation_energy.setdefault(name,np.zeros(shape[2]))
                    activation_energy[name]+=np.square(row['rms'])*calls/70*dataset_weight
            samples[name].append((gain,dataset_weight))
        for layer,rows in data['attention'].items():
            m,n,h,hk=rows[0]['shape'];attention[layer]=(m,2048,h,hk)
            # 2048 is an explicit preparation envelope, not a changed generation input.
            if any(row['shape'][1]>2048 for row in rows):raise ValueError('Calibration exceeds planned attention envelope')
            for row in rows:head_samples[layer,row['phase'],row['branch']].append((row,dataset_weight))
    gains={name:sum(g*w for g,w in records) for name,records in samples.items()}
    if a.ffn_calibration:
        fm=json.loads((a.ffn_calibration/'manifest.json').read_text())
        if not fm['complete'] or fm['identity']['cohort_sha256']!=sha(a.cohort):raise ValueError('Complete matching full-FFN calibration required')
        ffn_gain=collections.Counter()
        for entry in cohort['cases']['calibration']:
            file=a.ffn_calibration/(entry['case']['id']+'.json')
            if sha(file)!=fm['cases'][entry['case']['id']]:raise ValueError('Changed FFN interventions')
            data=json.loads(file.read_text());weight=.5/(9 if entry['benchmark']=='rbench' else 7)
            for name,rows in data['modules'].items():
                for row in rows:
                    calls=PHASES[row['phase']+1]-PHASES[row['phase']];e=row['errors']
                    for side in ('up','down','gate'):
                        if 'without_'+side+'_w' in e:ffn_gain[name+'.'+side+'_proj']+=(e['without_'+side+'_w']-e['three'])*calls/70*weight
        for name,total in ffn_gain.items():
            local=np.maximum(gains[name],0.)
            gains[name]=(local/local.sum() if local.sum()>0 else np.ones_like(local)/len(local))*max(0,total)
        write(a.output/'ffn_block_gains.json',dict(complete=True,gains=dict(ffn_gain),scope='Full FFN output marginal gains rescale per-operator N proposals. Combined trajectory still requires evaluation.'))
    if a.weight_axis=='k':
        import torch
        torch.set_num_threads(8)
        for name,entry in am['modules'].items():
            asset=torch.load(a.calibration/'rtn_assets'/entry['file'],weights_only=True,map_location='cpu')
            proxy=asset['score'].double().sum(0).numpy()*activation_energy[name].reshape(-1,128).mean(1)
            total=max(0,float(gains[name].sum()))
            gains[name]=(proxy/proxy.sum() if proxy.sum()>0 else np.ones_like(proxy)/len(proxy))*total
    choices=rank_bundles(gains,modules,a.weight_axis);weight_full_bytes=sum(n*k*9//16 for m,n,k in modules.values())
    # Protect every linear A block at 2.5. Lower caps use nonzero A floors and
    # measured head-specific half/full quotas, never global all-or-nothing layer deletion.
    profiles=[(2.5,1.,.45,1.),(2.,.75,.15,1.),(1.5,.375,.05,.75),(1.25,.125,.03,.5)]
    if a.ffn_calibration:profiles=profiles[:1]
    if a.profile=='balanced':
        if not a.ffn_calibration:raise ValueError('Balanced followup requires full FFN calibration')
        profiles=[(2.5,.875,.625,.75)]
    if a.budgets:
        profiles=[row for row in profiles if row[0] in a.budgets]
        if {row[0] for row in profiles}!=set(a.budgets):raise ValueError('Unsupported profile budgets')
    import torch
    from thunderworld_fp4.cosmos.assets import expand,compact
    from thunderworld_fp4.cosmos.settings import RunConfig,LinearSpec
    torch.set_num_threads(8);candidates=[]
    def head_counts(layer,ph,branch,side,fraction):
        # Fixed half/full counts use the smallest observed tile count; current
        # shape and actual coverage are reported in every request ledger.
        samples_=head_samples[layer,ph,branch];nt=min(math.ceil(row['shape'][1]/128) for row,w in samples_);h=samples_[0][0]['shape'][2]
        counts=[nt//2]*h
        if fraction==.5:return counts
        benefit=np.zeros(h)
        for row,w in samples_:
            benefit+=(np.asarray(row['errors'][side+'half'])-np.asarray(row['errors']['three']))*w
        total=math.floor(h*nt*fraction);order=np.argsort(-benefit,kind='stable')
        for head in order:
            add=min(nt-counts[head],total-sum(counts));counts[head]+=add
        return list(map(int,counts))
    def build(cap,af,wf,hf):
        plan=dict(schema_version=1,task='i2v',steps=35,phases=list(PHASES),budget=cap,model_identity=cm['model_identity'],
            cohort_sha256=sha(a.cohort),linears={},attention={},design=dict(w1_axis='N128 with full K' if a.weight_axis=='n' else 'K128 with full N',w1_static_across_phase_cfg=True,
                a_floor_fraction=af,w1_density_cap=wf,attention_key_tokens_planned_max=2048,calibration_manifest_sha256=sha(a.calibration/'manifest.json')))
        for branch in range(2):
            for ph in range(4):
                for name,(m,n,k) in modules.items():plan['linears'][f'{branch}|{ph}|{name}']=dict(a_blocks=math.ceil(k//128*af),w_blocks=[],cross=False)
                for layer,(m,n,h,hk) in attention.items():
                    row=dict(qk_terms=3,pv_terms=3)
                    if hf==1:row['full_attention']=True
                    else:row.update(qk_counts=head_counts(layer,ph,branch,'qk',hf),pv_counts=head_counts(layer,ph,branch,'pv',hf))
                    plan['attention'][f'{branch}|{ph}|{layer}']=row
        return plan
    def ledger(plan):
        rows=[]
        for key,setting in plan['linears'].items():
            branch,ph,name=key.split('|',2);cost=linear_cost(*modules[name],**setting)
            rows.append(dict(cost,name=name,branch=int(branch),phase=int(ph)))
        for key,setting in plan['attention'].items():
            branch,ph,name=key.split('|',2);m,n,h,hk=attention[name]
            setting=dict(setting)
            if setting.pop('full_attention',False):setting.update(qk_counts=[math.ceil(n/128)]*h,pv_counts=[math.ceil(n/128)]*h)
            else:
                # With fixed selected counts, only base work grows with N, so
                # the largest ratio occurs at the smallest legal key length.
                n=m+1
                if max(setting['qk_counts']+setting['pv_counts'])>math.ceil(n/128):raise ValueError('Fixed quota exceeds minimum input envelope')
            rows.append(dict(attention_cost(m,n,h,hk,**setting),name=name,branch=int(branch),phase=int(ph)))
        return compile_request(rows,plan['budget'],expected_modules=list(modules),expected_layers=list(attention))
    def export(name,plan):
        folder=a.output/name;folder.mkdir();budget=ledger(plan);RepairPlan(plan)
        write(folder/'plan.json',plan);write(folder/'compute.json',budget)
        bank=folder/'assets';bank.mkdir();manifest=dict(complete=False,precision='fp4',model_identity=am['model_identity'],modules={},cohort_sha256=sha(a.cohort))
        stats=collections.Counter()
        for module,entry in am['modules'].items():
            asset=expand(torch.load(a.calibration/'rtn_assets'/entry['file'],map_location='cpu',weights_only=True),1.)
            union=sorted({i for key,row in plan['linears'].items() if key.split('|',2)[-1]==module for i in row['w_blocks']})
            packed=compact(asset,union);file=bank/entry['file'];torch.save(packed,file)
            manifest['modules'][module]=dict(file=file.name,sha256=sha(file))
            stats['file_bytes']+=file.stat().st_size
            for key,value in packed.items():
                if isinstance(value,torch.Tensor):stats['tensor_bytes']+=value.numel()*value.element_size()
            stats['w1_bytes']+=len(union)*128**2*9//16
        manifest['complete']=True;write(bank/'manifest.json',manifest)
        config=RunConfig(resource_mode='shared-calibration',compute_budget=plan['budget'],budget_plan=str((folder/'plan.json').resolve()),ffn=LinearSpec(0,0),projection=LinearSpec(0,0),share_pack=False)
        write(folder/'config.json',config.to_dict());stats['w1_density']=stats['w1_bytes']/weight_full_bytes
        candidate=dict(name=name,budget=plan['budget'],config=config.to_dict(),assets=str(bank.resolve()),assets_sha256=sha(bank/'manifest.json'),plan_sha256=sha(folder/'plan.json'),cost=dict(stats),bound_ratio=budget['ratio'])
        candidates.append(candidate);write(folder/'candidate.json',candidate)
    for cap,af,wf,hf in profiles:
        plan=build(cap,af,wf,hf);base=ledger(plan)
        chosen,extra,size=select_bundles(choices,math.floor(cap*base['base_mma'])-base['total_mma'],math.floor(wf*weight_full_bytes))
        for key,row in plan['linears'].items():
            name=key.split('|',2)[-1];kb=modules[name][2]//128
            if a.weight_axis=='n':row['w_blocks']=sorted(n*kb+k for n in chosen.get(name,[]) for k in range(kb))
            else:row['w_blocks']=sorted(n*kb+k for n in range(modules[name][1]//128) for k in chosen.get(name,[]))
        if a.ffn_calibration:plan['design']['ffn_calibration_sha256']=sha(a.ffn_calibration/'manifest.json')
        label=f'b{cap:g}_ffn_aware_{a.weight_axis}' if a.ffn_calibration else f'b{cap:g}_protected_{a.weight_axis}'
        if a.profile=='balanced':label=f'b{cap:g}_balanced_{a.weight_axis}'
        export(label,plan)
        if cap==2.5 and not a.ffn_calibration and a.weight_axis=='n':
            for variant in ('uniform_n','energy_n'):
                other=copy.deepcopy(plan)
                for name,(m,n,k) in modules.items():
                    count=len(chosen.get(name,[]));nb=n//128;kb=k//128
                    if variant=='uniform_n':indices=[int(i*nb/count) for i in range(count)] if count else []
                    else:
                        entry=am['modules'][name];asset=torch.load(a.calibration/'rtn_assets'/entry['file'],weights_only=True,map_location='cpu')
                        indices=torch.argsort(asset['score'].sum(1),descending=True,stable=True)[:count].tolist()
                    ids=sorted(ni*kb+ki for ni in indices for ki in range(kb))
                    for key,row in other['linears'].items():
                        if key.split('|',2)[-1]==name:row['w_blocks']=ids
                if ledger(other)['total_mma']!=ledger(plan)['total_mma']:raise ValueError('Position control must be compute matched')
                other['design']['position_control']=variant;export('b2.5_'+variant,other)
    # Exact old quotas with RTN bytes isolate representation from allocation.
    if not a.ffn_calibration and not a.skip_legacy_control:
        old_name='b2.5_marginal';prior=old['methods'][old_name];plan=json.loads(Path(prior['config']['budget_plan']).read_text())
        export('b2.5_old_quota_rtn',plan)
    write(a.output/'candidates.json',dict(complete=True,candidates=candidates,cohort_sha256=sha(a.cohort),
        scope='Calibration proposals only. Static physically compressed W1; native full-input A residuals. No quality or speed promotion.'))
    objective='Equal datasets/categories; phase-call-weighted native ACT2 minus full-three local NMSE per N128.'
    if a.ffn_calibration:objective+=' FFN operator totals rescaled using native full-FFN W-repair interventions.'
    if a.weight_axis=='k':objective+=' K128 proposals use activation RMS squared times residual-scale energy, normalized to those operator totals; not measured K-block marginal gains.'
    write(a.output/'weight_benefits.json',dict(complete=True,gains={k:v.tolist() for k,v in gains.items()},
        objective=objective+' Full trajectory validation required.'))


if __name__=='__main__':main()
