"""Reserve untouched local seeds and freeze a cohort before validation starts."""
import json
from pathlib import Path
import subprocess
from baselines import sha,write


def admit_run(cohort,split,seeds,method,selection=None,configuration_sha256=None):
    if len(seeds)!=len(set(seeds)) or not seeds:raise ValueError('Unique nonempty seeds required')
    if split=='trajectory' and cohort is None:
        if not set(seeds).issubset({7103,7104}):raise ValueError('Development seeds only without a frozen cohort')
        return dict(split=split,selection_eligible=True)
    if cohort is None:raise ValueError('Held-out runs require a frozen cohort')
    value=json.loads(Path(cohort).read_text())
    if value.get('seed_status')!='frozen':raise ValueError('Cohort has not been audited and frozen')
    if set(seeds)!=set(value['seeds'][split]):raise ValueError('Run the complete declared split')
    receipt=dict(split=split,cohort_sha256=sha(cohort),selection_eligible=split!='final')
    if split=='final':
        if selection is None:raise ValueError('Final evaluation requires immutable selected candidates')
        frozen=json.loads(Path(selection).read_text())
        if frozen.get('cohort_sha256')!=sha(cohort) or method not in frozen.get('methods',[]):
            raise ValueError('Method is not in the predeclared final comparison')
        if not configuration_sha256 or frozen.get('configuration_sha256',{}).get(method)!=configuration_sha256:
            raise ValueError('Final configuration differs from the frozen selection')
        receipt['selection_sha256']=sha(selection)
    return receipt


def freeze_cohort(source,history,output):
    output=Path(output)
    if output.exists():raise FileExistsError('Frozen cohorts are immutable: '+str(output))
    value=json.loads(Path(source).read_text())
    if value.get('seed_status')!='reserved-not-audited-not-frozen':raise ValueError('Expected an unconsumed proposal')
    attempts=[]
    for offset in range(0,1000001,100000):
        seeds=[s+offset for role in ('validation','final') for s in value['seeds'][role]]
        alternatives='|'.join(map(str,seeds))
        pattern=r'"seed"\s*:\s*('+alternatives+r')\s*[,}]'
        command=['rg','--hidden','--no-ignore','-l','-g','*.json','-g','*.log',pattern,str(history)]
        check=subprocess.run(command,capture_output=True,text=True)
        files=subprocess.run(['rg','--files','--hidden','--no-ignore','-g','seed*.npz',str(history)],capture_output=True,text=True)
        matches=[line for line in files.stdout.splitlines() if Path(line).stem in {f'seed{s}' for s in seeds}]
        if check.returncode not in (0,1) or files.returncode not in (0,1) or check.stderr or files.stderr:
            raise RuntimeError('Seed audit could not inspect the declared history root')
        attempts.append(dict(offset=offset,executed_record_matches=check.stdout.splitlines(),output_matches=matches))
        if not check.stdout and not matches:break
    else:raise RuntimeError('No unused cohort in declared search range')
    for role in ('validation','final'):value['seeds'][role]=[s+offset for s in value['seeds'][role]]
    value.update(seed_status='frozen',seed_audit=dict(proposal_sha256=sha(source),history_root=str(Path(history).resolve()),
        attempts=attempts,scope='Local execution records only; no visibility into remote machines'))
    from thunderworld_fp4.cosmos.cohorts import Cohort
    Cohort(value)
    write(output,value)
    return value
