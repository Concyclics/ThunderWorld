"""Sequential receiver matrix; each case has a separate process and result file."""
import argparse,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--weights',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--suite',choices=['quick','main','stress'],default='quick');p.add_argument('--kind',choices=['both','ffn','attention'],default='both');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
m=json.loads((a.data/'manifest.json').read_text());jobs=[]
rows={'quick':[128],'main':[128,512],'stress':[2048]}[a.suite]
if a.kind in ('both','ffn'):
    if a.weights is None:p.error('FFN requires --weights')
    modes=['a75_w6'] if a.suite=='quick' else ['base','a50','a75','full_act2','a75_w6','a93_w6']
    for item in m['ffn']:
        for n in rows:
            for mode in modes:
                jobs.append(['--kind','ffn','--asset',item['file'],'--preset',mode,'--m',str(n),'--weights',str(a.weights),'--block-k','128','--tile-m','32'])
if a.kind in ('both','attention'):
    modes=['balanced','head_guided'] if a.suite=='quick' else ['fp4_base','full_act2','balanced','quality','full_dual','head_guided']
    for file in ['attention_av_step17.pt','attention_robot_step17.pt']:
        for n in ([128] if a.suite=='quick' else [128,560] if a.suite=='main' else [2048]):
            for mode in modes:
                jobs.append(['--kind','attention','--asset',file,'--preset',mode,'--m',str(n),'--tile-m','32'])
commands=[]
for i,job in enumerate(jobs):
    target=a.output/f'{i:03d}_{job[1]}_{Path(job[3]).stem}_{job[5]}_m{job[7]}.json'
    cmd=[sys.executable,str(HERE/'bench.py'),'--data',str(a.data),'--output',str(target),'--rounds','1' if a.suite=='quick' else '5','--rep-ms','10' if a.suite=='quick' else '40',*job]
    commands.append(cmd)
if a.dry_run:print(json.dumps(commands,indent=2));raise SystemExit
if a.output.exists():raise FileExistsError('Use a fresh matrix output directory')
a.output.mkdir(parents=True);(a.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
for i,cmd in enumerate(commands):
    print(f'[{i+1}/{len(commands)}] '+str(cmd[-len(jobs[i]):]),flush=True)
    with (a.output/f'{i:03d}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Completed',len(commands),'cases')
