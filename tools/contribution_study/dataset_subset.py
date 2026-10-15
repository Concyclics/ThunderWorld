"""Freeze a stratified RBench/PAI-Bench scene subset before generation."""
import argparse,collections,hashlib,json,random,subprocess
from pathlib import Path
from baselines import sha,write

SOURCES={'rbench':('DAGroup-PKU/RBench','6bdccf349ff5a8f68302428351e94f34ecd62450'),
         'paibench':('shi-labs/physical-ai-bench-generation','c36b7beb69bd7c4adcedc1431407fce98f9dfe44')}


def stratified(groups,total,seed=170917):
    keys=sorted(groups);counts={k:total//len(keys) for k in keys}
    for k in sorted(keys,key=lambda k:(-len(groups[k]),k))[:total%len(keys)]:counts[k]+=1
    selected=[]
    for k in keys:
        rng=random.Random(f'{seed}:{k}');pool=sorted(groups[k],key=lambda r:r['source_id'])
        selected.extend((k,r) for r in rng.sample(pool,counts[k]))
    return sorted(selected,key=lambda x:(x[0],x[1]['source_id']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cosmos','frozen','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--hf',required=True);a=p.parse_args()
    if (a.output/'manifest.json').exists():raise FileExistsError('Already frozen')
    a.output.mkdir(parents=True,exist_ok=True)
    root=a.cosmos/'evaluation/cosmos3/generator';groups={'rbench':collections.defaultdict(list),'paibench':collections.defaultdict(list)};prompt_hashes={}
    for path in sorted((root/'rbench/assets/prompts').glob('*.json')):
        prompt_hashes[str(path)]=sha(path);category=path.stem.removesuffix('_prompts')
        for r in json.loads(path.read_text()):groups['rbench'][category].append(dict(source_id=r['name'],image_path='imgs/'+r['image_path'],prompt=r['prompt']))
    path=root/'paibench_g/assets/i2v_prompts.json';prompt_hashes[str(path)]=sha(path)
    for r in json.loads(path.read_text()):
        category=r['image_name'].split('_')[0]
        groups['paibench'][category].append(dict(source_id=r['video_id'],image_path='condition_image/'+r['image_name'],prompt=r['prompt_en']))
    proposal=[]
    for benchmark,table in groups.items():
        for category,row in stratified(table,36):proposal.append(dict(benchmark=benchmark,category=category,**row))
    write(a.output/'selection.before-download.json',dict(sources=SOURCES,proposal=proposal,prompt_hashes=prompt_hashes))
    for benchmark,(repo,revision) in SOURCES.items():
        command=[a.hf,'download',repo,*[r['image_path'] for r in proposal if r['benchmark']==benchmark],
            '--type','dataset','--revision',revision,'--local-dir',str(a.output/'data'/benchmark),'--max-workers','4']
        with (a.output/(benchmark+'.download.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    cases=[];image_hashes=set()
    for i,row in enumerate(proposal):
        image=a.output/'data'/row['benchmark']/row['image_path'];digest=sha(image)
        if digest in image_hashes:raise ValueError('Duplicate input image; freeze a replacement before generation')
        image_hashes.add(digest)
        case=dict(id=row['benchmark']+'__'+row['source_id'],task='i2v',prompt=row['prompt'],image=str(image.resolve()),height=192,width=320,frames=61,
            negative_prompt='distorted objects, flickering, blurry, low quality')
        cases.append(dict(benchmark=row['benchmark'],category=row['category'],source_id=row['source_id'],case=case,image_sha256=digest,seeds=[201001+2*i,201002+2*i]))
    frozen=json.loads((a.frozen/'selection.json').read_text());methods={}
    names=['bf16','fp8','fp4_base','fp4_three','fp4_four']+[r['selected'] for r in frozen['decisions']]
    source_manifest=json.loads((a.frozen/'manifest.json').read_text());jobs={r['name']:r for r in source_manifest['runs']}
    for name in names:
        result_path=a.frozen/'generation'/name/'result.json';receipt=json.loads(result_path.read_text())
        if not receipt['complete']:raise ValueError('Incomplete fixed configuration')
        cfg=receipt['runtime']['config'];entry=dict(config=cfg,origin_receipt_sha256=sha(result_path),origin_configuration_sha256=receipt['configuration_sha256'])
        args=jobs[name]['args']
        if '--assets' in args:
            asset=Path(args[args.index('--assets')+1]);entry.update(assets=str(asset),assets_sha256=sha(asset/'manifest.json'),plan_sha256=sha(cfg['budget_plan']))
        methods[name]=entry
    common=source_manifest['common_args'];model=common[common.index('--model')+1]
    write(a.output/'manifest.json',dict(schema=1,frozen=True,model=model,sources=SOURCES,prompt_hashes=prompt_hashes,cases=cases,methods=methods,
        previous_selection_sha256=sha(a.frozen/'selection.json'),steps=35,resource_mode='shared-calibration',debug='off',execution='graph',
        statistics=dict(unit='scene: average two seeds before stratified resampling',replicates=10000,bootstrap_seed=170917,
            primary='LPIPS with SSIM/L2 mean guardrails',family_size=14,psnr='supplementary; per-video global RGB MSE, data_range=1',
            aggregation='equal categories within each dataset; equal datasets for combined summary'),
        scope='Frozen-policy cross-scene BF16 agreement. Original prompt / prompt_en, common negative prompt; 192x320x61, 35 steps, CFG6. Not official RBench/PAI-Bench leaderboard protocol or task-success score. No calibration/selection on this subset.'))


if __name__=='__main__':main()
