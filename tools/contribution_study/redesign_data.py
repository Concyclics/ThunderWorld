"""Freeze new disjoint multi-scene calibration/development/validation/final inputs."""
import argparse,collections,datetime,hashlib,json,random,subprocess
from pathlib import Path
from baselines import sha,write
from dataset_subset import SOURCES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cosmos','previous','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--hf',required=True);a=p.parse_args()
    if (a.output/'cohort.json').exists():raise FileExistsError('Cohort already frozen')
    old=json.loads(a.previous.read_text());excluded={(r['benchmark'],r['source_id']) for r in old['cases']}
    root=a.cosmos/'evaluation/cosmos3/generator';groups=collections.defaultdict(list);prompt_hashes={}
    for path in sorted((root/'rbench/assets/prompts').glob('*.json')):
        prompt_hashes[str(path)]=sha(path);category=path.stem.removesuffix('_prompts')
        for r in json.loads(path.read_text()):
            if ('rbench',r['name']) not in excluded:groups['rbench',category].append(dict(source_id=r['name'],image_path='imgs/'+r['image_path'],prompt=r['prompt']))
    path=root/'paibench_g/assets/i2v_prompts.json';prompt_hashes[str(path)]=sha(path)
    for r in json.loads(path.read_text()):
        if ('paibench',r['video_id']) not in excluded:
            groups['paibench',r['image_name'].split('_')[0]].append(dict(source_id=r['video_id'],image_path='condition_image/'+r['image_name'],prompt=r['prompt_en']))
    counts={'calibration':1,'development':1,'validation':2,'final':None};splits={role:[] for role in counts}
    for (benchmark,category),pool in sorted(groups.items()):
        pool=sorted(pool,key=lambda x:x['source_id']);random.Random(f'redesign-v1:{benchmark}:{category}').shuffle(pool)
        offset=0
        for role,count in counts.items():
            count=count if count is not None else (4 if benchmark=='rbench' else 6 if category=='human' else 5)
            if len(pool)<offset+count:raise ValueError('Insufficient disjoint stratum')
            splits[role].extend(dict(benchmark=benchmark,category=category,**r) for r in pool[offset:offset+count]);offset+=count
    a.output.mkdir(parents=True,exist_ok=True)
    proposal=dict(previous_manifest_sha256=sha(a.previous),sources=SOURCES,prompt_hashes=prompt_hashes,splits=splits)
    write(a.output/'selection.before-download.json',proposal)
    for benchmark,(repo,revision) in SOURCES.items():
        files=[r['image_path'] for rows in splits.values() for r in rows if r['benchmark']==benchmark]
        with (a.output/(benchmark+'.download.log')).open('w') as log:
            subprocess.run([a.hf,'download',repo,*files,'--type','dataset','--revision',revision,'--local-dir',str(a.output/'data'/benchmark),'--max-workers','4'],stdout=log,stderr=subprocess.STDOUT,check=True)
    from PIL import Image
    def pixel_sha(path):
        with Image.open(path) as image:
            image=image.convert('RGB');return hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
    used={pixel_sha(r['case']['image']) for r in old['cases']}
    used.add(pixel_sha('/home/yufeng/thunderworld/ThunderWorld-joint-runtime/data_local/e2e_inputs/robot_153.jpg'))
    sequence=301001;cases={}
    for role,entries in splits.items():
        cases[role]=[]
        for r in entries:
            image=a.output/'data'/r['benchmark']/r['image_path'];pixels=pixel_sha(image)
            if pixels in used:raise ValueError('Duplicate decoded image; do not freeze or run this proposal')
            used.add(pixels);seed_count=1 if role=='calibration' else 2
            case=dict(id=r['benchmark']+'__'+r['source_id'],task='i2v',prompt=r['prompt'],image=str(image.resolve()),
                height=192,width=320,frames=61,negative_prompt='distorted objects, flickering, blurry, low quality')
            cases[role].append(dict(benchmark=r['benchmark'],category=r['category'],source_id=r['source_id'],case=case,
                image_sha256=sha(image),pixel_sha256=pixels,seeds=list(range(sequence,sequence+seed_count))))
            sequence+=seed_count
    # Audit saved generated samples throughout this workspace before consuming these seeds.
    files=subprocess.run(['rg','--files','--hidden','--no-ignore','-g','seed*.npz','/home/yufeng/thunderworld'],capture_output=True,text=True)
    if files.returncode not in (0,1) or files.stderr:raise RuntimeError('Seed archive audit failed')
    new_seeds={f'seed{s}' for rows in cases.values() for r in rows for s in r['seeds']}
    matches=[f for f in files.stdout.splitlines() if Path(f).stem in new_seeds]
    if matches:raise ValueError('New seeds already used: '+str(matches[:3]))
    write(a.output/'cohort.json',dict(schema=1,frozen=True,created_at=datetime.datetime.now().astimezone().isoformat(),
        model=old['model'],sources=SOURCES,previous_manifest_sha256=sha(a.previous),prompt_hashes=prompt_hashes,cases=cases,
        counts={k:len(v) for k,v in cases.items()},seeds_audit='No matching saved seed NPZ in local workspace; no remote history claim',
        steps=35,scope='New inputs disjoint by source id and decoded pixels from the earlier 72 and robot_153. No semantic-near-duplicate guarantee. Final inputs reserved, no final generation before validation selection.'))
    print(json.dumps({k:len(v) for k,v in cases.items()}))


if __name__=='__main__':main()
