"""Freeze portable policies, raw summaries and source provenance (no model/video payloads)."""
import argparse
import collections
import gzip
import json
from pathlib import Path
import shutil
import subprocess
from baselines import sha,write


def packed(path,value):
    path.write_bytes(gzip.compress(json.dumps(value,ensure_ascii=False,indent=2).encode(),mtime=0))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);a=p.parse_args()
    a.archive.mkdir(exist_ok=True,parents=True);repo=Path(__file__).resolve().parents[2]
    for filename in ('verification.json','gpu_probe.json','calibration_complete.json'):
        source=a.study/filename
        if not json.loads(source.read_text())['complete']:raise ValueError('Incomplete implementation check')
        shutil.copyfile(source,a.archive/filename)
    catalog=json.loads((a.study/'candidates/candidates.json').read_text());bundle=[]
    for c in catalog['candidates']:
        folder=Path(c['config']['budget_plan']).parent
        bundle.append(dict(candidate=c,plan=json.loads((folder/'plan.json').read_text()),
            assets_manifest=json.loads((Path(c['assets'])/'manifest.json').read_text()),
            bounds=json.loads((folder/'bounds.json').read_text()) if (folder/'bounds.json').exists() else None,
            allocation=json.loads((folder/'allocation.json').read_text()) if (folder/'allocation.json').exists() else None))
    packed(a.archive/'policy_bundle.json.gz',dict(complete=True,candidates=bundle))
    probe_counts=collections.Counter()
    for cap in (1.5,2.,2.5):
        folder=a.study/'calibration'/f'b{cap:g}';manifest=json.loads((folder/'manifest.json').read_text())
        if sha(folder/'definitions.json')!=manifest['definitions_sha256']:raise ValueError('Changed calibration definitions')
        data={}
        for case,digest in manifest['cases'].items():
            file=folder/(case+'.json')
            if sha(file)!=digest:raise ValueError('Changed calibration probe')
            data[case]=json.loads(file.read_text())
            for key,row in data[case]['groups'].items():
                if not row['baseline_bitwise_equal']:raise ValueError('Unqualified numerical reference')
                probe_counts[key.split('|')[0]]+=1
        packed(a.archive/f'calibration_b{cap:g}.json.gz',dict(manifest=manifest,cases=data,definitions=json.loads((folder/'definitions.json').read_text())))
    generations=[];counts={}
    for split in ('development','validation'):
        folder=a.study/split
        if not folder.exists():continue
        if not json.loads((folder/'workflow.json').read_text())['complete']:raise ValueError('Incomplete quality workflow')
        manifest=json.loads((folder/'manifest.json').read_text());rows=[]
        for method in manifest['methods']:
            for case in manifest['cases']:
                file=folder/'generation'/method/case['case']['id']/'result.json'
                receipt=json.loads(file.read_text())
                if not receipt['complete']:raise ValueError('Incomplete output receipt')
                rows.append(dict(receipt_path=str(file),receipt_sha256=sha(file),receipt=receipt))
        generations.append(dict(split=split,manifest=manifest,receipts=rows))
        counts[split]=dict(scenes=len(manifest['cases']),methods=len(manifest['methods']),videos=sum(len(x['receipt']['rows']) for x in rows),
                           new_videos=sum(len(x['receipt']['rows']) for x in rows if 'imported_from' not in x['receipt']))
    packed(a.archive/'generation_index.json.gz',dict(complete=True,runs=generations))
    files=list((repo/'packages/fp4-repair/src/thunderworld_fp4').rglob('*.py'))
    files+=list((repo/'tools/contribution_study').glob('*.py'));hashes={}
    for file in files:
        relative=file.relative_to(repo);target=a.archive/'source'/relative;target.parent.mkdir(exist_ok=True,parents=True)
        shutil.copyfile(file,target);hashes[str(relative)]=sha(file)
    write(a.archive/'implementation.json',dict(complete=True,counts=counts,sources=hashes,baseline_bitwise_probes=dict(probe_counts),
         checkout_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),uncommitted=True,
         performance_eligible=False,final_test_run=False,
         scope='Current source snapshot; calibration source fingerprints are preserved in raw manifests. W1 binary assets, original BF16 weights and videos remain local.'))
    print(json.dumps(counts))


if __name__=='__main__':main()
