"""Prepare native FP8 / ordinary min-max FP4 baselines and freeze local launch profiles."""
import json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs5090/baseline_compare_20260914'

def call(label,args):
    dest=OUT/label
    if (dest/'result.json').exists():
        assert json.loads((dest/'result.json').read_text())['complete'];return dest
    print('Running',label,flush=True)
    with (OUT/f'{label}.log').open('w') as log:
        subprocess.run([sys.executable,'-m','thunderworld_fp4.cosmos',*map(str,args),'--output',str(dest)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    return dest

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'configs').mkdir(exist_ok=True);(OUT/'profiles').mkdir(exist_ok=True)
    for name,source in [('bf16','bf16'),('fp8','fp8'),('fp4','base')]:
        config=json.loads((ROOT/f'tools/joint_inference/configs/{source}.json').read_text())
        (OUT/f'configs/{name}.json').write_text(json.dumps(config,indent=2)+'\n')
    config=json.loads((OUT/'configs/fp4.json').read_text())
    config.update(linear_backend='cutlass-fused',cutlass_library=str(ROOT/'build/cutlass5090_20260914/repair_n128.so'),cutlass_small_library=str(ROOT/'build/cutlass5090_20260914/repair_n64.so'),joint_tuning=True)
    (OUT/'configs/fp4_cutlass.json').write_text(json.dumps(config,indent=2)+'\n')
    identity=json.loads((ROOT/'runs5090/joint_e2e_20260914/profiles/bf16.json').read_text())['identity']
    for name in ('bf16','fp8'):(OUT/f'profiles/{name}.json').write_text(json.dumps(dict(schema_version=1,identity=identity,entries={},missing=[],hits=[]),indent=2)+'\n')
    for name in ('fp8','fp4'):
        call('prepare_'+name,['prepare','--model',ROOT/'models/Cosmos3-Edge-a9d944e','--config',OUT/f'configs/{name}.json'])
    for name in ('fp4','fp4_cutlass'):
        dest=call('tune_'+name,['warmup','--model',ROOT/'models/Cosmos3-Edge-a9d944e','--config',OUT/f'configs/{name}.json','--assets',OUT/'prepare_fp4/assets','--case',ROOT/'tools/joint_inference/cases/robot_observe.json','--steps','35','--tune'])
        shutil.copyfile(dest/'tuning.json',OUT/f'profiles/{name}.json')
    (OUT/'setup_complete.json').write_text(json.dumps(dict(complete=True,scope='Repository FP8 W8A8 and ordinary uncalibrated min-max FP4; no repair; fixed model/pinned Diffusers; local frozen launch profiles'),indent=2)+'\n')
if __name__=='__main__':main()
