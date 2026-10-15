"""Resume verified model preparation and serialized full-request smoke checks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    src = root/'src/ThunderWorld'
    lock = (root/'logs/setup.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    state = dict(complete=False, pid=os.getpid(), started=time.time(), models={})
    def save():
        state['updated'] = time.time()
        target = root/'logs/setup-status.json'
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(state,indent=2)+'\n')
        tmp.replace(target)
    model_files = json.loads((root/'data/model_files.json').read_text())
    for model in ('edge','nano'):
        current = state['models'].setdefault(model,{})
        current['stage'] = 'waiting_for_model_and_recipes';save()
        folder = next(n for n,s in model_files.items() if s['repo'].endswith('-'+model.capitalize()))
        fixed = root/'models'/folder
        package = root/'data'/model
        while not (fixed/'thunderworld_verified.json').exists() or not (package/'package.json').exists():
            receipt = root/'logs/model_download.json'
            if receipt.exists() and json.loads(receipt.read_text()).get('error'):
                raise RuntimeError('Model download failed; see '+str(receipt))
            time.sleep(15)
        def run(stage, command):
            current['stage'] = stage;save()
            log = root/'logs'/f'{model}-{stage}.log'
            with log.open('w') as stream:
                result = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,cwd=src)
            if result.returncode:
                current.update(stage='failed', failed_operation=stage, returncode=result.returncode,
                               log=str(log));save()
                raise RuntimeError('Failed '+stage+'; see '+str(log))
        dest = root/'assets'/model
        if not (dest/'restored.json').exists():
            # A partially reconstructed bank is never exposed as ready. Preserve
            # failed progress separately before a clean reconstruction attempt.
            if dest.exists():
                dest.rename(dest.with_name(dest.name+f'.incomplete-{int(time.time())}'))
            run('restore-assets',[sys.executable,str(src/'tools/fp4_handoff/bundle.py'),'restore',
                '--package',str(package),'--model',str(fixed),'--output',str(dest),'--threads','4'])
        current['assets_verified'] = True;save()
        for method in ('bf16','fp8','fp4_base','b1.5_protected_n','b2_joint','b2.5_ffn_guarded'):
            result = root/'results'/f'{model}-{method}-smoke'
            if (result/'result.json').exists() and json.loads((result/'result.json').read_text()).get('complete'):
                continue
            if result.exists():
                result = result.with_name(result.name+f'-retry-{int(time.time())}')
            run(method+'-smoke',[sys.executable,str(src/'tools/thor_handoff/run.py'),
                '--root',str(root),'--model',model,'--method',method,'--repeats','1',
                '--output',str(result),'--save-outputs'])
            current.setdefault('smoke_results',{})[method] = str(result/'result.json');save()
        current.update(stage='complete',complete=True);save()
    state['complete'] = True;state['finished'] = time.time();save()


if __name__ == '__main__':
    main()
