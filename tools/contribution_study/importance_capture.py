"""Capture full-M native teacher panels on 16 calibration scenes, without row clipping.

Each scene supplies one phase/CFG pair; all eight pairs occur twice. Full cross-
scene phase statistics from the earlier 16 x 8 calibration remain the scale priors.
"""
import argparse
import json
from pathlib import Path
import time

from baselines import sha, write
from dataset_infer import clear_case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--after', type=Path)
    a = p.parse_args()
    if a.after:
        while True:
            status = json.loads(a.after.read_text()) if a.after.exists() else {}
            if status.get('paused'):
                raise RuntimeError('Previous experiment paused')
            if status.get('complete'):
                break
            time.sleep(5)
    cohort = json.loads(a.cohort.read_text())
    if len(cohort['cases']['calibration']) != 16 or not cohort['frozen']:
        raise ValueError('Expected frozen 16-scene calibration')
    a.output.mkdir(parents=True, exist_ok=True)
    identity = dict(cohort_sha256=sha(a.cohort), source_sha256=sha(__file__))
    path = a.output / 'manifest.json'
    state = dict(complete=False, identity=identity, cases={}, weights={}, rows=[], performance_eligible=False,
                 scope='Full M, one phase/CFG panel per calibration scene per operator; all 8 pairs twice across 16 scenes. Not all 16x8 combinations.')
    if path.exists():
        state = json.loads(path.read_text())
        if state['identity'] != identity:
            raise ValueError('Changed capture resume')
        if state['complete']:
            return
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    with shared_worker():
        state['resource'] = select_shared_device()
        import torch
        from thunderworld_fp4.cosmos import CosmosRunner, RunConfig
        from thunderworld_fp4.cosmos.settings import phase
        torch.set_num_threads(8)
        runner = CosmosRunner(cohort['model'], RunConfig(precision='bf16', attention_backend='bf16', execution='eager', resource_mode='shared-calibration'), ar_cache=True, first_frame=True).prepare()
        state['model_identity'] = runner.identity
        active = [False]
        target = [0, 0, 0]
        seen = set()
        handles = []
        for index, name in enumerate(runner.engine.targets):
            file = a.output / f'weight_{index:03d}.pt'
            if not file.exists():
                torch.save(runner.pipe.transformer.get_submodule(name).weight.detach().cpu(), file)
            if name in state['weights'] and sha(file) != state['weights'][name]['sha256']:
                raise ValueError('Changed teacher weights')
            state['weights'][name] = dict(file=file.name, sha256=sha(file))
        write(path, state)

        def hook(name, index):
            def callback(module, args, output):
                engine = runner.engine
                if not active[0] or engine.warming or name in seen:
                    return
                if (phase(engine.step, engine.steps), engine.branch) != tuple(target[:2]):
                    return
                seen.add(name)
                x = args[0].detach().reshape(-1, module.in_features).contiguous().cpu()
                y = output.detach().reshape(-1, module.out_features).contiguous().cpu()
                folder = a.output / f'layer_{index:03d}'
                folder.mkdir(exist_ok=True)
                file = folder / f'scene_{target[2]:02d}.pt'
                torch.save(dict(x=x, y=y), file)
                state['rows'].append(dict(name=name, scene_index=target[2], phase=target[0], branch=target[1],
                                          file=str(file.relative_to(a.output)), sha256=sha(file), shape=[len(x), y.shape[1], x.shape[1]]))
            return callback

        try:
            for index, name in enumerate(runner.engine.targets):
                handles.append(runner.pipe.transformer.get_submodule(name).register_forward_hook(hook(name, index)))
            for index, entry in enumerate(cohort['cases']['calibration']):
                case = entry['case']['id']
                if case in state['cases']:
                    continue
                state['rows'] = [r for r in state['rows'] if r['scene_index'] != index]
                if sha(entry['case']['image']) != entry['image_sha256']:
                    raise ValueError('Changed image')
                target[:] = [index % 4, (index // 4) % 2, index]
                clear_case(runner)
                runner.warmup(entry['case'], 35)
                seen.clear()
                active[0] = True
                with torch.inference_mode():
                    runner.infer(entry['case'], entry['seeds'][0], 35)
                active[0] = False
                if seen != set(runner.engine.targets):
                    raise ValueError('Incomplete full-layer capture')
                state['cases'][case] = dict(scene_index=index, phase=target[0], branch=target[1], seed=entry['seeds'][0])
                write(path, state)
                print(json.dumps(dict(completed=len(state['cases']), total=16)), flush=True)
            state['complete'] = True
            state.pop('paused', None)
            state.pop('error', None)
            write(path, state)
        except Exception as error:
            state.update(paused=True, error=repr(error))
            write(path, state)
            raise
        finally:
            for handle in handles:
                handle.remove()
            runner.engine.clear()


if __name__ == '__main__':
    main()
