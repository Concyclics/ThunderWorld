"""Resumable, full-M, all-generation-layer development capture for compute search."""
import argparse
import hashlib
import json
from pathlib import Path
from thunderworld_fp4.cosmos.cohorts import Cohort
from thunderworld_fp4.cosmos.resources import select_shared_device, shared_worker


def atomic_json(path, value):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'case', 'output', 'cohort'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--student-config',type=Path)
    p.add_argument('--student-assets',type=Path)
    p.add_argument('--reference-capture',type=Path)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.student_config and not a.reference_capture:p.error('Student capture requires original BF16 weights')
    role='trajectory' if a.student_config else 'calibration'
    cohort = Cohort.load(a.cohort)
    case = json.loads(a.case.read_text())
    if case.get('split') != 'development' or case['task'] != cohort.value['task']:
        raise ValueError('Capture requires matching development case')
    if case.get('image'):
        case['image'] = str((a.case.parent / case['image']).resolve())
    a.output.mkdir(parents=True, exist_ok=a.resume)
    manifest_path = a.output / 'manifest.json'
    with shared_worker():
        resource = select_shared_device()
        import torch
        from thunderworld_fp4.cosmos import CosmosRunner, RunConfig
        from thunderworld_fp4.cosmos.settings import phase
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        from dataclasses import replace
        config=RunConfig(precision='bf16',attention_backend='bf16',execution='eager',resource_mode='shared-calibration')
        if a.student_config:config=replace(RunConfig.from_dict(json.loads(a.student_config.read_text())),execution='eager',debug='trace',resource_mode='shared-calibration')
        runner=CosmosRunner(a.model,config,assets=a.student_assets,audio=case['task']=='av').prepare()
        teacher=json.loads((a.reference_capture/'manifest.json').read_text()) if a.reference_capture else None
        if teacher and teacher['model_identity']!=runner.identity:raise ValueError('Student/teacher model mismatch')
        if teacher and teacher.get('calibration_round',0)>=1:raise ValueError('Only one student recalibration round is allowed')
        manifest = dict(complete=False, split='development', cohort=cohort.value,
                        cohort_sha256=cohort.fingerprint, trajectory='student' if teacher else 'teacher', calibration_round=1 if teacher else 0, role=role, config=config.to_dict(), case=case, model_identity=runner.identity,
                        resource=resource, performance_eligible=False, weights={}, rows=[],
                        completed_seeds=[], scope='All generation layers, full M, four phases, all actual CFG branches')
        if a.resume and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if (manifest['cohort_sha256'] != cohort.fingerprint or manifest['case'] != case
                    or manifest['model_identity'] != runner.identity or RunConfig.from_dict(manifest.get('config',config.to_dict()))!=config):
                raise ValueError('Resume provenance mismatch')
            for row in manifest['rows']:
                if hashlib.sha256((a.output / row['file']).read_bytes()).hexdigest() != row['sha256']:
                    raise ValueError('Corrupt captured panel')
        atomic_json(manifest_path, manifest)
        active_seed = [None]
        seen = {(r['seed'], r['kind'], r['name'], r['branch'], r['phase']) for r in manifest['rows']}
        def save_row(kind, name, payload):
            e = runner.engine
            ph = phase(e.step, e.steps)
            key = (active_seed[0], kind, name, e.branch, ph)
            if key in seen or e.warming:
                return
            layer = name.split('.')[1] if kind == 'linear' else name
            folder = a.output / f'layer_{int(layer):03d}'
            folder.mkdir(exist_ok=True)
            path = folder / f'{len(manifest["rows"]):05d}.pt'
            data = payload()
            tmp = path.with_suffix('.pt.tmp')
            torch.save(data, tmp)
            tmp.replace(path)
            manifest['rows'].append(dict(seed=active_seed[0], kind=kind, name=name,
                branch=e.branch, phase=ph, step=e.step, file=str(path.relative_to(a.output)),
                shapes={k: list(v.shape) for k, v in data.items()},
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            seen.add(key)
            atomic_json(manifest_path, manifest)
        handles = []
        def hook(name):
            def callback(module, args, out):
                def payload():
                    x=args[0].detach().reshape(-1,module.in_features)
                    gold=out.detach().reshape(-1,module.out_features)
                    if teacher:
                        w=torch.load(a.reference_capture/teacher['weights'][name],map_location='cpu',weights_only=True).cuda()
                        gold=torch.nn.functional.linear(x,w)
                    payload=dict(x=x.cpu().contiguous(),y=gold.cpu().contiguous())
                    if teacher:payload['candidate_y']=out.detach().reshape_as(gold).cpu().contiguous()
                    return payload
                save_row('linear',name,payload)
            return callback
        try:
            for name in runner.engine.targets:
                module = runner.pipe.transformer.get_submodule(name)
                if name not in manifest['weights']:
                    path = a.output / f'weight_{len(manifest["weights"]):03d}.pt'
                    if teacher:
                        import shutil
                        shutil.copyfile(a.reference_capture/teacher['weights'][name],path)
                    else:torch.save(module.weight.detach().cpu(),path)
                    manifest['weights'][name] = path.name
                handles.append(module.register_forward_hook(hook(name)))
            atomic_json(manifest_path, manifest)
            def attention(layer,q,k,v,out):
                def payload():
                    gold=out
                    if teacher:
                        gold=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),enable_gqa=True).transpose(1,2)
                    payload=dict(q=q.cpu(),k=k.cpu(),v=v.cpu(),y=gold.cpu())
                    if teacher:payload['candidate_y']=out.cpu()
                    return payload
                save_row('attention',str(layer),payload)
            runner.engine.capture_attention=attention
            runner.warmup(case, 35)
            for seed in cohort.seeds[role]:
                if seed in manifest['completed_seeds']:
                    continue
                active_seed[0] = seed
                with torch.inference_mode():
                    runner.infer(case, seed, 35)
                manifest['completed_seeds'].append(seed)
                atomic_json(manifest_path, manifest)
                print(json.dumps(dict(seed=seed, panels=len(manifest['rows']))), flush=True)
            expected = len(cohort.seeds[role]) * 4 * (1 if case['task']=='policy' else 2) * (
                len(runner.engine.targets) + len(runner.pipe.transformer.layers))
            if len(manifest['rows']) != expected:
                raise RuntimeError(f'Incomplete coverage: {len(manifest["rows"])} != {expected}')
            manifest.update(complete=True, expected_panels=expected)
            manifest.pop('error', None)
            atomic_json(manifest_path, manifest)
        except Exception as error:
            manifest.update(complete=False, error=repr(error), paused=True)
            atomic_json(manifest_path, manifest)
            raise
        finally:
            for handle in handles:
                handle.remove()
            runner.engine.clear()
            del runner
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
