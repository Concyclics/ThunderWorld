"""Post hoc robustness check: restore rare stored-encoding tie groups to control.

The flagged groups are defined by coding bytes, never by review output. Original
frozen results are preserved. This is an audit, not another searched candidate.
"""
import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from baselines import sha, write


def run(root):
    protocol = json.loads((root/'protocol.json').read_text())
    original = json.loads((root/'progress.json').read_text())
    audit_path = root/'report/encoding_audit.json'
    audit = json.loads(audit_path.read_text())
    path = root/'report/boundary_check.json'
    identity = dict(source_sha256=sha(__file__), protocol_sha256=sha(root/'protocol.json'), audit_sha256=sha(audit_path))
    state = dict(complete=False, identity=identity, modules={}, performance_eligible=False)
    if path.exists():
        state = json.loads(path.read_text())
        if state['identity'] != identity:
            raise ValueError('Changed sensitivity-check protocol')
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    with shared_worker():
        state['resource'] = select_shared_device(12288)
        import torch
        from outlier_detail import patch_asset
        from outlier_detail_study import checked_load
        from importance_joint_calibrate import objective
        from thunderworld_fp4.cosmos.assets import expand
        from thunderworld_fp4.cosmos.linears import RepairLinear
        from thunderworld_fp4.cosmos.settings import LinearSpec, RunConfig
        torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32 = False
        benchmarks = [c['benchmark'] for c in protocol['cases']]
        fit, review = protocol['fit'], protocol['review']
        for name, flagged in audit['boundary_groups'].items():
            if name in state['modules']:
                continue
            opdata = original['modules'][name]; item = protocol['modules'][name]
            ids, na = item['support'], item['a_blocks']
            raw = checked_load(Path(protocol['spec']['assets'])/item['asset']['file'], item['asset']['sha256'])
            old = expand(raw, ids)
            patch = torch.load(root/'operators'/name/'patches.pt', weights_only=True)
            joint = {key: value.clone() for key, value in patch['methods']['joint'].items()}
            bad = torch.tensor(flagged)
            for key in ('c0', 's0', 'c1', 's1'):
                joint[key].view(torch.uint8)[bad] = patch['methods']['scale_only'][key].view(torch.uint8)[bad]
            # Scalar metadata uses its own dtype, not byte indexing.
            for key in ('loss', 'removed', 'base_scale', 'residual_scale'):
                joint[key][bad] = patch['methods']['scale_only'][key][bad]
            asset = patch_asset(old, joint, patch['groups']['nr'], patch['groups']['kg'], patch['groups']['blocks'])
            m, n, k = opdata['shape']; panels = []
            for row in item['panel_rows']:
                panel = checked_load(Path(protocol['capture'])/row['file'], row['sha256'])
                panels.append((panel['x'].cuda(), panel['y'].cuda().float()))
            results = {}
            for tag, cross in (('three', False), ('four', True)):
                ls = LinearSpec(0, 0, bool(cross and na), activation_blocks=na, weight_support=tuple(ids))
                controller = SimpleNamespace(warming=False, config=RunConfig.from_dict(protocol['spec']['config']), linear_spec=lambda *_: ls)
                with torch.inference_mode():
                    native = RepairLinear(SimpleNamespace(in_features=k, out_features=n), controller, name, ls, asset)
                    native.prepare_shape(m, ls)
                    errors = [float((native(x).float()-y).square().sum()/y.square().sum().clamp_min(1e-20)) for x, y in panels]
                if not np.isfinite(errors).all():
                    raise ValueError('Nonfinite output')
                results[tag] = dict(panel_nmse=errors, fit_objective=objective([errors[i] for i in fit], [benchmarks[i] for i in fit]),
                                    review_mean=sum(errors[i] for i in review)/len(review))
            with np.load(root/'operators'/name/'groups.npz') as arrays:
                energy = arrays['joint__energy'].astype(np.float64).sum()
                sse = arrays['joint__dual_sse'].astype(np.float64).sum()
                sse += arrays['scale_only__dual_sse'][flagged].astype(np.float64).sum()-arrays['joint__dual_sse'][flagged].astype(np.float64).sum()
                weight_nmse = float(sse/energy)
            parents = list(dict.fromkeys([opdata['chosen'][label] for label in ('rtn', 'scale_only', 'max_scale')]))
            candidates = {label: opdata['raw_native'][label] for label in parents}
            candidates['joint_without_boundary_groups'] = results
            chosen = min(candidates, key=lambda label: candidates[label]['three']['fit_objective'])
            state['modules'][name] = dict(restored_groups=flagged, raw_native=results,
                raw_weight_nmse=weight_nmse, chosen=chosen, accepted_native=candidates[chosen],
                accepted_weight_nmse=weight_nmse if chosen == 'joint_without_boundary_groups' else opdata['metrics'][chosen]['dual_nmse'])
            write(path, state)
            print(json.dumps(dict(operator=name, chosen=chosen, done=len(state['modules']), total=len(audit['boundary_groups']))), flush=True)
            del native, panels, panel, asset, old, raw
            gc.collect(); torch.cuda.empty_cache()
    accepted, raw_values, weights = [], [], []
    for name, op in original['modules'].items():
        if name in state['modules']:
            entry = state['modules'][name]
            accepted.append(entry['accepted_native']['three']['review_mean'])
            raw_values.append(entry['raw_native']['three']['review_mean'])
            weights.append(entry['accepted_weight_nmse'])
        else:
            accepted.append(op['accepted_native']['joint']['three']['review_mean'])
            raw_values.append(op['raw_native']['joint']['three']['review_mean'])
            weights.append(op['metrics'][op['chosen']['joint']]['dual_nmse'])
    scale = np.mean([op['accepted_native']['scale_only']['three']['review_mean'] for op in original['modules'].values()])
    weight_scale = np.mean([op['metrics'][op['chosen']['scale_only']]['dual_nmse'] for op in original['modules'].values()])
    state.update(complete=True, summary=dict(restored_groups=sum(map(len, audit['boundary_groups'].values())),
        raw_three_review=float(np.mean(raw_values)), accepted_three_review=float(np.mean(accepted)),
        accepted_three_vs_scale_pct=float(100*(np.mean(accepted)/scale-1)),
        accepted_weight_nmse=float(np.mean(weights)), accepted_weight_vs_scale_pct=float(100*(np.mean(weights)/weight_scale-1))))
    write(path, state)
    print(json.dumps(state['summary'], indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    run(p.parse_args().root)
