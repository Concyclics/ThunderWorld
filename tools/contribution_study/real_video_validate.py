"""Check frozen generation receipts and independently recompute recorded-video errors."""
import argparse
import json
import math
import os
from pathlib import Path

from baselines import sha, write


def validate(root):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    import lpips
    from skimage.metrics import structural_similarity
    torch.set_num_threads(4)
    torch.hub.set_dir('/tmp/thunderworld-lpips-cache')
    lpips_model = lpips.LPIPS(net='alex', version='0.1').eval().cpu()
    manifest = root / 'edge/final/manifest.json'
    study = json.loads(manifest.read_text())
    refs = json.loads(Path(study['reference_manifest']).read_text())
    receipts, checks, sampled = {}, [], []
    model_identity = None
    target_set = None
    samples = {('bf16', 'paibench_c__task_0054'), ('fp4_base', 'paibench_c__task_0259'),
               ('fp4_sixteen', 'paibench_c__task_0549')}
    for name, spec in study['methods'].items():
        prepared = json.loads((root / 'edge/final/generation' / name / 'prepared.json').read_text())
        runtime = prepared['runtime']
        if model_identity is None:
            model_identity = runtime['model_identity']
        if runtime['model_identity'] != model_identity:
            raise ValueError('Different model versions')
        adapter = prepared['adapter']
        if spec.get('expansion'):
            if (adapter['execution_type'] != 'native_fp4_tensor_core' or
                    adapter['specification'] != spec['expansion'] or
                    not adapter['current_input_activation_packing']):
                raise ValueError('Incorrect expansion backend or semantics')
            targets = set(adapter['targets'])
            if target_set is None:
                target_set = targets
            if targets != target_set or len(targets) != 168:
                raise ValueError('Unmatched FP4 projection scope')
        for entry in study['cases']:
            cid = entry['case']['id']
            folder = root / 'edge/final/generation' / name / cid
            receipt = folder / 'result.json'
            result = json.loads(receipt.read_text())
            identity = result['identity']
            if (not result['complete'] or identity['manifest_sha256'] != sha(manifest) or
                    identity['method'] != name or identity['case'] != entry['case'] or
                    identity['seeds'] != entry['seeds'] or
                    identity['image_sha256'] != refs['cases'][cid]['input_image_sha256'] or
                    len(result['rows']) != 1):
                raise ValueError('Changed generation input')
            row = result['rows'][0]
            if (row['seed'] != entry['seeds'][0] or row['steps'] != 35 or row['calls'] != 70 or
                    row['debug'] != 'off' or row['execution'] != 'graph' or row['performance_eligible']):
                raise ValueError('Changed execution settings')
            # Exported video metadata points here; preserve the complete method log.
            log = folder / 'generation.log'
            target = root / 'logs' / (name + '.log')
            if not log.exists():
                log.symlink_to(target)
            if log.resolve() != target.resolve() or not target.is_file():
                raise ValueError('Missing generation log')
            source = folder / row['outputs']
            if sha(source) != row['outputs_sha256']:
                raise ValueError('Corrupt generated output')
            ref = refs['cases'][cid]
            if sha(ref['source_file']) != ref['source_sha256'] or sha(ref['aligned_file']) != ref['aligned_sha256']:
                raise ValueError('Changed real video')
            with np.load(source) as data:
                video = data['video']
            with np.load(ref['aligned_file']) as data:
                reference = data['video']
            if video.shape != (61, 192, 320, 3) or reference.shape != video.shape:
                raise ValueError('Changed dimensions')
            record = json.loads((root / 'edge/final/metrics_real' / name / (cid + '.json')).read_text())
            if record['provenance']['generation_receipt_sha256'] != sha(receipt):
                raise ValueError('Metrics from a different request')
            for scope, start in [('future', 1), ('all_frames', 0)]:
                a, b = video[start:].astype(np.float64), reference[start:].astype(np.float64)
                if not np.isfinite(a).all() or not np.isfinite(b).all() or min(a.min(), b.min()) < 0 or max(a.max(), b.max()) > 1:
                    raise ValueError('Invalid raw RGB range')
                error_sum = float(np.sum((a-b)**2))
                mse = error_sum / a.size
                values = dict(mse=mse, rmse=math.sqrt(mse), psnr_db=10*math.log10(1/mse),
                              relative_l2=math.sqrt(error_sum / float(np.sum(b*b))))
                published = record['rows'][0][scope]
                differences = {k: abs(v-published[k]) for k, v in values.items()}
                if max(differences.values()) > 1e-10:
                    raise ValueError('Independent signal metric disagrees')
                checks.append(dict(method=name, case_id=cid, scope=scope, absolute_errors=differences))
            if (name, cid) in samples:
                ssim, perceptual = [], []
                with torch.inference_mode():
                    for a, b in zip(video, reference):
                        ssim.append(structural_similarity(a, b, data_range=1., channel_axis=-1))
                        x = torch.from_numpy(a).permute(2, 0, 1)[None]*2-1
                        y = torch.from_numpy(b).permute(2, 0, 1)[None]*2-1
                        perceptual.append(float(lpips_model(x, y).item()))
                differences = {}
                for scope, start in [('future', 1), ('all_frames', 0)]:
                    published = record['rows'][0][scope]
                    differences[scope] = dict(ssim=abs(float(np.mean(ssim[start:]))-published['ssim_mean']),
                        lpips=abs(float(np.mean(perceptual[start:]))-published['lpips']))
                    if max(differences[scope].values()) > 1e-6:
                        raise ValueError('Framewise perceptual recomputation disagrees')
                sampled.append(dict(method=name, case_id=cid, absolute_errors=differences))
            receipts[str(receipt.relative_to(root))] = sha(receipt)
    state = dict(complete=True, manifest_sha256=sha(manifest), source_sha256=sha(__file__),
        generation_receipts=receipts, matched_requests=len(receipts), checked_signal_scopes=len(checks),
        signal_checks=checks, perceptual_spot_checks=sampled,
        perceptual_scope='Three declared clips; single-frame LPIPS evaluation versus original batched evaluation',
        model_identity=model_identity, fp4_projection_count=len(target_set), performance_eligible=False)
    write(root / 'validation.json', state)
    return dict(complete=True, matched_requests=len(receipts), signal_scopes=len(checks),
                perceptual_spot_checks=len(sampled))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.root.resolve())))
