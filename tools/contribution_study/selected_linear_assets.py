"""Restore full offline W1 controls while preserving every selected W0 bit.

Use original BF16 weights and the frozen prescale/codec, never inference inputs
or output metrics. Verify W0, outer factors and already stored W1 byte-for-byte.
"""
import copy
import json
from pathlib import Path

from baselines import sha, write


def restore(root, selected_manifest, capture_manifest):
    import torch
    from thunderworld_fp4._calibrated import pack
    from thunderworld_fp4.cosmos.assets import compact
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    selected = json.loads(selected_manifest.read_text())
    capture = json.loads(capture_manifest.read_text())
    if not capture['complete']:
        raise ValueError('Incomplete original weight capture')
    provenance = dict(selected_manifest_sha256=sha(selected_manifest),
        capture_manifest_sha256=sha(capture_manifest), source_sha256=sha(__file__))
    with shared_worker(), torch.inference_mode():
        resource = select_shared_device()
        torch.set_num_threads(8)
        for origin in ('b1.5_protected_n', 'b2_joint'):
            source = Path(selected['methods'][origin]['assets'])
            original = json.loads((source/'manifest.json').read_text())
            if sha(source/'manifest.json') != selected['methods'][origin]['assets_sha256']:
                raise ValueError('Changed selected bank')
            if original['model_identity'] != capture['model_identity']:
                raise ValueError('Different model for full-W1 restoration')
            bank = root/'full_control_assets'/origin
            bank.mkdir(parents=True, exist_ok=True)
            output = bank/'manifest.json'
            manifest = dict(complete=False, precision='fp4', model_identity=original['model_identity'],
                modules={}, cohort_sha256=original['cohort_sha256'], provenance=provenance,
                selected_bank_sha256=sha(source/'manifest.json'), resource=resource)
            if output.exists():
                saved = json.loads(output.read_text())
                if saved['provenance'] != provenance or saved['selected_bank_sha256'] != manifest['selected_bank_sha256']:
                    raise ValueError('Changed full-bank resume identity')
                manifest = saved
            for index, (name, entry) in enumerate(original['modules'].items()):
                if name in manifest['modules']:
                    done = manifest['modules'][name]
                    if sha(bank/done['file']) != done['sha256']:
                        raise ValueError('Changed restored asset')
                    continue
                if sha(source/entry['file']) != entry['sha256']:
                    raise ValueError('Changed compressed selected asset')
                asset = torch.load(source/entry['file'], map_location='cpu', weights_only=True)
                if asset.get('mandatory_w_blocks'):
                    raise ValueError('Pure-A control cannot discard mandatory outlier W1')
                row = capture['weights'][name]
                weight_file = capture_manifest.parent/row['file']
                if sha(weight_file) != row['sha256']:
                    raise ValueError('Changed original BF16 weight')
                w = torch.load(weight_file, map_location='cpu', weights_only=True).cuda()
                d = asset['prescale'].cuda()
                base, res = asset.get('weight_scale', (1., 1.))
                c, s, f, _ = pack((w.float()/d).bfloat16().contiguous(),
                                  torch.ones_like(d), True, True, w, base=base, res=res)
                full = {k: copy.deepcopy(v) for k,v in asset.items() if k not in (
                    'storage_schema','weight_support_fraction','residual_indices',
                    'codes_base','codes_residual','scales_base','scales_residual')}
                full.update(codes=c.cpu(), scales=s.cpu(), factor=f.cpu())
                check = compact(full, asset['residual_indices'].tolist())
                for key in ('codes_base','codes_residual','scales_base','scales_residual','factor','prescale'):
                    if not torch.equal(check[key].contiguous().reshape(-1).view(torch.uint8),
                                       asset[key].contiguous().reshape(-1).view(torch.uint8)):
                        raise ValueError(f'Restored numerical representation differs: {name}/{key}')
                target = bank/f'{index:03d}.pt'
                torch.save(compact(full, 1.), target)
                manifest['modules'][name] = dict(file=target.name, sha256=sha(target),
                    shape=list(w.shape), selected_base_bitwise_equal=True,
                    selected_residual_bitwise_equal=True, source_weight_sha256=row['sha256'])
                write(output, manifest)
                if (index+1)%24 == 0:
                    print(json.dumps(dict(bank=origin, completed=index+1, total=len(original['modules']))), flush=True)
                del w, d, c, s, f, full, check, asset
            manifest['complete'] = True
            write(output, manifest)
    return dict(complete=True, provenance=provenance)
