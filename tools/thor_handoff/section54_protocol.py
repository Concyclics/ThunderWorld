"""CPU identity/asset checks and small reporting helpers for the FP4 retest."""
import csv
import hashlib
import json
from pathlib import Path
import struct
from build_baselines import sha, write

EXTERNAL = {'arcquant':'arcquant50','arcquant100':'arcquant100','sharq_perf':'sharq','awq_marlin':'awq_marlin'}


def model_dimensions(model):
    result = {}
    for file in sorted((Path(model)/'transformer').glob('*.safetensors')):
        with file.open('rb') as stream:
            header = json.loads(stream.read(struct.unpack('<Q', stream.read(8))[0]))
        for name, meta in header.items():
            if not name.endswith('.weight'):continue
            module = name[:-7]
            if ('.mlp_moe_gen.' in module or module.endswith(tuple('.self_attn.'+v for v in
                    ('add_q_proj','add_k_proj','add_v_proj','to_add_out')))) and len(meta['shape'])==2:
                if module in result:raise ValueError('Duplicate checkpoint tensor')
                result[module]=meta['shape']
    if not result:raise ValueError('Missing checkpoint projection headers')
    return result


def freeze_assets(root, model_key, model, methods, dimensions, output):
    paths = dict(arc_orders=str(root/'data/arc_orders' if model_key=='edge' else root/'data/nano/arc_orders'),
                 awq_marlin=str(root/'assets/awq_marlin' if model_key=='edge' else root/'assets/nano/awq_marlin'))
    identity={str(f.relative_to(model)):sha(f) for f in [model/'transformer/config.json',model/'transformer/diffusion_pytorch_model.safetensors.index.json'] if f.exists()}
    receipts={}
    for key,needed in [('arc_orders',any(m.startswith('arcquant') for m in methods)),('awq_marlin','awq_marlin' in methods)]:
        if not needed:continue
        folder=Path(paths[key]); manifest=json.loads((folder/'manifest.json').read_text())
        if not manifest['complete'] or manifest['model_identity']!=identity:raise ValueError('Asset model mismatch: '+key)
        entries=manifest['modules'].values() if key=='awq_marlin' else [manifest]
        if set(manifest['modules'] if key=='awq_marlin' else manifest['tensor_hashes'])!=set(dimensions):
            raise ValueError('Asset target names mismatch: '+key)
        total=0
        for entry in entries:
            f=folder/entry['file']
            if sha(f)!=entry['sha256']:raise ValueError('Asset checksum mismatch: '+str(f))
            total+=f.stat().st_size
        receipts[key]=dict(path=str(folder),manifest_sha256=sha(folder/'manifest.json'),payload_bytes=total,
                           model_identity=identity,targets=len(dimensions),complete=True)
    write(output/'assets-verified.json',receipts)
    return paths,receipts


def write_csv(path, rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def statistics_rows(records, method, model, manifest_sha):
    import statistics
    from attribution_report import percentile
    result=[]
    for case in ['all']+sorted({r['case'] for r in records}):
        data=[r for r in records if case=='all' or r['case']==case]
        times=[r['wall_seconds'] for r in data]
        median=statistics.median(times)
        result.append(dict(model=model,method_id=method,case=case,rounds=len({r['round'] for r in data}),
            requests=len(data),median_seconds=median,p90_seconds=percentile(times,.9),
            new_frame_fps_at_median=60/median,peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in data),
            peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in data),manifest_sha256=manifest_sha,
            performance_eligible=False))
    return result
