"""Same-input external baseline work and physical storage, without fake budgets."""
import argparse
import json
from pathlib import Path
from baselines import sha, write


def runtime_source(path):
    visited = set()
    while path not in visited:
        visited.add(path)
        receipt = json.loads(path.read_text())
        if receipt.get('runtime'):
            return receipt['runtime']
        prepared = path.parents[1]/'prepared.json'
        if prepared.exists():
            data = json.loads(prepared.read_text())
            if data['identity']['manifest_sha256'] != receipt['identity']['manifest_sha256']:
                raise ValueError('Changed preparation provenance')
            return data['runtime']
        source = receipt.get('imported_from')
        if not source:
            return None
        path = Path(source['path'])
        if sha(path) != source['sha256']:
            raise ValueError('Changed imported runtime evidence')
    raise ValueError('Cyclic imported runtime provenance')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    parent = json.loads((a.geometry/'manifest.json').read_text())
    rows = []
    for study, methods in (('external', ('bf16', 'fp8', 'arcquant', 'sharq')),
                           ('awq_native', ('awq_marlin',))):
        root = a.root/study
        manifest = json.loads((root/'manifest.json').read_text())
        if manifest['cases'] != parent['cases'] or manifest['model'] != parent['model']:
            raise ValueError('Unmatched method inputs/model')
        for name in methods:
            for entry in manifest['cases']:
                case = entry['case']['id']
                source = a.geometry/'generation/b2_joint'/case
                ledger_path = source/'compute.json'
                geometry = json.loads(ledger_path.read_text())
                witness = json.loads((source/'result.json').read_text())
                if not witness['complete'] or any(r['compute_sha256'] != sha(ledger_path) for r in witness['rows']):
                    raise ValueError('Unverified request geometry')
                request = root/'generation'/name/case/'result.json'
                receipt = json.loads(request.read_text())
                if not receipt['complete'] or receipt['identity']['case'] != entry['case']:
                    raise ValueError('Incomplete/mismatched external request')
                projection = sum(2*r['shape'][0]*r['shape'][1]*r['shape'][2]*r['calls']
                                 for r in geometry['rows'] if r['kind'] == 'linear')
                attention = sum(4*r['shape'][0]*r['shape'][1]*r['shape'][2]*r['shape'][4]*r['calls']
                                for r in geometry['rows'] if r['kind'] == 'attention')
                work = dict(bf16_attention_flops=attention)
                if name in ('bf16', 'awq_marlin'):
                    work['bf16_projection_flops'] = projection
                elif name == 'fp8':
                    work['fp8_projection_flops'] = projection
                elif name == 'arcquant':
                    work['fp4_dense_projection_flops'] = projection*3//2
                else:
                    work.update(fp4_dense_projection_flops=projection,
                        fp4_structured_sparse_nonzero_flops=projection//2,
                        structured_sparse_dense_equivalent_flops=projection)
                runtime = runtime_source(request)
                memory = dict(peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in receipt['rows']),
                              peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in receipt['rows']))
                if runtime:
                    memory.update(resident_model_state_bytes=runtime['resident_model_state_bytes'],
                                  support_buffer_bytes=runtime['support_buffer_bytes'])
                    if name not in ('bf16', 'awq_marlin'):
                        memory['registered_generation_weight_bytes'] = runtime['prepared_weight_bytes']
                if name == 'bf16':
                    shapes = {r['name']: r['shape'] for r in geometry['rows'] if r['kind'] == 'linear'}
                    memory['modeled_bf16_weight_payload_bytes'] = sum(2*n*k for _, n, k in shapes.values())
                if name == 'awq_marlin':
                    adapter = receipt['adapter']
                    memory.update(registered_generation_weight_bytes=adapter['resident_generation_weight_bytes'],
                                  adapter_workspace_bytes=adapter['scratch_bytes'])
                rows.append(dict(study=study, method=name, case=case, benchmark=entry['benchmark'],
                    logical_work=work, memory=memory, geometry_sha256=sha(ledger_path),
                    receipt_sha256=sha(request), actual_precision=receipt.get('actual_generation_precision', name),
                    padded_mma_qualified=False, performance_eligible=False))
    result = dict(complete=True, rows=rows, source_sha256=sha(__file__), performance_eligible=False,
        scope='Same-request generation projection/attention logical FLOPs, not issued tile counts or FP4 budget ratios. '
              'FP4 dense, 2:4 nonzero and dense-equivalent work remain separate; never add the latter two. '
              'Common BF16 AR prefill, codecs, prescale, packing, selection, dequantization and non-MMA work are excluded '
              'from these logical FLOPs but remain part of future complete-request timing. '
              'No native Marlin weight tensor is expanded to a persistent BF16 weight. '
              'Physical adapter weight/workspace bytes are used instead of the generic external Parameter-blind counter. '
              'Peak memory reflects each method execution mode; eager ARC/SharQ and Graph methods have different Graph storage.')
    a.output.mkdir(parents=True, exist_ok=True)
    write(a.output/'baseline_costs.json', result)
    lines = ['# Baseline work and physical storage', '', result['scope'], '',
        '| Method | Projection arithmetic | Generation weight MiB | Allocated peak GiB range |',
        '|---|---|---:|---:|']
    labels = dict(bf16='BF16', fp8='FP8', arcquant='1.5 dense FP4 (fixed 50% channels)',
                  sharq='1 dense FP4 + 0.5 nonzero 2:4 FP4', awq_marlin='BF16 MMA, packed NVFP4 weights')
    for name in labels:
        own = [r for r in rows if r['method'] == name]
        sizes = {r['memory'].get('registered_generation_weight_bytes') for r in own}
        size = 'unavailable' if None in sizes else '/'.join(f'{v/2**20:.3f}' for v in sorted(sizes))
        if name == 'bf16':
            size = '/'.join(f'{v/2**20:.3f}' for v in sorted({r['memory']['modeled_bf16_weight_payload_bytes'] for r in own}))+' (modeled matrix payload)'
        peaks = [r['memory']['peak_allocated_bytes']/2**30 for r in own]
        lines.append(f'| {name} | {labels[name]} | {size} | {min(peaks):.3f}–{max(peaks):.3f} |')
    lines += ['', 'The decoded AWQ reference is excluded from W4 storage claims. Weight bytes are not whole-model VRAM. '
              'BF16 generation matrix payload is derived from distinct observed N/K shapes; its generic prepared counter is empty, not zero actual weight memory. '
              'Other weight columns count registered deployment tensors; raw JSON keeps complete resident model state separately.']
    (a.output/'BASELINE_COSTS.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
