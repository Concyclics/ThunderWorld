"""Reconcile the unquantized resident portion with checkpoint components."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

GIB = 2**30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-directory', type=Path, required=True)
    args = parser.parse_args()
    folder = args.audit_directory
    audit = json.loads((folder / 'audit.json').read_text())
    checkpoints = json.loads((folder / 'checkpoint_inventory.json').read_text())
    output, rows = {}, []
    for model in ('edge', 'nano'):
        ck = next(v for k, v in checkpoints.items() if model in k.lower())
        p = {k: v['bytes'] for k, v in ck['transformer_partition'].items()}
        a = audit['models'][model]
        if p['generation_ffn'] + p['generation_attention_projections'] != a['bf16_target_bytes']:
            raise ValueError('Generation target partition disagrees with execution shapes')
        groups = {
            'AR_understanding_FFN': p['understanding_ffn'],
            'AR_understanding_attention_projections': p['understanding_attention_projections'],
            'text_embedding': p['text_embedding'],
            'language_output_head': p['language_output_head'],
            'VAE': ck['components']['vae']['bytes'],
            # Cosmos3 explicitly lists time_embedder in _keep_in_fp32_modules.
            'timestep_embedding_FP32': 2 * p['timestep_embedding'],
            'action_heads': p['action_heads'],
            'audio_heads': p.get('audio_heads', 0),
            'video_input_output_projections': p['video_input_output_projections'],
            'norms': p['norms'],
        }
        groups['remaining_buffers'] = a['unchanged_resident_bytes'] - sum(groups.values())
        # The residual equals the 64-element FP32 rotary-frequency buffer.
        # No optional vision encoder or sound tokenizer is needed to reconcile
        # this I2V resident inventory; their disk presence is not GPU residency.
        if groups['remaining_buffers'] != 256:
            raise ValueError('Unexpected resident remainder; inspect loaded components')
        ar = groups['AR_understanding_FFN'] + groups['AR_understanding_attention_projections']
        output[model] = dict(unquantized_resident_bytes=a['unchanged_resident_bytes'],
            component_bytes=groups, exact_reconciliation=True,
            AR_fraction_of_unquantized=ar/a['unchanged_resident_bytes'],
            estimates_not_measurements=dict(
                AR_W8_payload_saving_bytes=ar/2,
                AR_W4_single_K16_scale_saving_bytes=ar*(1-4.5/16),
                removable_I2V_lm_head_weight_bytes=groups['language_output_head']),
            scope='BF16 checkpoint component sizes reconciled with postwarm resident state. Runtime adds FP32 timestep weights and a 256-byte buffer. No new live-tensor inventory or equivalence experiment.')
        for name, size in groups.items():
            rows.append(dict(model=model, component=name, bytes=size, gib=size/GIB,
                fraction_of_unquantized=size/a['unchanged_resident_bytes']))
    (folder / 'unquantized_partition.json').write_text(json.dumps(dict(models=output,
        sources={name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
                 for name in ['audit.json', 'checkpoint_inventory.json']},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')
    with (folder/'unquantized_partition.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(output))


if __name__ == '__main__':
    main()
