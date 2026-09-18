"""CPU-only two-model benchmark inventory. Does not start any generation job."""
import argparse
import ast
import collections
import copy
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ('bf16', 'fp8', 'fp4_base', 'awq_marlin', 'arcquant', 'arcquant100',
           'sharq_perf', 'b1.5_protected_n', 'b2_joint', 'b2.5_ffn_guarded')
ANCHORS = ('fp4_three', 'fp4_four')
OURS = PRIMARY[-3:]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Inventory snapshots are immutable; choose a new output directory')
    w = a.workspace
    cohort_file = w/'repair_redesign_20260917/cohort.json'; cohort = read(cohort_file)
    old_file = w/'fp4_baseline_sweep_20260918/manifest.json'; old = read(old_file)
    optimized_file = w/'fp4_next_20260918/optimized-study/manifest.json'; optimized = read(optimized_file)
    anchor_file = w/'paper_evidence_20260918/cross/manifest.json'; anchor = read(anchor_file)
    if old['model'] != optimized['model'] or old['model'] != cohort['model'] or old['model'] != anchor['model']:
        raise ValueError('Changed model between evidence sources')
    tree = ast.parse((ROOT/'tools/cosmos_e2e/download.py').read_text())
    models = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(x, ast.Name) and x.id == 'MODELS' for x in node.targets))
    model_rows = {}
    for name, (repo, revision, folder) in models.items():
        path = w/'ThunderWorld/artifacts/huggingface/models'/folder
        config = read(path/'transformer/config.json')
        index_file = path/'transformer/diffusion_pytorch_model.safetensors.index.json'
        index = read(index_file)['weight_map']
        targets = sorted(k.removesuffix('.weight') for k in index if k.startswith('layers.') and k.endswith('.weight') and
                         ('.mlp_moe_gen.' in k or '.self_attn.add_' in k or '.self_attn.to_add_out.' in k))
        model_rows[name] = dict(repo=repo, revision=revision, local_path=str(path),
            architecture={key: config[key] for key in ('num_hidden_layers', 'num_attention_heads', 'num_key_value_heads',
                                                       'hidden_size', 'intermediate_size', 'hidden_act', 'head_dim')},
            model_identity={str(f.relative_to(path)): sha(f) for f in (path/'transformer/config.json', index_file)},
            generation_projection_count=len(targets), generation_targets=targets,
            frozen_numeric_assets=name == 'edge')
    final = cohort['cases']['final']
    if len(final) != 72 or collections.Counter(e['benchmark'] for e in final) != {'rbench': 36, 'paibench': 36}:
        raise ValueError('Changed pre-reserved final cohort')
    if any(len(e['seeds']) != 2 or sha(e['case']['image']) != e['image_sha256'] for e in final):
        raise ValueError('Changed input/seed count')
    source_ids = {(e['benchmark'], e['source_id']) for e in final}
    pixels = {e['pixel_sha256'] for e in final}
    if len(source_ids) != 72 or len(pixels) != 72:
        raise ValueError('Duplicate final inputs')
    disjoint = {}
    for role in ('calibration', 'development', 'validation'):
        other = cohort['cases'][role]
        disjoint[role] = dict(source_ids=len(source_ids & {(e['benchmark'], e['source_id']) for e in other}),
                              decoded_pixels=len(pixels & {e['pixel_sha256'] for e in other}))
        if any(disjoint[role].values()):
            raise ValueError('Final inputs overlap fitting cohorts')
    # Match any numeric seed token, including case_seedNNN and other saved-output
    # conventions. This is a local filename audit, not proof about remote jobs.
    files = subprocess.run(['rg', '--files', '--hidden', '--no-ignore', '-g', '*.npz', str(w)], capture_output=True, text=True)
    if files.returncode not in (0, 1) or files.stderr:
        raise RuntimeError('Output archive audit failed: '+files.stderr[:500])
    seeds = {s for e in final for s in e['seeds']}
    matches = [path for path in files.stdout.splitlines() if seeds & set(map(int, re.findall(r'\d+', Path(path).stem)))]
    catalog = {}; proofs = {}
    for name in (*PRIMARY, *ANCHORS):
        source = optimized if name in OURS else anchor if name in ANCHORS else old
        spec = copy.deepcopy(source['methods'][name])
        if spec.get('assets') and sha(Path(spec['assets'])/'manifest.json') != spec['assets_sha256']:
            raise ValueError('Changed offline assets: '+name)
        if spec['config'].get('budget_plan') and sha(spec['config']['budget_plan']) != spec['plan_sha256']:
            raise ValueError('Changed plan: '+name)
        if name in OURS:
            proof = optimized['execution_revision']['qualification'][name]; result = read(proof['receipt'])
            if sha(proof['receipt']) != proof['sha256'] or not result['complete'] or len(result['cases']) != 16:
                raise ValueError('Missing complete norm qualification')
            rows = [r for case in result['cases'].values() for r in case['rows']]
            if len(rows) != 32 or not all(r['source_hashes_match'] is True for r in rows):
                raise ValueError('Execution optimization changed output bytes')
            if spec['runtime_options'] != result['identity']['effective_spec']['runtime_options']:
                raise ValueError('Changed qualified runtime options')
            original = old['methods'][name]
            if read(original['config']['budget_plan']) != read(spec['config']['budget_plan']):
                raise ValueError('Changed numerical repair plan')
            proofs[name] = proof
        catalog[name] = dict(tier='primary' if name in PRIMARY else 'quality_anchor',
            edge=dict(status='frozen_configuration_available__new_cohort_preflight_pending', spec=spec),
            nano=dict(status='model_specific_calibration_and_qualification_pending' if name in OURS or name in ('awq_marlin', 'arcquant', 'arcquant100') else 'current_model_and_execution_qualification_pending',
                      spec=None, do_not_reuse_edge_assets=True),
            edge_external_adapter=old.get('adapters', {}).get(name),
            attention=spec['config']['attention_backend'], execution=spec['config']['execution'])
    total_requests = len(final)*2
    result = dict(schema=1, inventory_complete=True, matrix_status='preparation_only', generation_started=False,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256=sha(__file__), models=model_rows, methods=catalog, primary_methods=list(PRIMARY),
        quality_anchors=list(ANCHORS), qualified_edge_execution=proofs,
        sources={str(path): sha(path) for path in (cohort_file, old_file, optimized_file, anchor_file)},
        audit=dict(local_npz_files_scanned=len(files.stdout.splitlines()), final_seed_filename_matches=matches,
                   final_images_present_and_hash_matched=True, disjoint=disjoint, remote_history_audited=False,
                   prior_final_state=read(w/'paper_evidence_20260918/final/orchestrator.json')),
        quality=dict(scope='I2V, full pre-reserved 72-scene subset; not the complete official datasets or their leaderboard protocol',
                     scenes_per_dataset={'rbench': 36, 'paibench': 36}, seeds_per_scene=2, seeds=[min(seeds), max(seeds)],
                     categories=dict(collections.Counter(e['benchmark']+'/'+e['category'] for e in final)),
                     task='i2v', steps=35, cfg=6, frames=61, height=192, width=320,
                     ar_precision='bf16', ar_prefill='fresh per request; same caching policy', debug='off',
                     metrics=['lpips', 'ssim', 'psnr', 'video_relative_l2'], fit_on_final=False,
                     aggregation='Report each dataset separately; category macro means; average paired seeds inside each scene before category-stratified bootstrap',
                     confidence='Paired 95% scene-cluster CI; simultaneous claims require predeclared familywise correction',
                     guard='Lower LPIPS, no mean SSIM/L2 regression; PSNR reported, not substituted for a failed guard',
                     primary_reference='ARCQuant 100%', secondary_references=['AWQ-lite/Marlin', 'SharQ', 'plain W4A4'],
                     fp8_gate='Separate strict consistency gate; failures remain in the results',
                     inferential_family_size=12, inferential_family='3 budgets x 2 models x 2 datasets vs ARCQuant100; separate family for FP8 claims',
                     one_model_one_method_requests=total_requests,
                     both_models_primary_requests=total_requests*len(PRIMARY)*2,
                     both_models_including_anchors_requests=total_requests*(len(PRIMARY)+len(ANCHORS))*2),
        performance=dict(devices=['RTX PRO 6000', 'RTX 5090'], separate_device_tables=True,
                         formal_latency=dict(cases='same 16 development scenes; frozen shapes/categories', rounds=5, seeds_per_scene_per_round=3,
                             method_order='fixed randomized interleaving in each round; independent load and complete warmup',
                             debug='off', idle_required=True, fully_prepared_required=True,
                             statistics=['median', 'p90', 'videos_per_second', 'output_fps'],
                             time_scope='AR prefill + 35 steps + video decode + CPU output; exclude offline preparation and file serialization'),
                         throughput=dict(sizes=[1,2,4,8], kind='independent model replicas, not merged GEMM batch or shared weights',
                             cases='one previously fixed RBench and one PAI input', rounds=5, waves=3,
                             quality_for_concurrency=False, baseline_specific_tuning=False,
                             stop='resource limit/OOM, or two doublings below 5% gain; do not lower precision or alter input'),
                         memory=['cold_load_peak', 'resident_weights', 'w1_union_and_indices', 'scratch', 'graph_allocated_peak', 'reserved_peak'],
                         readiness='Current diagnostic workers hardcode performance_eligible=false; formal common scheduler and device coverage qualification remain required'),
        mechanism_appendix=dict(required=['single_a_act2', 'single_w_act2', 'joint2_act2'], optional=['joint1.5_act2'],
            shared_attention='full FP4 ACT2, all base blocks retained; K/V are activations, not model weights',
            full_cohort_extra_requests=total_requests*3*2,
            not_in_primary_table=['FourOverSix scale control', 'ViDiT-Q inspired schedule', 'SageAttention3 operator comparison', 'ModelOpt lite-SVDQuant operator reference'],
            note='Do not present mechanism reproductions or operator-only ports as complete official Cosmos baselines'),
        extensions=dict(tasks=['Edge policy', 'Nano policy', 'Nano AV'], status='separate task-specific cohorts/assets/protocol required',
                        note='RBench/PAI I2V inputs do not define an action-success benchmark or paired audio reference'),
        launch_gaps=[
            'Generalize one fresh-output worker to baseline_runtime.make_runner for every method; retain optional prior-hash checks, never require nonexistent final output hashes.',
            'dataset_infer.py bypasses runtime_options and ARC/SharQ adapters; baseline_sweep.quality admits only ARC100/SharQ. Neither is a complete final-matrix launcher.',
            'Persist exact qualified Edge assets/plans currently referenced under /tmp, with hash checks and independent immutable output directories.',
            'Build Nano-specific assets/plans and ARC orders/AWQ calibration; qualify its 36 layers, 32 heads and SwiGLU without reusing Edge quota tables.',
            'Preflight actual final input shapes and compute caps without reading final quality to change policies; missing support is a failure, not a silent fallback.',
            'Qualify PRO6000/5090 separately; no comparison of different GPUs or profiling-enabled measurements.',
            'Archive remote run receipts for final seed/case audit before claiming globally unseen evaluation.'
        ])
    a.output.mkdir(parents=True)
    (a.output/'inventory.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    (a.output/'final_cases.json.gz').write_bytes(gzip.compress(json.dumps(final, ensure_ascii=False).encode(), mtime=0))
    # External dependency/build identities stay explicit alongside frozen specs.
    (a.output/'edge_external.json').write_text(json.dumps(old.get('external', {}), indent=2)+'\n')
    print(json.dumps(dict(models={name: row['generation_projection_count'] for name,row in model_rows.items()},
        primary=len(PRIMARY), anchors=len(ANCHORS), samples=result['quality']['both_models_including_anchors_requests'],
        local_seed_matches=len(matches), generation_started=False)))


if __name__ == '__main__':
    main()
