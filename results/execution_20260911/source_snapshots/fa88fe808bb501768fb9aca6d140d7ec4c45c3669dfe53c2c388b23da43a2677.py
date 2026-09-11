"""Fresh full-trajectory head sampling and controlled quality ablations."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image
import diffusers.utils.import_utils as imports
imports._xformers_available = False
from diffusers import Cosmos3OmniPipeline, Cosmos3OmniTransformer, UniPCMultistepScheduler
from diffusers.models.autoencoders.autoencoder_cosmos3_audio import Cosmos3AVAEAudioTokenizer
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import CosmosActionCondition
from diffusers.utils import export_to_video
from runtime import ROOT, AttentionStudy, Observer
sys.path.insert(0, str(ROOT / 'tools/cosmos_e2e'))
from benchmark import quality


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cases():
    return [
        dict(id='robot_observe', task='i2v', split='development', image='data_local/e2e_inputs/robot_153.jpg',
             prompt='A stationary camera observes a robot arm moving smoothly toward an object on the table. The lighting and background remain unchanged.'),
        dict(id='robot_table', task='i2v', split='evaluation', image='data_local/finding_inputs/robot_table.png',
             prompt='The robot arm picks up a small object from the table and places it beside the other objects. A stationary camera records the movement.'),
        dict(id='waterfall', task='t2v', split='development',
             prompt='A waterfall flows steadily over rocks into a clear pool in a forest. Water splashes naturally. A stationary camera captures the scene.'),
        dict(id='pottery', task='t2v', split='evaluation',
             prompt='A close view of a potter shaping a clay bowl on a spinning pottery wheel. Hands move carefully as the bowl becomes wider. The camera remains steady.'),
        dict(id='policy_observe', task='policy', split='development', image='data_local/e2e_inputs/robot_153.jpg',
             prompt='Move the robot gripper toward the object on the table.'),
        dict(id='policy_table', task='policy', split='evaluation', image='data_local/finding_inputs/robot_table.png',
             prompt='Pick up the object and place it next to the other objects.'),
        dict(id='av_waterfall', task='av', split='development',
             prompt='A waterfall flows over rocks in a forest, with synchronized rushing water and splashing sounds. The camera remains steady.'),
        dict(id='av_pottery', task='av', split='evaluation',
             prompt='A potter shapes a clay bowl on a spinning wheel. The quiet room has the soft sound of the wheel rotating and hands touching wet clay.'),
    ]


def call_args(case, seed, steps):
    args = dict(prompt=case['prompt'], negative_prompt='distorted objects, flickering, blurry, low quality',
                num_inference_steps=steps, generator=torch.Generator(device='cuda').manual_seed(seed),
                output_type='np', add_resolution_template=False, add_duration_template=False,
                enable_safety_check=False, enable_sound=case['task'] == 'av')
    if case['task'] == 'policy':
        args.update(action=CosmosActionCondition(mode='policy', chunk_size=16, domain_name='bridge_orig_lerobot',
                    resolution_tier=256, image=Image.open(ROOT / case['image']).convert('RGB'),
                    view_point='third_person_view'), fps=5., guidance_scale=1.)
    else:
        args.update(height=192, width=320, num_frames=61, fps=24., guidance_scale=6.)
        if 'image' in case:
            args['image'] = Image.open(ROOT / case['image']).convert('RGB')
    return args


def signal_quality(signal, reference):
    a, b = np.asarray(signal, np.float64), np.asarray(reference, np.float64)
    return dict(shape=list(a.shape), relative_l2=float(np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-12)),
                rmse=float(np.sqrt(np.mean((a-b)**2))), max_abs=float(np.max(np.abs(a-b))))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['edge', 'nano'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', nargs='+')
    p.add_argument('--seeds', nargs='+', type=int, default=[6101, 6102])
    p.add_argument('--steps', type=int, default=35)
    p.add_argument('--sample', action='store_true')
    p.add_argument('--samples-per-role', type=int, default=32)
    p.add_argument('--modes', nargs='+', default=['bf16'])
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--policy', type=Path)
    p.add_argument('--budget-policy',type=Path)
    p.add_argument('--capture-budget',type=Path)
    p.add_argument('--share-ffn-pack',action='store_true')
    p.add_argument('--phase-graph',action='store_true')
    p.add_argument('--fuse-ffn',action='store_true')
    a = p.parse_args()
    if a.share_ffn_pack and not a.budget_policy:raise ValueError('Shared FFN packing requires budget runtime')
    if a.fuse_ffn and not a.budget_policy:raise ValueError('FFN fusion requires budget runtime')
    if a.phase_graph and (not a.budget_policy or a.sample or a.capture_budget or any(m!='bf16' and not m.startswith('ffn_') for m in a.modes)):
        raise ValueError('Phase graph requires budget policy and BF16/FFN-only modes without sampling/capture')
    if a.capture_budget and (a.modes!=['bf16'] or a.sample):
        raise ValueError('Budget capture requires only BF16, separate from head sampling')
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    selected = [c for c in cases() if (a.cases is None or c['id'] in a.cases) and (a.model == 'nano' or c['task'] != 'av')]
    if a.modes[0] != 'bf16':
        raise ValueError('BF16 reference must run first')
    if a.sample and a.modes != ['bf16']:
        raise ValueError('Observation and performance/ablation requests must be separate')
    model_path = ROOT / 'models' / ('Cosmos3-Edge-a9d944e' if a.model == 'edge' else 'Cosmos3-Nano-411f42a')
    config = json.loads((model_path / 'transformer/config.json').read_text())
    source_paths=[*Path(__file__).parent.glob('*.py'), *(ROOT/'tools/auto_budget').glob('*.py'), ROOT/'experiments/ffn_block_repair_20260907/native.py',
                  *sorted((ROOT/'packages/fp4-repair/src/thunderworld_fp4').glob('*.py'))]
    source_paths.append(ROOT/'tools/cosmos_e2e/graph.py')
    snapshot=a.output/'source_snapshots';snapshot.mkdir()
    hashes={}
    for f in source_paths:
        content=f.read_bytes();sha=hashlib.sha256(content).hexdigest()
        hashes[str(f.relative_to(ROOT))]=sha
        (snapshot/(sha+'.py')).write_bytes(content)
    report = dict(complete=False, model=a.model, model_config=config,
                  args={k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()}, cases=selected,
                  source_sha256=hashes,
                  model_revision={'edge':'a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba',
                                  'nano':'411f42a8fdfb8c5b2583cb8786e0938f49796eaa'}[a.model],
                  diffusers_commit=subprocess.check_output(['git','-C',str(ROOT/'data_local/e2e_src/diffusers'),'rev-parse','HEAD'],text=True).strip(),
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  scope='Full Generator encode/denoise/decode; eager for all modes; sampling wall time is not a performance result',
                  policy_input_limit='Robot observation images; bridge_orig_lerobot interface/domain selected, without matched action ground truth. Not a task success benchmark.',
                  ffn='BF16 for baseline/sampling. All attention ablations share model-local RTN W4 + full ACT2 FFNs. FFN term ablations keep BF16 attention.',
                  rows=[])
    for c in selected:
        if 'image' in c:
            c['image_sha256'] = digest(ROOT / c['image'])
    def save():
        (a.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    save()
    transformer, info = Cosmos3OmniTransformer.from_pretrained(str(model_path / 'transformer'),
                torch_dtype=torch.bfloat16, local_files_only=True, output_loading_info=True)
    report['loading_info'] = info
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise RuntimeError(info)
    audio = None
    if any(c['task'] == 'av' for c in selected):
        audio = Cosmos3AVAEAudioTokenizer.from_pretrained(str(model_path / 'sound_tokenizer'),
                                    torch_dtype=torch.bfloat16, local_files_only=True)
    pipe = Cosmos3OmniPipeline.from_pretrained(str(model_path), transformer=transformer, sound_tokenizer=audio,
                safety_checker=None, enable_safety_checker=False, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.transformer.set_attention_backend('_native_flash')
    pipe.vae.enable_tiling()
    pipe.to('cuda')
    if audio is not None:
        # Decode saved audio latents in a separate process after model release.
        # This avoids WDDM memory spill from decoder convolution workspaces.
        audio.to('cpu')
        pipe.decode_sound = lambda latent: latent
        report['audio_placement']='Audio decoder stays on CPU. Save generated sound latents; waveform decoding runs separately after the Transformer is released. AV wall time excludes waveform decode.'
        torch.cuda.empty_cache()
    study = AttentionStudy(pipe.transformer)
    capture=None;budget_controller=None
    if a.capture_budget or a.budget_policy:
        sys.path.insert(0,str(ROOT/'tools/auto_budget'))
    if a.capture_budget:
        from capture import Capture
        capture=Capture(pipe.transformer,study,a.capture_budget)
    if a.budget_policy:
        from dynamic import BudgetController
        budget_policy=json.loads(a.budget_policy.read_text())
        if budget_policy['model_revision']!=report['model_revision']:raise ValueError('Budget/model revision mismatch')
        budget_controller=BudgetController(study,budget_policy,share_pack=a.share_ffn_pack)
        report['budget_policy']=budget_policy
        report['budget_scope']='Layer/phase/CFG A/W tier allocation; partial A supports use current-input GPU energy ranks; BF16 attention'
        report['ffn']='BF16 baseline; budget modes share model-local RTN W0/W1 and choose base, half/full A, full W, or their first-order combination. BF16 attention. W1 remains allocated across tiers.'
    if a.policy is not None:
        study.group_policy=json.loads(a.policy.read_text())
        if study.group_policy['model'] != a.model:
            raise ValueError('Policy/model mismatch')
        report['group_policy']=study.group_policy
    references, anchors, repair_references = {}, {}, {}
    modules = None
    phase_graph=None
    if a.phase_graph:
        from phase_graph import PhaseGraph
        phase_graph=PhaseGraph(pipe.transformer.forward,study,budget_controller,lambda:modules or [],a.steps)
        pipe.transformer.forward=phase_graph
        report['scope']='Full Generator encode/denoise/decode; phase/branch-aware whole-transformer CUDA Graph for all modes; current inputs copied and outputs owned'
    sketches = []
    def output_sketch(module, args, output):
        values = []
        outputs = (output.sample, output.sound, output.action) if hasattr(output, 'sample') else output
        for value in outputs:
            if value is None:
                values.append(np.zeros(256, np.float32))
            else:
                tensor = value[0].flatten()
                indices = torch.linspace(0, len(tensor)-1, 256, device=tensor.device).long()
                values.append(tensor[indices].float().cpu().numpy())
        sketches.append(np.stack(values))
    pipe.transformer.register_forward_hook(output_sketch)
    try:
        for mode in a.modes:
            if phase_graph is not None:phase_graph.clear()
            if mode != 'bf16':
                if budget_controller is None:from ffn import install
                else:from dynamic import install
                if modules is None:
                    modules = install(pipe.transformer) if budget_controller is None else install(pipe.transformer,budget_controller)
                    report['replaced_ffn_linears']=len(modules)
                    if a.fuse_ffn:
                        from fusion import install as install_fusion
                        report['fused_ffn_layers']=install_fusion(pipe.transformer)
                    assert len(modules)==len(pipe.transformer.layers)*(2 if config['hidden_act']=='relu2' else 3)
                    gc.collect(); torch.cuda.empty_cache()
                ffn_mode = mode if mode.startswith('ffn_') else 'ffn_act2'
                if budget_controller is None:
                    for m in modules:m.configure(ffn_mode)
                else:
                    budget_controller.mode=mode.removeprefix('ffn_') if mode.startswith(('ffn_auto','ffn_uniform')) else mode
            study.mode = mode
            for case in selected:
                if phase_graph is not None:phase_graph.clear()
                if budget_controller is not None:budget_controller.task=case['task']
                if capture is not None:capture.enabled=False
                pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config,
                     flow_shift=10. if case['task']=='policy' else 3., use_karras_sigmas=False)
                branches = 1 if case['task'] == 'policy' else 2
                if not a.sample:
                    if phase_graph is not None:phase_graph.warming=True
                    if budget_controller is not None:budget_controller.warming=True
                    study.observer = None; study.reset(2, branches); sketches.clear()
                    pipe(**call_args(case, 6099, 2))
                    if budget_controller is not None:budget_controller.warming=False
                    if phase_graph is not None:phase_graph.warming=False
                for seed in a.seeds:
                    for repeat in range(a.rounds):
                        study.reset(a.steps, branches); sketches.clear()
                        if phase_graph is not None:phase_graph.reset_stats()
                        if budget_controller is not None:budget_controller.reset_pack()
                        if modules is not None:
                            for m in modules:
                                m.calls=0
                                if budget_controller is not None:m.base_flops=0;m.selected_extra_flops=0;m.tier_calls={}
                        if capture is not None:
                            capture.case=case['id'];capture.task=case['task'];capture.seed=seed;capture.enabled=True
                        study.observer = Observer(len(pipe.transformer.layers), config['num_attention_heads'],
                            a.steps, branches, samples_per_role=a.samples_per_role) if a.sample else None
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                        start = time.perf_counter()
                        result = pipe(**call_args(case, seed, a.steps))
                        if capture is not None:capture.enabled=False
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                        assert study.call == a.steps * branches, (study.call,a.steps,branches)
                        assert len(sketches) == study.call
                        ffn_calls=sum(m.calls for m in modules) if modules is not None else 0
                        if modules is not None:assert ffn_calls==len(modules)*study.call
                        video = np.asarray(result.video)
                        assert np.isfinite(video).all() and video.ndim == 4
                        signals = {'video': video, 'trajectory': np.stack(sketches)}
                        if result.action is not None:
                            signals['action'] = result.action[0].float().cpu().numpy()
                        if result.sound is not None:
                            signals['sound_latent'] = result.sound.float().cpu().numpy()
                        assert all(np.isfinite(v).all() for v in signals.values())
                        key = (case['id'], seed)
                        if mode == 'bf16':
                            references[key] = signals
                        if mode == 'ffn_act2':
                            anchors[key] = signals
                        if mode == 'attn_dual50':
                            repair_references[key] = signals
                        row = dict(mode=mode, case=case['id'], seed=seed, repeat=repeat,
                            wall_seconds=elapsed, transformer_seconds=sum(s.elapsed_time(e) for s,e in study.events)/1000,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(), counts=study.counts,
                            cache_max_age=max(study.ages,default=0), calls=study.call,
                            ffn_calls=ffn_calls,
                            video_shape=list(video.shape), sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in signals.items()})
                        if phase_graph is not None:
                            row['phase_graph']=dict(captures=phase_graph.captures,replays=phase_graph.replays,phase_hits=dict(phase_graph.phase_hits))
                        if mode != 'bf16':
                            row['vs_bf16'] = {k: quality(v,references[key][k]) if k=='video' else signal_quality(v,references[key][k]) for k,v in signals.items()}
                        if budget_controller is not None and modules is not None:
                            row['budget_actual']=dict(base_flops=sum(m.base_flops for m in modules),extra_flops=sum(m.selected_extra_flops for m in modules))
                            from collections import Counter
                            counts=Counter()
                            for m in modules:counts.update(m.tier_calls)
                            row['budget_actual']['tier_calls']=dict(counts)
                            row['budget_actual']['shared_pack_hits']=budget_controller.pack_hits
                            if budget_controller.pack_cache:raise RuntimeError('Unconsumed FFN shared packing entry')
                        if key in anchors and mode != 'ffn_act2':
                            row['vs_ffn_act2'] = {k: quality(v,anchors[key][k]) if k=='video' else signal_quality(v,anchors[key][k]) for k,v in signals.items()}
                        if key in repair_references and mode.startswith(('mask_reuse','delta_reuse')):
                            row['vs_current_repair'] = {k: quality(v,repair_references[key][k]) if k=='video' else signal_quality(v,repair_references[key][k]) for k,v in signals.items()}
                        stem = f'{mode}_{case["id"]}_seed{seed}_r{repeat}'
                        np.savez_compressed(a.output / (stem + '_outputs.npz'), **signals)
                        if repeat == 0:
                            export_to_video(list((np.clip(video,0,1)*255).round().astype(np.uint8)),
                                str(a.output / (stem + '.mp4')), fps=5 if case['task']=='policy' else 24, macro_block_size=1)
                        if study.observer is not None:
                            study.observer.save(a.output / (stem + '_heads.npz'))
                        study.observer = None
                        report['rows'].append(row); save()
                        print(mode, case['id'], seed, repeat, 'wall',round(elapsed,2),
                              'SSIM',row.get('vs_bf16',{}).get('video',{}).get('ssim_mean'), flush=True)
        if capture is not None:capture.save(report['model_revision'])
        report['complete'] = True; save()
    except Exception as error:
        report['error'] = repr(error); save(); raise


if __name__ == '__main__':
    main()
