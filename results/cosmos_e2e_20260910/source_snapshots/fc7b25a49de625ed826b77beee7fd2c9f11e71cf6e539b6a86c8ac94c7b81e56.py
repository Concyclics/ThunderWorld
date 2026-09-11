"""Paired full encode -> denoise -> decode benchmark on one local SM120 GPU."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import torch
import diffusers.utils.import_utils as import_utils

# The inherited optional xformers wheel targets another Torch version. Use SDPA.
import_utils._xformers_available = False
from diffusers import Cosmos3OmniPipeline, Cosmos3OmniTransformer, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from PIL import Image
from skimage.metrics import structural_similarity
from adapter import Controller, PreparedLinear, install_ffn, install_attention_dispatch, ROOT
from graph import GraphForward


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quality(video, reference):
    a, b = video.astype(np.float64), reference.astype(np.float64)
    mse = float(np.mean((a-b)**2))
    return {'mse': mse, 'psnr_db': float(-10*np.log10(mse)) if mse else None,
            'identical': bool(np.array_equal(video, reference)),
            'ssim_mean': float(np.mean([structural_similarity(x, y, data_range=1., channel_axis=-1)
                                       for x, y in zip(video, reference)])),
            'relative_l2': float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-12))}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=35)
    p.add_argument('--frames', type=int, default=61)
    p.add_argument('--height', type=int, default=192)
    p.add_argument('--width', type=int, default=320)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--seed', type=int, default=5090)
    p.add_argument('--quality-seed', type=int, default=5091)
    p.add_argument('--offload', choices=['resident', 'model'], default='resident')
    p.add_argument('--allow-split-k', action='store_true', help='Experimental: rounding changes can amplify across quantized layers')
    p.add_argument('--cuda-graph', action='store_true', help='Same whole-transformer replay adapter for every mode; resident only')
    p.add_argument('--modes', nargs='+', default=['bf16', 'ffn_original', 'ffn_optimized', 'joint_original', 'joint_optimized'])
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Use a fresh result directory')
    if a.modes[0] != 'bf16' or not set(a.modes) <= {'bf16', 'ffn_original', 'ffn_optimized', 'joint_original', 'joint_optimized'}:
        raise ValueError('Begin with BF16, followed by supported FP4 variants')
    if a.cuda_graph and a.offload != 'resident':
        raise ValueError('CUDA Graph requires resident weights')
    if min(a.rounds, a.steps, a.frames) < 1 or torch.cuda.get_device_capability() != (12, 0):
        raise ValueError('Positive counts and SM120 GPU required')
    a.output.mkdir(parents=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    import diffusers, transformers, triton
    revision = subprocess.check_output(['git', '-C', str(ROOT/'data_local/e2e_src/diffusers'), 'rev-parse', 'HEAD'], text=True).strip()
    source_paths = [Path(__file__), Path(__file__).with_name('adapter.py'), Path(__file__).with_name('graph.py'), ROOT/'experiments/ffn_block_repair_20260907/native.py',
                    *sorted((ROOT/'packages/fp4-repair/src/thunderworld_fp4').glob('*.py'))]
    report = {'complete': False, 'scope': 'Full image encode, tokenization, denoising and VAE decode; excludes loading, JIT, offline weight packing, safety checker and video file encoding',
              'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              'environment': {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'triton': triton.__version__,
                              'diffusers': diffusers.__version__, 'diffusers_commit': revision, 'transformers': transformers.__version__,
                              'optional_xformers': 'disabled; incompatible inherited wheel'},
              'source_sha256': {str(f.relative_to(ROOT)): digest(f) for f in source_paths},
              'image_sha256': digest(a.image), 'offload_policy': a.offload,
              'quantization': 'Generator FFNs only: current model RTN W4, identity prescale, full ACT2, no W1 repair; no Nano calibration transfer',
              'attention': 'Joint variants use generation-only 75% dual-first-order repair, uniform heads; causal understanding attention stays SDPA',
              'model_config': json.loads((a.model/'transformer/config.json').read_text()), 'rows': []}
    def save():
        (a.output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    save()
    print('Loading', a.model, flush=True)
    transformer, loading_info = Cosmos3OmniTransformer.from_pretrained(
        str(a.model/'transformer'), torch_dtype=torch.bfloat16, local_files_only=True, output_loading_info=True)
    report['transformer_loading_info'] = loading_info
    save()
    if any(loading_info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError(f'Checkpoint/source mismatch: {loading_info}')
    pipe = Cosmos3OmniPipeline.from_pretrained(str(a.model), torch_dtype=torch.bfloat16,
                                             transformer=transformer,
                                             sound_tokenizer=None, safety_checker=None, enable_safety_checker=False,
                                             local_files_only=True)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3., use_karras_sigmas=False)
    pipe.transformer.set_attention_backend('_native_flash')
    pipe.vae.enable_tiling()
    def place_pipeline():
        if a.offload == 'model':
            pipe.enable_model_cpu_offload(gpu_id=0)
        else:
            pipe.to('cuda')
    place_pipeline()
    controller = Controller(allow_split_k=a.allow_split_k)
    report['baseline_source_sha256'] = controller.baseline_hashes
    install_attention_dispatch(controller)
    graph_forward = None
    if a.cuda_graph:
        graph_forward = GraphForward(pipe.transformer.forward, controller)
        pipe.transformer.forward = graph_forward
    prompt = ('A stationary camera observes a robot arm in a workspace. The robot arm moves slowly and smoothly '
              'toward an object on the table. The background, lighting, and camera position remain unchanged. '
              'Objects keep their shape and position except for the deliberate motion of the arm.')
    negative = 'distorted objects, flickering, abrupt camera motion, blurry, low quality'
    image = Image.open(a.image).convert('RGB')
    report['prompt'], report['negative_prompt'] = prompt, negative
    events = []
    def pre(module, args):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        events.append([event, None])
    def post(module, args, out):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        events[-1][1] = event
    pipe.transformer.register_forward_pre_hook(pre)
    pipe.transformer.register_forward_hook(post)

    def run(seed, steps):
        events.clear()
        controller.calls = {'ffn': 0, 'attention': 0}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = pipe(prompt=prompt, negative_prompt=negative, image=image,
                      height=a.height, width=a.width, num_frames=a.frames, fps=24.,
                      num_inference_steps=steps, guidance_scale=6., enable_sound=False,
                      generator=torch.Generator(device='cuda').manual_seed(seed), output_type='np',
                      add_resolution_template=False, add_duration_template=False, enable_safety_check=False)
        torch.cuda.synchronize()
        wall = time.perf_counter()-start
        video = np.asarray(result.video)
        if video.ndim != 4 or video.shape[-1] != 3 or not np.isfinite(video).all():
            raise AssertionError(('Unexpected decoded video', video.shape))
        times = [s.elapsed_time(e)/1000 for s, e in events]
        assert len(times) == steps*2, (len(times), steps)
        expected_ffn = len(report.get('replaced_modules', []))*len(times) if controller.mode != 'bf16' else 0
        expected_attention = len(pipe.transformer.layers)*len(times) if controller.mode.startswith('joint') else 0
        assert controller.calls == {'ffn': expected_ffn, 'attention': expected_attention}, controller.calls
        row = {'seed': seed, 'steps': steps, 'wall_seconds': wall, 'transformer_call_seconds': times,
               'transformer_seconds': sum(times), 'transformer_seconds_excluding_first_call': sum(times[1:]),
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'operator_calls': dict(controller.calls),
               'video_shape': list(video.shape), 'video_sha256': hashlib.sha256(video.tobytes()).hexdigest()}
        return video, row

    installed = False
    references = {}
    original_videos = {}
    try:
        for mode in a.modes:
            if graph_forward is not None:
                graph_forward.clear()
            if mode != 'bf16' and not installed:
                pipe.remove_all_hooks()
                pipe.to('cpu')
                gc.collect()
                torch.cuda.empty_cache()
                report['replaced_modules'] = install_ffn(pipe.transformer, controller)
                assert len(report['replaced_modules']) == report['model_config']['num_hidden_layers']*(2 if report['model_config']['hidden_act']=='relu2' else 3)
                place_pipeline()
                installed = True
            controller.mode = mode
            controller.tuning = True
            print('Warmup', mode, flush=True)
            run(a.seed, min(2, a.steps))
            controller.tuning = False
            for i, seed in enumerate([a.seed]*a.rounds + [a.quality_seed]):
                video, row = run(seed, a.steps)
                row.update(mode=mode, round=i, timing_sample=i<a.rounds)
                if mode == 'bf16':
                    references.setdefault(seed, video.copy())
                else:
                    row['vs_bf16'] = quality(video, references[seed])
                if mode.endswith('original'):
                    original_videos[(mode.split('_')[0], seed)] = video.copy()
                if mode.endswith('optimized') and (mode.split('_')[0], seed) in original_videos:
                    row['vs_original_fp4'] = quality(video, original_videos[(mode.split('_')[0], seed)])
                if i in (0, a.rounds):
                    path = a.output/f'{mode}_seed{seed}'
                    np.save(path.with_suffix('.npy'), video)
                    export_to_video(list((np.clip(video, 0, 1)*255).round().astype(np.uint8)), str(path.with_suffix('.mp4')), fps=24, macro_block_size=1)
                    row['video_file'] = str(path.with_suffix('.mp4'))
                report['rows'].append(row)
                report['launches'], report['searches'] = controller.launches, controller.searches
                report['attention_shapes'] = sorted(controller.attention_shapes)
                if graph_forward is not None:
                    report['graph_counts'] = {'captures': graph_forward.captures, 'replays': graph_forward.replays}
                save()
                print(mode, i, 'wall', round(row['wall_seconds'], 3), 'transformer', round(row['transformer_seconds'], 3),
                      'ssim', row.get('vs_bf16', {}).get('ssim_mean'), flush=True)
        report['summary'] = {}
        for mode in a.modes:
            rows = [r for r in report['rows'] if r['mode']==mode and r['timing_sample']]
            values = [r['wall_seconds'] for r in rows]
            report['summary'][mode] = {'wall_median_seconds': statistics.median(values), 'wall_min_seconds': min(values),
                                       'wall_max_seconds': max(values), 'timing_samples': len(values)}
        report['complete'] = True
        save()
    except Exception as error:
        report['error'] = repr(error)
        save()
        raise


if __name__ == '__main__':
    main()
