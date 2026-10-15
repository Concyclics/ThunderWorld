"""Pinned Hub release discovery and inference; importing this module needs no CUDA."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def registry():
    return json.loads((Path(__file__).parent / 'presets/releases.json').read_text())


def resolve_release(model, profile='2x', directory=None, local_files_only=False):
    entry = registry()['models'][model]
    if directory is not None:
        root = Path(directory).resolve()
    else:
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(entry['repo_id'], revision=entry['revision'],
            allow_patterns=['*.json', '*.md', '*.txt', 'LICENSE', 'NOTICE', '.gitattributes',
                            'infer.py', 'runtime/**', f'profiles/{profile}/**'],
            ignore_patterns=[f'profiles/{other}/**' for other in ('1.5x', '2x') if other != profile],
            local_files_only=local_files_only))
    spec = json.loads((root / 'release.json').read_text())
    if spec['repo_id'] != entry['repo_id'] or spec['base_model']['revision'] != entry['base_required']['revision']:
        raise ValueError('Release/model identity mismatch')
    if profile not in spec['profiles'] or not (root / 'infer.py').is_file():
        raise ValueError('Incomplete release or unknown profile')
    return root


def inference_command(root, profile, *, base_model=None, image=None, prompt=None,
                      case_json=None, seed=12345, output=None, verify_only=False,
                      debug='off', execution='graph'):
    cmd = [sys.executable, str(Path(root) / 'infer.py'), '--profile', profile]
    if base_model is not None:
        cmd += ['--base-model', str(Path(base_model).resolve())]
    if verify_only:
        return cmd + ['--verify-only']
    if case_json:
        if image or prompt:
            raise ValueError('Use --case-json or --image/--prompt, not both')
        cmd += ['--case-json', str(Path(case_json).resolve())]
    elif image and prompt:
        cmd += ['--image', str(Path(image).resolve()), '--prompt', prompt]
    else:
        raise ValueError('Inference needs --image and --prompt, or --case-json')
    if output is None:
        raise ValueError('An output directory is required')
    return cmd + ['--seed', str(seed), '--output', str(Path(output).resolve()),
                  '--debug', debug, '--execution', execution]


def doctor(gpu=False):
    versions = {}
    for name in ('torch', 'triton', 'diffusers', 'transformers', 'huggingface-hub'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = dict(python=sys.version.split()[0], packages=versions,
                  qualified_environment='Python 3.12, PyTorch 2.11/cu130, Triton 3.6; requirements-runtime.txt',
                  native_backend='SM120', performance_eligible=False)
    if gpu:
        import torch
        report['gpus'] = [dict(name=torch.cuda.get_device_name(i),
            capability=list(torch.cuda.get_device_capability(i)),
            memory_bytes=torch.cuda.get_device_properties(i).total_memory)
            for i in range(torch.cuda.device_count())]
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=('edge', 'nano', 'super'), default='edge')
    p.add_argument('--profile', choices=('1.5x', '2x'), default='2x')
    p.add_argument('--release-dir', type=Path, help='Reuse a local complete/selected-profile release')
    p.add_argument('--base-model', type=Path)
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--download-only', action='store_true')
    p.add_argument('--verify-only', action='store_true', help='Verify downloaded asset hashes on CPU')
    p.add_argument('--doctor', action='store_true')
    p.add_argument('--check-gpu', action='store_true')
    p.add_argument('--list-models', action='store_true')
    p.add_argument('--image', type=Path)
    p.add_argument('--prompt')
    p.add_argument('--case-json', type=Path)
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--output', type=Path)
    p.add_argument('--debug', choices=('off', 'timing', 'trace'), default='off')
    p.add_argument('--execution', choices=('graph', 'eager'), default='graph')
    args = p.parse_args(argv)
    if args.doctor:
        print(json.dumps(doctor(args.check_gpu), indent=2)); return
    if args.list_models:
        print(json.dumps(registry(), indent=2)); return
    if not (args.verify_only or args.download_only):
        inference_command('.', args.profile, base_model=args.base_model, image=args.image,
            prompt=args.prompt, case_json=args.case_json, seed=args.seed, output=args.output)
    root = resolve_release(args.model, args.profile, args.release_dir, args.local_files_only)
    if args.download_only:
        print(root); return
    cmd = inference_command(root, args.profile, base_model=args.base_model, image=args.image,
        prompt=args.prompt, case_json=args.case_json, seed=args.seed, output=args.output,
        verify_only=args.verify_only, debug=args.debug, execution=args.execution)
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    if args.local_files_only:
        env['HF_HUB_OFFLINE'] = '1'
    subprocess.run(cmd, env=env, check=True)


if __name__ == '__main__':
    main()
