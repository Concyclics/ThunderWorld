"""Download only the fixed Generator components; reuse existing receiver files."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from huggingface_hub import HfApi

MODELS = {
    'edge': ('nvidia/Cosmos3-Edge', 'a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba', 'Cosmos3-Edge-a9d944e'),
    'nano': ('nvidia/Cosmos3-Nano', '411f42a8fdfb8c5b2583cb8786e0938f49796eaa', 'Cosmos3-Nano-411f42a'),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--models-dir', type=Path, default=Path('models'))
    p.add_argument('--with-sound', action='store_true', help='Include Nano sound_tokenizer for AV generation/decoding')
    a = p.parse_args()
    if a.with_sound and a.model != 'nano':p.error('--with-sound requires --model nano')
    repo, revision, folder = MODELS[a.model]
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    assert info.sha == revision
    components = ('transformer/', 'vae/', 'text_tokenizer/', 'scheduler/')
    if a.with_sound:components += ('sound_tokenizer/',)
    files = [f for f in info.siblings if f.rfilename == 'model_index.json'
             or f.rfilename.startswith(components)]
    target = a.models_dir/folder
    target.mkdir(parents=True, exist_ok=True)
    receipt = {'repo': repo, 'revision': revision, 'complete': False,
               'files': {f.rfilename: {'bytes': f.size} for f in files}}
    (target/'generator_download.json').write_text(json.dumps(receipt, indent=2)+'\n')
    hf = Path(sys.executable).with_name('hf')
    subprocess.run([str(hf), 'download', repo, *[f.rfilename for f in files], '--revision', revision,
                    '--local-dir', str(target), '--max-workers', '3'], check=True)
    for f in files:
        assert (target/f.rfilename).stat().st_size == f.size, f.rfilename
    receipt['complete'] = True
    (target/'generator_download.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
