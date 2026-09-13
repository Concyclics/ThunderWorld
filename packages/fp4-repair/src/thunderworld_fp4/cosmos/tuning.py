"""Device/kernel-bound launch profiles, explicit misses and fresh retiming."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import random
import statistics
import os
import subprocess
import torch
import triton
import triton.testing


def identity():
    root = Path(__file__).resolve().parents[1]
    paths = list(root.glob('_*.py')) + list((root/'cutlass').glob('*.py')) + [Path(__file__).with_name('linears.py')]
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    hardware = subprocess.check_output(['nvidia-smi', '-i', os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
        '--query-gpu=uuid,driver_version,power.limit', '--format=csv,noheader,nounits'], text=True).strip()
    return dict(gpu=prop.name, sm=prop.multi_processor_count, capability=[prop.major, prop.minor],
                hardware=hardware, cuda=torch.version.cuda,
                torch=torch.__version__, triton=triton.__version__,
                kernels={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})


class LaunchTable:
    def __init__(self, path=None, *, tune=False):
        self.identity = identity()
        self.entries = {}
        self.tune = tune
        self.misses = set()
        self.hits = set()
        if path:
            data = json.loads(Path(path).read_text())
            if data['identity'] != self.identity: raise ValueError('Tuning device/kernel mismatch')
            self.entries = data['entries']

    @staticmethod
    def key(kind, shape, spec):
        return json.dumps([kind, shape, spec], sort_keys=True, separators=(',', ':'))

    def get(self, key, default):
        if key in self.entries:
            self.hits.add(key)
            return self.entries[key]['launch']
        self.misses.add(key)
        return default

    def search(self, key, candidates, apply, fn):
        if key in self.entries or not self.tune: return
        apply(candidates[0])
        reference = fn().clone()
        accepted = []
        for launch in candidates:
            try:
                apply(launch)
                output = fn()
                if not torch.equal(reference, output): continue
                samples = [triton.testing.do_bench_cudagraph(fn, rep=10) for _ in range(3)]
                accepted.append(dict(launch=launch, search_ms=statistics.median(samples)))
            except (triton.OutOfResources, triton.CompilationError):
                continue
        if not accepted: raise RuntimeError('No numerically equivalent launch candidate')
        # Fresh interleaved measurement of the selected candidate and anchor.
        winner = min(accepted, key=lambda v:v['search_ms'])['launch']
        choices = [candidates[0], winner]
        samples = [[], []]
        rng = random.Random(120)
        for _ in range(5):
            order = [0, 1]; rng.shuffle(order)
            for i in order:
                apply(choices[i])
                samples[i].append(triton.testing.do_bench_cudagraph(fn, rep=10))
        index = int(statistics.median(samples[1]) < statistics.median(samples[0])*.97)
        self.entries[key] = dict(launch=choices[index], candidates=accepted,
                                 confirmation_samples_ms=samples, anchor=choices[0])
        self.misses.discard(key)
        apply(choices[index])

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(schema_version=1, identity=self.identity,
            entries=self.entries, missing=sorted(self.misses), hits=sorted(self.hits)), indent=2)+'\n')
