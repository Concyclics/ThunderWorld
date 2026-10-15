"""Instance-owned Wan VAE graphs with current inputs and owned outputs.

Capture the entire causal chunk loop: its feature caches are reconstructed on
every replay. Enable before warmup. Replacing weights requires a fresh runner.
The graph pool retains intermediates, trading GPU memory for fewer host launches.
"""
import time
import torch
from .trees import signature


class CodecGraphs:
    def __init__(self, vae, max_graphs=8):
        self.vae = vae
        self.cache = {}
        self.warming = True
        self.max_graphs = max_graphs
        self.capture_seconds = 0.
        self.replays = 0
        self.original_encode = vae._encode
        self.original_decode = vae._decode

        def encode(x):
            return self.run('encode', x)

        def decode(x, return_dict=True):
            from diffusers.models.autoencoders.vae import DecoderOutput
            y = self.run('decode', x)
            return DecoderOutput(sample=y) if return_dict else (y,)

        vae._encode = encode
        vae._decode = decode

    def run(self, kind, x):
        if kind not in ('encode', 'decode'):
            raise ValueError(f'Unknown codec operation: {kind}')
        if torch.is_grad_enabled() or self.vae.training:
            raise RuntimeError('Codec graphs require evaluation/inference mode')
        if not x.is_cuda:
            raise ValueError('Codec graphs require CUDA input')
        # Changing these switches changes the captured causal/tiled algorithm.
        options = tuple((k, str(getattr(self.vae, k, None))) for k in
            ('use_tiling', 'use_slicing', 'tile_sample_min_height', 'tile_sample_min_width',
             'tile_sample_stride_height', 'tile_sample_stride_width', 'spatial_compression_ratio'))
        key = (kind, signature(x), options, torch.cuda.current_stream(x.device).cuda_stream)
        if key not in self.cache:
            if not self.warming:
                raise RuntimeError('Unwarmed codec shape/options/stream')
            if len(self.cache) >= self.max_graphs:
                raise RuntimeError('Codec graph capacity reached; use a fresh runner')
            fn = self.original_encode if kind == 'encode' else lambda v: self.original_decode(v, return_dict=False)[0]
            static = x.clone()
            for _ in range(2):
                fn(static)
            torch.cuda.synchronize(x.device)
            graph = torch.cuda.CUDAGraph()
            start = time.perf_counter()
            with torch.cuda.graph(graph):
                out = fn(static)
            torch.cuda.synchronize(x.device)
            self.capture_seconds += time.perf_counter() - start
            self.cache[key] = (static, graph, out)
        static, graph, out = self.cache[key]
        static.copy_(x)
        graph.replay()
        self.replays += 1
        return out.clone()

    def receipt(self):
        return dict(enabled=True, captures=len(self.cache), max_graphs=self.max_graphs,
            capture_seconds=self.capture_seconds, replays=self.replays,
            keys=[str(k) for k in self.cache],
            scope='Whole causal codec loop; current-input copies and owned outputs; graph memory retained')
