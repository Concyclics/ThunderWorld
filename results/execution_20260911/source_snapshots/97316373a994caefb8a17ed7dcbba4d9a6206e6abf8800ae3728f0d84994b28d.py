"""Bounded whole-transformer graph replay with current-input copies and owned outputs.

Only used for resident inference. Clear before replacing or moving model weights.
Python shape/CFG/mode metadata is part of the key; tensor values are never cached.
"""
import torch


def signature(value):
    if isinstance(value, torch.Tensor):
        return ('tensor', tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device))
    if isinstance(value, dict):
        return ('dict', tuple((k, signature(v)) for k, v in sorted(value.items())))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(signature(v) for v in value))
    if value is None or isinstance(value, (str, bool, int, float)):
        return (type(value).__name__, value)
    raise TypeError(f'Unsupported graph metadata: {type(value)}')


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_tree(v) for v in value)
    if isinstance(value, list):
        return [clone_tree(v) for v in value]
    return value


def copy_tree(target, source):
    if isinstance(source, torch.Tensor):
        target.copy_(source)
    elif isinstance(source, dict):
        for k in source:
            copy_tree(target[k], source[k])
    elif isinstance(source, (tuple, list)):
        for x, y in zip(target, source):
            copy_tree(x, y)


class GraphForward:
    def __init__(self, forward, controller, max_entries=2):
        self.forward = forward
        self.controller = controller
        self.max_entries = max_entries
        self.cache = {}
        self.captures = 0
        self.replays = 0

    def clear(self):
        torch.cuda.synchronize()
        self.cache.clear()

    @torch.inference_mode()
    def __call__(self, *args, **kwargs):
        if args:
            raise ValueError('Cosmos pipeline graph adapter expects keyword inputs')
        key = (self.controller.mode, torch.cuda.current_stream().cuda_stream, signature(kwargs))
        counts_before = dict(self.controller.calls)
        if key not in self.cache:
            if not self.controller.tuning:
                raise RuntimeError('Unwarmed transformer shape/CFG branch in timed request')
            if len(self.cache) >= self.max_entries:
                self.clear()
            static = clone_tree(kwargs)
            # Warm libraries and all FP4 specializations outside capture.
            warm = torch.cuda.Stream()
            warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):
                for _ in range(3):
                    self.forward(**static)
            torch.cuda.current_stream().wait_stream(warm)
            torch.cuda.synchronize()
            before_capture = dict(self.controller.calls)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self.forward(**static)
            counts = {k: self.controller.calls[k]-before_capture[k] for k in counts_before}
            self.cache[key] = (static, graph, output, counts)
            self.captures += 1
        static, graph, output, counts = self.cache[key]
        copy_tree(static, kwargs)
        graph.replay()
        self.replays += 1
        # Replay executes the captured operators without revisiting Python counters.
        self.controller.calls = {k: counts_before[k]+counts[k] for k in counts_before}
        return clone_tree(output)
