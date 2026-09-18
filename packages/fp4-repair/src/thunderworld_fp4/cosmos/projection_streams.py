"""Fork/join independent projections within one request and CUDA Graph.

Each projection retains its own quantizer, residual selector, and scratch.
This does not make a stateful CosmosRunner safe for concurrent requests.
"""
import torch


class ProjectionStreams:
    def __init__(self, device=None):
        self.streams = tuple(torch.cuda.Stream(device=device) for _ in range(3))

    def __call__(self, x, projections):
        if len(projections) != 3:
            raise ValueError('Expected three independent Q/K/V projections')
        parent = torch.cuda.current_stream(x.device)
        outputs = []
        for stream, projection in zip(self.streams, projections):
            stream.wait_stream(parent)
            with torch.cuda.stream(stream):
                x.record_stream(stream)
                outputs.append(projection(x))
        for stream, output in zip(self.streams, outputs):
            parent.wait_stream(stream)
            output.record_stream(parent)
        return tuple(outputs)
