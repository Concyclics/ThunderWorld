"""Optional Cosmos integration. Configuration imports do not initialize CUDA."""
from .settings import RunConfig, LinearSpec

__all__ = ["RunConfig", "LinearSpec", "CosmosRunner"]


def __getattr__(name):
    if name == "CosmosRunner":
        from .runner import CosmosRunner
        return CosmosRunner
    raise AttributeError(name)
