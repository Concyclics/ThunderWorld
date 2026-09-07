"""CPU-only configuration, explicit layer/step/branch scope, and presets."""
from dataclasses import asdict, dataclass, field
from importlib.resources import files
import json
import math
from pathlib import Path


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _fraction(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")


@dataclass(frozen=True)
class RepairConfig:
    """Coverage selects residual work, never deletes attention connections.

    terms=2: A0B0 + A1B0; terms=3: also A0B1; terms=4: also A1B1.
    head_budgets overrides coverage for named query heads; unspecified heads
    use coverage. Terms apply to both QK and PV. Blocks remain 128x128 tokens.
    """

    coverage: float = 0.5
    terms: int = 3
    head_budgets: tuple[tuple[int, float], ...] = ()
    block_size: int = 128
    query_tile: int | None = None
    num_warps: int = 4
    splits: int = 2
    lazy_p: bool = True
    fast_p: bool = True

    def __post_init__(self):
        _fraction(self.coverage, "coverage")
        for name, allowed in (("terms", (2, 3, 4)), ("block_size", (128,)),
                              ("num_warps", (4, 8)), ("splits", (1, 2, 4, 8))):
            value = getattr(self, name)
            if type(value) is not int or value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        if self.query_tile is not None and (
            type(self.query_tile) is not int or self.query_tile not in (16, 32, 64)
        ):
            raise ValueError("query_tile must be None, 16, 32, or 64; BM128 failed validation")
        if type(self.lazy_p) is not bool or type(self.fast_p) is not bool:
            raise ValueError("lazy_p and fast_p must be booleans")
        budgets = tuple(tuple(pair) for pair in self.head_budgets)
        seen = set()
        for pair in budgets:
            if len(pair) != 2:
                raise ValueError("head_budgets entries must be (head, coverage)")
            head, fraction = pair
            _integer(head, "head")
            _fraction(fraction, "head coverage")
            if head in seen:
                raise ValueError(f"duplicate head budget: {head}")
            seen.add(head)
        object.__setattr__(self, "head_budgets", tuple(sorted(budgets)))

    def head_counts(self, query_heads: int, key_tokens: int) -> tuple[int, ...]:
        _integer(query_heads, "query_heads", 1)
        _integer(key_tokens, "key_tokens", 1)
        tiles = math.ceil(key_tokens / self.block_size)
        counts = [math.ceil(tiles * self.coverage)] * query_heads
        for head, fraction in self.head_budgets:
            if head >= query_heads:
                raise ValueError(f"head {head} exceeds query head count {query_heads}")
            counts[head] = math.ceil(tiles * fraction)
        return tuple(counts)

    def cost(self, query_heads: int, key_tokens: int) -> dict:
        """Padded-tile product count; excludes all packing/selection/softmax costs."""
        counts = self.head_counts(query_heads, key_tokens)
        fraction = sum(counts) / (query_heads * math.ceil(key_tokens / self.block_size))
        return {"selected_tile_fraction": fraction,
                "matrix_product_multiplier": 1 + (self.terms - 1) * fraction,
                "scope": "equal-cost padded tiles; external repair_mask replaces these quotas"}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict):
        if not isinstance(value, dict):
            raise ValueError("repair configuration must be an object")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown repair settings: {sorted(unknown)}")
        return cls(**value)

    @classmethod
    def from_json(cls, path):
        value = json.loads(Path(path).read_text())
        if "schema_version" in value:
            if value["schema_version"] != 1:
                raise ValueError("unsupported preset schema_version")
            value = value["config"]
        return cls.from_dict(value)


PRESETS = ("balanced", "quality", "full_dual", "full_four", "full_act2", "fp4_base")


def preset(name: str = "balanced") -> RepairConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {PRESETS}")
    value = json.loads(files("thunderworld_fp4").joinpath("presets", name + ".json").read_text())
    return RepairConfig.from_dict(value["config"])


@dataclass(frozen=True)
class RepairRule:
    """First matching rule wins; ranges are zero-based, half-open [start, end)."""

    config: RepairConfig
    layers: tuple[int, ...] | None = None
    steps: tuple[int, int] | None = None
    branches: tuple[str, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.config, RepairConfig):
            raise ValueError("rule config must be RepairConfig")
        if self.layers is not None:
            object.__setattr__(self, "layers", tuple(self.layers))
            if not self.layers:
                raise ValueError("layers must be nonempty")
            for layer in self.layers:
                _integer(layer, "layer")
        if self.steps is not None:
            object.__setattr__(self, "steps", tuple(self.steps))
            if len(self.steps) != 2:
                raise ValueError("steps must be [start, end)")
            start, end = self.steps
            _integer(start, "start step")
            _integer(end, "end step", 1)
            if start >= end:
                raise ValueError("step range must be nonempty")
        if self.branches is not None:
            object.__setattr__(self, "branches", tuple(self.branches))
            if not self.branches or any(not isinstance(b, str) or not b for b in self.branches):
                raise ValueError("branches must contain nonempty strings")


@dataclass(frozen=True)
class RepairPolicy:
    default: RepairConfig = field(default_factory=RepairConfig)
    rules: tuple[RepairRule, ...] = ()

    def __post_init__(self):
        if not isinstance(self.default, RepairConfig):
            raise ValueError("policy default must be RepairConfig")
        object.__setattr__(self, "rules", tuple(self.rules))
        if any(not isinstance(rule, RepairRule) for rule in self.rules):
            raise ValueError("policy rules must be RepairRule values")

    def resolve(self, *, layer=None, step=None, branch=None) -> RepairConfig:
        if layer is not None:
            _integer(layer, "layer")
        if step is not None:
            _integer(step, "step")
        for attribute, value in (("layers", layer), ("steps", step), ("branches", branch)):
            if value is None and any(getattr(rule, attribute) is not None for rule in self.rules):
                raise ValueError(f"policy requires explicit {attribute} context")
        for rule in self.rules:
            if rule.layers is not None and layer not in rule.layers:
                continue
            if rule.steps is not None and not rule.steps[0] <= step < rule.steps[1]:
                continue
            if rule.branches is not None and branch not in rule.branches:
                continue
            return rule.config
        return self.default


@dataclass(frozen=True)
class LinearRepairConfig:
    """Independent FFN/projection ACT2 budget on M128/N128/K128 tiles.

    Output blocks are channel groups (Q/K/V output groups may correspond to
    heads). Coverage selects important activation K blocks per M block.
    An explicit [ceil(M/128), N/128, K/128] mask overrides this selection.
    """

    coverage: float = 1.0
    output_budgets: tuple[tuple[int, float], ...] = ()
    query_tile: int = 32
    num_warps: int = 4

    def __post_init__(self):
        _fraction(self.coverage, "FFN coverage")
        if type(self.query_tile) is not int or self.query_tile not in (16, 32):
            raise ValueError("linear query_tile must be 16 or 32")
        if type(self.num_warps) is not int or self.num_warps not in (4, 8):
            raise ValueError("linear num_warps must be 4 or 8")
        # Reuse strict integer/fraction/duplicate checks, with channel semantics.
        validated = RepairConfig(coverage=self.coverage, head_budgets=self.output_budgets)
        object.__setattr__(self, "output_budgets", validated.head_budgets)

    def output_counts(self, out_features, in_features):
        _integer(out_features, "out_features", 1)
        _integer(in_features, "in_features", 1)
        if out_features <= 0 or in_features <= 0 or out_features % 128 or in_features % 128:
            raise ValueError("linear N and K must be positive multiples of 128")
        return RepairConfig(coverage=self.coverage, head_budgets=self.output_budgets).head_counts(
            out_features // 128, in_features)


@dataclass(frozen=True)
class InferenceConfig:
    """FFN and attention configuration are separate, including projections."""

    ffn: LinearRepairConfig = field(default_factory=LinearRepairConfig)
    attention: RepairConfig = field(default_factory=RepairConfig)
    attention_projection: LinearRepairConfig = field(default_factory=LinearRepairConfig)

    def __post_init__(self):
        if not isinstance(self.ffn, LinearRepairConfig) or not isinstance(self.attention_projection, LinearRepairConfig):
            raise ValueError("ffn/attention_projection must be LinearRepairConfig")
        if not isinstance(self.attention, RepairConfig):
            raise ValueError("attention must be RepairConfig")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {"ffn", "attention", "attention_projection"}:
            raise ValueError("expected separate ffn, attention, attention_projection sections")
        return cls(ffn=LinearRepairConfig(**value.get("ffn", {})),
                   attention=RepairConfig.from_dict(value.get("attention", {})),
                   attention_projection=LinearRepairConfig(**value.get("attention_projection", {})))

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self):
        return asdict(self)
