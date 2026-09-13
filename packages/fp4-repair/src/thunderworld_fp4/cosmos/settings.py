"""Immutable execution settings; independent projection and attention budgets."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from ..config import RepairConfig, RepairPolicy, RepairRule


@dataclass(frozen=True)
class LinearSpec:
    activation: float = 1.
    weight: float = 0.
    cross: bool = False
    activation_blocks: int | None = None
    weight_support: tuple[int, ...] | None = None

    def __post_init__(self):
        for x in (self.activation, self.weight):
            if isinstance(x, bool) or not math.isfinite(x) or not 0 <= x <= 1:
                raise ValueError("Repair fractions must be finite and in [0,1]")
        if self.activation_blocks is not None and (type(self.activation_blocks) is not int or self.activation_blocks < 0):
            raise ValueError('A quota must be a nonnegative integer')
        if self.weight_support is not None:
            ids=tuple(self.weight_support)
            if len(set(ids))!=len(ids) or any(type(i) is not int or i<0 for i in ids):
                raise ValueError('W support must contain unique nonnegative integers')
            object.__setattr__(self,'weight_support',ids)
        has_a = self.activation if self.activation_blocks is None else self.activation_blocks
        has_w = self.weight if self.weight_support is None else bool(self.weight_support)
        if type(self.cross) is not bool or self.cross and not (has_a and has_w):
            raise ValueError("Cross repair requires both residuals")

    def counts(self, n, k):
        a=math.ceil(self.activation*(k//128)) if self.activation_blocks is None else self.activation_blocks
        w=math.ceil(self.weight*(n//128)*(k//128)) if self.weight_support is None else len(self.weight_support)
        if a>k//128 or (self.weight_support and max(self.weight_support)>=n//128*(k//128)):
            raise ValueError('Repair quota/support exceeds shape')
        return a,w

    @property
    def key(self):
        return json.dumps({k:v for k,v in asdict(self).items() if v is not None}, sort_keys=True, separators=(',', ':'))


TIERS = {"base": LinearSpec(0, 0), "a50": LinearSpec(.5, 0),
         "a75": LinearSpec(.75, 0), "act2": LinearSpec(),
         "wgt2": LinearSpec(0, 1), "a50w": LinearSpec(.5, 1),
         "dual": LinearSpec(1, 1), "four": LinearSpec(1, 1, True)}


@dataclass(frozen=True)
class RunConfig:
    resource_mode: str = "exclusive"
    compute_budget: float | None = None
    budget_plan: str | None = None
    precision: str = "fp4"
    debug: str = "off"
    execution: str = "graph"
    ffn: LinearSpec = field(default_factory=LinearSpec)
    projection: LinearSpec = field(default_factory=LinearSpec)
    attention_backend: str = "fp4"
    attention: RepairPolicy = field(default_factory=lambda: RepairPolicy(
        RepairConfig(coverage=.5, terms=3, query_tile=32, transpose_v=True)))
    fuse_ffn: bool = True
    share_pack: bool = False
    kernel_variant: str = 'optimized'
    linear_backend: str = 'triton'
    cutlass_library: str | None = None
    cutlass_small_library: str | None = None
    strict_tuning: bool = False
    # A separate frozen FFN policy uses task|branch|phase|layer.part keys.
    ffn_policy: str | None = None
    ffn_policy_mode: str = "auto50"

    def __post_init__(self):
        if self.budget_plan and self.compute_budget is None:
            raise ValueError('A budget plan requires an explicit compute budget')
        if self.budget_plan and self.ffn_policy:
            raise ValueError('Choose a unified budget plan or a legacy FFN policy')
        if self.resource_mode not in ('exclusive', 'shared-calibration'):
            raise ValueError('Invalid resource mode')
        if self.compute_budget is not None:
            if self.compute_budget not in (1., 1.125, 1.25, 1.5) or isinstance(self.compute_budget, bool):
                raise ValueError('Compute budget must be 1, 1.125, 1.25 or 1.5')
            if self.precision != 'fp4' or self.attention_backend != 'fp4':
                raise ValueError('Compute budgets require joint FP4')
        if self.precision not in ('bf16', 'fp8', 'fp4'):
            raise ValueError('precision must be bf16/fp8/fp4')
        if self.debug not in ('off', 'timing', 'trace') or self.execution not in ('graph', 'eager'):
            raise ValueError('Invalid debug/execution mode')
        if self.attention_backend not in ('bf16', 'fp4'):
            raise ValueError('Invalid attention backend')
        if self.linear_backend not in ('triton','cutlass-fused'):
            raise ValueError('Invalid linear backend')
        if self.linear_backend=='cutlass-fused':
            if self.precision!='fp4' or self.kernel_variant!='optimized' or not self.cutlass_library:
                raise ValueError('CUTLASS requires optimized FP4 and an offline library')
        elif self.cutlass_library is not None or self.cutlass_small_library is not None:
            raise ValueError('A CUTLASS library requires the cutlass-fused backend')
        if self.kernel_variant not in ('optimized','legacy'):
            raise ValueError('Invalid kernel variant')
        if self.precision != 'fp4' and self.attention_backend != 'bf16':
            raise ValueError('BF16/FP8 controls must retain BF16 attention')
        if not isinstance(self.ffn, LinearSpec) or not isinstance(self.projection, LinearSpec):
            raise ValueError('Expected independent LinearSpec objects')
        if not isinstance(self.attention,RepairPolicy):raise ValueError('Expected RepairPolicy')
        if any(type(getattr(self,key)) is not bool for key in ('fuse_ffn','share_pack','strict_tuning')):
            raise ValueError('Execution switches must be boolean')

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for key in ('ffn', 'projection'):
            if key in value: value[key] = LinearSpec(**value[key])
        if 'attention' in value:
            p = dict(value['attention'])
            value['attention'] = RepairPolicy(
                RepairConfig.from_dict(p.get('default', {})),
                tuple(RepairRule(**{**r, 'config': RepairConfig.from_dict(r['config'])}) for r in p.get('rules', [])))
        return cls(**value)


def phase(step, steps):
    return min(3, sum(step >= math.ceil(steps*b) for b in (.2, .5, .8)))
