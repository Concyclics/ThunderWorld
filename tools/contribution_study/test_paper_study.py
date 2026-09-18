import copy
import pytest

from paper_study import cross_variant
from paper_mot import role_tiles
from paper_outlier import effective_spec
from thunderworld_fp4 import RepairConfig, RepairPolicy
from thunderworld_fp4.cosmos.settings import RunConfig, LinearSpec


def anchor():
    return dict(config=RunConfig(ffn=LinearSpec(1, 1, True), projection=LinearSpec(1, 1, True),
        attention=RepairPolicy(RepairConfig(coverage=1, terms=4))).to_dict(), origin_receipt_sha256='parent')


@pytest.mark.parametrize('family', ['projection', 'ffn', 'qk', 'pv'])
def test_one_cross_family_changes_and_parent_is_immutable(family):
    original = anchor()
    saved = copy.deepcopy(original)
    child = cross_variant(original, family)
    cfg = RunConfig.from_dict(child['config'])
    assert original == saved
    assert 'origin_receipt_sha256' not in child
    assert cfg.projection.cross == (family != 'projection')
    assert cfg.ffn.cross == (family != 'ffn')
    assert cfg.attention.default.terms == (3 if family == 'qk' else 4)
    assert (cfg.attention.default.pv_terms or cfg.attention.default.terms) == (3 if family == 'pv' else 4)
    for key in ('activation', 'weight', 'activation_blocks', 'weight_support'):
        for part in ('ffn', 'projection'):
            assert child['config'][part][key] == original['config'][part][key]
    assert cfg.attention.default.head_counts(4, 417) == (4,)*4
    assert cfg.attention.default.pv_counts(4, 417) == (4,)*4


def test_reject_nonfour_and_budgeted_assets():
    bad = anchor()
    bad['config']['ffn']['cross'] = False
    with pytest.raises(ValueError):
        cross_variant(bad, 'qk')
    bad = anchor()
    bad['assets'] = 'calibrated'
    with pytest.raises(ValueError):
        cross_variant(bad, 'ffn')
    with pytest.raises(ValueError):
        cross_variant(anchor(), 'unknown')


@pytest.mark.parametrize('ar,expected,mixed', [(31, [], [0]), (128, [0], []), (300, [0, 1], [2])])
def test_role_tiles_preserve_mixed_boundary(ar, expected, mixed):
    roles = role_tiles(960, 960+ar)
    assert roles['pure_ar'] == expected
    assert roles['mixed'] == mixed
    assert not set(roles['pure_dm']) & set(expected+mixed)
    assert set(roles['pure_dm']+expected+mixed) == set(range((960+ar+127)//128))


def test_cross_is_empty_if_either_repair_side_is_empty():
    assert not effective_spec(0, [0], True).cross
    assert not effective_spec(4, [], True).cross
    assert effective_spec(4, [0], True).cross
    assert not effective_spec(4, [0], False).cross
