import io
import json
from pathlib import Path
import tarfile

import pytest
import torch

from bundle import relative, check_package, corrections, apply_corrections, _hashes, rebuild
from baselines import sha, write
from arc_orders import load_orders
from portable import tensor_hash


@pytest.mark.parametrize('name', ['../outside', '/tmp/outside', '.'])
def test_package_rejects_path_escape(tmp_path, name):
    with pytest.raises(ValueError):relative(tmp_path, name)


def test_package_detects_modified_asset(tmp_path):
    path = tmp_path/'recipe.pt'
    path.write_bytes(b'fixed')
    write(tmp_path/'package.json', dict(schema='fp4-baselines-handoff-v1', complete=True,
                                      files={'recipe.pt':sha(path)}))
    check_package(tmp_path)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Changed package'):check_package(tmp_path)


@pytest.mark.parametrize('support', [[], [0], [0,1,2,3]])
def test_recipe_reconstructs_fp8_scales_and_exact_bytes(support):
    from thunderworld_fp4.cosmos.calibration import quantize_weight
    from thunderworld_fp4.cosmos.assets import compact
    torch.manual_seed(17)
    weight = torch.randn(256,256).bfloat16()
    prescale = torch.linspace(.5,2.,256)
    codes,scales,factor = quantize_weight(weight,prescale,base=.875,residual=1.125)
    source = compact(dict(codes=codes,scales=scales,factor=factor,prescale=prescale,
                          score=torch.arange(4).reshape(2,2),weight_scale=(.875,1.125)),support)
    metadata = {k:v for k,v in source.items() if k not in
                ('codes_base','scales_base','codes_residual','scales_residual')}
    restored = rebuild(weight, metadata)
    # Exercise real byte correction, including encoded FP8 scale bytes.
    source['codes_base'][0,0] ^= 1
    source['scales_base'].view(torch.uint8)[0,0] ^= 1
    patch = corrections(restored, source)
    assert _hashes(apply_corrections(restored,patch)) == _hashes(source)
    assert _hashes(restored) != _hashes(source)


def test_recipe_refuses_unrepresented_outlier_removal():
    with pytest.raises(ValueError, match='Outlier'):
        rebuild(torch.zeros(128,128), {'mandatory_w_blocks':[0]})


def test_arc_order_loader_checks_identity_hash_and_permutation(tmp_path):
    orders = {'layer':torch.tensor([2,0,3,1])}
    torch.save(orders,tmp_path/'orders.pt')
    manifest = dict(schema='arc-channel-order-v1',complete=True,model_identity={'model':'fixed'},
        panels_per_module=16,file='orders.pt',sha256=sha(tmp_path/'orders.pt'),
        tensor_hashes={k:tensor_hash(v) for k,v in orders.items()})
    write(tmp_path/'manifest.json',manifest)
    assert torch.equal(load_orders(tmp_path,{'model':'fixed'},['layer'])['layer'],orders['layer'])
    with pytest.raises(ValueError):load_orders(tmp_path,{'model':'other'},['layer'])
    with pytest.raises(ValueError):load_orders(tmp_path,{'model':'fixed'},['missing'])
    orders['layer'] = torch.tensor([2,0,3,3])
    torch.save(orders,tmp_path/'orders.pt')
    manifest['sha256'] = sha(tmp_path/'orders.pt')
    manifest['tensor_hashes'] = {k:tensor_hash(v) for k,v in orders.items()}
    write(tmp_path/'manifest.json',manifest)
    with pytest.raises(ValueError, match='permutation'):load_orders(tmp_path,{'model':'fixed'},['layer'])


@pytest.mark.parametrize('member', ['../escape', 'symlink'])
def test_release_extraction_rejects_unsafe_members(tmp_path, member):
    from download import extract
    path = tmp_path/'bad.tar.gz'
    with tarfile.open(path,'w:gz') as output:
        info = tarfile.TarInfo(member)
        if member == 'symlink':
            info.type = tarfile.SYMTYPE;info.linkname = '/tmp'
        output.addfile(info,io.BytesIO(b''))
    with pytest.raises(ValueError):extract(path,tmp_path/'target')


def test_native_repair_validation_does_not_require_external_builds():
    from baseline_runtime import verify
    verify({'methods':{'ours':{'config':{}}}}, 'ours')


def test_throughput_reserve_is_explicit_and_frozen(tmp_path):
    from baseline_throughput_suite import prepare
    study = tmp_path/'study';study.mkdir()
    write(study/'manifest.json',dict(cases=[{'benchmark':n,'case':{'id':n}} for n in ('rbench','paibench')],
        batch_protocol={'methods':['bf16'],'sizes':[1,2,4,8]}))
    prepare(study,tmp_path/'speed',2048)
    assert json.loads((tmp_path/'speed/suite.json').read_text())['reserve_mib']==2048
    with pytest.raises(ValueError):prepare(study,tmp_path/'invalid',0)
