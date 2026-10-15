"""Native-only Thor adapters. Never replace an unsupported kernel with BF16."""
import importlib
import json
import os
from pathlib import Path
import sys

from build_baselines import sha


def build_record(root, method, folder=None):
    folder = Path(folder) if folder else root/'external'/(method+'-sm110')
    record = json.loads((folder/'build.json').read_text())
    if not record['complete'] or record['architecture'] != 'sm_110a':
        raise ValueError('Unqualified build target: '+method)
    if sha(record['library']) != record['library_sha256']:
        raise ValueError('Baseline binary changed: '+method)
    for name, digest in record['patches'].items():
        if sha(folder/name) != digest:
            raise ValueError('Baseline patch changed')
    return folder, record


def sharq_class(root):
    folder, record = build_record(root, 'sharq')
    originals = json.loads((root/'data/baseline_sources.json').read_text())['sharq']['files']
    for name, digest in originals.items():
        if name.startswith('model/') or name == 'sharq_loader.py':
            if sha(folder/name) != digest:
                raise ValueError('SharQ Python quantization changed')
    os.environ['SHARQ_KERNEL_BUILD'] = str(folder/'kernels/build')
    if 'quantize' in sys.modules and Path(sys.modules['quantize'].__file__).resolve() != folder/'model/quantize.py':
        raise ValueError('Conflicting baseline modules: use a separate process')
    sys.path[:0] = [str(folder/'model'), str(folder)]
    return importlib.import_module('qLinearLayer').QLinearLayer, record


def install(runner, root, method, asset_paths=None, arc_build=None):
    method = {'arcquant':'arcquant50', 'sharq_perf':'sharq'}.get(method,method)
    asset_paths = asset_paths or {}
    if method not in ('arcquant50','arcquant100','sharq','awq_marlin'):return None
    if runner.engine.cache or runner._warmed:raise ValueError('Install baseline before warmup/capture')
    targets = list(runner.engine.targets)
    dimensions = {name:[runner.pipe.transformer.get_submodule(name).out_features,
                       runner.pipe.transformer.get_submodule(name).in_features] for name in targets}
    if method.startswith('arcquant'):
        from external import install_arc
        folder, record = build_record(root, 'arcquant', arc_build)
        receipt = install_arc(runner, folder, Path(asset_paths.get('arc_orders',root/'data/arc_orders')),
                              .5 if method == 'arcquant50' else 1.)
    elif method == 'sharq':
        from external import SharQLinear
        official, record = sharq_class(root)
        for name in runner.engine.targets:
            source = runner.pipe.transformer.get_submodule(name)
            op = SharQLinear(source, official)
            op.official.extra_fusion = True
            runner.pipe.transformer.set_submodule(name, op)
            runner.engine.linears[name] = op
        receipt = dict(method='SharQ', execution_type='official_native_sm110_cosmos_adapter',
                    build=record, extra_fusion=True, cosmos_rmsnorm_fusion=False,
                    performance_eligible=False)
    else:
        from awq_marlin import install as install_awq
        receipt = install_awq(runner, Path(asset_paths.get('awq_marlin',root/'assets/awq_marlin')))
    if set(runner.engine.linears)!=set(targets):raise ValueError('Incomplete external projection coverage')
    receipt.update(targets=targets,dimensions=dimensions,
        gemm_dtype='W4A16_BF16_Marlin' if method=='awq_marlin' else 'FP4_tensor_core',
        attention_backend='bf16',execution='graph' if method=='awq_marlin' else 'eager',
        adapter_source_sha256=sha(__file__),asset_paths={k:str(v) for k,v in asset_paths.items()},
        no_fallback=True)
    for name in targets:
        op=runner.engine.linears[name]
        if type(op).__name__ not in ('ARCLinear','SharQLinear','PackedAWQLinear'):
            raise ValueError('Unexpected high precision fallback: '+name)
        if [op.out_features,op.in_features]!=dimensions[name]:raise ValueError('Wrong dimensions')
    return receipt
