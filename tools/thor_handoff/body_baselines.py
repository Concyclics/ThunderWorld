"""Attach the existing native baseline adapters to the body-only experiment."""
import json
from pathlib import Path
import sys


def install(runner, manifest, method):
    if method not in manifest.get('adapters', {}):
        return None
    if manifest.get('baseline_platform') == 'Thor':
        from baseline_adapter import install as thor_install
        receipt = thor_install(runner, Path(manifest['external_root']), method,
                               manifest['external_assets'], manifest['arc_build'])
        receipt.update(ar_extension=runner.expansion, performance_eligible=False)
        return receipt
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'contribution_study'))
    from baseline_runtime import verify
    from external import install_arc, install_sharq
    verify(manifest, method)
    spec = manifest['adapters'][method]
    if runner.config.execution != 'eager' or runner.config.attention_backend != 'bf16':
        raise ValueError('Pinned native ARC/SharQ require eager execution and BF16 attention')
    if spec['method'] == 'arcquant':
        ext = manifest['external']
        receipt = install_arc(runner, ext['arc_build'], ext['capture'], spec['fraction'])
    elif spec['method'] == 'sharq':
        ext = manifest['external']
        lock = json.loads(Path(ext['sources']).read_text())
        source = lock['sources']['sharq']
        receipt = install_sharq(runner, source['path'], source['commit'])
        for op in runner.engine.linears.values():
            op.official.extra_fusion = spec['extra_fusion']
        receipt.update(extra_fusion=spec['extra_fusion'], cosmos_rmsnorm_fusion=False)
    else:
        raise ValueError('Unknown body baseline adapter')
    if set(runner.engine.linears) != set(runner.engine.targets):
        raise AssertionError('Incomplete native baseline coverage')
    receipt.update(ar_extension=runner.expansion,
        scope='Official native generation linears; optional AR FFN extension is separately labelled',
        performance_eligible=False)
    return receipt
