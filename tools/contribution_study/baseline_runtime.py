"""Frozen, existing execution paths for the matched FP4 baseline supplement.

No baseline kernels or launch parameters are tuned here. ARC and SharQ retain
their official default-stream kernels and eager execution. SharQ's performance
variant enables its existing epilogue flag; no Cosmos RMSNorm port is implied.
"""
import json
from pathlib import Path

from baselines import sha


def verify(manifest, name):
    spec = manifest['methods'][name]
    if spec.get('assets') and sha(Path(spec['assets'])/'manifest.json') != spec['assets_sha256']:
        raise ValueError('Changed offline repair assets')
    if spec['config'].get('budget_plan') and sha(spec['config']['budget_plan']) != spec['plan_sha256']:
        raise ValueError('Changed repair plan')
    ext = manifest.get('external', {})
    external = manifest.get('adapters', {}).get(name)
    checks = []
    if external:
        checks.append((Path(ext['sources']), ext['sources_sha256']))
        if external['method'] == 'arcquant':
            checks.extend(((Path(ext['arc_build'])/'build.json', ext['arc_build_sha256']),
                           (Path(ext['capture'])/'manifest.json', ext['capture_sha256'])))
        elif external['method'] == 'sharq' and ext.get('sharq_build'):
            checks.append((Path(ext['sharq_build'])/'contribution-build.json', ext['sharq_build_sha256']))
    for path, digest in checks:
        if sha(path) != digest:
            raise ValueError('Changed official baseline input: '+str(path))
    adapter = spec.get('baseline_adapter')
    if adapter:
        implementations = {'marlin_awq': 'awq_marlin.py', 'awq': 'external.py'}
        if adapter['method'] not in implementations:
            raise ValueError('Unknown AWQ adapter')
        if (sha(Path(adapter['assets'])/'manifest.json') != adapter['assets_sha256'] or
            sha(Path(__file__).with_name(implementations[adapter['method']])) != adapter['implementation_sha256']):
            raise ValueError('Changed AWQ input')
        if (spec.get('assets') or spec['config']['precision'] != 'bf16' or
                spec['config']['attention_backend'] != 'bf16'):
            raise ValueError('AWQ adapter requires BF16 dispatch and attention')


def make_runner(manifest, name):
    from thunderworld_fp4.cosmos import CosmosRunner, RunConfig
    verify(manifest, name)
    spec = manifest['methods'][name]
    cfg = RunConfig.from_dict(spec['config'])
    if cfg.debug != 'off' or cfg.resource_mode != 'shared-calibration':
        raise ValueError('Frozen debug-off diagnostic configuration required')
    # New experiments can freeze exact execution options without changing the
    # historical presets or their numerical policies. External adapters retain
    # their original runner unless separately qualified.
    options = dict(spec.get('runtime_options', {}))
    allowed = {'fuse_calibrated', 'fuse_pointwise', 'static_rotary', 'codec_graph',
               'fuse_norm', 'qkv_streams'}
    if set(options) - allowed or any(type(v) is not bool for v in options.values()):
        raise ValueError('Unsupported frozen runtime options')
    if options and (name in manifest.get('adapters', {}) or spec.get('baseline_adapter')):
        raise ValueError('External baseline execution changes need separate qualification')
    runner = CosmosRunner(manifest['model'], cfg, assets=spec.get('assets'),
                          ar_cache=True, first_frame=True, **options).prepare()
    adapter = None
    external = manifest.get('adapters', {}).get(name)
    if external and external['method'] == 'arcquant':
        from external import install_arc
        ext = manifest['external']
        if cfg.execution != 'eager':
            raise ValueError('Official ARC host-alpha path requires eager')
        adapter = install_arc(runner, ext['arc_build'], ext['capture'], external['fraction'])
    elif external and external['method'] == 'sharq':
        from external import install_sharq
        lock = json.loads(Path(manifest['external']['sources']).read_text())
        entry = lock['sources']['sharq']
        adapter = install_sharq(runner, entry['path'], entry['commit'])
        if manifest['external'].get('sharq_build'):
            build = json.loads((Path(manifest['external']['sharq_build'])/'contribution-build.json').read_text())
            if not build['complete'] or adapter['library_sha256'] != build['library_sha256']:
                raise ValueError('Loaded SharQ extension differs from the prepared build')
        enabled = external['extra_fusion']
        for layer in runner.engine.linears.values():
            layer.official.extra_fusion = enabled
        adapter.update(extra_fusion=enabled, cosmos_rmsnorm_fusion=False,
            scope='Official native sparse/residual kernels; existing epilogue fusion flag only. '
                  'Cosmos RMSNorm is outside the wrapper; no baseline-specific fusion port.')
    elif spec.get('baseline_adapter'):
        if spec['baseline_adapter']['method'] == 'marlin_awq':
            from awq_marlin import install
        else:
            from external import install_awq as install
        adapter = install(runner, spec['baseline_adapter']['assets'])
    return runner, adapter


def default_stream_only(manifest, name):
    # Pinned upstream launches omit a stream or call CUTLASS run() with the
    # default stream. Do not silently move these kernels to a Torch side stream.
    return manifest.get('adapters', {}).get(name, {}).get('method') in ('arcquant', 'sharq')
