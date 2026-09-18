"""True NVFP4 AWQ export and the installed official Marlin W4A16 kernel.

Export starts from original weights and the pinned AWQ calibration, never by
requantizing a decoded BF16 checkpoint. Native accumulation is a new numerical
path: operator checks do not qualify full-video quality or request performance.
"""
import argparse
import copy
import json
from pathlib import Path

from baselines import sha, write


def exact_codes_from_codebook(decoded, scale, factor):
    """Lossless extraction at the *original* quantizer scales, never re-fit them.

    Fake and real ModelOpt paths can make different boundary-rounding choices.
    Every frozen BF16 value must exactly equal a value in its original E2M1
    codebook after BF16 rounding; otherwise export fails. No tolerance or nearest
    approximate value is permitted. Signed zero is preserved as well.
    """
    import torch
    n, k = decoded.shape
    if decoded.dtype != torch.bfloat16 or k % 16 or scale.numel() != n*k//16:
        raise ValueError('Expected BF16 NVFP4 K16 representation')
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=decoded.device)
    scales = scale.float().reshape(n, k//16)*factor
    output = torch.empty((n, k//2), device=decoded.device, dtype=torch.uint8)
    for start in range(0, n, 64):
        values = decoded[start:start+64]
        grid = (scales[start:start+64, :, None]*levels).bfloat16()
        matches = values.abs().reshape(len(values), k//16, 16, 1) == grid[:, :, None, :]
        if not matches.any(-1).all():
            raise ValueError('Frozen AWQ value is outside its original FP4 codebook; lossless export is impossible')
        ids = matches.to(torch.int32).argmax(-1).reshape_as(values).to(torch.uint8)
        ids |= torch.signbit(values).to(torch.uint8)*8
        output[start:start+len(values)] = ids[:, 0::2] | (ids[:, 1::2] << 4)
    return output


def export(a):
    capture = json.loads((a.capture/'manifest.json').read_text())
    reference = json.loads((a.reference/'manifest.json').read_text())
    if not capture['complete'] or not reference['complete'] or capture['model_identity'] != reference['model_identity']:
        raise ValueError('Matching complete source and AWQ reference required')
    names = a.modules or list(reference['modules'])
    identity = dict(capture_sha256=sha(a.capture/'manifest.json'), reference_sha256=sha(a.reference/'manifest.json'),
        source_sha256=sha(__file__), scale_export_sha256=sha(Path(__file__).with_name('awq_scales.py')), modules=names)
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output/'manifest.json'
    state = dict(complete=False, identity=identity, modules={}, model_identity=reference['model_identity'],
        actual_precision='NVFP4_AWQ_LITE_W4A16', performance_eligible=False,
        scope='Original-weight official AWQ recalibration and actual E2M1/E4M3 codes/scales. '
              'Require bitwise decoded equality to official forward and existing AWQ quality assets. '
              'Marlin output accumulation requires separate full-generation qualification.')
    if path.exists():
        state = json.loads(path.read_text())
        if state['identity'] != identity:
            raise ValueError('Changed AWQ export resume')
        if state['complete']:
            return
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    with shared_worker():
        state['resource'] = select_shared_device()
        import torch
        import modelopt.torch.quantization as mtq
        from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
        from modelopt.torch.quantization.tensor_quant import DISABLE_TRITON_KERNEL
        from awq_scales import export_scales
        if DISABLE_TRITON_KERNEL:
            raise ValueError('Frozen AWQ reference used the official Triton fake quantizer')
        torch.set_num_threads(8)
        config = copy.deepcopy(mtq.NVFP4_AWQ_LITE_CFG)
        config['quant_cfg'] += [{'quantizer_name': '*input_quantizer', 'enable': False},
                                {'quantizer_name': '*output_quantizer', 'enable': False}]
        state['configuration'] = config
        try:
            with torch.inference_mode():
                for name in names:
                    if name in state['modules']:
                        item = state['modules'][name]
                        if sha(a.output/item['file']) != item['sha256']:
                            raise ValueError('Changed packed export')
                        continue
                    w = torch.load(a.capture/capture['weights'][name], weights_only=True).cuda()
                    linear = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, device='meta', dtype=torch.bfloat16)
                    linear.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
                    model = torch.nn.Sequential(linear)
                    rows = [r for r in capture['rows'] if r['kind'] == 'linear' and r['name'] == name]
                    if len(rows) != 16:
                        raise ValueError('Expected unchanged sixteen full-M panels')
                    panels = []
                    for row in rows:
                        file = a.capture/row['file']
                        if sha(file) != row['sha256']:
                            raise ValueError('Changed AWQ calibration panel')
                        panels.append(torch.load(file, weights_only=True)['x'].cuda())
                    def loop(current):
                        for x in panels:
                            current(x)
                    mtq.quantize(model, config, forward_loop=loop)
                    module = model[0]
                    tq = module.weight_quantizer
                    if module.input_quantizer.is_enabled or not tq.is_enabled or tq.pre_quant_scale is not None:
                        raise ValueError('Unsupported AWQ scale folding or activation quantization')
                    decoded = tq(module.weight).detach()
                    scale, global_scale = NVFP4QTensor.get_weights_scaling_factor_from_quantizer(tq, module.weight)
                    if not tq.is_nvfp4_dynamic:
                        raise ValueError('Only the frozen dynamic NVFP4 AWQ quantizer is qualified')
                    actual_scale = export_scales(module.weight, global_scale)
                    scale_differences = int((actual_scale.float() != scale.float()).sum())
                    scale = actual_scale
                    quant, scale, global_scale = NVFP4QTensor.quantize(module.weight, 16,
                        weights_scaling_factor=scale, weights_scaling_factor_2=global_scale, try_tensorrt=False)
                    restored = quant.dequantize(dtype=torch.bfloat16, scale=scale, double_scale=global_scale, block_sizes={-1: 16})
                    real_export_differences = int((restored != decoded).sum())
                    if real_export_differences:
                        # Recover the original fake path's exact boundary choices
                        # at unchanged scales, rather than accept a nearby weight.
                        exact = exact_codes_from_codebook(decoded, scale, global_scale)
                        quant = NVFP4QTensor(module.weight.shape, torch.bfloat16, exact)
                        restored = quant.dequantize(dtype=torch.bfloat16, scale=scale, double_scale=global_scale, block_sizes={-1: 16})
                    prescale = module.input_quantizer.pre_quant_scale
                    if prescale is None:
                        prescale = torch.ones(w.shape[1], device=w.device, dtype=w.dtype)
                    item = reference['modules'][name]
                    if sha(a.reference/item['file']) != item['sha256']:
                        raise ValueError('Changed decoded AWQ reference')
                    old = torch.load(a.reference/item['file'], weights_only=True)
                    if not torch.equal(restored, decoded) or not torch.equal(decoded.cpu(), old['weight']) or not torch.equal(prescale.cpu(), old['prescale']):
                        comparisons = {}
                        for label, left, right in (('real_vs_official_fake', restored, decoded),
                            ('official_fake_vs_frozen', decoded.cpu(), old['weight']),
                            ('prescale_vs_frozen', prescale.cpu(), old['prescale'])):
                            comparisons[label] = dict(bitwise_equal=torch.equal(left, right),
                                different_elements=int((left != right).sum()),
                                max_abs=float((left.float()-right.float()).abs().max()),
                                relative_l2=float((left.float()-right.float()).norm()/right.float().norm().clamp_min(1e-20)))
                        write(a.output/'export_mismatch.json', dict(complete=True, module=name, comparisons=comparisons,
                            scope='Rejected real export; do not substitute its codes for the frozen fake-quantized AWQ quality baseline.'))
                        raise ValueError('Native export does not reproduce the frozen AWQ representation')
                    for x in panels:
                        if not torch.equal(model(x), torch.nn.functional.linear(x*prescale, restored)):
                            raise ValueError('Official forward differs from exported representation')
                    asset = dict(codes=quant._quantized_data.cpu(), scales=scale.cpu(), factor=global_scale.cpu(),
                        prescale=prescale.cpu(), shape=list(w.shape))
                    file = a.output/(name+'.pt')
                    torch.save(asset, file)
                    state['modules'][name] = dict(file=file.name, sha256=sha(file),
                        tensor_bytes=sum(v.numel()*v.element_size() for v in asset.values() if isinstance(v, torch.Tensor)),
                        decoded_bitwise_equal=True, original_weight_calibration=True,
                        stock_real_export_different_elements=real_export_differences,
                        python_vs_official_triton_scale_different_groups=scale_differences,
                        extraction='Exact original-scale codebook inversion' if real_export_differences else 'Official real quantizer')
                    write(path, state)
                    print(json.dumps(dict(module=name, completed=len(state['modules']), total=len(names))), flush=True)
                    del panels, model, module, linear, tq, w, decoded, quant, restored
                    torch.cuda.empty_cache()
            state.update(complete=True, complete_model=set(names) == set(reference['modules']))
            state.pop('paused', None)
            state.pop('error', None)
            write(path, state)
        except Exception as error:
            state.update(paused=True, error=repr(error))
            write(path, state)
            raise


def make_linear(asset):
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_fp4_layer_for_marlin, apply_fp4_marlin_linear
    class PackedAWQLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            n, k = asset['shape']
            self.in_features = self.input_size_per_partition = k
            self.out_features = self.output_size_per_partition = n
            self.params_dtype = torch.bfloat16
            self.weight = torch.nn.Parameter(asset['codes'].cuda(), requires_grad=False)
            self.weight_scale = torch.nn.Parameter(asset['scales'].cuda(), requires_grad=False)
            self.weight_global_scale = torch.nn.Parameter(asset['factor'].cuda(), requires_grad=False)
            self.register_buffer('prescale', asset['prescale'].cuda())
            self.bias = None
            prepare_fp4_layer_for_marlin(self)
            workspace = self.workspace
            del self.workspace
            self.register_buffer('workspace', workspace)

        def forward(self, x):
            y = apply_fp4_marlin_linear(x*self.prescale, self.weight, self.weight_scale,
                self.weight_global_scale, self.workspace, self.out_features, self.in_features,
                use_fp32_reduce=True)
            return y if self.bias is None else y+self.bias
    return PackedAWQLinear()


def install(runner, assets):
    """Install before capture; preserve the existing Cosmos inference engine."""
    import torch
    import vllm
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4
    assets = Path(assets)
    manifest = json.loads((assets/'manifest.json').read_text())
    if (not manifest['complete'] or not manifest.get('complete_model') or
        manifest['model_identity'] != runner.identity or set(manifest['modules']) != set(runner.engine.targets)):
        raise ValueError('Complete matching all-layer native AWQ assets required')
    if runner.engine.cache or runner._warmed:
        raise ValueError('Install AWQ before warmup and Graph capture')
    weights, scratch = {}, {}
    for name, entry in manifest['modules'].items():
        file = assets/entry['file']
        if sha(file) != entry['sha256'] or not entry['decoded_bitwise_equal']:
            raise ValueError('Changed/unqualified native AWQ representation')
        source = runner.pipe.transformer.get_submodule(name)
        op = make_linear(torch.load(file, weights_only=True))
        if (op.out_features, op.in_features) != (source.out_features, source.in_features):
            raise ValueError('Mismatched AWQ operator shape')
        if source.bias is not None:
            del op.bias
            op.register_buffer('bias', source.bias.detach().clone())
        runner.pipe.transformer.set_submodule(name, op)
        runner.engine.linears[name] = op
        for value in (*op.parameters(), op.prescale, op.bias):
            if value is not None:
                storage = value.untyped_storage()
                weights[storage.data_ptr()] = storage.nbytes()
        storage = op.workspace.untyped_storage()
        scratch[storage.data_ptr()] = storage.nbytes()
    return dict(method='NVFP4-AWQ-lite-W4A16-Marlin', execution_type='official_native_marlin_cosmos_adapter',
        assets_sha256=sha(assets/'manifest.json'), vllm_version=vllm.__version__,
        adapter_sha256=sha(__file__), marlin_utility_sha256=sha(marlin_utils_fp4.__file__),
        resident_generation_weight_bytes=sum(weights.values()), scratch_bytes=sum(scratch.values()),
        generic_prepared_weight_bytes_qualified=False,
        performance_eligible=False, storage_comparison_eligible=True,
        limitations=['Whole-request quality must be remeasured for Marlin accumulation',
                     'Current-input AWQ prescale is included as a separate operation',
                     'Use adapter weight/scratch accounting; generic prepared_weight_bytes omits external Parameters',
                     'Shared calibration timings are ineligible'])


def probe(a):
    manifest = json.loads((a.assets/'manifest.json').read_text())
    if not manifest['complete']:
        raise ValueError('Complete original-weight packed export required')
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    with shared_worker():
        resource = select_shared_device()
        import torch
        import vllm
        from thunderworld_fp4.cosmos.calibration import decode
        torch.set_num_threads(8)
        rows = []
        with torch.inference_mode():
            for name, entry in manifest['modules'].items():
                if sha(a.assets/entry['file']) != entry['sha256']:
                    raise ValueError('Changed packed asset')
                asset = torch.load(a.assets/entry['file'], weights_only=True)
                op = make_linear(asset)
                # Independent decode multiplies group/global scales before FP4 values.
                scales = asset['scales'].cuda().float()*asset['factor'].cuda()
                gold_weight = decode(asset['codes'].cuda(), scales, 1.).bfloat16()
                for m in (129, 416, 960, 1024):
                    gen = torch.Generator(device='cuda').manual_seed(180918+m)
                    x = torch.randn(m, op.in_features, device='cuda', dtype=torch.bfloat16, generator=gen)
                    for _ in range(3):
                        op(x)
                    graph = torch.cuda.CUDAGraph()
                    fixed = x.clone()
                    torch.cuda.synchronize()
                    with torch.cuda.graph(graph):
                        output = op(fixed)
                    for changed in (x, x*.75+.125, torch.zeros_like(x)):
                        fixed.copy_(changed)
                        graph.replay()
                        eager = op(changed)
                        if not torch.equal(output, eager):
                            raise ValueError('Marlin graph/current-input output differs or is nondeterministic')
                        ref = torch.nn.functional.linear(changed*op.prescale, gold_weight)
                        error = float((output.float()-ref.float()).square().sum()/ref.float().square().sum().clamp_min(1e-20))
                        if error > 1e-4 or not torch.isfinite(output).all():
                            raise ValueError('Marlin decoded-reference NMSE exceeds the predeclared 1e-4 operator screen')
                        rows.append(dict(name=name, m=m, n=op.out_features, k=op.in_features, nmse=error,
                            graph_eager_bitwise_equal=True, input_case='zero' if not changed.any() else 'nonzero'))
                    del graph, output
        write(a.output, dict(complete=True, rows=rows, resource=resource, vllm_version=vllm.__version__,
            source_sha256=sha(__file__), assets_sha256=sha(a.assets/'manifest.json'),
            performance_eligible=False, full_generation_qualified=False,
            scope='Official Marlin NVFP4 W4A16 operator/Graph screen, BF16 inputs and FP32 reduction. '
                  'Independent decoded-weight reference; changed accumulation needs full trajectory quality qualification. No latency measurement.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('export', 'probe'))
    p.add_argument('--output', type=Path, required=True)
    for key in ('capture', 'reference', 'assets'):
        p.add_argument('--'+key, type=Path)
    p.add_argument('--modules', nargs='+')
    a = p.parse_args()
    if a.operation == 'export' and (a.capture is None or a.reference is None):
        p.error('export requires --capture and --reference')
    if a.operation == 'probe' and a.assets is None:
        p.error('probe requires --assets')
    (export if a.operation == 'export' else probe)(a)


if __name__ == '__main__':
    main()
