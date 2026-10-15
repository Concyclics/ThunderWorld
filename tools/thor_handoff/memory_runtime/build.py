"""Create an isolated memory variant from the hash-verified Thor v6 package."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def replace(text, before, after, count=1):
    if text.count(before) != count:
        raise ValueError('Unexpected parent source: ' + before[:100])
    return text.replace(before, after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/chenhan/thunderworld'))
    parser.add_argument('--name', default='cutlass-memory-v7')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    parent = args.root / 'variants/cutlass-mainloop-v6'
    old = json.loads((parent / 'variant.json').read_text())
    for name, digest in old['variant_hashes'].items():
        if sha(parent / 'thunderworld_fp4' / name) != digest:
            raise ValueError('Parent source changed: ' + name)
    target = args.root / 'variants' / args.name
    target.mkdir(parents=True, exist_ok=False)
    package = target / 'thunderworld_fp4'
    shutil.copytree(parent / 'thunderworld_fp4', package, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    source = target / 'source'
    source.mkdir()
    shutil.copy2(here / 'compact_pack.py', package / '_sm110_compact_pack.py')
    shutil.copy2(here / 'host_assets.py', package / '_host_assets.py')
    def edit(path, before, after):
        path.write_text(replace(path.read_text(), before, after))
    packing = package / 'cutlass/packing.py'
    edit(packing, 'def pack_input(x, op,', 'def pack_input_dense(x, op,')
    with packing.open('a') as stream:
        stream.write('''\n\ndef pack_input(x, op, **kwargs):
    from .._sm110_compact_pack import compact_bytes, pack_compact
    if compact_bytes(x.shape[0], x.shape[1], op.na)['enabled']:
        return pack_compact(x, op, **kwargs)
    return pack_input_dense(x, op, **kwargs)
''')
    linears = package / 'cosmos/linears.py'
    edit(linears, '    def export(self):\n        if self.weight_union is not None:',
         '''    def export(self):
        if getattr(self, '_host_assets_released', False):
            raise RuntimeError('Prepared CPU assets released; new shape/export needs a fresh runner')
        if self.weight_union is not None:''')
    runner = package / 'cosmos/runner.py'
    text = runner.read_text()
    start = text.index('            assets={}\n')
    end = text.index('        start=time.perf_counter()', start)
    text = text[:start] + '''            from .._host_assets import LazyAssets
            assets=LazyAssets(self.assets_path,manifest,self.config,policy)
''' + text[end:]
    text = replace(text, '        if key in self._warmed:return\n', '''        if key in self._warmed:return
        if getattr(self, '_host_assets_released', False):
            raise RuntimeError('Prepared CPU assets released; warm a new shape in a fresh runner')
''')
    runner.write_text(text)
    with runner.open('a') as stream:
        stream.write('''\n    def release_host_assets(self):
        from .._host_assets import release_host_assets
        return release_host_assets(self)
''')
    backend = package / '_sm110_cutlass.py'
    text = backend.read_text()
    text = replace(text, "'wloc','mask')", "'wloc','mask','ar','ars','aloc')")
    text = replace(text, "'dense','bank_blocks')]", "'dense','bank_blocks','compact')]")
    text = replace(text, "'cutlass-sm110-fused-mainloop':", "'cutlass-sm110-compact-mainloop':")
    text = replace(text, '        if op.na and ac.shape[0]<2:raise ValueError',
                   "        compact=getattr(direct,'compact_repair',None)\n        if op.na and ac.shape[0]<2 and compact is None:raise ValueError")
    text = replace(text, '        args=RepairIO(*[t.data_ptr()',
                   '''        rc,rs,locations=compact if compact is not None else (ac,sc,op.wloc)
        if compact is not None:
            blocks=triton.cdiv(self.m,128)*op.na
            if rc.shape!=(blocks,128,64) or rs.shape!=(blocks,128,8) or locations.shape!=(triton.cdiv(self.m,128),op.k//128):
                raise ValueError('Invalid compact activation repair shape')
            if rc.dtype!=torch.uint8 or rs.dtype!=torch.float8_e4m3fn or locations.dtype!=torch.int32:
                raise ValueError('Invalid compact activation repair dtype')
            if any(t.device!=x.device or not t.is_contiguous() for t in compact):
                raise ValueError('Invalid compact activation repair storage')
        args=RepairIO(*[t.data_ptr()''')
    text = replace(text, 'op.wloc,op.am)],', 'op.wloc,op.am,rc,rs,locations)],')
    text = replace(text, 'int(op.dense_w),op.wr.shape[0])', 'int(op.dense_w),op.wr.shape[0],int(compact is not None))')
    backend.write_text(text)
    # Addressing changes only. The MMA loops, selected masks, term order and
    # single BF16 epilogue are byte-for-byte unchanged from v6 source.
    header = (here.parent / 'sm110_repair_collective.cuh').read_text()
    header = replace(header, '  uint8_t const *mask;',
                     '  uint8_t const *mask;\n  uint8_t const *ar, *ars;\n  int const *aloc;')
    header = replace(header, 'shared, dense, bank_blocks;', 'shared, dense, bank_blocks, compact;')
    header = replace(header, 'auto ap=p.ac+int64_t(p.na>0)*p.mp*p.k/2;',
                     'auto ap=p.compact?p.ar:p.ac+int64_t(p.na>0)*p.mp*p.k/2;')
    header = replace(header, '''    auto ar=make_tensor(recast_ptr<typename Base::TmaInternalElementA>(ap),
        make_layout(make_shape(p.m,p.k,1),args.dA));''', '''    int arows=p.compact?((p.m+127)/128)*p.na*128:p.m;
    int acols=p.compact?128:p.k;
    typename Base::StrideA ad=cutlass::make_cute_packed_stride(typename Base::StrideA{}, {arows,acols,1});
    auto ar=make_tensor(recast_ptr<typename Base::TmaInternalElementA>(ap),
        make_layout(make_shape(arows,acols,1),ad));''')
    header = replace(header, 'auto ar=p_->tma_ar.get_tma_tensor(make_shape(p.m,p.k,1));',
                     'auto ar=p_->tma_ar.get_tma_tensor(make_shape(p.compact?((p.m+127)/128)*p.na*128:p.m,p.compact?128:p.k,1));')
    header = replace(header, 'inputs.ar(_,mb,kb,0)',
                     'inputs.ar(_,p.compact?p.aloc[mb*(p.k/128)+kb]:mb,p.compact?0:kb,0)')
    header = replace(header, '    auto asp=p.as+int64_t(ra)*p.mp*(p.k/16)+int64_t(mb)*128*(p.k/16)+kb*8;',
                     '''    auto asp=(ra && p.compact)?p.ars+int64_t(p.aloc[mb*(p.k/128)+kb])*1024:
        p.as+int64_t(ra)*p.mp*(p.k/16)+int64_t(mb)*128*(p.k/16)+kb*8;''')
    header = replace(header, 'asp+int64_t(row)*(p.k/16)+col', 'asp+int64_t(row)*(p.compact?8:p.k/16)+col')
    (source / 'sm110_repair_collective.cuh').write_text(header)
    shutil.copy2(here.parent / 'cutlass_sm110_fused.cu', source / 'cutlass_sm110_fused.cu')
    cutlass = args.root / 'external/arcquant/third-party/cutlass'
    library = target / 'cutlass_sm110.so'
    command = ['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','--expt-relaxed-constexpr','--expt-extended-lambda',
               '-gencode','arch=compute_110a,code=sm_110a','--shared','-Xcompiler=-fPIC',
               '-I'+str(cutlass/'include'),'-I'+str(cutlass/'tools/util/include'),
               str(source/'cutlass_sm110_fused.cu'),'-o',str(library)]
    with (target/'build.log').open('w') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    build = dict(backend='cutlass-sm110-compact-mainloop', architecture='sm_110a', tile=[128,128,128],
                 command=command, sources={p.name:sha(p) for p in source.iterdir()}, library_sha256=sha(library))
    library.with_suffix('.build.json').write_text(json.dumps(build, indent=2)+'\n')
    receipt = dict(old)
    receipt.update(created=time.time(), parent=dict(path=str(parent),sha256=sha(parent/'variant.json')),
                   name=args.name, qualified=False, memory_optimized=True,
                   library_sha256=sha(library),build_receipt_sha256=sha(library.with_suffix('.build.json')),
                   variant_hashes={str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*.py'))},
                   memory_source_hashes={p.name:sha(p) for p in sorted(here.glob('*.py'))})
    for key in ('qualification','sanitizer_receipts'):
        receipt.pop(key, None)
    (target/'variant.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(variant=str(target),library_sha256=sha(library),qualified=False)),flush=True)


if __name__ == '__main__': main()
