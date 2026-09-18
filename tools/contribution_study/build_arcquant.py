"""Build an isolated ARCQuant adaptation adding Cosmos K9216 instantiations."""
import argparse,difflib,json,os,shutil,subprocess,sys
from pathlib import Path
from baselines import git,sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('sources','output','pybind11'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();entry=json.loads(a.sources.read_text())['sources']['arcquant']
    source=Path(entry['path'])
    if git(source,'rev-parse','HEAD')!=entry['commit']:raise ValueError('Upstream changed')
    a.output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(source/'kernels',a.output/'kernels',ignore=shutil.ignore_patterns('build*'))
    (a.output/'third-party').mkdir()
    (a.output/'third-party/cutlass').symlink_to(source/'third-party/cutlass',target_is_directory=True)
    patches=[]
    for filename in ('bindings.cpp','reorder.cu'):
        path=a.output/'kernels/src'/filename;before=path.read_text();after=before
        if filename=='bindings.cpp':
            for side in ('X','W'):
                needle=f'        CASE_REORDER_{side}_16(2048)'
                if after.count(needle)!=1:raise ValueError('Unexpected upstream dispatch')
                after=after.replace(needle,f'        CASE_REORDER_{side}_16(9216)\n'+needle)
        else:
            for side in ('x','w'):
                after+=f'\ntemplate void run_reorder_{side}_bf16_nvfp4<16, 9216>(bf16_t*, int, int16_t*, uint8_t*, sf_t*, int, int);\n'
        path.write_text(after)
        patches.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
            fromfile='a/kernels/src/'+filename,tofile='b/kernels/src/'+filename))
    (a.output/'cosmos.patch').write_text(''.join(patches))
    env={**os.environ,'TORCH_CUDA_ARCH_LIST':'12.0a'}
    commands=[['cmake','-S',str(a.output/'kernels'),'-B',str(a.output/'kernels/build'),
        '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc',
        '-DPython3_EXECUTABLE='+sys.executable,'-Dpybind11_DIR='+str(a.pybind11)],
        ['cmake','--build',str(a.output/'kernels/build'),'--target','agemm','-j','4']]
    manifest=dict(complete=False,source_commit=entry['commit'],cutlass_commit=git(source/'third-party/cutlass','rev-parse','HEAD'),
        patches={'cosmos.patch':sha(a.output/'cosmos.patch')},commands=commands,architecture='sm_120a',
        performance_eligible=False,scope='Only K9216 dispatch and template instantiation; upstream quantization unchanged.')
    write(a.output/'build.json',manifest)
    for i,cmd in enumerate(commands):
        with (a.output/f'build_{i}.log').open('w') as stream:
            subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    manifest.update(complete=True,library_sha256=sha(a.output/'kernels/build/agemm.so'))
    write(a.output/'build.json',manifest)


if __name__=='__main__':main()
