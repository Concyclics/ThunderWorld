"""ARCQuant K25600 launch legality port; upstream scalar quantization unchanged.

The upstream K16 packer assigns one thread to each group in a whole row. K25600
would require 1600 threads. For wide K only, partition groups across CTAs and
gather the original BF16 elements directly; each group's arithmetic is copied
verbatim. Smaller shapes retain upstream kernels. Quality-only qualification.
"""
import argparse
import difflib
import json
from pathlib import Path
import shutil
import subprocess

from baselines import sha, write


def patch(source):
    if 'reorder_wide_x_kernel' in source:
        raise ValueError('Wide-K patch already installed')
    kernels = []
    for side in ('x', 'w'):
        start = source.index('template <int bdx, int GROUP_SIZE, int HIDDEN_DIM>\n__global__ void reorder_'+side+'_kernel(')
        end = source.index('\n}', start) + 2
        original = source[start:end]
        revised = original.replace('reorder_'+side+'_kernel', 'reorder_wide_'+side+'_kernel')
        a = revised.index('  // One block solves one row of hidden states.')
        b = revised.index('  // Local memory stores', a)
        revised = revised[:a] + revised[b:]
        revised = revised.replace('  int tx = threadIdx.x;\n  int tid = tx;',
            '  int tid = blockIdx.y * blockDim.x + threadIdx.x;\n  if (tid >= bdx) return;')
        a = revised.index('  #pragma unroll\n  for(int i = 0;i < iters;++i)')
        b = revised.index('  // Reorder', a)
        revised = revised[:a] + revised[b:]
        revised = revised.replace('input_smem[reorder_index[offset]]', 'input[reorder_index[offset]]')
        revised = revised.replace('  cta.sync();\n', '')
        kernels.append(revised)
    anchor = 'template<int group_size, int hidden_dim>\nvoid run_reorder_x_bf16_nvfp4('
    revised = source.replace(anchor, '\n\n'.join(kernels)+'\n\n'+anchor, 1)
    for side, rows, layout in [('x', 'seq_len', 'SFA'), ('w', 'out_features', 'SFB')]:
        anchor = f'  Tensor q_scale_tensor = cute::make_tensor(q_scale, filter_zeros(nvfp4::get_layout{layout}({rows}, KQ + KE)));\n'
        prefix = f'''  if constexpr (hidden_dim / group_size > 1024) {{
    dim3 wide_grids({rows}, (hidden_dim / group_size + 255) / 256);
    reorder_wide_{side}_kernel<hidden_dim / group_size, group_size, hidden_dim><<<wide_grids, 256>>>(
      hidden_states, reorder_index, q_out, q_scale_tensor, KQ, KE);
    return;
  }}
'''
        # There is also a group32 launcher; only replace the first group16 one.
        if anchor not in revised:
            raise ValueError('Upstream launcher changed')
        revised = revised.replace(anchor, anchor+prefix, 1)
    return revised


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    a = p.parse_args()
    root = a.build.resolve()
    manifest_path = root/'build.json'
    value = json.loads(manifest_path.read_text())
    source = root/'kernels/src/reorder.cu'
    if 'wide_k_port' not in value:
        history = root/'history/instantiation_only'
        history.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, history/'build.json')
        shutil.copy2(source, history/'reorder.cu')
        shutil.copy2(root/'kernels/build/agemm.so', history/'agemm.so')
        before = source.read_text()
        after = patch(before)
        source.write_text(after)
        (root/'wide_k.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
            fromfile='a/kernels/src/reorder.cu', tofile='b/kernels/src/reorder.cu')))
        value['patches']['wide_k.patch'] = sha(root/'wide_k.patch')
        value.update(complete=False, wide_k_port=dict(source_sha256=sha(__file__),
            group_size=16, threads=256, minimum_groups=1025,
            changes='Multi-CTA group ownership and direct BF16 gather only; scalar arithmetic unchanged',
            performance_eligible=False))
        write(manifest_path, value)
    elif value['wide_k_port']['source_sha256'] != sha(__file__):
        raise ValueError('Wide-K builder changed')
    with (root/'wide_build.log').open('a') as log:
        subprocess.run(value['commands'][1], stdout=log, stderr=subprocess.STDOUT, check=True)
    value.update(complete=True, library_sha256=sha(root/'kernels/build/agemm.so'))
    write(manifest_path, value)


if __name__ == '__main__':
    main()
