"""Further attribution using the pinned model's fixed execution order.

This is source-order inference, not module-tagged timing. Assert all 70 calls
have the expected GEMM sequence and retain the supporting counts. No GPU needed.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();es=json.loads(a.trace.read_text())['traceEvents']
    spans=sorted((e for e in es if e.get('cat')=='gpu_user_annotation' and e['name']=='TW/transformer'),key=lambda e:e['ts'])
    assert len(spans)==70
    kernels=sorted((e for e in es if e.get('cat')=='kernel'),key=lambda e:e['ts'])
    sums=defaultdict(lambda:dict(count=0,kernel_ms=0.));position=0
    for span in spans:
        while position<len(kernels) and kernels[position]['ts']<span['ts']:position+=1
        group=[]
        while position<len(kernels) and kernels[position]['ts']<span['ts']+span['dur']:
            group.append(kernels[position]);position+=1
        bf=[e for e in group if 'gemm' in e['name'].lower() and 'bf16' in e['name'] and 'GemmUniversal' not in e['name'] and 'splitK' not in e['name']]
        fp=[e for e in group if 'gemm' in e['name'].lower() and 'bf16' not in e['name'] and 'GemmUniversal' not in e['name'] and e['name']!='gemm']
        assert len(bf)==170
        assert len(fp)==3
        assert 'magma_sgemmEx' in fp[0]['name'] and 'simt_sgemm' in fp[1]['name'] and 'gemmk1' in fp[2]['name']
        labels=['vision_proj_in']+[name for _ in range(28) for name in ('AR_Q','AR_K','AR_V','AR_out','AR_FFN_up','AR_FFN_down')]+['vision_proj_out']
        mapping={id(e):label for e,label in zip(bf,labels)}
        mapping.update({id(e):label for e,label in zip(fp,('FP32_time_embed_linear1','FP32_time_embed_linear2','FP32_rotary_frequency'))})
        last=None
        for e in group:
            if id(e) in mapping:label=last=mapping[id(e)]
            elif 'splitKreduce' in e['name']:
                assert last and last.startswith('AR_')
                label=last+'_splitK'
            else:continue
            sums[label]['count']+=1;sums[label]['kernel_ms']+=e['dur']/1000
    total=sum(v['kernel_ms'] for v in sums.values())
    ar=sum(v['kernel_ms'] for k,v in sums.items() if k.startswith('AR_'))
    cpu={e.get('args',{}).get('External id'):e for e in es if e.get('cat')=='cpu_op'}
    codec_gemm=[e for e in kernels if 'gemm' in e['name'].lower() and 'bf16' in e['name'] and
                cpu.get(e['args'].get('External id'),{}).get('name')=='aten::cudnn_convolution']
    # Filter to the GEMM category; convolution symbols also contain implicit_gemm.
    codec_gemm=[e for e in codec_gemm if 'cudnn' not in e['name'].lower()]
    result=dict(complete=True,trace_sha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(),
        method='Source-order inference, verified sequence lengths on all 70 Transformer calls; split-K attached to preceding source GEMM. Codec classification uses External id -> aten::cudnn_convolution.',
        roles=dict(sums),AR_gemm_and_splitK_ms=ar,transformer_high_precision_gemm_ms=total,
        codec_convolution_lowered_to_bf16_gemm=dict(count=len(codec_gemm),kernel_ms=sum(e['dur'] for e in codec_gemm)/1000),
        decoder_tiling=dict(latent_shape=[12,20],tile=16,stride=12,tile_shapes=[[12,16],[12,8]],
            input_area_ratio=(12*16+12*8)/(12*20),scope='Input tile area only, not a measured FLOP or speed ratio'),
        encoder_tiling=dict(patchified_shape=[96,160],tile=256,stride=192,tile_count=1))
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
