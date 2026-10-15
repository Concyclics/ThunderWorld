"""Join pinned upstream provenance with explicit, bounded adaptation qualifications."""
import argparse,json
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.workspace;lock=root/'contribution_baselines_20260917/sources.lock.json'
    sources=json.loads(lock.read_text())['sources']
    definitions={
        'arcquant':dict(execution_type='official_native_cosmos_adapter',evidence='contribution_arcquant_probe_20260917.json',
            limits=['fixed 50% channels, no upstream model-wide proportion search','eager-only host alpha','K9216 dispatch/template patch'],
            build='contribution_arcquant_adapted_20260917/build.json'),
        'sharq':dict(execution_type='official_native_cosmos_adapter',evidence='contribution_sharq_adapter_probe_20260917.json',
            limits=['extra_fusion=False','native and official simulation are not bitwise equivalent'],
            build='contribution_baselines_20260917/sharq/kernels/build_cmake_sm120a/contribution-build.json'),
        'fouroversix':dict(execution_type='mechanism_reproduction',evidence='contribution_mechanism_all_20260917/report.json',
            limits=['scale choice only; not official native Cosmos port','full trajectory improvement failed on development seeds']),
        'viditq':dict(execution_type='collected_source_and_inspired_control',evidence=None,
            limits=['no faithful full ViDiT-Q Cosmos reproduction','existing measured layer/phase allocation is only an inspired control']),
        'sageattention':dict(execution_type='official_native_operator_adapter',evidence='contribution_sageattention_v2_20260917/probe.json',
            build='contribution_sageattention_v2_20260917/build.json',limits=['three layers, all 16 panels each','GQA expanded to query heads','K copy included; full generation not ported']),
        'nunchaku':dict(execution_type='official_source_collected_alternative_lite_reference',evidence='contribution_lowrank_20260917.json',
            limits=['executed reference is official NVIDIA ModelOpt lite-SVDQuant, not Nunchaku','six operator shapes only','BF16 low-rank costs separate; no native performance claim']),
        'modelopt':dict(execution_type='official_calibration_decoded_quality_reference',evidence='contribution_awq_20260917/assets/manifest.json',
            limits=['NVFP4_AWQ_LITE_CFG with A quantizers disabled before calibration','decoded BF16 cached weights, not packed W4 performance/storage']),
    }
    rows={}
    for name,definition in definitions.items():
        row={**sources[name],**definition,'performance_eligible':False}
        for field in ('evidence','build'):
            relative=definition.get(field)
            if relative:
                path=root/relative;value=json.loads(path.read_text())
                if not value['complete']:raise ValueError('Incomplete qualification '+name)
                row[field]=dict(path=str(path),sha256=sha(path))
        rows[name]=row
    write(a.output,dict(complete=True,sources_lock_sha256=sha(lock),baselines=rows,
        scope='Method identities and qualifications are separate from official source collection; none of these shared experiments is a native performance qualification.'))


if __name__=='__main__':main()
