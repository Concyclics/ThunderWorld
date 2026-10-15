"""Run existing research workers with explicit package paths and Super placement."""
import argparse
import os
import runpy
import sys
from common import ROOT, STUDY


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=('edge', 'nano', 'super'), required=True)
    p.add_argument('--script', required=True)
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    a = p.parse_args()
    script = (STUDY / a.script).resolve()
    if script.parent != STUDY.resolve() or not script.is_file() or script.suffix != '.py':
        raise ValueError('Worker must name a bundled contribution-study script')
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if devices:
        os.environ.setdefault('TW_SHARED_CALIBRATION_GPUS', devices)
    sys.path.insert(0, str(ROOT / 'tools/joint_inference'))
    sys.path.insert(0, str(ROOT / 'experiments/cutlass_fused_repair'))
    if a.model == 'super':
        from super_runtime import activate
        activate()
    argv = a.arguments[1:] if a.arguments[:1] == ['--'] else a.arguments
    sys.argv = [str(script), *argv]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
