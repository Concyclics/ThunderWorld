"""Fetch pinned official baseline sources and reproduce their SM120a builds."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from bundle import ROOT, check_package

# This script shares a filename with the study helper; import by exact path.
_spec = importlib.util.spec_from_file_location('study_baselines', ROOT/'tools/contribution_study/baselines.py')
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('fetch','build'))
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True, help='Receiver-local official checkout directory')
    parser.add_argument('--arc-build', type=Path)
    parser.add_argument('--pybind11', type=Path, help='CMake package directory; otherwise use python -m pybind11')
    args = parser.parse_args()
    check_package(args.package)
    lock = args.package/'sources.lock.json'
    if args.operation == 'fetch':
        result = _helpers.collect(args.sources, ['arcquant','sharq'], source_lock=lock)
        for entry in result['sources'].values():
            subprocess.run(['git','-C',entry['path'],'submodule','update','--init','--depth=1','third-party/cutlass'], check=True)
    else:
        if args.arc_build is None:parser.error('build requires --arc-build')
        local_lock = args.sources/'sources.lock.json'
        local = json.loads(local_lock.read_text())['sources']
        pinned = json.loads(lock.read_text())['sources']
        for name in ('arcquant','sharq'):
            if any(local[name][key] != pinned[name][key] for key in ('url','commit')):
                raise ValueError('Wrong upstream version')
        pybind = args.pybind11 or Path(subprocess.check_output([sys.executable,'-m','pybind11','--cmakedir'], text=True).strip())
        subprocess.run([sys.executable,str(ROOT/'tools/contribution_study/build_arcquant.py'),
            '--sources',str(local_lock),'--output',str(args.arc_build),'--pybind11',str(pybind)], check=True)
        subprocess.run([sys.executable,str(ROOT/'tools/contribution_study/build_sharq.py'),
            '--sources',str(local_lock),'--pybind11',str(pybind)], check=True)


if __name__ == '__main__':main()
