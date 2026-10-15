"""Contribution experiment entry point; no simulated timings enter performance tables."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from baselines import SOURCES, collect, sha, write

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='operation',required=True)
    b = sub.add_parser('collect-baselines')
    b.add_argument('--output',type=Path,required=True)
    b.add_argument('--names',nargs='+',choices=list(SOURCES),default=list(SOURCES))
    b.add_argument('--lock',type=Path,help='Fetch the exact previously recorded official commits')
    m = sub.add_parser('mechanism')
    for key in ('capture','assets','output'):
        m.add_argument('--'+key,type=Path,required=True)
    m.add_argument('--modules',nargs='+')
    m.add_argument('--resume',action='store_true')
    m.add_argument('--threshold',type=float,choices=(2.,3.),default=2.)
    for op, script in [('capture','capture_budget.py'),('search','search_budget.py'),
                       ('evaluate','run_compute_cohort.py'),('paper','paper_study.py')]:
        child = sub.add_parser(op,add_help=False)
        child.set_defaults(script=script)
        child.add_argument('args',nargs=argparse.REMAINDER)
    r=sub.add_parser('report');source=r.add_mutually_exclusive_group(required=True)
    source.add_argument('--study',type=Path);source.add_argument('--workspace',type=Path)
    r.add_argument('--output',type=Path)
    f=sub.add_parser('freeze-cohort')
    for key in ('cohort','history','output'):f.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    if args.operation=='collect-baselines':
        collect(args.output,args.names,args.lock)
    elif args.operation=='mechanism':
        from mechanism import run
        run(args)
    elif args.operation=='report':
        if args.workspace:
            if args.output is None:p.error('--workspace report requires --output')
            subprocess.run([sys.executable,str(Path(__file__).with_name('archive_report.py')),
                '--workspace',str(args.workspace),'--output',str(args.output)],check=True)
        else:
            from report import report
            report(args.study)
    elif args.operation=='freeze-cohort':
        from protocol import freeze_cohort
        freeze_cohort(args.cohort,args.history,args.output)
    else:
        forwarded=args.args[1:] if args.args[:1]==['--'] else args.args
        script=ROOT/'tools/joint_inference'/args.script
        if args.operation=='paper':script=Path(__file__).with_name('paper_study.py')
        if args.operation=='evaluate' and forwarded[:1]==['external']:
            script=Path(__file__).with_name('study_workflow.py');forwarded=forwarded[1:]
        elif args.operation=='search' and forwarded[:1]==['head-coordinate']:
            script=Path(__file__).with_name('head_coordinate.py');forwarded=forwarded[1:]
        subprocess.run([sys.executable,str(script),*forwarded],check=True)


if __name__=='__main__':main()
