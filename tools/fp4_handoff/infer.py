"""Compatible handoff CLI backed by the common fresh-seed quality worker."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--method', required=True)
    p.add_argument('--cases', nargs='+')
    p.add_argument('--require-source-hash', action='store_true')
    a = p.parse_args()
    from dataset_infer import run
    run(a.study/'manifest.json', a.method, a.study/'generation'/a.method,
        cases=a.cases, require_source_hash=a.require_source_hash)


if __name__ == '__main__':
    main()
