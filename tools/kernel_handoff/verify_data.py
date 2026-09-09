"""CPU-only checksum verification for the transferred workload bundle."""
import argparse,hashlib,json
from pathlib import Path

def verify(root):
    root=Path(root).resolve();m=json.loads((root/'manifest.json').read_text())
    if m.get('schema_version')!=1 or not m.get('complete'):raise ValueError('Incomplete or unsupported bundle')
    for name,entry in m['files'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root):raise ValueError('Manifest path escapes bundle')
        if p.stat().st_size!=entry['bytes']:raise ValueError(f'Size mismatch: {name}')
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(8<<20),b''):h.update(block)
        if h.hexdigest()!=entry['sha256']:raise ValueError(f'SHA256 mismatch: {name}')
    return m
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);a=p.parse_args();m=verify(a.data)
    print(f"Verified {len(m['files'])} files; {m['total_bytes']/2**20:.2f} MiB")
