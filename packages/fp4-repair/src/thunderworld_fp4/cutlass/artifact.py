"""CPU-only offline artifact validation. A binary never selects a device profile."""
import hashlib
import json
from pathlib import Path
from .build import source_hashes


def verify_library(path):
    path=Path(path).resolve()
    meta=json.loads(path.with_suffix('.build.json').read_text())
    if meta.get('schema_version')!=1 or meta.get('backend')!='cutlass-fused-sm120':
        raise ValueError('Unsupported CUTLASS build manifest')
    if meta.get('tile') not in ([128,128,128],[128,64,128]):raise ValueError('Unsupported CUTLASS tile')
    if meta.get('sources')!=source_hashes():raise ValueError('Stale CUTLASS source/header build')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if meta.get('library_sha256')!=digest:raise ValueError('CUTLASS library digest mismatch')
    return dict(backend=meta['backend'],library_sha256=digest,sources=meta['sources'],tile=meta['tile'])
