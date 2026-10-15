"""Copy a runtime, then add I2V trimming/coverage without modifying frozen sources."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def change(text, before, after):
    if text.count(before) != 1: raise ValueError('Parent source mismatch: ' + before[:100])
    return text.replace(before, after)


def build(parent, output):
    package = parent / 'thunderworld_fp4'
    receipt = parent / 'variant.json'
    old = json.loads(receipt.read_text()) if receipt.exists() else {}
    for name, digest in old.get('variant_hashes', {}).items():
        if sha(package/name) != digest: raise ValueError('Changed parent: ' + name)
    if old.get('library_sha256') and sha(parent/'cutlass_sm110.so') != old['library_sha256']:
        raise ValueError('Changed parent binary')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(package, output/'thunderworld_fp4', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in parent.glob('cutlass_sm110*'):
        if path.is_file(): shutil.copy2(path, output/path.name)
    here = Path(__file__).resolve().parent
    shutil.copy2(here/'expanded_runtime.py', output/'thunderworld_fp4/_expanded_runtime.py')
    runner = output/'thunderworld_fp4/cosmos/runner.py'
    s = runner.read_text()
    s = change(s, 'qkv_streams=False):', 'qkv_streams=False,expansion=None):')
    s = change(s, '        self.path=Path(model);', '        self.expansion=expansion;self.expansion_report=None\n        self.path=Path(model);')
    s = change(s, "        self.pipe.to('cuda')\n", '')
    s = change(s, '        self.engine=JointEngine(self.pipe.transformer,self.config,self.table,assets)\n',
        '''        from .._expanded_runtime import install, audit_protected
        self.expansion_report=install(self)
        self.engine=JointEngine(self.pipe.transformer,self.config,self.table,assets)
        self.pipe.to('cuda')
        audit_protected(self)
''')
    s = change(s, '    def _context(self,case,steps,audio_output):\n', '''    def _context(self,case,steps,audio_output):
        if self.expansion is not None and case['task']!='i2v':
            raise ValueError('Expanded runtime is I2V-only; use an untrimmed runner for other tasks')
''')
    s = change(s, 'return dict(config=self.config.to_dict(),model_identity=self.identity,',
        'return dict(expansion=self.expansion_report,config=self.config.to_dict(),model_identity=self.identity,')
    runner.write_text(s)
    hashes = {str(p.relative_to(output/'thunderworld_fp4')):sha(p)
              for p in sorted((output/'thunderworld_fp4').rglob('*.py'))}
    value = dict(schema=1, parent=str(parent), parent_receipt_sha256=sha(receipt) if receipt.exists() else None,
        parent_sources={str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*.py'))},
        variant_hashes=hashes, qualified=False, scope='I2V trimming and expanded quantization experiment',
        library_sha256=old.get('library_sha256'), builder_sha256=sha(__file__))
    (output/'variant.json').write_text(json.dumps(value,indent=2)+'\n')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.parent,a.output)
    print(a.output)
