"""Prepare the two public NVIDIA example images used by the trajectory runner.

Only fixed public assets are downloaded. ZIP members are read by name without
extracting paths. Existing files are validated and never silently replaced.
"""
from pathlib import Path
import argparse,hashlib,io,json,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[2]
BASE='https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/'
ASSETS=[('data_local/e2e_inputs/robot_153.jpg','robot_153.jpg','886e6cfa8a0d6c4d96459cfe79b51c56aff4d63e1c523f8366f1771ea9c2788d'),('data_local/finding_inputs/robot_example.zip','robot_example.zip','61279fcfb58827bc84957a4b4f2dd810a66331b6253ba259cff516754f36ede6')]
TABLE='3f3e3b7d84d4e3f70aade11bc430edf3bfab07aa36a22dc4c5dd37761eaa81bc'
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args();receipts=[]
 for relative,name,expected in ASSETS:
  path=ROOT/relative
  if not path.exists():
   if a.check_only:raise FileNotFoundError(path)
   with urllib.request.urlopen(BASE+name,timeout=120) as response:body=response.read()
   if sha(body)!=expected:raise ValueError('Public asset checksum mismatch: '+name)
   path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
  if sha(path.read_bytes())!=expected:raise ValueError('Existing file checksum mismatch: '+relative)
  receipts.append(dict(path=relative,url=BASE+name,sha256=expected))
 target=ROOT/'data_local/finding_inputs/robot_table.png'
 if not target.exists():
  if a.check_only:raise FileNotFoundError(target)
  import av
  archive=ROOT/ASSETS[1][0]
  with zipfile.ZipFile(archive) as z:
   with z.open('robot_example/robot_input.mp4') as f:
    with av.open(io.BytesIO(f.read())) as video:frame=next(video.decode(video=0)).to_image()
  frame.save(target)
 actual=sha(target.read_bytes())
 if actual!=TABLE:raise ValueError('Decoded PNG differs from the archived image hash; inspect decoder/Pillow versions before claiming an identical-input reproduction')
 receipts.append(dict(path=str(target.relative_to(ROOT)),sha256=actual,source='First decoded frame of robot_example/robot_input.mp4'))
 print(json.dumps(dict(passed=True,receipts=receipts),indent=2))
if __name__=='__main__':main()
