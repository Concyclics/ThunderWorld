import sys
from pathlib import Path
from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device

with shared_worker():
    print(select_shared_device(),flush=True)
    import torch,pytest
    torch.set_num_threads(8)
    raise SystemExit(pytest.main([str(Path(__file__).with_name('test_bridge.py')),'-q',*sys.argv[1:]]))
