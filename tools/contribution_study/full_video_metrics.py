"""All-frame BF16 consistency metrics, with reusable reference features.

Definitions match cosmos/quality.py: AlexNet LPIPS 0.1, skimage's default
7x7/sample-covariance RGB SSIM, and clip-global float64 MSE/PSNR (range=1).
The GPU SSIM implementation must pass the independent qualification command
before a cohort can run. There is no resizing, frame sampling, or AMP.
"""
import hashlib
import io
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_video(record, shape=(61, 192, 320, 3)):
    """Read each archive once; verify both its file and decoded signal hashes."""
    data = Path(record['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != record['npz_sha256']:
        raise ValueError('NPZ hash mismatch: ' + record['path'])
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if archive.files != ['video']:
            raise ValueError('Expected video-only I2V output')
        video = archive['video']
    if video.dtype != np.float32 or video.shape != tuple(shape):
        raise ValueError(f'Unexpected video dtype/shape: {video.dtype}, {video.shape}')
    video = np.ascontiguousarray(video)
    if hashlib.sha256(memoryview(video).cast('B')).hexdigest() != record['video_sha256']:
        raise ValueError('Decoded video hash mismatch: ' + record['path'])
    validate_video(video)
    return video


def validate_video(video):
    if video.dtype != np.float32 or video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError('Metrics require float32 THWC RGB input')
    if min(video.shape[:3]) < 1 or min(video.shape[1:3]) < 7:
        raise ValueError('Empty or undersized video')
    if not np.isfinite(video).all() or video.min() < 0 or video.max() > 1:
        raise ValueError('Metrics require finite [0,1] video; no normalization allowed')


def uniform7(x):
    # scipy.ndimage.uniform_filter filters axis 0 and then axis 1, using
    # double sums and float32 intermediate outputs. Valid windows exactly
    # correspond to skimage's three-pixel crop, avoiding border padding.
    y = F.avg_pool2d(x.double(), (7, 1), stride=1).float()
    return F.avg_pool2d(y.double(), (1, 7), stride=1).float()


class VideoMetrics:
    def __init__(self, device='cuda:0', threads=2):
        import lpips
        torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.hub.set_dir(os.environ.get('TW_LPIPS_CACHE', '/tmp/thunderworld-lpips-cache'))
        self.device = torch.device(device)
        self.model = lpips.LPIPS(net='alex', version='0.1').eval().to(self.device)
        self.lpips_module = lpips
        self.batch_size = 8
        self.reference = None
        self.reference64 = None
        self.reference_norm = None
        self.cache = []

    def tensor(self, array):
        return torch.from_numpy(array).permute(0, 3, 1, 2).float().to(self.device)

    def features(self, value):
        value = self.model.scaling_layer(value * 2 - 1)
        return [self.lpips_module.normalize_tensor(t) for t in self.model.net(value)]

    @torch.inference_mode()
    def set_reference(self, reference):
        validate_video(reference)
        self.cache = []
        self.reference = reference
        self.reference64 = np.asarray(reference, np.float64)
        self.reference_norm = max(float(np.linalg.norm(self.reference64)), 1e-12)
        for start in range(0, len(reference), self.batch_size):
            y = self.tensor(reference[start:start+self.batch_size])
            uy = uniform7(y)
            vy = (49 / 48) * (uniform7(y * y) - uy * uy)
            self.cache.append((y, uy, vy, self.features(y)))

    @torch.inference_mode()
    def compare(self, candidate):
        validate_video(candidate)
        if self.reference is None or candidate.shape != self.reference.shape:
            raise ValueError('Missing or mismatched reference')
        delta = np.asarray(candidate, np.float64) - self.reference64
        mse = float(np.mean(np.square(delta)))
        relative_l2 = float(np.linalg.norm(delta) / self.reference_norm)
        del delta
        lpips_frames, ssim_frames = [], []
        for block, start in enumerate(range(0, len(candidate), self.batch_size)):
            x = self.tensor(candidate[start:start+self.batch_size])
            y, uy, vy, ref_features = self.cache[block]
            ux = uniform7(x)
            vx = (49 / 48) * (uniform7(x * x) - ux * ux)
            vxy = (49 / 48) * (uniform7(x * y) - ux * uy)
            a1, a2 = 2 * ux * uy + 0.0001, 2 * vxy + 0.0009
            b1, b2 = ux ** 2 + uy ** 2 + 0.0001, vx + vy + 0.0009
            s = (a1 * a2) / (b1 * b2)
            # skimage accumulates spatially in float64, then casts each RGB
            # channel mean to float32 before averaging channels and frames.
            channel_means = s.double().mean(dim=(-2, -1)).cpu().numpy().astype(np.float32)
            ssim_frames.extend(channel_means.mean(axis=1).tolist())
            features = self.features(x)
            value = 0
            for layer, (feature, ref_feature) in enumerate(zip(features, ref_features)):
                diff = (feature - ref_feature) ** 2
                value = value + self.lpips_module.spatial_average(self.model.lins[layer](diff), keepdim=True)
            lpips_frames.extend(value.flatten().cpu().tolist())
        metrics = dict(mse=mse, rmse=float(np.sqrt(mse)), relative_l2=relative_l2,
            psnr_db=float(-10 * np.log10(mse)) if mse else None, psnr_exact=mse == 0,
            lpips=float(np.mean(lpips_frames)),
            ssim_mean=float(np.mean(np.asarray(ssim_frames, dtype=np.float32))))
        if not all(np.isfinite(v) for v in metrics.values() if v is not None):
            raise ValueError('Non-finite metric')
        return dict(video=metrics, per_frame=dict(lpips=lpips_frames, ssim=ssim_frames))


def environment():
    import importlib.metadata
    import lpips
    from baselines import sha
    weights = Path(lpips.__file__).parent / 'weights/v0.1/alex.pth'
    alex = Path(os.environ.get('TW_LPIPS_CACHE', '/tmp/thunderworld-lpips-cache')) / 'checkpoints/alexnet-owt-7be5be79.pth'
    return dict(versions={k: importlib.metadata.version(k)
                         for k in ('numpy', 'torch', 'torchvision', 'lpips', 'scikit-image', 'scipy')},
        weights={str(weights): sha(weights), str(alex): sha(alex)},
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), batch_size=8,
        amp=False, tf32=False, cudnn_deterministic=True, cudnn_benchmark=False)
