"""Static scientific comparison figures from measured times and decoded frames."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    d = json.loads(a.result.read_text())
    assert d['complete'] and d['args']['steps']==35
    modes = ['bf16', 'ffn_optimized', 'joint_optimized']
    labels = ['BF16', 'Optimized FP4 FFN', 'Optimized FP4 FFN + attention']
    seed = d['args']['seed']
    videos = [np.load(a.result.parent/f'{mode}_seed{seed}.npy', mmap_mode='r') for mode in modes]
    indices = np.linspace(0, len(videos[0])-1, 4).astype(int)
    fig, axes = plt.subplots(3, 4, figsize=(12, 7.6), constrained_layout=True)
    for row, (video, label) in enumerate(zip(videos, labels)):
        for col, frame in enumerate(indices):
            ax = axes[row, col]
            ax.imshow(video[frame], vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if row==0: ax.set_title(f'Frame {frame+1}')
            if col==0: ax.set_ylabel(label, fontsize=9)
    fig.suptitle(f'{a.label}: same input and seed {seed}, 35 steps (256p pilot)', fontsize=14)
    a.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output/'frames.png', dpi=150)
    plt.close(fig)

    values = [d['summary'][m]['wall_median_seconds'] for m in d['args']['modes']]
    low = [v-d['summary'][m]['wall_min_seconds'] for m,v in zip(d['args']['modes'], values)]
    high = [d['summary'][m]['wall_max_seconds']-v for m,v in zip(d['args']['modes'], values)]
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    name_map = {'bf16':'BF16', 'ffn_original':'FP4 FFN\noriginal', 'ffn_optimized':'FP4 FFN\noptimized',
                'joint_original':'FP4 joint\noriginal', 'joint_optimized':'FP4 joint\noptimized'}
    names = [name_map[m] for m in d['args']['modes']]
    bars = ax.bar(names, values, yerr=[low, high], capsize=4, color=['#65758B','#8FBFA8','#26785D','#A0B8D8','#315B91'])
    ax.bar_label(bars, labels=[f'{v:.3f} s' for v in values], padding=8)
    ax.set_ylim(0, max(values)*1.25)
    ax.set_ylabel('Full Generator latency (seconds)')
    ax.set_title(f'{a.label} — median and range of 3 requests; shared CUDA Graph' if d['args'].get('cuda_graph') else a.label)
    fig.savefig(a.output/'latency.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
