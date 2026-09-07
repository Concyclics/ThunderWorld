# FFN activation-channel and weight-block repair

This experiment compares actual K-channel block support with a small offline
weight residual bank. It leaves `thunderworld_fp4.FP4Linear` and production
model defaults unchanged.

- `study.py`: AV-calibrated static supports and robot local linear replay;
  activation energy vs residual-weighted selection, K16/64/128/256, and equal
  total A/W repair budgets. K16 is a numerical comparison, not a native path.
- `native.py`: experimental SM120 backend. Full dense W4 base plus selected
  A1W0 and A0W1 in one native FP4 GEMM/FP32 accumulator; optional A1W1 cross term.
  W1 N128/K tiles are packed offline into CSR storage. Activations are current.
- `native_v1.py`: initial separate-index-loop implementation, preserved.
- `benchmark.py`: initial M32 operator comparison with activation packing.
- `tune.py`: compares interleaved W0 reuse, K128/256 and M32/64/128 execution.
- `ffn_chain.py`: full local SiLU(gate)*up -> down with candidate intermediate
  activations, referenced to current BF16 FFN operators.
- `smoke.py`: tail rows, cross terms, graph replay, immutable W0/W1 checks.

Run sequentially from the ThunderWorld repository with the Cosmos3 environment:

```bash
CUDA_VISIBLE_DEVICES=3 python experiments/ffn_block_repair_20260907/study.py
CUDA_VISIBLE_DEVICES=3 python experiments/ffn_block_repair_20260907/benchmark.py
CUDA_VISIBLE_DEVICES=3 python experiments/ffn_block_repair_20260907/tune.py
CUDA_VISIBLE_DEVICES=3 python experiments/ffn_block_repair_20260907/ffn_chain.py
CUDA_VISIBLE_DEVICES=3 compute-sanitizer --tool memcheck --error-exitcode 1 \
  python experiments/ffn_block_repair_20260907/smoke.py
CUDA_VISIBLE_DEVICES=3 compute-sanitizer --tool racecheck --error-exitcode 1 \
  python experiments/ffn_block_repair_20260907/smoke.py
```

The scripts use existing local model/capture assets and output under
`results/f4/ffn_block_repair_20260907`. They do not download assets. Performance
measurements must run sequentially; this run used shared physical GPU 3.

`Native(wc, ws, wg, d, am, wm, b=128, cross=False, bm=32, interleave=True)`
expects both full quantized planes as offline construction inputs, a boolean
activation support `[ceil(M/128),K]` with equal selected block counts per row,
and a boolean weight support `[N,K]`. Inputs must already be block-constant at
K=`b` and N128. Constructor clones W0 and gathers only selected W1 tiles; release
the original full preparation planes for standalone deployment. The object's
`wr`, `wrs`, `wi`, `off` contain only the sparse residual data and metadata.
Runtime accepts contiguous BF16 `[M,K]` for the prepared M128 support, on SM120.

W0 and W1 use the same outer factor; this allows accumulation before one output
scaling operation. W0 uses local pinned BF16 weights, community AWQ prescale,
and RTN packing. It is **not** the community checkpoint's optimized packed W4.
No online weight quantization, tensor decoding, training, 2:4 sparse MMA, or
cross-step activation/output cache is involved. Producer kernels remain
separate from GEMM. Partial A repair still reserves full two-plane activation
scratch; the memory reduction here concerns W1 storage, not A scratch capacity.

The current prototype uses one shared K block size for A and W repair; budgets
are independent. Frozen support is a starting point: layer/phase-specific
calibration can supply different supports, but this experiment did not fit an
adaptive cross-step controller. See the linked report for accuracy and speed
limits before integrating this prototype.

[Chinese report](../../docs/FFN_BLOCK_REPAIR_20260907.md)
