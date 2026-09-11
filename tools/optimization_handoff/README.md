# Optimized runtime handoff tools

Read [the canonical handoff](../../docs/OPTIMIZATION_HANDOFF.md) for budget
semantics, model setup, integration limits and receiver experiments.
Its setup includes exporting the immutable `9b51f0d` baseline from Git history;
the original-vs-optimized adapter tests require that source snapshot.

| File | Purpose |
|---|---|
| `profiles/edge.json`, `profiles/nano.json` | Compact runtime schedules and launch choices; no raw model data |
| `export_profiles.py` | Deterministically rebuild exports from full archived policies |
| `prepare_inputs.py` | Fetch checksum-pinned public inputs and reconstruct the table image; `--check-only` verifies local assets |
| `retime.py` | Frozen-choice A/W retiming; requires local real-weight payloads, CUDA and SM120 |
| `audit.py` | CPU aggregation of fresh retiming and archived stage comparisons |
| `verify_release.py` | Verify runtime profile equivalence, release counts and canonical document links without loading weights |

Run from the repository root:

```bash
python tools/optimization_handoff/verify_release.py
python tools/optimization_handoff/export_profiles.py
bash tools/kernel_handoff/python5090.sh tools/optimization_handoff/prepare_inputs.py --check-only
bash tools/kernel_handoff/python5090.sh tools/optimization_handoff/retime.py \
  --payload-root runs5090 --output runs5090/receiver_retime.json
```

`retime.py` expects the historical shape payloads in
`PAYLOAD_ROOT/auto_budget_{edge,nano}_bench`. It repeats the M316 probe for M416
when needed. On a new capture distribution, first regenerate the shape benchmark
with the calibration tools; missing historical shapes require explicit probe
preparation rather than silently treating different inputs as a reproduction.

The shipped audit is in `results/optimization_handoff_20260911`. It preserves
sample timings, local output errors and source hashes. Do not overwrite it for
a receiver experiment. `audit.py --help` shows how to select another input/output.
`retime.json` is not a policy file: the experimental trajectory wrappers expect
their own benchmark schemas including selected launch configurations.

`verification.json` and `smoke_{edge,nano}.json` record release checks. The smoke
runs contain only two denoising steps and are not quality/speed claims. Raw output
arrays and model payloads remain receiver-local. The full historical scientific
reports and source snapshots remain under `results/` and `reports/`.
