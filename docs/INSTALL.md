# Installation

Use the repository root as the installation source. The former package-only installation remains available for kernel developers, but the root also exposes `tw-infer` and the release registry.

## Qualified inference environment

- Linux x86-64, Python 3.12.
- PyTorch 2.11.0 with CUDA 13.0 wheels, Triton 3.6.0.
- Diffusers revision `77c13f8a49c457f10a89862bb129bdc930b88929` with Cosmos3 support. An arbitrary older PyPI release is insufficient.
- NVIDIA SM120: RTX PRO 6000 Blackwell / RTX 5090. This describes native instruction support, not guaranteed model fit.
- An NVIDIA driver compatible with the selected CUDA wheel. CUTLASS builds additionally require a matching CUDA toolkit / C++ compiler.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-runtime.txt
python -m pip install -e '.[hub]'
tw-infer --doctor --check-gpu
```

Authenticate with `hf auth login` if the pinned NVIDIA base requires account access. Never put a token in a script or commit. The model launcher downloads the original base and one selected profile; `--base-model` and `--release-dir` reuse local copies. The base models remain governed by their upstream license.

## CPU-only inspection

```bash
python -m pip install -e '.[hub,test]'
tw-infer --list-models
tw-infer --doctor
tw-infer --model edge --profile 2x --download-only
tw-infer --model edge --profile 2x --verify-only
```

These operations do not initialize CUDA. `--verify-only` reads and hashes the selected packed assets without downloading the original base. `--local-files-only` requires cached releases; for inference it also prevents the child loader from downloading the base.

## Calibration and evaluation extras

```bash
python -m pip install -r requirements-evaluation.txt
```

The **FP8 baseline** needs the native vLLM `scaled_fp8_quant` and `cutlass_scaled_mm` extensions. The qualified environment used **vLLM 0.25.1** with the same PyTorch/CUDA ABI. Install or build a compatible vLLM wheel in a separate evaluation environment if its dependencies conflict; do not silently downgrade the inference environment or replace native FP8 with fake quantization.

ARCQuant and SharQ use their pinned official CUDA builds. AWQ calibration needs the pinned NVIDIA Model Optimizer source; [baseline instructions](BASELINES.md) explain collection and build steps. vLLM and ModelOpt are not prerequisites for downloading or running the released ThunderWorld profiles.

## Device limits

The released runtime targets SM120. Super places DM on logical GPU 0 and unchanged BF16 AR on logical GPU 1; its preparation admission thresholds are 88 GiB and 72 GiB of free memory. A pair of RTX 5090s is not a qualified Super setup. Edge/Nano also materialize the original base during preparation; inspect the actual free-memory requirement rather than treating the packed asset size as total VRAM.

The paper's SM110 Jetson AGX Thor port has a separate compiler and attention implementation under `tools/thor_handoff/`. It is not selected by the Hub launcher, and SM120 binaries must not be copied to Thor as if they were native builds.

## Tests

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/public packages/fp4-repair/tests
```

CPU runs skip CUDA tests. On a dedicated SM120 device, select it explicitly and run the relevant CUDA tests and Compute Sanitizer qualification described in [PERFORMANCE.md](PERFORMANCE.md). A CPU pass is not evidence of kernel speed or full-trajectory quality.
