# Third-party materials

The [Apache-2.0 license](LICENSE) covers ThunderWorld project code. It does not relicense dependencies, datasets, or model weights. Preserve existing copyright headers when modifying kernels.

| Material | Source and terms |
|---|---|
| Paper artwork in `docs/assets/` | Han Chen et al., [ThunderWorld](https://www.preprints.org/manuscript/202610.0689), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original manuscript figures, unmodified; hashes in `docs/assets/provenance.json`. |
| Cosmos3 and derived FP4 weights | [NVIDIA Cosmos3](https://huggingface.co/nvidia), OpenMDW-1.1. Each released model includes its own license and notice. Weights are downloaded separately. |
| CUTLASS / CuTe | [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass), BSD-3-Clause. Build against the pinned revision specified by the relevant build receipt. |
| Triton | [triton-lang/triton](https://github.com/triton-lang/triton), MIT. |
| PyTorch | [pytorch/pytorch](https://github.com/pytorch/pytorch), BSD-style license and bundled third-party notices. |
| Diffusers / Transformers | [Hugging Face](https://github.com/huggingface), Apache-2.0. |
| Baseline sources | See `configs/public/baselines.lock.json`. The collector records upstream license files and hashes; check each repository's license before redistribution. We fetch upstream code into an ignored directory, retaining its notices. |
| RBench inputs | [DAGroup-PKU/RBench](https://huggingface.co/datasets/DAGroup-PKU/RBench). Download separately under the dataset's terms. |
| PAI-Bench-G inputs | [shi-labs/physical-ai-bench-generation](https://huggingface.co/datasets/shi-labs/physical-ai-bench-generation). Download separately under the dataset's terms. |
| Benchmark prompt lists | [NVIDIA/cosmos](https://github.com/NVIDIA/cosmos), fixed revision in `tools/public/data.py`; retain upstream notices. |

The repository contains calibration identifiers and results, not copies of the full source datasets or original checkpoints. A baseline source lock is a provenance record, not a blanket license grant.
