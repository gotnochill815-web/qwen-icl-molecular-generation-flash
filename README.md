# Qwen3.8-Flash-Next: Zero-Shot Molecular Generation

This repository implements a zero-shot molecular generation pipeline using **Qwen3.8-Flash-Next**. The model is prompted to generate SMILES strings satisfying specified molecular property targets. Generated molecules are evaluated using RDKit and the resulting metrics are aggregated across independent experiments.

The experiments cover four molecular properties and selected combinations:

- LogP
- TPSA
- SAS
- QED
- TPSA + SAS
- SAS + LogP
- LogP + SAS + TPSA
- TPSA + LogP

Each experiment has its own configuration, manifest, generation output, smoke-test output, and evaluation results.

## 1. Repository Structure

```text
qwen-icl-molecular-generation-flash/
│
├── configs/
│   ├── logp_zero_shot.json
│   ├── tpsa_zero_shot.json
│   ├── sas_zero_shot.json
│   ├── qed_zero_shot.json
│   ├── tpsa_sas_zero_shot.json
│   ├── sas_logp_zero_shot.json
│   ├── logp_sas_tpsa_zero_shot.json
│   └── tpsa_logp_zero_shot.json
│
├── src/
│   ├── common.py
│   ├── prepare_dataset.py
│   ├── build_manifest.py
│   ├── generate_flash.py
│   ├── generate_flash_vllm.py
│   ├── smoke_test.py
│   ├── evaluate_flash.py
│   ├── evaluate_chemfm_property.py
│   ├── evaluate_chemfm_multi.py
│   └── aggregate_flash_results.py
│
├── requirements.txt
├── requirements-vllm.txt
├── requirements-test-py39.txt
├── .gitignore
└── README.md
```

## 2. Experimental Design

All experiments use **zero-shot generation**, with no molecular examples supplied in the generation prompt.

| Setting | Value |
|---|---:|
| Model | Qwen3.8-Flash-Next |
| Number of experiments | 8 |
| Shots | 0 |
| Targets per experiment | 4 |
| Generations per target | 25 |
| Total generations per experiment | 100 |
| Total planned generations | 800 |
| Temperature | 0.7 |
| Random seed | 42 |

Each configuration defines one independent experiment. This keeps manifests, raw outputs, and evaluation results separate and makes it easier to compare property conditions.

### Property evaluation

Molecular properties are computed from generated SMILES using RDKit:

- **LogP:** Wildman-Crippen molecular LogP.
- **TPSA:** Topological polar surface area.
- **SAS:** Synthetic accessibility score.
- **QED:** Quantitative estimate of drug-likeness.

For multi-property experiments, generated molecules are evaluated against all requested properties.

## 3. Requirements

The main environment uses Python 3.11 or later.

Install the project dependencies:

```bash
pip install -r requirements.txt
```

PyTorch should be installed separately with a CUDA-compatible build matching the available GPU environment.

For vLLM-based generation:

```bash
pip install -r requirements-vllm.txt
```

The vLLM configuration requires **vLLM 0.30.0 or later**.

The file `requirements-test-py39.txt` records the separate older Python 3.9 testing environment.

## 4. Dataset

The manifest builder uses a molecular dataset containing SMILES strings and computed molecular properties.

The input CSV should contain these columns:

```text
smiles
logp
tpsa
sas
qed
```

Update the dataset path in each experiment configuration before running the pipeline.

The dataset is used to construct target property conditions. For single-property experiments, target values are selected using quantiles. For multi-property experiments, combinations of property values are sampled using the configured random seed.

## 5. Configuration

Each experiment is configured independently under `configs/`.

For example:

```text
configs/logp_zero_shot.json
```

The configuration specifies:

- Experiment name and requested properties.
- Dataset and validation data paths.
- Number of targets and generations.
- Sampling parameters.
- Model identifiers.
- Manifest and job-list paths.
- Raw generation output paths.
- Smoke-test output paths.
- Evaluation and aggregation directories.
- vLLM runtime settings.

The eight configuration files are intended to be run independently. Avoid pointing multiple experiments to the same output or manifest paths.

## 6. Build Experiment Manifests

Build the target conditions and generation jobs for all experiments:

```bash
for config in configs/*.json; do
    python src/build_manifest.py --config "$config"
done
```

Each experiment produces four target conditions and 100 generation jobs.

To build only one experiment:

```bash
python src/build_manifest.py --config configs/logp_zero_shot.json
```

## 7. Run Molecular Generation

The repository provides two generation backends:

- `generate_flash.py`: Transformers-based generation.
- `generate_flash_vllm.py`: vLLM-based generation.

For vLLM generation, run:

```bash
python src/generate_flash_vllm.py --config configs/logp_zero_shot.json
```

To run all eight experiments sequentially:

```bash
for config in configs/*.json; do
    python src/generate_flash_vllm.py --config "$config"
done
```

The generation script uses the experiment manifest and records generated outputs with their corresponding job IDs and molecular properties.

Generation can be resumed: jobs already present in the output are skipped.

The vLLM runner performs preflight checks before model initialization. These include:

- vLLM version.
- CUDA availability.
- Number of visible GPUs relative to tensor parallelism.
- Available GPU memory across the selected tensor-parallel GPUs.
- Available host memory when PLE CPU offloading is enabled.
- Validity of engine and batch configuration values.

The default vLLM checkpoint is:

```text
Inferact/Qwen3.8-Flash-Next-NVFP4
```

The configuration also specifies the model's PLE offloading settings.

**Hardware note:** The default NVFP4 configuration specifies a minimum GPU memory requirement of 130 GiB and at least 51 GiB of available host memory for PLE CPU offloading. Additional runtime headroom is necessary. Check the requirements against the actual GPU and host environment before launching inference. An 80 GB A100 does not satisfy the configured 130 GiB GPU-memory requirement.

## 8. Smoke Testing

Before running actual model inference, run the smoke test for each experiment:

```bash
for config in configs/*.json; do
    python src/smoke_test.py --config "$config"
done
```

The smoke test uses predefined molecular examples rather than loading the language model. It checks that the experiment pipeline can:

1. Read the experiment configuration and manifest.
2. Create generation records.
3. Parse and validate SMILES using RDKit.
4. Store output in the experiment-specific smoke-test location.

Smoke-test records can then be passed to `evaluate_flash.py` using the corresponding configured smoke-output path. Set `--output-root` to the experiment's configured evaluation directory so that the aggregation script can find the resulting summary.

## 9. Evaluate Generated Molecules

Evaluate the actual generation outputs:

```bash
for config in configs/*.json; do
    python src/evaluate_flash.py --config "$config"
done
```

Evaluation checks generated records against the corresponding manifest and computes molecular validity and property-based evaluation metrics.

The evaluation code uses separate evaluators for single-property and supported multi-property experiments.

Evaluation results are written to experiment-specific directories.

## 10. Aggregate Results

Aggregate the evaluation summaries for each experiment:

```bash
for config in configs/*.json; do
    python src/aggregate_flash_results.py --config "$config"
done
```

The aggregation script checks the number of completed generations against the expected number of jobs.

An experiment is marked as **partial** when fewer than the expected number of generations have been evaluated. This is expected during smoke testing, where only a small subset of jobs is processed.

## 11. Validation Status

The repository's eight experiment pipelines have been smoke-tested.

| Validation | Result |
|---|---|
| Experiment configurations | 8 checked |
| Manifest generation | Passed for all 8 |
| Smoke-test pipelines | Passed for all 8 |
| Stub molecules evaluated | 16 |
| RDKit SMILES validation | 16/16 valid |
| Evaluation and aggregation | Passed for all 8 |
| Python compilation | Passed |

These tests validate the experiment orchestration, record format, SMILES handling, evaluation, and aggregation logic.

**Important:** Smoke tests use predefined molecules. They do not establish that the Qwen model has successfully completed actual inference. Model-generation results must be validated separately.

## 12. References

- [Qwen3.8-Flash-Next](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4)
- [vLLM](https://github.com/vllm-project/vllm)
- [RDKit](https://www.rdkit.org/)
- [GuacaMol](https://github.com/BenevolentAI/guacamol)

---

**Project:** Zero-Shot In-Context Molecular Generation  
**Model:** Qwen3.8-Flash-Next  
**Generation framework:** Transformers / vLLM  
**Molecular evaluation:** RDKit
