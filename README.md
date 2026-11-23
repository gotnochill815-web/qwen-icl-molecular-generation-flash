# Zero-Shot Molecular Generation with Qwen3.8-Flash-Next

A configurable pipeline for zero-shot molecular generation conditioned on
molecular properties using the Hugging Face checkpoint
`Qwen/Qwen3.8-Flash-Next`.

The repository supports a Transformers generation backend and an optional
vLLM backend, with RDKit-based molecular validation and evaluation.

**Scope:** Zero-shot molecular generation only (`shots = 0`).

## Experiments

Eight experiments are configured:

| Experiment | Conditioning properties | Jobs |
|---|---|---:|
| logp | LogP | 100 |
| tpsa | TPSA | 100 |
| sas | SAS | 100 |
| qed | QED | 100 |
| tpsa_sas | TPSA + SAS | 100 |
| sas_logp | SAS + LogP | 100 |
| logp_sas_tpsa | LogP + SAS + TPSA | 100 |
| tpsa_logp | TPSA + LogP | 100 |
| **Total** | | **800** |

Each experiment uses four targets, one replicate per target, and 25
generations per target.

The generation configuration uses:
- Temperature: 0.7
- Top-p: 0.95
- Maximum new tokens: 128
- Repetition penalty: 1.0
- Sampling enabled
- Random seed: 42

## Repository structure

```text
.
├── configs/
│   └── flash_zero_shot.json
├── src/
│   ├── common.py
│   ├── build_manifest.py
│   ├── prepare_dataset.py
│   ├── smoke_test.py
│   ├── generate_flash.py
│   ├── generate_flash_vllm.py
│   ├── evaluate_flash.py
│   ├── evaluate_chemfm_property.py
│   ├── evaluate_chemfm_multi.py
│   └── aggregate_flash_results.py
├── requirements.txt
├── README.md
└── .gitignore
```

Generated datasets, manifests, model weights, logs, and experiment outputs
are intentionally excluded from Git.

## Environment

Model-free validation environment:

| Component | Version |
|---|---|
| Python | 3.9.25 |
| PyTorch | 2.8.0+cu128 |
| CUDA build | 12.8 |
| Transformers | 4.57.6 |
| Accelerate | 1.10.1 |
| Hugging Face Hub | 0.36.2 |
| Tokenizers | 0.22.2 |
| Safetensors | 0.7.0 |
| RDKit | 2025.9.2 |
| NumPy | 1.26.4 |
| Pandas | 2.3.3 |

The table above records the model-free validation environment. Flash-Next uses Python 3.11 and Transformers 5.17.0. The model-class imports and architecture mapping were checked, but checkpoint loading and GPU generation have not yet been validated. Actual GPU availability depends on the cluster allocation. The older dependency snapshot is retained in `requirements-test-py39.txt`.

## Installation

Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
```

Install a PyTorch build compatible with your CUDA environment. The tested
environment used PyTorch 2.8.0 with CUDA 12.8 support.

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Check the environment:

```bash
python -m pip check
```

The vLLM backend is optional. Its runtime dependencies and compatibility
must be installed and validated separately. Do not install vLLM into an
existing cluster environment without checking its PyTorch, CUDA, Python,
and model-architecture compatibility.

## Dataset preparation

The dataset preparation script expects an existing GuacaMol dataset
directory containing:

```text
guacamol_full/
├── train.csv
└── val.csv
```

The input CSV files must contain:
- `smiles`
- `qed`, `logp`, `sas`, `tpsa`
- `mean_qed`, `mean_logp`, `mean_sas`, `mean_tpsa`
- `std_qed`, `std_logp`, `std_sas`, `std_tpsa`

The property columns are expected to contain normalized values. The
preparation script converts them back to their original scales using the
provided means and standard deviations.

Run:

```bash
python src/prepare_dataset.py \
  --data-dir /path/to/guacamol_full
```

Processed files are written to:

```text
outputs/processed/guacamol_full/
├── train.csv
├── val.csv
└── metadata.json
```

If the processed dataset is already available, this step can be skipped.

## Build zero-shot manifests

Generate target and job manifests:

```bash
python src/build_manifest.py \
  --config configs/flash_zero_shot.json
```

This creates:

```text
manifests/
├── flash_zero_shot_targets.json
└── flash_zero_shot_jobs.jsonl
```

The configured experiment plan contains 800 unique generation jobs.

## Model generation

### Transformers backend

Request a suitable GPU allocation before running inference.

Start with a single-job test:

```bash
python src/generate_flash.py \
  --config configs/flash_zero_shot.json \
  --limit 1
```

Run the complete experiment:

```bash
python src/generate_flash.py \
  --config configs/flash_zero_shot.json
```

The generator saves records to:

```text
outputs/raw/qwen_flash_zero_shot.jsonl
```

It tracks completed job IDs and skips jobs already present in the output,
allowing interrupted runs to be resumed.

### Optional vLLM backend

After installing and validating a compatible vLLM environment:

```bash
python src/generate_flash_vllm.py \
  --config configs/flash_zero_shot.json \
  --limit 1
```

The vLLM runtime has not yet been validated for this checkpoint in the
tested environment.

## Evaluation

Evaluate generated records:

```bash
python src/evaluate_flash.py \
  --input outputs/raw/qwen_flash_zero_shot.jsonl \
  --output-root outputs/evaluation/flash_zero_shot
```

Alternatively, evaluate one experiment:

```bash
python src/evaluate_flash.py \
  --input outputs/raw/qwen_flash_zero_shot.jsonl \
  --experiment logp \
  --output-root outputs/evaluation/flash_zero_shot
```

The evaluator:
- Validates generated SMILES using RDKit.
- Calculates the requested molecular descriptors.
- Calculates validity, uniqueness, and novelty.
- Calculates absolute property errors.
- Reports overall and per-target metrics.
- Validates generation records against the Flash job manifest.

Each experiment receives its own directory containing:

```text
<experiment>/
├── molecules.csv
├── summary.csv
└── summary.json
```

## Aggregate results

Aggregate all configured experiments:

```bash
python src/aggregate_flash_results.py \
  --input-root outputs/evaluation/flash_zero_shot \
  --output-dir outputs/evaluation/flash_zero_shot/all_experiments
```

The aggregator checks the expected number of manifest jobs and labels
results as complete or partial.

It reads only the Flash evaluation directory, not legacy experiment
results.

## Model-free smoke testing

The repository provides a smoke test that uses predefined valid SMILES
instead of loading model weights.

Generate stub records:

```bash
python src/smoke_test.py \
  --config configs/flash_zero_shot.json
```

Evaluate them:

```bash
python src/evaluate_flash.py \
  --input outputs/raw/qwen_flash_smoke_test.jsonl \
  --output-root outputs/evaluation/flash_smoke_test
```

Aggregate them:

```bash
python src/aggregate_flash_results.py \
  --input-root outputs/evaluation/flash_smoke_test \
  --output-dir outputs/evaluation/flash_smoke_test/all_experiments
```

These tests validate manifest selection, record creation, RDKit evaluation,
and aggregation. Stub molecules are not model-generation results and must
not be reported as Qwen performance.

## Validation status

Validated:
- Python compilation of the source directory.
- Configuration and manifest generation.
- 800 unique zero-shot manifest jobs.
- RDKit property calculation.
- Transformers and vLLM command-line interfaces.
- Model-free smoke testing across all eight experiments.
- Evaluation of all eight experiments.
- Aggregation of all eight experiments.
- Separation from legacy Qwen 27B result directories.

Not yet validated:
- Loading the actual Qwen3.8-Flash-Next checkpoint.
- GPU inference with the Transformers backend.
- GPU inference with the vLLM backend.
- Full 800-job generation.

## Reproducibility

The experiment configuration is stored in:

```text
configs/flash_zero_shot.json
```

Modify that file to change targets, generation parameters, output paths,
or model settings.

Generated outputs and manifests are not committed to the repository.
Regenerate manifests using `src/build_manifest.py` before generation.
