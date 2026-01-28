# Hard Reasoning Benchmark via Model Disagreement

This repository provides a fully reproducible pipeline for constructing and evaluating a hard reasoning benchmark derived from model–model disagreement.

The core idea is to identify prompts that induce high epistemic disagreement between strong language models, and to use these prompts as a principled benchmark for evaluating hard reasoning capability.

The benchmark is constructed without human annotation and is model-agnostic at evaluation time.

## Overview

The pipeline consists of four stages:

1. Local or automatic loading of reasoning datasets
2. Token-level disagreement computation between two foundation models
3. Construction of a hardness-ranked benchmark
4. Evaluation of multiple models on the benchmark

The entire pipeline can be executed end-to-end using a single command.

## Datasets

The following datasets are used:

- GSM8K
- ARC-Easy
- ARC-Challenge
- MATH (relative algebra subset)

Datasets are accessed via Hugging Face and cached locally. No manual dataset download is required.

## Models

Disagreement computation is performed between:

- Qwen3-8B (base)
- DeepSeek-Math-7B (base)

Evaluation can be run on either or both models independently.

All models are loaded via Hugging Face and cached automatically.

## Repository Structure

```
.
├── configs
│   └── paths.yaml
├── data
│   ├── gsm8k
│     └── train.jsonl
│     └── test.jsonl
│   ├── arc-challenge
│     └── test-00000-of-00001.parquet
│     └── train-00000-of-00001.parquet
│     └── validation-00000-of-00001.parquet
│   ├── arc-easy
│     └── test-00000-of-00001.parquet
│     └── train-00000-of-00001.parquet
│     └── validation-00000-of-00001.parquet
│   └── math
│     └── with_steps_flist_relative_algebra.txt
│     └── no_steps_flist_relative_algebra.txt
├── scripts
│   ├── run_disagreement.sh
│   ├── build_benchmark.sh
│   ├── eval_all_models.sh
│   └── run_all.sh
├── src
│   └── utils
│     └── io.py
│   ├── frame.py
│   ├── txt_csv.py
│   ├── build_hard_reasoning_benchmark.py
│   ├── eval_qwen3_on_hrb_all.py
│   └── eval_deepseek_on_hrb_all.py
├── results
│   └── (generated outputs)
└── README.md
````

## Installation

Create a Python environment with the following dependencies:

```bash
pip install torch transformers datasets pandas tqdm pyyaml
````

A CUDA-enabled GPU is required for full-scale runs.

## Configuration

All paths are centralized in `configs/paths.yaml`:

```yaml
data_root: data
results_root: results

datasets:
  gsm8k: data/gsm8k
  arc_easy: data/arc-easy
  arc_challenge: data/arc-challenge
  math: data/math

models:
  qwen3: Qwen/Qwen3-8B
  deepseek: deepseek-ai/deepseek-math-7b
```

These paths work out of the box and do not require modification for standard usage.

## Running the Full Pipeline

To run the complete pipeline from scratch:

```bash
bash scripts/run_all.sh
```

This performs the following steps:

1. Computes token-level disagreement scores
2. Converts raw outputs into CSV format
3. Builds the hard reasoning benchmark
4. Evaluates all configured models

All outputs are written to the `results/` directory.

## Intermediate Scripts

You may also run stages independently:

### Disagreement computation

```bash
bash scripts/run_disagreements.sh
```

### Benchmark construction

```bash
bash scripts/build_benchmark.sh
```

### Model evaluation

```bash
bash scripts/eval_all_models.sh
```

## Outputs

Key generated files include:

* `results/disagreement_scores.csv`
* `results/hard_reasoning_benchmark.csv`
* `results/qwen3_8b_hrb_all_results.csv`
* `results/deepseek_7b_hrb_all_results.csv`

Each evaluation file contains:

* Prompt
* Hardness bucket
* Hardness score
* Model response
* Latency
* Token usage statistics

## Hardness Definition

Hardness is defined as:

```
hardness_score = disagreement_score × (1 + used_tokens_pct / 100)
```

## Citation

If you use this benchmark or methodology, please cite the corresponding paper.

```
@article{hard_reasoning_disagreement,
  title={Hard Reasoning Benchmarks via Model Disagreement},
  author={Durgesh Rao, Dr. Shirshendu Layek},
  year={2026}
}
```
## License

This repository is released under the MIT License.
