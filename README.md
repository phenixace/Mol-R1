# Mol-R1: Towards Explicit Long-CoT Reasoning in Molecule Discovery

[[Paper]](https://arxiv.org/abs/2508.08401)

This repository contains the code and data for **Mol-R1**, a framework that improves explainability and reasoning performance of R1-like Explicit Long-CoT reasoning LLMs in text-based molecule generation.

## Overview

Mol-R1 introduces two key components:

- **PRID** (Prior Regulation via In-context Distillation): A distillation strategy that generates paired reasoning traces guided by in-context examples with known caption-molecule pairs.
- **MoIA** (Molecular Iterative Adaptation): An iterative training strategy combining SFT with RPO (Reinforced Policy Optimization), with rejection-sampling distillation to continuously expand the training set.

```
┌──────────────────────────────────────────────────────────┐
│              MoIA: Molecular Iterative Adaptation         │
│                                                          │
│   ┌─────┐     ┌─────┐     ┌────────────────────────┐    │
│   │ SFT │ ──▶ │ RPO │ ──▶ │ Rejection-Sampling     │    │
│   │     │     │     │     │ Distillation (RS/PRID)  │    │
│   └─────┘     └─────┘     └──────────┬─────────────┘    │
│      ▲                               │                   │
│      └──────── merge new data ───────┘                   │
│                                                          │
│   Stop when: data stops growing OR max steps reached     │
└──────────────────────────────────────────────────────────┘
```

## Project Structure

```
Mol-R1/
├── data/
│   ├── raw/                    # Original CheBI-20 dataset
│   │   └── chebi-20/
│   │       ├── train.txt       # 26,406 training samples
│   │       ├── test.txt        # 3,300 test samples
│   │       └── validation.txt  # 3,300 validation samples
│   ├── PRID-4o/                # GPT-4o PRID distilled reasoning traces (1,054 samples)
│   │   └── prid_4o_train.json
│   ├── PRID-G/                 # Gemma PRID distilled reasoning traces (7,285 samples)
│   │   └── prid_g_train.json
│   ├── RS-G/                   # Gemma rejection-sampling distilled traces (8,946 samples)
│   │   └── rs_g_train.json
│   └── MoIA/                   # MoIA iterative training data (T=0,1,2)
│       ├── T0.json             # Iteration 0: seed data (1,054 samples)
│       ├── T1.json             # Iteration 1: expanded data (7,285 samples)
│       └── T2.json             # Iteration 2: further expanded (8,946 samples)
├── src/
│   ├── train/
│   │   ├── sft_train.py        # SFT training
│   │   └── grpo_train.py       # RPO training
│   ├── eval/
│   │   ├── predict.py          # Multi-GPU inference
│   │   └── evaluate.py         # Evaluation metrics
│   └── distill/
│       ├── prid_distill.py     # PRID: Prior Regulation via In-context Distillation
│       ├── rejection_sampling_distill.py  # RS: Rejection-Sampling Distillation
│       └── moia.py             # MoIA: full iterative training loop
├── scripts/
│   ├── run_sft_eval.sh         # SFT + eval pipeline
│   ├── run_sft_rpo.sh          # SFT + RPO + eval pipeline
│   ├── run_moia.sh             # MoIA iterative training loop
│   ├── run_prid.sh             # PRID distillation
│   └── run_rs.sh               # RS distillation
├── requirements.txt
└── README.md
```

## Data

### Raw Dataset
CheBI-20 (Caption → Molecule) in TSV format: `CID \t SMILES \t description`

### Distilled Reasoning Traces

| Dataset | Method | Source | Samples | Description |
|---------|--------|--------|---------|-------------|
| **PRID-4o** | PRID | GPT-4o | 1,054 | In-context distilled reasoning traces from GPT-4o |
| **PRID-G** | PRID | Gemma-3-12B | 7,285 | In-context distilled reasoning traces from Gemma |
| **RS-G** | RS | Gemma-3-12B | 8,946 | Rejection-sampling filtered reasoning traces from Gemma |

### MoIA Iterative Data

| File | Iteration | Samples | Description |
|------|-----------|---------|-------------|
| `T0.json` | T=0 | 1,054 | Seed data (PRID-4o) |
| `T1.json` | T=1 | 7,285 | After 1st MoIA iteration |
| `T2.json` | T=2 | 8,946 | After 2nd MoIA iteration |

Each sample contains a `<think>...</think><answer>SMILES</answer>` formatted reasoning trace.

## Installation

```bash
pip install -r requirements.txt
```

## Hyper-parameters

### SFT
| Parameter | Value |
|-----------|-------|
| gpu_number (A800) | 2 |
| per_device_train_batch_size | 1 |
| gradient_accumulation_steps | 4 |
| learning_rate | 1e-5 |
| num_train_epochs | 5 |
| lr_scheduler_type | cosine |
| warmup_ratio | 0.1 |

### RPO (Reinforced Policy Optimization)
| Parameter | Value |
|-----------|-------|
| gpu_number (A800) | 8 |
| learning_rate | 1e-6 |
| weight_decay | 1e-2 |
| kl_coef | 1e-2 |
| n (num_generations) | 5 |
| rollout.temperature | 1.0 |
| global_batch_size | 128 |
| rollout_batch_size | 512 |
| micro_batch_size_per_device_for_update | 4 |

### Inference
| Parameter | Value |
|-----------|-------|
| temperature | 0.6 |
| top_p | 0.9 |
| max_tokens | 10000 |

## Quick Start

### 1. Distillation

```bash
# PRID with GPT-4o
bash scripts/run_prid.sh gpt-4o "" $OPENAI_API_KEY

# PRID with Gemma (via Ollama)
bash scripts/run_prid.sh gemma3:12b http://localhost:11434/v1/ ollama

# Rejection-Sampling with Gemma
bash scripts/run_rs.sh gemma3:12b http://localhost:11434/v1/
```

### 2. SFT + Evaluation

```bash
bash scripts/run_sft_eval.sh RS-G facebook/galactica-125m 2
```

### 3. SFT + RPO Pipeline

```bash
bash scripts/run_sft_rpo.sh RS-G facebook/galactica-125m 2 8
```

### 4. MoIA Iterative Training

```bash
# Start from T0 seed data
bash scripts/run_moia.sh data/MoIA/T0.json facebook/galactica-125m 5
```

The MoIA loop will:
- SFT cold-start → RPO → generate predictions on raw train data
- Score predictions via rejection sampling (threshold >= 8/10)
- Merge new high-quality data into training set
- Repeat until data stops growing or max steps reached

### 5. Custom Training

```bash
# SFT only
torchrun --nproc_per_node=2 src/train/sft_train.py \
    --mode cot \
    --model_name facebook/galactica-125m \
    --data_path data/RS-G/rs_g_train.json \
    --output_dir outputs/my_sft \
    --epochs 5 --lr 1e-5 --batch_size 1 --grad_accum 4

# RPO on top of SFT
torchrun --nproc_per_node=8 src/train/grpo_train.py \
    --sft_model_dir outputs/my_sft/final \
    --data_path data/RS-G/rs_g_train.json \
    --output_dir outputs/my_rpo \
    --epochs 2 --lr 1e-6 --num_generations 5 --temperature 1.0 --beta 0.01

# Evaluate
torchrun --nproc_per_node=8 src/eval/predict.py \
    --mode cot \
    --model_dir outputs/my_rpo/final \
    --test_path data/raw/chebi-20/test.txt \
    --output_path outputs/predictions.txt \
    --temperature 0.6 --top_p 0.9 --max_tokens 10000

python src/eval/evaluate.py outputs/predictions.txt
```

## Evaluation Metrics

- **Exact Match**: InChI-based exact match rate
- **BLEU**: Character-level BLEU score
- **Levenshtein**: Edit distance
- **Validity**: RDKit molecular validity rate
- **MACCS/RDK/Morgan FTS**: Fingerprint Tanimoto similarity

## Citation

```bibtex
@article{li2025molr1,
  title={Mol-R1: Towards Explicit Long-CoT Reasoning in Molecule Discovery},
  author={Li, Jiatong and Wang, Weida and Zhang, Qinggang and Li, Junxian and Zhang, Di and Zheng, Changmeng and Zhang, Shufei and Wei, Xiaoyong and Li, Qing},
  journal={arXiv preprint arXiv:2508.08401},
  year={2025}
}
```
