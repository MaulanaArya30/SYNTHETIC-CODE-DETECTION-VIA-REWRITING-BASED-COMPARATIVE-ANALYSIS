# Synthetic Code Detection via Rewriting-Based Comparative Analysis

> **Paper:** "Comparative Analysis of Rewriting-Based Large Language Model-Generated Synthetic Code Detection"  
> **Authors:** Maulana Arya Alambana, Arif Nurwidyantoro  
> **Affiliation:** Department of Computer Science and Electronics, Universitas Gadjah Mada  
> **Status:** Submitted to IEEE ICAIIA 2026

---

## Overview

This repository contains the implementation for a **zero-shot synthetic code detection framework** that distinguishes AI-generated Python code from human-written code — without requiring any labeled training data.

The core idea: LLM-generated code, when rewritten by another language model, tends to produce outputs highly similar to the original (because it already aligns with the model's internal distribution). Human-written code, by contrast, undergoes more substantial changes during rewriting. We exploit this behavioral difference as the detection signal.

The study compares two code embedding architectures — **CodeT5** and **GraphCodeBERT** — each evaluated with and without **SimCSE** fine-tuning, across 11 stylistically diverse prompting variants from the Pan et al. (2024) dataset.

---

## Method

```
Original Code (Human / AI)
        │
        ▼
  Code Rewriting (Qwen2.5-Coder-7B-Instruct)
        │
        ▼
   m Rewritten Variants
        │
        ▼
  Code Embedding (CodeT5 / GraphCodeBERT + SimCSE)
        │
        ▼
  Cosine Similarity (original vs. rewritten)
        │
        ▼
  Pairwise Comparison → Detection Decision
```

**Rewriter:** Qwen2.5-Coder-7B-Instruct (nucleus sampling, top_p=0.95, temp=0.9)  
**Embedding models:** CodeT5-base, GraphCodeBERT (both optionally fine-tuned with SimCSE)  
**Similarity metric:** Cosine similarity, averaged across m rewrites  
**Rewrite amounts evaluated:** m = 2, 4, 8

---

## Key Results

| Model | AUROC (m=2) | AUROC (m=4) | AUROC (m=8) | PA (m=8) |
|---|---|---|---|---|
| CodeT5-BASE | 0.5733 | **0.5904** | 0.5843 | 0.5973 |
| CodeT5-SimCSE | 0.5682 | 0.5786 | 0.5882 | **0.6116** |
| GraphCodeBERT-BASE | 0.5661 | 0.5716 | 0.5756 | 0.5829 |
| GraphCodeBERT-SimCSE | 0.5656 | 0.5758 | 0.5837 | 0.6085 |

- All configurations exceed the random-guessing baseline (AUROC = 0.50)
- **CodeT5-based models** achieve a higher average AUROC (0.5805) than **GraphCodeBERT-based models** (0.5731)
- Peak single-variant result: GraphCodeBERT-BASE on V13 (Add Dead Code) → **AUROC = 0.8529**
- Identifier renaming variants (V8–V10) collapse detection to near-random (avg. AUROC ~0.51–0.53)

---

## Dataset

The dataset is sourced from [Pan et al. (2024)](https://doi.org/10.1145/3639474.3640068), publicly available on [Figshare](https://figshare.com/articles/dataset/Replication_Package/24298036).

It contains 5,069 problem-solution pairs (LeetCode, Kaggle, Quescol) with 11 prompting variants:

| Variant | Description |
|---|---|
| V1 | Without Modification |
| V4 | Solution Without Comments |
| V5 | Assertion Test Code |
| V6 | Solution with Test Cases |
| V7 | Unittest Test Case |
| V8 | Replace Variable Names |
| V9 | Replace Function Names |
| V10 | Replace Variables and Functions |
| V11 | Long Method |
| V12 | Short Method |
| V13 | Add Dead Code |

---

## Repository Structure

```
├── data/               # Dataset CSVs and rewrite cache (JSON)
├── models/             # Saved SimCSE fine-tuned encoder weights
├── notebooks/          # Jupyter notebooks for exploration and evaluation
├── src/                # Source code
│   ├── rewriting/      # Code rewriting pipeline (Qwen2.5-Coder)
│   ├── embedding/      # SimCSE training for CodeT5 and GraphCodeBERT
│   ├── scoring/        # Cosine similarity scoring and evaluation
│   └── utils/          # Shared utilities
├── pyproject.toml      # Poetry dependency definition
└── poetry.lock
```

---

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
# Clone the repository
git clone https://github.com/MaulanaArya30/SYNTHETIC-CODE-DETECTION-VIA-REWRITING-BASED-COMPARATIVE-ANALYSIS.git
cd SYNTHETIC-CODE-DETECTION-VIA-REWRITING-BASED-COMPARATIVE-ANALYSIS

# Install dependencies
poetry install

# Activate the environment
poetry shell
```

**Requirements:** Python 3.9+, CUDA-compatible GPU recommended for rewriting and embedding steps.

---

## Usage

### 1. Download the dataset

Download the Pan et al. (2024) dataset from [Figshare](https://figshare.com/articles/dataset/Replication_Package/24298036) and place the CSV files under `data/`.

### 2. Generate rewrites (Phase 1)

```bash
python src/rewriting/generate_rewrites.py --variant V1 --m 8
```

Rewrites are cached as JSON files under `data/rewrites/` to avoid redundant GPU computation across model configurations.

### 3. Train SimCSE embeddings

```bash
# Fine-tune CodeT5 with SimCSE
python src/embedding/train_simcse.py --model codet5

# Fine-tune GraphCodeBERT with SimCSE
python src/embedding/train_simcse.py --model graphcodebert
```

### 4. Run scoring and evaluation (Phase 2)

```bash
python src/scoring/evaluate.py --model codet5-simcse --m 8
```

Results (AUROC and Pairwise Accuracy) are logged per variant and aggregated.

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@inproceedings{alambana2026synthetic,
  title     = {Comparative Analysis of Rewriting-Based Large Language Model-Generated Synthetic Code Detection},
  author    = {Alambana, Maulana Arya and Nurwidyantoro, Arif},
  booktitle = {2026 IEEE International Conference on AI Implementation \& Applications (ICAIIA)},
  year      = {2026},
  note      = {Under Review}
}
```

---

## References

- Ye et al. (2025). [Uncovering LLM-Generated Code: A Zero-Shot Synthetic Code Detector via Code Rewriting.](https://doi.org/10.1609/aaai.v39i1.32082) *AAAI 2025.*
- Pan et al. (2024). [Assessing AI Detectors in Identifying AI-Generated Code.](https://doi.org/10.1145/3639474.3640068) *ICSE-SEET 2024.*
- Wang et al. (2021). [CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models.](https://doi.org/10.18653/v1/2021.emnlp-main.685) *EMNLP 2021.*
- Gao et al. (2021). [SimCSE: Simple Contrastive Learning of Sentence Embeddings.](https://doi.org/10.18653/v1/2021.emnlp-main.552) *EMNLP 2021.*
