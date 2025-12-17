# Faithful-and-Self-Reflective-Summarization-Agent

This repository contains an end-to-end **abstractive summarization system** with automated factuality evaluation and parameter-efficient fine-tuning.
The project focuses on **engineering reliable ML pipelines**, reproducible experiments, and practical evaluation of self-reflection mechanisms in large language models.

---

## Project Overview

* **Task**: Abstractive summarization on the CNN/DailyMail dataset
* **Models**: HuggingFace LLaMA-3 (8B) for generation; BART (evaluation-only) for QAFactEval
* **Tech Stack**: PyTorch, HuggingFace Transformers, LoRA, AWS GPU, QAFactEval
* **Focus**: Pipeline design, evaluation infrastructure, and failure analysis of reflection-based learning

The project implements and compares:

1. A baseline summarization pipeline
2. An inference-time self-reflective pipeline (draft → critique → revise)
3. A LoRA-based fine-tuning pipeline
4. Automated evaluation using ROUGE, BERTScore, and QAFactEval

---

## Repository Structure

```
.
├── scripts/                 # Executable entry points
│   ├── data                 # Data used from Huggingface
│   │   ├── cnn_dm_train.jsonl
│   │   ├── cnn_dm_test.jsonl
│   │   └── cnn_dm_val.jsonl
│   ├── run_baseline.py
│   ├── run_self_refine.py
│   ├── train_lora.py
│   ├── eval_metrics.py
│   └── eval_qafacteval.py
│
├── requirements/            # Dependency definitions
│   ├── requirements_for_baseline.txt
│   ├── requirements_for_eval.txt
│   ├── requirements_LoRA.txt
│   └── requirements_qafacteval.txt
│
├── assets/                  # Evaluation figures
│   └── metrics_*.png
│
├── README.md
└── .gitignore
```

> **Note**: Large datasets, model checkpoints, and generated outputs are intentionally excluded from version control.

---

## Installation

### Baseline / Generation

```bash
pip install -r requirements/requirements_for_baseline.txt
```

### LoRA Fine-Tuning

```bash
pip install -r requirements/requirements_LoRA.txt
```

### Evaluation (Metrics)

```bash
pip install -r requirements/requirements_for_eval.txt
```

### Evaluation (QAFactEval)

```bash
pip install -r requirements/requirements_qafacteval.txt
```

> QAFactEval was tested with PyTorch 1.6. GPU acceleration is supported but optional.

---

## Usage

### 1. Run Baseline Summarization

```bash
python scripts/run_baseline.py
```

### 2. Run Self-Reflective Summarization

```bash
python scripts/run_self_refine.py
```

### 3. Train with LoRA

```bash
python scripts/train_lora.py
```

### 4. Evaluate Outputs

```bash
# ROUGE / BERTScore
python scripts/eval_metrics.py

# QAFactEval (factual consistency)
python scripts/eval_qafacteval.py
```

---

## Key Findings

* Inference-time self-reflection can **degrade factual consistency** when critique signals are noisy.
* Fine-tuning on low-quality reflection artifacts may **internalize errors instead of correcting them**.
* Factuality evaluation (QAFactEval) introduces significant computational overhead and requires careful environment isolation.

These results highlight practical **failure modes and trade-offs** in reflection-based ML systems under realistic compute constraints.

---

## Engineering Highlights

* Modular pipeline design separating **generation, training, and evaluation**
* Reproducible experiment workflows on **AWS GPU instances**
* Explicit dependency isolation for evaluation tooling
* Clear separation between **core models** and **evaluation-only dependencies**

---

## Disclaimer

This repository is intended for **educational and experimental purposes**.
It demonstrates system design, evaluation infrastructure, and engineering trade-offs rather than proposing a production-ready summarization model.


