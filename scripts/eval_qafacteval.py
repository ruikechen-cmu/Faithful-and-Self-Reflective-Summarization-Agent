import json
import os
from typing import List, Dict, Any

import numpy as np
import torch
from qafacteval import QAFactEval

# ======= You need to modify these two paths =======
# 1. Your jsonl results file (draft / refined_1 / refined_2 are all inside it)
FILE_PATH = "./outputs/baseline_llama3_cnn_dm/generations_self_refine_1step_test.jsonl"
# FILE_PATH = "./outputs/baseline_llama3_cnn_dm/generations_finetuned_test.jsonl"

# 2. QAFactEval models directory
QAFACTEVAL_MODEL_FOLDER = "./QAFactEval/models"

# Whether to use GPU if possible
USE_GPU = True


def load_items(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def init_qafacteval() -> QAFactEval:
    """
    Initialize QAFactEval; use GPU if available, otherwise fall back to CPU.
    """
    if USE_GPU and torch.cuda.is_available():
        cuda_device = 0
    else:
        cuda_device = -1

    print(f"[QAFactEval] Using device: {'cuda:0' if cuda_device == 0 else 'cpu'}")

    m = QAFACTEVAL_MODEL_FOLDER
    kwargs = {
        "cuda_device": cuda_device,
        "use_lerc_quip": True,
        "verbose": True,
        "generation_batch_size": 4,
        "answering_batch_size": 4,
        "lerc_batch_size": 4,
    }

    metric = QAFactEval(
        lerc_quip_path=f"{m}/quip-512-mocha",
        generation_model_path=f"{m}/generation/model.tar.gz",
        answering_model_dir=f"{m}/answering",
        lerc_model_path=f"{m}/lerc/model.tar.gz",
        lerc_pretrained_model_path=f"{m}/lerc/pretraining.tar.gz",
        **kwargs,
    )
    return metric


def extract_lerc_quip_scores(results: Any) -> List[float]:
    """
    QAFactEval uses an older nested return format; this parser is made robust:
    supports several common nesting patterns and extracts lerc_quip scores per document.
    """
    scores = []

    for doc_res in results:
        cand_res = None

        # Case 1: list/tuple, first element is candidate result
        if isinstance(doc_res, (list, tuple)):
            first = doc_res[0]
            if isinstance(first, list) and first:
                cand_res = first[0]
            elif isinstance(first, dict):
                cand_res = first

        # Case 2: directly a dict
        elif isinstance(doc_res, dict):
            cand_res = doc_res

        if not isinstance(cand_res, dict):
            continue

        qa_eval = cand_res.get("qa-eval", {})
        if not isinstance(qa_eval, dict):
            continue

        val = None
        # Try different possible key names from different versions
        for key in ["lerc_quip", "lerc_quip_score", "lerc_quip_scores"]:
            if key in qa_eval:
                val = qa_eval[key]
                break

        if isinstance(val, (int, float)):
            scores.append(float(val))
        elif isinstance(val, list) and val:
            # Some implementations give a score per QA pair; take average
            try:
                scores.append(float(np.mean(val)))
            except Exception:
                continue

    return scores


def compute_qafacteval_average(metric: QAFactEval,
                               sources: List[str],
                               predictions: List[str]) -> float:
    """
    Compute average QAFactEval (lerc_quip) score for a batch of [source, prediction] pairs.
    """
    assert len(sources) == len(predictions)
    candidates = [[pred] for pred in predictions]  # One candidate per document

    print(f"[QAFactEval] Scoring {len(sources)} samples...")
    results = metric.score_batch_qafacteval(
        sources,
        candidates,
        return_qa_pairs=True,
    )

    scores = extract_lerc_quip_scores(results)
    if not scores:
        raise RuntimeError("No lerc_quip scores parsed from QAFactEval results.")

    avg = float(np.mean(scores))
    print(f"[QAFactEval] Parsed {len(scores)} scores, average = {avg:.4f}")
    return avg


def main():
    items = load_items(FILE_PATH)
    print(f"Loaded {len(items)} examples from {FILE_PATH}")

    if not items:
        print("File is empty. Exiting.")
        return

    # Original article text, used for generating QA
    articles = [it.get("article", "") for it in items]

    # Auto-detect which stages are present
    example_keys = items[0].keys()
    stages = []

    if "draft" in example_keys:
        stages.append(("draft", "draft"))

    for k in example_keys:
        if k.startswith("refined_"):
            stages.append((k, k))

    if not stages:
        # Fallback: if only prediction exists
        if "prediction" in example_keys:
            stages.append(("prediction", "prediction"))

    print("Stages to evaluate:", stages)
    if not stages:
        print("No summary fields found. Exiting.")
        return

    metric = init_qafacteval()

    summary_results = {}
    for stage_name, key in stages:
        preds = [it.get(key, "") for it in items]
        print(f"\n==== Computing QAFactEval for stage {stage_name} ====")
        try:
            score = compute_qafacteval_average(metric, articles, preds)
        except Exception as e:
            print(f"[Error] Failed computing for stage {stage_name}: {e}")
            score = None

        summary_results[stage_name] = {
            "QAFactEval_lerc_quip_avg": score
        }

    # Save results to a json file in the same directory
    base_dir = os.path.dirname(FILE_PATH)
    base_name = os.path.splitext(os.path.basename(FILE_PATH))[0]
    out_path = os.path.join(base_dir, f"qafacteval_{base_name}_metrics.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)

    print("\n=== All Done ===")
    print(f"Results saved to: {out_path}")
    print("Preview:")
    print(json.dumps(summary_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
