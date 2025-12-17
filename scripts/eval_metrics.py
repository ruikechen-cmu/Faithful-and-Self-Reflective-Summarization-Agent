# Compute ROUGE and BERTScore and generate plots
# The input format: item = {"id": doc_id, "article": art, "reference": ref, "draft": y0, "feedback_1": fb_1, "refined_1": y1, ……}

import json
import evaluate
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# === Path configuration ===
FILE_PATH = "./outputs/baseline_llama3_cnn_dm/generations_finetuned_test.jsonl"
# FILE_PATH = "./outputs/baseline_llama3_cnn_dm/generations_self_refine_1step_test.jsonl"

OUTPUT_PREFIX = "metrics_cnn_dm"  # Will generate metrics_cnn_dm_step0.png, step1.png, ...

QAFACTEVAL_METRICS_PATH = "./outputs/baseline_llama3_cnn_dm/qafacteval_generations_finetuned_test_metrics.json"
# QAFACTEVAL_METRICS_PATH = "./outputs/baseline_llama3_cnn_dm/qafacteval_generations_self_refine_1step_test_metrics.json"


def compute_rouge_bertscore(predictions, references, device):
    """Given predictions and references, compute ROUGE and BERTScore, return a dict."""
    rouge = evaluate.load('rouge')
    rouge_results = rouge.compute(predictions=predictions, references=references)

    bertscore = evaluate.load('bertscore')
    bert_results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        device=device,
        batch_size=32
    )

    scores = {
        "ROUGE-1": rouge_results['rouge1'],
        "ROUGE-2": rouge_results['rouge2'],
        "ROUGE-L": rouge_results['rougeL'],
        "BERTScore (F1)": float(np.mean(bert_results['f1']))
    }
    return scores


def plot_scores(scores, title, output_path):
    """Given a scores dict, draw a bar chart and save it."""
    print(f"Generating figure: {output_path}")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    df = pd.DataFrame(list(scores.items()), columns=['Metric', 'Score'])

    ax = sns.barplot(x='Metric', y='Score', data=df, palette="viridis", hue='Metric', legend=False)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=12)

    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('Score', fontsize=14)
    plt.xlabel('', fontsize=14)
    max_val = max(scores.values()) if scores else 1.0
    upper = max(1.0, max_val * 1.1)
    plt.ylim(0, upper)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Figure saved: {output_path}")


def load_qafacteval_metrics(path: str):
    """
    Load files like:
      {
        "draft": {"QAFactEval_lerc_quip_avg": 3.72},
        "refined_1": {"QAFactEval_lerc_quip_avg": 3.53}
      }
    Return dict {stage_name: score}
    """
    if not path or not os.path.exists(path):
        print(f"[QAFactEval] Metrics file not found: {path}, QAFactEval plot will be skipped.")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stage2score = {}
    for stage_name, stage_dict in data.items():
        if isinstance(stage_dict, dict) and "QAFactEval_lerc_quip_avg" in stage_dict:
            stage2score[stage_name] = float(stage_dict["QAFactEval_lerc_quip_avg"])
    print(f"[QAFactEval] Loaded stage scores: {stage2score}")
    return stage2score


def compute_and_plot(json_file_path):
    print(f"Reading data: {json_file_path}")
    items = []

    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))

    if len(items) == 0:
        print("File is empty. Exiting.")
        return

    # Extract all references
    references = [it.get('reference', "") for it in items]

    # === Auto-detect what “stages” exist in this file ===
    # Support:
    #   baseline-only: prediction
    #   self-refine: draft, refined_1, refined_2, ...
    example = items[0].keys()

    has_draft = 'draft' in example
    refined_keys = []
    for k in example:
        if k.startswith("refined_"):
            try:
                step = int(k.split("_")[1])
                refined_keys.append((step, k))
            except:
                continue
    refined_keys = sorted(refined_keys, key=lambda x: x[0])  # sort by step

    # Decide which stages to evaluate
    # Format: list of (name, key_in_json)
    eval_stages = [
        ("draft", "draft"),         # baseline or finetune uses this
        ("refined_1", "refined_1"), # self-refine
    ]

    print("Evaluating the following stages:")
    for name, key in eval_stages:
        print(f"  - {name}  (field: {key})")

    qafacteval_stage_scores = load_qafacteval_metrics(QAFACTEVAL_METRICS_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Generate a plot for each stage
    base_dir = os.path.dirname(json_file_path)
    base_name = os.path.splitext(os.path.basename(json_file_path))[0]

    is_finetune_run = ("finetune" in base_name.lower()) or ("lora" in base_name.lower())

    for idx, (stage_name, field_key) in enumerate(eval_stages):

        predictions = [it.get(field_key, "") for it in items]
        if len(predictions) == 0:
            print(f"[Warning] Stage {stage_name} has no predictions, skipping.")
            continue

        print(f"\n=== Computing metrics: {stage_name} ===")
        scores = compute_rouge_bertscore(predictions, references, device=device)

        if field_key in qafacteval_stage_scores:
            qa_raw = qafacteval_stage_scores[field_key]
            scores["QAFactEval (lerc_quip)"] = float(qa_raw)
        else:
            print(f"[QAFactEval] No QAFactEval score found for stage {field_key}, skipping.")

        # Determine final stage name
        if field_key == "draft":
            if is_finetune_run:
                short_name = "lora_finetune"
            else:
                short_name = "baseline"
        elif field_key == "refined_1":
            short_name = "selfrefine"
        else:
            short_name = stage_name

        safe_name = short_name.replace(" ", "_")

        out_png_name = f"{OUTPUT_PREFIX}_{base_name}_{safe_name}.png"
        out_png_path = os.path.join(base_dir, out_png_name)

        plot_scores(scores, title=short_name, output_path=out_png_path)

    print("\nAll stages evaluated.")


if __name__ == "__main__":
    compute_and_plot(FILE_PATH)
