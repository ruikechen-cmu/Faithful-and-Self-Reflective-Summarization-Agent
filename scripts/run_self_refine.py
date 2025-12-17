# This is run_self_refine, returning summaries from LLaMA-3 8B. Optionally support LLaMA-3 8B self-reflection or multi-step refinement.
# Input: CNN DailyMail. Output id, article, draft / feedback / refined * k rounds
# Baseline output format:
#   item = {"id": doc_id, "article": art, "reference": ref, "draft": y0, "feedback_1": fb_1, "refined_1": y1, …}
# --self_refine 0 means no refinement and directly output the result. --self_refine k means performing k rounds of refinement and outputting draft / feedback / refined processes.
'''
# Single-round baseline, run only on val
python run_self_refine.py ^
  --split val ^
  --self_refine 0 ^
  --data_dir "./data" ^
  --output_dir "./outputs/baseline_llama3_cnn_dm"

# Run 1-step Self-Refine on test (draft + feedback_1 + refined_1)
python run_self_refine.py ^
  --split test ^
  --self_refine 1 ^
  --data_dir "./data" ^
  --output_dir "./outputs/baseline_llama3_cnn_dm"
'''


import os
import argparse
import json
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import evaluate
import numpy as np


# ======================
# Command-line arguments
# ======================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline: LLaMA-3 8B summarization + optional Self-Refine on CNN/DailyMail (local jsonl)"
    )
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="HuggingFace model id")  # This is the switch for loading llama3 8B

    # Same as baseline_1: train / val / test
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="which split jsonl to use: train / val / test",
    )

    parser.add_argument("--max_samples", type=int, default=4,
                        help="limit number of samples for quick baseline (None for full)")  # How many articles to load
    parser.add_argument("--batch_size", type=int, default=2)  # Increase depending on GPU memory
    parser.add_argument("--max_new_tokens", type=int, default=1024)  # Truncate / pad length for summarization
    parser.add_argument("--output_dir", type=str, default="outputs/baseline_llama3_cnn_dm")  # Output directory
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu; default auto")  # Device
    parser.add_argument("--load_4bit", action="store_true",
                        help="use 4-bit quantization (requires bitsandbytes)")  # Memory-saving option

    # self_refine switch: 0 = single-shot baseline; 1 = one self-refine step (draft → feedback → rewrite)
    parser.add_argument("--self_refine", type=int, default=0,
                        help="0: single-shot baseline; k: k-step Self-Refine (draft + feedback + refine)")

    # New: your local dataset directory
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"D:\CMU\10623-GenAI\project\data",
        help="directory where cnn_dm_train/val/test jsonl are stored",
    )

    return parser.parse_args()


# ======================
# Prompt construction
# ======================

SYSTEM_PROMPT = (
    "You are a helpful AI assistant that writes concise, factual summaries of news articles. "
    "Your summaries must be faithful to the source document and avoid hallucinating details."
)

FEEDBACK_SYSTEM_PROMPT = (
    "You are a careful reviewer who critiques summaries of news articles. "
    "Your feedback should be actionable and specific, pointing out factual errors, "
    "missing important information, hallucinations, and unclear or vague wording."
)

REFINE_SYSTEM_PROMPT = (
    "You are an expert editor who rewrites summaries of news articles based on feedback. "
    "You must follow the feedback while keeping the summary faithful to the article and avoiding hallucinations."
)


def build_prompt(article: str) -> List[Dict]:
    """Step 0: Generate the initial summary y0 from the article"""
    user_prompt = (
        "Please write a concise, factual summary (1–3 sentences) of the following news article.\n"
        "Directly output the summary without introductory phrases (e.g., 'Here is the summary').\n\n"
        f"Article:\n{article}\n\nSummary:"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_feedback_prompt(article: str, current_summary: str) -> List[Dict]:
    """Generate feedback fb_t based only on the article and the current summary output"""
    user_prompt = (
        "Point out:"
        "- any factual errors or contradictions with the article,"
        "- any clearly important information from the article that is missing (focus on main events, not side details),"
        "- any hallucinated details that are not supported by the article,"
        "- any unclear or vague phrases that could confuse the reader."
        "Do NOT nitpick stylistic choices that do not affect factual accuracy or clarity."
        f"Article:\n{article}\n\n"
        f"Current summary:\n{current_summary}\n\n"
        "Critique:"
    )

    return [
        {"role": "system", "content": FEEDBACK_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_refine_prompt(article: str,
                        summaries_history: List[str],
                        feedbacks_history: List[str]) -> List[Dict]:
    """
    For each refinement step, the input contains the article + all previous summaries + their corresponding feedback.
    Alignment rule:
        summaries_history[i] corresponds to feedbacks_history[i] (if available).
    Examples:
        step=1: summaries=[y0], feedbacks=[fb1] -> y0 + fb1
        step=2: summaries=[y0,y1], feedbacks=[fb1,fb2] -> y0+fb1, y1+fb2
    """
    history_parts = []
    for idx, y in enumerate(summaries_history):
        if idx < len(feedbacks_history):
            fb = feedbacks_history[idx]
            history_parts.append(
                f"Version {idx} summary:\n{y}\n\n"
                f"Feedback on version {idx}:\n{fb}\n"
            )
        else:
            history_parts.append(
                f"Version {idx} summary:\n{y}\n"
            )

    history_block = "\n".join(history_parts)

    user_prompt = (
        "You are given a news article, along with previous summaries and feedback on them. "
        "Write an improved summary (3–5 sentences) that:\n"
        "- fixes the issues mentioned in the feedback,\n"
        "- is faithful to the article,\n"
        "- avoids hallucinating any unsupported details,\n"
        "- is clear, concise, and coherent.\n"
        "Do NOT mention the feedback itself or explain your changes; just output the new summary.\n\n"
        f"Article:\n{article}\n\n"
        f"Previous summaries and feedback:\n{history_block}\n\n"
        "Improved summary:"
    )

    return [
        {"role": "system", "content": REFINE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

# ======================
# Model and data loading
# ======================

def load_model_and_tokenizer(model_name: str, device: str = None, load_4bit: bool = False):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if load_4bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
        )

    # If not using quantization and device_map is not set, manually move to CUDA
    if not load_4bit and device == "cuda" and not hasattr(model, "hf_device_map"):
        model.to(device)

    model.eval()
    return tokenizer, model, device


# ===== Load CNN/DailyMail from local jsonl =====
def load_cnn_dailymail_from_jsonl(
    data_dir: str,
    split: str,
    max_samples: int = None,
):
    """
    Load data from local jsonl.

    Expected filenames:
        cnn_dm_train.jsonl
        cnn_dm_val.jsonl
        cnn_dm_test.jsonl

    Each json line must contain at least:
        - "id"
        - "article"
        - "reference" (you previously replaced 'highlights' with 'reference')
      If still named "highlights", compatibility is preserved.
    """
    filename = f"cnn_dm_{split}.jsonl"
    path = os.path.join(data_dir, filename)
    print(f"Loading CNN/DailyMail from {path} ...")

    articles = []
    references = []
    ids = []

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            art = obj["article"]
            ref = obj.get("reference", obj.get("highlights"))
            doc_id = obj["id"]

            articles.append(art)
            references.append(ref)
            ids.append(doc_id)

    print(f"Loaded {len(articles)} examples from split = {split}")
    return articles, references, ids


# ======================
# General batch generation function
# ======================

def generate_from_prompts(
    model,
    tokenizer,
    prompts_text: List[str],
    batch_size: int = 2,
    max_new_tokens: int = 128,
    temperature: float = 0.1,
    top_p: float = 0.9,
    desc: str = "Generating",
) -> List[str]:
    """Given chat-template-expanded text prompts, generate outputs in batches and return only the new generation."""
    outputs_all = []

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    for i in tqdm(range(0, len(prompts_text), batch_size), desc=desc):
        batch_prompts = prompts_text[i:i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
            add_special_tokens=False,
        ).to(model.device)

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,        # deterministic mode
                top_p=1.0,
                eos_token_id=terminators,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Extract only newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = gen[:, input_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        decoded = [d.strip() for d in decoded]
        outputs_all.extend(decoded)

    return outputs_all


# ======================
# Baseline single-shot summarization
# ======================

def generate_summaries_single_shot(
    model,
    tokenizer,
    articles: List[str],
    batch_size: int = 2,
    max_new_tokens: int = 128,
) -> List[str]:
    prompts_text = [
        tokenizer.apply_chat_template(
            build_prompt(a),
            tokenize=False,
            add_generation_prompt=True,
        )
        for a in articles
    ]

    summaries = generate_from_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts_text=prompts_text,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        desc="Generating single-shot summaries",
    )
    return summaries


# ======================
# k-step Self-Refine
# ======================

def generate_summaries_self_refine_k(
    model,
    tokenizer,
    articles: List[str],
    k_steps: int,
    batch_size: int = 2,
    max_new_tokens: int = 128,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    k-step Self-Refine:
        y0: initial draft
        for t=1..k:
            fb_t = feedback on y_{t-1}
            y_t  = refine(article, [y0..y_{t-1}], [fb1..fb_t])

    Return:
        all_summaries: list of length (k+1), each element a list [y_i] of length N
        all_feedbacks: list of length k, each element a list [fb_{i+1}] of length N
    """
    assert k_steps >= 1
    N = len(articles)

    # ---- Step 0: initial drafts y0 ----
    print(f"[Self-Refine] Step 0: Generating initial drafts (y0)...")
    draft_prompts = [
        tokenizer.apply_chat_template(
            build_prompt(a),
            tokenize=False,
            add_generation_prompt=True,
        )
        for a in articles
    ]
    y0_list = generate_from_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts_text=draft_prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        desc="Generating drafts y0",
    )

    all_summaries: List[List[str]] = [y0_list]
    all_feedbacks: List[List[str]] = []

    current_summaries = y0_list

    # ---- Self-Refine iterations ----
    for step in range(1, k_steps + 1):
        print(f"[Self-Refine] Step {step}: feedback + refine")

        # 1) Generate feedback fb_step
        fb_prompts = [
            tokenizer.apply_chat_template(
                build_feedback_prompt(art, cur_sum),
                tokenize=False,
                add_generation_prompt=True,
            )
            for art, cur_sum in zip(articles, current_summaries)
        ]
        fb_list = generate_from_prompts(
            model=model,
            tokenizer=tokenizer,
            prompts_text=fb_prompts,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            desc=f"Generating feedback fb{step}",
        )
        all_feedbacks.append(fb_list)

        # 2) Generate refined summaries y_step based on all history
        refine_prompts = []
        for i in range(N):
            summaries_history_i = [s_list[i] for s_list in all_summaries]   # y0..y_{step-1}
            feedbacks_history_i = [f_list[i] for f_list in all_feedbacks]   # fb1..fb_step
            refine_prompts.append(
                tokenizer.apply_chat_template(
                    build_refine_prompt(
                        article=articles[i],
                        summaries_history=summaries_history_i,
                        feedbacks_history=feedbacks_history_i,
                    ),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        refined_list = generate_from_prompts(
            model=model,
            tokenizer=tokenizer,
            prompts_text=refine_prompts,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            desc=f"Refining summaries y{step}",
        )

        all_summaries.append(refined_list)
        current_summaries = refined_list

    return all_summaries, all_feedbacks


# ======================
# main
# ======================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load data — From local jsonl
    articles, references, ids = load_cnn_dailymail_from_jsonl(
        data_dir=args.data_dir,
        split=args.split,
        max_samples=args.max_samples,
    )

    # 2. Load model
    tokenizer, model, device_str = load_model_and_tokenizer(
        args.model_name, args.device, args.load_4bit
    )

    # 3. Generate
    if args.self_refine == 0:
        print("Running single-shot baseline (no Self-Refine)...")
        predictions = generate_summaries_single_shot(
            model=model,
            tokenizer=tokenizer,
            articles=articles,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        all_summaries = None
        all_feedbacks = None
        k_steps = 0
    else:
        k_steps = args.self_refine
        print(f"Running {k_steps}-step Self-Refine...")
        all_summaries, all_feedbacks = generate_summaries_self_refine_k(
            model=model,
            tokenizer=tokenizer,
            articles=articles,
            k_steps=k_steps,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        predictions = all_summaries[-1]

    # 4. Save results (JSONL)
    tag = f"baseline" if args.self_refine == 0 else f"self_refine_{k_steps}step"
    output_file = os.path.join(args.output_dir, f"generations_{tag}_{args.split}.jsonl")
    print(f"Saving generations to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, (art, ref, pred, doc_id) in enumerate(
            zip(articles, references, predictions, ids)
        ):
            if args.self_refine == 0:
                item = {
                    "id": doc_id,
                    "article": art,
                    "reference": ref,
                    "draft": pred,
                }
            else:
                item = {
                    "id": doc_id,
                    "article": art,
                    "reference": ref,
                }

                # y0
                item["draft"] = all_summaries[0][idx]

                # Each round's feedback and refined summary
                k_steps = args.self_refine
                for step in range(1, k_steps + 1):
                    fb_t = all_feedbacks[step - 1][idx]
                    y_t = all_summaries[step][idx]
                    item[f"feedback_{step}"] = fb_t
                    item[f"refined_{step}"] = y_t

            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
