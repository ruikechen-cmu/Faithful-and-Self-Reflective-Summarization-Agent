# This is run_baseline, directly returning summaries from LLaMA-3 8B without any extra post-processing
# Input: local CNN DailyMail jsonl with id, article, reference
# Output: id, article text, reference summary, model summary
# Returned format:
#   item = {"id": doc_id, "article": art, "reference": ref, "draft": pred}
# Run commands:
# 1. Train split
'''
python run_baseline.py ^
  --split train ^
  --data_dir "fill in" ^
  --output_dir "fill in"

# 2. Validation split
python run_baseline.py ^
  --split val ^
  --data_dir "fill in" ^
  --output_dir "fill in"

# 3. Test split
python run_baseline.py ^
  --split test ^
  --data_dir "fill in" ^
  --output_dir "fill in"
'''

import os
import argparse
import json
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from peft import PeftModel

# ========================
# Command-line arguments
# ========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline: LLaMA-3 8B single-shot summarization on CNN/DailyMail (local jsonl)"
                    "optionally with LoRA fine-tuned adapter"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model id",
    )

    # Now using train / val / test splits corresponding to your jsonl files
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="which split jsonl to use: train / val / test",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="limit number of samples (None for all lines in jsonl)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=5, help="batch size for generation"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="max new tokens to generate for summary",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/baseline_llama3_cnn_dm",
        help="directory to save generations",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda / cpu; default auto",
    )
    parser.add_argument(
        "--load_4bit",
        action="store_true",
        help="use 4-bit quantization (requires bitsandbytes)",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"D:\CMU\10623-GenAI\project\data",
        help="directory where cnn_dm_train/val/test jsonl are stored",
    )
    
    parser.add_argument(
        "--lora_dir",
        type=str,
        default=None,
        help="path to LoRA fine-tuned adapter (output_dir of LoRA training). "
             "If None, use raw base model.",
    )

    return parser.parse_args()


# ========================
# Prompt construction
# ========================
SYSTEM_PROMPT = (
    "You are a helpful AI assistant that writes concise, factual summaries of news articles. "
    "Your summaries must be faithful to the source document and avoid hallucinating details."
)


def build_prompt(article: str) -> List[Dict]:
    """
    Wrap one CNN/DailyMail article into the dialogue format required by LLaMA-3 8B Instruct.
    """
    user_prompt = (
        "Please write a concise, factual summary (1-3 sentences) of the following news article.\n"
        "Directly output the summary without introductory phrases (e.g., 'Here is the summary').\n\n"
        f"Article:\n{article}\n\nSummary:"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# ========================
# Load model and tokenizer
# ========================
def load_model_and_tokenizer(model_name: str, 
                             device: str = None, 
                             load_4bit: bool = False,
                             lora_dir: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer_source = lora_dir if (lora_dir is not None and os.path.isdir(lora_dir)) else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Decoder-only models use left padding
    tokenizer.padding_side = "left"
    
    # For simplicity and stability, do not combine LoRA inference with 4bit here.
    if lora_dir is not None and load_4bit:
        print("[Warning] Using LoRA adapter and load_4bit together is not tested. "
              "Ignoring --load_4bit and loading full-precision base + LoRA.")
        load_4bit = False

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
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
        )
        
        # If LoRA exists, load LoRA adapter
        if lora_dir is not None:
            print(f"[Info] Loading LoRA adapter from {lora_dir} ...")
            model = PeftModel.from_pretrained(base_model, lora_dir)
        else:
            model = base_model

    # Manually push to device if device_map is not used
    if not load_4bit and device == "cuda" and not hasattr(model, "hf_device_map"):
        model.to(device)

    model.eval()
    return tokenizer, model, device


# ========================
# Load CNN/DailyMail jsonl
# ========================
def load_cnn_dailymail_from_jsonl(
    data_dir: str, split: str, max_samples: int = None
):
    """
    Load data from local jsonl.

    Expected filenames:
        cnn_dm_train.jsonl
        cnn_dm_val.jsonl
        cnn_dm_test.jsonl

    Each json line must contain:
        - "id"
        - "article"
        - "reference"   (your previous processing replaced 'highlights' with 'reference')
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
            ref = obj.get("reference", obj.get("highlights"))  # compatibility if still using 'highlights'
            doc_id = obj["id"]

            articles.append(art)
            references.append(ref)
            ids.append(doc_id)

    print(f"Loaded {len(articles)} examples from split = {split}")
    return articles, references, ids


# ========================
# Generate summaries in batch
# ========================
def generate_summaries(
    model,
    tokenizer,
    articles: List[str],
    batch_size: int = 2,
    max_new_tokens: int = 128,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> List[str]:
    summaries = []

    # The stop tokens for LLaMA-3
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    for i in tqdm(range(0, len(articles), batch_size), desc="Generating summaries"):
        batch_articles = articles[i : i + batch_size]

        # 1. Apply chat template to produce prompt text
        prompts_text = [
            tokenizer.apply_chat_template(
                build_prompt(a), tokenize=False, add_generation_prompt=True
            )
            for a in batch_articles
        ]

        # 2. Batch tokenize
        inputs = tokenizer(
            prompts_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
            add_special_tokens=False,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False if temperature == 0 else True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=terminators,
                pad_token_id=tokenizer.pad_token_id,
            )

        # 3. Decode only the newly generated part
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded = [d.strip() for d in decoded]
        summaries.extend(decoded)

    return summaries


# ========================
# main
# ========================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load data from jsonl
    articles, references, ids = load_cnn_dailymail_from_jsonl(
        data_dir=args.data_dir,
        split=args.split,
        max_samples=args.max_samples,
    )

    # 2. Load model
    tokenizer, model, device_str = load_model_and_tokenizer(
        args.model_name, args.device, args.load_4bit
    )

    # 3. Generate summaries
    summaries = generate_summaries(
        model=model,
        tokenizer=tokenizer,
        articles=articles,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    # 4. Save results to jsonl
    tag = "finetuned" if args.lora_dir is not None else "baseline"
    output_file = os.path.join(
        args.output_dir, f"generations_{tag}_{args.split}.jsonl"
    )
    print(f"Saving generations to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        for art, ref, pred, doc_id in zip(articles, references, summaries, ids):
            item = {
                "id": doc_id,
                "article": art,
                "reference": ref,
                "draft": pred,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
