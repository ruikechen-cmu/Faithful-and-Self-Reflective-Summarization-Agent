import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,  # Use Seq2Seq collator to better handle padding and masks
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


# =========================================================
# Interface 1: Base model & paths
# =========================================================
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  # Base model
TRAIN_FILE = "./outputs/baseline_llama3_cnn_dm/generations_self_refine_1step_train.jsonl"  # Training data: one JSON per line
VAL_FILE = "./outputs/baseline_llama3_cnn_dm/generations_self_refine_1step_val.jsonl"      # Validation data (optional)
OUTPUT_DIR = "./llama3-self-reflective-lora"  # LoRA output directory
# Maximum sequence length to avoid OOM (adjust according to GPU memory: 2048, 4096, 8192, etc.)
MAX_SEQ_LENGTH = 1024


# =========================================================
# Interface 2: Data field names (must match JSON keys)
#   Each line example:
#   {"article": "...source X...", "refined": "...reflection-improved summary Yn..."}
# =========================================================
DOC_FIELD = "article"
Yn_FIELD = "refined_1"


# =========================================================
# Interface 3: Prompt template (prompt format)
# Only keep the System Prompt content; chat format is handled by tokenizer
# =========================================================
SYSTEM_PROMPT = "Write a concise, factual summary (3–5 sentences) of the given news article."


# =========================================================
# Interface 4: LoRA configuration (r, alpha, and which modules to apply LoRA to)
#   target_modules must match the actual model architecture
# =========================================================
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


def build_prompt_messages(article: str) -> List[Dict]:
    """
    Build message list in Llama-3-Instruct chat format.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": article}
    ]


def load_and_prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # LLaMA-3 usually has no pad_token; use eos_token as padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def preprocess_function(example: Dict, tokenizer):
    """
    Convert (X, Yn) into:
        input_ids, attention_mask, labels
    Only Yn (target summary) contributes to the loss (prompt tokens get label = -100).
    """
    article = example[DOC_FIELD]
    refined = example[Yn_FIELD]

    # Use apply_chat_template to generate standard chat-format prompt
    messages = build_prompt_messages(article)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Append EOS token to the target
    target = refined + tokenizer.eos_token

    # Encode prompt and target separately so we can mask labels on the prompt part
    prompt_enc = tokenizer(
        prompt,
        add_special_tokens=False,
    )
    target_enc = tokenizer(
        target,
        add_special_tokens=False,
    )

    input_ids = prompt_enc["input_ids"] + target_enc["input_ids"]
    attention_mask = [1] * len(input_ids)

    # Prompt part does not contribute to loss, so set labels to -100 there
    labels = [-100] * len(prompt_enc["input_ids"]) + target_enc["input_ids"]

    # Truncation logic:
    # Simple tail truncation to prevent overly long sequences and avoid OOM
    if len(input_ids) > MAX_SEQ_LENGTH:
        input_ids = input_ids[:MAX_SEQ_LENGTH]
        attention_mask = attention_mask[:MAX_SEQ_LENGTH]
        labels = labels[:MAX_SEQ_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = load_and_prepare_tokenizer()

    # ==========================
    # 1. Load dataset
    # ==========================
    # If using HF Hub, replace with load_dataset("xxx/yyy")
    data_files = {"train": TRAIN_FILE}
    if os.path.exists(VAL_FILE):
        data_files["validation"] = VAL_FILE

    raw_datasets = load_dataset("json", data_files=data_files)

    # Preprocessing
    column_names = list(raw_datasets["train"].features)

    def preprocess_wrapper(example):
        return preprocess_function(example, tokenizer)

    tokenized_datasets = raw_datasets.map(
        preprocess_wrapper,
        batched=False,
        remove_columns=column_names,
        desc="Tokenizing",
    )
    
    # ==========================
    # 2. Load base model + attach LoRA
    # ==========================
    print("Loading model...")
    # Core fix: use device_map={"": 0} instead of device_map="auto"
    # This avoids the "expected device meta but got cuda:0" error
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},  # Force loading to GPU 0 to avoid meta device conflicts
    )

    # Enable gradient checkpointing (must be called before get_peft_model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # ==========================
    # 3. Data collator
    # ==========================
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    # ==========================
    # 4. Training arguments
    # ==========================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps" if "validation" in tokenized_datasets else "no",
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,  # Enable here; works together with enable_input_require_grads() above
        report_to="none",
    )

    # ==========================
    # 5. Trainer
    # ==========================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ==========================
    # 6. Start training
    # ==========================
    print("Starting training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
