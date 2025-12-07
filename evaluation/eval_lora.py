"""
Train + evaluate a LoRA adapter for the Go code-to-text task, and compare it
against the frozen base model on the same validation subset.

Usage example:
    python evaluation/eval_lora.py \
        --data-dir data/processed/code_x_glue_go_lenfiltered_sampled \
        --base-model gpt2 \
        --lora-output-dir outputs/lora/gpt2

    python evaluation/eval_lora.py \
        --data-dir data/processed/code_x_glue_go_lenfiltered_sampled \
        --base-model meta-llama/Llama-3.2-3B-Instruct \
        --lora-output-dir outputs/lora/llama

    nohup python3 evaluation/eval_lora.py --data-dir data/processed/code_x_glue_go_lenfiltered_sampled --base-model gpt2 --lora-output-dir outputs/lora/gpt2 > logs/stdout_gpt 2> logs/stderr_gpt &
    nohup python3 evaluation/eval_lora.py --data-dir data/processed/code_x_glue_go_lenfiltered_sampled --base-model meta-llama/Llama-3.2-3B-Instruct --lora-output-dir outputs/lora/llama > logs/stdout_llama 2> logs/stderr_llama &

    ## Loading existing LoRA checkpoint
    nohup python3 evaluation/eval_lora.py --load-lora-dir outputs/lora/gpt2 --data-dir data/processed/code_x_glue_go_lenfiltered_sampled --base-model gpt2 --lora-output-dir outputs/lora/gpt2 > logs/stdout_gpt 2> logs/stderr_gpt &
    nohup python3 evaluation/eval_lora.py --load-lora-dir outputs/lora/llama --data-dir data/processed/code_x_glue_go_lenfiltered_sampled --base-model meta-llama/Llama-3.2-3B-Instruct --lora-output-dir outputs/lora/llama > logs/stdout_llama 2> logs/stderr_llama &
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

from peft import LoraConfig, get_peft_model, PeftModel
from eval_icl import compute_bleu, compute_rouge_l, compute_meteor, compute_bertscore

import sys
sys.path.append("scripts")
from run_icl import build_prompt_from_file

DEFAULT_DATA_DIR = "data/processed/code_x_glue_go_lenfiltered_sampled"
DEFAULT_OUTPUT_DIR = "outputs/lora"
DEFAULT_BASE_MODEL = "gpt2"

# ---------------------------------------------------------------------------
# Helper functions for building prompts and formatting
# ---------------------------------------------------------------------------

def build_generation_prompt(code, model_type):
    model_type = model_type.lower()
    if "gpt2" in model_type:
        return build_prompt_from_file("prompts/gpt2_0shot.txt", code)
    return build_prompt_from_file("prompts/llama_0shot.txt", code)

def build_training_text(code, docstring, tokenizer, model_type):
    return f"{build_generation_prompt(code, model_type)}{docstring.strip()}{tokenizer.eos_token}"

def extract_description(full_text):
    marker = "Description:"
    idx = full_text.rfind(marker)
    if idx != -1:
        return full_text[idx + len(marker):].strip()
    return full_text.strip()

# ---------------------------------------------------------------------------
# Function for computing all metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(refs: List[str], preds: List[str]) -> Dict[str, float]:
    return {
        "BLEU": compute_bleu(refs, preds),
        "ROUGE-L": compute_rouge_l(refs, preds),
        "METEOR": compute_meteor(refs, preds),
        "BERTScore": compute_bertscore(refs, preds)
    }

# ---------------------------------------------------------------------------
# Functions for tokenizing the dataset
# ---------------------------------------------------------------------------

def tokenize_supervised_dataset(ds, tokenizer, max_seq_length, model_type):
    def _tokenize(batch):
        texts = [
            build_training_text(code, docstring, tokenizer, model_type)
            for code, docstring in zip(batch["code"], batch["docstring"])
        ]
        tokenized = tokenizer(texts, max_length = max_seq_length, truncation = True, padding = False)
        return tokenized

    return ds.map(_tokenize, batched = True, remove_columns = ds.column_names)

# ---------------------------------------------------------------------------
# Functions for loading the base model and ensuring the pad token is set
# ---------------------------------------------------------------------------

def ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

def load_base_model(model_name):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = dtype)
    return model

def infer_default_target_modules(model_type):
    model_type = model_type.lower()
    if "llama" in model_type:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    return ["c_attn"]

def can_use_bf16():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8

def train_lora(base_model, tokenizer, train_dataset, args):
    resolved_output_dir = resolve_non_conflicting_output_dir(args.lora_output_dir)

    target_modules = infer_default_target_modules(base_model.config.model_type)

    lora_config = LoraConfig(
        r = args.lora_r,
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        bias = "none",
        target_modules = target_modules,
        task_type = "CAUSAL_LM"
    )

    peft_model = get_peft_model(base_model, lora_config)
    if hasattr(peft_model, "print_trainable_parameters"):
        peft_model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False)

    training_args = TrainingArguments(
        output_dir = resolved_output_dir,
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        learning_rate = args.learning_rate,
        num_train_epochs = args.num_train_epochs,
        logging_steps = args.logging_steps,
        save_strategy = "epoch",
        save_total_limit = 1,
        fp16 = torch.cuda.is_available() and not can_use_bf16(),
        bf16 = can_use_bf16(),
        report_to = "none",
        optim = "paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch"
    )

    if hasattr(peft_model.config, "use_cache"):
        peft_model.config.use_cache = False

    trainer = Trainer(
        model = peft_model,
        args = training_args,
        train_dataset = train_dataset,
        data_collator = data_collator
    )
    trainer.train()

    os.makedirs(resolved_output_dir, exist_ok = True)
    peft_model.save_pretrained(resolved_output_dir)
    tokenizer.save_pretrained(resolved_output_dir)

    save_training_parameters(args, lora_config, target_modules, training_args, resolved_output_dir)

    return peft_model, resolved_output_dir

# ---------------------------------------------------------------------------
# Functions for loading trained LoRA model
# ---------------------------------------------------------------------------

def resolve_non_conflicting_output_dir(output_dir):
    if not os.path.exists(output_dir):
        return output_dir
    res_dir = output_dir
    idx = 1
    while os.path.exists(res_dir):
        res_dir = output_dir + "_" + str(idx)
        idx += 1
    return res_dir

def lora_checkpoint_exists(output_dir):
    config_path = os.path.join(output_dir, "adapter_config.json")
    weight_paths = [
        os.path.join(output_dir, "adapter_model.bin"),
        os.path.join(output_dir, "adapter_model.safetensors"),
    ]
    return os.path.isfile(config_path) and any(os.path.isfile(p) for p in weight_paths)

def load_or_train_lora(train_ds, tokenizer, args):
    base_model = load_base_model(args.base_model)
    if args.load_lora_dir:
        if lora_checkpoint_exists(args.load_lora_dir):
            print(f"Found existing LoRA checkpoint in {args.load_lora_dir}; loading it.")
            return PeftModel.from_pretrained(base_model, args.load_lora_dir), args.load_lora_dir
        else:
            raise ValueError(f"LoRA checkpoint not found in {args.load_lora_dir}")

    print("\nTokenizing train subset for LoRA fine-tuning...")
    tokenized_train = tokenize_supervised_dataset(
        train_ds, tokenizer, args.max_seq_length, args.base_model
    )
    print("Training LoRA adapter...")
    return train_lora(base_model, tokenizer, tokenized_train, args)

# ---------------------------------------------------------------------------
# Functions for evaluating the model
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, tokenizer, dataset, max_seq_length, max_new_tokens, temperature, top_p, device, model_type):
    model.eval()
    max_context = getattr(model.config, "max_position_embeddings", max_seq_length)
    records: List[Dict] = []

    for idx, example in enumerate(dataset):
        prompt = build_generation_prompt(example["code"], model_type)
        inputs = tokenizer(
            prompt,
            return_tensors = "pt",
            truncation = True,
            max_length = min(max_seq_length, max_context),
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        prompt_len = inputs["input_ids"].shape[1]
        available = max_context - prompt_len
        effective_new_tokens = max(1, min(max_new_tokens, available))

        outputs = model.generate(
            **inputs,
            max_new_tokens = effective_new_tokens,
            do_sample = temperature > 0,
            temperature = temperature if temperature > 0 else 1.0,
            top_p = top_p,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.pad_token_id,
            repetition_penalty = 1.2,
        )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens = True)
        prediction = extract_description(decoded)
        reference = example["docstring"]

        records.append(
            {
                "id": int(example["id"]) if "id" in example else idx,
                "code": example["code"],
                "reference": reference,
                "generated": prediction
            }
        )

    refs = [rec["reference"] for rec in records]
    preds = [rec["generated"] for rec in records]
    metrics = compute_all_metrics(refs, preds)
    return metrics, records

def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, "w", encoding = "utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

def save_metrics(results, path):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(results, f, indent = 2)

def save_training_parameters(args, lora_config, target_modules, training_args, output_dir):
    os.makedirs(output_dir, exist_ok = True)
    payload = {
        "base_model": args.base_model,
        "data_dir": args.data_dir,
        "eval_split": args.eval_split,
        "max_seq_length": args.max_seq_length,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "training": {
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "logging_steps": args.logging_steps,
            "optimizer": training_args.optim,
            "fp16": training_args.fp16,
            "bf16": training_args.bf16,
        },
        "lora": {
            "r": lora_config.r,
            "alpha": lora_config.lora_alpha,
            "dropout": lora_config.lora_dropout,
            "target_modules": list(target_modules),
        },
    }
    params_path = os.path.join(output_dir, "training_parameters.json")
    with open(params_path, "w", encoding = "utf-8") as f:
        json.dump(payload, f, indent = 2)


# ---------------------------------------------------------------------------
# Function for parsing command line arguments
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type = str, default = DEFAULT_DATA_DIR)
    parser.add_argument("--base-model", type = str, default = DEFAULT_BASE_MODEL)
    parser.add_argument("--eval-split", type = str, default = "validation")
    parser.add_argument("--max-seq-length", type = int, default = 1024)
    parser.add_argument("--max-new-tokens", type = int, default = 64)
    parser.add_argument("--temperature", type = float, default = 0.0)
    parser.add_argument("--top-p", type = float, default = 0.95)

    parser.add_argument("--per-device-train-batch-size", type = int, default = 2)
    parser.add_argument("--gradient-accumulation-steps", type = int, default = 8)
    parser.add_argument("--num-train-epochs", type = float, default = 1.0)
    parser.add_argument("--learning-rate", type = float, default = 2e-4)
    parser.add_argument("--logging-steps", type = int, default = 25)

    parser.add_argument("--lora-r", type = int, default = 16)
    parser.add_argument("--lora-alpha", type = int, default = 32)
    parser.add_argument("--lora-dropout", type = float, default = 0.05)

    parser.add_argument(
        "--load-lora-dir",
        type = str,
        default = None
    )
    parser.add_argument(
        "--lora-output-dir",
        type = str,
        default = DEFAULT_OUTPUT_DIR
    )
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading dataset from {args.data_dir}")
    dataset = load_from_disk(args.data_dir)
    train_ds = dataset["train"]
    eval_ds = dataset[args.eval_split]

    tokenizer_source = args.load_lora_dir if args.load_lora_dir else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    ensure_pad_token(tokenizer)

    # ------------------------------------------------------------------
    # Baseline evaluation
    # ------------------------------------------------------------------
    print(f"\nEvaluating frozen base model: {args.base_model}")
    baseline_model = load_base_model(args.base_model).to(device)
    baseline_metrics, baseline_records = evaluate_model(
        model = baseline_model,
        tokenizer = tokenizer,
        dataset = eval_ds,
        max_seq_length = args.max_seq_length,
        max_new_tokens = args.max_new_tokens,
        temperature = args.temperature,
        top_p = args.top_p,
        device = device,
        model_type = args.base_model
    )
    del baseline_model
    torch.cuda.empty_cache()

    results = [{"Model": "baseline", **baseline_metrics}]

    # ------------------------------------------------------------------
    # LoRA Training / Loading and Evaluation
    # ------------------------------------------------------------------
    lora_model, resolved_output_dir = load_or_train_lora(train_ds, tokenizer, args)

    lora_model.to(device)
    print("\nEvaluating LoRA-adapted model...")
    lora_metrics, lora_records = evaluate_model(
        model = lora_model,
        tokenizer = tokenizer,
        dataset = eval_ds,
        max_seq_length = args.max_seq_length,
        max_new_tokens = args.max_new_tokens,
        temperature = args.temperature,
        top_p = args.top_p,
        device = device,
        model_type = args.base_model
    )

    # Save predictions for baseline and LoRA models
    save_jsonl(
        baseline_records,
        os.path.join(resolved_output_dir, args.base_model.replace("/", "_") + "_baseline_predictions.jsonl"),
    )
    save_jsonl(
        lora_records,
        os.path.join(resolved_output_dir, args.base_model.replace("/", "_") + "_lora_predictions.jsonl"),
    )
    results.append({"Model": "lora", **lora_metrics})

    results_path = os.path.join(resolved_output_dir, "metrics.json")

    save_metrics(results, results_path)

if __name__ == "__main__":
    main()