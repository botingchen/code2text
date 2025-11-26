"""
In-context learning evaluation for the 11-667 final project.

- Models:
    - gpt2
    - meta-llama/Llama-3.2-3B-Instruct
- Task:
    Go code -> natural language description (code-to-text)
- Dataset:
    data/processed/code_x_glue_go_lenfiltered_sampled
      splits: train / validation / test

This script:
    - Loads the processed dataset
    - Builds few-shot prompts from the train split
    - Runs ICL generation on the validation split
    - Saves outputs for each model to JSONL for later evaluation
"""

from __future__ import annotations  

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import torch
from datasets import load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# -------------------------
# Configuration
# -------------------------

DATA_DIR = "data/processed/code_x_glue_go_lenfiltered_sampled"
OUTPUT_DIR = "outputs/icl"

# Model names
GPT2_MODEL_NAME = "gpt2"
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# Few-shot & generation settings
NUM_FEW_SHOT_EXAMPLES = 4
MAX_NEW_TOKENS = 64  # upper bound; we may reduce this per-example based on context

RANDOM_SEED = 42  # for future use if you randomize few-shot selection


# -------------------------
# Prompt builders
# -------------------------

def build_gpt2_prompt(
    few_shot_examples: List[Dict[str, str]],
    code: str,
) -> str:
    """Pattern-style prompt for GPT-2: 'Code:' / 'Description:' pairs."""
    parts: List[str] = []

    for ex in few_shot_examples:
        parts.append("Code:\n" + ex["code"])
        parts.append("Description:\n" + ex["docstring"].strip())
        parts.append("")  # blank line separator

    parts.append("Code:\n" + code)
    parts.append("Description:\n")

    return "\n".join(parts)


def build_llama_prompt(
    few_shot_examples: List[Dict[str, str]],
    code: str,
) -> str:
    """Instruction-style prompt for LLaMA-3.2-3B-Instruct."""
    header = (
        "You are an expert Go developer. "
        "Given a Go function, you write a concise, high-level English description of what the function does. "
        "Focus on the purpose and behavior, not low-level implementation details.\n\n"
        "Here are some examples:\n\n"
    )

    example_blocks: List[str] = []
    for i, ex in enumerate(few_shot_examples, start=1):
        block = (
            f"[Example {i}]\n"
            f"Code:\n{ex['code']}\n"
            f"Description:\n{ex['docstring'].strip()}\n"
        )
        example_blocks.append(block)

    target_block = (
        "\nNow describe the following function.\n\n"
        f"Code:\n{code}\n"
        "Description:\n"
    )

    return header + "\n".join(example_blocks) + target_block


# -------------------------
# Data & ICL utilities
# -------------------------

def pick_few_shot_examples(
    train_ds: Dataset,
    k: int,
) -> List[Dict[str, str]]:
    """
    Pick k few-shot examples from the train split.
    For now this is deterministic (first k).
    """
    k = min(k, len(train_ds))
    examples = [train_ds[i] for i in range(k)]
    return [{"code": ex["code"], "docstring": ex["docstring"]} for ex in examples]


def extract_generated_description(full_text: str) -> str:
    """
    Extract the model's description from the full decoded text by taking
    everything after the last occurrence of 'Description:'.
    """
    marker = "Description:"
    last_idx = full_text.rfind(marker)
    if last_idx != -1:
        return full_text[last_idx + len(marker):].strip()
    return full_text.strip()


@dataclass
class ICLModelConfig:
    """Configuration for one ICL model run."""
    name: str
    prompt_builder: Callable[[List[Dict[str, str]], str], str]
    output_filename: str


# -------------------------
# Core generation logic
# -------------------------

def generate_for_model(
    model_cfg: ICLModelConfig,
    few_shot_examples: List[Dict[str, str]],
    val_ds: Dataset,
) -> None:
    """
    Run ICL generation on the validation split for a single model,
    and save results to JSONL under OUTPUT_DIR.
    """
    model_name = model_cfg.name
    output_path = os.path.join(OUTPUT_DIR, model_cfg.output_filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # use dtype instead of torch_dtype to avoid deprecation warning
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Model context length (fallback to 1024 if not set)
    max_context = getattr(model.config, "max_position_embeddings", 1024)

    print(f"Running generation on {len(val_ds)} validation examples...")
    with open(output_path, "w", encoding="utf-8") as f_out:
        for idx, ex in enumerate(val_ds):
            code = ex["code"]
            reference = ex["docstring"]

            prompt = model_cfg.prompt_builder(few_shot_examples, code)

            # Tokenize prompt first, without forcing a max_length yet
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_context,  # clamp prompt itself to context window
            ).to(device)

            input_len = inputs["input_ids"].shape[1]

            # Compute how many new tokens we can safely generate
            # so that input_len + new_tokens <= max_context
            available_for_generation = max_context - input_len
            if available_for_generation <= 0:
                # Edge case: prompt already fills the context
                # We skip generation or set 1 token to avoid errors.
                effective_max_new_tokens = 1
            else:
                effective_max_new_tokens = min(MAX_NEW_TOKENS, available_for_generation)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=effective_max_new_tokens,
                    do_sample=False,       # deterministic for now
                    temperature=1.0,
                    top_p=1.0,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_desc = extract_generated_description(full_text)

            record = {
                "id": idx,
                "model_name": model_name,
                "code": code,
                "reference": reference,
                "prompt": prompt,
                "generated": generated_desc,
            }
            f_out.write(json.dumps(record) + "\n")

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(val_ds)} examples")

    print(f"Saved outputs for {model_name} to: {output_path}")


# -------------------------
# Main
# -------------------------

def main() -> None:
    print(f"Loading processed dataset from: {DATA_DIR}")
    ds = load_from_disk(DATA_DIR)
    train_ds = ds["train"]
    val_ds = ds["validation"]

    print(f"Train size: {len(train_ds)}, Validation size: {len(val_ds)}")

    few_shot_examples = pick_few_shot_examples(train_ds, NUM_FEW_SHOT_EXAMPLES)
    print(f"Using {len(few_shot_examples)} few-shot examples in prompts.")

    model_configs: List[ICLModelConfig] = [
        # ICLModelConfig(
        #     name=GPT2_MODEL_NAME,
        #     prompt_builder=build_gpt2_prompt,
        #     output_filename="val_gpt2_icl.jsonl",
        # ),
        ICLModelConfig(
            name=LLAMA_MODEL_NAME,
            prompt_builder=build_llama_prompt,
            output_filename="val_llama3.2_3b_instruct_icl.jsonl",
        ),
    ]

    for cfg in model_configs:
        generate_for_model(
            model_cfg=cfg,
            few_shot_examples=few_shot_examples,
            val_ds=val_ds,
        )

    print("\nDone with ICL generation for all models.")


if __name__ == "__main__":
    main()
