"""
Evaluate ICL model outputs for the 11-667 final project.

Reads JSONL output from scripts/run_icl.py and computes:
    - BLEU
    - ROUGE-L
    - METEOR
    - BERTScore

Usage:
    python scripts/eval_icl_outputs.py

Make sure you have:
    pip install evaluate pandas
"""

from __future__ import annotations

import json
import os
from typing import List, Dict

import evaluate
import pandas as pd

# -------------------------
# Config
# -------------------------

OUTPUT_DIR = "outputs/icl"

# Map a friendly model/prompt label to a JSONL file.
# Update these filenames to match your actual outputs.
EVAL_FILES = {
    "gpt2_pattern_4shot": "val_gpt2_icl.jsonl",
    "llama3.2_3b_instruct_4shot": "val_llama3.2_3b_instruct_icl.jsonl",
}

# -------------------------
# I/O helpers
# -------------------------

def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# -------------------------
# Metrics (using evaluate)
# -------------------------

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")
bertscore_metric = evaluate.load("bertscore")


def compute_bleu(references: List[str], predictions: List[str]) -> float:
    """
    BLEU expects:
      - predictions: List[str]
      - references: List[List[str]]
    where each inner list is the set of reference texts for that prediction.
    """
    score = bleu_metric.compute(
        predictions=predictions,
        references=[[ref] for ref in references],  # one ref per example
    )
    return score["bleu"]


def compute_rouge_l(references: List[str], predictions: List[str]) -> float:
    """
    ROUGE-L wrapper that works across different `evaluate` versions.

    Some versions return:
      score["rougeL"] -> float
    Others return:
      score["rougeL"] -> object with .mid.fmeasure

    This handles both.
    """
    score = rouge_metric.compute(
        predictions=predictions,
        references=references,
        rouge_types=["rougeL"],
    )
    val = score["rougeL"]

    # Case 1: val is a plain float (numpy.float64 or Python float)
    if isinstance(val, (float, int)):
        return float(val)

    # Case 2: val is a rouge_score.scoring.Score object
    # which has a .mid.fmeasure attribute
    if hasattr(val, "mid"):
        mid = val.mid
        if isinstance(mid, dict):
            # Sometimes mid is a dict
            return float(mid.get("fmeasure", 0.0))
        if hasattr(mid, "fmeasure"):
            return float(mid.fmeasure)

    # Fallback: try to cast to float
    return float(val)


def compute_meteor(references: List[str], predictions: List[str]) -> float:
    """
    METEOR expects:
      - predictions: List[str]
      - references: List[str]
    """
    score = meteor_metric.compute(
        predictions=predictions,
        references=references,
    )
    return score["meteor"]

def compute_bertscore(references: List[str], predictions: List[str]) -> float:
    """
    Compute BERTScore (F1) using a pretrained English model.
    We average the F1 scores over all examples.
    """
    result = bertscore_metric.compute(
        predictions=predictions,
        references=references,
        model_type="microsoft/deberta-base-mnli",  # or "roberta-large" if you have more VRAM
        lang="en",
        rescale_with_baseline=True,
    )
    # result["f1"] is a list of per-example scores
    f1_scores = result["f1"]
    return float(sum(f1_scores) / len(f1_scores))

# -------------------------
# Main
# -------------------------

def main() -> None:
    rows = []

    for label, filename in EVAL_FILES.items():
        jsonl_path = os.path.join(OUTPUT_DIR, filename)

        print(f"\nLoading predictions for {label} from {jsonl_path}")
        data = load_jsonl(jsonl_path)

        # These are raw strings, not token lists.
        preds = [ex["generated"] for ex in data]
        refs = [ex["reference"] for ex in data]

        bleu = compute_bleu(refs, preds)
        rouge_l = compute_rouge_l(refs, preds)
        meteor = compute_meteor(refs, preds)
        bertscore = compute_bertscore(refs, preds)

        rows.append({
            "Model+Prompt": label,
            "BLEU": bleu,
            "ROUGE-L": rouge_l,
            "METEOR": meteor,
            "BERTScore": bertscore,
        })

    df = pd.DataFrame(rows)
    print("\n=== ICL Evaluation Results (Validation Set) ===\n")
    print(df.to_string(index=False))

    out_csv = os.path.join(OUTPUT_DIR, "icl_eval_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved CSV to {out_csv}")


if __name__ == "__main__":
    main()
