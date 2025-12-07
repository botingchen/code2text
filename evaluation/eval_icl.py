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

import argparse     
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
    "llama3.2_3b_instruct_4shot": "val_llama3.2_3b_4shot.jsonl",
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
        lang="en"
    )
    # result["f1"] is a list of per-example scores
    f1_scores = result["f1"]
    return float(sum(f1_scores) / len(f1_scores))

# -------------------------
# Main
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ICL model outputs using BLEU, ROUGE-L, METEOR, and BERTScore."
    )
    parser.add_argument(
        "--input-files",
        type=str,
        nargs="+",
        help="Path(s) to input JSONL file(s) to evaluate. Can specify multiple files.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        help="Labels for each input file (optional). Should match the number of input files.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=os.path.join(OUTPUT_DIR, "icl_eval_results.csv"),
        help="Path to save the evaluation CSV (default: outputs/icl/icl_eval_results.csv)",
    )
    
    args = parser.parse_args()
    
    # Determine which files to evaluate
    if args.input_files:
        input_files = args.input_files
        # Generate labels: use provided labels or derive from filenames
        if args.labels:
            if len(args.labels) != len(input_files):
                print("Warning: Number of labels doesn't match number of input files. Using filenames as labels.")
                labels = [os.path.splitext(os.path.basename(f))[0] for f in input_files]
            else:
                labels = args.labels
        else:
            labels = [os.path.splitext(os.path.basename(f))[0] for f in input_files]
        
        eval_files = dict(zip(labels, input_files))
    else:
        # Fall back to default EVAL_FILES
        print("No input files specified. Using default EVAL_FILES.")
        eval_files = {label: os.path.join(OUTPUT_DIR, filename) for label, filename in EVAL_FILES.items()}
    
    rows = []

    for label, jsonl_path in eval_files.items():
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

    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
