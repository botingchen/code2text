"""
Visualize length distributions:

1. Original train split
2. Filtered train split (<= MAX_CODE_TOKENS)
3. Sampled 2K train split (stratified sampling)

Requires:
- Raw dataset in HF cache (from earlier load)
- Processed dataset stored at data/processed/code_x_glue_go_lenfiltered_2k
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, load_from_disk, Dataset

# -------------------------
# Config
# -------------------------

DATASET_NAME = "google/code_x_glue_ct_code_to_text"
SUBSET_NAME = "go"

PROCESSED_PATH = "data/processed/code_x_glue_go_lenfiltered_2k"
MAX_CODE_TOKENS = 450   # or your chosen threshold
PLOTS_DIR = "plots/sample_vs_original"


# -------------------------
# Helpers
# -------------------------

def compute_lengths(ds: Dataset):
    return np.array([len(x) for x in ds["code_tokens"]])


def filter_by_max_length(ds, max_len):
    return ds.filter(lambda ex: len(ex["code_tokens"]) <= max_len)


def plot_three_histograms(
    original,
    filtered,
    sampled,
    output_file,
    clip_x=800,
    bins=60,
):
    """Plot original vs filtered vs sampled distributions."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Clip long tails for visualization
    orig_clipped = np.clip(original, 0, clip_x)
    filt_clipped = np.clip(filtered, 0, clip_x)
    samp_clipped = np.clip(sampled, 0, clip_x)

    plt.figure(figsize=(9, 6))

    plt.hist(orig_clipped, bins=bins, alpha=0.5, label="Original train", density=True)
    plt.hist(filt_clipped, bins=bins, alpha=0.5, label=f"Filtered (<= {MAX_CODE_TOKENS})", density=True)
    plt.hist(samp_clipped, bins=bins, alpha=0.7, label="Sampled 2K (stratified)", density=True)

    plt.title("Code Length Distribution: Original vs Filtered vs Sampled (Go)")
    plt.xlabel("Code length (# tokens, clipped)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"Saved plot to {output_file}")


def print_stats(name, lengths):
    print(f"\n[{name}]")
    print(f"  count: {len(lengths)}")
    print(f"  min:   {lengths.min()}")
    print(f"  max:   {lengths.max()}")
    print(f"  mean:  {lengths.mean():.2f}")
    for p in [50, 75, 90, 95, 99]:
        print(f"  p{p}:   {np.percentile(lengths, p):.2f}")


# -------------------------
# Main
# -------------------------

def main():
    print("Loading original dataset (cached)...")
    raw = load_dataset(DATASET_NAME, SUBSET_NAME)
    train_raw = raw["train"]

    print("Computing original train lengths...")
    lengths_original = compute_lengths(train_raw)
    print_stats("Original", lengths_original)

    print("Filtering by max length...")
    train_filtered = filter_by_max_length(train_raw, MAX_CODE_TOKENS)
    lengths_filtered = compute_lengths(train_filtered)
    print_stats("Filtered", lengths_filtered)

    print(f"Loading sampled dataset from {PROCESSED_PATH}...")
    processed = load_from_disk(PROCESSED_PATH)
    train_sampled = processed["train"]

    lengths_sampled = compute_lengths(train_sampled)
    print_stats("Sampled 2K", lengths_sampled)

    output_file = os.path.join(
        PLOTS_DIR, "code_len_original_filtered_sampled.png"
    )

    plot_three_histograms(
        original=lengths_original,
        filtered=lengths_filtered,
        sampled=lengths_sampled,
        output_file=output_file,
        clip_x=800,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
