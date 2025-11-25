"""
Data pipeline for the 11-667 final project.

- Uses google/code_x_glue_ct_code_to_text (Go subset)
- Filters out long code sequences (by code_tokens length)
- Applies stratified sampling to train, validation, and test splits
- Preserves diversity across code length
- Saves all splits to disk
"""

from __future__ import annotations

import os
from typing import List, Set

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

# -------------------------
# Configuration
# -------------------------

DATASET_NAME = "google/code_x_glue_ct_code_to_text"
SUBSET_NAME = "go"

# Filter threshold
MAX_CODE_TOKENS = 450

# Sample sizes per split
TRAIN_SAMPLES = 2000
VAL_SAMPLES = 500      # you can change this number
TEST_SAMPLES = 500     # you can change this number

# Number of bins for stratified sampling
NUM_LENGTH_BINS = 10

# Random seed
RANDOM_SEED = 42

# Output directory
OUTPUT_DIR = "data/processed/code_x_glue_go_lenfiltered_sampled"


# -------------------------
# Helpers
# -------------------------

def add_code_len_column(ds: Dataset) -> Dataset:
    """Add a 'code_len' column = length of code_tokens."""
    return ds.map(
        lambda ex: {"code_len": len(ex["code_tokens"])},
        desc="Computing code length",
    )


def filter_by_max_length(ds: Dataset, max_len: int) -> Dataset:
    """Filter out examples whose code_tokens length exceeds max_len."""
    return ds.filter(
        lambda ex: len(ex["code_tokens"]) <= max_len,
        desc=f"Filtering examples with code_len > {max_len}",
    )


def stratified_sample_by_length(
    ds: Dataset,
    n_samples: int,
    num_bins: int,
    seed: int,
) -> Dataset:
    """
    Sample n_samples examples from ds, approximately preserving the
    distribution of code_len via simple stratified sampling over length bins.
    """
    if len(ds) <= n_samples:
        return ds

    rng = np.random.default_rng(seed)

    code_lens = np.array(ds["code_len"], dtype=np.int32)

    # Percentile-based binning
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(code_lens, percentiles)

    # Handle degenerate case
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) == 1:
        indices = rng.choice(len(ds), size=n_samples, replace=False)
        return ds.select(sorted(indices.tolist()))

    # Assign to bins
    bin_indices = np.digitize(code_lens, bin_edges[1:-1], right=True)
    all_indices = np.arange(len(ds))

    # Collect per-bin index sets
    bin_to_indices = []
    for b in range(len(bin_edges) - 1):
        mask = bin_indices == b
        bin_to_indices.append(all_indices[mask])

    # Allocate target samples per bin
    target_per_bin = max(n_samples // num_bins, 1)

    selected = set()
    remaining = n_samples

    # First pass: equal sampling per bin
    for bin_idxs in bin_to_indices:
        if remaining <= 0:
            break
        if len(bin_idxs) == 0:
            continue

        take = min(target_per_bin, len(bin_idxs), remaining)
        chosen = rng.choice(bin_idxs, size=take, replace=False)
        selected.update(chosen.tolist())
        remaining = n_samples - len(selected)

    # Second pass: fill remainder from leftover pool
    if remaining > 0:
        remaining_pool = np.setdiff1d(
            all_indices, np.array(list(selected), dtype=np.int64), assume_unique=True
        )
        extra_take = min(remaining, len(remaining_pool))
        extra = rng.choice(remaining_pool, size=extra_take, replace=False)
        selected.update(extra.tolist())

    final_indices = sorted(selected)
    if len(final_indices) > n_samples:
        final_indices = final_indices[:n_samples]

    return ds.select(final_indices)


# -------------------------
# Main pipeline
# -------------------------

def build_go_dataset() -> None:
    print(f"Loading dataset: {DATASET_NAME} ({SUBSET_NAME})...")
    raw = load_dataset(DATASET_NAME, SUBSET_NAME)

    print("Adding code_len column...")
    with_len = DatasetDict({
        split_name: add_code_len_column(split_ds)
        for split_name, split_ds in raw.items()
    })

    print(f"Filtering examples with code_len > {MAX_CODE_TOKENS}...")
    filtered = DatasetDict({
        split_name: filter_by_max_length(split_ds, MAX_CODE_TOKENS)
        for split_name, split_ds in with_len.items()
    })

    # ------------ TRAIN ------------
    print(f"Stratified sampling: train ({TRAIN_SAMPLES})...")
    sampled_train = stratified_sample_by_length(
        filtered["train"],
        n_samples=TRAIN_SAMPLES,
        num_bins=NUM_LENGTH_BINS,
        seed=RANDOM_SEED,
    )

    # ------------ VALIDATION ------------
    print(f"Stratified sampling: validation ({VAL_SAMPLES})...")
    sampled_val = stratified_sample_by_length(
        filtered["validation"],
        n_samples=VAL_SAMPLES,
        num_bins=NUM_LENGTH_BINS,
        seed=RANDOM_SEED,
    )

    # ------------ TEST ------------
    print(f"Stratified sampling: test ({TEST_SAMPLES})...")
    sampled_test = stratified_sample_by_length(
        filtered["test"],
        n_samples=TEST_SAMPLES,
        num_bins=NUM_LENGTH_BINS,
        seed=RANDOM_SEED,
    )

    # Drop helper column
    sampled_train = sampled_train.remove_columns(["code_len"])
    sampled_val = sampled_val.remove_columns(["code_len"])
    sampled_test = sampled_test.remove_columns(["code_len"])

    processed = DatasetDict(
        {
            "train": sampled_train,
            "validation": sampled_val,
            "test": sampled_test,
        }
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving processed dataset to {OUTPUT_DIR}...")
    processed.save_to_disk(OUTPUT_DIR)

    print("\nDone. Final dataset sizes:")
    for split_name, split_ds in processed.items():
        print(f"  {split_name}: {len(split_ds)} examples")


if __name__ == "__main__":
    build_go_dataset()
