#!/usr/bin/env python3
# quality_analysis.py
#
# Generic script for:
# - Completeness analysis
# - Confidence tiering
# - Hallmark density
#
# Usage example:
#   python quality_analysis.py \
#       --input data.csv \
#       --output data_with_metrics.csv \
#       --id-column id \
#       --hallmarks col_a col_b col_c
#
# Requires: pandas, numpy

import argparse
import os
import sys

import numpy as np
import pandas as pd


def infer_file_type(path):
    lower = path.lower()
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".tsv") or lower.endswith(".tab"):
        return "tsv"
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return "excel"
    raise ValueError("Unsupported file extension for " + path)


def read_input(path):
    file_type = infer_file_type(path)
    if file_type == "csv":
        df = pd.read_csv(path)
    elif file_type == "tsv":
        df = pd.read_csv(path, sep="\t")
    elif file_type == "excel":
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file type " + file_type)
    return df


def write_output(df, path):
    file_type = infer_file_type(path)
    if file_type == "csv":
        df.to_csv(path, index=False)
    elif file_type == "tsv":
        df.to_csv(path, sep="\t", index=False)
    elif file_type == "excel":
        df.to_excel(path, index=False)
    else:
        raise ValueError("Unsupported file type " + file_type)


def compute_completeness(df, id_column):
    # All non-ID columns are attributes used for completeness
    attribute_cols = [c for c in df.columns if c != id_column]

    # Define non-null / non-empty check
    df_attributes = df[attribute_cols].copy()

    # Convert empty strings and whitespace-only strings to
    def to_nan_if_empty(x):
        if isinstance(x, str) and x.strip() == "":
            return np.
        return x

    df_attributes = df_attributes.applymap(to_nan_if_empty)

    non_null_matrix = ~df_attributes.isna()

    completeness_count = non_null_matrix.sum(axis=1)
    total_attributes = len(attribute_cols)

    # Avoid divide-by-zero (e.g., if there are no attribute columns)
    if total_attributes == 0:
        completeness_ratio = pd.Series(np., index=df.index)
    else:
        completeness_ratio = completeness_count / float(total_attributes)

    df["completeness_count"] = completeness_count
    df["completeness_ratio"] = completeness_ratio

    return df, attribute_cols


def assign_confidence_tier(df, high_threshold=0.8, medium_threshold=0.5):
    # Assumes completeness_ratio is already present
    def tier_from_ratio(r):
        if pd.isna(r):
            return "Unknown"
        if r >= high_threshold:
            return "High"
        if r >= medium_threshold:
            return "Medium"
        return "Low"

    df["confidence_tier"] = df["completeness_ratio"].apply(tier_from_ratio)
    return df


def compute_hallmark_density(df, hallmark_cols):
    # Handle case where no hallmarks are specified
    if hallmark_cols is None or len(hallmark_cols) == 0:
        df["hallmark_count"] = 0
        df["hallmark_ratio"] = np.
        return df

    existing_hallmarks = [c for c in hallmark_cols if c in df.columns]

    if len(existing_hallmarks) == 0:
        df["hallmark_count"] = 0
        df["hallmark_ratio"] = np.
        return df

    df_h = df[existing_hallmarks].copy()

    def to_nan_if_empty(x):
        if isinstance(x, str) and x.strip() == "":
            return np.
        return x

    df_h = df_h.applymap(to_nan_if_empty)
    non_null_h = ~df_h.isna()

    hallmark_count = non_null_h.sum(axis=1)
    total_hallmarks = len(existing_hallmarks)

    hallmark_ratio = hallmark_count / float(total_hallmarks)

    df["hallmark_count"] = hallmark_count
    df["hallmark_ratio"] = hallmark_ratio

    return df


def summarize(df, id_column, attribute_cols, hallmark_cols):
    print("=== Basic Info ===")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("ID column:", id_column)
    print("Attribute columns (used for completeness):", len(attribute_cols))

    if hallmark_cols is None or len(hallmark_cols) == 0:
        print("Hallmark columns: None specified")
    else:
        existing_hallmarks = [c for c in hallmark_cols if c in df.columns]
        missing_hallmarks = [c for c in hallmark_cols if c not in df.columns]
        print("Hallmark columns (requested):", hallmark_cols)
        print("Hallmark columns (found):", existing_hallmarks)
        if missing_hallmarks:
            print("Hallmark columns (missing in data):", missing_hallmarks)

    print("\n=== Completeness Summary ===")
    print(df["completeness_ratio"].describe())

    print("\n=== Confidence Tier Distribution ===")
    print(df["confidence_tier"].value_counts(dropna=False))

    print("\n=== Hallmark Density Summary ===")
    if "hallmark_ratio" in df.columns:
        print(df["hallmark_ratio"].describe())
    else:
        print("Hallmark metrics not computed (no hallmarks provided).")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute completeness, confidence tiering, and hallmark density for a tabular dataset."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input file path (csv, tsv, xls, xlsx).",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output file path (csv, tsv, xls, xlsx).",
    )

    parser.add_argument(
        "--id-column",
        default="id",
        help="Name of the ID column (default: id).",
    )

    parser.add_argument(
        "--hallmarks",
        nargs="*",
        default=None,
        help="List of hallmark columns (space-separated). Example: --hallmarks col_a col_b",
    )

    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.8,
        help="Minimum completeness_ratio for High confidence (default: 0.8).",
    )

    parser.add_argument(
        "--medium-threshold",
        type=float,
        default=0.5,
        help="Minimum completeness_ratio for Medium confidence (default: 0.5). Below this is Low.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print("Input file not found: " + args.input, file=sys.stderr)
        sys.exit(1)

    df = read_input(args.input)

    if args.id_column not in df.columns:
        print("ID column " + args.id_column + " not found in input.", file=sys.stderr)
        print("Available columns:", list(df.columns), file=sys.stderr)
        sys.exit(1)

    df, attribute_cols = compute_completeness(df, args.id_column)
    df = assign_confidence_tier(
        df,
        high_threshold=args.high_threshold,
        medium_threshold=args.medium_threshold,
    )
    df = compute_hallmark_density(df, args.hallmarks)

    summarize(df, args.id_column, attribute_cols, args.hallmarks)

    write_output(df, args.output)
    print("\nProcessed data written to:", args.output)


if __name__ == "__main__":
    main()