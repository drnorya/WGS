#!/usr/bin/env python3
"""
AMR and Virulence Factor Analysis

This script:
- Reads a table of gene-level annotations (AMR + virulence factors)
- Cleans and normalizes the data
- Produces:
  - Summary tables per sample and per group (AMR, Virulence, Other)
  - Top abundant genes and subcategories
  - Basic statistics and plots (if matplotlib/seaborn are installed)

Expected input columns (you can customize names below):
- sample_id       : Sample name / ID
- gene_id         : Gene / feature identifier
- category        : High-level category label (e.g. AMR, Virulence, Other)
- sub_category    : Finer category (e.g. beta_lactamase, efflux_pump, adhesion)
- abundance       : Numeric abundance (read counts, RPKM, TPM, coverage, etc.)

Usage example:

    python amr_virulence_analysis.py \
        --input annotations.csv \
        --output_prefix results/amr_virulence

Author: Julius (https://julius.ai)
Created by: Julius AI (AI assistants for data science)
"""

import argparse
import os
import sys

import pandas as pd

# Optional plotting imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze AMR genes and virulence factors from an annotation table"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV/TSV file with AMR and virulence annotations"
    )
    parser.add_argument(
        "--sep",
        default=None,
        help="Field separator: auto-detect by extension if not provided "
             "(.csv -> ',', .tsv/.txt -> '\\t')"
    )
    parser.add_argument(
        "--output_prefix",
        required=True,
        help="Prefix for output files (e.g. results/amr_virulence)"
    )
    parser.add_argument(
        "--min_abundance",
        type=float,
        default=0.0,
        help="Minimum abundance threshold to keep a gene/feature (default 0.0)"
    )
    parser.add_argument(
        "--amr_labels",
        nargs="+",
        default=["AMR", "Resistance", "Antibiotic_resistance"],
        help="Labels in 'category' column that represent AMR (default: AMR Resistance Antibiotic_resistance)"
    )
    parser.add_argument(
        "--virulence_labels",
        nargs="+",
        default=["Virulence", "Virulence_factor", "VF"],
        help="Labels in 'category' column that represent virulence factors "
             "(default: Virulence Virulence_factor VF)"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of top genes/features to report and plot (default 20)"
    )

    return parser.parse_args()


def auto_sep(filepath, user_sep):
    if user_sep is not None:
        return user_sep
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".tsv", ".txt"]:
        return "\t"
    return ","


def load_data(filepath, sep):
    if not os.path.exists(filepath):
        print("ERROR: Input file not found: " + filepath)
        sys.exit(1)
    try:
        df = pd.read_csv(filepath, sep=sep)
    except Exception as e:
        print("ERROR: Could not read input file: " + str(e))
        sys.exit(1)
    if df.empty:
        print("ERROR: Input file is empty.")
        sys.exit(1)
    return df


def check_required_columns(df, col_config):
    missing = []
    for logical_name, col in col_config.items():
        if logical_name == "sub_category":
            # sub_category is optional; skip strict checking
            continue
        if col not in df.columns:
            missing.append(col)
    if missing:
        print("ERROR: Missing required columns: " + ", ".join(missing))
        print("Available columns: " + ", ".join(df.columns.astype(str)))
        sys.exit(1)


def preprocess(df, col_config, min_abundance):
    abundance_col = col_config["abundance"]

    # Coerce abundance to numeric
    df[abundance_col] = pd.to_numeric(df[abundance_col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[abundance_col])
    df = df[df[abundance_col] >= min_abundance]
    after = len(df)

    print("Kept " + str(after) + " of " + str(before) + " rows after abundance filtering.")

    # Normalize text columns (strip spaces)
    for key in ["sample", "gene", "category", "sub_category"]:
        col = col_config.get(key)
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def label_groups(df, col_config, amr_labels, virulence_labels):
    category_col = col_config["category"]

    amr_set = set([x.lower() for x in amr_labels])
    vir_set = set([x.lower() for x in virulence_labels])

    def classify(cat):
        c = str(cat).lower().strip()
        if c in amr_set:
            return "AMR"
        if c in vir_set:
            return "Virulence"
        return "Other"

    df["group"] = df[category_col].apply(classify)
    return df


def summarize_by_sample_group(df, col_config):
    sample_col = col_config["sample"]
    abundance_col = col_config["abundance"]

    # Total abundance per sample/group
    grp = df.groupby([sample_col, "group"], as_index=False)[abundance_col].sum()
    grp = grp.rename(columns={abundance_col: "total_abundance"})

    # Pivot to wide format
    pivot = grp.pivot(index=sample_col, columns="group", values="total_abundance").fillna(0)
    pivot = pivot.reset_index()

    # Also compute relative abundance per sample
    group_cols = [c for c in pivot.columns if c not in [sample_col]]
    pivot["total_all_groups"] = pivot[group_cols].sum(axis=1)
    for c in group_cols:
        pivot[c + "_rel"] = pivot[c] / pivot["total_all_groups"].replace(0, float(""))

    return pivot


def summarize_by_feature(df, col_config):
    gene_col = col_config["gene"]
    subcat_col = col_config.get("sub_category")
    abundance_col = col_config["abundance"]
    sample_col = col_config["sample"]

    # Gene-level summary
    gene_summary = (
        df.groupby(["group", gene_col], as_index=False)
        .agg(
            total_abundance=(abundance_col, "sum"),
            mean_abundance=(abundance_col, "mean"),
            n_samples=(sample_col, "nunique")
        )
    )

    # Sub-category summary (if available)
    if subcat_col in df.columns:
        subcat_summary = (
            df.groupby(["group", subcat_col], as_index=False)
            .agg(
                total_abundance=(abundance_col, "sum"),
                mean_abundance=(abundance_col, "mean"),
                n_genes=(gene_col, "nunique"),
                n_samples=(sample_col, "nunique")
            )
        )
    else:
        subcat_summary = None

    return gene_summary, subcat_summary


def get_top_features(gene_summary, top_n):
    # Within each group, get top N genes by total_abundance
    top_list = []
    for group_name, sub in gene_summary.groupby("group"):
        sub_sorted = sub.sort_values("total_abundance", ascending=False).head(top_n)
        top_list.append(sub_sorted)
    if not top_list:
        return gene_summary.iloc[0:0].copy()
    top_features = pd.concat(top_list, ignore_index=True)
    return top_features


def save_tables(df_sample_group, gene_summary, subcat_summary, top_features, output_prefix):
    base_dir = os.path.dirname(output_prefix)
    if base_dir and not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    # Sample x group summary
    sample_out = output_prefix + ".sample_group_summary.csv"
    df_sample_group.to_csv(sample_out, index=False)

    # Gene-level summary
    gene_out = output_prefix + ".gene_summary.csv"
    gene_summary.to_csv(gene_out, index=False)

    # Sub-category summary
    if subcat_summary is not None:
        subcat_out = output_prefix + ".subcategory_summary.csv"
        subcat_summary.to_csv(subcat_out, index=False)

    # Top features
    top_out = output_prefix + ".top_features.csv"
    top_features.to_csv(top_out, index=False)


def make_plots(df_sample_group, top_features, col_config, output_prefix):
    if not PLOTTING_AVAILABLE:
        print("matplotlib/seaborn not available; skipping plots.")
        return

    sample_col = col_config["sample"]

    # Plot 1: Stacked bar of AMR vs Virulence vs Other per sample
    try:
        group_cols = [c for c in df_sample_group.columns
                      if c not in [sample_col, "total_all_groups"]
                      and not c.endswith("_rel")]

        df_melt = df_sample_group.melt(
            id_vars=[sample_col],
            value_vars=group_cols,
            var_name="group",
            value_name="total_abundance"
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df_melt,
            x=sample_col,
            y="total_abundance",
            hue="group"
        )
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Sample")
        plt.ylabel("Total abundance")
        plt.title("AMR vs Virulence vs Other abundance per sample")
        plt.tight_layout()
        plt.savefig(output_prefix + ".sample_group_stacked.png", dpi=300)
        plt.close()
    except Exception as e:
        print("Could not create sample_group_stacked plot: " + str(e))

    # Plot 2: Top features by total abundance within each group
    try:
        if top_features.empty:
            print("No top features to plot.")
            return

        plt.figure(figsize=(10, 8))
        # Create a combined label for plotting
        gene_col = col_config["gene"]
        top_features["gene_label"] = top_features["group"] + " | " + top_features[gene_col].astype(str)
        top_features_sorted = top_features.sort_values("total_abundance", ascending=True)

        sns.barplot(
            data=top_features_sorted,
            x="total_abundance",
            y="gene_label",
            hue="group",
            dodge=False
        )
        plt.xlabel("Total abundance")
        plt.ylabel("Gene (by group)")
        plt.title("Top features by total abundance within each group")
        plt.tight_layout()
        plt.savefig(output_prefix + ".top_features.png", dpi=300)
        plt.close()
    except Exception as e:
        print("Could not create top_features plot: " + str(e))


def main():
    args = parse_args()

    sep = auto_sep(args.input, args.sep)
    df = load_data(args.input, sep)

    # Customize these names if your columns differ
    col_config = {
        "sample": "sample_id",
        "gene": "gene_id",
        "category": "category",
        "sub_category": "sub_category",  # optional
        "abundance": "abundance",
    }

    check_required_columns(df, col_config)

    df = preprocess(df, col_config, args.min_abundance)
    df = label_groups(df, col_config, args.amr_labels, args.virulence_labels)

    df_sample_group = summarize_by_sample_group(df, col_config)
    gene_summary, subcat_summary = summarize_by_feature(df, col_config)
    top_features = get_top_features(gene_summary, args.top_n)

    save_tables(df_sample_group, gene_summary, subcat_summary, top_features, args.output_prefix)
    make_plots(df_sample_group, top_features, col_config, args.output_prefix)

    print("Analysis completed.")
    print("Main outputs:")
    print("  " + args.output_prefix + ".sample_group_summary.csv")
    print("  " + args.output_prefix + ".gene_summary.csv")
    if subcat_summary is not None:
        print("  " + args.output_prefix + ".subcategory_summary.csv")
    print("  " + args.output_prefix + ".top_features.csv")
    if PLOTTING_AVAILABLE:
        print("Plots saved with prefix: " + args.output_prefix)


if __name__ == "__main__":
    main()