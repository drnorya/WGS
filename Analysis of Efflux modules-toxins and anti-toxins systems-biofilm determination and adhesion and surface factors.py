#!/usr/bin/env python3
"""
functional_module_scan.py

Scan bacterial genomes/proteomes for:
- Efflux modules
- Toxin–antitoxin systems
- Biofilm determinants
- Adhesion / surface factors

Can work in 2 modes:
1) HMM-based (recommended) using hmmscan results
2) Keyword-based using an annotation table (e.g. Prokka or Bakta output)

USAGE EXAMPLES:

1) HMM-based (assuming you already ran hmmscan and have domtblout files):
    python functional_module_scan.py \
        --mode hmm \
        --hmmscan_dir hmmscan_results/ \
        --hmmscan_suffix .domtblout \
        --out_prefix my_project

2) Keyword-based (from an annotation table with at least: genome_id, gene_id, product):
    python functional_module_scan.py \
        --mode keyword \
        --annotation_table annotations.tsv \
        --out_prefix my_project

Author: Julius (https://julius.ai), Julius AI
"""

import os
import sys
import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

############################################
# 1. Category definitions / keyword rules  #
############################################

EFFLUX_KEYWORDS = [
    "efflux pump",
    "multidrug efflux",
    "rnd efflux",
    "smr family",
    "mfs transporter",
    "abc transporter permease",
    "acriflavine resistance protein",
    "mdt",
    "mex",      # Pseudomonas efflux
    "ade",      # Acinetobacter efflux
    "norA",
    "emr",
    "tet efflux",
]

TOXIN_KEYWORDS = [
    "toxin-antitoxin",
    "toxin component",
    "toxin protein",
    "relE",
    "mazF",
    "hicA",
    "yafQ",
    "hipA",
    "parE",
    "chpA",
    "endoRNase toxin",
    "vapC",
]

ANTITOXIN_KEYWORDS = [
    "antitoxin",
    "relB",
    "mazE",
    "hicB",
    "hipB",
    "parD",
    "chpB",
    "vapB",
]

BIOFILM_KEYWORDS = [
    "biofilm",
    "pel polysaccharide",
    "psl polysaccharide",
    "alginate biosynthesis",
    "curli",
    "csgA",
    "bcsA",  # cellulose synthesis
    "cellulose synthase",
    "exopolysaccharide",
    "eps biosynthesis",
    "extracellular polysaccharide",
]

ADHESION_KEYWORDS = [
    "adhesin",
    "adhesion",
    "fimbrial",
    "fimbriae",
    "pili",
    "pilus",
    "type i fimbrial",
    "type iv pilus",
    "autotransporter",
    "surface protein",
    "fibronectin-binding",
    "collagen-binding",
    "intimin",
    "invasin",
]


############################################
# 2. Utility functions                     #
############################################

def load_domtblout(domtblout_path, evalue_cutoff=1e-5, score_cutoff=30):
    """
    Parse HMMER domtblout file and return a pandas DataFrame.

    Filters by evalue and score.
    """
    records = []
    with open(domtblout_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 23:
                continue

            target_name = parts[0]
            query_name = parts[3]
            full_seq_evalue = float(parts[6])
            full_seq_score = float(parts[7])
            records.append({
                "target": target_name,
                "query": query_name,
                "evalue": full_seq_evalue,
                "score": full_seq_score
            })

    if not records:
        return pd.DataFrame(columns=["target", "query", "evalue", "score"])

    df_hits = pd.DataFrame(records)
    df_hits = df_hits[(df_hits["evalue"] <= evalue_cutoff) &
                      (df_hits["score"] >= score_cutoff)]
    return df_hits


def classify_hmm_hit(query_name):
    """
    Classify an HMM profile (query_name) into one of the categories.

    This depends on how you name your HMMs. Here we assume the HMM
    names contain indicative substrings like:
    efflux, toxin, antitoxin, biofilm, adhesion, fimbrial, pilus
    """
    q = query_name.lower()

    if "efflux" in q or "rnd" in q or "mfs" in q or "smr" in q:
        return "efflux_module"
    if "toxin" in q or "toxin_" in q or re.search(r"\b(relE|mazF|hicA|yafQ|hipA|vapC)\b", q):
        return "toxin"
    if "antitoxin" in q or re.search(r"\b(relB|mazE|hicB|hipB|vapB)\b", q):
        return "antitoxin"
    if "biofilm" in q or "pel_" in q or "psl_" in q or "alginate" in q or "curli" in q:
        return "biofilm_determinant"
    if "adhesin" in q or "adhesion" in q or "fimbr" in q or "pilus" in q or "pili" in q or "autotransporter" in q:
        return "adhesion_surface"
    return "other"


def keyword_match(text, keyword_list):
    """
    Case-insensitive keyword match. Returns True if any keyword is found.
    """
    if pd.isna(text):
        return False
    t = str(text).lower()
    for kw in keyword_list:
        if kw.lower() in t:
            return True
    return False


def classify_annotation_row(row):
    """
    Classify a gene based on annotation text using keyword lists.
    Expects a column 'product' or 'annotation' in the row.
    """
    product = None
    if "product" in row:
        product = row["product"]
    elif "annotation" in row:
        product = row["annotation"]
    else:
        return "other"

    if keyword_match(product, EFFLUX_KEYWORDS):
        return "efflux_module"
    if keyword_match(product, TOXIN_KEYWORDS):
        return "toxin"
    if keyword_match(product, ANTITOXIN_KEYWORDS):
        return "antitoxin"
    if keyword_match(product, BIOFILM_KEYWORDS):
        return "biofilm_determinant"
    if keyword_match(product, ADHESION_KEYWORDS):
        return "adhesion_surface"
    return "other"


############################################
# 3. HMM-based pipeline                    #
############################################

def run_hmm_mode(hmmscan_dir, hmmscan_suffix, out_prefix):
    """
    Collect hmmscan domtblout files from hmmscan_dir, classify hits,
    summarize per-genome and per-gene.
    """
    domtblout_files = [
        os.path.join(hmmscan_dir, f)
        for f in os.listdir(hmmscan_dir)
        if f.endswith(hmmscan_suffix)
    ]

    if not domtblout_files:
        print("No domtblout files found with suffix " + hmmscan_suffix + " in " + hmmscan_dir)
        sys.exit(1)

    all_hits = []

    for path in domtblout_files:
        genome_id = os.path.basename(path).replace(hmmscan_suffix, "")
        df = load_domtblout(path)
        if df.empty:
            continue
        df["genome_id"] = genome_id
        df["category"] = df["query"].apply(classify_hmm_hit)
        all_hits.append(df)

    if not all_hits:
        print("No significant HMM hits found across domtblout files")
        sys.exit(0)

    df_all = pd.concat(all_hits, ignore_index=True)

    df_all.to_csv(out_prefix + "_gene_level_hits.tsv", sep="\t", index=False)

    summary = (
        df_all[df_all["category"] != "other"]
        .groupby(["genome_id", "category"])["target"]
        .nunique()
        .reset_index()
        .pivot(index="genome_id", columns="category", values="target")
        .fillna(0)
        .astype(int)
    )

    summary.to_csv(out_prefix + "_module_summary.tsv", sep="\t")

    print("Gene-level hits table written to: " + out_prefix + "_gene_level_hits.tsv")
    print("Summary table written to: " + out_prefix + "_module_summary.tsv")

    make_plots(summary, out_prefix)


############################################
# 4. Keyword-based pipeline                #
############################################

def run_keyword_mode(annotation_table, out_prefix):
    """
    Keyword-based classification of genes from an annotation table.

    Expects at least:
        genome_id, gene_id, product
    or:
        genome_id, locus_tag, annotation
    """
    df = pd.read_csv(annotation_table, sep=None, engine="python")

    if "genome_id" not in df.columns:
        print("Error: annotation_table must contain a 'genome_id' column.")
        sys.exit(1)

    if "gene_id" not in df.columns and "locus_tag" not in df.columns:
        print("Error: annotation_table must contain 'gene_id' or 'locus_tag' column.")
        sys.exit(1)

    id_col = "gene_id" if "gene_id" in df.columns else "locus_tag"

    df["category"] = df.apply(classify_annotation_row, axis=1)

    out_cols = ["genome_id", id_col, "category"]
    if "product" in df.columns:
        out_cols.append("product")
    elif "annotation" in df.columns:
        out_cols.append("annotation")

    df[out_cols].to_csv(out_prefix + "_gene_level_hits.tsv", sep="\t", index=False)

    summary = (
        df[df["category"] != "other"]
        .groupby(["genome_id", "category"])[id_col]
        .nunique()
        .reset_index()
        .pivot(index="genome_id", columns="category", values=id_col)
        .fillna(0)
        .astype(int)
    )

    summary.to_csv(out_prefix + "_module_summary.tsv", sep="\t")

    print("Gene-level hits table written to: " + out_prefix + "_gene_level_hits.tsv")
    print("Summary table written to: " + out_prefix + "_module_summary.tsv")

    make_plots(summary, out_prefix)


############################################
# 5. Plotting                              #
############################################

def make_plots(summary_df, out_prefix):
    """
    Make simple barplots of module counts per genome.
    """
    categories = ["efflux_module", "toxin", "antitoxin",
                  "biofilm_determinant", "adhesion_surface"]
    for cat in categories:
        if cat not in summary_df.columns:
            summary_df[cat] = 0

    plt.figure(figsize=(10, 6))
    summary_df[categories].sum().plot(kind="bar")
    plt.ylabel("Total gene count")
    plt.title("Total counts per category across genomes")
    plt.tight_layout()
    out_path = out_prefix + "_total_counts_barplot.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved plot: " + out_path)

    plt.figure(figsize=(12, 6))
    summary_df[categories].plot(kind="bar", stacked=False)
    plt.ylabel("Gene count")
    plt.title("Module counts per genome")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path2 = out_prefix + "_per_genome_barplot.png"
    plt.savefig(out_path2, dpi=300)
    plt.close()
    print("Saved plot: " + out_path2)


############################################
# 6. Argument parser / main                #
############################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan genomes/proteomes for efflux modules, TA systems, biofilm and adhesion determinants."
    )

    parser.add_argument(
        "--mode",
        choices=["hmm", "keyword"],
        required=True,
        help="Analysis mode: 'hmm' = use HMMER domtblout; 'keyword' = annotation keyword search."
    )

    parser.add_argument(
        "--hmmscan_dir",
        help="Directory with hmmscan domtblout files (for mode=hmm)."
    )
    parser.add_argument(
        "--hmmscan_suffix",
        default=".domtblout",
        help="File suffix for domtblout files in hmmscan_dir (default: .domtblout)."
    )

    parser.add_argument(
        "--annotation_table",
        help="Annotation table file (TSV/CSV; for mode=keyword). Must contain 'genome_id' and 'gene_id' or 'locus_tag', plus 'product' or 'annotation'."
    )

    parser.add_argument(
        "--out_prefix",
        required=True,
        help="Prefix for output files (TSV tables and plots)."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "hmm":
        if args.hmmscan_dir is None:
            print("Error: --hmmscan_dir is required for mode=hmm.")
            sys.exit(1)
        run_hmm_mode(args.hmmscan_dir, args.hmmscan_suffix, args.out_prefix)

    elif args.mode == "keyword":
        if args.annotation_table is None:
            print("Error: --annotation_table is required for mode=keyword.")
            sys.exit(1)
        run_keyword_mode(args.annotation_table, args.out_prefix)


if __name__ == "__main__":
    main()