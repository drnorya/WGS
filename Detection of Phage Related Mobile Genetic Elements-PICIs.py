#!/usr/bin/env python3
# PICI / phage hallmark analysis pipeline
# - parse GenBank hallmark regions
# - flag hallmark genes from product annotations
# - call PICI-like vs other
# - export tables + summary figures

import os
import sys
import glob

import pandas as pd
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns


###############################################
# 0. CONFIGURATION
###############################################

# Folder containing your hallmark-region files
INPUT_DIR = "input_regions"   # change if needed

# Output folder for tables and figures
OUT_DIR = "pici_outputs"

os.makedirs(OUT_DIR, exist_ok=True)


###############################################
# 1. HELPER FUNCTIONS
###############################################

def is_gbk(path):
    lower = path.lower()
    return lower.endswith(".gbk") or lower.endswith(".gb") or lower.endswith(".genbank")


def is_fasta(path):
    lower = path.lower()
    return lower.endswith(".fasta") or lower.endswith(".fa") or lower.endswith(".fna")


def extract_cds_from_gbk(gbk_path):
    """
    Parse a GenBank file of a hallmark region and return a list of CDS records
    as dicts (one CDS per row).
    """
    records_out = []
    file_id = os.path.basename(gbk_path)

    for rec in SeqIO.parse(gbk_path, "genbank"):
        for feat in rec.features:
            if feat.type != "CDS":
                continue

            start = int(feat.location.start)
            end = int(feat.location.end)
            strand = int(feat.location.strand) if feat.location.strand is not None else 1

            # Qualifiers are lists; get first if exists
            qualifiers = feat.qualifiers
            product = qualifiers.get("product", [""])[0]
            gene = qualifiers.get("gene", [""])[0]
            locus_tag = qualifiers.get("locus_tag", [""])[0]
            protein_id = qualifiers.get("protein_id", [""])[0]

            # Use translation length if present, else infer
            aa_len = None
            if "translation" in qualifiers:
                seq_aa = qualifiers["translation"][0]
                aa_len = len(seq_aa)
            else:
                # approximate from nucleotide length
                aa_len = max(1, abs(end - start) // 3)

            if not protein_id:
                protein_id = file_id + "|" + rec.id + "|" + "CDS_" + str(start) + "_" + str(end)

            records_out.append(
                dict(
                    file=file_id,
                    contig=rec.id,
                    protein_id=protein_id,
                    start=start,
                    end=end,
                    strand=strand,
                    gene=gene if gene else None,
                    locus_tag=locus_tag if locus_tag else None,
                    product=product,
                    aa_len=aa_len,
                )
            )

    return records_out


def flag_hallmarks(df):
    """
    Add boolean hallmark columns based on 'product' annotation.
    """
    def has_any(s, terms):
        s = str(s).lower()
        for t in terms:
            if t in s:
                return True
        return False

    df = df.copy()

    df["kw_Integrase"] = df["product"].apply(
        lambda s: has_any(
            s,
            ["integrase", "tyrosine recombinase", "site-specific recombinase"]
        )
    )

    df["kw_Primase_helicase"] = df["product"].apply(
        lambda s: has_any(
            s,
            ["p4 family primase", "p4-family primase", "phage plasmid primase, p4", "p4 family", "primase-helicase", "primase helicase"]
        )
    )

    df["kw_Terminase_small"] = df["product"].apply(
        lambda s: has_any(
            s,
            ["terminase small", "small terminase"]
        )
    )

    df["kw_Terminase_large"] = df["product"].apply(
        lambda s: has_any(
            s,
            ["terminase large", "large terminase"]
        )
    )

    df["kw_Portal"] = df["product"].apply(
        lambda s: has_any(
            s,
            ["portal protein", "portal vertex", "portal protein gp"]
        )
    )

    df["kw_Capsid"] = df["product"].apply(
        lambda s: has_any(
            s,
            ["capsid protein", "major capsid", "minor capsid", "head protein"]
        )
    )

    df["kw_Repressor_regulator"] = df["product"].apply(
        lambda s: has_any(
            s,
            ["repressor", "lysogeny regulator", "cro-like", "transcriptional regulator"]
        )
    )

    return df


def summarize_by_region(df):
    """
    Region-level summary table from per-CDS hallmark flags.
    """
    grp = df.groupby("file")

    sum_rows = []
    for file_id, sub in grp:
        has_int = bool(sub["kw_Integrase"].any())
        has_rep = bool(sub["kw_Primase_helicase"].any())
        has_pack = bool(sub["kw_Terminase_small"].any())
        has_terL = bool(sub["kw_Terminase_large"].any())
        has_struct = bool(sub["kw_Portal"].any() or sub["kw_Capsid"].any())

        n_prot = sub.shape[0]

        # Simple, conservative rule:
        # PICI-like: integrase + P4 primase/helicase + TerS present
        # Anything else = other (or prophage-like if you want to extend later)
        if has_int and has_rep and has_pack:
            call = "PICI-like (Integrase + P4-primase/helicase + TerS)"
        else:
            call = "other"

        sum_rows.append(
            dict(
                file=file_id,
                n_proteins=n_prot,
                has_integrase=has_int,
                has_rep=has_rep,
                has_packaging=has_pack,
                has_terL=has_terL,
                has_struct=has_struct,
                call=call,
            )
        )

    return pd.DataFrame(sum_rows)


###############################################
# 2. LOAD INPUT FILES AND PARSE GENBANK
###############################################

def main():
    gbk_files = []
    fasta_files = []

    if not os.path.isdir(INPUT_DIR):
        print("Input directory does not exist: " + INPUT_DIR)
        sys.exit(1)

    for path in glob.glob(os.path.join(INPUT_DIR, "*")):
        if is_gbk(path):
            gbk_files.append(path)
        elif is_fasta(path):
            fasta_files.append(path)

    if len(gbk_files) == 0:
        print("No GenBank files (.gbk/.gb) found in " + INPUT_DIR)
        sys.exit(1)

    print("Found GenBank files:")
    for p in gbk_files:
        print("  " + p)

    # Parse all GenBank CDS
    all_cds_records = []
    for gbk_path in gbk_files:
        cds_list = extract_cds_from_gbk(gbk_path)
        all_cds_records.extend(cds_list)

    df_cds = pd.DataFrame(all_cds_records)
    if df_cds.shape[0] == 0:
        print("No CDS records found in GenBank files.")
        sys.exit(1)

    # Add hallmark flags
    df_cds = flag_hallmarks(df_cds)

    ###############################################
    # 3. REGION-LEVEL SUMMARY AND CLASSIFICATION
    ###############################################

    df_regions = summarize_by_region(df_cds)

    # Save core tables
    cds_out = os.path.join(OUT_DIR, "domain_signal_motif_keyword_table.csv")
    reg_out = os.path.join(OUT_DIR, "hallmark_region_calls.csv")

    df_cds.to_csv(cds_out, index=False)
    df_regions.to_csv(reg_out, index=False)

    print("Wrote per-CDS table to: " + cds_out)
    print("Wrote region summary table to: " + reg_out)

    ###############################################
    # 4. MANUSCRIPT-STYLE TABLES
    ###############################################

    # Table 1: region-level hallmark summary
    cols_keep = [
        "file",
        "n_proteins",
        "has_integrase",
        "has_rep",
        "has_packaging",
        "has_terL",
        "has_struct",
        "call",
    ]
    table1 = df_regions[cols_keep].copy()
    table1 = table1.sort_values(["call", "file"])
    t1_path = os.path.join(OUT_DIR, "Table1_PICI_hallmark_summary_by_region.csv")
    table1.to_csv(t1_path, index=False)
    print("Wrote Table 1 to: " + t1_path)

    # Table 2: per-gene evidence for PICI-like regions
    pici_files = df_regions[df_regions["call"].str.contains("PICI-like", na=False)]["file"].tolist()
    df_cds_pici = df_cds[df_cds["file"].isin(pici_files)].copy()

    flag_cols = [c for c in df_cds_pici.columns if c.startswith("kw_")]

    show_cols = [
        "file",
        "protein_id",
        "start",
        "end",
        "strand",
        "gene",
        "locus_tag",
        "product",
        "aa_len",
    ] + flag_cols

    show_cols = [c for c in show_cols if c in df_cds_pici.columns]
    table2 = df_cds_pici[show_cols].sort_values(["file", "start"])
    t2_path = os.path.join(OUT_DIR, "Table2_PICI_like_regions_per_gene_evidence.csv")
    table2.to_csv(t2_path, index=False)
    print("Wrote Table 2 to: " + t2_path)

    ###############################################
    # 5. FIGURES: HEATMAP + KEYWORD COUNTS
    ###############################################

    # Figure 1: presence/absence heatmap
    heat_cols = [
        "has_integrase",
        "has_rep",
        "has_packaging",
        "has_terL",
        "has_struct",
    ]

    heat_df = df_regions[["file"] + heat_cols].set_index("file").astype(int)

    plt.figure(figsize=(7, 0.5 + 0.4 * len(heat_df)))
    sns.heatmap(
        heat_df,
        annot=True,
        cbar=False,
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        fmt="d",
    )
    plt.title("Hallmark presence/absence by region")
    plt.ylabel("Region")
    plt.xlabel("Hallmark")
    plt.tight_layout()
    f1_path = os.path.join(OUT_DIR, "Figure1_PICI_hallmark_heatmap.png")
    plt.savefig(f1_path, dpi=300)
    plt.close()
    print("Wrote Figure 1 to: " + f1_path)

    # Figure 2: keyword counts by region
    kw_cols = [c for c in df_cds.columns if c.startswith("kw_")]
    kw_counts = df_cds.groupby("file")[kw_cols].sum().reset_index()
    kw_counts = kw_counts.sort_values("file")

    plt.figure(figsize=(9, 3.0))
    for c in kw_cols:
        plt.plot(
            kw_counts["file"],
            kw_counts[c],
            marker="o",
            label=c.replace("kw_", ""),
        )
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("CDS count")
    plt.title("Keyword-based hallmark counts by region")
    plt.legend(fontsize=7, ncol=3, frameon=False)
    plt.tight_layout()
    f2_path = os.path.join(OUT_DIR, "Figure2_keyword_counts_by_region.png")
    plt.savefig(f2_path, dpi=300)
    plt.close()
    print("Wrote Figure 2 to: " + f2_path)

    print("Done.")


if __name__ == "__main__":
    main()