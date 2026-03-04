#!/usr/bin/env python3

# Prophage mobility, indelibility, and induction potential analysis
# Ready-to-use script. Reads prophage region and gene tables, applies
# keyword-based heuristics, and outputs a summary table.

import argparse
import sys
import pandas as pd
import numpy as np
from collections import defaultdict


def read_table(path):
    if path is None:
        return None
    if path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        # default: TSV
        return pd.read_csv(path, sep="\t")


def normalize_cols(df):
    # Lowercase and strip column names for robustness
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def contains_any(text, keywords):
    if not isinstance(text, str):
        return False
    text_low = text.lower()
    for kw in keywords:
        if kw in text_low:
            return True
    return False


def count_keywords_in_group(products, keywords):
    if products is None or len(products) == 0:
        return 0
    count = 0
    for prod in products:
        if contains_any(prod, keywords):
            count += 1
    return count


def classify_mobility(products):
    # Keyword groups
    integrase_kw = [
        "integrase",
        "recombinase",
        "xerC".lower(),
        "xerD".lower(),
        "tyrosine recombinase",
        "serine recombinase",
        "excisionase",
        "attp",
        "attb"
    ]
    conjugation_kw = [
        "conjugative",
        "tra",
        "trb",
        "type iv secretion",
        "t4ss",
        "pilus",
        "type 4 secretion"
    ]
    structural_kw = [
        "capsid",
        "head",
        "tail",
        "portal",
        "baseplate",
        "scaffold",
        "scaffolding",
        "virion"
    ]
    lytic_kw = [
        "holin",
        "endolysin",
        "lysin",
        "lysis",
        "spanin"
    ]

    n_integrase = count_keywords_in_group(products, integrase_kw)
    n_conj = count_keywords_in_group(products, conjugation_kw)
    n_struct = count_keywords_in_group(products, structural_kw)
    n_lytic = count_keywords_in_group(products, lytic_kw)

    n_phage_like = n_struct + n_lytic

    # Simple heuristic rules
    if n_integrase > 0 and n_phage_like > 0:
        return "integrative_prophage"
    if n_integrase == 0 and (n_struct >= 2 or n_conj > 0 or n_phage_like >= 3):
        return "putatively_mobile"
    if n_phage_like <= 1 and n_integrase == 0:
        return "defective_or_remnant"
    return "uncertain"


def classify_indelibility(left_prod, right_prod):
    if left_prod is None and right_prod is None:
        return "unknown"

    tRNA_kw = ["trna", "tmrna", "ssra"]
    conserved_kw = [
        "dna gyrase",
        "topoisomerase",
        "rna polymerase",
        "ribosomal protein",
        "ribosome",
        "elongation factor",
        "recA".lower(),
        "dna polymerase"
    ]
    mobile_kw = [
        "transposase",
        "insertion sequence",
        "is element",
        "integron",
        "mobile element",
        "resolvase",
        "insertion sequence"
    ]
    hyp_kw = ["hypothetical protein"]

    texts = []
    if isinstance(left_prod, str):
        texts.append(left_prod)
    if isinstance(right_prod, str):
        texts.append(right_prod)

    if len(texts) == 0:
        return "unknown"

    has_trna = any(contains_any(t, tRNA_kw) for t in texts)
    has_conserved = any(contains_any(t, conserved_kw) for t in texts)
    has_mobile = any(contains_any(t, mobile_kw) for t in texts)
    mostly_hyp = all(contains_any(t, hyp_kw) for t in texts)

    if has_trna or has_conserved:
        return "indelible_like"
    if has_mobile or mostly_hyp:
        return "variable_site"
    return "unknown"


def classify_induction_potential(products, mobility_class):
    if products is None or len(products) == 0:
        return "unknown"

    repressor_kw = [
        "repressor",
        "ci repressor",
        "cro",
        "cro-like",
        "antirepressor",
        "anti-repressor",
        "lysogenic",
        "transcriptional regulator"
    ]
    lytic_kw = [
        "holin",
        "endolysin",
        "lysin",
        "lysis",
        "spanin"
    ]
    structural_kw = [
        "capsid",
        "head",
        "tail",
        "portal",
        "baseplate",
        "scaffold",
        "scaffolding",
        "virion"
    ]

    n_rep = count_keywords_in_group(products, repressor_kw)
    n_lytic = count_keywords_in_group(products, lytic_kw)
    n_struct = count_keywords_in_group(products, structural_kw)

    if mobility_class == "defective_or_remnant":
        return "low"

    # high: integrative plus repressor plus lytic plus some structure
    if (mobility_class == "integrative_prophage" and
        n_rep > 0 and n_lytic > 0 and n_struct > 0):
        return "high"

    # medium: some lytic and some structure or repressor
    if (n_lytic > 0 and n_struct > 0) or (n_rep > 0 and (n_lytic > 0 or n_struct > 0)):
        return "medium"

    # low: few phage-like genes
    if n_lytic == 0 and n_struct == 0 and n_rep == 0:
        return "low"

    return "unknown"


def summarize_prophage(prophage_df, gene_df, flank_df=None):
    prophage_df = normalize_cols(prophage_df)
    gene_df = normalize_cols(gene_df)
    if flank_df is not None:
        flank_df = normalize_cols(flank_df)

    required_p_cols = ["genome_id", "prophage_id", "contig_id", "start", "end"]
    required_g_cols = ["genome_id", "prophage_id", "gene_id", "start", "end", "product"]

    for col in required_p_cols:
        if col not in prophage_df.columns:
            raise ValueError("Missing column in prophage table: " + col)

    for col in required_g_cols:
        if col not in gene_df.columns:
            raise ValueError("Missing column in gene table: " + col)

    # Optional columns for flank_df
    if flank_df is not None:
        flank_required = [
            "genome_id",
            "prophage_id",
            "left_flank_gene",
            "right_flank_gene",
            "left_flank_product",
            "right_flank_product"
        ]
        for col in flank_required:
            if col not in flank_df.columns:
                raise ValueError("Missing column in flank table: " + col)

    results = []
    grouped = gene_df.groupby(["genome_id", "prophage_id"])

    for (genome_id, prophage_id), group in grouped:
        products = list(group["product"].astype(str).fillna(""))

        mobility_class = classify_mobility(products)

        # Flanking gene info if available
        left_prod = None
        right_prod = None
        indelibility_class = "unknown"
        if flank_df is not None:
            sub = flank_df[
                (flank_df["genome_id"] == genome_id) &
                (flank_df["prophage_id"] == prophage_id)
            ]
            if len(sub) > 0:
                row = sub.iloc[0]
                left_prod = row.get("left_flank_product", None)
                right_prod = row.get("right_flank_product", None)
                indelibility_class = classify_indelibility(left_prod, right_prod)

        induction_potential = classify_induction_potential(products, mobility_class)

        # Prophage coordinates
        psub = prophage_df[
            (prophage_df["genome_id"] == genome_id) &
            (prophage_df["prophage_id"] == prophage_id)
        ]
        if len(psub) == 0:
            # If missing, still record with NaNs
            contig_id = np.
            start = np.
            end = np.
            caller = np.
        else:
            prow = psub.iloc[0]
            contig_id = prow.get("contig_id", np.)
            start = prow.get("start", np.)
            end = prow.get("end", np.)
            caller = prow.get("caller", np.)

        res = {
            "genome_id": genome_id,
            "prophage_id": prophage_id,
            "contig_id": contig_id,
            "start": start,
            "end": end,
            "caller": caller,
            "mobility_class": mobility_class,
            "indelibility_class": indelibility_class,
            "induction_potential": induction_potential,
            "n_genes": len(group),
        }
        results.append(res)

    res_df = pd.DataFrame(results)
    return res_df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze prophage mobility, indelibility, and induction potential "
                    "from prophage and gene annotation tables."
    )
    parser.add_argument(
        "--prophage-table",
        required=True,
        help="TSV/CSV with prophage regions (columns: genome_id, prophage_id, contig_id, start, end, [caller])"
    )
    parser.add_argument(
        "--gene-table",
        required=True,
        help="TSV/CSV with genes inside prophages (columns: genome_id, prophage_id, gene_id, start, end, product)"
    )
    parser.add_argument(
        "--flank-table",
        required=False,
        default=None,
        help="Optional TSV/CSV with flanking host genes "
             "(columns: genome_id, prophage_id, left_flank_gene, right_flank_gene, "
             "left_flank_product, right_flank_product)"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output TSV file with summarized prophage classification"
    )

    args = parser.parse_args()

    try:
        prophage_df = read_table(args.prophage_table)
        gene_df = read_table(args.gene_table)
        flank_df = read_table(args.flank_table) if args.flank_table is not None else None

        res_df = summarize_prophage(prophage_df, gene_df, flank_df)

        res_df.to_csv(args.out, sep="\t", index=False)
    except Exception as e:
        sys.stderr.write("Error: " + str(e) + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()