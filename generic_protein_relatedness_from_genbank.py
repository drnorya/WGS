#!/usr/bin/env python3
"""Generic protein relatedness analysis from GenBank files.

This script:
- Accepts one or more GenBank files as input
- Scans CDS features for user-configurable keyword patterns (e.g. integrase, recombinase)
- Extracts matching protein sequences and writes a FASTA and a metadata table
- Computes a pairwise global identity matrix between all extracted proteins
- Converts identity to a distance matrix (1 - identity)
- Performs average-linkage hierarchical clustering
- Exports a Newick dendrogram tree
- Generates a heatmap and dendrogram figure

By default it searches for integrase/recombinase-like proteins, but
keywords can be changed via the command line.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from Bio import SeqIO
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from scipy.spatial.distance import squareform


def feature_matches_keywords(feature, keywords):
    """Return True if a CDS feature contains any of the given keywords."""
    if feature.type != "CDS":
        return False

    text_fields = []
    for key in ["product", "gene", "note", "function", "inference", "db_xref"]:
        vals = feature.qualifiers.get(key, [])
        for v in vals:
            text_fields.append(str(v).lower())

    joined = " ".join(text_fields)
    for kw in keywords:
        if kw.lower() in joined:
            return True
    return False


def extract_hits_and_fasta(genbank_files, keywords, output_dir):
    """Parse GenBank files, collect CDS features matching keywords, write table + FASTA."""
    records = []
    fasta_lines = []

    for gb_path in genbank_files:
        gb_path = Path(gb_path)
        if not gb_path.exists():
            print("Warning: missing GenBank file " + str(gb_path), file=sys.stderr)
            continue

        for rec in SeqIO.parse(str(gb_path), "genbank"):
            for feat in rec.features:
                if not feature_matches_keywords(feat, keywords):
                    continue
                if "translation" not in feat.qualifiers:
                    continue

                prot_seq = feat.qualifiers["translation"][0]
                start = int(feat.location.start) + 1
                end = int(feat.location.end)
                strand = feat.location.strand

                product = ";".join(feat.qualifiers.get("product", [""]))
                gene = ";".join(feat.qualifiers.get("gene", [""]))
                locus_tag = ";".join(feat.qualifiers.get("locus_tag", [""]))

                rec_id = rec.id
                region_id = gb_path.stem

                label_parts = [region_id, rec_id]
                if locus_tag:
                    label_parts.append(locus_tag)
                elif gene:
                    label_parts.append(gene)
                else:
                    label_parts.append("CDS" + str(start) + "_" + str(end))
                seq_id = "|".join(label_parts)

                records.append({
                    "source_file": gb_path.name,
                    "record_id": rec_id,
                    "seq_id": seq_id,
                    "locus_tag": locus_tag,
                    "gene": gene,
                    "product": product,
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "aa_length": len(prot_seq),
                    "aa_sequence": prot_seq,
                })

                fasta_lines.append(">" + seq_id)
                fasta_lines.append(prot_seq)

    df = pd.DataFrame.from_records(records)

    hits_csv = Path(output_dir) / "hits_metadata.csv"
    df.to_csv(hits_csv, index=False)

    fasta_path = Path(output_dir) / "hits_proteins.faa"
    if len(fasta_lines) > 0:
        with open(fasta_path, "w") as fh:
            fh.write("
".join(fasta_lines) + "
")

    return df


def compute_identity_matrix(df, output_dir, gap_open=-10.0, gap_extend=-0.5):
    """Compute all-by-all global protein identity matrix using BLOSUM62."""
    seqs = list(df["aa_sequence"].values)
    labels = list(df["seq_id"].values)
    n = len(seqs)
    ident_mat = np.zeros((n, n), dtype=float)

    matrix = matlist.blosum62

    for i in range(n):
        ident_mat[i, i] = 1.0
        for j in range(i + 1, n):
            a = seqs[i]
            b = seqs[j]
            alns = pairwise2.align.globalds(a, b, matrix, gap_open, gap_extend, one_alignment_only=True)
            if not alns:
                ident = 0.0
            else:
                aln_a, aln_b, score, start, end = alns[0]
                matches = 0
                length = len(aln_a)
                for k in range(length):
                    if aln_a[k] == aln_b[k] and aln_a[k] != "-":
                        matches += 1
                ident = float(matches) / float(length) if length > 0 else 0.0
            ident_mat[i, j] = ident
            ident_mat[j, i] = ident

    ident_df = pd.DataFrame(ident_mat, index=labels, columns=labels)

    ident_csv = Path(output_dir) / "pairwise_identity_matrix.csv"
    ident_df.to_csv(ident_csv)
    ident_tsv = Path(output_dir) / "pairwise_identity_matrix.tsv"
    ident_df.to_csv(ident_tsv, sep="	")

    dist_df = 1.0 - ident_df
    np.fill_diagonal(dist_df.values, 0.0)

    dist_csv = Path(output_dir) / "pairwise_distance_matrix_1_minus_identity.csv"
    dist_df.to_csv(dist_csv)
    dist_tsv = Path(output_dir) / "pairwise_distance_matrix_1_minus_identity.tsv"
    dist_df.to_csv(dist_tsv, sep="	")

    return ident_df, dist_df


def linkage_to_newick(Z, labels):
    """Convert a SciPy linkage matrix to a Newick string."""
    tree = to_tree(Z, rd=False)

    def build_newick(node, parent_dist):
        if node.is_leaf():
            name = labels[node.id]
            length = max(parent_dist - node.dist, 0.0)
            return name + ":" + str(length)
        left = build_newick(node.get_left(), node.dist)
        right = build_newick(node.get_right(), node.dist)
        length = max(parent_dist - node.dist, 0.0)
        return "(" + left + "," + right + ")" + ":" + str(length)

    return build_newick(tree, tree.dist) + ";"


def make_tree_and_figures(ident_df, dist_df, output_dir, prefix="protein_relatedness"):
    """Create a Newick dendrogram and PNG/PDF figures (heatmap + dendrogram)."""
    labels = list(ident_df.index)
    condensed = squareform(dist_df.values, checks=False)
    Z = linkage(condensed, method="average")

    newick_str = linkage_to_newick(Z, labels)
    newick_path = Path(output_dir) / (prefix + "_average_linkage.newick")
    with open(newick_path, "w") as fh:
        fh.write(newick_str + "
")

    plt.figure(figsize=(10, 8))
    sns.heatmap(ident_df, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.title("Pairwise global identity")
    plt.tight_layout()
    heat_png = Path(output_dir) / (prefix + "_heatmap.png")
    heat_pdf = Path(output_dir) / (prefix + "_heatmap.pdf")
    plt.savefig(heat_png, dpi=300)
    plt.savefig(heat_pdf)
    plt.close()

    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=labels, leaf_rotation=90)
    plt.title("Average-linkage clustering on 1 - identity")
    plt.ylabel("Distance (1 - identity)")
    plt.tight_layout()
    dend_png = Path(output_dir) / (prefix + "_dendrogram.png")
    dend_pdf = Path(output_dir) / (prefix + "_dendrogram.pdf")
    plt.savefig(dend_png, dpi=300)
    plt.savefig(dend_pdf)
    plt.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generic protein relatedness (identity + dendrogram) from GenBank files."
    )
    parser.add_argument(
        "genbank",
        nargs="+",
        help="One or more GenBank files to scan.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output",
        help="Output directory (default: output).",
    )
    parser.add_argument(
        "-k",
        "--keywords",
        default="integrase,site-specific recombinase,tyrosine recombinase,serine recombinase,phage integrase,recombinase,xerC,xerD",
        help=(
            "Comma-separated list of case-insensitive keyword fragments used to filter CDS features "
            "(default: integrase/recombinase-related terms)."
        ),
    )
    parser.add_argument(
        "--gap-open",
        type=float,
        default=-10.0,
        help="Gap opening penalty for global alignments (default: -10.0).",
    )
    parser.add_argument(
        "--gap-extend",
        type=float,
        default=-0.5,
        help="Gap extension penalty for global alignments (default: -0.5).",
    )
    parser.add_argument(
        "--prefix",
        default="protein_relatedness",
        help="Filename prefix for Newick + figure outputs (default: protein_relatedness).",
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    keywords = [kw.strip() for kw in args.keywords.split(",") if kw.strip()]

    print("Scanning GenBank files:")
    for gb in args.genbank:
        print("  " + gb)
    print("Using keywords:")
    print("  " + ", ".join(keywords))
    print("Output directory:")
    print("  " + str(output_dir))

    print("Step 1: Extract matching CDS features and proteins...")
    df_hits = extract_hits_and_fasta(args.genbank, keywords, output_dir)
    print("Number of hits:")
    print(len(df_hits))
    if df_hits.empty:
        print("No hits found; nothing to cluster. Exiting.")
        return 0

    print("Step 2: Compute pairwise global identity matrix...")
    ident_df, dist_df = compute_identity_matrix(
        df_hits,
        output_dir,
        gap_open=args.gap_open,
        gap_extend=args.gap_extend,
    )

    print("Step 3: Build dendrogram and figures...")
    make_tree_and_figures(ident_df, dist_df, output_dir, prefix=args.prefix)

    print("Done. Key outputs:")
    print("  Hits table:         " + str(output_dir / "hits_metadata.csv"))
    print("  Proteins FASTA:     " + str(output_dir / "hits_proteins.faa"))
    print("  Identity matrix:    " + str(output_dir / "pairwise_identity_matrix.tsv"))
    print("  Distance matrix:    " + str(output_dir / "pairwise_distance_matrix_1_minus_identity.tsv"))
    print("  Newick dendrogram:  " + str(output_dir / (args.prefix + "_average_linkage.newick")))
    print("  Heatmap figure:     " + str(output_dir / (args.prefix + "_heatmap.png")))
    print("  Dendrogram figure:  " + str(output_dir / (args.prefix + "_dendrogram.png")))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
