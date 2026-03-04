#!/usr/bin/env python3
"""
Predict and extract prophage regions from PhageMiner results.

Inputs
------
- A GenBank file with the full bacterial genome or contig(s).
- A TSV/CSV with PhageMiner / hallmark results (e.g. integrases, recombinases,
  phage genes), with at least:
    - contig / seqid column
    - start coordinate (1-based)
    - end coordinate (1-based)
    - an ID column (gene_id, locus_tag, etc.)

Approach
--------
1. Read the hallmark hits table.
2. Group by contig and sort by position.
3. Cluster hits into candidate prophage regions using a max-gap threshold.
4. Filter clusters by minimum number of hits.
5. For each region, extract the corresponding subsequence from the GenBank,
   copy overlapping features, shift their coordinates, and write:
   - one .gbk
   - one .fasta

Dependencies
-----------
pip install biopython pandas
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation


def log(msg):
    sys.stderr.write(str(msg) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict and extract prophage regions from PhageMiner hallmark hits."
    )

    parser.add_argument(
        "--input-gbk",
        required=True,
        help="Input GenBank file with full genome/contigs (e.g. ASM2515056.gbk).",
    )
    parser.add_argument(
        "--hallmark-tsv",
        required=True,
        help="TSV/CSV with hallmark hits (integrases/phage genes) from PhageMiner.",
    )
    parser.add_argument(
        "--sep",
        default="\t",
        help="Separator for hallmark file (default: tab). Use ',' for CSV.",
    )
    parser.add_argument(
        "--contig-col",
        required=True,
        help="Column name in hallmark file with contig/seqid (e.g. 'contig').",
    )
    parser.add_argument(
        "--start-col",
        required=True,
        help="Column name with hit start coordinate (1-based).",
    )
    parser.add_argument(
        "--end-col",
        required=True,
        help="Column name with hit end coordinate (1-based).",
    )
    parser.add_argument(
        "--id-col",
        required=True,
        help="Column name with feature ID (gene_id, locus_tag, etc.).",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=2,
        help="Minimum number of hallmark hits to keep a region (default: 2).",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=20000,
        help="Maximum gap between consecutive hallmark hits within a region (bp, default: 20000).",
    )
    parser.add_argument(
        "--flank",
        type=int,
        default=0,
        help="Optional flanking sequence to add on both sides of each region (bp, default: 0).",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Output prefix for region files (e.g. ASM2515056_hallmark).",
    )

    return parser.parse_args()


def load_hallmark_table(path, sep, contig_col, start_col, end_col, id_col):
    df = pd.read_csv(path, sep=sep)

    for col in [contig_col, start_col, end_col, id_col]:
        if col not in df.columns:
            raise ValueError(
                "Column " + col + " not found in hallmark table " + str(path)
            )

    df = df.copy()
    df = df.dropna(subset=[contig_col, start_col, end_col])
    df[start_col] = pd.to_numeric(df[start_col], errors="coerce")
    df[end_col] = pd.to_numeric(df[end_col], errors="coerce")
    df = df.dropna(subset=[start_col, end_col])

    df[start_col] = df[start_col].astype(int)
    df[end_col] = df[end_col].astype(int)

    df["start_min"] = df[[start_col, end_col]].min(axis=1)
    df["end_max"] = df[[start_col, end_col]].max(axis=1)

    df = df.sort_values([contig_col, "start_min"]).reset_index(drop=True)

    return df


def cluster_hits(df, contig_col, max_gap, min_hits):
    """
    Cluster hallmark hits into candidate prophage regions by contig.

    Returns a list of dicts:
    {
      "contig": ...,
      "start": ...,
      "end": ...,
      "n_hits": ...,
      "hit_indices": [row indices in df],
    }
    """
    regions = []

    for contig, grp in df.groupby(contig_col):
        grp = grp.sort_values("start_min")
        current = None

        for idx, row in grp.iterrows():
            s = int(row["start_min"])
            e = int(row["end_max"])

            if current is None:
                current = {
                    "contig": contig,
                    "start": s,
                    "end": e,
                    "n_hits": 1,
                    "hit_indices": [idx],
                }
            else:
                if s - current["end"] <= max_gap:
                    if s < current["start"]:
                        current["start"] = s
                    if e > current["end"]:
                        current["end"] = e
                    current["n_hits"] += 1
                    current["hit_indices"].append(idx)
                else:
                    regions.append(current)
                    current = {
                        "contig": contig,
                        "start": s,
                        "end": e,
                        "n_hits": 1,
                        "hit_indices": [idx],
                    }

        if current is not None:
            regions.append(current)

    regions = [r for r in regions if r["n_hits"] >= min_hits]

    # Sort by contig, then start
    regions.sort(key=lambda r: (r["contig"], r["start"]))

    # Assign region IDs per contig: R1, R2, ...
    by_contig = {}
    for r in regions:
        contig = r["contig"]
        if contig not in by_contig:
            by_contig[contig] = 0
        by_contig[contig] += 1
        r["region_id"] = by_contig[contig]

    return regions


def index_genbank_by_id(gbk_path):
    """
    Read GenBank and return a dict {seqid: record}.
    """
    seqid_to_rec = {}
    for rec in SeqIO.parse(str(gbk_path), "genbank"):
        seqid_to_rec[rec.id] = rec
    if not seqid_to_rec:
        raise ValueError("No records found in GenBank file " + str(gbk_path))
    return seqid_to_rec


def extract_region_record(
    rec,
    region_start_1based,
    region_end_1based,
    new_id,
    new_description,
    flank_bp=0,
):
    """
    Extract a sub-record [region_start_1based..region_end_1based] with optional flanks
    and copy overlapping features, shifting coordinates.
    """
    seqlen = len(rec.seq)

    # Apply flanks and clamp to sequence bounds
    start_1 = max(1, region_start_1based - flank_bp)
    end_1 = min(seqlen, region_end_1based + flank_bp)

    start0 = start_1 - 1
    end0 = end_1

    sub_seq = rec.seq[start0:end0]

    sub_rec = SeqRecord(
        sub_seq,
        id=new_id,
        name=new_id,
        description=new_description,
    )

    # Basic annotations
    sub_rec.annotations["molecule_type"] = rec.annotations.get(
        "molecule_type", "DNA"
    )
    sub_rec.annotations["topology"] = rec.annotations.get("topology", "linear")
    if "data_file_division" in rec.annotations:
        sub_rec.annotations["data_file_division"] = rec.annotations[
            "data_file_division"
        ]
    if "date" in rec.annotations:
        sub_rec.annotations["date"] = rec.annotations["date"]
    sub_rec.annotations["source"] = rec.annotations.get(
        "source", rec.description
    )
    sub_rec.annotations["organism"] = rec.annotations.get("organism", "")
    sub_rec.annotations["taxonomy"] = rec.annotations.get("taxonomy", [])

    kept_feats = 0
    for feat in rec.features:
        f_start = int(feat.location.start)
        f_end = int(feat.location.end)

        if f_end <= start0 or f_start >= end0:
            continue

        new_start = max(f_start, start0) - start0
        new_end = min(f_end, end0) - start0

        new_loc = FeatureLocation(
            new_start, new_end, strand=feat.location.strand
        )
        new_feat = SeqFeature(
            location=new_loc, type=feat.type, qualifiers=feat.qualifiers
        )
        sub_rec.features.append(new_feat)
        kept_feats += 1

    return sub_rec, kept_feats, (start_1, end_1)


def main():
    args = parse_args()

    hallmark_path = Path(args.hallmark_tsv)
    gbk_path = Path(args.input_gbk)
    out_prefix = Path(args.out_prefix)

    log("Loading hallmark hits from " + str(hallmark_path))
    df = load_hallmark_table(
        hallmark_path,
        sep=args.sep,
        contig_col=args.contig_col,
        start_col=args.start_col,
        end_col=args.end_col,
        id_col=args.id_col,
    )
    log("Loaded " + str(df.shape[0]) + " hallmark hits")

    log("Indexing GenBank file " + str(gbk_path))
    seqid_to_rec = index_genbank_by_id(gbk_path)
    log("Found " + str(len(seqid_to_rec)) + " sequence records in GenBank")

    log("Clustering hallmark hits into candidate prophage regions")
    regions = cluster_hits(
        df,
        contig_col=args.contig_col,
        max_gap=args.max_gap,
        min_hits=args.min_hits,
    )
    log("Identified " + str(len(regions)) + " candidate prophage regions")

    if not regions:
        log("No regions passed filters (min_hits = " + str(args.min_hits) + ")")
        sys.exit(0)

    summary_rows = []

    for region in regions:
        contig = region["contig"]
        start = region["start"]
        end = region["end"]
        n_hits = region["n_hits"]
        region_id = region["region_id"]

        if contig not in seqid_to_rec:
            log(
                "WARNING: contig "
                + str(contig)
                + " not found in GenBank. Skipping region "
                + str(region_id)
            )
            continue

        rec = seqid_to_rec[contig]

        region_label = "R" + str(region_id)
        new_id = (
            out_prefix.name + "_" + region_label
        )  # e.g. ASM2515056_hallmark_R1
        description = (
            "Predicted prophage region "
            + region_label
            + " on "
            + contig
            + " "
            + str(start)
            + ".."
            + str(end)
            + " ("
            + str(n_hits)
            + " hallmark hits)"
        )

        sub_rec, kept_feats, (used_start, used_end) = extract_region_record(
            rec,
            region_start_1based=start,
            region_end_1based=end,
            new_id=new_id,
            new_description=description,
            flank_bp=args.flank,
        )

        gbk_out = out_prefix.parent / (new_id + ".gbk")
        fa_out = out_prefix.parent / (new_id + ".fasta")

        SeqIO.write([sub_rec], str(gbk_out), "genbank")
        SeqIO.write([sub_rec], str(fa_out), "fasta")

        log(
            "Wrote region "
            + region_label
            + " (contig "
            + contig
            + ", "
            + str(used_start)
            + "-"
            + str(used_end)
            + ", "
            + str(len(sub_rec.seq))
            + " bp, "
            + str(kept_feats)
            + " features) to:"
        )
        log("  " + str(gbk_out))
        log("  " + str(fa_out))

        summary_rows.append(
            {
                "region_label": region_label,
                "contig": contig,
                "start": start,
                "end": end,
                "used_start_with_flank": used_start,
                "used_end_with_flank": used_end,
                "length_bp": len(sub_rec.seq),
                "n_hallmark_hits": n_hits,
                "n_features": kept_feats,
                "gbk_path": str(gbk_out),
                "fasta_path": str(fa_out),
            }
        )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_tsv = out_prefix.parent / (out_prefix.name + "_regions_summary.tsv")
        summary_df.to_csv(summary_tsv, sep="\t", index=False)
        log("Wrote region summary to " + str(summary_tsv))


if __name__ == "__main__":
    main()