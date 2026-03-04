#!/usr/bin/env python3

"""
att_boundary_scan.py

Detect att-like terminal direct repeats in prophage/hallmark regions.

Features:
- Reads a FASTA file with one or more region sequences.
- Optionally reads matching GenBank files to annotate integrase/recombinase/tRNA/etc.
  features near the sequence termini.
- Searches for terminal direct repeats near both ends using:
  - mismatch-tolerant search (default ≤2 mismatches)
  - exact-match search (0 mismatches)
- Outputs CSV tables summarizing:
  - best mismatch-tolerant repeat per region
  - best exact-match repeat per region
  - end-proximal features from GenBank (if provided)

Dependencies:
- Biopython
- pandas
"""

import argparse
from collections import defaultdict

from Bio import SeqIO
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect att-like boundary repeats in prophage/hallmark regions"
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="FASTA file with one or more hallmark/prophage region sequences",
    )
    parser.add_argument(
        "--genbank",
        nargs="*",
        default=[],
        help="Optional GenBank files with matching region sequences (same IDs)",
    )
    parser.add_argument(
        "--out_prefix",
        required=True,
        help="Prefix for output tables (e.g. results/att_scan)",
    )
    parser.add_argument(
        "--edge_window",
        type=int,
        default=8000,
        help="Number of bp from each end to search for repeats (default: 8000)",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=18,
        help="Minimum repeat length (bp) for reporting (default: 18)",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=120,
        help="Maximum repeat length (bp) for reporting (default: 120)",
    )
    parser.add_argument(
        "--seed_k",
        type=int,
        default=17,
        help="Seed k-mer length for initial matching (default: 17)",
    )
    parser.add_argument(
        "--max_mismatches",
        type=int,
        default=2,
        help="Maximum allowed mismatches in extended repeat (default: 2)",
    )
    return parser.parse_args()


def hamming(a, b):
    """Count mismatches between two equal-length strings."""
    mm = 0
    for x, y in zip(a, b):
        if x != y:
            mm += 1
    return mm


def extend_with_mism(left, right, i, j, k, max_mism):
    """
    Given seed match starting at left[i:i+k] and right[j:j+k],
    extend to the left and right while allowing up to max_mism mismatches.
    """
    l0 = i
    r0 = j
    l1 = i + k
    r1 = j + k

    mm = hamming(left[i:i + k], right[j:j + k])
    if mm > max_mism:
        return None

    # extend left
    while l0 > 0 and r0 > 0:
        add_mm = 1 if left[l0 - 1] != right[r0 - 1] else 0
        if mm + add_mm > max_mism:
            break
        mm += add_mm
        l0 -= 1
        r0 -= 1

    # extend right
    while l1 < len(left) and r1 < len(right):
        add_mm = 1 if left[l1] != right[r1] else 0
        if mm + add_mm > max_mism:
            break
        mm += add_mm
        l1 += 1
        r1 += 1

    return l0, l1, r0, r1, mm


def best_repeat_for_seq(seq_str, edge_window, seed_k, min_len, max_len, max_mismatches):
    """
    Find the best terminal direct repeat between left and right edges of seq_str.

    Returns a dict with:
      repeat_len, mismatches, attL_1based, attR_1based, repeat_left, repeat_right
    or None if nothing passes filters.
    """
    seq_len = len(seq_str)
    left = seq_str[:min(edge_window, seq_len)]
    right_start = max(0, seq_len - edge_window)
    right = seq_str[right_start:]

    if len(left) < seed_k or len(right) < seed_k:
        return None

    prefix_k = 10  # short prefix to index seeds

    right_pref = defaultdict(list)
    for j in range(0, len(right) - seed_k + 1):
        pref = right[j:j + prefix_k]
        right_pref[pref].append(j)

    best = None
    for i in range(0, len(left) - seed_k + 1):
        left_seed = left[i:i + seed_k]
        pref = left_seed[:prefix_k]
        cand_positions = right_pref.get(pref, [])
        if not cand_positions:
            continue

        for j in cand_positions:
            seg = right[j:j + seed_k]
            if hamming(left_seed, seg) > max_mismatches:
                continue

            ext = extend_with_mism(left, right, i, j, seed_k, max_mismatches)
            if ext is None:
                continue

            l0, l1, r0, r1, mm = ext
            rep_len = l1 - l0
            if rep_len < min_len or rep_len > max_len:
                continue

            # score: prioritize length heavily, small penalty for mismatches
            score = rep_len * 1000000 - mm * 50000
            if best is None or score > best[0]:
                best = (score, l0, l1, r0, r1, mm)

    if best is None:
        return None

    _, l0, l1, r0, r1, mm = best
    rep_len = l1 - l0
    left_rep = left[l0:l1]
    right_rep = right[r0:r1]

    # map window coordinates back to full-sequence 1-based coordinates
    attL_start = l0 + 1
    attL_end = l1
    attR_start_full = right_start + r0 + 1
    attR_end_full = right_start + r1

    return {
        "repeat_len": rep_len,
        "mismatches": mm,
        "attL_1based": str(attL_start) + "-" + str(attL_end),
        "attR_1based": str(attR_start_full) + "-" + str(attR_end_full),
        "repeat_left": left_rep,
        "repeat_right": right_rep,
    }


def exact_repeat_for_seq(seq_str, edge_window, min_len, max_len):
    """
    Convenience wrapper for exact-match repeats: calls best_repeat_for_seq with max_mismatches=0.
    For exact matching, we use seed_k = min_len to ensure seeds cover at least the min length.
    """
    return best_repeat_for_seq(
        seq_str=seq_str,
        edge_window=edge_window,
        seed_k=min_len,
        min_len=min_len,
        max_len=max_len,
        max_mismatches=0,
    )


def load_fasta_sequences(fasta_path):
    """Return dict: region_id -> uppercase sequence string."""
    region_seq = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        region_seq[rec.id] = str(rec.seq).upper()
    return region_seq


def load_genbank_features(gbk_paths, end_window=2000):
    """
    Load GenBank features within end_window bp of either terminus.

    Returns a DataFrame with:
      region, length_bp, feature_type, start_1based, end_1based, strand,
      near_end (left/right), annotation
    """
    records = []
    if not gbk_paths:
        return pd.DataFrame(records)

    from Bio import SeqIO as _SeqIO

    for gbk in gbk_paths:
        for rec in _SeqIO.parse(gbk, "genbank"):
            region_id = rec.id
            seqlen = len(rec.seq)
            for feat in rec.features:
                ftype = feat.type
                if ftype not in ["CDS", "tRNA", "tmRNA", "rRNA", "repeat_region"]:
                    continue

                start = int(feat.location.start) + 1  # 1-based
                end = int(feat.location.end)
                strand = int(feat.location.strand) if feat.location.strand is not None else 0

                near_end = None
                if start <= end_window or end <= end_window:
                    near_end = "left"
                elif (seqlen - end) <= end_window or (seqlen - start) <= end_window:
                    near_end = "right"

                if near_end is None:
                    continue

                anno_parts = []
                for k in ["gene", "product", "locus_tag"]:
                    if k in feat.qualifiers:
                        anno_parts.append(k + "=" + feat.qualifiers[k][0])
                annotation = "; ".join(anno_parts) if anno_parts else ""

                records.append(
                    {
                        "region": region_id,
                        "length_bp": seqlen,
                        "feature_type": ftype,
                        "start_1based": start,
                        "end_1based": end,
                        "strand": strand,
                        "near_end": near_end,
                        "annotation": annotation,
                    }
                )

    return pd.DataFrame(records)


def main():
    args = parse_args()

    # 1) Load sequences
    region_seq = load_fasta_sequences(args.fasta)

    # 2) Mismatch-tolerant best repeat per region
    mismatch_rows = []
    for region, seq_str in region_seq.items():
        res = best_repeat_for_seq(
            seq_str=seq_str,
            edge_window=args.edge_window,
            seed_k=args.seed_k,
            min_len=args.min_len,
            max_len=args.max_len,
            max_mismatches=args.max_mismatches,
        )
        row = {
            "region": region,
            "length_bp": len(seq_str),
            "att_found_mismatch": res is not None,
            "strand": "+",
        }
        if res is not None:
            row.update(res)
        mismatch_rows.append(row)

    df_mismatch = pd.DataFrame(mismatch_rows)

    # 3) Exact-match best repeat per region
    exact_rows = []
    for region, seq_str in region_seq.items():
        res = exact_repeat_for_seq(
            seq_str=seq_str,
            edge_window=args.edge_window,
            min_len=args.min_len,
            max_len=args.max_len,
        )
        row = {
            "region": region,
            "length_bp": len(seq_str),
            "att_found_exact": res is not None,
        }
        if res is not None:
            row.update(res)
        exact_rows.append(row)

    df_exact = pd.DataFrame(exact_rows)

    # 4) GenBank end-proximal features, if provided
    df_end_feats = load_genbank_features(args.genbank, end_window=2000)

    # 5) Merge repeat calls with end-feature evidence
    if not df_end_feats.empty:
        end_evidence = (
            df_end_feats.groupby("region", as_index=False)
            .agg(
                {
                    "feature_type": lambda x: ",".join(sorted(set(x))),
                    "annotation": lambda x: " | ".join(list(x)[:4]),
                }
            )
            .rename(
                columns={
                    "feature_type": "end_features",
                    "annotation": "end_feature_annotations",
                }
            )
        )
        final_df = df_mismatch.merge(end_evidence, on="region", how="left")
    else:
        final_df = df_mismatch.copy()
        final_df["end_features"] = ""
        final_df["end_feature_annotations"] = ""

    col_order = [
        "region",
        "length_bp",
        "att_found_mismatch",
        "strand",
        "repeat_len",
        "mismatches",
        "attL_1based",
        "attR_1based",
        "repeat_left",
        "repeat_right",
        "end_features",
        "end_feature_annotations",
    ]
    final_df = final_df.reindex(columns=col_order)

    # 6) Write outputs
    out_prefix = args.out_prefix
    final_df.to_csv(out_prefix + "_att_like_boundaries_main.csv", index=False)
    df_exact.to_csv(out_prefix + "_exact_match_att_scan.csv", index=False)
    df_mismatch.to_csv(out_prefix + "_mismatch_tolerant_att_scan.csv", index=False)
    if not df_end_feats.empty:
        df_end_feats.to_csv(out_prefix + "_end_features_near_termini.csv", index=False)

    print("Wrote:")
    print(out_prefix + "_att_like_boundaries_main.csv")
    print(out_prefix + "_exact_match_att_scan.csv")
    print(out_prefix + "_mismatch_tolerant_att_scan.csv")
    if not df_end_feats.empty:
        print(out_prefix + "_end_features_near_termini.csv")


if __name__ == "__main__":
    main()